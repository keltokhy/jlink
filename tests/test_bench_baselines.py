"""Offline checks of benchmark scoring, source preparation and pipeline integration."""

import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import zipfile

import httpx
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# pytest's console entry point need not put the repository root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import baselines, prepare
from bench import run as runner
from bench.report import render_report
from bench.data import load_dataset, sample_left, validate_dataset, write_dataset


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("benchmark tests must not make network requests")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)


def pairs(rows=()):
    return pd.DataFrame(rows, columns=["left_id", "right_id"])


def records(names, prefix=""):
    return pd.DataFrame({"id": [f"{prefix}{i}" for i in range(len(names))], "name": names})


def metadata():
    return {"entity": "firm", "definition": "The same company.", "on": ["name"], "how": "many-to-one"}


def test_pair_scorer_counts_sets_and_empty_cases():
    truth = pairs([("a", "x"), ("a", "x"), ("b", "y")])
    links = pairs([("a", "x"), ("c", "z"), ("c", "z")])
    result = baselines.score_against_truth(links, truth, candidates=pairs([("a", "x")]))
    assert result == {"tp": 1, "fp": 1, "fn": 1, "precision": .5, "recall": .5,
                      "f1": .5, "pairs_completeness": .5}
    assert baselines.score_against_truth(pairs(), pairs())["f1"] == 0
    with pytest.raises(ValueError, match="right_id"):
        baselines.score_against_truth(pd.DataFrame({"left_id": []}), truth)
    with pytest.raises(ValueError, match="missing IDs"):
        baselines.score_against_truth(pairs([(None, "x")]), truth)


def test_exact_normalization_fields_and_missing_values():
    left = pd.DataFrame({"id": ["001", "002", "003", "004"],
                         "name": ["Société & Co.", "a b", "x", ""], "city": ["NY", "c", None, "NY"]})
    right = pd.DataFrame({"key": [7, 8, 9, 10, 11],
                          "label": ["societe and co", "a", "x", "", "SOCIETE & CO"],
                          "town": ["ny", "b c", None, "ny", "NY"]})
    truth = pairs([("001", 7), ("001", 11)])
    result = baselines.exact_match(left, right, truth, on=[("name", "label"), ("city", "town")],
                                   left_id="id", right_id="key")
    assert set(map(tuple, result.links[["left_id", "right_id"]].values)) == {("001", 7), ("001", 11)}
    assert result.metrics["f1"] == 1
    assert result.tuned_on_truth is False


@pytest.mark.parametrize("method", [baselines.exact_match, baselines.jaro_winkler, baselines.tfidf_cosine])
def test_methods_validate_and_support_index_ids(method):
    left, right = records(["same"]), records(["same"])
    assert method(left, right, pairs([(0, 0)]), on="name").metrics["f1"] == 1
    with pytest.raises(ValueError, match="duplicate IDs"):
        method(pd.concat([left, left]), right, pairs(), on="name")
    with pytest.raises(ValueError, match="no column 'missing'"):
        method(left, right, pairs(), on="missing")
    with pytest.raises(ValueError, match="absent"):
        method(left, right, pairs([(99, 0)]), on="name")
    left.loc[0, "id"] = None
    with pytest.raises(ValueError, match="missing"):
        method(left, right, pairs(), on="name", left_id="id")


def test_jaro_first_character_block_recall_and_tie_break():
    left = records(["apple", "ibm", ""], "L")
    right = records(["apple", "apple", "zibm", ""], "R")
    truth = pairs([("L0", "R0"), ("L1", "R2")])
    result = baselines.jaro_winkler(left, right, truth, on="name", left_id="id", right_id="id")
    assert result.candidate_count == 2
    assert result.blocking_recall == .5
    assert result.links.right_id.tolist() == ["R0"]
    assert result.metrics["recall"] == .5
    assert result.tuned_on_truth


def test_tfidf_sparse_chunks_against_independent_dense_reference():
    from sklearn.feature_extraction.text import TfidfVectorizer

    left, right = records(["beta", "alpha", "gamut", "q"], "L"), records(["alpha", "betas", "gamma"], "R")
    truth = pairs([("L0", "R1"), ("L1", "R0"), ("L2", "R2")])
    result = baselines.tfidf_cosine(left, right, truth, on="name", left_id="id", right_id="id", chunk_size=1)
    vectors = TfidfVectorizer(analyzer="char", ngram_range=(2, 4)).fit_transform(
        left.name.tolist() + right.name.tolist())
    dense = (vectors[:4] @ vectors[4:].T).toarray()
    assert result.candidate_count == np.count_nonzero(dense)
    assert result.blocking_recall == 1
    assert result.metrics["f1"] == 1
    for row in result.links.itertuples():
        i, j = int(row.left_id[1:]), int(row.right_id[1:])
        assert j == np.argmax(dense[i])
        assert row.sim == pytest.approx(dense[i, j])
    repeated = baselines.tfidf_cosine(left, right, truth, on="name", left_id="id", right_id="id",
                                        chunk_size=50)
    assert_frame_equal(result.links, repeated.links)


def test_tfidf_zero_vectors_empty_frames_and_deterministic_ties():
    for a, b in [(["a", ""], ["b", ""]), ([], ["abc"]), (["abc"], [])]:
        result = baselines.tfidf_cosine(records(a), records(b), pairs(), on="name")
        assert result.links.empty
        assert result.candidate_count == 0
    result = baselines.tfidf_cosine(records(["same"]), records(["same", "same"]), pairs([(0, 0)]), on="name")
    assert result.links.right_id.tolist() == [0]
    with pytest.raises(ValueError, match="chunk_size"):
        baselines.tfidf_cosine(records([]), records([]), pairs(), on="name", chunk_size=0)


def test_threshold_grouped_ties_and_reject_all():
    best = pd.DataFrame({"left_id": [0, 1, 2], "right_id": [0, 1, 2], "sim": [.9, .8, .8]})
    truth = pairs([(0, 0), (1, 1)])
    links, threshold = baselines.tune_threshold(best, truth)
    assert threshold == .8
    assert len(links) == 3  # cannot cherry-pick only the true .8 pair
    assert baselines.tune_threshold(best, pairs([(5, 5)]))[1] is None
    assert baselines.tune_threshold(best, pairs())[0].empty
    for score in [float("nan"), float("inf"), 1.1, -.1, "bad"]:
        with pytest.raises(ValueError, match="finite"):
            baselines.tune_threshold(best.assign(sim=score), truth)
    with pytest.raises(ValueError, match="one pair per left_id"):
        baselines.tune_threshold(pd.concat([best, best]), truth)


def test_threshold_matches_exhaustive_search_with_rounded_random_scores():
    rng = np.random.default_rng(712)
    for _ in range(30):
        best = pd.DataFrame({"left_id": range(30), "right_id": range(30),
                             "sim": np.round(rng.random(30), 1)})
        truth = pairs([(i, i) for i in range(30) if rng.random() < .4] + [(99, 99)])
        links, _ = baselines.tune_threshold(best, truth)
        expected = max(baselines.score_against_truth(best.loc[best.sim >= t], truth)["f1"]
                       for t in best.sim.unique())
        assert baselines.score_against_truth(links, truth)["f1"] == pytest.approx(expected)


def test_threshold_f1_ties_prefer_fewer_links():
    best = pd.DataFrame({"left_id": range(4), "right_id": range(4), "sim": [.9, .8, .8, .8]})
    links, threshold = baselines.tune_threshold(best, pairs([(0, 0), (3, 3)]))
    assert threshold == .9  # F1 is 2/3 at both thresholds
    assert len(links) == 1


def test_layout_dangling_ids_duplicates_and_roundtrip(tmp_path):
    left, right = records(["a", "b"]), records(["a", "a"])
    truth = pairs([("0", "0"), ("0", "0"), ("0", "1")])
    meta = write_dataset(tmp_path / "nber-firms", left, right, truth, metadata())
    assert meta["validation"]["truth_duplicate_pairs"] == 1
    assert meta["validation"]["truth_left_ids_with_multiple_matches"] == 1
    assert meta["validation"]["right_duplicate_records_excluding_id"] == 1
    assert meta["row_counts"]["truth"] == 2
    loaded = load_dataset("nber-firms", tmp_path)
    assert loaded[0].id.tolist() == ["0", "1"]
    assert len(loaded[2]) == 2
    with pytest.raises(ValueError, match="absent"):
        validate_dataset(left, right, pairs([("unknown", "0")]), metadata())
    with pytest.raises(ValueError, match="prepare.py"):
        load_dataset("abt-buy", tmp_path)
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("../../elsewhere", tmp_path)


def test_left_sampling_restricts_truth_with_fixed_seed():
    left = records(["company"] * 100)
    left.index = range(300, 400)
    truth = pairs([(str(i), "right") for i in range(100)])
    sampled, restricted = sample_left(left, truth, 11)
    assert len(sampled) == len(restricted) == 11
    assert set(restricted.left_id) == set(sampled.id)
    assert_frame_equal(sampled, sample_left(left, truth, 11)[0])
    assert sample_left(left, truth, 1000)[0].equals(left)
    for n in [0, -2, 2.5, True]:
        with pytest.raises(ValueError, match="--sample"):
            sample_left(left, truth, n)


def test_nber_excludes_unknowns_preserves_leading_zeros_and_reports_collapses():
    raw = pd.DataFrame({"assignee": ["01", "01", "01", "02", "03"],
                        "assname": ["Alias", "Z alias", "Alias", "No label", "Other"],
                        "cusip": ["000001", "000002", "000001", "", "000001"],
                        "cname": ["Parent", "Other parent", "Parent", "Unknown", "Parent old"]})
    left, right, truth, diagnostics = prepare.build_nber(raw)
    assert left.id.tolist() == ["01", "03"]
    assert left.name.tolist() == ["Alias", "Other"]
    assert right.id.tolist() == ["000001", "000002"]
    assert right.name.tolist() == ["Parent", "Other parent"]
    assert len(truth.drop_duplicates()) == 3
    assert diagnostics["excluded_incomplete_rows"] == 1
    assert diagnostics["source_duplicate_rows"] == 1
    assert diagnostics["cusips_with_multiple_names"] == 1


def test_download_checksum_receipt_cache_and_changed_source(tmp_path, monkeypatch):
    body = b"public test data"
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, content=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    checksum = prepare._sha256(body)
    path, receipt = prepare.download("https://example.test/data.zip", checksum, tmp_path)
    assert receipt["downloaded_at"]
    assert path.read_bytes() == body
    assert prepare.download("https://example.test/data.zip", checksum, tmp_path)[1] == receipt
    assert len(calls) == 1
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cached source"):
        prepare.download("https://example.test/data.zip", checksum, tmp_path)
    with pytest.raises(ValueError, match="source.*changed"):
        prepare.download("https://example.test/changed.zip", "0" * 64, tmp_path)


def test_leipzig_original_mapping_and_amazon_column_rename(tmp_path, monkeypatch):
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Amazon.csv", 'id,title,manufacturer,price\n001,Product,Maker,20\n')
        archive.writestr("GoogleProducts.csv", 'id,name,manufacturer,price\n002,Product,Maker,20\n')
        archive.writestr("Amzon_GoogleProducts_perfectMapping.csv", "idAmazon,idGoogleBase\n001,002\n")
    monkeypatch.setattr(prepare, "download", lambda *args: (path, {"downloaded_at": "2026-09-18"}))
    left, right, truth, meta = prepare._leipzig("amazon-google", tmp_path)
    assert left.name.tolist() == ["Product"]
    assert truth.left_id.tolist() == ["001"]
    assert meta["license"] == "CC-BY-4.0"
    validate_dataset(left, right, truth, meta)


def test_prepare_continues_after_unreachable_source(monkeypatch, capsys):
    visited = []

    def fake(name):
        visited.append(name)
        if name == "abt-buy":
            raise httpx.ConnectError("unreachable")
        return {"row_counts": {}, "validation": {}}

    monkeypatch.setattr(prepare, "prepare", fake)
    assert prepare.main(["abt-buy", "febrl4"]) == 1
    assert visited == ["abt-buy", "febrl4"]
    assert "unreachable" in capsys.readouterr().err


def test_febrl_uses_bundled_pinned_data_offline():
    left, right, truth, meta = prepare._febrl()
    assert (len(left), len(right), len(truth)) == (5000, 5000, 5000)
    assert validate_dataset(left, right, truth, meta)["truth_ids_verified"]
    assert all(len(source["sha256"]) == 64 for source in meta["sources"])


def test_legacy_nber_zip_fallback_verifies_payload_crc(tmp_path, monkeypatch):
    path = tmp_path / "legacy.zip"
    info = zipfile.ZipInfo("match.csv")
    info.extra = b"\xa0\x0c\x99\x0e"  # invalid extra-field length, as in the original archive
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, b"assignee,cusip\n001,000001\n")
    with pytest.raises(ValueError, match="pinned NBER"):
        prepare._nber_csv(path)
    monkeypatch.setattr(prepare, "_sha256", lambda _: prepare.NBER_SHA256)
    assert prepare._nber_csv(path) == b"assignee,cusip\n001,000001\n"
    corrupted = bytearray(path.read_bytes())
    struct.pack_into("<I", corrupted, 14, 0)  # incorrect stored CRC, otherwise valid payload
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="CRC"):
        prepare._nber_csv(path)


def test_offline_harness_never_imports_live_modules(tmp_path, monkeypatch):
    left, right = records(["alpha", "beta"]), records(["alpha", "beta", "gamma"])
    write_dataset(tmp_path / "data" / "nber-firms", left, right, pairs([("0", "0"), ("1", "1")]), metadata())
    imported = []

    def missing(name, required):
        imported.append(name)
        assert name == "jlink.block"
        return None

    monkeypatch.setattr(runner, "_optional_module", missing)
    report = runner.run("nber-firms", sample=1, data_dir=tmp_path / "data", out_dir=tmp_path / "out")
    assert imported == ["jlink.block"]
    assert report["blocking"]["status"] == "unavailable"
    assert report["live"] == {"status": "not_requested", "api_calls": 0}
    assert report["evaluated_counts"] == {"left": 1, "right": 3, "truth": 1}
    assert json.loads((tmp_path / "out" / "nber-firms.json").read_text()) == report
    assert all(value["metrics"]["f1"] == 1 for value in report["baselines"].values())
    for name in runner.NAMES:
        copy = dict(report, dataset=name)
        (tmp_path / "out" / f"{name}.json").write_text(json.dumps(copy))
    markdown = render_report(tmp_path / "out")
    assert "upper bounds" in markdown and "Blocking recall" in markdown
    assert "| nber-firms | 1 | 3 | 1 | 1; seed 0 |" in markdown


def test_lazy_import_only_swallows_absent_module(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("missing", name=name)

    monkeypatch.setattr(runner.importlib, "import_module", missing)
    assert runner._optional_module("jlink.linker", ("Linker",)) is None

    def broken(name):
        raise ModuleNotFoundError("internal dependency broken", name="internal_dependency")

    monkeypatch.setattr(runner.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="internal dependency"):
        runner._optional_module("jlink.linker", ("Linker",))


def test_pipeline_paths_follow_contract_with_fake_modules(monkeypatch):
    left, right = records(["alpha"]), records(["alpha"])
    truth = pairs([("0", "0")])
    candidates = truth.assign(sim=1.0, block="ngrams:name")
    calls = []

    def ngrams(*columns, k=10):
        assert columns == ("name",) and k == 10
        return "blocker"

    def propose(a, b, *, on, left_id, right_id, blockers):
        assert on == ["name"] and left_id == right_id == "id" and blockers == ["blocker"]
        return candidates

    class Linker:
        def __init__(self, *, entity, definition, on, blockers):
            assert entity == "firm" and definition == "The same company."
            assert on == ["name"] and blockers == ["blocker"]

        def link(self, a, b, *, left_id, right_id, how, threshold, min_margin, budget):
            assert left_id == right_id == "id" and how == "many-to-one"
            assert threshold == .5 and min_margin == 0 and budget == .07
            calls.append("fake link")
            return SimpleNamespace(links=candidates, candidates=candidates,
                                   meter=SimpleNamespace(calls=0, cached=1, retries=0,
                                                         input_tokens=0, cost=0.0))

    modules = {"jlink.block": SimpleNamespace(ngrams=ngrams, candidates=propose),
               "jlink.linker": SimpleNamespace(Linker=Linker),
               "jlink.audit": SimpleNamespace(score_against_truth=baselines.score_against_truth)}
    monkeypatch.setattr(runner, "_optional_module", lambda name, required: modules[name])
    blocked, _ = runner._blocking(left, right, truth, metadata())
    assert blocked["pairs_completeness"] == 1
    assert blocked["estimated_cost_usd"] == pytest.approx(330 * .042 / 1_000_000)
    # Exercise only this fake; never pass --live to the real CLI.
    live = runner._live(left, right, truth, metadata(), .07)
    assert live["metrics"]["f1"] == 1 and calls == ["fake link"]
    assert live["meter"]["cost"] == 0


@pytest.mark.parametrize("budget", [-1, float("inf"), float("nan"), True])
def test_harness_rejects_invalid_budgets(budget):
    with pytest.raises(ValueError, match="--budget"):
        runner.run("nber-firms", budget=budget)

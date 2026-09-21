"""No-network regressions for evidence boundaries, splitting, resolution, and cached replay."""

import importlib.util
import json
from pathlib import Path
import sys

import httpx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import evaluation as ev
from bench import heldout
from bench.data import write_dataset
from bench.probabilistic import ECMBaseline, comparison_vectors


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("offline benchmarks may not send network requests")
    monkeypatch.setattr(httpx.Client, "send", fail)
    monkeypatch.setattr(httpx.AsyncClient, "send", fail)


def pairs(rows=()):
    return pd.DataFrame(rows, columns=["left_id", "right_id"])


def records(n=12):
    left = pd.DataFrame({"id": [f"L{i:03}" for i in range(n)],
                         "name": [f"Company number {i}" for i in range(n)],
                         "city": [f"City number {i}" for i in range(n)]})
    right = left.assign(id=[f"R{i:03}" for i in range(n)])
    return left, right, pairs(zip(left.id, right.id))


def meta():
    return {"entity": "firm", "definition": "Same firm.", "on": ["name", "city"], "how": "one-to-one"}


def test_split_is_entity_disjoint_reproducible_and_order_invariant():
    left, right, truth = records()
    # A many-to-many component and a duplicate signature must never leak across partitions.
    truth = pd.concat([truth, pairs([("L000", "R001")])], ignore_index=True)
    left.loc[2, ["name", "city"]] = left.loc[3, ["name", "city"]].to_numpy()
    split = ev.entity_split(left, right, truth, on=meta()["on"])
    shuffled = ev.entity_split(left.sample(frac=1), right.sample(frac=1), truth.sample(frac=1),
                               on=meta()["on"])
    pd.testing.assert_frame_equal(split, shuffled)
    assert set(split.split) == {"dev", "test"}
    assert split.groupby("group").split.nunique().max() == 1
    look = split.set_index(["side", "id"])
    for a, b in truth.itertuples(index=False, name=None):
        assert look.loc[("left", a), "split"] == look.loc[("right", b), "split"]
    assert look.loc[("left", "L002"), "group"] == look.loc[("left", "L003"), "group"]
    for part in ("dev", "test"):
        a, b, gold = ev.subset(left, right, truth, split, part)
        assert set(gold.left_id) <= set(a.id) and set(gold.right_id) <= set(b.id)


def test_split_refuses_impossible_and_missing_truth_ids():
    left, right, truth = records(1)
    with pytest.raises(ValueError, match="two observed"):
        ev.entity_split(left, right, truth, on="name")
    with pytest.raises(ValueError, match="strictly"):
        ev.entity_split(left, right, truth, on="name", dev_fraction=1)
    with pytest.raises(ValueError, match="absent"):
        ev.entity_split(left, right, pairs([("unknown", "R000")]), on="name")


def test_explicit_unmatched_removes_only_known_counterparts():
    left, right, truth = records(10)
    a, b, gold, unmatched = ev.make_unmatched(left, right, truth, fraction=.2)
    assert (len(a), len(b), len(gold), len(unmatched)) == (8, 8, 6, 4)
    assert set(unmatched.loc[unmatched.side.eq("left"), "id"]) == set(a.id) - set(gold.left_id)
    assert set(unmatched.loc[unmatched.side.eq("right"), "id"]) == set(b.id) - set(gold.right_id)
    with pytest.raises(ValueError, match="complete one-to-one"):
        ev.make_unmatched(left, right, truth.iloc[:5], fraction=.2)
    with pytest.raises(ValueError, match="below"):
        ev.make_unmatched(left, right, truth, fraction=.5)


def test_stage_metrics_separate_blocking_unjudged_and_resolution():
    truth = pairs([("a", "x"), ("b", "y"), ("c", "z")])
    candidates = pairs([("a", "x"), ("a", "y"), ("b", "y")]).assign(sim=.5)
    scores = candidates.assign(p=[.9, .8, np.nan])
    metrics, links = ev.evaluate_stages(candidates, scores, truth, how="many-to-one", threshold=.5,
                                       complete=True)
    assert metrics["candidates"]["recall"] == 2/3
    conditional = metrics["judge_conditional_on_candidates"]
    assert conditional["recall"] == .5 and conditional["fp"] == 1
    assert conditional["unjudged_pairs"] == 1 and conditional["brier"] == pytest.approx(.325)
    assert metrics["final_assignment"]["recall"] == 1/3
    assert metrics["final_assignment"]["precision"] == 1
    assert ev.pair_set(links) == {("a", "x")}
    with pytest.raises(ValueError, match="each candidate"):
        ev.evaluate_stages(candidates, scores.iloc[:2], truth, how="many-to-one", threshold=.5, complete=True)


def test_nber_style_unknown_pairs_never_become_negatives():
    truth = pairs([("a", "x"), ("c", "z")])
    candidates = pairs([("a", "x"), ("b", "y")]).assign(sim=.7)
    metrics, _ = ev.evaluate_stages(candidates, candidates.assign(p=.9), truth,
                                   how="many-to-many", threshold=.5, complete=False)
    for key in ("judge_conditional_on_candidates", "final_assignment"):
        assert metrics[key]["precision"] is None and metrics[key]["f1"] is None
        assert metrics[key]["fp"] is None and metrics[key]["unlisted_predictions_unknown"] == 1
    assert metrics["judge_conditional_on_candidates"]["brier"] is None
    assert metrics["final_assignment"]["recall"] == .5


def test_final_unmatched_counts_records_not_pairs():
    candidates = pairs([("a", "x"), ("a", "y")]).assign(sim=.7)
    labels = pd.DataFrame({"side": ["left", "left", "right"], "id": ["a", "b", "z"]})
    metrics, _ = ev.evaluate_stages(candidates, candidates.assign(p=.9), pairs(), how="many-to-many",
                                   threshold=.5, complete=True, unmatched=labels)
    assert metrics["final_assignment"]["explicit_unmatched"]["left"] == {
        "records": 2, "false_linked": 1, "correct_abstention_rate": .5}
    assert metrics["final_assignment"]["explicit_unmatched"]["right"]["correct_abstention_rate"] == 1


def test_threshold_selection_scores_final_cardinality_and_reject_all():
    scores = pairs([("a", "x"), ("a", "y"), ("b", "x")]).assign(p=[.9, .8, .8], sim=.5)
    truth = pairs([("a", "y"), ("b", "x")])
    threshold, trials = ev.choose_threshold(scores, truth, how="one-to-one", grid=[.5, .85, 1])
    assert threshold == .5 and next(t for t in trials if t["threshold"] == .5)["f1"] == 1
    assert ev.choose_threshold(scores, pairs([("c", "z")]), how="one-to-one", grid=[.5])[0] is None
    with pytest.raises(ValueError, match="empty"):
        ev.choose_threshold(scores, pairs(), how="one-to-one", grid=[.5])
    for grid in ([], [float("nan")], [-.1], [True]):
        with pytest.raises(ValueError, match="grid"):
            ev.choose_threshold(scores, truth, how="one-to-one", grid=grid)


@pytest.mark.parametrize("value", [1.1, -.1, np.inf, -np.inf])
def test_invalid_cached_probabilities_rejected(value):
    with pytest.raises(ValueError, match="between"):
        ev.validate_scores(pairs([("a", "b")]).assign(p=value))


def test_shared_candidate_baselines_and_blocker_extension_hook():
    left, right, truth = records(3)
    candidates = ev.propose(left, right, on=meta()["on"], specs=[{"kind": "exact", "columns": ["name"]}])
    for method in ("exact", "jaro_winkler", "tfidf"):
        scores = ev.string_scores(left, right, candidates, on=meta()["on"], method=method)
        assert ev.pair_set(scores) == ev.pair_set(candidates) == ev.pair_set(truth)
        assert np.allclose(scores.p, 1)
    class AllPairs(ev.block.Blocker):
        name = "custom"
        def pairs(self, a, b):
            return np.array([(i, j) for i in range(len(a)) for j in range(len(b))])
    custom = ev.propose(left, right, on=meta()["on"], specs=[{"kind": "custom"}],
                        registry={"custom": lambda *columns: AllPairs()})
    assert len(custom) == 9
    with pytest.raises(ValueError, match="unknown blocker"):
        ev.propose(left, right, on="name", specs=[{"kind": "made_up"}])


@pytest.mark.skipif(importlib.util.find_spec("recordlinkage") is None, reason="optional bench dependency")
def test_ecm_uses_separate_fields_and_fits_only_development():
    left, right, _ = records(12)
    candidates = pairs([(a, b) for a in left.id for b in right.id])
    features = comparison_vectors(left, right, candidates, on=meta()["on"])
    # Use diverse binary patterns to test real estimator fit/prob without degenerate features.
    features.iloc[:] = np.random.default_rng(12).integers(0, 2, size=features.shape)
    model = ECMBaseline().fit(features)
    before = json.dumps(model.settings, sort_keys=True)
    p = model.score(features.iloc[:8])
    assert len(p) == 8 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert json.dumps(model.settings, sort_keys=True) == before
    with pytest.raises(ValueError, match="multiple"):
        ECMBaseline().fit(features[["name"]])


def test_heldout_roundtrip_reuses_scores_offline_and_does_not_overwrite(tmp_path):
    left, right, truth = records(20)
    write_dataset(tmp_path / "data" / "febrl4", left, right, truth, meta())
    cache = tmp_path / "cache"
    cache.mkdir()
    candidates = ev.propose(left, right, on=meta()["on"], specs=[{"kind": "ngrams", "kwargs": {"k": 3}}])
    candidates.assign(p=[.9 if pair in ev.pair_set(truth) else .1
                         for pair in candidates[["left_id", "right_id"]].itertuples(index=False, name=None)],
                      source="jev").to_csv(cache / "scores.csv", index=False)
    (cache / "settings.json").write_text(json.dumps(dict(meta(), model="offline-fake")))
    report = heldout.run("febrl4", data_dir=tmp_path / "data", out_dir=tmp_path / "out", cached_run=cache,
                         config={"candidate_source": "cached", "threshold_grid": [.3, .5, .7]})
    assert report["new_api_calls"] == 0 and report["new_api_cost_usd"] == 0
    assert report["results"]["cached_jev"]["test"]["final_assignment"]["f1"] == 1
    assert report["results"]["jaro_winkler"]["dev_trials"]
    assert len(report["source_sha256"]) == 4
    assert json.loads((tmp_path / "out" / "febrl4" / "report.json").read_text()) == report
    with pytest.raises(ValueError, match="already"):
        heldout.run("febrl4", data_dir=tmp_path / "data", out_dir=tmp_path / "out")


def test_positive_only_harness_disables_f1_tuning_and_unmatched_fabrication(tmp_path):
    left, right, truth = records()
    write_dataset(tmp_path / "data" / "nber-firms", left, right, truth, meta())
    report = heldout.run("nber-firms", data_dir=tmp_path / "data", out_dir=tmp_path / "out")
    assert report["results"]["tfidf"]["dev_trials"] == []
    assert report["results"]["tfidf"]["test"]["final_assignment"]["f1"] is None
    with pytest.raises(ValueError, match="only for FEBRL"):
        heldout.run("nber-firms", data_dir=tmp_path / "data", out_dir=tmp_path / "unmatched",
                     config={"unmatched_fraction": .1})


def test_resolver_fallback_warnings_are_preserved_in_metrics():
    candidates = pairs([(f"L{i:04}", "R") for i in range(2001)]).assign(sim=.5)
    metrics, links = ev.evaluate_stages(candidates, candidates.assign(p=.9), pairs([("L0000", "R")]),
                                       how="one-to-one", threshold=.5, complete=True)
    assert len(links) == 1
    assert "greedy" in metrics["final_assignment"]["resolver_warnings"][0]


def test_raw_string_scores_are_not_reported_as_calibrated_probabilities():
    candidates = pairs([("a", "x")]).assign(sim=.5)
    metrics, _ = ev.evaluate_stages(candidates, candidates.assign(p=.7, source="jaro_winkler"), candidates,
                                   how="one-to-one", threshold=.5, complete=True, score_is_probability=False)
    assert metrics["judge_conditional_on_candidates"]["brier"] is None
    assert metrics["judge_conditional_on_candidates"]["calibration"] == []
    assert metrics["judge_conditional_on_candidates"]["by_source"]["jaro_winkler"]["tp"] == 1


def test_configured_candidates_missing_from_cache_remain_unjudged(tmp_path):
    left, right, truth = records(12)
    write_dataset(tmp_path / "data" / "febrl4", left, right, truth, meta())
    cache = tmp_path / "cache"
    cache.mkdir()
    # Only exact true pairs were previously scored; the configured pool also includes competitors.
    truth.assign(p=.9, sim=1., source="jev", block="exact").to_csv(cache / "scores.csv", index=False)
    (cache / "settings.json").write_text(json.dumps(dict(meta(), model="offline-fake")))
    report = heldout.run("febrl4", data_dir=tmp_path / "data", out_dir=tmp_path / "out", cached_run=cache,
                         config={"threshold_grid": [.5]})
    stages = report["results"]["cached_jev"]["test"]
    assert stages["judge_conditional_on_candidates"]["unjudged_pairs"] > 0
    assert (stages["judge_conditional_on_candidates"]["scored_pairs"]
            == report["split"]["counts"]["test"]["truth"])

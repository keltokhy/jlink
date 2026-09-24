"""End to end against a fake Jev: block, judge, resolve, report, save and load."""

import json

import pandas as pd
import pytest

import jlink
from fakes import FakeJev

LEFT = pd.DataFrame({
    "gvkey": ["001", "002", "003", "004"],
    "name": ["International Business Machines Corp", "Acme Widgets Inc.", "Zeta Holdings", "Northwind Traders"],
    "state": ["NY", "OH", "CA", "WA"]})
RIGHT = pd.DataFrame({
    "id": [10, 11, 12, 13, 14],
    "firm": ["IBM", "ACME WIDGETS, INC", "Zeta Holdings LLC", "Zeta Capital", "Southwind Traders"],
    "st": ["NY", "OH", "CA", "CA", "WA"]})
TRUE = {("International Business Machines Corp", "IBM"), ("Acme Widgets Inc.", "ACME WIDGETS, INC"),
        ("Zeta Holdings", "Zeta Holdings LLC")}


def oracle(state, question):
    pair = (state["record_a"]["name"], state["record_b"]["name"])
    return 0.96 if pair in TRUE else 0.6 if pair == ("Zeta Holdings", "Zeta Capital") else 0.03


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(name, raising=False)


def linker():
    return jlink.Linker("firm", [("name", "firm"), ("state", "st")], "Subsidiaries are different firms.",
                        blockers=[jlink.block.ngrams(("name", "firm"), k=3), jlink.block.initials(("name", "firm"))])


def run(**kw):
    kw = {"left_id": "gvkey", "right_id": "id", "progress": False} | kw
    return linker().link(LEFT, RIGHT, transport=FakeJev(oracle).transport, **kw)


def test_links_firms_including_the_acronym():
    result = run()
    got = set(zip(result.links["left_id"], result.links["right_id"]))
    assert got == {("001", 10), ("002", 11), ("003", 12)}
    assert result.links["left_id"].is_unique and result.links["right_id"].is_unique
    assert {"p", "sim", "block", "source", "margin"} <= set(result.links.columns)
    ibm = result.scores.query("left_id == '001' and right_id == 10").iloc[0]
    assert "initials" in ibm["block"]


def test_relink_needs_no_new_calls():
    result = run()
    many = result.relink(how="many-to-many", threshold=0.5)
    assert ("003", 13) in set(zip(many.links["left_id"], many.links["right_id"]))
    strict = result.relink(min_margin=0.5)
    assert ("003", 12) not in set(zip(strict.links["left_id"], strict.links["right_id"]))
    assert many.meter is result.meter


def test_merged_report_and_methods():
    result = run()
    merged = result.merged()
    assert len(merged) == 3 and {"name", "firm", "state", "st", "p"} <= set(merged.columns)
    report, methods = result.report(), result.methods()
    assert "Links: 3 (one-to-one" in report and "Rule: Record A and record B refer to the same firm." in report
    assert "Records with no candidate pair: 0 of 4 left, 0 of 5 right" in report
    assert "4 records to 5 records" in methods and "Subsidiaries are different firms." in methods
    assert "one-to-one" not in methods  # prose, not parameter names


def test_save_and_load_round_trip(tmp_path):
    result = run()
    back = jlink.load(result.save(tmp_path / "out"))
    pd.testing.assert_frame_equal(back.links[["left_id", "right_id"]], result.links[["left_id", "right_id"]])
    assert back.links["left_id"].iloc[0] in {"001", "002", "003"}  # leading zeros survive
    assert len(back.relink(how="many-to-many").links) >= 3
    with pytest.raises(ValueError, match="loaded from disk"):
        back.merged()
    assert len(back.merged(LEFT, RIGHT)) == 3


def test_float_ids_survive_save_load_merge_and_review(tmp_path):
    # Stata often stores numeric IDs as doubles, including values no integer column would hold.
    left = LEFT.assign(gvkey=[1001.0, 1002.0, 1003.5, 2.0 ** 60])
    result = linker().link(left, RIGHT, left_id="gvkey", right_id="id", progress=False,
                           transport=FakeJev(oracle).transport)
    directory = result.save(tmp_path / "out")
    assert json.loads((directory / "settings.json").read_text())["id_kinds"] == {
        "left_id": "float", "right_id": "int"}
    back = jlink.load(directory)
    assert back.scores["left_id"].dtype == back.links["left_id"].dtype == "float64"
    pd.testing.assert_frame_equal(back.scores[["left_id", "right_id"]], result.scores[["left_id", "right_id"]])
    pd.testing.assert_frame_equal(back.links[["left_id", "right_id", "p", "margin"]],
                                  result.links[["left_id", "right_id", "p", "margin"]])
    pd.testing.assert_frame_equal(back.merged(left, RIGHT), result.merged())
    assert len(back.merged(left, RIGHT)) == 3
    applied = jlink.create_review(back, left=left, right=RIGHT).apply()
    assert list(zip(applied.links.left_id, applied.links.right_id)) == list(
        zip(result.links.left_id, result.links.right_id))
    assert {type(v) for v in applied.links.left_id} == {float}


def test_float_ids_saved_as_text_by_earlier_versions_still_load_as_text(tmp_path):
    left = LEFT.assign(gvkey=[1001.0, 1002.0, 1003.5, 1004.0])
    directory = linker().link(left, RIGHT, left_id="gvkey", right_id="id", progress=False,
                              transport=FakeJev(oracle).transport).save(tmp_path / "out")
    settings = json.loads((directory / "settings.json").read_text())
    settings["id_kinds"]["left_id"] = "str"
    (directory / "settings.json").write_text(json.dumps(settings))
    assert set(jlink.load(directory).links["left_id"]) == {"1001.0", "1002.0", "1003.5"}


def test_audit_sample_shows_both_records():
    sample = run().audit_sample(n=6)
    assert {"is_match", "a_name", "b_name", "a_state", "b_state", "p", "weight", "selected"} <= set(
        sample.columns)


def test_audit_uses_actual_result_links_after_relink_and_load(tmp_path):
    result = run()
    for current in (result, result.relink(how="many-to-many"), result.relink(min_margin=.5),
                    result.relink(threshold=.99), jlink.load(result.save(tmp_path / "saved"))):
        sample = current.audit_sample(n=len(current.scores))
        expected = set(zip(current.links.left_id, current.links.right_id))
        selected = sample.loc[sample.selected]
        assert set(zip(selected.left_id, selected.right_id)) == expected
        assert sample.selected.dtype == bool
    # Membership comes from the actual table, even if it differs from current settings.
    result.links = result.links.iloc[:0]
    assert not result.audit_sample(n=len(result.scores)).selected.any()


def test_estimate_makes_no_calls():
    est = linker().estimate(LEFT, RIGHT, left_id="gvkey", right_id="id")
    assert est["pairs"] > 0 and est["dollars"] < 0.01


def test_bad_arguments():
    with pytest.raises(ValueError, match="`how` must be one of"):
        run(how="1:1")
    with pytest.raises(ValueError, match="duplicate IDs"):
        linker().link(pd.concat([LEFT, LEFT]), RIGHT, left_id="gvkey", right_id="id", progress=False)


def test_literal_na_ids_survive_save_relink_and_merge(tmp_path):
    left = pd.DataFrame({"key": ["NA", "NULL", "001"], "name": ["Alpha", "Beta", "Gamma"]})
    right = pd.DataFrame({"key": ["NULL", "001", "NA"], "name": ["Alpha", "Beta", "Gamma"]})
    result = jlink.Linker("firm", "name", exact_shortcut=True).link(left, right, left_id="key", right_id="key", progress=False,
                                               transport=FakeJev().transport)
    back = jlink.load(result.save(tmp_path / "out"))
    pd.testing.assert_frame_equal(back.links[["left_id", "right_id"]], result.links[["left_id", "right_id"]])
    assert len(back.relink(threshold=0.9).merged(left, right)) == 3
    assert back.scores["error"].isna().all()


def test_missing_scores_and_literal_errors_are_separate_from_id_strings(tmp_path):
    result = run()
    # Exercise the persisted tables without issuing more requests.
    result.scores = pd.DataFrame({
        "left_id": ["NA", "NULL", "001", ""], "right_id": ["NULL", "001", "NA", "nan"],
        "p": [0.12345678901234568, float("nan"), float("nan"), 0.8], "sim": [0.8, 0.7, 0.6, 0.5],
        "block": ["test"] * 4, "source": ["jev", "error", "unjudged", "error"],
        "error": [pd.NA, "NA", None, "NULL"],
    })
    result = result.relink(threshold=0.1)
    back = jlink.load(result.save(tmp_path / "nulls"))
    assert back.scores["left_id"].tolist() == ["NA", "NULL", "001", ""]
    assert back.scores["right_id"].tolist() == ["NULL", "001", "NA", "nan"]
    assert back.scores["p"].isna().tolist() == [False, True, True, False]
    assert back.scores.loc[0, "p"] == result.scores.loc[0, "p"]
    assert back.scores["error"].isna().tolist() == [True, False, True, False]
    assert back.scores.loc[1, "error"] == "NA" and back.scores.loc[3, "error"] == "NULL"


def test_invalid_budget_is_rejected_before_blocking(monkeypatch):
    configured = linker()

    def blocked(*args, **kwargs):
        pytest.fail("invalid budgets must be rejected before blocking")

    monkeypatch.setattr(configured, "candidates", blocked)
    with pytest.raises(ValueError, match="budget"):
        configured.link(LEFT, RIGHT, budget=-1)


def test_merged_accepts_ids_already_named_left_id_and_right_id(tmp_path):
    left = pd.DataFrame({"left_id": ["NA"], "name": ["Alpha"]})
    right = pd.DataFrame({"right_id": ["NULL"], "name": ["ALPHA"]})
    result = jlink.Linker("firm", "name", exact_shortcut=True).link(left, right, left_id="left_id", right_id="right_id", progress=False)
    back = jlink.load(result.save(tmp_path / "named"))
    assert len(back.relink().merged(left, right)) == 1


@pytest.mark.parametrize("dedupe", [False, True])
def test_estimate_exposes_short_record_assumptions_and_accepts_token_input(dedupe):
    estimator = jlink.Linker("article", "text", blockers=[jlink.block.exact("group")])
    short = pd.DataFrame({"text": ["a"] * 10, "group": [1] * 10})
    long = short.assign(text="a" * 50_000)
    estimates = [estimator.estimate(frame, *([] if dedupe else [frame])) for frame in (short, long)]
    assert estimates[0]["dollars"] == estimates[1]["dollars"]  # pair-count scenario, no length inference
    assumptions = {"tokens_per_pair": 330, "price_per_million_tokens": 0.042, "pairs_per_second": 200,
                   "token_basis": "short_records", "throughput_basis": "short_records"}
    assert estimates[0]["assumptions"] == estimates[1]["assumptions"] == assumptions
    supplied = estimator.estimate(long, *([] if dedupe else [long]), tokens_per_pair=10_000)
    assert supplied["dollars"] == round(supplied["pairs"] * 10_000 * 0.042 / 1e6, 4)
    assert supplied["assumptions"] == assumptions | {"tokens_per_pair": 10_000, "token_basis": "caller_supplied"}
    assert supplied["seconds"] == estimates[0]["seconds"]  # no new throughput calibration


@pytest.mark.parametrize("tokens", [True, 0, -1, float("nan"), float("inf"), "330"])
def test_estimate_rejects_invalid_token_assumptions(tokens):
    with pytest.raises(ValueError, match="tokens_per_pair"):
        linker().estimate(LEFT, RIGHT, tokens_per_pair=tokens)


def test_default_identity_link_retains_main_saved_settings(tmp_path):
    result = run()
    assert "style" not in result.settings  # absence means identity, as in origin/main
    directory = result.save(tmp_path / "run")
    assert "style" not in json.loads((directory / "settings.json").read_text())
    back = jlink.load(directory)
    assert back.settings == result.settings and back.report() == result.report()
    assert back.methods() == result.methods()


def test_default_blocking_searches_both_directions_and_report_counts_unpaired_records():
    # Eleven closer names crowd "Acme" out of "Acme Corp"'s forward ten; searching from "Acme" finds it.
    left = pd.DataFrame({"name": ["Acme Corp", "Zzyzx"]})
    right = pd.DataFrame({"firm": [f"Acme Corp {i}" for i in range(11)] + ["Acme"]})
    on = [("name", "firm")]
    forward = jlink.block.candidates(left, right, on=on, blockers=[jlink.block.ngrams(*on, k=10)])
    assert (0, 11) not in set(zip(forward.left_id, forward.right_id))
    result = jlink.Linker("firm", on).link(left, right, progress=False,
                                           transport=FakeJev(lambda state, question: 0.1).transport)
    assert result.settings["blockers"] == ["ngrams:name=firm", "ngrams-reverse:name=firm"]
    assert (0, 11) in set(zip(result.scores.left_id, result.scores.right_id))
    assert ("Records with no candidate pair: 1 of 2 left, 0 of 12 right (these cannot be linked"
            in result.report())

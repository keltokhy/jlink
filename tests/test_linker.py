"""End to end against a fake Jev: block, judge, resolve, report, save and load."""

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


def test_audit_sample_shows_both_records():
    sample = run().audit_sample(n=6)
    assert {"is_match", "a_name", "b_name", "a_state", "b_state", "p", "weight"} <= set(sample.columns)


def test_estimate_makes_no_calls():
    est = linker().estimate(LEFT, RIGHT, left_id="gvkey", right_id="id")
    assert est["pairs"] > 0 and est["dollars"] < 0.01


def test_bad_arguments():
    with pytest.raises(ValueError, match="`how` must be one of"):
        run(how="1:1")
    with pytest.raises(ValueError, match="duplicate IDs"):
        linker().link(pd.concat([LEFT, LEFT]), RIGHT, left_id="gvkey", right_id="id", progress=False)

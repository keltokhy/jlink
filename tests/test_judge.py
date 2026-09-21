import warnings

import numpy as np
import pandas as pd
import pytest

from fakes import FakeJev
from jlink.core import JevFatal
from jlink.judge import judge, question

LEFT = pd.DataFrame({"gvkey": ["a1", "a2", "a3"],
                     "name": ["International Business Machines", "Acme Widgets Inc.", "Zeta Holdings"],
                     "year": [1985.0, np.nan, 2001.0]})
RIGHT = pd.DataFrame({"id": [10, 11, 12], "firm": ["IBM Corp", "ACME WIDGETS, INC", "Omega Partners"],
                      "year": [1985, 1990, 2001]})
ON = [("name", "firm"), "year"]


def pairs(rows):
    return pd.DataFrame(rows, columns=["left_id", "right_id", "sim"]).assign(block="test")


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(name, raising=False)


def run(cands, fake, **kw):
    kw = {"on": ON, "entity": "firm", "left_id": "gvkey", "right_id": "id", "progress": False} | kw
    return judge(cands, LEFT, RIGHT, transport=fake.transport, **kw)


def test_states_questions_and_scores():
    fake = FakeJev(lambda state, q: 0.97 if "IBM" in state["record_b"]["name"] else 0.02)
    scores, meter = run(pairs([("a1", 10, 0.1), ("a3", 12, 0.3)]), fake, definition="Subsidiaries are different firms.")
    assert list(scores["p"]) == [0.97, 0.02] and list(scores["source"]) == ["jev", "jev"]
    assert meter.calls == 2
    # judged in descending similarity; labels come from the left column names; 1985.0 is sent as 1985
    assert fake.bodies[0]["state"] == {"record_a": {"name": "Zeta Holdings", "year": 2001},
                                       "record_b": {"name": "Omega Partners", "year": 2001}}
    assert fake.bodies[1]["state"]["record_a"] == {"name": "International Business Machines", "year": 1985}
    assert fake.bodies[0]["questions"]["match"] == question("firm", "Subsidiaries are different firms.")
    assert "Record A and record B refer to the same firm. Subsidiaries" in question("firm", "Subsidiaries x")["instructions"]


def test_missing_fields_are_dropped_from_the_state():
    fake = FakeJev()
    run(pairs([("a2", 12, 0.2)]), fake)
    assert fake.bodies[0]["state"]["record_a"] == {"name": "Acme Widgets Inc."}


def test_explicitly_trusted_identical_records_cost_nothing():
    left = pd.DataFrame({"name": ["Acme Widgets, Inc."]}, index=["x"])
    right = pd.DataFrame({"name": ["ACME WIDGETS INC"]}, index=["y"])
    fake = FakeJev()
    scores, meter = judge(pairs([("x", "y", 1.0)]), left, right, on="name", entity="firm",
                          progress=False, transport=fake.transport, exact_shortcut=True)
    assert scores.loc[0, "p"] == 1.0 and scores.loc[0, "source"] == "exact" and not fake.bodies


def test_equal_names_are_judged_by_default_and_can_be_rejected():
    frame = pd.DataFrame({"name": ["John Smith"]})
    fake = FakeJev(lambda state, rule: .02)
    scores, meter = judge(pairs([(0, 0, 1.0)]), frame, frame, on="name", entity="person",
                          definition="Identical names alone do not establish identity.",
                          progress=False, transport=fake.transport)
    assert scores.p.tolist() == [.02] and scores.source.tolist() == ["jev"]
    assert meter.calls == 1


def test_equal_names_with_no_budget_remain_unresolved_by_default():
    frame = pd.DataFrame({"name": ["John Smith"]})
    fake = FakeJev()
    with pytest.warns(UserWarning, match="budget ran out"):
        scores, meter = judge(pairs([(0, 0, 1.0)]), frame, frame, on="name", entity="person",
                              budget=0, progress=False, transport=fake.transport)
    assert scores.source.tolist() == ["unjudged"] and scores.p.isna().all()
    assert not fake.bodies and meter.calls == 0


def test_budget_leaves_the_least_similar_pairs_unjudged():
    fake = FakeJev(cost=0.01)
    cands = pairs([("a1", 10, 0.1), ("a1", 11, 0.9), ("a2", 11, 0.8), ("a3", 12, 0.2), ("a3", 10, 0.05)])
    with pytest.warns(UserWarning, match="budget ran out"):
        scores, _ = run(cands, fake, budget=0.02, concurrency=1)
    judged = scores[scores["source"] == "jev"]
    assert set(judged["sim"]) == {0.9, 0.8} and scores["p"].isna().sum() == 3


def test_second_run_is_free():
    cands = pairs([("a1", 10, 0.1), ("a2", 11, 0.9)])
    run(cands, FakeJev())
    fake = FakeJev()
    scores, meter = run(cands, fake)
    assert not fake.bodies and meter.calls == 0 and meter.cached == 2 and scores["p"].notna().all()


def test_one_failed_pair_is_recorded_not_raised():
    import httpx

    class Flaky(FakeJev):
        def __call__(self, request):
            if b"Zeta" in request.content:
                return httpx.Response(422, json={"detail": [{"msg": "state too large"}]})
            return super().__call__(request)

    with pytest.warns(UserWarning, match="1 pairs failed"):
        scores, _ = run(pairs([("a1", 10, 0.5), ("a3", 12, 0.4)]), Flaky())
    assert list(scores["source"]) == ["jev", "error"] and "state too large" in scores.loc[1, "error"]


def test_bad_key_raises_and_stops_calling():
    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "No auth credentials found", "code": 401}})

    cands = pairs([("a1", 10, s / 100) for s in range(1)] + [("a2", 11, 0.5), ("a3", 12, 0.4)])
    with pytest.raises(JevFatal, match="401"):
        judge(cands, LEFT, RIGHT, on=ON, entity="firm", left_id="gvkey", right_id="id", progress=False,
              concurrency=1, transport=httpx.MockTransport(handler))
    assert len(calls) <= 2


def test_input_errors_name_the_problem():
    fake = FakeJev()
    with pytest.raises(ValueError, match="no column 'nme'"):
        run(pairs([("a1", 10, 0.1)]), fake, on=[("nme", "firm")])
    with pytest.raises(ValueError, match="not in the right data"):
        run(pairs([("a1", 99, 0.1)]), fake)
    with pytest.raises(ValueError, match="entity"):
        run(pairs([("a1", 10, 0.1)]), fake, entity=" ")


def test_runs_inside_a_running_event_loop():
    import asyncio

    async def notebook_cell():
        return run(pairs([("a1", 10, 0.1)]), FakeJev(lambda s, q: 0.8))

    scores, _ = asyncio.run(notebook_cell())
    assert scores.loc[0, "p"] == 0.8


def test_exact_shortcut_preserves_field_boundaries():
    left = pd.DataFrame({"first": ["Mary Ann"], "last": ["Smith"]})
    right = pd.DataFrame({"first": ["Mary"], "last": ["Ann Smith"]})
    fake = FakeJev(lambda s, q: 0.12)
    scores, _ = judge(pairs([(0, 0, 1.0)]), left, right, on=["first", "last"], entity="person",
                      progress=False, transport=fake.transport, exact_shortcut=True)
    assert scores.loc[0, "source"] == "jev"
    assert scores.loc[0, "p"] == 0.12 and len(fake.bodies) == 1


def test_zero_budget_keeps_cached_and_exact_scores_without_paid_calls():
    cached = pairs([("a1", 10, 0.1)])
    run(cached, FakeJev(lambda s, q: 0.8), on=[("name", "firm")])
    fake = FakeJev()
    # The cache hit comes after a miss in descending similarity order.
    cands = pairs([("a3", 12, 0.9), ("a2", 11, 1.0), ("a1", 10, 0.1)])
    with pytest.warns(UserWarning, match="budget ran out"):
        scores, meter = run(cands, fake, on=[("name", "firm")], budget=0, exact_shortcut=True)
    assert not fake.bodies and meter.calls == 0 and meter.cached == 1
    assert scores["source"].tolist() == ["unjudged", "exact", "jev"]
    assert scores.loc[2, "p"] == 0.8 and pd.isna(scores.loc[0, "p"])


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, pd.NaT, np.datetime64("NaT", "ns"),
                                     "", "   ", "---"])
def test_incomplete_records_are_never_accepted_as_exact(missing):
    left = pd.DataFrame({"first": ["Mary"], "last": [missing]})
    right = pd.DataFrame({"first": ["MARY"], "last": [missing]})
    fake = FakeJev()
    scores, _ = judge(pairs([(0, 0, 1.0)]), left, right, on=["first", "last"], entity="person",
                      progress=False, transport=fake.transport, exact_shortcut=True)
    assert scores.loc[0, "source"] == "jev" and len(fake.bodies) == 1


def test_normalized_complete_fields_still_match_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    left = pd.DataFrame({"name": ["Ácme & Sons, Inc."], "year": [1985.0]})
    right = pd.DataFrame({"name": ["ACME and SONS INC"], "year": [1985]})
    scores, meter = judge(pairs([(0, 0, 1.0)]), left, right, on=["name", "year"], entity="firm",
                          progress=False, exact_shortcut=True)
    assert scores.loc[0, "source"] == "exact" and meter.calls == 0


@pytest.mark.parametrize("budget", [-1, float("inf"), -float("inf"), float("nan"), True, False, "0"])
def test_invalid_budgets_fail_before_api_setup(budget, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ValueError, match="budget.*finite nonnegative"):
        run(pairs([("a1", 10, 0.1)]), FakeJev(), budget=budget)


def test_none_budget_is_explicitly_unlimited():
    scores, meter = run(pairs([("a1", 10, 0.1), ("a3", 12, 0.2)]), FakeJev(cost=10.0),
                        budget=None, concurrency=1)
    assert scores["p"].notna().all() and meter.cost == 20.0


def test_positive_exhausted_budget_still_reads_later_cache_hits():
    run(pairs([("a1", 10, 0.1)]), FakeJev(lambda s, q: 0.7))
    fake = FakeJev(cost=0.02)
    with pytest.warns(UserWarning, match="budget ran out"):
        scores, meter = run(pairs([("a3", 12, 0.9), ("a2", 11, 0.8), ("a1", 10, 0.1)]), fake,
                            budget=0.01, concurrency=1)
    assert scores["source"].tolist() == ["jev", "unjudged", "jev"]
    assert scores.loc[2, "p"] == 0.7 and meter.cached == 1 and len(fake.bodies) == 1


def test_in_flight_calls_finish_and_may_overshoot_positive_budget():
    import asyncio
    import httpx

    fake = FakeJev(cost=0.02)

    async def handler(request):
        await asyncio.sleep(0.01)
        return fake(request)

    cands = pairs([("a1", 10, 0.9), ("a1", 11, 0.8), ("a3", 12, 0.7), ("a3", 10, 0.6)])
    with pytest.warns(UserWarning, match="budget ran out"):
        scores, meter = judge(cands, LEFT, RIGHT, on=ON, entity="firm", left_id="gvkey", right_id="id",
                              budget=0.001, concurrency=2, progress=False, transport=httpx.MockTransport(handler))
    assert meter.calls == 2 and meter.cost == 0.04
    assert scores["source"].tolist() == ["jev", "jev", "unjudged", "unjudged"]


def test_zero_budget_can_read_cache_without_credentials(monkeypatch):
    cands = pairs([("a1", 10, 0.1)])
    run(cands, FakeJev(), api="openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    scores, meter = run(cands, FakeJev(), api="openrouter", budget=0)
    assert meter.calls == 0 and meter.cached == 1 and scores.loc[0, "source"] == "jev"


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_bad_concurrency_fails_instead_of_hanging(concurrency):
    with pytest.raises(ValueError, match="concurrency.*positive integer"):
        run(pairs([("a1", 10, 0.1)]), FakeJev(), concurrency=concurrency)

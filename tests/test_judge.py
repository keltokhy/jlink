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


def test_identical_records_cost_nothing():
    left = pd.DataFrame({"name": ["Acme Widgets, Inc."]}, index=["x"])
    right = pd.DataFrame({"name": ["ACME WIDGETS INC"]}, index=["y"])
    fake = FakeJev()
    scores, meter = judge(pairs([("x", "y", 1.0)]), left, right, on="name", entity="firm",
                          progress=False, transport=fake.transport)
    assert scores.loc[0, "p"] == 1.0 and scores.loc[0, "source"] == "exact" and not fake.bodies


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

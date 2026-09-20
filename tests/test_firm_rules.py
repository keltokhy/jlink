import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.firm_rules import ARMS, baselines, evaluate, load_fixture, requests, run
from fakes import FakeJev
from jlink.judge import judge


def saved_rows(fixture):
    pairs = {p["pair_id"]: p for p in fixture["pairs"]}
    rows = []
    for req in requests(fixture):
        gold = pairs[req["pair_id"]]["labels"][req["rule_id"]]
        response = {"match": {"type": "noul", "noul": .9 if gold else .1}}
        rows.append({"request_sha256": req["request_sha256"], "model": req["payload"]["model"],
                     "recorded_at": "2026-09-20T00:00:00Z", "response": response,
                     "provider_response": {"answers": response}, "evidence_kind": "offline_fake"})
    return rows


def test_requests_do_not_expose_labels_or_annotations_and_skip_unknowns():
    fixture = load_fixture()
    prepared = requests(fixture)
    known = sum(v is not None for p in fixture["pairs"] for v in p["labels"].values())
    assert len(prepared) == len(ARMS) * known
    assert len({r["request_sha256"] for r in prepared}) == len(prepared)
    for req in prepared:
        state = req["payload"]["state"]
        assert set(state) == {"record_a", "record_b"}
        for record in state.values():
            assert not set(record) & {"labels", "rationale", "source_ids", "pair_id", "split", "group_id"}
            if req["arm"] == "names_and_dates":
                assert set(record) == {"name", "as_of"}


def test_perfect_predictions_rule_flips_and_unknown_denominators():
    fixture = load_fixture()
    rows = saved_rows(fixture)
    report = evaluate(fixture, rows)
    assert report["status"] == "offline_fake" and report["missing_responses"] == 0
    assert report["unknown_labels_excluded_per_arm"] > 0
    for split in report["metrics"].values():
        for arm in split.values():
            assert arm["overall"]["accuracy"] == 1
            assert arm["overall"]["coverage"] == 1
            flips = arm["rule_changing_pairs"]
            assert flips["n"] == flips["all_known_rules_correct"]
    partial = evaluate(fixture, rows[:1])
    overall = partial["metrics"]["dev"]["rich"]["overall"]
    assert overall["accuracy"] == 1 and overall["coverage"] < 1
    assert overall["correct_fraction_of_expected"] < 1


def test_response_tampering_duplicates_mixed_evidence_and_nonfinite_rejected():
    fixture = load_fixture()
    rows = saved_rows(fixture)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(fixture, rows + rows[:1])
    broken = copy.deepcopy(rows[:1])
    broken[0]["response"]["match"]["noul"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate(fixture, broken)
    broken = copy.deepcopy(rows[:2])
    broken[0]["evidence_kind"] = "live_model"
    with pytest.raises(ValueError, match="pool"):
        evaluate(fixture, broken)
    broken = copy.deepcopy(rows[:1])
    broken[0]["provider_response"] = {"answers": {}}
    with pytest.raises(ValueError, match="preserved"):
        evaluate(fixture, broken)


def test_group_split_and_boolean_labels_are_validated(tmp_path):
    fixture = load_fixture()
    fixture["pairs"][-1]["group_id"] = fixture["pairs"][0]["group_id"]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match="disjoint"):
        load_fixture(path)
    fixture = load_fixture()
    fixture["pairs"][0]["labels"]["legal_entity"] = 1
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match="booleans"):
        load_fixture(path)


def test_frozen_requests_match_public_judge_payload(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    fixture = load_fixture()
    req = requests(fixture)[0]
    fake = FakeJev()
    state = req["payload"]["state"]
    columns = list(state["record_a"])
    left = pd.DataFrame([dict(state["record_a"], id="a")])
    right = pd.DataFrame([dict(state["record_b"], id="b")])
    candidates = pd.DataFrame({"left_id": ["a"], "right_id": ["b"], "sim": [0.], "block": ["given"]})
    judge(candidates, left, right, on=columns, entity="firm", definition=fixture["rules"][req["rule_id"]],
          left_id="id", right_id="id", api="openrouter", model=fixture["model_settings"]["model"],
          exact_shortcut=False, cache=False, progress=False, transport=fake.transport)
    assert fake.bodies == [req["payload"]]


def test_baseline_threshold_uses_development_only_and_rule_blind_ceiling():
    fixture = load_fixture()
    baseline = baselines(fixture)
    edited = copy.deepcopy(fixture)
    for pair in edited["pairs"]:
        if pair["split"] == "test":
            pair["labels"] = {k: not v if v is not None else None for k, v in pair["labels"].items()}
    assert baseline["threshold"] == baselines(edited)["threshold"]
    for part in baseline["partitions"].values():
        assert part["name_tfidf"]["correct"] <= part["rule_blind_oracle_correct"] < part["n"]


def test_offline_default_and_budget_limited_fake_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.delenv("JEV_URL", raising=False)
    offline = run(out=tmp_path / "frozen")
    assert offline["status"] == "pending" and offline["new_api_calls"] == 0
    fake = FakeJev(cost=.01)
    report = run(out=tmp_path / "fake", live=True, budget=.025, transport=fake.transport)
    assert len(fake.bodies) == 3
    assert report["status"] == "offline_fake" and report["missing_responses"] > 0
    assert report["new_api_calls"] == 0 and report["new_api_cost_usd"] == 0
    assert (tmp_path / "fake/frozen.json").is_file()
    assert (tmp_path / "fake/responses.jsonl").is_file()


@pytest.mark.parametrize("kwargs", [{"budget": 1}, {"live": True}, {"live": True, "budget": None}])
def test_spending_requires_live_and_positive_finite_budget(tmp_path, kwargs):
    with pytest.raises(ValueError, match="budget"):
        run(out=tmp_path, **kwargs)

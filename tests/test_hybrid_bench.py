import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data import write_dataset
from bench.hybrid import retrieval_metrics, run, union_candidates
from fakes import FakeJev
from test_embeddings import Encoder


@pytest.fixture
def corpus(tmp_path):
    left = pd.DataFrame({"id": [f"l{i}" for i in range(20)], "name": [f"Firm {i}" for i in range(20)]})
    right = pd.DataFrame({"id": [f"r{i}" for i in range(20)], "name": [f"Company {i}" for i in range(20)]})
    truth = pd.DataFrame({"left_id": left.id, "right_id": right.id})
    metadata = {"entity": "firm", "definition": "same firm", "on": ["name"], "how": "many-to-one"}
    root = tmp_path / "data"
    write_dataset(root / "nber-firms", left, right, truth, metadata)
    lookup = {name: [int(j == i) for j in range(20)] for i in range(20)
              for name in (f"Firm {i}", f"Company {i}")}
    return root, Encoder(lookup)


def test_split_retrieval_artifacts_and_incomplete_gold_policy(corpus, tmp_path, monkeypatch):
    root, encoder = corpus
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(name, raising=False)
    fake = FakeJev(lambda s, q: .8)
    report = run("nber-firms", out=tmp_path / "run", data_dir=root, encoder=encoder, k=1,
                 live=True, budget=.1, transport=fake.transport)
    assert report["new_api_calls"] > 0 and report["new_api_cost_usd"] < .1
    for partition in ("dev", "test"):
        entry = report["partitions"][partition]
        assert entry["retrieval"]["hybrid"]["candidate_recall"] == 1
        assert entry["retrieval"]["hybrid"]["lost_matches"] == 0
        assert entry["same_judge_comparison"]["hybrid"]["final_assignment"]["precision"] is None
    json.dumps(report, allow_nan=False)
    assert (tmp_path / "run/test-union-scores.csv").is_file()
    split = pd.read_csv(tmp_path / "run/split.csv")
    assert split.groupby("group").split.nunique().max() == 1
    with pytest.raises(ValueError, match="already contains"):
        run("nber-firms", out=tmp_path / "run", data_dir=root, encoder=encoder)


def test_no_judging_is_default(corpus, tmp_path, monkeypatch):
    root, encoder = corpus
    import bench.hybrid as module

    monkeypatch.setattr(module, "judge", lambda *a, **kw: pytest.fail("no API setup in retrieval-only mode"))
    report = run("nber-firms", out=tmp_path / "run", data_dir=root, encoder=encoder, k=1)
    assert report["judge_status"] == "not_run" and report["new_api_calls"] == 0
    assert "same_judge_comparison" not in report["partitions"]["test"]


def test_union_retains_zero_lexical_score_for_semantic_candidates():
    a = pd.DataFrame({"left_id": ["a"], "right_id": ["b"], "sim": [0.], "block": ["semantic"]})
    b = a.assign(block="lexical")
    result = union_candidates(a, b)
    assert len(result) == 1 and result.sim.iloc[0] == 0
    assert result.block.iloc[0] == "semantic+lexical"
    assert retrieval_metrics(result, a, b)["recovered_matches"] == 0


@pytest.mark.parametrize("kwargs", [{"budget": 1}, {"live": True}, {"live": True, "budget": None}])
def test_spending_requires_explicit_live_and_finite_budget(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run("nber-firms", out=tmp_path / "absent", **kwargs)
    assert not (tmp_path / "absent").exists()

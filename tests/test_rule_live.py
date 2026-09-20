from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.rule_live import run
from fakes import FakeJev


def test_live_requires_explicit_positive_budget(tmp_path):
    for kwargs in ({}, {"budget": .1}, {"live": True}, {"live": True, "budget": None}):
        with pytest.raises(ValueError):
            run(out=tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()


def test_fake_responses_are_bound_saved_and_never_called_live(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.delenv("JEV_URL", raising=False)
    fake = FakeJev(lambda s, q: .8, cost=.01)
    report = run(out=tmp_path / "out", live=True, budget=.025, transport=fake.transport)
    assert len(fake.bodies) == 3
    assert report["status"] == "offline_fake_only" and report["missing_responses"] == 33
    assert report["new_api_calls"] == 0 and report["new_api_cost_usd"] == 0
    assert (tmp_path / "out/responses.jsonl").is_file()
    assert "provider_response" in (tmp_path / "out/responses.jsonl").read_text()

"""Cross-feature checks for saved runs, blocking, review, and final-link evaluation."""

import importlib
import json
from pathlib import Path

import pandas as pd

import jlink
from fakes import FakeJev

cli = importlib.import_module("jlink.cli").cli


def test_review_cli_preserves_probability_at_acceptance_boundary(tmp_path):
    left = pd.DataFrame({"id": ["NA"], "name": ["Alpha"]})
    right = pd.DataFrame({"id": ["NULL"], "name": ["Beta"]})
    scores = pd.DataFrame({"left_id": ["NA"], "right_id": ["NULL"],
                           "p": [0.9999999999999999], "sim": [0.8],
                           "block": ["fixture"], "source": ["jev"], "error": [None]})
    settings = dict(on=[["name", "name"]], left_id="id", right_id="id",
                    how="one-to-one", threshold=1.0, min_margin=None)
    original = jlink.Result(jlink.resolve(scores, threshold=1.0), scores, settings)
    original.save(tmp_path / "run")
    left.to_csv(tmp_path / "left.csv", index=False)
    right.to_csv(tmp_path / "right.csv", index=False)
    cli(["review", "create", str(tmp_path / "run"), "--left", str(tmp_path / "left.csv"),
         "--right", str(tmp_path / "right.csv"), "-o", str(tmp_path / "review.html"),
         "--artifact", str(tmp_path / "review.json")])
    review = jlink.read_review(tmp_path / "review.json")
    assert review.candidates.p.iloc[0] == scores.p.iloc[0]
    assert review.apply().links.empty  # Merely reopening a saved run must not select a new link.


def test_command_line_alone_reaches_the_review_workflow(tmp_path, monkeypatch, capsys):
    """link --save writes what Result.save() writes, so review create needs no Python session."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for key in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(key, raising=False)
    fake = FakeJev(lambda state, question: 0.9 if state["record_a"]["name"][:4] == state["record_b"]["name"][:4]
                   else 0.1)
    linker = importlib.import_module("jlink.linker")
    real = linker.judge
    monkeypatch.setattr(linker, "judge", lambda *a, **kw: real(*a, **{**kw, "transport": fake.transport}))
    pd.DataFrame({"id": ["001", "NA"], "name": ["Acme Inc", "Zeta LLC"]}).to_csv(tmp_path / "left.csv", index=False)
    pd.DataFrame({"id": ["7", "8"], "name": ["Acme Incorporated", "Zeta Holdings"]}).to_csv(
        tmp_path / "right.csv", index=False)
    run = tmp_path / "linkage"
    cli(["link", str(tmp_path / "left.csv"), str(tmp_path / "right.csv"), "--on", "name", "--entity", "firm",
         "--left-id", "id", "--right-id", "id", "--block", "ngrams:name:2", "--no-cache",
         "-o", str(tmp_path / "links.csv"), "--save", str(run)])
    assert sorted(p.name for p in run.iterdir()) == ["links.csv", "scores.csv", "settings.json"]
    saved = jlink.load(run)
    assert set(zip(saved.links.left_id, saved.links.right_id)) == {("001", "7"), ("NA", "8")}
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "links.csv", dtype=str, keep_default_na=False)[
        ["left_id", "right_id"]], saved.links[["left_id", "right_id"]])
    assert saved.settings["question"] == fake.bodies[0]["questions"]["match"]["instructions"]
    assert saved.settings["inputs"]["left"]["compared"]["columns"] == ["name"]
    calls = len(fake.bodies)
    cli(["review", "create", str(run), "--left", str(tmp_path / "left.csv"), "--right", str(tmp_path / "right.csv"),
         "-o", str(tmp_path / "review.html"), "--artifact", str(tmp_path / "review.json")])
    review = jlink.read_review(tmp_path / "review.json")
    review.decide("NA", "8", "reject", reviewer="cli-only")
    review.save(tmp_path / "decided.json")
    cli(["review", "apply", str(tmp_path / "decided.json"), "-o", str(tmp_path / "reviewed.json")])
    assert "1 reviewed links" in capsys.readouterr().err and len(fake.bodies) == calls


def test_grouped_cached_run_survives_review_and_selected_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for key in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(key, raising=False)
    left = pd.DataFrame({"id": ["NA", "001", "orphan"],
                         "name": ["Acme", "Alfa", "Unpaired"], "state": ["NY", "NY", "XX"]})
    right = pd.DataFrame({"id": ["NULL", "002"], "name": ["Acme", "Alpha"], "region": ["NY", "NY"]})
    blocker = jlink.block.within(jlink.block.ngrams("name", k=2, min_sim=0, reverse=True),
                                 ("state", "region"))
    fake = FakeJev(lambda state, question: 0.8 if state["record_a"]["name"] == "Alfa"
                  and state["record_b"]["name"] == "Alpha" else 0.1)
    linker = jlink.Linker("firm", "name", blockers=[blocker], api="openrouter",
                          model="offline-pinned", cache=tmp_path / "answers.sqlite", exact_shortcut=True)
    first = linker.link(left, right, left_id="id", right_id="id", transport=fake.transport,
                        progress=False)
    calls = len(fake.bodies)
    cached = linker.link(left, right, left_id="id", right_id="id", transport=fake.transport,
                         budget=0, progress=False)
    assert len(fake.bodies) == calls and cached.meter.calls == 0
    assert cached.settings["resolved_models"] == ["offline-pinned"]
    assert cached.settings["blocker_configs"] == [blocker.to_config()]
    restored = jlink.load(cached.save(tmp_path / "run"))
    assert restored.candidates.attrs["blocking"] == cached.settings["blocking"]
    pd.testing.assert_series_equal(restored.scores.p, first.scores.p)
    assert len(restored.merged(left, right)) == 2

    review = jlink.create_review(restored, left=left, right=right)
    review.decide("001", "002", "reject", reviewer="integration-test")
    reviewed = jlink.read_review(review.save(tmp_path / "review.json")).apply()
    assert set(zip(reviewed.links.left_id, reviewed.links.right_id)) == {("NA", "NULL")}
    pd.testing.assert_series_equal(reviewed.scores.p, restored.scores.p)
    assert reviewed.review["source"]["settings"]["inputs"] == cached.settings["inputs"]
    sample = jlink.audit_sample(reviewed.scores, links=reviewed.links, n=len(reviewed.scores))
    truth = {("NA", "NULL"), ("001", "002")}
    sample["is_match"] = [int(pair in truth) for pair in zip(sample.left_id, sample.right_id)]
    evaluation = jlink.evaluate(sample, mode="selected", n_boot=20)
    assert evaluation.precision[0] == 1.0 and evaluation.recall[0] == 0.5
    assert jlink.evaluate(sample, n_boot=20).recall[0] == 1.0
    assert int(reviewed.selected_scores().selected.sum()) == 1


def test_benchmark_registry_supports_integrated_grouped_reverse_search(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from bench.evaluation import propose

    left = pd.DataFrame({"id": ["a", "b"], "name": ["Alpha", "Alpha"], "state": ["NY", "CA"]})
    right = pd.DataFrame({"id": ["x", "y"], "name": ["Alfa", "Alfa"], "state": ["CA", "NY"]})
    candidates = propose(left, right, on=["name"], specs=[{
        "kind": "within_state", "columns": ["name"], "kwargs": {"k": 1, "reverse": True}}],
        registry={"within_state": lambda *cols, **kw: jlink.block.within(
            jlink.block.ngrams(*cols, **kw), "state")})
    assert set(zip(candidates.left_id, candidates.right_id)) == {("a", "y"), ("b", "x")}
    config = candidates.attrs["blocking"]["passes"][0]["config"]
    assert config["type"] == "within" and config["blocker"]["reverse"] is True
    json.dumps(candidates.attrs["blocking"], allow_nan=False)


def test_resume_finishes_a_saved_run_from_the_command_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for key in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(key, raising=False)
    fake = FakeJev(lambda state, question: 0.9 if state["record_a"]["name"][:4] == state["record_b"]["name"][:4]
                   else 0.1)
    linker = importlib.import_module("jlink.linker")
    real = linker.judge
    monkeypatch.setattr(linker, "judge", lambda *a, **kw: real(*a, **{**kw, "transport": fake.transport}))
    left, right, run = tmp_path / "left.csv", tmp_path / "right.csv", tmp_path / "linkage"
    pd.DataFrame({"id": ["001", "NA"], "name": ["Acme Inc", "Zeta LLC"]}).to_csv(left, index=False)
    pd.DataFrame({"id": ["7", "8"], "name": ["Acme Incorporated", "Zeta Holdings"]}).to_csv(right, index=False)
    cli(["link", str(left), str(right), "--on", "name", "--entity", "firm", "--left-id", "id",
         "--right-id", "id", "--no-cache", "-j", "1", "--budget", "0.00001", "-o", str(tmp_path / "links.csv"),
         "--save", str(run)])
    cut = jlink.load(run)
    first = len(fake.bodies)
    assert 0 < first < len(cut.scores) and (cut.scores["source"] == "unjudged").any()
    cli(["resume", str(run), str(left), str(right), "--no-cache"])
    assert "0 pairs still unjudged or failed" in capsys.readouterr().err
    done = jlink.load(run)
    assert len(fake.bodies) == len(cut.scores)  # each pair was sent once across both commands
    assert (done.scores["source"] == "jev").all() and len(done.settings["resumes"]) == 1
    assert set(zip(done.links.left_id, done.links.right_id)) == {("001", "7"), ("NA", "8")}

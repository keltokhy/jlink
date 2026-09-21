"""Saved runs explain configuration, input identity, and mixed/legacy model origins offline."""

import json
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd
import pytest

import jlink
from fakes import FakeJev
from jlink.core import Cache
from jlink.judge import question
from jlink.provenance import blocker_config, frame_fingerprint


class ResolvedJev(FakeJev):
    def __init__(self, resolved="jev-1"):
        super().__init__(lambda s, q: 0.8)
        self.resolved = resolved

    def __call__(self, request):
        data = super().__call__(request).json()
        data["model"] = self.resolved
        return httpx.Response(200, json=data)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(name, raising=False)


def run(*, left=None, fake=None, **kwargs):
    left = pd.DataFrame({"name": ["Alpha", "Beta"]}, index=["NA", "001"]) if left is None else left
    right = pd.DataFrame({"name": ["Alpha LLC", "Beta Corp"]}, index=["NULL", "002"])
    linker = jlink.Linker("firm", "name", api="openrouter", model="jev-alias", exact_shortcut=False,
                         blockers=[jlink.block.ngrams("name", k=2, min_sim=0)])
    return linker.link(left, right, progress=False, transport=(fake or ResolvedJev()).transport, **kwargs)


def test_cached_run_preserves_resolved_identity_and_requested_settings(tmp_path):
    first = run()
    fake = ResolvedJev("jev-2")
    second = run(fake=fake, budget=0)
    assert not fake.bodies
    assert second.settings["requested_api"] == second.settings["provider"] == "openrouter"
    assert second.settings["requested_model"] == second.settings["request_model"] == "jev-alias"
    assert second.settings["resolved_models"] == ["jev-1"]
    assert second.settings["model"] == second.meter.model == "jev-1"
    assert second.settings["unknown_model_answers"] == 0
    assert second.scores["score_origin"].eq("cache").all()
    pd.testing.assert_series_equal(first.scores["answered_at"], second.scores["answered_at"])
    back = jlink.load(second.save(tmp_path / "out"))
    assert back.settings == second.settings
    assert back.meter.cached == 4 and back.meter.model == "jev-1"
    assert back.meter.answer_provenance == second.meter.answer_provenance
    assert back.scores["provider"].eq("openrouter").all()


def test_mixed_model_versions_are_not_collapsed_to_last_answer(tmp_path):
    run()
    left = pd.DataFrame({"name": ["Alpha", "Beta", "Gamma"]}, index=["NA", "001", "003"])
    result = run(left=left, fake=ResolvedJev("jev-2"))
    assert result.settings["resolved_models"] == ["jev-1", "jev-2"]
    assert result.settings["model"] == result.meter.model == ""
    assert result.meter.calls == 2 and result.meter.cached == 4
    assert set(result.scores["model"]) == {"jev-1", "jev-2"}
    assert "jev-1, jev-2" in result.methods()
    back = jlink.load(result.save(tmp_path / "mixed"))
    assert back.settings["resolved_models"] == ["jev-1", "jev-2"]
    assert back.meter.calls == 2 and back.meter.cost == result.meter.cost


def test_legacy_cache_does_not_get_the_current_requested_identity():
    store = Cache()
    state = {"record_a": {"name": "Alpha"}, "record_b": {"name": "Alpha LLC"}}
    store.put(Cache.key("jev-alias", state, question("firm")), {"type": "noul", "noul": 0.7})
    store.db.close()
    with pytest.warns(UserWarning, match="budget ran out"):
        result = run(budget=0)
    assert result.settings["resolved_models"] == []
    assert result.settings["unknown_model_answers"] == 1
    row = result.scores.loc[result.scores["source"] == "jev"].iloc[0]
    assert pd.isna(row["model"]) and pd.isna(row["provider"]) and pd.isna(row["answered_at"])
    assert "unknown model identity" in result.report()


def test_env_model_and_provider_selection_are_distinct_from_arguments(monkeypatch):
    monkeypatch.setenv("JEV_API", "openrouter")
    monkeypatch.setenv("JEV_MODEL", "env-alias")
    frame = pd.DataFrame({"name": ["Alpha"]})
    result = jlink.Linker("firm", "name", exact_shortcut=False).link(
        frame, frame, progress=False, transport=ResolvedJev().transport)
    assert result.settings["requested_api"] is result.settings["requested_model"] is None
    assert result.settings["provider"] == "openrouter" and result.settings["request_model"] == "env-alias"
    assert result.settings["resolved_models"] == ["jev-1"]


def test_builtin_configuration_includes_all_dataclass_parameters():
    passes = [jlink.block.ngrams(("name", "firm"), k=17, n=(3, 5), min_sim=0.25, name="neighbors"),
              jlink.block.exact("state", ("year", "yr")), jlink.block.initials("name", min_len=4)]
    configs = [blocker_config(b) for b in passes]
    if all(callable(getattr(b, "to_config", None)) for b in passes):
        # Newer blocking modules own their public schema; preserve it without reinterpretation.
        assert configs == [b.to_config() for b in passes]
        assert '"k": 17' in json.dumps(configs[0]) and '"min_sim": 0.25' in json.dumps(configs[0])
        return
    assert configs[0]["parameters"] == {"fields": [["name", "name", "firm"]], "name": "neighbors",
                                        "k": 17, "n": [3, 5], "min_sim": 0.25}
    assert configs[1]["parameters"]["fields"] == [["state", "state", "state"], ["year", "year", "yr"]]
    assert configs[2]["parameters"]["min_len"] == 4
    json.dumps(configs, allow_nan=False)


def test_serialization_protocol_preserves_nested_group_and_reverse_options():
    class FutureBlocker(jlink.block.Blocker):
        name = "grouped"

        def to_config(self):
            return {"type": "within", "columns": [["state", "st"]], "missing": "match",
                    "blocker": {"type": "ngrams", "k": 7, "n": [2, 5], "min_sim": 0.2, "reverse": True}}

    blocker = FutureBlocker()
    assert blocker_config(blocker) == blocker.to_config()


def test_nested_dataclass_config_and_opaque_custom_blocker_are_explicit():
    @dataclass
    class LegacyNgrams(jlink.block.Blocker):
        to_config = None  # Exercise fallback even when the base class gains a public method.
        k: int = 7
        name: str = "neighbors"

    @dataclass
    class Group(jlink.block.Blocker):
        to_config = None
        child: jlink.block.Blocker
        groups: tuple = ("state",)
        reverse: bool = True
        name: str = "group"

    config = blocker_config(Group(LegacyNgrams()))
    assert config["parameters"]["child"]["parameters"]["k"] == 7
    assert config["parameters"]["groups"] == ["state"] and config["parameters"]["reverse"] is True

    class Custom(jlink.block.Blocker):
        to_config = None
        name = "custom"

    assert blocker_config(Custom())["reconstructable"] is False
    assert blocker_config(Custom())["serialization"] == "opaque"


def test_default_and_empty_blockers_are_distinct():
    frame = pd.DataFrame({"name": ["Alpha"]})
    default = jlink.Linker("firm", "name").link(frame, frame, progress=False,
                                                   transport=ResolvedJev().transport)
    empty = jlink.Linker("firm", "name", blockers=[]).link(frame, frame, progress=False)
    assert '"k": 10' in json.dumps(default.settings["blocker_configs"])
    assert empty.settings["blockers"] == empty.settings["blocker_configs"] == []
    assert empty.scores.empty and empty.settings["unknown_model_answers"] == 0


def test_fingerprint_is_typed_fieldwise_ordered_and_deterministic():
    frame = pd.DataFrame({"first": ["Mary Ann", "Jose"], "last": ["Smith", None]}, index=["001", "NA"])

    def fingerprint(value):
        return frame_fingerprint(value, id_column=None, columns=["first", "last"], side="left")["sha256"]

    digest = fingerprint(frame)
    assert fingerprint(frame.copy()) == digest
    moved = frame.copy()
    moved.loc["001", ["first", "last"]] = ["Mary", "Ann Smith"]
    assert fingerprint(moved) != digest
    assert fingerprint(frame.iloc[::-1]) != digest
    assert fingerprint(frame.set_axis([1, "NA"])) != digest
    assert fingerprint(frame.assign(last=["Smith", np.nan])) == digest


def test_compared_fingerprints_exclude_unrelated_columns_but_full_hash_covers_blocking_inputs():
    a = run(left=pd.DataFrame({"name": ["Alpha"], "state": ["NY"]}))
    b = run(left=pd.DataFrame({"name": ["Alpha"], "state": ["CA"]}))
    assert a.settings["inputs"]["left"]["compared"] == b.settings["inputs"]["left"]["compared"]
    assert a.settings["inputs"]["left"]["full"]["sha256"] != b.settings["inputs"]["left"]["full"]["sha256"]
    assert b.settings["exact_shortcut"] is False
    assert b.settings["exact_policy"] == "all_fields_nonempty_and_equal_v1"
    assert b.settings["concurrency"] == 32 and b.settings["max_pairs"] == 5_000_000
    assert b.settings["runtime"]["pandas"] == pd.__version__


def test_old_saved_results_load_without_inventing_provenance(tmp_path):
    directory = run().save(tmp_path / "legacy")
    path = directory / "settings.json"
    settings = json.loads(path.read_text())
    legacy_keys = {"jlink", "date", "entity", "definition", "question", "on", "left_id", "right_id",
                   "blockers", "how", "threshold", "min_margin", "budget", "n_left", "n_right", "model",
                   "calls", "cached", "input_tokens", "dollars", "seconds", "id_kinds"}
    path.write_text(json.dumps({k: v for k, v in settings.items() if k in legacy_keys}))
    for filename in ("scores.csv", "links.csv"):
        frame = pd.read_csv(directory / filename, dtype={"left_id": str, "right_id": str}, keep_default_na=False)
        frame.drop(columns=["model", "provider", "score_origin", "answered_at"]).to_csv(directory / filename,
                                                                                        index=False)
    back = jlink.load(directory)
    assert back.settings["model_identity_status"] == "legacy_unverified"
    assert "resolved_models" not in back.settings and "inputs" not in back.settings
    assert back.meter.model == "" and "legacy model identity unverified" in back.methods()
    assert set(back.relink().links["left_id"]) == {"NA", "001"}


def test_optional_blocking_diagnostics_survive_save_load(monkeypatch, tmp_path):
    original = jlink.Linker.candidates
    diagnostic = {"schema_version": 1, "pair_count": 4, "passes": [{"added_pairs": 4}]}

    def with_diagnostics(*args, **kwargs):
        result = original(*args, **kwargs)
        result.attrs["blocking"] = diagnostic
        return result

    monkeypatch.setattr(jlink.Linker, "candidates", with_diagnostics)
    result = run()
    assert result.settings["blocking"] == diagnostic
    back = jlink.load(result.save(tmp_path / "diagnostics"))
    assert back.settings["blocking"] == back.candidates.attrs["blocking"] == diagnostic


def test_legacy_methods_do_not_claim_the_corrected_exact_policy():
    result = run()
    result.scores.loc[0, "source"] = "exact"
    result.settings.pop("exact_policy")
    assert "its fieldwise and missing-value policy was not recorded" in result.methods()


def test_custom_dataclass_with_opaque_state_is_marked_incomplete_without_breaking():
    @dataclass
    class Custom(jlink.block.Blocker):
        to_config = None
        name: str = "custom"
        callback: object = lambda: None

    config = blocker_config(Custom())
    assert config["parameters"] == {"name": "custom"}
    assert config["configuration_complete"] is False
    assert "callback" in config["unserialized_fields"]


@pytest.mark.parametrize("task", ["link", "dedupe"])
def test_invalid_blocker_provenance_is_rejected_before_judging(task):
    class InvalidConfig(jlink.block.Blocker):
        name = "invalid_config"

        def pairs(self, left, right):
            return np.array([[0, 1]], dtype=np.int64)

        def to_config(self):
            return {"unsupported": np.array([1, 2])}

    frame = pd.DataFrame({"name": ["Alpha", "Beta"]})
    fake = FakeJev(lambda state, question: 0.9)
    linker = jlink.Linker("firm", "name", blockers=[InvalidConfig()], cache=False)
    with pytest.raises(ValueError, match="cannot be saved as JSON: ndarray"):
        getattr(linker, task)(frame, *([frame] if task == "link" else []),
                              progress=False, transport=fake.transport)
    assert not fake.bodies


@pytest.mark.parametrize("task", ["link", "dedupe"])
def test_blocker_provenance_is_captured_before_judging(task):
    class MutableConfig(jlink.block.Blocker):
        name = "mutable_config"
        generation = "before"

        def pairs(self, left, right):
            return np.array([[0, 1]], dtype=np.int64)

        def to_config(self):
            return {"generation": self.generation}

    blocker = MutableConfig()

    def answer(state, question):
        blocker.generation = "after"
        return 0.9

    frame = pd.DataFrame({"name": ["Alpha", "Beta"]})
    fake = FakeJev(answer)
    linker = jlink.Linker("firm", "name", blockers=[blocker], cache=False)
    result = getattr(linker, task)(frame, *([frame] if task == "link" else []),
                                  progress=False, transport=fake.transport)
    assert len(fake.bodies) == 1
    assert result.settings["blocker_configs"] == [{"generation": "before"}]

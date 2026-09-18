"""CLI contract tests: all downstream computation is replaced by offline stand-ins."""

import importlib
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from jlink.io import read_table

command = importlib.import_module("jlink.cli")


@pytest.fixture
def downstream(monkeypatch):
    calls = SimpleNamespace()
    pairs = pd.DataFrame({"left_id": ["00123", "00456"], "right_id": ["00007", "00008"],
                          "block": ["ngrams:name", "ngrams:name"], "sim": [0.9, 0.8]})
    scores = pairs.assign(p=[0.98, 0.4], source="jev", error=None)
    links = scores.iloc[:1].assign(margin=float("nan"))
    calls.result = SimpleNamespace(links=links, scores=scores, candidates=pairs,
                                   meter=SimpleNamespace(calls=2, cost=0.001),
                                   report=lambda: "# Firm linkage\n\nOne link.\n")

    class Linker:
        def __init__(self, *, entity, definition, on, blockers, api=None, model=None, cache=True,
                     concurrency=32):
            calls.constructor = dict(entity=entity, definition=definition, on=on, blockers=blockers,
                                     api=api, model=model, cache=cache, concurrency=concurrency)

        def link(self, left, right, *, left_id=None, right_id=None, how="one-to-one", threshold=0.5,
                 min_margin=None, budget=5.0, progress=True):
            calls.link = dict(left=left, right=right, left_id=left_id, right_id=right_id, how=how,
                              threshold=threshold, min_margin=min_margin, budget=budget)
            return calls.result

    linker = ModuleType("jlink.linker")
    linker.Linker = Linker
    monkeypatch.setitem(sys.modules, "jlink.linker", linker)
    block = importlib.import_module("jlink.block")
    for kind in ("ngrams", "exact", "initials"):
        factory = lambda *cols, _kind=kind, **kwargs: (_kind, cols, kwargs)
        monkeypatch.setattr(block, kind, factory, raising=False)

    def candidates(left, right, **kwargs):
        calls.candidates = kwargs
        return pairs

    monkeypatch.setattr(block, "candidates", candidates, raising=False)
    audit = importlib.import_module("jlink.audit")

    def sample(frame, **kwargs):
        calls.sample = frame, kwargs
        return frame[["left_id", "right_id", "p"]].assign(bin="(0.95, 1]", weight=1.0, is_match="")

    def evaluate(frame, *, threshold, mode):
        calls.evaluate = frame, threshold
        calls.evaluate_mode = mode
        return SimpleNamespace(
            summary=lambda: "precision 0.9; recall among candidate pairs 0.8",
            to_markdown=lambda: "| precision | recall among candidate pairs |\n| 0.9 | 0.8 |",
        )

    monkeypatch.setattr(audit, "audit_sample", sample, raising=False)
    monkeypatch.setattr(audit, "evaluate", evaluate, raising=False)
    return calls


@pytest.fixture
def inputs(tmp_path):
    left = tmp_path / "survey firms.csv"
    right = tmp_path / "company registry.csv"
    pd.DataFrame({"id": ["00123", "00456"], "name": ["Acme Inc", "Other Firm"],
                  "city": ["Boston", "NY"], "state": ["MA", "NY"]}).to_csv(left, index=False)
    pd.DataFrame({"rid": ["00007", "00008"], "name": ["Acme", "Someone Else"],
                  "town": ["Boston", "NY"], "city": ["Boston", "NY"],
                  "st": ["MA", "NY"], "state": ["MA", "NY"]}).to_csv(right, index=False)
    return str(left), str(right)


def link_args(inputs):
    return ["link", *inputs, "--on", "name", "--entity", "firm", "--left-id", "id", "--right-id", "rid"]


def assert_error(args, capsys, expected):
    with pytest.raises(SystemExit) as exc:
        command.cli(args)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("jlink: ") and captured.err.count("\n") == 1
    assert "Traceback" not in captured.err
    assert expected in captured.err


@pytest.mark.parametrize("subcommand", [None, "link", "estimate", "audit", "evaluate"])
def test_help(subcommand, capsys):
    with pytest.raises(SystemExit) as exc:
        command.cli(([subcommand] if subcommand else []) + ["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "Example:" in help_text and ".dta" in help_text and "firm" in help_text
    if subcommand is None:
        assert "jev-link" in help_text and "python -m jlink" in help_text and "Java" in help_text


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        command.cli(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "jlink 0.1.0"


def test_stdout_and_summary(downstream, inputs, capsys):
    command.cli(link_args(inputs))
    captured = capsys.readouterr()
    assert captured.out.startswith("left_id,right_id,") and "00123,00007" in captured.out
    assert "2 left records, 2 right records; 2 candidate pairs; 1 links; 2 calls; $0.0010" in captured.err
    assert downstream.constructor == dict(entity="firm", definition="", on=["name"], blockers=None,
                                          api=None, model=None, cache=True, concurrency=32)
    assert downstream.link["left"].id.tolist() == ["00123", "00456"]


def test_link_all_options_and_files(downstream, inputs, tmp_path, capsys):
    output, scores, report = [tmp_path / file for file in ("links.dta", "scores.parquet", "report.md")]
    rules = ["ngrams:name:10", "ngrams:name+city:20", "exact:state", "exact:state=st", "initials:name"]
    args = link_args(inputs) + ["--on", "city=town", "--define", 'Same firm, including "Inc."',
                                "--how", "many-to-one", "--threshold", "0.8", "--min-margin", "-0.2",
                                "--budget", "0", "--api", "openrouter", "--model", "fake/jev",
                                "--no-cache", "-j", "4", "-o", str(output), "--scores", str(scores),
                                "--report", str(report)]
    for rule in rules:
        args.extend(["--block", rule])
    command.cli(args)
    assert capsys.readouterr().out == ""
    assert read_table(output).left_id.tolist() == ["00123"]
    assert len(read_table(scores)) == 2
    assert report.read_text() == downstream.result.report()
    assert downstream.constructor["on"] == ["name", ("city", "town")]
    assert downstream.constructor["blockers"] == [
        ("ngrams", ("name",), {"k": 10}), ("ngrams", ("name", "city"), {"k": 20}),
        ("exact", ("state",), {}), ("exact", (("state", "st"),), {}), ("initials", ("name",), {}),
    ]
    for key, value in dict(how="many-to-one", threshold=0.8, min_margin=-0.2, budget=0).items():
        assert downstream.link[key] == value
    for key, value in dict(api="openrouter", model="fake/jev", cache=False, concurrency=4).items():
        assert downstream.constructor[key] == value


@pytest.mark.parametrize("rule", ["", "random:name", "ngrams:name", "ngrams:name:0", "ngrams:name:-1",
                                  "ngrams:name:1.5", "ngrams:name:nan", "ngrams::10", "exact:name:10",
                                  "exact:=st", "exact:a=b=c", "exact:name+", "initials:name+city",
                                  "initials:name:2"])
def test_reject_bad_block(inputs, rule, capsys):
    assert_error(link_args(inputs) + ["--block", rule], capsys, "accepted forms are")


@pytest.mark.parametrize("args, expected", [([], "required"), (["wat"], "invalid choice"),
    (["link"], "required"), (["link", "a.csv", "b.csv", "--on", "name", "--entity", "firm", "-j", "0"],
                              "positive whole number")])
def test_parser_errors(args, expected, capsys):
    assert_error(args, capsys, expected)


@pytest.mark.parametrize("option,value", [("--threshold", "nan"), ("--threshold", "1.1"),
    ("--budget", "-1"), ("--budget", "inf"), ("--min-margin", "2"), ("-j", "1.2"), ("--how", "random")])
def test_invalid_values(inputs, option, value, capsys):
    assert_error(link_args(inputs) + [option, value], capsys, option)


def test_missing_columns_and_ids(inputs, capsys):
    assert_error(link_args(inputs) + ["--on", "missing"], capsys, "column 'missing'")
    assert_error(link_args(inputs) + ["--right-id", "wrong"], capsys, "column 'wrong'")
    assert_error(link_args(inputs) + ["--block", "exact:unknown"], capsys, "column 'unknown'")
    assert_error(link_args(inputs) + ["--on", "city="], capsys, "left_column=right_column")
    left = read_table(inputs[0])
    left["id"] = "00123"
    left.to_csv(inputs[0], index=False)
    assert_error(link_args(inputs), capsys, "duplicate IDs")


def test_output_errors_before_link(downstream, inputs, tmp_path, capsys):
    assert_error(link_args(inputs) + ["-o", "bad.xlsx"], capsys, "bad.xlsx")
    assert_error(link_args(inputs) + ["-o", inputs[0]], capsys, "separate output file")
    assert_error(link_args(inputs) + ["-o", str(tmp_path / "missing" / "out.csv")], capsys, "out.csv")
    assert not hasattr(downstream, "link")


def test_one_line_runtime_error(downstream, inputs, capsys, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("problem with column 'name'\nmore details")

    monkeypatch.setattr(sys.modules["jlink.linker"].Linker, "link", broken)
    assert_error(link_args(inputs), capsys, "column 'name' more details")


def test_estimate_never_constructs_linker(downstream, inputs, capsys):
    command.cli(["estimate", *inputs, "--on", "name", "--left-id", "id", "--right-id", "rid"])
    captured = capsys.readouterr()
    assert "2 candidate pairs" in captured.out and "$0.000028" in captured.out
    assert "200 pairs/second" in captured.out and "No API calls" in captured.out
    assert not hasattr(downstream, "constructor")


def test_audit_and_evaluate(downstream, tmp_path, inputs, capsys):
    scores, audit = tmp_path / "scores.csv", tmp_path / "audit.csv"
    downstream.result.scores.to_csv(scores, index=False)
    command.cli(["audit", str(scores), "-n", "5", "--left", inputs[0], "--right", inputs[1],
                 "--on", "name", "--on", "city=town", "--left-id", "id", "--right-id", "rid",
                 "-o", str(audit)])
    frame, kwargs = downstream.sample
    assert frame.p.dtype.kind == "f" and frame.left_id.iloc[0] == "00123"
    assert kwargs["n"] == 5 and kwargs["on"] == ["name", ("city", "town")]
    labeled = read_table(audit)
    labeled["is_match"] = ["yes", "no"]
    labeled.to_csv(audit, index=False)
    command.cli(["evaluate", str(audit), "--threshold", "0.8", "--markdown"])
    assert "| precision |" in capsys.readouterr().out
    frame, threshold = downstream.evaluate
    assert threshold == 0.8 and frame.weight.dtype.kind == "f"
    assert downstream.evaluate_mode == "threshold"
    assert frame.left_id.iloc[0] == "00123" and frame.is_match.tolist() == ["yes", "no"]
    command.cli(["evaluate", str(audit)])
    assert "recall among candidate pairs" in capsys.readouterr().out


def test_audit_without_sources_and_numeric_errors(downstream, tmp_path, capsys):
    scores, audit = tmp_path / "scores.csv", tmp_path / "audit.csv"
    downstream.result.scores.to_csv(scores, index=False)
    command.cli(["audit", str(scores), "-o", str(audit)])
    assert downstream.sample[1] == {"n": 200}
    assert_error(["audit", str(scores), "--left", "left.csv", "-o", str(audit)], capsys, "together")
    bad = pd.DataFrame({"left_id": ["01"], "right_id": ["02"], "p": ["not a probability"]})
    bad.to_csv(scores, index=False)
    assert_error(["audit", str(scores), "-o", str(audit)], capsys, "column 'p'")


def test_audit_aligns_numeric_stata_ids(downstream, tmp_path):
    paths = ("left.dta", "right.dta", "scores.csv", "a.csv")
    left, right, scores, output = [tmp_path / name for name in paths]
    pd.DataFrame({"id": [123], "name": ["Acme"]}).to_stata(left, write_index=False)
    pd.DataFrame({"id": [7], "name": ["Acme"]}).to_stata(right, write_index=False)
    pd.DataFrame({"left_id": [123], "right_id": [7], "p": [0.98]}).to_csv(scores, index=False)
    command.cli(["audit", str(scores), "--left", str(left), "--right", str(right), "--on", "name",
                 "--left-id", "id", "--right-id", "id", "-o", str(output)])
    assert downstream.sample[0].left_id.iloc[0] == 123


def test_selected_cli_roundtrip_and_old_audit_files(tmp_path, capsys):
    scores, links, output = [tmp_path / file for file in ("scores.csv", "links.csv", "audit.csv")]
    frame = pd.DataFrame({"left_id": ["001", "001"], "right_id": ["007", "008"], "p": [.9, .8]})
    frame.to_csv(scores, index=False)
    frame.iloc[:1].to_csv(links, index=False)
    command.cli(["audit", str(scores), "--links", str(links), "-o", str(output)])
    sample = read_table(output)
    assert set(sample.selected) == {"True", "False"}
    assert set(sample.left_id) == {"001"}
    sample["is_match"] = sample.right_id.eq("007")
    sample.to_csv(output, index=False)
    command.cli(["evaluate", str(output), "--mode", "selected", "--markdown"])
    summary = capsys.readouterr().out
    assert "Final-link evaluation using saved selected membership" in summary
    assert "| Precision | 1.0000 |" in summary
    assert "| Judged-candidate recall | 1.0000 |" in summary
    command.cli(["evaluate", str(output)])
    assert "Precision 0.5000" in capsys.readouterr().out
    command.cli(["evaluate", str(output), "--threshold", "0.85"])
    assert "Precision 1.0000" in capsys.readouterr().out
    sample.drop(columns="selected").to_csv(output, index=False)
    command.cli(["evaluate", str(output)])
    assert "Pair-scoring evaluation at threshold 0.5" in capsys.readouterr().out
    assert_error(["evaluate", str(output), "--mode", "selected"], capsys, "'selected' column")
    command.cli(["audit", str(scores), "-o", str(output)])
    assert "selected" not in read_table(output)


def test_selected_cli_aligns_ids_with_sources_and_protects_links(tmp_path, capsys):
    left, right, scores, links, output = [tmp_path / name for name in
                                          ("left.dta", "right.dta", "scores.csv", "links.csv", "audit.csv")]
    pd.DataFrame({"id": [123], "name": ["Acme"]}).to_stata(left, write_index=False)
    pd.DataFrame({"id": [7, 8], "name": ["ACME", "Other"]}).to_stata(right, write_index=False)
    frame = pd.DataFrame({"left_id": [123, 123], "right_id": [7, 8], "p": [.9, .8]})
    frame.to_csv(scores, index=False)
    frame.iloc[:1].to_csv(links, index=False)
    args = ["audit", str(scores), "--links", str(links), "--left", str(left), "--right", str(right),
            "--on", "name", "--left-id", "id", "--right-id", "id"]
    command.cli([*args, "-o", str(output)])
    sample = read_table(output)
    assert sample.loc[sample.selected.eq("True"), "right_id"].tolist() == ["7"]
    assert_error([*args, "-o", str(links)], capsys, "separate output file")
    frame.iloc[:1].assign(right_id=999).to_csv(links, index=False)
    assert_error(["audit", str(scores), "--links", str(links), "-o", str(output)], capsys,
                 "absent from scores")


def test_selected_cli_rejects_invalid_flags(tmp_path, capsys):
    path = tmp_path / "labeled.csv"
    pd.DataFrame({"left_id": ["A"], "right_id": ["B"], "p": [.9], "bin": ["all"],
                  "weight": [1], "is_match": [1], "selected": ["yes"]}).to_csv(path, index=False)
    assert_error(["evaluate", str(path), "--mode", "selected"], capsys, "nonmissing 1/0 or True/False")


@pytest.mark.parametrize("program", [[sys.executable, "-m", "jlink"], ["jev-link"], ["jlink"]])
def test_installed_entrypoints(program):
    result = subprocess.run([*program, "--version"], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and result.stdout.strip() == "jlink 0.1.0"
    result = subprocess.run([*program, "estimate", "absent.csv", "right.csv", "--on", "name"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2 and result.stderr.startswith("jlink:")
    assert result.stderr.count("\n") == 1 and "absent.csv" in result.stderr

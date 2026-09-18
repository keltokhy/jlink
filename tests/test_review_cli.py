"""Real offline CLI round trips, including saved NA IDs and authoritative JSON output."""
import importlib
import json

import pandas as pd
import pytest

import jlink

cli = importlib.import_module("jlink.cli").cli


@pytest.fixture
def source(tmp_path):
    left = pd.DataFrame({"id": ["001", "NA", "003"], "name": ["Acme", "North", "No candidate"]})
    right = pd.DataFrame({"id": ["NULL", "B"], "name": ["Acme Labs", "Northstar"]})
    scores = pd.DataFrame({"left_id": ["001", "001", "NA"], "right_id": ["NULL", "B", "B"],
                           "p": [.8, .79, float("nan")], "sim": [.9, .8, .5],
                           "source": ["jev", "jev", "unjudged"], "block": "fake", "error": None})
    settings = dict(left_id="id", right_id="id", on=[["name", "name"]],
                    how="one-to-one", threshold=.5, min_margin=None)
    result = jlink.Result(jlink.resolve(scores), scores, settings, _left=left, _right=right)
    result.save(tmp_path / "run")
    left.to_csv(tmp_path / "left.csv", index=False)
    right.to_csv(tmp_path / "right.csv", index=False)
    return tmp_path


def create_args(source):
    return ["review", "create", str(source / "run"), "--left", str(source / "left.csv"),
            "--right", str(source / "right.csv"), "-o", str(source / "review.html"),
            "--artifact", str(source / "review.json")]


def test_create_page_apply_round_trip(source, capsys):
    cli(create_args(source))
    assert "no API calls" in capsys.readouterr().err
    assert (source / "review.html").read_text().startswith("<!doctype html>")
    review = jlink.read_review(source / "review.json")
    review.decide("NA", "B", "accept", reviewer="Reviewer 1").save(source / "review.json")
    cli(["review", "page", str(source / "review.json"), "-o", str(source / "reopened.html")])
    assert "Reviewer 1" in (source / "reopened.html").read_text()
    cli(["review", "apply", str(source / "review.json"), "-o", str(source / "applied.json"),
         "--csv", str(source / "links.csv"), "--threshold", ".99", "--no-margin"])
    document = json.loads((source / "applied.json").read_text())
    assert len(document["links"]["rows"]) == 1
    row = document["links"]["rows"][0]
    assert row["left_id"] == {"type": "str", "value": "NA"} and row["p"] is None
    assert document["settings"]["threshold"] == .99
    assert document["review"]["history"][0]["reviewer"] == "Reviewer 1"
    assert "manual_accept" in (source / "links.csv").read_text()


def test_conflicts_fail_without_partial_output(source, capsys):
    cli(create_args(source))
    review = jlink.read_review(source / "review.json")
    review.decide("001", "NULL", "accept", reviewer="A").decide("001", "B", "accept", reviewer="A")
    review.save(source / "review.json")
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        cli(["review", "apply", str(source / "review.json"), "-o", str(source / "applied.json")])
    assert error.value.code == 2
    assert "conflicting manual acceptances" in capsys.readouterr().err
    assert not (source / "applied.json").exists()
    cli(["review", "apply", str(source / "review.json"), "-o", str(source / "applied.json"),
         "--how", "many-to-many"])
    assert len(json.loads((source / "applied.json").read_text())["links"]["rows"]) == 2


def test_output_cannot_overwrite_inputs(source, capsys):
    cli(create_args(source))
    before = (source / "review.json").read_bytes()
    with pytest.raises(SystemExit):
        cli(["review", "apply", str(source / "review.json"), "-o", str(source / "review.json")])
    assert (source / "review.json").read_bytes() == before
    assert "repeats an input" in capsys.readouterr().err


def test_saved_integer_ids_restore_before_alignment(source):
    settings_path = source / "run" / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["id_kinds"]["left_id"] = "int"
    settings_path.write_text(json.dumps(settings))
    left = pd.read_csv(source / "left.csv", keep_default_na=False, dtype=str)
    left["id"] = [str(2**53 + 1), str(2**53 + 3), str(2**53 + 5)]
    left.to_csv(source / "left.csv", index=False)
    for file in ("scores.csv", "links.csv"):
        frame = pd.read_csv(source / "run" / file, keep_default_na=False, dtype=str)
        frame["left_id"] = frame.left_id.map({"001": str(2**53 + 1), "NA": str(2**53 + 3)})
        frame.to_csv(source / "run" / file, index=False)
    cli(create_args(source))
    review = jlink.read_review(source / "review.json")
    assert review.to_dict()["source"]["records"]["left"][0]["id"] == {"type": "int", "value": str(2**53 + 1)}
    assert review.apply().links.iloc[0].left_id == 2**53 + 1


@pytest.mark.parametrize("args", [["--help"], ["review", "--help"], ["review", "create", "--help"],
                                 ["review", "apply", "--help"], ["review", "page", "--help"]])
def test_review_is_discoverable(args, capsys):
    with pytest.raises(SystemExit) as exit:
        cli(args)
    assert exit.value.code == 0
    assert "review" in capsys.readouterr().out

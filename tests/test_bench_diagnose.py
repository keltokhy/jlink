import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.diagnose import diagnose_pairs, run


def test_distinguishes_missing_rejected_and_unjudged_gold():
    left = pd.DataFrame({"id": ["a"], "name": ["item"]})
    right = pd.DataFrame({"id": ["w", "x", "y", "z"], "name": ["w", "x", "y", "z"]})
    gold = pd.DataFrame({"left_id": ["a"] * 3, "right_id": ["w", "x", "y"]})
    scores = pd.DataFrame({"left_id": ["a"] * 3, "right_id": ["x", "y", "z"],
                           "p": [.2, float("nan"), .9]})
    result = diagnose_pairs(scores, gold, left, right, on=["name"])
    assert result["retrieval_misses"] == 1
    assert result["unjudged_gold_candidates"] == 1
    assert result["judged_false_negatives"] == 1
    assert result["final_missed_links"] == 3
    assert result["false_positives"] == 1


def test_conflicts_use_judge_payload_including_mapped_fields_and_missing_values():
    left = pd.DataFrame({"id": ["a"], "title": ["item"], "detail": [""]})
    right = pd.DataFrame({"id": ["NA", "NULL"], "name": ["item", "item"],
                          "detail": [None, ""], "hidden": ["different", "evidence"]})
    gold = pd.DataFrame({"left_id": ["a"], "right_id": ["NA"]})
    scores = pd.DataFrame({"left_id": ["a", "a"], "right_id": ["NA", "NULL"], "p": [.9, .9]})
    result = diagnose_pairs(scores, gold, left, right, on=[("title", "name"), "detail"])
    assert result["identical_judge_inputs_with_mixed_labels"] == 1
    assert result["pairs_in_mixed_groups"] == 2
    assert {r["right_id"] for r in result["mixed_groups"][0]} == {"NA", "NULL"}
    left["hidden"] = "different"
    result = diagnose_pairs(scores, gold, left, right, on=[("title", "name"), "detail", "hidden"])
    assert result["identical_judge_inputs_with_mixed_labels"] == 0


def test_positive_only_crosswalk_cannot_be_used_for_error_labels(tmp_path):
    with pytest.raises(ValueError, match="positive-only"):
        run("nber-firms", saved_run=tmp_path, out=tmp_path / "report.json")

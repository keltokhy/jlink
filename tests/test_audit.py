from io import StringIO

import numpy as np
import pandas as pd
import pytest

from jlink.audit import Evaluation, audit_sample, evaluate, score_against_truth


def scores(probabilities):
    return pd.DataFrame({"left_id": np.arange(len(probabilities)),
                         "right_id": np.arange(len(probabilities)) + 100, "p": probabilities})


def labeled(p, y, weights=None, bins=None):
    return pd.DataFrame({"p": p, "is_match": y, "weight": weights if weights is not None else 1,
                         "bin": bins if bins is not None else "all"})


def test_sampling_is_reproducible_shuffled_and_uses_only_judged_pairs():
    frame = scores([.01] * 50 + [.1] * 30 + [.3] * 20 + [.6] * 20 + [.9] * 10 + [1.] * 10 + [np.nan])
    original = frame.copy(deep=True)
    sample = audit_sample(frame, n=30, seed=13)
    pd.testing.assert_frame_equal(sample, audit_sample(frame, n=30, seed=13))
    assert not sample.equals(audit_sample(frame, n=30, seed=14))
    assert len(sample) == 30 and sample.left_id.is_unique
    assert not sample.bin.is_monotonic_increasing
    assert not sample.p.isna().any()
    assert sample.is_match.isna().all()
    assert sample.groupby("bin", observed=True).size().tolist() == [5] * 6
    assert sample.weight.sum() == len(frame) - 1
    assert sample.columns.tolist() == ["is_match", "p", "bin", "weight", "left_id", "right_id"]
    pd.testing.assert_frame_equal(frame, original)


def test_capped_allocation_redistributes_until_full():
    sample = audit_sample(scores([.01] + [.1] * 2 + [.3] * 15 + [.9] * 100), n=20)
    assert sample.groupby("bin", observed=True).size().tolist() == [1, 2, 9, 8]
    assert sample.weight.sum() == pytest.approx(118)
    assert len(audit_sample(scores([.1, .2, .8]), n=999)) == 3


def test_right_closed_bins_include_zero_and_one():
    sample = audit_sample(scores([0, .05, .2, .5, .8, .95, 1]), n=7).sort_values("left_id")
    assert sample.bin.astype(str).tolist() == ["[0, 0.05]", "[0, 0.05]", "(0.05, 0.2]", "(0.2, 0.5]",
                                             "(0.5, 0.8]", "(0.8, 0.95]", "(0.95, 1]"]
    assert (sample.weight == 1).all()


def test_fields_are_interleaved_using_explicit_ids_and_mapped_columns():
    left = pd.DataFrame({"id": [8, 3], "name": ["Alpha", "Beta"], "city": ["Cairo", "New York"]})
    right = pd.DataFrame({"id": [20, 10], "title": ["B", "A"], "town": ["NYC", "Cairo"]})
    frame = pd.DataFrame({"left_id": [3, 8], "right_id": [20, 10], "p": [.8, .9]})
    sample = audit_sample(frame, left=left, right=right, on=[("name", "title"), ("city", "town")],
                          left_id="id", right_id="id").sort_values("left_id")
    assert sample.columns.tolist() == ["is_match", "a_name", "b_name", "a_city", "b_city",
                                       "p", "bin", "weight", "left_id", "right_id"]
    assert sample.a_name.tolist() == ["Beta", "Alpha"]
    assert sample.b_name.tolist() == ["B", "A"]
    assert sample.b_city.tolist() == ["NYC", "Cairo"]


def test_fields_default_to_frame_indexes():
    left = pd.DataFrame({"name": ["a", "b"]}, index=[2, 1])
    right = pd.DataFrame({"name": ["A", "B"]}, index=[4, 3])
    sample = audit_sample(pd.DataFrame({"left_id": [1], "right_id": [4], "p": [.9]}),
                          left=left, right=right, on="name")
    assert sample.loc[0, "a_name"] == "b" and sample.loc[0, "b_name"] == "A"


@pytest.mark.parametrize("probabilities", [[], [np.nan], [.1, .9]])
def test_empty_and_zero_samples(probabilities):
    sample = audit_sample(scores(probabilities), n=0)
    assert sample.empty
    assert sample.columns.tolist() == ["is_match", "p", "bin", "weight", "left_id", "right_id"]


def test_sampling_less_than_number_of_bins_preserves_missing_strata():
    sample = audit_sample(scores([.1, .7, .9]), n=1)
    sample["is_match"] = 1
    result = evaluate(sample, n_boot=10)
    assert np.isnan(result.recall[0])
    assert (result.calibration.n == 0).sum() == 2
    assert "No labels in bin" in result.summary()


def test_weighted_point_estimates_and_calibration():
    frame = labeled([.9, .8, .7, .2, .1], [1, 0, 1, 1, 0], [10, 2, 3, 20, 5],
                    ["high", "high", "high", "low", "low"])
    result = evaluate(frame, n_boot=199, seed=5)
    assert isinstance(result, Evaluation)
    assert result.precision[0] == pytest.approx(13 / 15)
    assert result.recall[0] == pytest.approx(13 / 33)
    assert result.f1[0] == pytest.approx(26 / 48)
    assert result.brier == pytest.approx(np.average((frame.p - frame.is_match) ** 2, weights=frame.weight))
    table = result.calibration.set_index("bin")
    assert table.loc["high", "n"] == 3
    assert table.loc["high", "mean_p"] == pytest.approx((9 + 1.6 + 2.1) / 15)
    assert table.loc["low", "match_rate"] == .8
    repeated = evaluate(frame, n_boot=199, seed=5)
    assert repeated.precision == result.precision and repeated.recall == result.recall
    assert repeated.f1 == result.f1


def test_estimates_are_invariant_to_large_weight_scale():
    frame = labeled([.9, .7, .2, .1], [1, 0, 1, 0], [2, 1, 3, 2])
    expected = evaluate(frame, n_boot=99)
    frame["weight"] = frame.weight * 5e307
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = evaluate(frame, n_boot=99)
    assert result.precision == pytest.approx(expected.precision)
    assert result.recall == pytest.approx(expected.recall)
    assert result.f1 == pytest.approx(expected.f1)
    assert result.brier == pytest.approx(expected.brier)


def test_bootstrap_resamples_each_stratum_and_retains_row_weights():
    frame = labeled([.8, .7, .2, .1], [1, 0, 1, 0], [2, 4, 10, 20], ["high", "high", "low", "low"])
    result = evaluate(frame, n_boot=299, seed=48)
    rng = np.random.default_rng(48)
    draws = np.column_stack([rng.integers(0, 2, size=(299, 2)), rng.integers(2, 4, size=(299, 2))])
    values = []
    for draw in draws:
        boot = frame.iloc[draw]
        tp = (boot.weight * boot.is_match * (boot.p >= .5)).sum()
        linked = boot.weight[boot.p >= .5].sum()
        true = (boot.weight * boot.is_match).sum()
        values.append([tp / linked, tp / true if true else np.nan, 2 * tp / (linked + true)])
    expected = np.nanquantile(values, [.025, .975], axis=0)
    for index, metric in enumerate([result.precision, result.recall, result.f1]):
        assert metric[1:] == pytest.approx(expected[:, index])


def test_labels_and_csv_round_trip():
    values = [1, 0, True, False, "Y", "n", " Yes ", "NO", 1.0, 0.0, "", None, np.nan, pd.NA, "  "]
    frame = labeled([.9] * len(values), values)
    result = evaluate(frame, n_boot=20)
    assert result.n_labeled == 10 and result.n_unlabeled == 5
    assert result.precision[0] == .5
    sample = audit_sample(scores([.1, .2, .7, .8]), n=4)
    sample["is_match"] = sample.p > .5
    reread = pd.read_csv(StringIO(sample.to_csv(index=False)))
    assert evaluate(reread, n_boot=20).precision[0] == 1


def test_blank_rows_are_dropped_before_numeric_validation():
    frame = labeled([.9, "not scored"], [1, ""], [2, "not weighted"])
    result = evaluate(frame, n_boot=10)
    assert result.precision[0] == 1 and result.n_unlabeled == 1


@pytest.mark.parametrize("p, y, undefined, reason", [
    ([.1, .2], [1, 0], "precision", "No predicted links"),
    ([.8, .9], [0, 0], "recall", "No true matches"),
    ([.1, .2], [0, 0], "f1", "No predicted links or true matches"),
])
def test_undefined_metrics_have_explanations(p, y, undefined, reason):
    with np.errstate(all="raise"):
        result = evaluate(labeled(p, y), n_boot=30)
    assert all(np.isnan(value) for value in getattr(result, undefined))
    assert reason in result.summary()


def test_no_labels_and_completely_unlabeled_bin():
    for frame in (labeled([], []), labeled([.8, .1], ["", None]),
                  labeled([.8, .1], [1, ""], bins=["high", "low"])):
        result = evaluate(frame, n_boot=20)
        assert np.isnan(result.precision).all() and np.isnan(result.brier)
        assert "No label" in result.summary()
    result = evaluate(labeled([.8, .1], [1, ""], bins=["high", "low"]), n_boot=20)
    assert result.calibration.n.tolist() == [1, 0]
    assert np.isnan(result.calibration.match_rate.iloc[1])


def test_summary_and_markdown_are_appendix_ready():
    result = evaluate(labeled([.1, .2, .7, .9], [0, 1, 0, 1], bins=["low", "low", "hi|gh", "hi|gh"]),
                      n_boot=50)
    text = result.to_markdown()
    assert "| Metric | Estimate | 95% interval |" in text
    assert "| Candidate-pair recall |" in text
    assert "| Weighted Brier score |" in text
    assert "Weighted match rate" in text and "hi\\|gh" in text
    assert "threshold 0.5" in text and "4 labeled pairs" in text
    assert "Recall is among candidate pairs" in result.summary()
    assert "lost in blocking" in result.summary()


def test_estimators_and_intervals_on_repeated_stratified_samples():
    # A fixed population with severe imbalance: an unweighted sample badly overstates recall.
    sizes = np.array([30_000, 20_000, 12_000, 8_000, 4_000, 2_000])
    probabilities = [.02, .1, .35, .65, .9, .98]
    rates = [.02, .1, .35, .65, .9, .98]
    population = scores(np.repeat(probabilities, sizes))
    truth = np.concatenate([np.arange(size) < round(size * rate) for size, rate in zip(sizes, rates)])
    predicted = population.p.to_numpy() >= .5
    tp = (truth & predicted).sum()
    target = np.array([tp / predicted.sum(), tp / truth.sum(), 2 * tp / (predicted.sum() + truth.sum())])
    estimates, covered, briers = [], [], []
    for seed in range(80):
        sample = audit_sample(population, n=900, seed=seed)
        sample["is_match"] = truth[sample.left_id.to_numpy()]
        result = evaluate(sample, n_boot=299, seed=seed + 10_000)
        metrics = np.array([result.precision, result.recall, result.f1])
        estimates.append(metrics[:, 0])
        covered.append((metrics[:, 1] <= target) & (target <= metrics[:, 2]))
        briers.append(result.brier)
    assert np.mean(estimates, axis=0) == pytest.approx(target, abs=.02)
    coverage = np.mean(covered, axis=0)
    assert ((coverage >= .85) & (coverage <= 1)).all(), coverage
    true_brier = np.mean((population.p.to_numpy() - truth) ** 2)
    assert np.mean(briers) == pytest.approx(true_brier, abs=.01)


@pytest.mark.parametrize("kwargs, message", [
    ({"n": -1}, "n"), ({"n": 2.5}, "n"), ({"bins": [0, .8, .2, 1]}, "bins"),
    ({"bins": [.1, 1]}, "bins"), ({"bins": [0, .9]}, "bins"), ({"bins": [0, np.nan, 1]}, "bins"),
    ({"seed": -1}, "seed"),
    ({"left": pd.DataFrame()}, "together"),
])
def test_invalid_sampling_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        audit_sample(scores([.8]), **kwargs)


@pytest.mark.parametrize("column, value, message", [
    ("is_match", "maybe", "is_match"), ("is_match", 2, "is_match"),
    ("weight", 0, "weight"), ("weight", -2, "weight"), ("weight", np.nan, "weight"),
    ("p", np.nan, "p"), ("p", 1.1, "p"), ("bin", None, "bin"),
])
def test_invalid_evaluation_data(column, value, message):
    frame = labeled([.9], [1]).astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        evaluate(frame, n_boot=10)


@pytest.mark.parametrize("kwargs, message", [
    ({"threshold": 1.1}, "threshold"), ({"n_boot": 0}, "n_boot"), ({"seed": "invalid"}, "seed"),
])
def test_invalid_evaluation_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        evaluate(labeled([.9], [1]), **kwargs)


def test_missing_or_duplicate_source_ids_and_missing_fields():
    frame = scores([.8])
    for left, message in [(pd.DataFrame({"name": ["a", "b"]}, index=[0, 0]), "duplicate"),
                          (pd.DataFrame({"name": ["a"]}, index=[2]), "absent"),
                          (pd.DataFrame({"other": ["a"]}), "name")]:
        with pytest.raises(ValueError, match=message):
            audit_sample(frame, left=left, right=pd.DataFrame({"name": ["b"]}, index=[100]), on="name")


def test_score_against_truth_counts_and_blocking_completeness():
    links = pd.DataFrame({"left_id": [1, 2, 3], "right_id": [11, 12, 13]})
    truth = pd.DataFrame({"left_id": [1, 2, 4, 5], "right_id": [11, 12, 14, 15]})
    candidates = pd.concat([links, truth.iloc[2:3]], ignore_index=True)
    result = score_against_truth(links, truth, candidates)
    assert result == {"precision": 2 / 3, "recall": .5, "f1": 4 / 7, "tp": 2, "fp": 1, "fn": 2,
                      "pairs_completeness": .75}
    assert "pairs_completeness" not in score_against_truth(links, truth)


def test_score_against_truth_empty_sets_and_duplicate_validation():
    one = pd.DataFrame({"left_id": [1], "right_id": [2]})
    empty = one.iloc[:0]
    result = score_against_truth(empty, one)
    assert np.isnan(result["precision"]) and result["recall"] == 0 and result["f1"] == 0
    result = score_against_truth(one, empty, empty)
    assert result["precision"] == 0 and np.isnan(result["recall"]) and np.isnan(result["pairs_completeness"])
    result = score_against_truth(empty, empty)
    assert all(np.isnan(result[name]) for name in ["precision", "recall", "f1"])
    with pytest.raises(ValueError, match="duplicate"):
        score_against_truth(pd.concat([one, one]), one)

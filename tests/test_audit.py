from io import StringIO

import numpy as np
import pandas as pd
import pytest

from jlink.audit import Evaluation, audit_sample, evaluate, score_against_truth
from jlink.resolve import resolve


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


def test_blank_rows_need_their_weight_but_not_a_probability():
    result = evaluate(labeled([.9, "not scored"], [1, ""], [2, 3]), n_boot=10)
    assert result.precision[0] == 1 and result.n_unlabeled == 1
    # A blank row's weight says how much of the bin its labeled rows stand for.
    for weight in ("not weighted", None, np.nan, np.inf):
        with pytest.raises(ValueError, match="every row, including rows with a blank label"):
            evaluate(labeled([.9, "not scored"], [1, ""], [2, weight]), n_boot=10)
    for weight in (0, -1):
        with pytest.raises(ValueError, match="positive sampling weights"):
            evaluate(labeled([.9, .8], [1, ""], [2, weight]), n_boot=10)
    with pytest.raises(ValueError, match="column 'weight' must contain numbers"):
        evaluate(labeled([.9, .8], [1, ""], ["not weighted", 3]), n_boot=10)


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


def test_unlabeled_bins_are_never_redistributed_and_their_weight_is_reported():
    frame = labeled([.9, .8, .3, .2, .1, .15], [1, 0, "", "", "", 0], [5, 5, 30, 30, 60, 60],
                    ["high", "high", "middle", "middle", "low", "low"])
    result = evaluate(frame, n_boot=20)
    for metric in (result.precision, result.recall, result.f1):
        assert np.isnan(metric).all()
    assert np.isnan(result.brier)
    text = result.summary()
    assert "No labels in bin(s) middle: population-wide estimates and intervals are NaN." in text
    assert "hold 31.6% of the sampled weight (middle 31.6%)" in text  # 60 of 190
    assert "reweighted" not in text  # nothing was estimated, so no adjustment is claimed
    # Bins that do have labels keep their own calibration rows.
    table = result.calibration.set_index("bin")
    assert table.n.to_dict() == {"high": 2, "middle": 0, "low": 1}
    assert table.loc["high", "match_rate"] == .5 and np.isnan(table.loc["middle", "match_rate"])
    frame.loc[5, "is_match"] = ""
    assert "hold 94.7% of the sampled weight (middle 31.6%, low 63.2%)" in evaluate(frame, n_boot=20).summary()
    # A populated bin that was never sampled has no rows, so its weight cannot be stated.
    sample = audit_sample(scores([.1, .7, .9]), n=1)
    sample["is_match"] = 1
    text = evaluate(sample, n_boot=10).summary()
    assert "had no sampled rows, so their weight is unknown" in text and "of the sampled weight" not in text


def test_summary_and_markdown_are_appendix_ready():
    result = evaluate(labeled([.1, .2, .7, .9], [0, 1, 0, 1], bins=["low", "low", "hi|gh", "hi|gh"]),
                      n_boot=50)
    text = result.to_markdown()
    assert "| Metric | Estimate | 95% interval |" in text
    assert "| Judged-candidate recall |" in text
    assert "| Weighted Brier score |" in text
    assert "Weighted match rate" in text and "hi\\|gh" in text
    assert "threshold 0.5" in text and "4 labeled pairs" in text
    assert "Recall is among judged candidate pairs only" in result.summary()
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


def full_audit():
    """The population design above as a deterministic, fully labeled audit: 150 rows a bin,
    no random sampling, and weights that vary within a bin."""
    sizes = np.array([30_000, 20_000, 12_000, 8_000, 4_000, 2_000])
    edges = [0, .05, .2, .5, .8, .95, 1.0]
    rates = [.02, .1, .35, .65, .9, .98]
    parts = []
    for b, (size, rate, low, high) in enumerate(zip(sizes, rates, edges[:-1], edges[1:])):
        k = np.arange(150)
        parts.append(pd.DataFrame({
            "p": low + (high - low) * (k + 1) / 151, "is_match": ((k * 7) % 150 < round(150 * rate)).astype(int),
            "weight": size / 150 * (1 + (k % 5) / 10), "bin": f"bin {b}",
            "left_id": b * 1000 + k, "right_id": b * 1000 + k, "selected": (low >= .5) & (k % 4 != 0)}))
    return pd.concat(parts, ignore_index=True)


# float.hex() of precision, recall and F1 (estimate, low, high), then Brier, then each bin's
# mean_p and match_rate, from the evaluator as it was BEFORE blank labels were reweighted
# (jlink main at 3531fb9). A fully labeled audit must keep giving exactly these numbers.
BEFORE_REWEIGHTING = {
    "threshold": ["0x1.921eebb885516p-1", "0x1.75e5ba8888a2cp-1", "0x1.a7bc467a51dcep-1",
                  "0x1.04787f5348f1ep-1", "0x1.d5279075a8883p-2", "0x1.1d342e0b7a344p-1",
                  "0x1.3c27955d8ac48p-1", "0x1.22fc4ad6ed917p-1", "0x1.4fa11464e420fp-1",
                  "0x1.a2b637320262bp-4"],
    "selected": ["0x1.8e56529e8a2b1p-1", "0x1.70fbda18068bcp-1", "0x1.a63c1b63fe218p-1",
                 "0x1.db9ddf52c3c3bp-2", "0x1.abcd314320f3bp-2", "0x1.09c23210e5b8dp-1",
                 "0x1.29d17fbc3e612p-1", "0x1.13b463c1641aap-1", "0x1.40a0c34d85c27p-1",
                 "0x1.a2b637320262bp-4"],
    "calibration": ["0x1.9a8112e0039dfp-6", "0x1.3579be02468aep-6", "0x1.0056cd7a67c1ap-3",
                    "0x1.999999999999ap-4", "0x1.66bd33e0ce281p-2", "0x1.626af37c048d3p-2",
                    "0x1.4cf8338a00adcp-1", "0x1.4df012345678bp-1", "0x1.c015b35e99f08p-1",
                    "0x1.ccccccccccccep-1", "0x1.f33a6efd66838p-1", "0x1.f579be02468aep-1"],
}


@pytest.mark.parametrize("mode", ["threshold", "selected"])
def test_fully_labeled_audits_give_exactly_the_numbers_they_gave_before_reweighting(mode):
    result = evaluate(full_audit(), threshold=.6, n_boot=299, seed=10_000, mode=mode)
    numbers = [*result.precision, *result.recall, *result.f1, result.brier]
    assert [float(value).hex() for value in numbers] == BEFORE_REWEIGHTING[mode]
    table = result.calibration[["mean_p", "match_rate"]].to_numpy().ravel()
    assert [float(value).hex() for value in table] == BEFORE_REWEIGHTING["calibration"]
    assert "reweighted" not in result.summary()


def test_labeled_pairs_take_over_the_weight_of_their_bins_blank_rows():
    frame = labeled([.9, .8, .3, .2, .1, .25], [1, 0, 1, "", None, 0], [5, 5, 10, 10, 10, 10],
                    ["high", "high", "low", "low", "low", "low"])
    result = evaluate(frame, n_boot=50)
    # "low" has 40 of weight on 4 sampled rows and 20 on its 2 labels: each label counts twice.
    assert result.precision[0] == pytest.approx(5 / 10)
    assert result.recall[0] == pytest.approx(5 / (5 + 20))  # 5 / 15 if blanks were merely dropped
    assert result.f1[0] == pytest.approx(2 * 5 / (10 + 25))
    assert result.brier == pytest.approx((5 * .01 + 5 * .64 + 20 * .49 + 20 * .0625) / 50)
    table = result.calibration.set_index("bin")  # within a bin the common factor cancels
    assert table.loc["low", "mean_p"] == pytest.approx(.275) and table.loc["low", "match_rate"] == .5
    assert (result.n_labeled, result.n_unlabeled) == (4, 2)
    assert "2 blank labels: labeled pairs were reweighted to their bin's full sampling weight (low x2)" \
        in result.summary()
    # Unequal weights inside a bin: the factor is total weight over labeled weight, 60 / 15.
    uneven = labeled([.9, .3, .2, .1], [1, 1, "", ""], [5, 15, 25, 20], ["high", "low", "low", "low"])
    assert evaluate(uneven, n_boot=20).recall[0] == pytest.approx(5 / (5 + 60))
    assert "(low x4)" in evaluate(uneven, n_boot=20).summary()
    # The same blank rate in every bin is a common scale and moves nothing.
    even = labeled([.9, .8, .3, .2], [1, "", 1, ""], [5, 5, 10, 10], ["high", "high", "low", "low"])
    assert evaluate(even, n_boot=20).recall[0] == pytest.approx(5 / 15)


@pytest.mark.parametrize("mode", ["threshold", "selected"])
def test_every_estimate_and_interval_uses_the_reweighted_pairs(mode):
    frame = full_audit()
    blank = (frame.index % 150 >= 40) & frame.bin.isin(["bin 0", "bin 1", "bin 2"])  # 110 of 150 in the low bins
    partial = frame.assign(is_match=frame.is_match.astype(object).mask(blank, ""))
    result = evaluate(partial, threshold=.6, n_boot=299, seed=4, mode=mode)
    # The same audit with the blank rows deleted and the factors applied by hand.
    total = frame.groupby("bin").weight.transform("sum")
    kept = frame.weight.where(~blank).groupby(frame.bin).transform("sum")
    by_hand = frame.assign(weight=frame.weight * total / kept).loc[~blank]
    expected = evaluate(by_hand, threshold=.6, n_boot=299, seed=4, mode=mode)
    for name in ("precision", "recall", "f1"):
        assert getattr(result, name) == pytest.approx(getattr(expected, name), rel=1e-12)
    assert result.brier == pytest.approx(expected.brier, rel=1e-12)
    pd.testing.assert_frame_equal(result.calibration, expected.calibration)
    # Merely dropping the blanks, as before, understates the low bins and overstates recall.
    dropped = evaluate(frame.loc[~blank], threshold=.6, n_boot=50, seed=4, mode=mode)
    assert dropped.recall[0] > result.recall[0] + .15
    assert result.recall[0] == pytest.approx(evaluate(frame, threshold=.6, n_boot=50, mode=mode).recall[0], abs=.05)


def test_bootstrap_matches_rescaling_each_replicate_to_its_bin_totals():
    # audit_sample gives one weight per bin, so every replicate draws the same labeled weight in a
    # bin, and rescaling that replicate to the bin's total is the evaluator's fixed factor.
    frame = full_audit()
    frame["weight"] = frame.groupby("bin").weight.transform("max")
    blank = (frame.index % 150 >= 40) & frame.bin.isin(["bin 0", "bin 1", "bin 2"])
    partial = frame.assign(is_match=frame.is_match.astype(object).mask(blank, ""))
    result = evaluate(partial, threshold=.6, n_boot=299, seed=21)
    rng = np.random.default_rng(21)
    rescaled, unscaled = np.zeros((299, 3)), np.zeros((299, 3))
    for name, rows in frame.loc[~blank].groupby("bin", sort=False):
        draws = rng.integers(len(rows), size=(299, len(rows)))
        truth, predicted = rows.is_match.to_numpy(), rows.p.to_numpy() >= .6
        parts = np.column_stack([predicted * truth, predicted, truth])[draws]
        weight = rows.weight.to_numpy()[draws]
        totals = (weight[:, :, None] * parts).sum(axis=1)
        unscaled += totals
        rescaled += totals * (frame.weight[frame.bin == name].sum() / weight.sum(axis=1))[:, None]

    def intervals(totals):
        tp, linked, true = totals.T
        return np.quantile(np.column_stack([tp / linked, tp / true, 2 * tp / (linked + true)]), [.025, .975], axis=0)

    for index, metric in enumerate([result.precision, result.recall, result.f1]):
        assert metric[1:] == pytest.approx(intervals(rescaled)[:, index], rel=1e-12)
    # Resampling the unscaled weights would centre recall's interval on the old, biased estimate.
    assert intervals(unscaled)[0, 1] > result.recall[2]


def test_uneven_blank_labels_no_longer_bias_recall_on_repeated_samples():
    # Known truth; labels missing completely at random WITHIN a bin, but in 70% of the rows of
    # the three low bins and none of the high ones: a labeler who starts with the likely matches.
    sizes = np.array([30_000, 20_000, 12_000, 8_000, 4_000, 2_000])
    rates = [.02, .1, .35, .65, .9, .98]
    population = scores(np.repeat([.02, .1, .35, .65, .9, .98], sizes))
    truth = np.concatenate([np.arange(size) < round(size * rate) for size, rate in zip(sizes, rates)])
    predicted = population.p.to_numpy() >= .5
    target = (truth & predicted).sum() / truth.sum()
    adjusted, dropped, covered, covered_before = [], [], [], []
    for seed in range(60):
        sample = audit_sample(population, n=600, seed=seed)
        blank = (np.random.default_rng(seed + 500).random(len(sample)) < .7) & (sample.p < .5).to_numpy()
        sample["is_match"] = pd.Series(truth[sample.left_id.to_numpy()].astype(float)).mask(blank)
        result = evaluate(sample, n_boot=199, seed=seed)
        before = evaluate(sample.loc[~blank], n_boot=199, seed=seed)  # no blank rows: nothing to adjust
        adjusted.append(result.recall[0])
        dropped.append(before.recall[0])
        covered.append(result.recall[1] <= target <= result.recall[2])
        covered_before.append(before.recall[1] <= target <= before.recall[2])
    assert np.mean(adjusted) == pytest.approx(target, abs=.02)
    assert np.mean(covered) >= .85, np.mean(covered)
    # What the evaluator reported before: recall far too high, inside intervals that miss the truth.
    assert np.mean(dropped) > target + .15 and np.mean(covered_before) < .2


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


def test_one_to_one_competition_separates_pair_scores_from_final_links():
    frame = pd.DataFrame({"left_id": ["A", "A"], "right_id": ["X", "Y"],
                          "p": [.9, .8], "sim": [1., 1.]})
    links = resolve(frame)
    assert links.right_id.tolist() == ["X"]
    sample = audit_sample(frame, links=links)
    sample["is_match"] = sample.right_id.eq("X")
    pair = evaluate(sample, n_boot=20)
    final = evaluate(sample, mode="selected", n_boot=20)
    assert pair.mode == "threshold" and pair.threshold == .5
    assert pair.precision[0] == .5 and pair.recall[0] == 1
    assert final.mode == "selected" and final.threshold is None
    assert final.precision[0] == final.recall[0] == final.f1[0] == 1
    assert final.brier == pair.brier
    pd.testing.assert_frame_equal(final.calibration, pair.calibration)
    assert "Pair-scoring evaluation at threshold 0.5" in pair.summary()
    assert "Final-link evaluation using saved selected membership" in final.to_markdown()
    assert "no threshold is reapplied" in final.summary()
    assert "Brier and calibration assess pair scores" in final.summary()
    assert "do not estimate blocking uncertainty" in final.summary()


@pytest.mark.parametrize("how, chosen", [
    ("one-to-one", {("A", "Y"), ("B", "X")}),
    ("many-to-one", {("A", "X"), ("B", "X")}),
    ("one-to-many", {("A", "X"), ("A", "Y")}),
    ("many-to-many", {("A", "X"), ("A", "Y"), ("B", "X")}),
])
def test_selection_is_actual_global_or_directional_assignment(how, chosen):
    frame = pd.DataFrame({"left_id": ["A", "A", "B"], "right_id": ["X", "Y", "X"],
                          "p": [.9, .8, .85], "sim": [1., 1., 1.]})
    sample = audit_sample(frame, links=resolve(frame, how=how), bins=(0, 1))
    selected = sample.loc[sample.selected]
    assert set(zip(selected.left_id, selected.right_id)) == chosen
    truth = {("A", "Y"), ("B", "X")}
    sample["is_match"] = [pair in truth for pair in zip(sample.left_id, sample.right_id)]
    result = evaluate(sample, mode="selected", n_boot=20)
    assert result.precision[0] == pytest.approx(len(chosen & truth) / len(chosen))
    assert result.recall[0] == pytest.approx(len(chosen & truth) / len(truth))


@pytest.mark.parametrize("options, count", [
    ({"min_margin": .2}, 0), ({"threshold": .95}, 0), ({"threshold": .85}, 1),
])
def test_margin_and_nondefault_threshold_are_already_reflected_in_selection(options, count):
    frame = pd.DataFrame({"left_id": ["A", "A"], "right_id": ["X", "Y"],
                          "p": [.9, .8], "sim": [1., 1.]})
    sample = audit_sample(frame, links=resolve(frame, **options))
    sample["is_match"] = sample.right_id.eq("X")
    assert sample.selected.sum() == count
    # A new evaluation threshold cannot alter the saved decision, in either direction.
    for threshold in (0, 1):
        result = evaluate(sample, mode="selected", threshold=threshold, n_boot=20)
        assert result.recall[0] == count
        assert result.f1[0] == count
        if not count:
            assert np.isnan(result.precision[0]) and "No selected links" in result.summary()
        else:
            assert result.precision[0] == 1


def test_low_threshold_selection_is_not_rethresholded_at_default_half():
    frame = scores([.3, .2]).assign(sim=1.)
    sample = audit_sample(frame, links=resolve(frame, threshold=.25), bins=(0, 1))
    sample["is_match"] = sample.p.eq(.3)
    result = evaluate(sample, mode="selected", n_boot=20)
    assert result.precision[0] == result.recall[0] == result.f1[0] == 1
    assert evaluate(sample, n_boot=20).recall[0] == 0


def test_selected_bootstrap_retains_weights_and_is_reproducible():
    frame = labeled([.8, .7, .6, .1], [1, 0, 1, 0], [2, 4, 10, 20], ["high", "high", "low", "low"])
    frame = frame.assign(left_id=range(4), right_id=range(4), selected=[True, True, False, False])
    result = evaluate(frame, mode="selected", n_boot=299, seed=48)
    assert result.precision[0] == pytest.approx(2 / 6)
    assert result.recall[0] == pytest.approx(2 / 12)
    assert result.f1[0] == pytest.approx(4 / 18)
    rng = np.random.default_rng(48)
    draws = np.column_stack([rng.integers(0, 2, size=(299, 2)), rng.integers(2, 4, size=(299, 2))])
    values = []
    for draw in draws:
        boot = frame.iloc[draw]
        tp = (boot.weight * boot.is_match * boot.selected).sum()
        linked = boot.weight[boot.selected].sum()
        true = (boot.weight * boot.is_match).sum()
        values.append([tp / linked, tp / true if true else np.nan, 2 * tp / (linked + true)])
    expected = np.nanquantile(values, [.025, .975], axis=0)
    repeated = evaluate(frame, mode="selected", n_boot=299, seed=48)
    for index, name in enumerate(("precision", "recall", "f1")):
        assert getattr(result, name)[1:] == pytest.approx(expected[:, index])
        assert getattr(result, name) == getattr(repeated, name)


def test_sampling_preserves_selection_with_unequal_bin_weights():
    frame = scores([.1] * 30 + [.9] * 10)
    links = frame.iloc[30:35]
    sample = audit_sample(frame, links=links, n=10, seed=8)
    assert sample.weight.sum() == len(frame)
    assert sample.selected.dtype == bool
    assert sample.selected.equals(sample.left_id.between(30, 34))
    sample["is_match"] = sample.left_id.mod(2).eq(0)
    result = evaluate(sample, mode="selected", n_boot=20)
    tp = (sample.weight * sample.selected * sample.is_match).sum()
    assert result.precision[0] == pytest.approx(tp / sample.weight[sample.selected].sum())
    assert result.recall[0] == pytest.approx(tp / sample.weight[sample.is_match].sum())


def test_unjudged_pairs_are_not_in_audit_or_recall_denominator():
    frame = scores([.9, .8, np.nan, np.nan]).assign(source=["exact", "jev", "unjudged", "error"])
    sample = audit_sample(frame, links=frame.iloc[:1], bins=(0, 1))
    assert set(sample.left_id) == {0, 1}
    assert sample.weight.sum() == 2
    sample["is_match"] = True
    result = evaluate(sample, mode="selected", n_boot=20)
    assert result.recall[0] == .5
    assert "excludes pairs without a probability" in result.summary()
    with pytest.raises(ValueError, match="without a judged 'p'"):
        audit_sample(frame, links=frame.iloc[2:3])


def test_selected_incomplete_labels_and_empty_strata():
    frame = labeled([.9, .8, .1], [1, "", 0], bins=["high", "high", "low"])
    frame = frame.assign(left_id=range(3), right_id=range(3), selected=[1, 0, 0])
    result = evaluate(frame, mode="selected", n_boot=20)
    assert result.n_labeled == 2 and result.n_unlabeled == 1
    assert result.precision[0] == 1
    assert "1 blank labels: labeled pairs were reweighted to their bin's full sampling weight (high x2)" \
        in result.summary()
    assert "related to the truth within a bin can still bias" in result.summary()
    frame.loc[2, "is_match"] = ""
    result = evaluate(frame, mode="selected", n_boot=20)
    assert np.isnan(result.precision).all() and np.isnan(result.brier)
    assert "No labels in bin(s) low" in result.summary()
    assert "hold 33.3% of the sampled weight (low 33.3%)" in result.summary()


def test_selected_csv_representations_and_old_files():
    values = [1, 0, True, False, " TRUE ", "false", 1.0, 0.0, "1", "0"]
    frame = labeled([.9] * len(values), [1] * len(values))
    frame = frame.assign(left_id=range(len(values)), right_id=range(len(values)), selected=values)
    for data in (frame, pd.read_csv(StringIO(frame.to_csv(index=False))),
                 pd.read_csv(StringIO(frame.to_csv(index=False)), dtype=str)):
        result = evaluate(data, mode="selected", n_boot=20)
        assert result.precision[0] == 1 and result.recall[0] == .5
    old = frame.drop(columns="selected")
    assert evaluate(old, n_boot=20).recall[0] == 1
    with pytest.raises(ValueError, match="from actual final links"):
        evaluate(old, mode="selected", n_boot=20)


@pytest.mark.parametrize("value", ["yes", "no", "maybe", "", " ", None, np.nan, pd.NA, 2, -1, .5, np.inf])
def test_invalid_selection_is_rejected_even_for_unlabeled_rows(value):
    frame = labeled([.9, .8], [1, ""]).assign(left_id=[0, 1], right_id=[0, 1], selected=[True, value])
    with pytest.raises(ValueError, match="column 'selected'.*nonmissing"):
        evaluate(frame, mode="selected", n_boot=20)


def test_selection_pair_keys_and_supplied_link_membership_are_validated():
    frame = scores([.9, .8])
    for links, error in ((pd.concat([frame.iloc[:1]] * 2), "duplicate"),
                         (frame.assign(left_id=["other", "absent"]), "absent from scores"),
                         (frame.assign(left_id=[None, None]), "missing IDs")):
        with pytest.raises(ValueError, match=error):
            audit_sample(frame, links=links)
    sample = audit_sample(frame, links=frame.iloc[:1]).assign(is_match=1)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(pd.concat([sample] * 2), mode="selected", n_boot=20)
    with pytest.raises(ValueError, match="left_id"):
        evaluate(sample.drop(columns="left_id"), mode="selected", n_boot=20)
    with pytest.raises(ValueError, match="mode must be"):
        evaluate(sample, mode="unknown")


def test_standalone_scores_do_not_infer_selection_and_empty_links_are_known_false():
    frame = scores([.9, .8]).assign(selected=True)
    assert "selected" not in audit_sample(frame)
    sample = audit_sample(frame, links=frame.iloc[:0])
    assert not sample.selected.any()
    empty = audit_sample(frame, links=frame.iloc[:1], n=0)
    assert empty.empty and empty.selected.dtype == bool

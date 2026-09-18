from fractions import Fraction
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import pytest

from jlink.resolve import resolve


def scores(rows):
    frame = pd.DataFrame(rows, columns=["left_id", "right_id", "p", "sim"])
    frame["block"] = "test"
    frame["source"] = np.where(frame["p"].isna(), "unjudged", "jev")
    frame["error"] = pd.NA
    return frame


def pairs(frame):
    return list(frame[["left_id", "right_id"]].itertuples(index=False, name=None))


def objective(frame):
    return tuple(sum((Fraction(float(value)) for value in frame[col]), Fraction()) for col in ("p", "sim"))


def brute_force(frame):
    groups = [group for _, group in frame.groupby("left_id", sort=True)]
    best = (Fraction(), Fraction())

    def visit(index, used, total):
        nonlocal best
        if index == len(groups):
            best = max(best, total)
            return
        visit(index + 1, used, total)
        for row in groups[index].itertuples():
            if row.right_id not in used:
                visit(index + 1, used | {row.right_id},
                      (total[0] + Fraction(row.p), total[1] + Fraction(row.sim)))

    visit(0, set(), (Fraction(), Fraction()))
    return best


@pytest.mark.parametrize("how, expected", [
    ("many-to-many", [("a", "x"), ("a", "y"), ("b", "x"), ("b", "z")]),
    ("many-to-one", [("a", "x"), ("b", "x")]),
    ("one-to-many", [("a", "x"), ("a", "y"), ("b", "z")]),
    ("one-to-one", [("a", "y"), ("b", "x")]),
])
def test_cardinality_modes(how, expected):
    frame = scores([("a", "x", .9, .9), ("a", "y", .8, .8),
                    ("b", "x", .85, .85), ("b", "z", .6, .6), ("c", "z", np.nan, .9)])
    result = resolve(frame, how=how, min_margin=-1)
    assert set(pairs(result)) == set(expected)
    assert list(result.columns) == [*frame.columns, "margin"]
    assert result.index.equals(pd.RangeIndex(len(result)))
    assert result["p"].is_monotonic_decreasing


def test_global_optimum_beats_greedy_and_never_invents_a_pair():
    frame = scores([("a", "x", .9, .9), ("a", "y", .8, .8), ("b", "x", .8, .8)])
    result = resolve(frame, min_margin=-1)
    assert set(pairs(result)) == {("a", "y"), ("b", "x")}
    assert result["p"].sum() == 1.6


def test_unmatched_ids_are_allowed_even_when_a_complete_matching_exists():
    frame = scores([("a", "x", .99, .9), ("a", "y", .1, .8), ("b", "x", .1, .8)])
    assert pairs(resolve(frame, threshold=0, min_margin=-1)) == [("a", "x")]


def test_components_and_transposed_rectangular_assignment():
    frame = scores([(1, "x", .8, .6), (2, "x", .9, .6), (2, "y", .8, .6),
                    (3, "y", .9, .6), (4, "z", .7, .5), (5, "u", .6, .4), (5, "v", .9, .6)])
    assert set(pairs(resolve(frame, min_margin=-1))) == {(2, "x"), (3, "y"), (4, "z"), (5, "v")}


def test_one_to_one_matches_brute_force_on_small_random_graphs():
    rng = np.random.default_rng(412)
    for _ in range(160):
        rows = [(a, b, float(rng.integers(0, 9) / 8), float(rng.integers(0, 9) / 8))
                for a in range(rng.integers(1, 5)) for b in range(rng.integers(1, 5)) if rng.random() < .75]
        frame = scores(rows)
        threshold = float(rng.choice([0, .25, .5]))
        eligible = frame[frame["p"] >= threshold]
        result = resolve(frame, threshold=threshold, min_margin=-1)
        assert objective(result) == brute_force(eligible)
        assert result["left_id"].is_unique and result["right_id"].is_unique
        assert set(pairs(result)) <= set(pairs(eligible))
        shuffled = resolve(frame.sample(frac=1, random_state=17), threshold=threshold, min_margin=-1)
        pd.testing.assert_frame_equal(result, shuffled)


def test_similarity_cannot_trade_away_a_tiny_probability_difference():
    high = np.nextafter(.5, 1)
    frame = scores([("a", "x", high, 0), ("a", "y", .5, 1),
                    ("b", "x", .5, 1), ("b", "y", high, 0)])
    result = resolve(frame, min_margin=-1)
    assert pairs(result) == [("a", "x"), ("b", "y")]
    assert objective(result) == brute_force(frame)


def test_near_ties_match_exact_brute_force_for_both_objectives():
    # An epsilon alone either loses probability or rounds away the similarity tie break.
    rng = np.random.default_rng(69)
    for _ in range(160):
        frame = scores([(a, b, .5 + rng.integers(-3, 4) * 2.**-52, rng.integers(0, 9) / 8)
                        for a in range(3) for b in range(3)])
        assert objective(resolve(frame, threshold=0, min_margin=-1)) == brute_force(frame)


def test_all_small_unweighted_graphs_prefer_earliest_id_pairs():
    # This also covers non-square components and swaps along alternating paths.
    for mask in range(1, 512):
        edges = [(a, b) for a in range(3) for b in range(3) if mask & (1 << (3 * a + b))]
        options = [subset for length in range(4) for subset in combinations(edges, length)
                   if len({a for a, _ in subset}) == length == len({b for _, b in subset})]
        expected = min(options, key=lambda subset: (-len(subset), subset))
        frame = scores([(a, b, .5, .5) for a, b in reversed(edges)])
        assert tuple(pairs(resolve(frame))) == expected


@pytest.mark.parametrize("how", ["many-to-one", "one-to-many", "one-to-one"])
def test_equal_probability_prefers_similarity(how):
    frame = scores([("a", "x", .8, .3), ("a", "y", .8, .9),
                    ("b", "x", .8, .9), ("b", "y", .8, .3)])
    assert pairs(resolve(frame, how=how)) == [("a", "y"), ("b", "x")]


def test_equal_probability_and_similarity_use_id_order():
    frame = scores([(a, b, .8, .6) for a in [30, 10, 20] for b in ["z", "x", "y"]])
    for seed in range(8):
        assert pairs(resolve(frame.sample(frac=1, random_state=seed))) == [(10, "x"), (20, "y"), (30, "z")]


def test_margin_uses_below_threshold_pairs_ties_and_both_sides():
    frame = scores([("a", "x", .9, .8), ("a", "y", .4, .7),
                    ("b", "x", .8, .7), ("c", "z", .7, .8), ("c", "u", .7, .8),
                    ("d", "v", .6, .8), ("d", "w", np.nan, .8), ("e", "q", .8, .8),
                    ("e", "r", .3, .8)])
    result = resolve(frame, how="many-to-many", min_margin=-1).set_index(["left_id", "right_id"])
    assert result.loc[("a", "x"), "margin"] == pytest.approx(.1)
    assert result.loc[("b", "x"), "margin"] == pytest.approx(-.1)
    assert result.loc[("e", "q"), "margin"] == pytest.approx(.5)
    assert result.loc[("c", "z"), "margin"] == 0
    assert np.isnan(result.loc[("d", "v"), "margin"])
    assert ("b", "x") not in pairs(resolve(frame, how="many-to-many", min_margin=0))
    assert ("b", "x") in pairs(resolve(frame, how="many-to-many"))  # no margin filter unless asked
    assert ("d", "v") in pairs(resolve(frame, min_margin=1))


def test_vectorized_margin_matches_rowwise_definition_with_repeated_index():
    rng = np.random.default_rng(50)
    frame = scores([(a, b, rng.choice([.1, .2, .5, .8, np.nan]), .7)
                    for a in range(20) for b in range(15) if rng.random() < .2])
    frame.index = [0] * len(frame)
    original = frame.copy(deep=True)
    result = resolve(frame, how="many-to-many", threshold=0, min_margin=-1)
    for row in result.itertuples():
        other = frame[((frame.left_id == row.left_id) | (frame.right_id == row.right_id))
                      & ~((frame.left_id == row.left_id) & (frame.right_id == row.right_id))]
        expected = row.p - other.p.max()
        assert row.margin == pytest.approx(expected, nan_ok=True)
    pd.testing.assert_frame_equal(frame, original)


def test_margin_is_applied_after_assignment_without_rematching():
    frame = scores([("a", "x", .9, .8), ("a", "y", .8, .8), ("b", "x", .8, .8)])
    assert resolve(frame, min_margin=0).empty
    assert set(pairs(resolve(frame))) == {("a", "y"), ("b", "x")}  # the optimal assignment, unfiltered


def test_large_component_warns_and_uses_deterministic_greedy():
    frame = scores([(0, "x", .9, .9), (0, "y", .8, .9), (1, "x", .8, .8)]
                   + [(a, "x", .6, .6) for a in range(2, 2002)])
    with pytest.warns(RuntimeWarning, match="2,002 left IDs.*greedy"):
        result = resolve(frame, min_margin=-1)
    assert pairs(result) == [(0, "x")]
    with pytest.warns(RuntimeWarning):
        shuffled = resolve(frame.sample(frac=1, random_state=2), min_margin=-1)
    pd.testing.assert_frame_equal(result, shuffled)


def test_exactly_2000_nodes_does_not_warn():
    frame = scores([(a, "x", .6, .6) for a in range(2000)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pairs(resolve(frame)) == [(0, "x")]


@pytest.mark.parametrize("frame", [scores([]), scores([("a", "x", np.nan, .4)])])
def test_empty_and_unjudged_scores(frame):
    result = resolve(frame)
    assert result.empty and "margin" in result


@pytest.mark.parametrize("kwargs, message", [
    ({"how": "invalid"}, "how"), ({"threshold": -1}, "threshold"),
    ({"threshold": np.inf}, "threshold"), ({"min_margin": np.nan}, "min_margin"),
])
def test_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve(scores([]), **kwargs)


@pytest.mark.parametrize("column, value, message", [
    ("p", "bad", "p"), ("p", 1.2, "p"), ("p", np.inf, "p"), ("p", .5 + .1j, "p"),
    ("sim", np.nan, "sim"), ("sim", -1, "sim"), ("left_id", None, "left_id"),
])
def test_invalid_data(column, value, message):
    frame = scores([("a", "b", .9, .8)]).astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        resolve(frame)


def test_duplicate_pairs_and_missing_columns():
    frame = scores([("a", "b", .9, .8), ("a", "b", .8, .7)])
    with pytest.raises(ValueError, match="duplicate"):
        resolve(frame)
    with pytest.raises(ValueError, match="sim"):
        resolve(frame.iloc[:1].drop(columns="sim"))

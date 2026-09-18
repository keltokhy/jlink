"""Offline blocking checks; run this file with --benchmark ROWS for the scale measurement."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from jlink import block
from jlink.fields import record_text


def _dense_cosines(left, right, a, b, n=(2, 4)):
    text = pd.concat([record_text(left, a), record_text(right, b)])
    try:
        vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=n, norm="l2").fit_transform(text)
    except ValueError as error:
        assert "empty vocabulary" in str(error)
        return np.zeros((len(left), len(right)))
    return np.clip((vectors[:len(left)] @ vectors[len(left):].T).toarray(), 0, 1)


def _dense_neighbors(sim, k, min_sim):
    pairs = []
    for i, scores in enumerate(sim):
        ordered = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
        pairs.extend((i, j) for j in ordered[:k] if scores[j] >= min_sim)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def test_exact_normalization_multiple_fields_and_positions():
    left = pd.DataFrame({"firm": ["Café & Co.", "cafe and co", "other", "a b", "a"],
                         "city": ["NY", "ny", "NY", "c", "b c"]}, index=[8, 3, 2, 1, 0])
    right = pd.DataFrame({"name": ["CAFE AND CO", "cafe & co", "other", "a", "a b"],
                          "town": ["ny", "NY", "LA", "b c", "c"]}, index=[9, 4, 3, 2, 1])
    result = block.exact(("firm", "name"), ("city", "town")).pairs(left, right)
    assert_array_equal(result, [[0, 0], [0, 1], [1, 0], [1, 1], [3, 4], [4, 3]])
    assert result.dtype == np.int64


def test_exact_does_not_match_missing_or_incomplete_keys():
    frame = pd.DataFrame({"name": [None, np.nan, pd.NA, "", "!!!", "Acme", "Acme"],
                          "city": ["NY"] * 6 + [None]})
    assert_array_equal(block.exact("name", "city").pairs(frame, frame), [[5, 5]])


def test_names_and_custom_names():
    assert block.exact("name", ("city", "town")).name == "exact:name,city=town"
    assert block.ngrams("name").name == "ngrams:name"
    assert block.initials(("name", "firm")).name == "initials:name=firm"
    for factory in (block.exact, block.ngrams, block.initials):
        assert factory("name", name="custom").name == "custom"


@pytest.mark.parametrize("k,min_sim,n", [(1, 0.1, (2, 4)), (3, 0.3, (2, 4)),
                                         (20, 0, (1, 3)), (4, 0, (8, 12))])
def test_ngrams_matches_dense_reference(k, min_sim, n):
    left = pd.DataFrame({"firm": ["Acme", "IBM", "International Business Machines", None],
                         "city": ["NY", "NY", "Paris", None]})
    right = pd.DataFrame({"name": ["Acme", "Acmé", "IBM", "International Business", "Zed", None],
                          "town": ["NY", "NY", "Paris", "Paris", "Cairo", None]})
    sim = _dense_cosines(left, right, ["firm", "city"], ["name", "town"], n)
    result = block.ngrams(("firm", "name"), ("city", "town"), k=k, n=n,
                          min_sim=min_sim).pairs(left, right)
    assert_array_equal(result, _dense_neighbors(sim, k, min_sim))
    assert result.dtype == np.int64


def test_ngrams_hand_checked_neighbors_and_threshold():
    left = pd.DataFrame({"name": ["ab", "zz", "", None]})
    right = pd.DataFrame({"name": ["zz", "ab", "ac", None]})
    assert_array_equal(block.ngrams("name", k=1).pairs(left, right), [[0, 1], [1, 0]])
    assert_array_equal(block.ngrams("name", k=2, min_sim=0.99).pairs(left, right), [[0, 1], [1, 0]])
    # Only the boundary bigram " a" is shared between ab and ac.
    a = pd.DataFrame({"name": ["ab"]})
    b = pd.DataFrame({"name": ["ac"]})
    score = _dense_cosines(a, b, ["name"], ["name"])[0, 0]
    assert_array_equal(block.ngrams("name", min_sim=score).pairs(a, b), [[0, 0]])
    assert block.ngrams("name", min_sim=score + 1e-6).pairs(a, b).shape == (0, 2)


@pytest.mark.parametrize("text,n", [([None, "", pd.NA, np.nan], (2, 4)), (["", None], (9, 10))])
def test_empty_vocabulary(text, n):
    frame = pd.DataFrame({"name": text})
    assert block.ngrams("name", n=n).pairs(frame, frame).shape == (0, 2)
    expected = [[i, j] for i in range(len(frame)) for j in range(2)]
    assert_array_equal(block.ngrams("name", k=2, n=n, min_sim=0).pairs(frame, frame), expected)


def test_unit_cosines_pass_a_threshold_of_one():
    frame = pd.DataFrame({"name": ["Acme", "International Business Machines", "AB", "Alpha Beta",
                                   "Long Long Long"]})
    assert_array_equal(block.ngrams("name", min_sim=1).pairs(frame, frame), [[i, i] for i in range(5)])


def test_char_wb_keeps_short_padded_words_for_long_ngrams():
    frame = pd.DataFrame({"name": ["a", "b"]})
    assert_array_equal(block.ngrams("name", n=(9, 10)).pairs(frame, frame), [[0, 0], [1, 1]])


def test_zero_similarity_neighbors_include_sparse_zeros_in_position_order():
    left = pd.DataFrame({"name": [None, "zz"]})
    right = pd.DataFrame({"name": ["ab", "cd", "zz", "ef"]})
    assert_array_equal(block.ngrams("name", k=3, min_sim=0).pairs(left, right),
                       [[0, 0], [0, 1], [0, 2], [1, 2], [1, 0], [1, 1]])


def test_chunk_sizes_do_not_change_ties_or_output(monkeypatch):
    left = pd.DataFrame({"name": ["Acme", "beta", "ab", None, "IBM"] * 7},
                        index=np.arange(100, 135)[::-1])
    right = pd.DataFrame({"name": ["Acme", "beta", "ab", "IBM", "xx"] * 7},
                         index=np.arange(200, 235)[::-1])
    blocker = block.ngrams("name", k=3, min_sim=0)
    expected_pairs = blocker.pairs(left, right)
    assert_array_equal(expected_pairs[:3], [[0, 0], [0, 5], [0, 10]])
    expected = block.candidates(left, right, on="name", blockers=[blocker])
    for rows in (1, 2, 9, 1000):
        monkeypatch.setattr(block, "_CHUNK_ROWS", rows)
        monkeypatch.setattr(block, "_PAIR_CHUNK_ROWS", rows)
        assert_array_equal(blocker.pairs(left, right), expected_pairs)
        assert_frame_equal(block.candidates(left, right, on="name", blockers=[blocker]), expected)


def test_sparse_product_is_bounded_and_not_densified(monkeypatch):
    original = csr_matrix.__matmul__
    seen = []

    def multiply(self, other):
        seen.append(self.shape[0])
        assert self.shape[0] <= 3
        return original(self, other)

    def forbidden(*args, **kwargs):
        raise AssertionError("blocking must not densify a similarity matrix")

    monkeypatch.setattr(block, "_PRODUCT_BYTES", 12 * 23 * 3)
    monkeypatch.setattr(csr_matrix, "__matmul__", multiply)
    monkeypatch.setattr(csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(csr_matrix, "todense", forbidden)
    left = pd.DataFrame({"name": ["Acme"] * 13})
    right = pd.DataFrame({"name": ["Acme"] * 23})
    result = block.candidates(left, right, on="name")
    assert len(result) == 130
    assert seen == [3, 3, 3, 3, 1]


def test_initials_both_directions_and_acronym_must_be_one_letter_token():
    left = pd.DataFrame({"firm": ["IBM", "International Business Machines", "B&Q", "ab2", "a",
                                   "ibm inc", None, "International Business Machines", "ab"]})
    right = pd.DataFrame({"name": ["International Business Machines", "IBM", "Business Quality",
                                    "Alpha Beta Two", "Alpha", "Alpha Beta", "I B M"]})
    assert_array_equal(block.initials(("firm", "name")).pairs(left, right),
                       [[0, 0], [0, 6], [1, 1], [7, 1], [8, 5]])
    assert_array_equal(block.initials(("firm", "name"), min_len=3).pairs(left, right),
                       [[0, 0], [0, 6], [1, 1], [7, 1]])


@pytest.mark.parametrize("ignored", ["and", "of", "the", "for", "inc", "corp", "co", "ltd", "llc",
                                      "plc", "company", "corporation", "incorporated", "limited"])
def test_initials_ignores_each_specified_word(ignored):
    left = pd.DataFrame({"name": ["AB", f"Alpha {ignored.upper()} Béta"]})
    right = pd.DataFrame({"name": [f"Alpha Beta {ignored}", "ab"]})
    assert_array_equal(block.initials("name").pairs(left, right), [[0, 0], [1, 1]])


def test_initials_empty_and_ignored_text_never_pairs():
    frame = pd.DataFrame({"name": [None, "", "!!!", "and of the", "inc corp ltd"]})
    assert block.initials("name").pairs(frame, frame).shape == (0, 2)


def test_candidates_union_provenance_all_on_similarity_and_order():
    left = pd.DataFrame({"id": ["z", "a"], "firm": ["Acme", "IBM"], "city": ["NY", "NY"]})
    right = pd.DataFrame({"rid": [8, 2, 1, 6],
                          "name": ["Acme", "Acme", "International Business Machines", "Acme"],
                          "town": ["LA", "NY", "NY", "NY"]})
    passes = [block.exact(("firm", "name")), block.initials(("firm", "name")),
              block.ngrams(("firm", "name"), k=1, n=(1, 1))]
    result = block.candidates(left, right, on=[("firm", "name"), ("city", "town")],
                              blockers=passes, left_id="id", right_id="rid")
    assert list(result.columns) == ["left_id", "right_id", "block", "sim"]
    assert list(zip(result.left_id, result.right_id)) == [("z", 2), ("z", 6), ("z", 8), ("a", 1), ("a", 8)]
    assert result.block.tolist() == ["exact:firm=name", "exact:firm=name", "exact:firm=name+ngrams:firm=name",
                                     "initials:firm=name", "ngrams:firm=name"]
    expected = _dense_cosines(left, right, ["firm", "city"], ["name", "town"])
    for row in result.itertuples():
        i = left.index[left.id == row.left_id][0]
        j = right.index[right.rid == row.right_id][0]
        assert row.sim == pytest.approx(expected[i, j], abs=1e-14)
    assert result.sim.iloc[2] < 1  # Name-only exact match is not a full-record cosine of one.
    assert not result.duplicated(["left_id", "right_id"]).any()
    assert_frame_equal(result, block.candidates(left, right, on=[("firm", "name"), ("city", "town")],
                                               blockers=passes, left_id="id", right_id="rid"))


def test_default_pass_and_index_ids():
    left = pd.DataFrame({"name": ["alpha", "beta"], "city": ["NY", "LA"]}, index=["z", "a"])
    right = pd.DataFrame({"firm": ["alpha", "beta"], "town": ["NY", "LA"]}, index=[7, 3])
    on = [("name", "firm"), ("city", "town")]
    assert_frame_equal(block.candidates(left, right, on=on),
                       block.candidates(left, right, on=on, blockers=[block.ngrams(*on)]))
    result = block.candidates(left, right, on=on, blockers=[block.exact(*on)])
    assert result.left_id.tolist() == ["z", "a"]
    assert result.right_id.tolist() == [7, 3]


def test_all_on_text_missing_still_scores_exact_and_initials_candidates():
    left = pd.DataFrame({"key": ["IBM", "acme"], "on": [None, pd.NA]})
    right = pd.DataFrame({"key": ["International Business Machines", "Acme"], "on": [np.nan, ""]})
    result = block.candidates(left, right, on="on", blockers=[block.exact("key"), block.initials("key")])
    assert list(zip(result.left_id, result.right_id, result.sim)) == [(0, 0, 0.0), (1, 1, 0.0)]


@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_empty_frames(side):
    left = pd.DataFrame({"name": ["Acme"]})
    right = left.copy()
    if side in ("left", "both"):
        left = left.iloc[:0]
    if side in ("right", "both"):
        right = right.iloc[:0]
    for factory in (block.exact, block.ngrams, block.initials):
        pairs = factory("name").pairs(left, right)
        assert pairs.shape == (0, 2) and pairs.dtype == np.int64
        result = block.candidates(left, right, on="name", blockers=[factory("name")])
        assert result.empty
        assert list(result.columns) == ["left_id", "right_id", "block", "sim"]
        assert result.sim.dtype == float


def test_explicit_empty_blockers():
    frame = pd.DataFrame({"name": ["Acme"]})
    assert block.candidates(frame, frame, on="name", blockers=[]).empty


@pytest.mark.parametrize("dtype,values", [("string", ["001", "002"]), ("Int64", [2, 1]),
                                         ("category", ["b", "a"])])
def test_id_values_and_dtypes_survive(dtype, values):
    frame = pd.DataFrame({"id": pd.Series(values, dtype=dtype), "name": ["Acme", "Beta"]})
    result = block.candidates(frame, frame, on="name", left_id="id", right_id="id",
                              blockers=[block.exact("name")])
    assert result.left_id.tolist() == values
    assert result.right_id.tolist() == values
    assert result.left_id.dtype == frame.id.dtype == result.right_id.dtype


def test_multiindex_ids_survive_as_tuples():
    frame = pd.DataFrame({"name": ["Acme", "Beta"]}, index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]))
    result = block.candidates(frame, frame, on="name", blockers=[block.exact("name")])
    assert result.left_id.tolist() == [("a", 1), ("b", 2)]


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("id_column", [None, "id"])
def test_duplicate_ids(side, id_column):
    good = pd.DataFrame({"id": [1, 2], "name": ["Acme", "Beta"]})
    bad = pd.DataFrame({"id": [1, 1], "name": ["Acme", "Beta"]}, index=[5, 5])
    left, right = (bad, good) if side == "left" else (good, bad)
    with pytest.raises(ValueError, match=f"{side}.*duplicate IDs"):
        block.candidates(left, right, on="name", left_id=id_column, right_id=id_column, blockers=[])


@pytest.mark.parametrize("side", ["left", "right"])
def test_missing_columns_name_the_side_and_column(side):
    good = pd.DataFrame({"name": ["Acme"], "id": [1]})
    bad = pd.DataFrame({"other": ["Acme"]})
    left, right = (bad, good) if side == "left" else (good, bad)
    for factory in (block.exact, block.ngrams, block.initials):
        with pytest.raises(ValueError, match=f"{side}.*'name'"):
            factory("name").pairs(left, right)
    with pytest.raises(ValueError, match=f"{side}.*'name'"):
        block.candidates(left, right, on="name", blockers=[])
    left, right = (good.drop(columns="id"), good) if side == "left" else (good, good.drop(columns="id"))
    with pytest.raises(ValueError, match=f"{side}.*'id'"):
        block.candidates(left, right, on="name", left_id="id", right_id="id")


def test_max_pairs_counts_union_not_repeated_proposals():
    frame = pd.DataFrame({"name": ["Acme", "Acme"]})
    passes = [block.exact("name", name="first"), block.exact("name", name="second")]
    result = block.candidates(frame, frame, on="name", blockers=passes, max_pairs=4)
    assert len(result) == 4
    assert result.block.tolist() == ["first+second"] * 4
    assert len(block.candidates(frame, frame, on="name", blockers=passes, max_pairs=None)) == 4
    with pytest.raises(ValueError, match=r"4 pairs.*max_pairs=3.*smaller `k`.*`exact`"):
        block.candidates(frame, frame, on="name", blockers=passes, max_pairs=3)
    with pytest.raises(ValueError, match="at least 1 pairs"):
        block.candidates(frame, frame, on="name", blockers=passes, max_pairs=0)
    assert block.candidates(frame, frame, on="name", blockers=[], max_pairs=0).empty


@pytest.mark.parametrize("kwargs,label", [({"k": 0}, "k"), ({"k": True}, "k"), ({"k": 1.5}, "k"),
                                         ({"n": (0, 4)}, "n"), ({"n": (4, 2)}, "n"),
                                         ({"n": (2,)}, "n"), ({"n": [2, 4]}, "n"),
                                         ({"min_sim": -0.1}, "min_sim"), ({"min_sim": 1.1}, "min_sim"),
                                         ({"min_sim": np.nan}, "min_sim"), ({"min_sim": "0.1"}, "min_sim")])
def test_invalid_ngram_options(kwargs, label):
    with pytest.raises(ValueError, match=label):
        block.ngrams("name", **kwargs)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_initials_min_len(value):
    with pytest.raises(ValueError, match="min_len"):
        block.initials("name", min_len=value)


@pytest.mark.parametrize("factory", [block.exact, block.ngrams, block.initials])
def test_invalid_columns_and_names(factory):
    with pytest.raises(ValueError, match="column name"):
        factory(("a", "b", "c"))
    for name in ("", "  ", 3):
        with pytest.raises(ValueError, match="name"):
            factory("name", name=name)


@pytest.mark.parametrize("value", [-1, 1.5, True, np.nan, "100"])
def test_invalid_max_pairs(value):
    frame = pd.DataFrame({"name": ["Acme"]})
    with pytest.raises(ValueError, match="max_pairs"):
        block.candidates(frame, frame, on="name", max_pairs=value)


@pytest.mark.parametrize("on", [None, [], 2, [("name", 1)]])
def test_invalid_on(on):
    frame = pd.DataFrame({"name": ["Acme"]})
    with pytest.raises(ValueError, match="on"):
        block.candidates(frame, frame, on=on)


@pytest.mark.parametrize("blockers", ["exact", ["exact"], block.exact("name")])
def test_invalid_blockers(blockers):
    frame = pd.DataFrame({"name": ["Acme"]})
    with pytest.raises(ValueError, match="blockers"):
        block.candidates(frame, frame, on="name", blockers=blockers)


@pytest.mark.parametrize("factory", [block.exact, block.ngrams])
def test_blocker_requires_columns(factory):
    with pytest.raises(ValueError, match="at least one field"):
        factory()


class _Fixed(block.Blocker):
    name = "fixed"

    def __init__(self, pairs):
        self.positions = pairs

    def pairs(self, left, right):
        return self.positions


@pytest.mark.parametrize("pairs", [np.array([0, 0]), np.array([[0.0, 0.0]]), [[0, 0]],
                                   np.array([[-1, 0]]), np.array([[1, 0]]), np.array([[0, 1]])])
def test_malformed_custom_blocker_pairs(pairs):
    frame = pd.DataFrame({"name": ["Acme"]})
    with pytest.raises(ValueError, match="blocker 'fixed'"):
        block.candidates(frame, frame, on="name", blockers=[_Fixed(pairs)])


def test_duplicate_custom_proposals_are_unioned():
    frame = pd.DataFrame({"name": ["Acme"]})
    result = block.candidates(frame, frame, on="name", blockers=[_Fixed(np.array([[0, 0], [0, 0]]))])
    assert len(result) == 1 and result.block.iloc[0] == "fixed"


def test_pairs_completeness_deduplicates_and_ignores_extra_columns():
    proposed = pd.DataFrame({"left_id": ["a", "a", "b", "extra"], "right_id": [1, 1, 2, 8], "sim": 1})
    truth = pd.DataFrame({"left_id": ["a", "a", "b", "c"], "right_id": [1, 1, 2, 3]})
    assert block.pairs_completeness(proposed, truth) == pytest.approx(2 / 3)
    assert block.pairs_completeness(truth, truth) == 1
    assert block.pairs_completeness(proposed.iloc[:0], truth) == 0
    assert np.isnan(block.pairs_completeness(proposed, truth.iloc[:0]))
    with pytest.raises(ValueError, match="truth.*'right_id'"):
        block.pairs_completeness(proposed, truth.drop(columns="right_id"))
    with pytest.raises(ValueError, match="candidates.*'left_id'"):
        block.pairs_completeness(proposed.drop(columns="left_id"), truth)


def _benchmark(rows):
    """Reproducible company-like corpus, including common tokens and duplicate names."""
    import json
    import resource
    import sys
    import time

    rng = np.random.default_rng(20260918)
    first = np.array("Acme Alder Alpine Atlas Beacon Blue Cedar Central Clear Coral Delta Eagle Elm Emerald "
                     "Fair First Golden Grand Green Harbor High Iron Lake Maple Metro North Oak Pacific "
                     "Pine Pioneer Prime Red River Silver South Summit Sun Union Valley West White".split(),
                     dtype=object)
    second = np.array("Bridge Brook Coast Creek Crown Field Forest Gate Glen Grove Hill Horizon Island "
                      "Lake Land Mountain Ocean Park Peak Point Port Ridge Rock Shore Spring Stone "
                      "Stream Tower Trail Tree View Village Vista Water Way Wood".split(), dtype=object)
    industry = np.array("Advisors Aerospace Agriculture Analytics Builders Chemicals Communications "
                        "Construction Consulting Design Development Electric Energy Engineering Foods "
                        "Healthcare Holdings Industries Insurance Logistics Manufacturing Materials "
                        "Media Mining Motors Networks Partners Properties Research Retail Services "
                        "Software Solutions Systems Technologies Trading Transport Ventures".split(),
                        dtype=object)
    suffix = np.array(["Inc", "Corp", "Company", "LLC", "Ltd", "Group", "Partners", ""], dtype=object)
    # Ten percent use a different legal suffix; ten percent abbreviate the first word.
    names = rng.choice(first, rows) + " " + rng.choice(second, rows) + " " + rng.choice(industry, rows)
    left_names = names + " " + rng.choice(suffix, rows)
    right_names = left_names.copy()
    right_names[::10] = names[::10] + " Limited"
    for i in range(1, rows, 10):
        words = str(right_names[i]).split()
        right_names[i] = words[0][0] + " " + " ".join(words[1:])
    right_names = right_names[rng.permutation(rows)]
    left, right = pd.DataFrame({"name": left_names}), pd.DataFrame({"name": right_names})
    started = time.perf_counter()
    pairs = block.ngrams("name", k=10).pairs(left, right)
    elapsed = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    print(json.dumps({"rows_per_side": rows, "pairs": len(pairs), "seconds": elapsed,
                      "peak_rss_mib": peak_bytes / 1024 ** 2, "seed": 20260918}), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=int, required=True, metavar="ROWS")
    _benchmark(parser.parse_args().benchmark)

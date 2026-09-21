"""Grouped/reverse candidate search, streaming limits, and public diagnostics."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix

from jlink import block


def _pairs(frame):
    return set(zip(frame.left_id, frame.right_id))


def test_exact_union_does_not_partition_but_within_searches_local_neighbors():
    left = pd.DataFrame({"name": ["Acme"], "state": ["NY"]})
    right = pd.DataFrame({"name": ["Acme", "Acme New York", "Acme NY"],
                          "state": ["CA", "NY", "NY"]})
    global_pass = block.ngrams("name", k=1)
    global_result = block.candidates(left, right, on="name", blockers=[global_pass])
    union = block.candidates(left, right, on="name", blockers=[global_pass, block.exact("state")])
    grouped = block.candidates(left, right, on="name", blockers=[block.within(global_pass, "state")])
    assert _pairs(global_result) == {(0, 0)}
    assert _pairs(union) == {(0, 0), (0, 1), (0, 2)}
    assert _pairs(grouped) == {(0, 2)}  # Search inside NY; post-filtering global top-1 yields nothing.


def test_mapped_groups_fit_vectors_separately_and_restore_original_ids(monkeypatch):
    left = pd.DataFrame({"id": pd.Series(["001", "002", "003", "004"], dtype="string"),
                         "firm": ["same"] * 4, "state": ["NY", "CA", "ny", "CA"]})
    right = pd.DataFrame({"rid": pd.Series([9, 8, 7, 6], dtype="Int64"),
                          "name": ["same"] * 4, "region": ["CA", "NY", "ca", "ny"]})
    left.index, right.index = [20, 3, 18, 7], [99, 12, 5, 40]
    observed = []
    original = block._vectors

    def vectors(a, b, *args):
        observed.append((a.index.tolist(), b.index.tolist()))
        return original(a, b, *args)

    monkeypatch.setattr(block, "_vectors", vectors)
    grouped = block.within(block.ngrams(("firm", "name"), k=1), ("state", "region"))
    assert_array_equal(grouped.pairs(left, right), [[0, 1], [2, 1], [1, 0], [3, 0]])
    assert observed == [([20, 18], [12, 40]), ([3, 7], [99, 5])]
    result = block.candidates(left, right, on=[("firm", "name")], blockers=[grouped],
                              left_id="id", right_id="rid")
    assert result.left_id.tolist() == ["001", "002", "003", "004"]
    assert result.right_id.tolist() == [8, 9, 8, 9]
    assert result.left_id.dtype == left.id.dtype
    assert result.right_id.dtype == right.rid.dtype


@pytest.mark.parametrize("missing,expected", [
    ("drop", [[3, 3]]),
    ("match", [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [3, 3]]),
])
def test_missing_group_components_are_explicit_and_never_wildcards(missing, expected):
    left = pd.DataFrame({"name": ["Acme"] * 4, "state": [None, "!!!", pd.NA, "NY"],
                         "year": ["2020", "2020", "2021", "2020"]})
    right = pd.DataFrame({"name": ["Acme"] * 4, "state": [np.nan, "", None, "ny"],
                          "year": ["2020", "2020", "2021", "2020"]})
    grouped = block.within(block.exact("name"), "state", "year", missing=missing)
    assert_array_equal(grouped.pairs(left, right), expected)


def test_nested_grouping_and_custom_blocker_keep_local_position_contract():
    class First(block.Blocker):
        name = "first"

        def pairs(self, left, right):
            return np.array([[0, 0]], dtype=np.int64)

    frame = pd.DataFrame({"name": ["AB", "Alpha Beta", "AB"], "state": ["NY"] * 3,
                          "year": [2020, 2021, 2020]}, index=[50, 30, 10])
    nested = block.within(block.within(First(), "year"), "state")
    assert_array_equal(nested.pairs(frame, frame), [[0, 0], [1, 1]])
    grouped_initials = block.within(block.initials("name"), "state")
    assert_array_equal(grouped_initials.pairs(frame, frame), [[0, 1], [1, 0], [1, 2], [2, 1]])


@pytest.mark.parametrize("values", [pd.Series([None, None, "NY"], dtype="category"),
                                    pd.Series([pd.NaT, pd.NaT, pd.Timestamp("2020-01-01")])])
def test_typed_group_nulls_follow_missing_policy(values):
    frame = pd.DataFrame({"name": ["Acme"] * 3, "group": values})
    assert_array_equal(block.within(block.exact("name"), "group").pairs(frame, frame), [[2, 2]])
    assert_array_equal(block.within(block.exact("name"), "group", missing="match").pairs(frame, frame),
                       [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
    assert_array_equal(block.exact("group").pairs(frame, frame), [[2, 2]])


@pytest.mark.parametrize("left_years,right_years", [
    ([1985, 1990, 2001], [1985.0, 1990.0, 2001.0, np.nan]),          # any missing value makes a column float
    (["1985", "1990", "2001"], ["1985.0", "1990.0", "2001.0", None]),  # the same column after a CSV round trip
    (np.array([1985, 1990, 2001], dtype=np.int32), ["1985", "1990.00", " 2001 ", ""]),
    (pd.array([1985, 1990, 2001], dtype="Int64"), pd.array([1985, 1990, 2001, None], dtype="Float64")),
    # Sixteen-digit identifiers are still exact as doubles, beyond where the judge's cleaning makes ints.
    ([4000000000000001, 4000000000000002, 4000000000000003], [4000000000000001.0, 4000000000000002.0,
                                                               4000000000000003.0, np.nan]),
])
def test_whole_number_keys_agree_across_integer_float_and_text_columns(left_years, right_years):
    left = pd.DataFrame({"name": ["Acme Corp", "Bolt Inc", "Cargo LLC"], "year": left_years})
    right = pd.DataFrame({"name": ["Acme Corporation", "Bolt Incorporated", "Cargo", "Acme Corp"],
                          "year": right_years})
    expected = [[0, 0], [1, 1], [2, 2]]
    assert_array_equal(block.exact("year").pairs(left, right), expected)
    grouped = block.within(block.ngrams("name", k=5, min_sim=0), "year")
    assert_array_equal(grouped.pairs(left, right), expected)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = block.candidates(left, right, on="name", blockers=[grouped])
    assert _pairs(result) == {(0, 0), (1, 1), (2, 2)}
    assert result.attrs["blocking"]["passes"][0]["proposed_pairs"] == 3


def test_numeric_keys_stay_distinct_unless_they_are_the_same_whole_number():
    left = pd.DataFrame({"key": [1985.5, 1985.0, "1985.05", "v1.0.0", "02139"]})
    right = pd.DataFrame({"key": ["1985", 1985.5, "1985.50", "v1", "1985.05", 2139]})
    assert_array_equal(block.exact("key").pairs(left, right), [[0, 1], [1, 0], [2, 4]])


@pytest.mark.parametrize("make", [lambda: block.exact("year"),
                                  lambda: block.within(block.ngrams("name", k=2), "year")])
def test_keyed_pass_without_any_shared_key_warns_instead_of_silently_proposing_nothing(make):
    left = pd.DataFrame({"name": ["Acme", "Bolt"], "year": [1985, 1990]})
    right = pd.DataFrame({"name": ["Acme", "Bolt"], "year": ["FY85", "FY90"]})
    with pytest.warns(UserWarning, match=r"proposed no pairs.*no value in common.*'1985'.*'fy85'"):
        result = block.candidates(left, right, on="name", blockers=[make()])
    assert result.empty
    with pytest.warns(UserWarning, match="right: no complete key"):
        block.candidates(left, right.assign(year=None), on="name", blockers=[make()])


def test_no_key_warning_for_empty_inputs_shared_keys_or_unkeyed_passes():
    left = pd.DataFrame({"name": ["Acme"], "year": [1985]})
    right = pd.DataFrame({"name": ["Zzzz"], "year": [1985.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The groups agree; it is the inner search that finds nothing.
        inner = block.candidates(left, right, on="name", blockers=[block.within(block.ngrams("name", k=1), "year")])
        unkeyed = block.candidates(left, right, on="name", blockers=[block.initials("name")])
        empty = block.candidates(left, right.iloc[:0], on="name", blockers=[block.exact("year")])
    assert inner.empty and unkeyed.empty and empty.empty


@pytest.mark.parametrize("right_state", ["CA", "NY"])
def test_groups_validate_inner_columns_even_without_matching_groups(right_state):
    left = pd.DataFrame({"state": ["NY"]})
    right = pd.DataFrame({"state": [right_state], "name": ["Acme"]})
    with pytest.raises(ValueError, match="left.*'name'"):
        block.within(block.ngrams("name"), "state").pairs(left, right)


@pytest.mark.parametrize("side", ["left", "right", "both", "disjoint"])
def test_empty_and_disjoint_groups(side):
    left = pd.DataFrame({"name": ["Acme"], "state": ["NY"]})
    right = left.copy()
    if side in ("left", "both"):
        left = left.iloc[:0]
    if side in ("right", "both"):
        right = right.iloc[:0]
    if side == "disjoint":
        right["state"] = "CA"
    result = block.within(block.ngrams("name"), "state").pairs(left, right)
    assert result.shape == (0, 2) and result.dtype == np.int64


def test_reverse_recovers_unselected_neighbors_with_mapped_fields_and_ids():
    left = pd.DataFrame({"firm": ["Acme", "Zulu"]}, index=["L9", "L1"])
    right = pd.DataFrame({"name": ["Acme", "Acme Inc", "Zulu"]}, index=["R8", "R3", "R2"])
    forward = block.ngrams(("firm", "name"), k=1)
    reverse = block.ngrams(("firm", "name"), k=1, reverse=True)
    assert_array_equal(forward.pairs(left, right), [[0, 0], [1, 2]])
    assert_array_equal(reverse.pairs(left, right), [[0, 0], [0, 1], [1, 2]])
    result = block.candidates(left, right, on=[("firm", "name")], blockers=[forward, reverse])
    assert _pairs(result) == {("L9", "R8"), ("L9", "R3"), ("L1", "R2")}
    assert result.set_index("right_id").loc["R3", "block"] == "ngrams-reverse:firm=name"
    assert result.attrs["blocking"]["passes"][1]["added_pairs"] == 1


@pytest.mark.parametrize("minimum", [0, 0.1, 1])
def test_reverse_equals_swapped_dense_search_and_is_chunk_invariant(monkeypatch, minimum):
    left = pd.DataFrame({"firm": ["Acme", "Acme", "Beta", None]}, index=[8, 5, 4, 1])
    right = pd.DataFrame({"name": ["Acme", "Beta", "Beet", None]}, index=[1, 4, 5, 8])
    # Forward swapped data is an independent reference for direction and mapped columns.
    expected = block.ngrams(("name", "firm"), k=2, min_sim=minimum).pairs(right, left)[:, ::-1]
    reverse = block.ngrams(("firm", "name"), k=2, min_sim=minimum, reverse=True)
    for chunk in (1, 3, 256):
        monkeypatch.setattr(block, "_CHUNK_ROWS", chunk)
        monkeypatch.setattr(block, "_PAIR_CHUNK_ROWS", chunk)
        assert_array_equal(reverse.pairs(left, right), expected)


def test_reverse_inside_group_never_multiplies_across_groups(monkeypatch):
    original = csr_matrix.__matmul__
    products = []

    def multiply(a, b):
        products.append((a.shape[0], b.shape[1]))
        assert a.shape[0] <= 3 and b.shape[1] <= 2
        return original(a, b)

    monkeypatch.setattr(csr_matrix, "__matmul__", multiply)
    left = pd.DataFrame({"name": ["same"] * 6, "state": ["NY", "CA", "FL"] * 2})
    right = pd.DataFrame({"name": ["same"] * 9, "state": ["NY", "CA", "FL"] * 3})
    grouped = block.within(block.ngrams("name", reverse=True, k=1), "state")
    pairs = grouped.pairs(left, right)
    assert len(pairs) == 9
    assert products == [(3, 2)] * 3
    assert (left.state.iloc[pairs[:, 0]].to_numpy() == right.state.iloc[pairs[:, 1]].to_numpy()).all()


@pytest.mark.parametrize("kind", ["exact", "within", "ngrams", "initials"])
def test_oversized_builtin_pass_stops_after_one_bounded_batch(monkeypatch, kind):
    frame = pd.DataFrame({"name": ["same"] * 20_000, "state": ["NY"] * 20_000})
    passes = {"exact": block.exact("name"),
              "within": block.within(block.exact("name"), "state"),
              "ngrams": block.ngrams("name", k=20_000, min_sim=0),
              "initials": block.initials("name")}
    if kind == "ngrams":
        frame["name"] = ""  # Zero vocabulary, still 400 million qualifying pairs.
    if kind == "initials":
        frame["name"] = "AB"
    right = frame.copy()
    if kind == "initials":
        right["name"] = "Alpha Beta"
    blocker = passes[kind]
    original = blocker.iter_pairs
    batches = []

    def batches_only(left, right):
        for batch in original(left, right):
            batches.append(len(batch))
            yield batch
            raise AssertionError("guard must stop before requesting another batch")

    def forbidden(*args, **kwargs):
        raise AssertionError("candidates must stream rather than call materializing pairs()")

    monkeypatch.setattr(block, "_PAIR_CHUNK_ROWS", 7)
    monkeypatch.setattr(blocker, "iter_pairs", batches_only)
    monkeypatch.setattr(blocker, "pairs", forbidden)
    with pytest.raises(ValueError, match=r"at least 4 pairs.*max_pairs=3.*within.*unions more pairs"):
        block.candidates(frame, right, on="name", blockers=[blocker], max_pairs=3)
    assert batches == [7]


def test_streamed_custom_duplicates_diagnostics_and_union_limit():
    class Duplicate(block.Blocker):
        name = "duplicates"

        def iter_pairs(self, left, right):
            yield np.array([[0, 0], [0, 0], [1, 1]])
            yield np.array([[1, 1]])

    left = pd.DataFrame({"name": ["Acme", "Beta"]})
    right = pd.DataFrame({"name": ["Acme", "Acme"]})
    result = block.candidates(left, right, on="name", max_pairs=3,
                              blockers=[Duplicate(), block.exact("name"), block.exact("name")])
    info = result.attrs["blocking"]
    assert info["schema_version"] == 1 and info["pair_count"] == 3
    counts = [(p["proposed_pairs"], p["unique_pairs"], p["added_pairs"], p["overlapping_pairs"],
               p["union_pairs"], p["exclusive_pairs"]) for p in info["passes"]]
    assert counts == [(4, 2, 2, 0, 2, 1), (2, 2, 1, 1, 3, 0), (2, 2, 0, 2, 3, 0)]
    assert info["passes"][0]["config"]["reconstructable"] is False
    assert json.loads(json.dumps(info)) == info


def test_builtin_configs_are_json_safe_and_preserve_nested_parameters():
    reverse = block.ngrams(("firm", "name"), k=np.int64(7), n=(np.int64(1), np.int64(3)),
                           min_sim=np.float64(0.25), reverse=np.bool_(True), name="backward")
    nested = block.within(block.within(reverse, "year", missing="match"), ("state", "region"))
    config = nested.to_config()
    assert config["columns"] == [["state", "region"]] and config["missing"] == "drop"
    assert config["blocker"]["missing"] == "match"
    assert config["blocker"]["blocker"] == {
        "type": "ngrams", "name": "backward", "columns": [["firm", "name"]],
        "k": 7, "n": [1, 3], "min_sim": 0.25, "reverse": True}
    assert json.loads(json.dumps(config)) == config
    assert block.exact("state").to_config() == {
        "type": "exact", "name": "exact:state", "columns": [["state", "state"]]}
    assert block.initials("name", min_len=3).to_config()["min_len"] == 3
    assert block.ngrams("name").to_config()["reverse"] is False
    assert block.ngrams("name", reverse=True).name == "ngrams-reverse:name"


@pytest.mark.parametrize("reverse", [1, "yes", None])
def test_invalid_reverse(reverse):
    with pytest.raises(ValueError, match="reverse.*boolean"):
        block.ngrams("name", reverse=reverse)


def test_invalid_within_parameters():
    with pytest.raises(ValueError, match="Blocker"):
        block.within("ngrams", "state")
    with pytest.raises(ValueError, match="missing"):
        block.within(block.ngrams("name"), "state", missing="fallback")
    with pytest.raises(ValueError, match="at least one field"):
        block.within(block.ngrams("name"))
    with pytest.raises(ValueError, match="name"):
        block.within(block.ngrams("name"), "state", name="")


def test_grouped_candidates_are_repeatable_and_full_table_similarities_are_unchanged():
    frame = pd.DataFrame({"name": ["Acme", "Beta", "Acme Inc"], "state": ["NY", "CA", "NY"]})
    passes = [block.within(block.ngrams("name", k=1), "state"), block.ngrams("name", reverse=True)]
    first = block.candidates(frame, frame, on=["name", "state"], blockers=passes)
    second = block.candidates(frame, frame, on=["name", "state"], blockers=passes)
    assert_frame_equal(first, second)
    assert first.attrs == second.attrs
    ungrouped = block.candidates(frame, frame, on=["name", "state"], blockers=[block.ngrams("name")])
    both = first.merge(ungrouped, on=["left_id", "right_id"])
    assert_array_equal(both.sim_x, both.sim_y)

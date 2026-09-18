"""Round-trip checks for identifiers, value types and actionable file errors."""

import builtins
import warnings

import numpy as np
import pandas as pd
import pytest

from jlink.io import read_table, write_table


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".CSV"])
def test_delimited_ids(suffix, tmp_path):
    frame = pd.DataFrame({"id": ["00123", "NA", "NULL", "nan", "1e3", ""],
                          "name": ['Smith, "Jones"', "Café", "a\tb", "one\ntwo", "Acme", None]})
    path = tmp_path / ("firm data" + suffix)
    write_table(frame, path)
    result = read_table(path)
    assert result.id.iloc[:5].tolist() == frame.id.iloc[:5].tolist()
    assert pd.isna(result.id.iloc[5])
    assert result.name.iloc[:5].tolist() == frame.name.iloc[:5].tolist()
    assert list(result.columns) == ["id", "name"]


@pytest.mark.parametrize("suffix", [".dta", ".parquet"])
def test_native_types(suffix, tmp_path):
    frame = pd.DataFrame({"id": ["00123", "00004"], "n": np.array([1, 2], dtype="int32"),
                          "p": [0.8, float("nan")], "name": ["Société", "東京"]})
    path = tmp_path / ("firm data" + suffix)
    write_table(frame, path)
    result = read_table(path)
    pd.testing.assert_frame_equal(result, frame)


def test_stata_value_labels_stay_numeric(tmp_path):
    path = tmp_path / "labeled.dta"
    frame = pd.DataFrame({"id": np.array([1, 2], dtype="int16")})
    frame.to_stata(path, write_index=False, value_labels={"id": {1: "First", 2: "Second"}})
    pd.testing.assert_frame_equal(read_table(path), frame)


def test_stata_repairs_names_once_without_mutation(tmp_path):
    names = ["firm name", "a" * 40, "a" * 39 + "b", "1id", "if", "", "firm_name"]
    frame = pd.DataFrame([["00123"] * len(names)], columns=names)
    original = frame.copy(deep=True)
    path = tmp_path / "links.dta"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_table(frame, path)
    assert len(caught) == 1
    assert "Stata column names changed" in str(caught[0].message)
    result = read_table(path)
    assert result.columns.is_unique
    assert all(len(name) <= 32 for name in result.columns)
    assert "firm name" not in result and "if" not in result
    assert result.iloc[0].tolist() == ["00123"] * len(names)
    pd.testing.assert_frame_equal(frame, original)


def test_stata_empty_nullable_and_dates(tmp_path):
    frame = pd.DataFrame({"id": pd.Series(["00123", pd.NA], dtype="string"),
                          "count": pd.Series([1, pd.NA], dtype="Int64"),
                          "error": [None, None], "when": pd.to_datetime(["2020-01-01", None])})
    path = tmp_path / "nullable.dta"
    write_table(frame, path)
    result = read_table(path)
    assert result.id.tolist() == ["00123", ""]
    assert result["count"].iloc[0] == 1 and pd.isna(result["count"].iloc[1])
    assert result.error.tolist() == ["", ""]
    pd.testing.assert_series_equal(result.when, frame.when)


@pytest.mark.parametrize("operation", [read_table, lambda path: write_table(pd.DataFrame({"id": [1]}), path)])
def test_missing_pyarrow_is_actionable(operation, monkeypatch, tmp_path):
    original = builtins.__import__

    def without_arrow(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("hidden for this test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_arrow)
    with pytest.raises(ValueError, match="table.parquet.*Parquet needs pyarrow"):
        operation(tmp_path / "table.parquet")


@pytest.mark.parametrize("suffix", [".xlsx", ".txt", ""])
def test_unsupported_format(suffix, tmp_path):
    with pytest.raises(ValueError, match="expected a .csv"):
        read_table(tmp_path / ("bad" + suffix))
    with pytest.raises(ValueError, match="expected a .csv"):
        write_table(pd.DataFrame(), tmp_path / ("bad" + suffix))


def test_bad_paths_and_bad_stata_columns(tmp_path):
    with pytest.raises(ValueError, match="cannot read.*missing.csv"):
        read_table(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="cannot write.*missing/links.csv"):
        write_table(pd.DataFrame({"id": [1]}), tmp_path / "missing" / "links.csv")
    with pytest.raises(ValueError, match="unique"):
        write_table(pd.DataFrame([[1, 2]], columns=["id", "id"]), tmp_path / "links.dta")
    with pytest.raises(ValueError, match="DataFrame"):
        write_table([1, 2], tmp_path / "links.csv")


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".dta", ".parquet"])
def test_empty_links_keep_schema(suffix, tmp_path):
    frame = pd.DataFrame({"left_id": pd.Series(dtype=str), "right_id": pd.Series(dtype=str),
                          "p": pd.Series(dtype=float)})
    path = tmp_path / ("empty" + suffix)
    write_table(frame, path)
    result = read_table(path)
    assert result.empty and list(result.columns) == list(frame.columns)


@pytest.mark.parametrize("dtype", ["int64", "Int64", "uint64"])
def test_stata_refuses_to_round_large_ids(tmp_path, dtype):
    frame = pd.DataFrame({"id": pd.Series([2**53 + 1], dtype=dtype)})
    with pytest.raises(ValueError, match="column 'id'.*lossless range"):
        write_table(frame, tmp_path / "large.dta")

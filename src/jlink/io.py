"""Read and write research tables without guessing that text IDs are numbers."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from pandas.errors import InvalidColumnName

FORMATS = (".csv", ".tsv", ".dta", ".parquet")


def _path(path: str | Path) -> Path:
    try:
        path = Path(path).expanduser()
    except (TypeError, ValueError):
        raise ValueError(f"{path!r}: expected a table file path") from None
    if path.suffix.lower() not in FORMATS:
        raise ValueError(f"{str(path)!r}: expected a .csv, .tsv, .dta or .parquet file")
    return path


def _require_arrow(path: Path) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ValueError(f"{str(path)!r}: Parquet needs pyarrow; "
                         "install it with 'uv add pyarrow' (or 'uv pip install pyarrow')") from None


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a table; delimited fields stay strings, including IDs such as '00123'."""
    path = _path(path)
    try:
        suffix = path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            # Only empty cells are missing: 'NA', 'NULL' and 'nan' can be real IDs.
            return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=str,
                               keep_default_na=False, na_values=[""])
        if suffix == ".dta":
            # Value labels must not turn numeric Stata IDs into categorical text.
            return pd.read_stata(path, convert_categoricals=False, preserve_dtypes=True)
        _require_arrow(path)
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        raise ValueError(f"cannot read {str(path)!r}: {exc}") from None


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    """Write without an index; warn once if Stata needs different column names."""
    path = _path(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{str(path)!r}: expected a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError(f"{str(path)!r}: column names must be unique")
    try:
        suffix = path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            frame.to_csv(path, sep="\t" if suffix == ".tsv" else ",", index=False)
        elif suffix == ".dta":
            _write_stata(frame, path)
        else:
            _require_arrow(path)
            frame.to_parquet(path, engine="pyarrow", index=False)
    except Exception as exc:
        raise ValueError(f"cannot write {str(path)!r}: {exc}") from None


def _write_stata(frame: pd.DataFrame, path: Path) -> None:
    data = frame.copy()
    # pandas handles Stata's reserved names, length limit and collisions. Give empty
    # names a nonempty starting point, and avoid mutating the caller's DataFrame.
    columns = []
    for i, name in enumerate(data.columns):
        name = str(name)
        if not name:
            name = f"column_{i + 1}"
            while name in columns or name in data.columns:
                name = "_" + name
        columns.append(name)
    if len(set(columns)) != len(columns):
        raise ValueError("column names must remain unique when converted to text")
    initial_renames = [(old, new) for old, new in zip(data.columns, columns) if old != new]
    data.columns = columns
    for column in data:
        # Stata has no nullable extension dtypes. Retain numeric values as numbers
        # and text as text; all-missing object columns must be writable too.
        series = data[column]
        if pd.api.types.is_integer_dtype(series.dtype) and (
            (series.dropna() > 2**53).any() or (series.dropna() < -(2**53)).any()
        ):
            raise ValueError(f"column {column!r} has integers outside Stata's lossless range; "
                             "store these IDs as strings or write a Parquet file")
        if isinstance(series.dtype, pd.StringDtype):
            data[column] = series.astype(object).where(series.notna(), "")
        elif pd.api.types.is_extension_array_dtype(series.dtype) and not isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            if pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
                data[column] = series.astype("float64" if series.isna().any() else "int64")
            elif pd.api.types.is_float_dtype(series.dtype):
                data[column] = series.astype("float64")
        elif series.dtype == object and series.isna().all():
            data[column] = series.fillna("")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InvalidColumnName)
        data.to_stata(path, write_index=False, version=118)
    messages = [f"{old!r} -> {new!r}" for old, new in initial_renames]
    for warning in caught:
        if issubclass(warning.category, InvalidColumnName):
            messages.append(" ".join(str(warning.message).split()))
        else:
            warnings.warn(str(warning.message), warning.category, stacklevel=3)
    if messages:
        warnings.warn(f"{str(path)!r}: Stata column names changed: {'; '.join(messages)}",
                      UserWarning, stacklevel=3)

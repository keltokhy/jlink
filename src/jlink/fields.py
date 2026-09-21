"""How records are described: the `on` argument, and the normalized text used for similarity."""

from __future__ import annotations

import math
import re
import unicodedata

import numpy as np
import pandas as pd

On = list  # items are "column" or ("left_column", "right_column")
_PUNCT = re.compile(r"[^\w\s]|_")
_SPACE = re.compile(r"\s+")
_WHOLE_DECIMAL = re.compile(r"[+-]?\d+\.0+")


def parse_on(on, *, unpaired: bool = False) -> list[tuple[str, str | None, str | None]]:
    """[(label, left_column, right_column), ...]. The label is the left column name.

    With ``unpaired=True`` an item may also be ``(left, None)`` or ``(None, right)``: a field that
    only one side has. It is shown to the judge under its own column name; the absent side is None.
    Steps that compare a left column with a right column keep the default and reject such items.
    """
    if isinstance(on, str):
        on = [on]
    if not on:
        raise ValueError("`on` must name at least one field to compare")
    out = []
    for item in on:
        pair = tuple(item) if isinstance(item, (tuple, list)) and len(item) == 2 else ()
        if isinstance(item, str):
            out.append((item, item, item))
        elif pair and all(isinstance(c, str) for c in pair):
            out.append((pair[0], pair[0], pair[1]))
        elif pair and sum(c is None for c in pair) == 1 and any(isinstance(c, str) for c in pair):
            if not unpaired:
                raise ValueError(f"the field {pair!r} exists on one side only, but this step compares a left "
                                 "column with a right column; one-sided fields are only shown to the judge")
            out.append((pair[0] or pair[1], pair[0], pair[1]))
        else:
            raise ValueError("each `on` item is a column name, a (left, right) pair of names, or a one-sided "
                             f"(left, None) or (None, right); got {item!r}")
    if any(lc is None or rc is None for _, lc, rc in out):
        for side, labels in (("left", [f[0] for f in out if f[1] is not None]),
                             ("right", [f[0] for f in out if f[2] is not None])):
            if not labels:
                raise ValueError(f"`on` gives the {side} records no field; "
                                 "the judge needs at least one per side")
            repeated = [label for label in labels if labels.count(label) > 1]
            if repeated:
                raise ValueError(f"`on` would show the {side} record two fields labeled {repeated[0]!r}; "
                                 "paired fields are labeled by their left column, so rename a column")
    return out


def side_fields(fields: list, side: str) -> list[tuple[str, str]]:
    """[(label, column), ...] for the fields that the left or the right records have."""
    index = 1 if side == "left" else 2
    return [(f[0], f[index]) for f in fields if f[index] is not None]


def check_columns(frame: pd.DataFrame, columns: list[str], side: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"the {side} data has no column {missing[0]!r}; its columns are {list(frame.columns)}")


def normalize(text: object) -> str:
    """Casefold, strip accents, turn & into 'and', drop punctuation, collapse whitespace."""
    if text is None or (isinstance(text, float) and text != text) or text is pd.NA:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold().replace("&", " and ")
    return _SPACE.sub(" ", _PUNCT.sub(" ", s)).strip()


def clean(value):
    """A JSON-ready value, or None if missing. Whole floats become ints: a year read as 1985.0 is 1985."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value) if value.is_integer() and abs(value) < 1e15 else value
    if isinstance(value, (bool, int)):
        return value
    text = str(value).strip()
    return text or None


def key_text(value: object) -> str:
    """Normalized text of one exact or group key component; empty if missing.

    Whole numbers agree however they are stored. A numeric column with one missing value is
    float in pandas, and a table written from it says "1985.0", so the integer 1985, the float
    1985.0 and the texts "1985" and "1985.0" all give "1985".
    """
    value = clean(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # clean() leaves whole floats from 1e15 up as floats; sixteen-digit keys exist
    elif isinstance(value, str) and _WHOLE_DECIMAL.fullmatch(value):
        value = value[:value.index(".")]
    return normalize(value)


def record_text(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """One normalized string per row: the listed columns joined by a space."""
    # astype(object) keeps the .str accessor working when a frame is empty or a column is all-missing floats
    parts = [frame[c].astype(object).map(normalize).astype(object) for c in columns]
    text = parts[0]
    for part in parts[1:]:
        text = text.str.cat(part, sep=" ")
    return text.str.strip()


def ids(frame: pd.DataFrame, id_column: str | None, side: str) -> pd.Index:
    """The frame's IDs, checked for uniqueness."""
    if id_column is None:
        values = frame.index
    else:
        check_columns(frame, [id_column], side)
        values = pd.Index(frame[id_column])
    if values.has_duplicates:
        where = f"column {id_column!r}" if id_column else "index"
        raise ValueError(f"the {side} data's {where} has duplicate IDs; IDs must be unique")
    return values

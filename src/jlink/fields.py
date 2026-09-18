"""How records are described: the `on` argument, and the normalized text used for similarity."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

On = list  # items are "column" or ("left_column", "right_column")
_PUNCT = re.compile(r"[^\w\s]|_")
_SPACE = re.compile(r"\s+")


def parse_on(on) -> list[tuple[str, str, str]]:
    """[(label, left_column, right_column), ...]. The label is the left column name."""
    if isinstance(on, str):
        on = [on]
    if not on:
        raise ValueError("`on` must name at least one field to compare")
    out = []
    for item in on:
        if isinstance(item, str):
            out.append((item, item, item))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and all(isinstance(c, str) for c in item):
            out.append((item[0], item[0], item[1]))
        else:
            raise ValueError(f"each `on` item is a column name or a (left, right) pair of names; got {item!r}")
    return out


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

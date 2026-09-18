"""The benchmark layout and checks shared by preparation and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jlink.fields import check_columns, ids, parse_on

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NAMES = ("febrl4", "dblp-acm", "abt-buy", "amazon-google", "nber-firms")
PAIR_COLUMNS = ["left_id", "right_id"]


def pair_set(frame: pd.DataFrame, label: str = "pairs") -> set[tuple]:
    """Read unique pairs, rejecting missing IDs rather than silently losing labels."""
    check_columns(frame, PAIR_COLUMNS, label)
    for col in PAIR_COLUMNS:
        if frame[col].isna().any() or frame[col].eq("").any():
            raise ValueError(f"the {label} column {col!r} contains missing IDs")
    return set(frame[PAIR_COLUMNS].itertuples(index=False, name=None))


def validate_dataset(left: pd.DataFrame, right: pd.DataFrame, truth: pd.DataFrame,
                     meta: dict) -> dict:
    """Check the contract, report repeated truth rows and non-unique match endpoints."""
    for key in ("entity", "definition", "on", "how"):
        if not meta.get(key):
            raise ValueError(f"meta.json must supply a nonempty {key!r}")
    if meta["how"] not in {"one-to-one", "many-to-one", "one-to-many", "many-to-many"}:
        raise ValueError("meta.json 'how' must be a supported linkage cardinality")
    fields = parse_on(meta["on"])
    for side, frame, position in (("left", left, 1), ("right", right, 2)):
        check_columns(frame, [f[position] for f in fields], side)
        values = ids(frame, "id", side)
        if values.isna().any() or values.isin([""]).any():
            raise ValueError(f"the {side} column 'id' contains missing IDs")
    pairs = pair_set(truth, "truth")
    for column, frame in (("left_id", left), ("right_id", right)):
        unknown = truth.loc[~truth[column].isin(frame["id"]), column]
        if len(unknown):
            raise ValueError(f"truth column {column!r} has {len(unknown)} IDs absent from "
                             f"the {column.removesuffix('_id')} data, e.g. {unknown.iloc[0]!r}")
    unique = truth.drop_duplicates(PAIR_COLUMNS)
    return {
        "left_duplicate_ids": 0, "right_duplicate_ids": 0,
        "truth_duplicate_pairs": len(truth) - len(pairs),
        "truth_left_ids_with_multiple_matches": int(unique.groupby("left_id").size().gt(1).sum()),
        "truth_right_ids_with_multiple_matches": int(unique.groupby("right_id").size().gt(1).sum()),
        "left_duplicate_records_excluding_id": int(left.drop(columns="id").duplicated().sum()),
        "right_duplicate_records_excluding_id": int(right.drop(columns="id").duplicated().sum()),
        "truth_ids_verified": True,
    }


def write_dataset(directory: Path, left: pd.DataFrame, right: pd.DataFrame,
                  truth: pd.DataFrame, meta: dict) -> dict:
    """Validate before writing; retain duplicate diagnostics while storing unique truth."""
    meta = dict(meta, validation=validate_dataset(left, right, truth, meta))
    truth = truth[PAIR_COLUMNS].drop_duplicates().reset_index(drop=True)
    meta["row_counts"] = {"left": len(left), "right": len(right), "truth": len(truth)}
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in (("left", left), ("right", right), ("truth", truth)):
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


def load_dataset(name: str, data_dir: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load a prepared dataset with a useful instruction when preparation is missing."""
    if name not in NAMES:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(NAMES)}")
    directory = Path(data_dir) / name
    try:
        meta = json.loads((directory / "meta.json").read_text())
        frames = [pd.read_parquet(directory / f"{part}.parquet") for part in ("left", "right", "truth")]
    except FileNotFoundError as exc:
        raise ValueError(f"dataset {name!r} is not prepared; "
                         f"run uv run python bench/prepare.py {name}") from exc
    validate_dataset(*frames, meta)
    return *frames, meta


def sample_left(left: pd.DataFrame, truth: pd.DataFrame, n: int | None,
                seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample only left records; keep the entire right search universe and matching truth."""
    if n is None:
        return left, truth
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("--sample must be a positive number of left records")
    # Sample positions and restore source order even when the frame has an arbitrary index.
    positions = pd.Series(range(len(left))).sample(n=min(n, len(left)), random_state=seed).sort_values()
    sampled = left.iloc[positions.to_numpy()].copy()
    return sampled, truth.loc[truth.left_id.isin(sampled.id)].copy()

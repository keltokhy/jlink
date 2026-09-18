"""Samples for hand labeling and weighted evidence about linkage quality."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fields import check_columns, ids, parse_on
from .resolve import _columns, _number, _numbers, _pairs, _probabilities


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be a whole number at least {minimum}")
    return int(value)


def _allocation(sizes: np.ndarray, n: int) -> np.ndarray:
    """Equal allocation, repeatedly redistributing places left by exhausted bins."""
    counts = np.zeros(len(sizes), dtype=np.int64)
    remaining = min(n, int(sizes.sum()))
    while remaining:
        available = np.flatnonzero(counts < sizes)
        share, remainder = divmod(remaining, len(available))
        extra = np.full(len(available), share, dtype=np.int64)
        extra[:remainder] += 1
        extra = np.minimum(extra, sizes[available] - counts[available])
        counts[available] += extra
        remaining -= int(extra.sum())
    return counts


def _fields(sample: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, on,
            left_id: str | None, right_id: str | None) -> list[str]:
    fields = parse_on(on)
    labels = [label for label, _, _ in fields]
    if len(set(labels)) != len(labels):
        raise ValueError("on must give each field a different left column name")
    positions = []
    for frame, column, side in ((left, left_id, "left"), (right, right_id, "right")):
        _columns(frame, [], f"the {side} data")
        values = ids(frame, column, side)
        if values.isna().any():
            raise ValueError(f"the {side} data has missing IDs; supply an ID for every record")
        indexer = values.get_indexer(sample[f"{side}_id"])
        if (indexer < 0).any():
            raise ValueError(f"sample column '{side}_id' contains IDs absent from the {side} data")
        positions.append(indexer)
    columns = []
    for label, a, b in fields:
        check_columns(left, [a], "left")
        check_columns(right, [b], "right")
        sample[f"a_{label}"] = left[a].iloc[positions[0]].reset_index(drop=True)
        sample[f"b_{label}"] = right[b].iloc[positions[1]].reset_index(drop=True)
        columns.extend([f"a_{label}", f"b_{label}"])
    return columns


def audit_sample(scores: pd.DataFrame, *, n: int = 200,
                 bins: tuple[float, ...] = (0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0), seed: int = 0,
                 left: pd.DataFrame | None = None, right: pd.DataFrame | None = None, on=None,
                 left_id: str | None = None, right_id: str | None = None) -> pd.DataFrame:
    """Draw an equally allocated stratified sample, with inverse inclusion weights.

    Nonempty bin categories are retained even when n is too small to sample every bin.
    Saving to CSV loses unused categories; label every sampled bin before evaluation.
    """
    n = _integer(n, "n", 0)
    seed = _integer(seed, "seed", 0)
    _pairs(scores, "scores")
    p = _probabilities(scores)
    try:
        edges = np.asarray(bins, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("bins must be increasing numeric boundaries from 0 to 1") from exc
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] != 0 or edges[-1] != 1 or (np.diff(edges) <= 0).any()):
        raise ValueError("bins must be strictly increasing boundaries starting at 0 and ending at 1")
    supplied = (left is not None, right is not None, on is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("provide left, right and on together to include fields for labeling")
    frame = scores.loc[p.notna(), ["left_id", "right_id"]].copy().reset_index(drop=True)
    frame["p"] = p.loc[p.notna()].to_numpy()
    labels = [f"{'[' if i == 0 else '('}{low:g}, {high:g}]"
              for i, (low, high) in enumerate(zip(edges[:-1], edges[1:]))]
    if len(set(labels)) != len(labels):
        # Preserve distinct labels even for boundaries that differ past six significant digits.
        labels = [f"{'[' if i == 0 else '('}{low!r}, {high!r}]"
                  for i, (low, high) in enumerate(zip(edges[:-1].tolist(), edges[1:].tolist()))]
    frame["bin"] = pd.cut(frame["p"], edges, labels=labels, right=True, include_lowest=True)
    frame["bin"] = frame["bin"].cat.remove_unused_categories()
    groups = frame.groupby("bin", observed=True, sort=True).indices
    counts = _allocation(np.asarray([len(group) for group in groups.values()]), n)
    rng = np.random.default_rng(seed)
    samples = []
    for group, count in zip(groups.values(), counts):
        if count:
            part = frame.iloc[rng.choice(group, size=count, replace=False)].copy()
            part["weight"] = len(group) / int(count)
            samples.append(part)
    if samples:
        sample = pd.concat(samples, ignore_index=True)
        sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    else:
        sample = frame.iloc[:0].copy()
        sample["weight"] = pd.Series(dtype=float)
    sample["is_match"] = pd.Series(pd.NA, index=sample.index, dtype="object")
    fields = _fields(sample, left, right, on, left_id, right_id) if all(supplied) else []
    return sample[["is_match", *fields, "p", "bin", "weight", "left_id", "right_id"]]


def _labels(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.casefold()
    mapping = {"1": 1.0, "1.0": 1.0, "true": 1.0, "y": 1.0, "yes": 1.0,
               "0": 0.0, "0.0": 0.0, "false": 0.0, "n": 0.0, "no": 0.0}
    blank = text.isna() | text.eq("").fillna(False)
    invalid = ~blank & ~text.isin(mapping)
    if invalid.any():
        bad = values.loc[invalid].iloc[0]
        raise ValueError(f"column 'is_match' has {bad!r}; use 1/0, True/False, y/n, yes/no, or blank")
    return text.map(mapping).astype(float)


def _metrics(totals: np.ndarray) -> np.ndarray:
    tp, linked, true = np.moveaxis(totals, -1, 0)
    numerator = np.stack([tp, tp, 2 * tp], axis=-1)
    denominator = np.stack([linked, true, linked + true], axis=-1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def _bootstrap(contributions: np.ndarray, groups: dict, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    totals = np.zeros((n_boot, 3))
    for group in groups.values():
        size = len(group)
        if not size:
            continue
        # Bound the temporary draws for large audits, while vectorizing over replicates.
        batch = max(1, 250_000 // size)
        values = contributions[group]
        for start in range(0, n_boot, batch):
            stop = min(n_boot, start + batch)
            draws = rng.integers(size, size=(stop - start, size))
            totals[start:stop] += values[draws].sum(axis=1)
    return _metrics(totals)


def _format(value: float) -> str:
    return "NaN" if np.isnan(value) else f"{value:.4f}"


@dataclass
class Evaluation:
    """Weighted audit estimates and percentile intervals from a stratified bootstrap."""

    precision: tuple[float, float, float]
    recall: tuple[float, float, float]
    f1: tuple[float, float, float]
    brier: float
    calibration: pd.DataFrame
    n_labeled: int
    n_unlabeled: int
    threshold: float
    _notes: tuple[str, ...] = field(default=(), repr=False)

    def summary(self) -> str:
        """Describe the estimand and explain unavailable estimates."""
        parts = [f"{self.n_labeled:,} labeled pairs; {self.n_unlabeled:,} blank labels dropped; "
                 f"threshold {self.threshold:g}."]
        for name in ("precision", "recall", "f1"):
            estimate, low, high = getattr(self, name)
            label = "F1" if name == "f1" else name.capitalize()
            parts.append(f"{label} {_format(estimate)} (95% CI {_format(low)} to {_format(high)}).")
        parts.append(f"Weighted Brier score {_format(self.brier)}.")
        parts.append("Estimates use sampling weights; 95% intervals use a bootstrap within bins.")
        parts.append("Recall is among candidate pairs and cannot see true matches lost in blocking.")
        parts.extend(self._notes)
        return " ".join(parts)

    def to_markdown(self) -> str:
        """Return appendix tables without requiring pandas' optional tabulate dependency."""
        lines = ["| Metric | Estimate | 95% interval |", "|:--|--:|:--|"]
        for name, label in (("precision", "Precision"), ("recall", "Candidate-pair recall"), ("f1", "F1")):
            estimate, low, high = getattr(self, name)
            lines.append(f"| {label} | {_format(estimate)} | {_format(low)} to {_format(high)} |")
        lines.append(f"| Weighted Brier score | {_format(self.brier)} | — |")
        lines.extend(["", "| Probability bin | Labeled pairs | Weighted mean p | Weighted match rate |",
                      "|:--|--:|--:|--:|"])
        for row in self.calibration.itertuples(index=False):
            label = str(row.bin).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            lines.append(f"| {label} | {row.n:,} | {_format(row.mean_p)} | {_format(row.match_rate)} |")
        lines.extend(["", self.summary()])
        return "\n".join(lines)


def evaluate(labeled: pd.DataFrame, *, threshold: float = 0.5,
             n_boot: int = 2000, seed: int = 0) -> Evaluation:
    """Estimate linkage accuracy using inverse inclusion weights and within-bin resampling.

    Entirely unlabeled strata make population-wide estimates unavailable. Partial labeling
    uses the supplied weights for the available judgments; it assumes missing labels do not
    introduce selection bias. F1 is 2 TP / (predicted links + true matches).
    """
    threshold = _number(threshold, "threshold", probability=True)
    n_boot = _integer(n_boot, "n_boot", 1)
    seed = _integer(seed, "seed", 0)
    _columns(labeled, ["p", "bin", "weight", "is_match"], "labeled data")
    frame = labeled.reset_index(drop=True).copy()
    y = _labels(frame["is_match"])
    keep = y.notna()
    n_labeled, n_unlabeled = int(keep.sum()), int((~keep).sum())
    if frame["bin"].isna().any():
        raise ValueError("column 'bin' must identify a sampling stratum for every pair")
    strata = (list(frame["bin"].cat.categories) if isinstance(frame["bin"].dtype, pd.CategoricalDtype)
              else list(pd.unique(frame["bin"])))
    judged = frame.loc[keep].reset_index(drop=True)
    p = _probabilities(judged, missing=False).to_numpy()
    weights = _numbers(judged, "weight").to_numpy()
    if (weights <= 0).any():
        raise ValueError("column 'weight' must contain positive sampling weights")
    if n_labeled:
        # Ratios are invariant to a common scale; avoid overflow with large population weights.
        weights = weights / weights.max()
    truth = y.loc[keep].to_numpy()
    predicted = p >= threshold
    contributions = weights[:, None] * np.column_stack([predicted * truth, predicted, truth])
    totals = contributions.sum(axis=0)
    groups = judged.groupby("bin", observed=True, sort=False).indices
    calibration = []
    empty = []
    for stratum in strata:
        group = groups.get(stratum, np.array([], dtype=int))
        if len(group):
            mean_p = float(np.average(p[group], weights=weights[group]))
            rate = float(np.average(truth[group], weights=weights[group]))
        else:
            mean_p = rate = float("nan")
            empty.append(stratum)
        calibration.append((stratum, len(group), mean_p, rate))
    table = pd.DataFrame(calibration, columns=["bin", "n", "mean_p", "match_rate"])
    notes = []
    estimates = _metrics(totals)
    intervals = np.full((3, 2), np.nan)
    brier = float(np.average((p - truth) ** 2, weights=weights)) if n_labeled else float("nan")
    if not n_labeled:
        notes.append("No labeled pairs: all estimates and intervals are NaN.")
    if empty:
        names = ", ".join(str(value) for value in empty)
        notes.append(f"No labels in bin(s) {names}: population-wide estimates and intervals are NaN.")
    if not n_labeled or empty:
        estimates[:] = np.nan
        brier = float("nan")
    else:
        if totals[1] == 0:
            notes.append("No predicted links at this threshold: precision and its interval are NaN.")
        if totals[2] == 0:
            notes.append("No true matches among labeled pairs: recall and its interval are NaN.")
        if totals[1] + totals[2] == 0:
            notes.append("No predicted links or true matches: F1 and its interval are NaN.")
        boot = _bootstrap(contributions, groups, n_boot, seed)
        for index, name in enumerate(("precision", "recall", "F1")):
            finite = np.isfinite(boot[:, index])
            if np.isfinite(estimates[index]) and finite.any():
                intervals[index] = np.quantile(boot[finite, index], [0.025, 0.975])
                if not finite.all():
                    notes.append(f"{n_boot - int(finite.sum()):,} bootstrap replicates with undefined {name} "
                                 "were excluded from that interval.")
        if any(len(group) == 1 for group in groups.values()):
            notes.append("A bin has only one label; its within-bin uncertainty cannot be estimated "
                         "by resampling. Label more pairs for reliable intervals.")
    if n_unlabeled and n_labeled:
        notes.append("Available labels retain their sampling weights; "
                     "selective missing labels can bias estimates.")
    metrics = [tuple(float(value) for value in (estimate, *interval))
               for estimate, interval in zip(estimates, intervals)]
    return Evaluation(*metrics, brier, table, n_labeled, n_unlabeled, threshold, tuple(notes))


def score_against_truth(links: pd.DataFrame, truth: pd.DataFrame,
                        candidates: pd.DataFrame | None = None) -> dict:
    """Compare unique linked pairs with known truth; optionally measure blocking recall."""
    _pairs(links, "links")
    _pairs(truth, "truth")
    index = pd.MultiIndex.from_frame(truth[["left_id", "right_id"]])
    linked = pd.MultiIndex.from_frame(links[["left_id", "right_id"]])
    tp = int(linked.isin(index).sum())
    fp, fn = len(links) - tp, len(truth) - tp
    precision, recall, f1 = _metrics(np.array([tp, len(links), len(truth)], dtype=float))
    result = {"precision": float(precision), "recall": float(recall), "f1": float(f1),
              "tp": tp, "fp": fp, "fn": fn}
    if candidates is not None:
        _pairs(candidates, "candidates")
        proposed = pd.MultiIndex.from_frame(candidates[["left_id", "right_id"]])
        result["pairs_completeness"] = float(index.isin(proposed).mean()) if len(index) else float("nan")
    return result

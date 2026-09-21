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
                 left_id: str | None = None, right_id: str | None = None,
                 links: pd.DataFrame | None = None) -> pd.DataFrame:
    """Draw an equally allocated stratified sample, with inverse inclusion weights.

    Nonempty bin categories are retained even when n is too small to sample every bin.
    Saving to CSV loses unused categories; sample and label every populated bin before export.
    Supplying the actual final links adds boolean selected membership for mode="selected"
    evaluation. Scores alone do not determine selection; no selected column is inferred.
    Pairs without a probability (unjudged or failed) are outside the sampling population.
    """
    n = _integer(n, "n", 0)
    seed = _integer(seed, "seed", 0)
    _pairs(scores, "scores")
    p = _probabilities(scores)
    selected = None
    if links is not None:
        _pairs(links, "links")
        candidates = pd.MultiIndex.from_frame(scores[["left_id", "right_id"]])
        chosen = pd.MultiIndex.from_frame(links[["left_id", "right_id"]])
        if not chosen.isin(candidates).all():
            raise ValueError("links contains pairs absent from scores; "
                             "use matching left_id/right_id values and types")
        selected = candidates.isin(chosen)
        if (selected & p.isna().to_numpy()).any():
            raise ValueError("links contains pairs without a judged 'p' in scores; supply scored final links")
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
    if selected is not None:
        frame["selected"] = selected[p.notna().to_numpy()]
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
    selection = ["selected"] if selected is not None else []
    return sample[["is_match", *fields, "p", *selection, "bin", "weight", "left_id", "right_id"]]


def _selected(values: pd.Series) -> pd.Series:
    """Read explicit membership, including CSV booleans; never use string truthiness."""
    text = values.astype("string").str.strip().str.casefold()
    mapping = {"1": True, "1.0": True, "true": True, "0": False, "0.0": False, "false": False}
    invalid = ~text.isin(mapping)
    if invalid.any():
        bad = values.loc[invalid].iloc[0]
        raise ValueError(f"column 'selected' has {bad!r}; use nonmissing 1/0 or True/False for every pair")
    return text.map(mapping).astype(bool)


def _blank(values: pd.Series) -> pd.Series:
    """Rows whose human label is missing or only whitespace."""
    text = values.astype("string").str.strip()
    return text.isna() | text.eq("").fillna(False)


def _labels(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.casefold()
    mapping = {"1": 1.0, "1.0": 1.0, "true": 1.0, "y": 1.0, "yes": 1.0,
               "0": 0.0, "0.0": 0.0, "false": 0.0, "n": 0.0, "no": 0.0}
    blank = _blank(values)
    invalid = ~blank & ~text.isin(mapping)
    if invalid.any():
        bad = values.loc[invalid].iloc[0]
        raise ValueError(f"column 'is_match' has {bad!r}; use 1/0, True/False, y/n, yes/no, or blank")
    return text.map(mapping).astype(float)


_RESTORE = "restore the 'weight' column from the file that audit_sample wrote"


def _sampling_weights(frame: pd.DataFrame, keep: pd.Series) -> np.ndarray:
    """Positive weights for every sampled row; a bin's blank rows carry part of its weight."""
    try:
        weights = _numbers(frame, "weight").to_numpy()
    except ValueError:
        _numbers(frame.loc[keep], "weight")  # a fault among the labeled rows keeps its usual message
        blank = pd.to_numeric(frame.loc[~keep, "weight"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        raise ValueError(f"column 'weight' has no usable number in {int((~np.isfinite(blank)).sum()):,} of the "
                         f"{len(blank):,} rows with a blank label. Every sampled row needs its sampling weight, "
                         "because the blank rows show how much of their bin the labeled pairs stand for; "
                         f"{_RESTORE}") from None
    if (weights <= 0).any():
        raise ValueError(f"column 'weight' must contain positive sampling weights, but {int((weights <= 0).sum()):,} "
                         f"rows have zero or less; {_RESTORE}")
    return weights


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
    threshold: float | None
    _notes: tuple[str, ...] = field(default=(), repr=False)
    mode: str = "threshold"

    def summary(self) -> str:
        """Describe the estimand and explain unavailable estimates."""
        estimand = (f"Pair-scoring evaluation at threshold {self.threshold:g}." if self.mode == "threshold"
                    else "Final-link evaluation using saved selected membership; no threshold is reapplied.")
        parts = [estimand, f"{self.n_labeled:,} labeled pairs; {self.n_unlabeled:,} labels left blank."]
        for name in ("precision", "recall", "f1"):
            estimate, low, high = getattr(self, name)
            label = "F1" if name == "f1" else name.capitalize()
            parts.append(f"{label} {_format(estimate)} (95% CI {_format(low)} to {_format(high)}).")
        parts.append(f"Weighted Brier score {_format(self.brier)}; Brier and calibration assess pair scores.")
        parts.append("Estimates use sampling weights; 95% intervals use a bootstrap within bins.")
        parts.append("Recall is among judged candidate pairs only; it excludes pairs without a probability "
                     "and true matches lost in blocking.")
        parts.append("Intervals condition on the judged candidates and fixed predictions; "
                     "they do not estimate blocking uncertainty or uncertainty from unjudged pairs.")
        parts.extend(self._notes)
        return " ".join(parts)

    def to_markdown(self) -> str:
        """Return appendix tables without requiring pandas' optional tabulate dependency."""
        lines = ["| Metric | Estimate | 95% interval |", "|:--|--:|:--|"]
        for name, label in (("precision", "Precision"), ("recall", "Judged-candidate recall"), ("f1", "F1")):
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
             n_boot: int = 2000, seed: int = 0, mode: str = "threshold") -> Evaluation:
    """Evaluate thresholded pair scores or explicit final-link membership, with audit weights.

    mode="threshold" (default) predicts p >= threshold, ignoring assignment and margins.
    mode="selected" uses the nonmissing selected column and does not reapply threshold.
    Brier and calibration assess pair probabilities in either mode. Recall is limited to
    judged candidate pairs, not all true links; bootstrap intervals condition on that pool.
    F1 is 2 TP / (predicted links + true matches).

    Blank labels: keep those rows, weights included. Within each bin the labeled pairs are
    reweighted to the total weight of all the bin's sampled rows, so a bin with more blanks
    is not underrepresented. This applies to every estimate, to Brier and to the bootstrap,
    and changes nothing when no label is blank. It assumes that within a bin a blank is
    unrelated to the truth; hard pairs left blank can still bias the result. A bin with
    sampled rows but no label cannot be estimated: population-wide estimates are then NaN
    and the summary states the share of the sampled weight those bins hold.
    """
    if mode not in ("threshold", "selected"):
        raise ValueError("mode must be 'threshold' for pair scores or 'selected' for final links")
    threshold = _number(threshold, "threshold", probability=True)
    n_boot = _integer(n_boot, "n_boot", 1)
    seed = _integer(seed, "seed", 0)
    _columns(labeled, ["p", "bin", "weight", "is_match"], "labeled data")
    if mode == "selected":
        if "selected" not in labeled:
            raise ValueError("mode='selected' needs a 'selected' column from actual final links; "
                             "use Result.audit_sample() or audit_sample(scores, links=links)")
        _pairs(labeled, "labeled data")
    frame = labeled.reset_index(drop=True).copy()
    selection = _selected(frame["selected"]) if mode == "selected" else None
    y = _labels(frame["is_match"])
    keep = y.notna()
    n_labeled, n_unlabeled = int(keep.sum()), int((~keep).sum())
    if frame["bin"].isna().any():
        raise ValueError("column 'bin' must identify a sampling stratum for every pair")
    strata = (list(frame["bin"].cat.categories) if isinstance(frame["bin"].dtype, pd.CategoricalDtype)
              else list(pd.unique(frame["bin"])))
    judged = frame.loc[keep].reset_index(drop=True)
    p = _probabilities(judged, missing=False).to_numpy()
    sampled = _sampling_weights(frame, keep)
    if len(sampled):
        # Ratios are invariant to a common scale; avoid overflow with large population weights.
        sampled = sampled / sampled.max()
    labeled_rows = keep.to_numpy()
    weights = sampled[labeled_rows]
    # A bin's sampled rows, labeled or not, stand for its whole population. Its labeled pairs
    # take over the weight of its blank rows; otherwise bins with more blanks would count for less.
    codes, names = pd.factorize(frame["bin"])
    bin_weight = np.bincount(codes, weights=sampled, minlength=len(names))
    labeled_weight = np.bincount(codes[labeled_rows], weights=weights, minlength=len(names))
    blanks = np.bincount(codes[~labeled_rows], minlength=len(names))
    adjusted = (blanks > 0) & (labeled_weight > 0)
    factors = np.divide(bin_weight, labeled_weight, out=np.ones(len(names)), where=adjusted)
    weights = weights * factors[codes[labeled_rows]]
    truth = y.loc[keep].to_numpy()
    predicted = p >= threshold if selection is None else selection.loc[keep].to_numpy()
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
        listed = ", ".join(str(value) for value in empty)
        notes.append(f"No labels in bin(s) {listed}: population-wide estimates and intervals are NaN.")
        shares = {name: bin_weight[i] / bin_weight.sum() for i, name in enumerate(names) if name in empty}
        if shares:
            detail = ", ".join(f"{name} {share:.1%}" for name, share in shares.items())
            notes.append(f"The unlabeled bins hold {sum(shares.values()):.1%} of the sampled weight ({detail}); "
                         "no labeled pair can stand in for them, so label pairs in every bin.")
        unsampled = [str(value) for value in empty if value not in shares]
        if unsampled:
            notes.append(f"Bin(s) {', '.join(unsampled)} had no sampled rows, so their weight is unknown.")
    if not n_labeled or empty:
        estimates[:] = np.nan
        brier = float("nan")
    else:
        if totals[1] == 0:
            description = ("No predicted links among labeled pairs at this threshold" if mode == "threshold"
                           else "No selected links among labeled pairs")
            notes.append(f"{description}: precision and its interval are NaN.")
        if totals[2] == 0:
            notes.append("No true matches among labeled pairs: recall and its interval are NaN.")
        if totals[1] + totals[2] == 0:
            links = "predicted" if mode == "threshold" else "selected"
            notes.append(f"No {links} links or true matches among labeled pairs: F1 and its interval are NaN.")
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
    if adjusted.any() and not empty:
        detail = ", ".join(f"{name} x{factors[i]:.3g}" for i, name in enumerate(names) if adjusted[i])
        notes.append("Rows with a blank label were kept as part of the sample: in each bin that has them, the "
                     f"labeled pairs were reweighted to the bin's full sampling weight ({detail}), so unequal "
                     "blank rates across bins do not shift the estimates. Labels left blank for reasons "
                     "related to the truth within a bin can still bias them.")
    metrics = [tuple(float(value) for value in (estimate, *interval))
               for estimate, interval in zip(estimates, intervals)]
    return Evaluation(*metrics, brier, table, n_labeled, n_unlabeled,
                      threshold if mode == "threshold" else None, tuple(notes), mode=mode)


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

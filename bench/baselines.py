"""Transparent string baselines, with explicitly optimistic truth-tuned thresholds."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from jlink.fields import check_columns, ids, normalize, parse_on, record_text

from bench.data import PAIR_COLUMNS, pair_set


@dataclass
class BaselineResult:
    """Predicted links, measured quality, and the restrictions used to obtain them."""

    links: pd.DataFrame
    threshold: float | None
    metrics: dict
    candidate_count: int
    blocking_recall: float
    best_match_recall: float
    method: str
    tuned_on_truth: bool

    def summary(self) -> dict:
        """Serializable results without large link tables."""
        return {"method": self.method, "threshold": self.threshold, "metrics": self.metrics,
                "candidate_count": self.candidate_count, "blocking_recall": self.blocking_recall,
                "best_match_recall": self.best_match_recall, "links": len(self.links),
                "tuned_on_truth": self.tuned_on_truth,
                "interpretation": "in-sample threshold oracle; upper bound for this fixed best-match method"
                if self.tuned_on_truth else "untuned normalized exact-match baseline"}


def score_against_truth(links: pd.DataFrame, truth: pd.DataFrame,
                        candidates: pd.DataFrame | None = None) -> dict:
    """Score unique ID pairs; missing predictions and empty truth use zero-valued rates."""
    predicted, actual = pair_set(links, "links"), pair_set(truth, "truth")
    tp = len(predicted & actual)
    fp, fn = len(predicted - actual), len(actual - predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(actual) if actual else 0.0
    result = {"precision": precision, "recall": recall, "f1": 2 * tp / (2 * tp + fp + fn)
              if 2 * tp + fp + fn else 0.0, "tp": tp, "fp": fp, "fn": fn}
    if candidates is not None:
        result["pairs_completeness"] = len(actual & pair_set(candidates, "candidates")) / len(actual) \
            if actual else 0.0
    return result


def _inputs(left, right, on, left_id, right_id, truth):
    fields = parse_on(on)
    check_columns(left, [f[1] for f in fields], "left")
    check_columns(right, [f[2] for f in fields], "right")
    left_ids, right_ids = ids(left, left_id, "left"), ids(right, right_id, "right")
    for side, values in (("left", left_ids), ("right", right_ids)):
        if values.isna().any() or values.isin([""]).any():
            raise ValueError(f"the {side} IDs must not be missing or empty")
    actual = pair_set(truth, "truth")
    for side, values, position in (("left", left_ids, 0), ("right", right_ids, 1)):
        unknown = {pair[position] for pair in actual} - set(values)
        if unknown:
            raise ValueError(f"truth has {len(unknown)} {side}_id values absent from the {side} data")
    return fields, left_ids, right_ids, actual


def _frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[*PAIR_COLUMNS, "sim"])


def tune_threshold(best: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    """Maximize F1 over every distinct score, including rejecting all; ties prefer fewer links."""
    check_columns(best, [*PAIR_COLUMNS, "sim"], "best matches")
    predicted, actual = pair_set(best, "best matches"), pair_set(truth, "truth")
    if len(predicted) != len(best) or best.left_id.duplicated().any():
        raise ValueError("best matches must contain at most one pair per left_id")
    scores = pd.to_numeric(best.sim, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("best matches column 'sim' must contain finite values between 0 and 1")
    ordered = best.assign(sim=scores).sort_values("sim", ascending=False, kind="stable")
    ordered = ordered.reset_index(drop=True)
    if ordered.empty or not actual:
        return ordered.iloc[:0].copy(), None
    correct = np.array([pair in actual for pair in ordered[PAIR_COLUMNS].itertuples(index=False, name=None)])
    cumulative = correct.cumsum()
    values = ordered.sim.to_numpy()
    # Evaluate only after the last member of each tie: a threshold cannot split equal scores.
    ends = np.flatnonzero(np.r_[values[:-1] != values[1:], True])
    best_end, best_tp, best_denominator = -1, 0, 1
    for end in ends:
        tp, denominator = int(cumulative[end]), int(end + 1 + len(actual))
        if tp * best_denominator > best_tp * denominator:
            best_end, best_tp, best_denominator = int(end), tp, denominator
    if best_end == -1:
        return ordered.iloc[:0].copy(), None
    return ordered.iloc[:best_end + 1].copy(), float(values[best_end])


def _result(best, truth, count, blocking_recall, method) -> BaselineResult:
    links, threshold = tune_threshold(best, truth)
    return BaselineResult(links, threshold, score_against_truth(links, truth), count, blocking_recall,
                          score_against_truth(best, truth)["recall"], method, True)


def exact_match(left: pd.DataFrame, right: pd.DataFrame, truth: pd.DataFrame, *, on,
                left_id: str | None = None, right_id: str | None = None) -> BaselineResult:
    """All pairs equal in every normalized field; any empty field prevents an exact match."""
    fields, li, ri, _ = _inputs(left, right, on, left_id, right_id, truth)
    a = [left[c].map(normalize).tolist() for _, c, _ in fields]
    b = [right[c].map(normalize).tolist() for _, _, c in fields]
    index = defaultdict(list)
    for j, key in enumerate(zip(*b)):
        if all(key):
            index[key].append(j)
    rows = []
    for i, key in enumerate(zip(*a)):
        if all(key):
            rows.extend((li[i], ri[j], 1.0) for j in index.get(key, ()))
    links = _frame(rows)
    metrics = score_against_truth(links, truth)
    return BaselineResult(links, 1.0, metrics, len(links), metrics["recall"], metrics["recall"],
                          "normalized exact on every field (all equal pairs)", False)


def jaro_winkler(left: pd.DataFrame, right: pd.DataFrame, truth: pd.DataFrame, *, on,
                 left_id: str | None = None, right_id: str | None = None) -> BaselineResult:
    """Choose one right record per left within first-character blocks, then tune the threshold."""
    import jellyfish

    fields, li, ri, actual = _inputs(left, right, on, left_id, right_id, truth)
    a = record_text(left, [f[1] for f in fields]).tolist() if len(left) else []
    b = record_text(right, [f[2] for f in fields]).tolist() if len(right) else []
    index = defaultdict(list)
    for j, text in enumerate(b):
        if text:
            index[text[0]].append(j)
    # A block is streamed per left row, never materialized as the full cross-product.
    rows, count = [], 0
    for i, text in enumerate(a):
        if not text:
            continue
        choices = index.get(text[0], ())
        count += len(choices)
        winner, highest = None, -1.0
        for j in choices:
            similarity = jellyfish.jaro_winkler_similarity(text, b[j])
            if similarity > highest:  # equal scores retain the earliest right source row
                winner, highest = j, similarity
        if winner is not None:
            rows.append((li[i], ri[winner], highest))
    ak, bk = dict(zip(li, a)), dict(zip(ri, b))
    covered = sum(bool(ak[x] and bk[y] and ak[x][0] == bk[y][0]) for x, y in actual)
    return _result(_frame(rows), truth, count, covered / len(actual) if actual else 0.0,
                   "Jaro-Winkler best per left; nonempty shared first-character block")


def tfidf_cosine(left: pd.DataFrame, right: pd.DataFrame, truth: pd.DataFrame, *, on,
                 left_id: str | None = None, right_id: str | None = None,
                 chunk_size: int = 128) -> BaselineResult:
    """Best positive character (2,4)-gram cosine; fit both sources and multiply sparse chunks."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive number of left records")
    fields, li, ri, actual = _inputs(left, right, on, left_id, right_id, truth)
    a = record_text(left, [f[1] for f in fields]).tolist() if len(left) else []
    b = record_text(right, [f[2] for f in fields]).tolist() if len(right) else []
    method = "TF-IDF best per left; character (2,4)-grams; positive cosine candidates"
    if not a or not b:
        return _result(_frame([]), truth, 0, 0.0, method)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False, dtype=np.float64)
    try:
        vectors = vectorizer.fit_transform(a + b)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        return _result(_frame([]), truth, 0, 0.0, method)
    va, vb = vectors[:len(a)], vectors[len(a):]
    right_transpose = vb.T.tocsr()
    rows, count = [], 0
    for start in range(0, len(a), chunk_size):
        similarities = (va[start:start + chunk_size] @ right_transpose).tocsr()
        similarities.eliminate_zeros()
        count += similarities.nnz
        for offset in range(similarities.shape[0]):
            begin, end = similarities.indptr[offset:offset + 2]
            if begin == end:
                continue
            scores = similarities.data[begin:end]
            positions = similarities.indices[begin:end]
            highest = scores.max()
            winner = positions[scores == highest].min()
            rows.append((li[start + offset], ri[winner], float(np.clip(highest, 0, 1))))
    left_positions, right_positions = dict(zip(li, range(len(a)))), dict(zip(ri, range(len(b))))
    ordered_truth = list(actual)
    if ordered_truth:
        true_scores = va[[left_positions[x] for x, _ in ordered_truth]].multiply(
            vb[[right_positions[y] for _, y in ordered_truth]]).sum(axis=1)
        covered = int(np.count_nonzero(np.asarray(true_scores).ravel() > 0))
    else:
        covered = 0
    return _result(_frame(rows), truth, count, covered / len(actual) if actual else 0.0, method)

"""Cheap, deterministic candidate passes before pairs are sent to the judge."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .fields import check_columns, ids, normalize, parse_on, record_text

# Bound even a fully populated sparse product to about 64 MiB (float64 + int32).
_CHUNK_ROWS = 256
_PRODUCT_BYTES = 64 * 1024 * 1024
_PAIR_CHUNK_ROWS = 8192
_SIM_NGRAMS = (2, 4)
_IGNORED = frozenset({
    "and", "of", "the", "for", "inc", "corp", "co", "ltd", "llc", "plc",
    "company", "corporation", "incorporated", "limited",
})


class Blocker:
    """A named pass returning pairs of row positions, independent of source IDs."""

    name: str

    def pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
        """Return an int64 array of shape (number of pairs, 2)."""
        raise NotImplementedError("implement pairs(left, right) for this blocking pass")


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"`{label}` must be a positive integer")


def _pass_fields(columns: tuple, name: str | None, kind: str) -> tuple[list, str]:
    fields = parse_on(columns)
    if name is None:
        names = [a if a == b else f"{a}={b}" for _, a, b in fields]
        name = f"{kind}:{','.join(names)}"
    if not isinstance(name, str) or not name.strip():
        raise ValueError("a blocker's `name` must be a nonempty string")
    return fields, name


def _columns(left: pd.DataFrame, right: pd.DataFrame, fields: list) -> tuple[list[str], list[str]]:
    for side, frame in (("left", left), ("right", right)):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"the {side} data must be a pandas DataFrame")
    a, b = [f[1] for f in fields], [f[2] for f in fields]
    check_columns(left, a, "left")
    check_columns(right, b, "right")
    return a, b


def _empty_pairs() -> np.ndarray:
    return np.empty((0, 2), dtype=np.int64)


def exact(*columns: str | tuple[str, str], name: str | None = None) -> Blocker:
    """Pair records equal in every listed field; incomplete keys never pair."""
    fields, name = _pass_fields(columns, name, "exact")
    return _Exact(fields, name)


@dataclass
class _Exact(Blocker):
    fields: list
    name: str

    def pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
        a, b = _columns(left, right, self.fields)
        groups = {}
        for j, key in enumerate(zip(*(right[c].map(normalize) for c in b))):
            if all(key):
                groups.setdefault(key, []).append(j)
        pairs = []
        for i, key in enumerate(zip(*(left[c].map(normalize) for c in a))):
            if all(key):
                pairs.extend((i, j) for j in groups.get(key, ()))
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def ngrams(*columns: str | tuple[str, str], k: int = 10, n: tuple[int, int] = (2, 4),
           min_sim: float = 0.1, name: str | None = None) -> Blocker:
    """Keep up to k cosine neighbors per left row; ties use right row position."""
    fields, name = _pass_fields(columns, name, "ngrams")
    _positive_int(k, "k")
    if not isinstance(n, tuple) or len(n) != 2:
        raise ValueError("`n` must be a (minimum, maximum) tuple of positive n-gram lengths")
    for value in n:
        _positive_int(value, "n")
    if n[0] > n[1]:
        raise ValueError("`n` must have its minimum n-gram length before its maximum")
    if (isinstance(min_sim, (bool, np.bool_)) or not isinstance(min_sim, Real)
            or not np.isfinite(min_sim) or not 0 <= min_sim <= 1):
        raise ValueError("`min_sim` must be a finite number between 0 and 1")
    return _Ngrams(fields, name, int(k), n, float(min_sim))


def _vectors(left: pd.DataFrame, right: pd.DataFrame, a: list[str], b: list[str],
             n: tuple[int, int]) -> tuple[csr_matrix, csr_matrix]:
    text = pd.concat([record_text(left, a), record_text(right, b)], ignore_index=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=n, norm="l2", dtype=np.float64)
    try:
        matrix = vectorizer.fit_transform(text).tocsr()
    except ValueError as error:
        if "empty vocabulary" not in str(error):
            raise
        matrix = csr_matrix((len(text), 0), dtype=np.float64)
    # A fixed feature traversal keeps dot-product rounding independent of row chunks.
    matrix.sort_indices()
    return matrix[:len(left)], matrix[len(left):]


def _top_k(positions: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Partition by score, resolving the entire boundary tie by position."""
    if len(scores) > k:
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        better = np.flatnonzero(scores > cutoff)
        tied = np.flatnonzero(scores == cutoff)
        remaining = k - len(better)
        if len(tied) > remaining:
            tied = tied[np.argpartition(positions[tied], remaining - 1)[:remaining]]
        keep = np.concatenate((better, tied))
        positions, scores = positions[keep], scores[keep]
    return positions[np.lexsort((positions, -scores))]


def _clip_cosines(scores: np.ndarray) -> np.ndarray:
    np.clip(scores, 0, 1, out=scores)
    # Normalized identical vectors can sum to just below one in floating point.
    scores[np.abs(scores - 1) <= 1e-14] = 1
    return scores


@dataclass
class _Ngrams(Blocker):
    fields: list
    name: str
    k: int
    n: tuple[int, int]
    min_sim: float

    def pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
        a, b = _columns(left, right, self.fields)
        if left.empty or right.empty:
            return _empty_pairs()
        x, y = _vectors(left, right, a, b, self.n)
        k = min(self.k, len(right))
        if not x.nnz or not y.nnz:
            if self.min_sim > 0:
                return _empty_pairs()
            return np.column_stack((np.repeat(np.arange(len(left), dtype=np.int64), k),
                                    np.tile(np.arange(k, dtype=np.int64), len(left))))
        transpose = y.T.tocsr()
        rows = max(1, min(_CHUNK_ROWS, _PRODUCT_BYTES // (12 * len(right))))
        output = np.empty((len(left) * k, 2), dtype=np.int64)
        count = 0
        for start in range(0, len(left), rows):
            product = (x[start:start + rows] @ transpose).tocsr()
            _clip_cosines(product.data)
            for local in range(product.shape[0]):
                lo, hi = product.indptr[local:local + 2]
                positions, scores = product.indices[lo:hi], product.data[lo:hi]
                keep = (scores >= self.min_sim) & (scores > 0)
                positions, scores = positions[keep], scores[keep]
                chosen = _top_k(positions, scores, k)
                if self.min_sim == 0 and len(chosen) < k:
                    # Implicit sparse zeros qualify too, including for missing text.
                    first = np.arange(k, dtype=np.int64)
                    zeros = first[~np.isin(first, chosen)][:k - len(chosen)]
                    chosen = np.concatenate((chosen, zeros))
                stop = count + len(chosen)
                output[count:stop, 0] = start + local
                output[count:stop, 1] = chosen
                count = stop
        return output[:count].copy()


def initials(column: str | tuple[str, str], min_len: int = 2, name: str | None = None) -> Blocker:
    """Pair an acronym with a token expansion in either direction, omitting filler words."""
    fields, name = _pass_fields((column,), name, "initials")
    _positive_int(min_len, "min_len")
    return _Initials(fields, name, int(min_len))


def _acronym_keys(text: str, min_len: int) -> tuple[str, str]:
    acronym = text if text.isalpha() and len(text) >= min_len else ""
    expanded = "".join(token[0] for token in text.split() if token not in _IGNORED)
    if len(expanded) < min_len:
        expanded = ""
    return acronym, expanded


@dataclass
class _Initials(Blocker):
    fields: list
    name: str
    min_len: int

    def pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
        a, b = _columns(left, right, self.fields)
        acronyms, expansions = {}, {}
        for j, text in enumerate(right[b[0]].map(normalize)):
            acronym, expanded = _acronym_keys(text, self.min_len)
            if acronym:
                acronyms.setdefault(acronym, []).append(j)
            if expanded:
                expansions.setdefault(expanded, []).append(j)
        pairs = []
        for i, text in enumerate(left[a[0]].map(normalize)):
            acronym, expanded = _acronym_keys(text, self.min_len)
            matches = set(expansions.get(acronym, ())) | set(acronyms.get(expanded, ()))
            pairs.extend((i, j) for j in sorted(matches))
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _checked_pairs(blocker: Blocker, left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    pairs = blocker.pairs(left, right)
    if (not isinstance(pairs, np.ndarray) or pairs.ndim != 2 or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)):
        raise ValueError(f"blocker {blocker.name!r} must return an integer array "
                         "with shape (number of pairs, 2)")
    if pairs.size and (np.any(pairs < 0) or np.any(pairs[:, 0] >= len(left))
                       or np.any(pairs[:, 1] >= len(right))):
        raise ValueError(f"blocker {blocker.name!r} returned a row position outside the left or right data")
    return pairs


def _pair_similarities(left: pd.DataFrame, right: pd.DataFrame, a: list[str], b: list[str],
                       pairs: np.ndarray) -> np.ndarray:
    x, y = _vectors(left, right, a, b, _SIM_NGRAMS)
    similarities = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), _PAIR_CHUNK_ROWS):
        chunk = pairs[start:start + _PAIR_CHUNK_ROWS]
        similarities[start:start + len(chunk)] = np.asarray(
            x[chunk[:, 0]].multiply(y[chunk[:, 1]]).sum(axis=1)
        ).ravel()
    return _clip_cosines(similarities)


def candidates(left: pd.DataFrame, right: pd.DataFrame, *, on: str | list[str | tuple[str, str]],
               blockers: list[Blocker] | None = None, left_id: str | None = None,
               right_id: str | None = None, max_pairs: int | None = 5_000_000) -> pd.DataFrame:
    """Union named passes and score every pair on all `on` fields using 2–4 character grams."""
    try:
        fields = parse_on(on)
    except TypeError as error:
        raise ValueError("`on` must list column names or (left, right) pairs of column names") from error
    a, b = _columns(left, right, fields)
    left_ids, right_ids = ids(left, left_id, "left"), ids(right, right_id, "right")
    if max_pairs is not None and (isinstance(max_pairs, (bool, np.bool_))
                                  or not isinstance(max_pairs, Integral) or max_pairs < 0):
        raise ValueError("`max_pairs` must be a nonnegative integer, or None for no limit")
    if blockers is None:
        blockers = [ngrams(*[(a, b) for _, a, b in fields], k=10)]
    if not isinstance(blockers, list) or any(not isinstance(p, Blocker) for p in blockers):
        raise ValueError("`blockers` must be a list of Blocker passes, or None for the default n-gram pass")
    union = {}
    for pass_number, blocker in enumerate(blockers):
        if not isinstance(getattr(blocker, "name", None), str) or not blocker.name.strip():
            raise ValueError("each blocker must have a nonempty string `name`")
        bit = 1 << pass_number
        for i, j in _checked_pairs(blocker, left, right):
            key = (int(i), int(j))
            union[key] = union.get(key, 0) | bit
        if max_pairs is not None and len(union) > max_pairs:
            raise ValueError(f"blocking produced {len(union):,} pairs, exceeding max_pairs={max_pairs:,}; "
                             "use a smaller `k` or a more selective `exact` pass")
    pairs = np.asarray(list(union), dtype=np.int64).reshape(-1, 2)
    labels = {mask: "+".join(p.name for i, p in enumerate(blockers) if mask & (1 << i))
              for mask in set(union.values())}
    blocks = [labels[mask] for mask in union.values()]
    del union
    sim = _pair_similarities(left, right, a, b, pairs) if len(pairs) else np.empty(0, dtype=float)
    order = np.lexsort((pairs[:, 1], -sim, pairs[:, 0]))
    return pd.DataFrame({
        "left_id": left_ids.take(pairs[:, 0]),
        "right_id": right_ids.take(pairs[:, 1]),
        "block": pd.Series(blocks, dtype=object),
        "sim": sim,
    }).iloc[order].reset_index(drop=True)


def pairs_completeness(candidates: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Share of distinct truth pairs proposed; NaN when there are no known truth pairs."""
    columns = ["left_id", "right_id"]
    for label, frame in (("candidates", candidates), ("truth", truth)):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"the {label} data must be a pandas DataFrame")
        check_columns(frame, columns, label)
    known = pd.MultiIndex.from_frame(truth[columns].drop_duplicates())
    if not len(known):
        return float("nan")
    proposed = pd.MultiIndex.from_frame(candidates[columns].drop_duplicates())
    return float(known.isin(proposed).mean())

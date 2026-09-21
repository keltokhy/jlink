"""Cheap, deterministic candidate passes before pairs are sent to the judge."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .fields import check_columns, ids, key_text, normalize, parse_on, record_text, side_fields

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

    def iter_pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> Iterator[np.ndarray]:
        """Yield pair batches. Override to enforce limits before allocating all pairs."""
        yield self.pairs(left, right)

    def to_config(self) -> dict:
        """JSON-safe description; custom passes must describe their own parameters."""
        return {"type": "custom", "name": self.name,
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "reconstructable": False}

    def _validate(self, left: pd.DataFrame, right: pd.DataFrame) -> None:
        pass


class _StreamingBlocker(Blocker):
    def pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
        batches = list(self.iter_pairs(left, right))
        return np.concatenate(batches) if batches else _empty_pairs()

    def _validate(self, left: pd.DataFrame, right: pd.DataFrame) -> None:
        _columns(left, right, self.fields)


def _field_config(kind: str, blocker: Blocker) -> dict:
    return {"type": kind, "name": blocker.name,
            "columns": [[a, b] for _, a, b in blocker.fields]}


def _pack_rows(rows) -> Iterator[np.ndarray]:
    """Pack (left position, right positions) without a Cartesian Python list."""
    output = np.empty((_PAIR_CHUNK_ROWS, 2), dtype=np.int64)
    count = 0
    for i, positions in rows:
        for start in range(0, len(positions), _PAIR_CHUNK_ROWS):
            chunk = positions[start:start + _PAIR_CHUNK_ROWS]
            offset = 0
            while offset < len(chunk):
                take = min(len(chunk) - offset, len(output) - count)
                output[count:count + take, 0] = i
                output[count:count + take, 1] = chunk[offset:offset + take]
                count += take
                offset += take
                if count == len(output):
                    yield output.copy()
                    count = 0
    if count:
        yield output[:count].copy()


def _keys(frame: pd.DataFrame, columns: list[str]):
    # Object conversion makes categorical nulls and datetime NaT follow the same
    # missing-key policy as None, rather than surviving map() as nonempty keys.
    # key_text lets an integer year meet the same year stored as a float or as "1985.0".
    return zip(*(frame[c].astype(object).where(frame[c].notna(), None).map(key_text)
                 for c in columns))


def _groups(frame: pd.DataFrame, columns: list[str], missing: str = "drop") -> dict:
    groups = {}
    for i, key in enumerate(_keys(frame, columns)):
        if missing == "match" or all(key):
            groups.setdefault(key, []).append(i)
    return groups


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"`{label}` must be a positive integer")


def _pass_fields(columns: tuple, name: str | None, kind: str) -> tuple[list, str]:
    for item in columns:
        if isinstance(item, (tuple, list)) and any(c is None for c in item):
            raise ValueError(f"`{kind}` compares a left column with a right column, so it cannot use the "
                             f"one-sided field {tuple(item)!r}; one-sided fields are only shown to the judge")
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


def _no_shared_key(blocker: Blocker, left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    """Why a keyed pass proposed nothing, if its two sides have no key in common."""
    a, b = _columns(left, right, blocker.fields)
    policy = getattr(blocker, "missing", "drop")
    ours, theirs = _groups(left, a, policy), _groups(right, b, policy)
    if ours.keys() & theirs.keys():
        return None
    examples = "; ".join(f"{side} example: {' | '.join(next(iter(groups)))!r}" if groups
                         else f"{side}: no complete key"
                         for side, groups in (("left", ours), ("right", theirs)))
    return (f"blocking pass {blocker.name!r} proposed no pairs: its key columns have no value in common "
            f"between the {len(left):,} left and {len(right):,} right records ({examples}). "
            "Check that both sides spell and code these columns the same way.")


def exact(*columns: str | tuple[str, str], name: str | None = None) -> Blocker:
    """Pair records equal in every listed field; incomplete keys never pair."""
    fields, name = _pass_fields(columns, name, "exact")
    return _Exact(fields, name)


@dataclass
class _Exact(_StreamingBlocker):
    fields: list
    name: str

    def to_config(self) -> dict:
        return _field_config("exact", self)

    def iter_pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> Iterator[np.ndarray]:
        a, b = _columns(left, right, self.fields)
        groups = _groups(right, b)
        yield from _pack_rows((i, groups.get(key, ())) for i, key in enumerate(_keys(left, a))
                              if all(key))


def within(blocker: Blocker, *columns: str | tuple[str, str], missing: str = "drop",
           name: str | None = None) -> Blocker:
    """Run a blocker separately inside normalized exact groups, then restore positions.

    ``missing='drop'`` excludes any incomplete group key. ``missing='match'`` treats
    empty normalized components as equal, while still requiring all other components
    to agree. N-gram TF-IDF is fitted independently within each matching group.
    """
    if not isinstance(blocker, Blocker):
        raise ValueError("`blocker` must be a Blocker pass")
    if missing not in ("drop", "match"):
        raise ValueError("`missing` must be 'drop' or 'match'")
    fields, group_name = _pass_fields(columns, name, "within")
    if name is None:
        group_name += f"[{missing}]({blocker.name})"
    return _Within(fields, group_name, blocker, missing)


@dataclass
class _Within(_StreamingBlocker):
    fields: list
    name: str
    blocker: Blocker
    missing: str

    def to_config(self) -> dict:
        return _field_config("within", self) | {
            "missing": self.missing, "blocker": self.blocker.to_config()}

    def _validate(self, left: pd.DataFrame, right: pd.DataFrame) -> None:
        super()._validate(left, right)
        self.blocker._validate(left, right)

    def iter_pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> Iterator[np.ndarray]:
        self._validate(left, right)
        a, b = _columns(left, right, self.fields)
        left_groups, right_groups = _groups(left, a, self.missing), _groups(right, b, self.missing)
        for key, left_positions in left_groups.items():
            if key not in right_groups:
                continue
            ia = np.asarray(left_positions, dtype=np.int64)
            ib = np.asarray(right_groups[key], dtype=np.int64)
            for pairs in _checked_batches(self.blocker, left.iloc[ia], right.iloc[ib]):
                yield np.column_stack((ia[pairs[:, 0]], ib[pairs[:, 1]]))


def ngrams(*columns: str | tuple[str, str], k: int = 10, n: tuple[int, int] = (2, 4),
           min_sim: float = 0.1, name: str | None = None, reverse: bool = False) -> Blocker:
    """Keep k right neighbors per left row, or k left per right with reverse=True.

    Ties use the neighbor's original row position. Union forward and reverse passes
    in ``candidates`` for symmetric search; the default remains forward only.
    """
    if not isinstance(reverse, (bool, np.bool_)):
        raise ValueError("`reverse` must be a boolean")
    fields, name = _pass_fields(columns, name, "ngrams-reverse" if reverse else "ngrams")
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
    return _Ngrams(fields, name, int(k), tuple(int(v) for v in n), float(min_sim), bool(reverse))


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
class _Ngrams(_StreamingBlocker):
    fields: list
    name: str
    k: int
    n: tuple[int, int]
    min_sim: float
    reverse: bool

    def to_config(self) -> dict:
        return _field_config("ngrams", self) | {
            "k": self.k, "n": list(self.n), "min_sim": self.min_sim, "reverse": self.reverse}

    def iter_pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> Iterator[np.ndarray]:
        a, b = _columns(left, right, self.fields)
        if left.empty or right.empty:
            return
        x, y = _vectors(left, right, a, b, self.n)
        if self.reverse:
            x, y = y, x
        for pairs in _pack_rows(self._neighbors(x, y)):
            yield pairs[:, ::-1].copy() if self.reverse else pairs

    def _neighbors(self, x: csr_matrix, y: csr_matrix):
        k = min(self.k, y.shape[0])
        if not x.nnz or not y.nnz:
            if self.min_sim == 0:
                first = np.arange(k, dtype=np.int64)
                for i in range(x.shape[0]):
                    yield i, first
            return
        transpose = y.T.tocsr()
        rows = max(1, min(_CHUNK_ROWS, _PRODUCT_BYTES // (12 * y.shape[0])))
        for start in range(0, x.shape[0], rows):
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
                yield start + local, chosen


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
class _Initials(_StreamingBlocker):
    fields: list
    name: str
    min_len: int

    def to_config(self) -> dict:
        return _field_config("initials", self) | {"min_len": self.min_len}

    def iter_pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> Iterator[np.ndarray]:
        a, b = _columns(left, right, self.fields)
        acronyms, expansions = {}, {}
        for j, text in enumerate(right[b[0]].map(normalize)):
            acronym, expanded = _acronym_keys(text, self.min_len)
            if acronym:
                acronyms.setdefault(acronym, []).append(j)
            if expanded:
                expansions.setdefault(expanded, []).append(j)

        def rows():
            for i, text in enumerate(left[a[0]].map(normalize)):
                acronym, expanded = _acronym_keys(text, self.min_len)
                matches = set(expansions.get(acronym, ())) | set(acronyms.get(expanded, ()))
                yield i, sorted(matches)

        yield from _pack_rows(rows())


def _checked_pairs(blocker: Blocker, left: pd.DataFrame, right: pd.DataFrame,
                   pairs: np.ndarray) -> np.ndarray:
    if (not isinstance(pairs, np.ndarray) or pairs.ndim != 2 or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)):
        raise ValueError(f"blocker {blocker.name!r} must return an integer array "
                         "with shape (number of pairs, 2)")
    if pairs.size and (np.any(pairs < 0) or np.any(pairs[:, 0] >= len(left))
                       or np.any(pairs[:, 1] >= len(right))):
        raise ValueError(f"blocker {blocker.name!r} returned a row position outside the left or right data")
    return pairs


def _checked_batches(blocker: Blocker, left: pd.DataFrame,
                     right: pd.DataFrame) -> Iterator[np.ndarray]:
    if not isinstance(getattr(blocker, "name", None), str) or not blocker.name.strip():
        raise ValueError("each blocker must have a nonempty string `name`")
    blocker._validate(left, right)
    for pairs in blocker.iter_pairs(left, right):
        yield _checked_pairs(blocker, left, right, pairs)


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
    """Union passes, score all `on` fields, and map positions to IDs.

    ``result.attrs['blocking']`` contains JSON-safe pass configurations and counts.
    ``max_pairs`` limits unique pairs; built-ins stream batches and stop on overflow.
    One-sided `on` fields, ``(left, None)`` or ``(None, right)``, join that side's text for
    ``sim``. The default n-gram pass searches the paired fields only.
    """
    try:
        fields = parse_on(on, unpaired=True)
    except TypeError as error:
        raise ValueError("`on` must list column names or (left, right) pairs of column names") from error
    paired = [f for f in fields if f[1] is not None and f[2] is not None]
    _columns(left, right, paired)
    a, b = [c for _, c in side_fields(fields, "left")], [c for _, c in side_fields(fields, "right")]
    check_columns(left, a, "left")
    check_columns(right, b, "right")
    left_ids, right_ids = ids(left, left_id, "left"), ids(right, right_id, "right")
    if max_pairs is not None and (isinstance(max_pairs, (bool, np.bool_))
                                  or not isinstance(max_pairs, Integral) or max_pairs < 0):
        raise ValueError("`max_pairs` must be a nonnegative integer, or None for no limit")
    if blockers is None:
        if not paired:
            raise ValueError("the default n-gram pass needs an `on` field that both sides have; every field "
                             "here is one-sided, so choose the passes yourself with `blockers=`")
        blockers = [ngrams(*[(lc, rc) for _, lc, rc in paired], k=10)]
    if not isinstance(blockers, list) or any(not isinstance(p, Blocker) for p in blockers):
        raise ValueError("`blockers` must be a list of Blocker passes, or None for the default n-gram pass")
    union = {}
    diagnostics = []
    for pass_number, blocker in enumerate(blockers):
        bit = 1 << pass_number
        before = len(union)
        proposed = unique = 0
        for batch in _checked_batches(blocker, left, right):
            proposed += len(batch)
            for i, j in batch:
                key = (int(i), int(j))
                previous = union.get(key, 0)
                unique += not bool(previous & bit)
                union[key] = previous | bit
                if max_pairs is not None and len(union) > max_pairs:
                    raise ValueError(
                        f"blocking produced at least {len(union):,} pairs, exceeding "
                        f"max_pairs={max_pairs:,} during {blocker.name!r}; use a smaller `k`, "
                        "wrap a pass with `within(...)`, or remove/better constrain broad passes; "
                        "adding an `exact` pass unions more pairs and does not restrict other passes")
        if not proposed and len(left) and len(right) and isinstance(blocker, (_Exact, _Within)):
            # Disagreeing keys lose every pair of this pass without any other sign.
            if (message := _no_shared_key(blocker, left, right)) is not None:
                warnings.warn(message, stacklevel=2)
        added = len(union) - before
        diagnostics.append({"pass": pass_number, "name": blocker.name, "config": blocker.to_config(),
                            "proposed_pairs": proposed, "unique_pairs": unique, "added_pairs": added,
                            "overlapping_pairs": unique - added, "union_pairs": len(union)})
    exclusive = {}
    for mask in union.values():
        if mask & (mask - 1) == 0:
            exclusive[mask] = exclusive.get(mask, 0) + 1
    for i, diagnostic in enumerate(diagnostics):
        diagnostic["exclusive_pairs"] = exclusive.get(1 << i, 0)
    pairs = np.asarray(list(union), dtype=np.int64).reshape(-1, 2)
    labels = {mask: "+".join(p.name for i, p in enumerate(blockers) if mask & (1 << i))
              for mask in set(union.values())}
    blocks = [labels[mask] for mask in union.values()]
    del union
    sim = _pair_similarities(left, right, a, b, pairs) if len(pairs) else np.empty(0, dtype=float)
    order = np.lexsort((pairs[:, 1], -sim, pairs[:, 0]))
    result = pd.DataFrame({
        "left_id": left_ids.take(pairs[:, 0]),
        "right_id": right_ids.take(pairs[:, 1]),
        "block": pd.Series(blocks, dtype=object),
        "sim": sim,
    }).iloc[order].reset_index(drop=True)
    result.attrs["blocking"] = {"schema_version": 1, "max_pairs": int(max_pairs) if max_pairs is not None else None,
                                "pair_count": len(result), "passes": diagnostics}
    return result


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


def embeddings(*columns: str | tuple[str, str], **kwargs) -> Blocker:
    """Optional local semantic neighbors; see ``jlink.embeddings.embeddings`` for options."""
    from .embeddings import embeddings as factory

    return factory(*columns, **kwargs)

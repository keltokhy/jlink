"""Turn scored candidate pairs into links, then screen for ambiguous matches."""

from __future__ import annotations

from collections import deque
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _columns(frame: pd.DataFrame, names: list[str], table: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{table} must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError(f"{table} has duplicate column names; give each column a unique name")
    for name in names:
        if name not in frame:
            raise ValueError(f"{table} needs a {name!r} column")


def _pairs(frame: pd.DataFrame, table: str) -> None:
    _columns(frame, ["left_id", "right_id"], table)
    for name in ("left_id", "right_id"):
        if frame[name].isna().any():
            raise ValueError(f"{table} column {name!r} has missing IDs; supply an ID for every pair")
    try:
        duplicated = frame.duplicated(["left_id", "right_id"]).any()
    except TypeError as exc:
        raise ValueError(f"{table} IDs must be scalar values, such as strings or integers") from exc
    if duplicated:
        raise ValueError(f"{table} has duplicate (left_id, right_id) pairs; keep one row per pair")


def _numbers(frame: pd.DataFrame, name: str, *, missing: bool = False) -> pd.Series:
    _columns(frame, [name], "the data")
    try:
        numeric = pd.to_numeric(frame[name], errors="raise")
        if pd.api.types.is_complex_dtype(numeric.dtype):
            raise ValueError("complex numbers are not valid probabilities or weights")
        values = numeric.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"column {name!r} must contain numbers") from exc
    if np.isinf(values).any() or (not missing and values.isna().any()):
        raise ValueError(f"column {name!r} must contain finite numbers" + (" or NaN" if missing else ""))
    return values


def _probabilities(frame: pd.DataFrame, name: str = "p", *, missing: bool = True) -> pd.Series:
    values = _numbers(frame, name, missing=missing)
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"column {name!r} must contain probabilities between 0 and 1")
    return values


def _number(value: float, name: str, *, probability: bool = False) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if probability and not 0 <= number <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _margin(scores: pd.DataFrame) -> np.ndarray:
    """Exclude each row itself from both group maxima, including tied maxima."""
    competitors = []
    p = scores["p"]
    for side in ("left_id", "right_id"):
        groups = scores[side]
        highest = p.groupby(groups, sort=False).transform("max")
        at_highest = p.eq(highest)
        count = at_highest.groupby(groups, sort=False).transform("sum")
        second = p.mask(at_highest).groupby(groups, sort=False).transform("max")
        competitors.append(highest.where(~at_highest | count.gt(1), second).to_numpy())
    return p.to_numpy() - np.fmax(*competitors)


def _integers(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Express finite binary floats in a shared, exact integer unit."""
    ratios = [float(value).as_integer_ratio() for value in values]
    denominator = max(den for _, den in ratios)
    return np.asarray([num * (denominator // den) for num, den in ratios], dtype=object), denominator


def _certificate(row: np.ndarray, col: np.ndarray, weights: np.ndarray,
                 matched: np.ndarray, n_columns: int) -> np.ndarray | None:
    """Find exact assignment duals, or detect a probability/similarity improvement.

    For a row assigned to a, column prices must satisfy v[j] >= v[a] + w[i,j] - w[i,a].
    Relax these difference constraints from zero. Unused columns must have price zero;
    an increasing cycle or a positive unused price proves the assignment is suboptimal.
    """
    keys = row * n_columns + col
    chosen = weights[np.searchsorted(keys, np.arange(len(matched)) * n_columns + matched)]
    unused = np.ones(n_columns, dtype=bool)
    unused[matched] = False
    prices = np.zeros(n_columns, dtype=object)
    for _ in range(len(matched) + 1):
        updated = prices.copy()
        np.maximum.at(updated, col, prices[matched[row]] + weights - chosen[row])
        if np.any(updated[unused] != 0):
            return None
        if np.array_equal(updated, prices):
            return prices
        prices = updated
    return None


def _exact_assignment(row: np.ndarray, col: np.ndarray, weights: np.ndarray,
                      n_rows: int, n_columns: int) -> np.ndarray:
    """Hungarian refinement using integers, for ties too close for floating-point SciPy.

    The usual row/column potentials and augmenting paths work with Python integers too.
    This uncommon fallback preserves both objectives exactly; a tiny epsilon cannot do so.
    """
    costs = [dict() for _ in range(n_rows)]
    for a, b, weight in zip(row, col, weights):
        costs[a][b + 1] = -weight
    forbidden = (n_rows + 1) * max(weights) + 1
    u, v = [0] * (n_rows + 1), [0] * (n_columns + 1)
    owner, previous = [0] * (n_columns + 1), [0] * (n_columns + 1)
    for a in range(1, n_rows + 1):
        owner[0] = a
        best = [forbidden] * (n_columns + 1)
        used = [False] * (n_columns + 1)
        current = 0
        while True:
            used[current] = True
            active = owner[current]
            delta, following = None, 0
            for b in range(1, n_columns + 1):
                if used[b]:
                    continue
                reduced = costs[active - 1].get(b, forbidden) - u[active] - v[b]
                if reduced < best[b]:
                    best[b], previous[b] = reduced, current
                if delta is None or best[b] < delta:
                    delta, following = best[b], b
            for b in range(n_columns + 1):
                if used[b]:
                    u[owner[b]] += delta
                    v[b] -= delta
                else:
                    best[b] -= delta
            current = following
            if owner[current] == 0:
                break
        while current:
            parent = previous[current]
            owner[current] = owner[parent]
            current = parent
    matched = np.empty(n_rows, dtype=np.int64)
    for b in range(1, n_columns + 1):
        if owner[b]:
            matched[owner[b] - 1] = b - 1
    return matched


def _canonical_matching(row: np.ndarray, col: np.ndarray, weights: np.ndarray,
                        matched: np.ndarray, prices: np.ndarray, n_real: int,
                        transposed: bool) -> np.ndarray:
    """Prefer the earliest ID pair on the exact optimal face, using alternating paths."""
    n_columns = len(prices)
    chosen = weights[np.searchsorted(row * n_columns + col,
                                     np.arange(len(matched)) * n_columns + matched)]
    tight = prices[col] == prices[matched[row]] + weights - chosen[row]
    tight_row, tight_col = row[tight], col[tight]
    adjacent = [[] for _ in matched]
    for a, b in zip(tight_row, tight_col):
        adjacent[a].append(b)
    real = tight_col < n_real
    a, b = tight_row[real], tight_col[real]
    order = np.lexsort((a, b) if transposed else (b, a))
    owners = np.full(n_columns, -1, dtype=np.int64)
    owners[matched] = np.arange(len(matched))
    locked_rows = np.zeros(len(matched), dtype=bool)
    locked_cols = np.zeros(n_columns, dtype=bool)
    for edge in order:
        start, target = a[edge], b[edge]
        if locked_rows[start] or locked_cols[target]:
            continue
        old = matched[start]
        if target != old:
            displaced = owners[target]
            if displaced < 0:
                if prices[target] != prices[old]:
                    continue
                owners[old] = -1
            else:
                queue = deque([displaced])
                visited = {start, displaced}
                parent = {}
                end = None
                while queue and end is None:
                    current = queue.popleft()
                    for candidate in adjacent[current]:
                        if candidate == target or locked_cols[candidate] or candidate in parent:
                            continue
                        parent[candidate] = current
                        following = owners[candidate]
                        if candidate == old or (following < 0 and prices[candidate] == prices[old]):
                            end = candidate
                            break
                        if following >= 0 and following not in visited and not locked_rows[following]:
                            visited.add(following)
                            queue.append(following)
                if end is None:
                    continue
                owners[old] = -1
                while True:
                    current = parent[end]
                    previous = matched[current]
                    matched[current], owners[end] = end, current
                    if current == displaced:
                        break
                    end = previous
            matched[start], owners[target] = target, start
        locked_rows[start] = locked_cols[target] = True
    return matched


def _assignment(p: np.ndarray, sim: np.ndarray, n_real: int,
                transposed: bool) -> tuple[np.ndarray, np.ndarray]:
    rows, matched = linear_sum_assignment(p, maximize=True)
    row, col = np.nonzero(np.isfinite(p))
    probability, _ = _integers(p[row, col])
    similarity, unit = _integers(sim[row, col])
    # One unit of probability outweighs every possible similarity difference.
    weights = probability * (len(rows) * unit + 1) + similarity
    prices = _certificate(row, col, weights, matched, p.shape[1])
    if prices is None:
        # A fast proposal for ordinary ties. Only an exact dual certificate can accept it.
        _, matched = linear_sum_assignment(p + 1e-9 * sim, maximize=True)
        prices = _certificate(row, col, weights, matched, p.shape[1])
    if prices is None:
        matched = _exact_assignment(row, col, weights, len(rows), p.shape[1])
        prices = _certificate(row, col, weights, matched, p.shape[1])
        assert prices is not None, "integer assignment must have an optimality certificate"
    matched = _canonical_matching(row, col, weights, matched, prices, n_real, transposed)
    return rows, matched


def _greedy(left: np.ndarray, right: np.ndarray, order: np.ndarray) -> np.ndarray:
    used_left, used_right = set(), set()
    chosen = []
    for edge in order:
        a, b = left[edge], right[edge]
        if a not in used_left and b not in used_right:
            chosen.append(edge)
            used_left.add(a)
            used_right.add(b)
    return np.asarray(chosen, dtype=np.int64)


def _one_to_one(pairs: pd.DataFrame) -> np.ndarray:
    try:
        left, left_ids = pd.factorize(pairs["left_id"], sort=True)
        right, right_ids = pd.factorize(pairs["right_id"], sort=True)
    except TypeError as exc:
        raise ValueError("left_id and right_id must contain sortable IDs, "
                         "such as strings or integers") from exc
    n_left, n_right = len(left_ids), len(right_ids)
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (left, n_left + right)),
                       shape=(n_left + n_right, n_left + n_right)).tocsr()
    _, labels = connected_components(graph, directed=False)
    component = labels[left]
    sizes = np.bincount(component)
    single = sizes[component] == 1
    chosen = [np.flatnonzero(single)]
    remaining = np.flatnonzero(~single)
    ordered = remaining[np.argsort(component[remaining], kind="stable")]
    groups = np.split(ordered, np.flatnonzero(np.diff(component[ordered])) + 1) if len(ordered) else []
    probabilities = pairs["p"].to_numpy()
    similarities = pairs["sim"].to_numpy()
    for edges in groups:
        a, row = np.unique(left[edges], return_inverse=True)
        b, col = np.unique(right[edges], return_inverse=True)
        if max(len(a), len(b)) > 2000:
            warnings.warn(
                f"one-to-one component has {len(a):,} left IDs and {len(b):,} right IDs; "
                "a side exceeds 2,000, so using greedy matching, which may not maximize total p",
                RuntimeWarning, stacklevel=3,
            )
            # pairs are already sorted by p, sim, left_id, right_id.
            chosen.append(_greedy(left, right, edges))
            continue
        if min(len(a), len(b)) == 1:
            chosen.append(edges[:1])
            continue
        # Put the smaller side in rows; one private dummy per row allows unmatched IDs.
        transposed = len(a) > len(b)
        if transposed:
            row, col = col, row
            a, b = b, a
        shape = (len(a), len(b) + len(a))
        p = np.full(shape, -np.inf)
        sim = np.zeros(shape)
        p[row, col] = probabilities[edges]
        sim[row, col] = similarities[edges]
        p[np.arange(len(a)), len(b) + np.arange(len(a))] = 0
        matched_rows, matched_cols = _assignment(p, sim, len(b), transposed)
        real = matched_cols < len(b)
        positions = np.full((len(a), len(b)), -1, dtype=np.int64)
        positions[row, col] = edges
        chosen.append(positions[matched_rows[real], matched_cols[real]])
    return np.concatenate(chosen)


def resolve(scores: pd.DataFrame, *, how: str = "one-to-one", threshold: float = 0.5,
            min_margin: float | None = None) -> pd.DataFrame:
    """Choose links under the requested cardinality rule, then apply the ambiguity margin.

    Margins use all judged pairs, including those below the threshold. Applying min_margin
    can remove links from the optimal assignment; the remaining pairs are not rematched.
    One-to-one ties maximize total sim, then prefer the earliest (left_id, right_id) pairs.
    """
    modes = ("many-to-many", "many-to-one", "one-to-many", "one-to-one")
    if how not in modes:
        raise ValueError(f"how must be one of {', '.join(modes)}; got {how!r}")
    threshold = _number(threshold, "threshold", probability=True)
    if min_margin is not None:
        min_margin = _number(min_margin, "min_margin")
    _pairs(scores, "scores")
    frame = scores.copy().reset_index(drop=True)
    frame["p"] = _probabilities(frame)
    frame["sim"] = _probabilities(frame, "sim", missing=False)
    frame["margin"] = _margin(frame)
    pairs = frame.loc[frame["p"].ge(threshold)]
    try:
        pairs = pairs.sort_values(["p", "sim", "left_id", "right_id"],
                                  ascending=[False, False, True, True], kind="stable")
    except TypeError as exc:
        raise ValueError("left_id and right_id must contain sortable IDs, "
                         "such as strings or integers") from exc
    if how == "many-to-one":
        pairs = pairs.drop_duplicates("left_id")
    elif how == "one-to-many":
        pairs = pairs.drop_duplicates("right_id")
    elif how == "one-to-one" and len(pairs):
        pairs = pairs.iloc[_one_to_one(pairs)]
    if min_margin is not None:
        pairs = pairs.loc[pairs["margin"].isna() | pairs["margin"].ge(min_margin)]
    return pairs.sort_values(["p", "left_id", "right_id"], ascending=[False, True, True],
                             kind="stable").reset_index(drop=True)

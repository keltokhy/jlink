"""Turn scored pairs within one table into clusters of records.

Connected components at a threshold chain records together: one wrong pair above the threshold
joins two groups of any size. The default here is average linkage in which every pair between two
clusters votes: the low probabilities that were also paid for, and the pairs that blocking found
too unlike to propose, count against a merge that one high pair suggests.
"""

from __future__ import annotations

import heapq
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .resolve import _integers, _number, _pairs, _probabilities

LINKAGES = ("average", "components")
UNPROPOSED = ("ignore", "nonmatch")


def cluster(scores: pd.DataFrame, *, ids=None, threshold: float = 0.5, linkage: str = "average",
            unproposed: str = "nonmatch") -> pd.DataFrame:
    """One row per record: ``id``, ``cluster_id`` and ``cluster_size``. Makes no API calls.

    ``scores`` holds each unordered pair once, as ``left_id``, ``right_id`` and ``p``. ``ids`` lists
    every record in table order, so records that no pair mentions become singletons; without it
    only records that appear in ``scores`` are returned, in order of first appearance.

    ``linkage="components"`` joins every pair with ``p >= threshold`` and everything reachable
    through such pairs. ``linkage="average"`` starts from single records and repeatedly merges the
    two clusters with the highest mean ``p`` between them, while that mean is at least
    ``threshold``. Candidate pairs without a probability (unjudged or failed) are never evidence.
    ``unproposed`` says what a pair that blocking never proposed counts as in that mean.
    ``"nonmatch"`` (the default) counts it as ``p = 0``: blocking found the two records too unlike
    to propose. One wrong pair then cannot join two groups, and a record joins a group only on
    evidence against enough of its members, so a true group that blocking covers only in part is
    split. ``"ignore"`` leaves such pairs out, so the mean is over judged pairs alone: partly
    covered groups stay whole, and two groups whose only judged pair is wrong merge on it.
    It applies to average linkage only. docs/dedupe.md measures both failure modes.

    Means are compared in exact integer arithmetic and ties go to the clusters that appear
    earliest in ``ids``, so the result does not depend on the row order of ``scores``.
    ``cluster_id`` numbers clusters by the first appearance of any member.
    """
    if linkage not in LINKAGES:
        raise ValueError(f"`linkage` must be one of {', '.join(LINKAGES)}; got {linkage!r}")
    if unproposed not in UNPROPOSED:
        raise ValueError(f"`unproposed` must be one of {', '.join(UNPROPOSED)}; got {unproposed!r}")
    threshold = _number(threshold, "threshold", probability=True)
    _pairs(scores, "scores")
    p = _probabilities(scores).to_numpy()
    try:
        if ids is None:
            records = pd.Index(pd.unique(scores[["left_id", "right_id"]].to_numpy().ravel(order="C")))
        else:
            records = pd.Index(ids)
        if records.has_duplicates or records.isna().any():
            raise ValueError("`ids` must list each record once, with no missing IDs")
        a, b = records.get_indexer(scores["left_id"]), records.get_indexer(scores["right_id"])
    except TypeError as exc:
        raise ValueError("record IDs must be scalar values, such as strings or integers") from exc
    if (a < 0).any() or (b < 0).any():
        absent = scores["left_id"].to_numpy()[a < 0] if (a < 0).any() else scores["right_id"].to_numpy()[b < 0]
        raise ValueError(f"scores name a record that is not in `ids`: {absent[0]!r}")
    if (a == b).any():
        raise ValueError(f"scores pair the record {scores['left_id'].to_numpy()[a == b][0]!r} with itself; "
                         "dedupe scores hold pairs of different records")
    low, high = np.minimum(a, b), np.maximum(a, b)
    if pd.MultiIndex.from_arrays([low, high]).has_duplicates:
        raise ValueError("scores list a pair in both orders; keep one row per unordered pair")

    n = len(records)
    if not n:
        return pd.DataFrame({"id": records, "cluster_id": np.empty(0, dtype=np.int64),
                             "cluster_size": np.empty(0, dtype=np.int64)})
    judged = np.isfinite(p)
    strong = judged & (np.nan_to_num(p) >= threshold)
    graph = coo_matrix((np.ones(int(strong.sum()), dtype=np.int8), (low[strong], high[strong])),
                       shape=(n, n))
    _, component = connected_components(graph, directed=False)
    labels = component.copy()
    if linkage == "average":
        # A merge needs a mean of at least the threshold, hence one pair at or above it, so every
        # cluster lies inside one component of the strong pairs. Work inside each component, with
        # all of its candidate pairs: the weak ones are the evidence against chaining.
        inside = np.flatnonzero(component[low] == component[high])
        inside = inside[np.argsort(component[low[inside]], kind="stable")]
        groups = np.split(inside, np.flatnonzero(np.diff(component[low[inside]])) + 1) if len(inside) else []
        sizes = np.bincount(component)
        weights = bar = None
        for edges in groups:
            # Nothing to decide where every pair supports the merge: all pairs strong, and under
            # "nonmatch" also none missing, which is a clique.
            members = int(sizes[component[low[edges[0]]]])
            if strong[edges].all() and (unproposed == "ignore" or 2 * len(edges) == members * (members - 1)):
                continue
            if weights is None:
                weights, _ = _integers(np.append(np.nan_to_num(p), threshold))
                weights, bar = weights[:-1], weights[-1]
            roots = _average_linkage(low[edges], high[edges], weights[edges], judged[edges], bar,
                                     count_unproposed=unproposed == "nonmatch")
            for node, root in roots.items():
                labels[node] = n + root  # distinct from every component label
    cluster_id, _ = pd.factorize(labels)
    sizes = np.bincount(cluster_id)[cluster_id]
    return pd.DataFrame({"id": records, "cluster_id": cluster_id.astype(np.int64),
                         "cluster_size": sizes.astype(np.int64)})


def _average_linkage(a: np.ndarray, b: np.ndarray, weights: np.ndarray, judged: np.ndarray, bar: int, *,
                     count_unproposed: bool) -> dict[int, int]:
    """Greedy average-linkage merges among the given nodes. Returns node -> earliest member of its cluster.

    Clusters are named by their earliest member. ``between[x][y]`` is (sum of p, judged pairs,
    candidate pairs without p) between clusters x and y, the sum in integer units shared with
    ``bar``. The mean divides by the judged pairs, or with ``count_unproposed`` by every pair
    between the two clusters except the candidates without p. A heap entry is stale once either
    cluster has merged since it was pushed, and is then skipped.
    """
    between: dict[int, dict[int, tuple[int, int, int]]] = {}
    for x, y, w, known in zip(a.tolist(), b.tolist(), weights.tolist(), judged.tolist()):
        entry = (w, 1, 0) if known else (0, 0, 1)
        between.setdefault(x, {})[y] = entry
        between.setdefault(y, {})[x] = entry
    members = {node: [node] for node in between}
    version = dict.fromkeys(between, 0)
    heap: list = []

    def offer(x: int, y: int) -> None:
        total, known, unknown = between[x][y]
        pairs = len(members[x]) * len(members[y]) - unknown if count_unproposed else known
        # Only a mean at or above the threshold can ever be merged, so weaker entries are never queued.
        if pairs > 0 and known and total >= bar * pairs:
            x, y = min(x, y), max(x, y)
            heapq.heappush(heap, (-Fraction(total, pairs), x, y, version[x], version[y]))

    for x, links in between.items():
        for y in links:
            if x < y:
                offer(x, y)
    while heap:
        _, x, y, seen_x, seen_y = heapq.heappop(heap)
        if version.get(x) != seen_x or version.get(y) != seen_y:
            continue
        # x < y, so the merged cluster keeps the name x: its earliest member.
        mine, theirs = between.pop(x), between.pop(y)
        del mine[y], theirs[x], version[y]
        for z, entry in theirs.items():
            seen = mine.get(z, (0, 0, 0))
            mine[z] = (seen[0] + entry[0], seen[1] + entry[1], seen[2] + entry[2])
        between[x] = mine
        members[x].extend(members.pop(y))
        version[x] += 1
        for z, entry in mine.items():
            between[z].pop(y, None)
            between[z][x] = entry
            offer(x, z)
    return {node: name for name, nodes in members.items() for node in nodes}

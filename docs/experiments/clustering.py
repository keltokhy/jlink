"""Offline clustering stress test. Run from the repository root with PYTHONPATH=src.

Synthetic scores only: no API client, no judging calls, no data files. Within a true group, each
record is scored high against the next `--reach` records of its group, taken in a ring; the default
reaches every other member, a clique. Each record also has random pairs with records of other groups,
scored low, except a share that is scored wrongly high. Because those pairs are random, a wrong pair
is almost always the only scored pair between its two groups, which is the hardest case for a rule
that weighs evidence. A small `--reach` is the opposite stress: true groups that blocking covers
only in part. See docs/dedupe.md for interpretation and recorded results.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import jlink


def corpus(records, size, cross, wrong, seed, reach):
    rng = np.random.default_rng(seed)
    group = np.arange(records) // size
    a = np.repeat(np.arange(records), reach)
    b = (a // size) * size + (np.tile(np.arange(1, reach + 1), records) + a % size) % size
    ring = pd.DataFrame({"left_id": np.minimum(a, b), "right_id": np.maximum(a, b)}).drop_duplicates()
    a, b, keep = ring.left_id.to_numpy(), ring.right_id.to_numpy(), np.ones(len(ring), dtype=bool)
    x = np.repeat(np.arange(records), cross)
    y = rng.integers(0, records, len(x))
    apart = group[x] != group[y]
    between = pd.DataFrame({"left_id": np.minimum(x, y)[apart], "right_id": np.maximum(x, y)[apart]})
    between = between.drop_duplicates(ignore_index=True)
    mistaken = rng.random(len(between)) < wrong
    between["p"] = np.where(mistaken, rng.uniform(0.6, 0.9, len(between)), rng.uniform(0, 0.2, len(between)))
    within = pd.DataFrame({"left_id": a[keep], "right_id": b[keep], "p": rng.uniform(0.8, 1, int(keep.sum()))})
    return pd.concat([within, between], ignore_index=True), group, int(mistaken.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=200_000)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--cross", type=int, default=10, help="random pairs with other groups, per record")
    parser.add_argument("--wrong", type=float, default=0.02, help="share of those pairs scored wrongly high")
    parser.add_argument("--reach", type=int, help="group members each record is scored against, in a ring "
                                                   "(default: all of them)")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.reach = args.group_size - 1 if args.reach is None else args.reach
    if not 1 <= args.reach < args.group_size:
        parser.error("--reach must be at least 1 and less than --group-size")
    scores, group, mistaken = corpus(args.records, args.group_size, args.cross, args.wrong, args.seed,
                                     args.reach)
    result = {"experiment": "clustering", **{k: v for k, v in vars(args).items() if k != "output"},
              "true_groups": args.records // args.group_size, "pairs": len(scores),
              "wrong_high_pairs": mistaken, "threshold": 0.5, "rules": []}
    for linkage, unproposed in (("components", "ignore"), ("average", "ignore"), ("average", "nonmatch")):
        begin = time.perf_counter()
        clusters = jlink.cluster(scores, ids=np.arange(args.records), linkage=linkage, unproposed=unproposed)
        seconds = time.perf_counter() - begin
        truth = clusters.assign(group=group).groupby("cluster_id").agg(
            groups=("group", "nunique"), size=("id", "size"))
        exact = int(((truth.groups == 1) & (truth["size"] == args.group_size)).sum())
        result["rules"].append({"linkage": linkage, "unproposed": unproposed, "seconds": seconds,
                                "clusters": len(truth), "largest_cluster": int(truth["size"].max()),
                                "true_groups_recovered_exactly": exact,
                                "records_in_mixed_clusters": int(truth.loc[truth.groups > 1, "size"].sum())})
    result["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["rules"], indent=1), flush=True)


if __name__ == "__main__":
    main()

"""Offline candidate experiments. Run from the repository root with PYTHONPATH=src.

No API client, judging calls, data downloads, or writes to the input directory.
See docs/blocking.md for commands, interpretation, and recorded results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from jlink import block
from jlink.fields import normalize


def peak_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024


def firms(root):
    files = [root / name for name in ("left.parquet", "right.parquet", "truth.parquet", "meta.json")]
    left, right, truth = [pd.read_parquet(p) for p in files[:3]]
    left["first_letter"] = left.name.map(normalize).str[:1]
    right["first_letter"] = right.name.map(normalize).str[:1]
    forward = lambda k=10: block.ngrams("name", k=k)
    reverse = lambda: block.ngrams("name", reverse=True)
    variants = {
        "forward_k10": [forward()],
        "reverse_k10": [reverse()],
        "symmetric_k10": [forward(), reverse()],
        "forward_k20": [forward(20)],
        "forward_k50": [forward(50)],
        "forward_k100": [forward(100)],
        "forward_k10_plus_initials": [forward(), block.initials("name")],
        "first_letter_k10": [block.within(forward(), "first_letter")],
        "first_letter_symmetric_k10": [block.within(p, "first_letter") for p in (forward(), reverse())],
    }
    results = []
    old_pairs = set()
    known = set(zip(truth.left_id, truth.right_id))
    for name, passes in variants.items():
        start = time.perf_counter()
        candidates = block.candidates(left, right, on="name", left_id="id", right_id="id", blockers=passes)
        elapsed = time.perf_counter() - start
        pairs = set(zip(candidates.left_id, candidates.right_id))
        if name == "forward_k10":
            old_pairs = pairs
        result = {"variant": name, "pairs": len(pairs), "truth_found": len(pairs & known),
                  "recall": block.pairs_completeness(candidates, truth), "seconds": elapsed,
                  "additional_truth_vs_forward_k10": len((pairs - old_pairs) & known),
                  "lost_truth_vs_forward_k10": len((old_pairs - pairs) & known),
                  "blocking": candidates.attrs["blocking"]}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "blocking"}), flush=True)
    return {"experiment": "firms", "left": len(left), "right": len(right), "truth": len(known),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "grouping": "Derived first normalized name character only; no state/year available. "
                        "No truth or ownership fields used in candidate generation.", "results": results}


def corpus(rows, groups):
    rng = np.random.default_rng(20260918)
    first = np.array("Acme Alder Alpine Atlas Beacon Blue Cedar Central Clear Coral Delta Eagle Elm Emerald "
                     "Fair First Golden Grand Green Harbor High Iron Lake Maple Metro North Oak Pacific "
                     "Pine Pioneer Prime Red River Silver South Summit Sun Union Valley West White".split(), dtype=object)
    second = np.array("Bridge Brook Coast Creek Crown Field Forest Gate Glen Grove Hill Horizon Island "
                      "Lake Land Mountain Ocean Park Peak Point Port Ridge Rock Shore Spring Stone "
                      "Stream Tower Trail Tree View Village Vista Water Way Wood".split(), dtype=object)
    industry = np.array("Advisors Aerospace Agriculture Analytics Builders Chemicals Communications "
                        "Construction Consulting Design Development Electric Energy Engineering Foods "
                        "Healthcare Holdings Industries Insurance Logistics Manufacturing Materials "
                        "Media Mining Motors Networks Partners Properties Research Retail Services "
                        "Software Solutions Systems Technologies Trading Transport Ventures".split(), dtype=object)
    suffix = np.array(["Inc", "Corp", "Company", "LLC", "Ltd", "Group", "Partners", ""], dtype=object)
    names = rng.choice(first, rows) + " " + rng.choice(second, rows) + " " + rng.choice(industry, rows)
    left_names = names + " " + rng.choice(suffix, rows)
    right_names = left_names.copy()
    right_names[::10] = names[::10] + " Limited"
    for i in range(1, rows, 10):
        words = str(right_names[i]).split()
        right_names[i] = words[0][0] + " " + " ".join(words[1:])
    state = np.arange(rows) % groups
    permutation = rng.permutation(rows)
    left = pd.DataFrame({"id": np.arange(rows), "name": left_names, "state": state})
    right = pd.DataFrame({"id": permutation, "name": right_names[permutation], "state": state[permutation]})
    return left, right


def scale(rows, groups, symmetric):
    left, right = corpus(rows, groups)
    passes = [block.ngrams("name")]
    if symmetric:
        passes.append(block.ngrams("name", reverse=True))
    if groups > 1:
        passes = [block.within(p, "state") for p in passes]
    start = time.perf_counter()
    candidates = block.candidates(left, right, on="name", left_id="id", right_id="id", blockers=passes)
    elapsed = time.perf_counter() - start
    found = int((candidates.left_id == candidates.right_id).sum())
    return {"experiment": "scale", "rows_per_side": rows, "groups": groups, "symmetric": symmetric,
            "pairs": len(candidates), "seconds": elapsed, "synthetic_truth_found": found,
            "synthetic_recall": found / rows, "peak_rss_mib": peak_mib(), "seed": 20260918,
            "blocking": candidates.attrs["blocking"]}


def guard(rows):
    frame = pd.DataFrame({"name": ["Acme"] * rows, "state": ["NY"] * rows})
    start = time.perf_counter()
    try:
        block.candidates(frame, frame, on="name", blockers=[block.within(block.exact("name"), "state")],
                         max_pairs=1000)
    except ValueError as error:
        return {"experiment": "guard", "rows_per_side": rows, "possible_exact_pairs": rows**2,
                "max_pairs": 1000, "error": str(error), "seconds": time.perf_counter() - start,
                "peak_rss_mib": peak_mib()}
    raise AssertionError("guard did not reject oversized exact group")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("firms", "scale", "guard"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--groups", type=int, default=50)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1 or args.groups < 1:
        parser.error("rows and groups must be positive")
    if args.experiment == "firms":
        if args.data_root is None:
            parser.error("firms needs --data-root")
        result = firms(args.data_root)
    elif args.experiment == "guard":
        result = guard(args.rows)
    else:
        result = scale(args.rows, args.groups, args.symmetric)
    result["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "packages": {p: importlib.metadata.version(p) for p in
                                          ("numpy", "pandas", "scipy", "scikit-learn")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("results", "blocking")}), flush=True)


if __name__ == "__main__":
    main()

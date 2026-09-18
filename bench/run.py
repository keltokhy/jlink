"""Run offline baselines, available jlink blocking, and an explicitly opted-in live pass."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.baselines import exact_match, jaro_winkler, score_against_truth, tfidf_cosine
from bench.data import DATA, NAMES, ROOT, load_dataset, sample_left


def _optional_module(name: str, required: tuple[str, ...]):
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return None
    return module if all(callable(getattr(module, attr, None)) for attr in required) else None


def _blocking(left, right, truth, meta):
    module = _optional_module("jlink.block", ("candidates", "ngrams"))
    if module is None:
        return {"status": "unavailable", "reason": "jlink.block is not implemented on this branch"}, None
    start = perf_counter()
    candidates = module.candidates(left, right, on=meta["on"], left_id="id", right_id="id",
                                   blockers=[module.ngrams(*meta["on"], k=10)])
    metrics = score_against_truth(candidates, truth, candidates=candidates)
    return {"status": "ok", "candidate_count": len(candidates), "k": 10,
            "pairs_completeness": metrics["pairs_completeness"], "seconds": perf_counter() - start,
            "estimated_cost_usd": len(candidates) * 330 * 0.042 / 1_000_000,
            "estimated_seconds_at_200_pairs_per_second": len(candidates) / 200,
            "estimate_assumptions": "330 input tokens/pair, $0.042/million input tokens; "
                                    "not a quote"}, candidates


def _live(left, right, truth, meta, budget):
    # No judge, linker or credential-handling imports happen on an offline run.
    module = _optional_module("jlink.linker", ("Linker",))
    if module is None:
        raise ValueError("--live requires jlink.linker; merge the pipeline implementation first")
    block = _optional_module("jlink.block", ("ngrams",))
    if block is None:
        raise ValueError("--live requires jlink.block; merge the blocking implementation first")
    start = perf_counter()
    linker = module.Linker(entity=meta["entity"], definition=meta["definition"], on=meta["on"],
                           blockers=[block.ngrams(*meta["on"], k=10)])
    result = linker.link(left, right, left_id="id", right_id="id", how=meta["how"],
                         threshold=0.5, min_margin=0.0, budget=budget)
    audit = _optional_module("jlink.audit", ("score_against_truth",))
    scorer = audit.score_against_truth if audit is not None else score_against_truth
    return {"status": "ok", "metrics": scorer(result.links, truth, candidates=result.candidates),
            "seconds": perf_counter() - start, "budget_usd": budget,
            "meter": {key: getattr(result.meter, key)
                      for key in ("calls", "cached", "retries", "input_tokens", "cost")},
            "threshold": 0.5, "how": meta["how"], "min_margin": 0.0}


def run(dataset: str, *, budget: float = 0.10, sample: int | None = None, live: bool = False,
        data_dir: Path = DATA, out_dir: Path = ROOT / "out") -> dict:
    """Run one dataset; offline mode never initializes the API client or reads credentials."""
    if (isinstance(budget, bool) or not isinstance(budget, (int, float))
            or not math.isfinite(budget) or budget < 0):
        raise ValueError("--budget must be a finite, nonnegative dollar amount")
    left, right, truth, meta = load_dataset(dataset, data_dir)
    full_counts = {"left": len(left), "right": len(right), "truth": len(truth)}
    left, truth = sample_left(left, truth, sample)
    start = perf_counter()
    results = {}
    methods = (("exact", exact_match), ("jaro_winkler", jaro_winkler), ("tfidf_cosine", tfidf_cosine))
    for name, method in methods:
        before = perf_counter()
        result = method(left, right, truth, on=meta["on"], left_id="id", right_id="id")
        results[name] = dict(result.summary(), seconds=perf_counter() - before)
    blocking, _ = _blocking(left, right, truth, meta)
    report = {
        "dataset": dataset, "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": meta["definition"], "on": meta["on"], "meta": meta,
        "source_counts": full_counts,
        "evaluated_counts": {"left": len(left), "right": len(right), "truth": len(truth)},
        "sample": {"requested_left_records": sample, "seed": 0,
                   "left_ids_sha256": hashlib.sha256(json.dumps(left.id.tolist()).encode()).hexdigest(),
                   "right_universe": "all records"},
        "baseline_policy": "Exact emits every equal pair. Similarity baselines select "
                           "one best right per left "
                           "(ties: earliest right source row) then maximize F1 on this same truth. These are "
                           "optimistic threshold upper bounds, not held-out estimates. "
                           "No one-to-one constraint.",
        "baselines": results, "blocking": blocking,
        "live": {"status": "not_requested", "api_calls": 0},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name)
                                     for name in ("numpy", "pandas", "scipy", "scikit-learn", "jellyfish")}},
    }
    if len(left) * 10 > 60_000:
        report["sampling_advice"] = ("Use --sample 500 for a small live pass; "
                                    "k=10 can propose over 60,000 pairs.")
    if live:
        report["live"] = _live(left, right, truth, meta, budget)
    report["seconds"] = perf_counter() - start
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{dataset}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=NAMES, required=True)
    parser.add_argument("--budget", type=float, default=0.10, help="live-pass dollar cap; default: 0.10")
    parser.add_argument("--sample", type=int,
                        help="sample N left records with seed 0; keep all right records")
    parser.add_argument("--live", action="store_true", help="explicitly authorize real API calls")
    args = parser.parse_args(argv)
    try:
        report = run(args.dataset, budget=args.budget, sample=args.sample, live=args.live)
    except (ValueError, OSError, ImportError) as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"dataset": args.dataset, "counts": report["evaluated_counts"],
                      "seconds": report["seconds"], "blocking": report["blocking"], "live": report["live"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

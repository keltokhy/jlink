"""Compare lexical, semantic, hybrid and optional LinkTransformer retrieval on frozen splits.

No paid calls unless --live is supplied. Optional --judge reuses Jev's cache at budget=0.
Each split is judged once on the union, isolating retrieval from differences in judgments.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data import DATA, NAMES, ROOT, PAIR_COLUMNS, load_dataset, pair_set
from bench.evaluation import entity_split, evaluate_stages, subset, validate_pairs
from bench.heldout import file_hash, write_json
from bench.live import RULES
from jlink import block
from jlink.embeddings import DEFAULT_MODEL
from jlink.fields import parse_on
from jlink.judge import judge, validate_budget


def union_candidates(*tables: pd.DataFrame) -> pd.DataFrame:
    """Keep lexical similarity unchanged and retain every contributing pass name."""
    rows = pd.concat(tables, ignore_index=True)
    return rows.groupby(PAIR_COLUMNS, sort=False, as_index=False).agg(
        sim=("sim", "first"), block=("block", lambda x: "+".join(dict.fromkeys(x))))


def retrieval_metrics(candidates, truth, reference=None):
    pairs, gold = pair_set(candidates), pair_set(truth)
    found = pairs & gold
    result = {"pairs": len(pairs), "known_matches_found": len(found), "known_matches": len(gold),
              "candidate_recall": len(found) / len(gold) if gold else None}
    if reference is not None:
        old = pair_set(reference)
        result.update(added_pairs=len(pairs - old), recovered_matches=len(found - old),
                      lost_matches=len((old & gold) - pairs))
    return result


def linktransformer_candidates(python, directory, left, right, on, model, revision, k):
    """Run the real optional package in its own environment, keeping its dependencies isolated."""
    directory.mkdir()
    left.to_parquet(directory / "left.parquet", index=False)
    right.to_parquet(directory / "right.parquet", index=False)
    write_json(directory / "request.json", {"on": on, "model": model, "revision": revision, "k": k})
    with (directory / "process.log").open("w") as log:
        process = subprocess.run([str(python), str(ROOT / "linktransformer_retrieval.py"), str(directory)],
                                 stdout=log, stderr=subprocess.STDOUT, check=False, timeout=1800,
                                 env=dict(os.environ, OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false"))
    if process.returncode:
        raise ValueError(f"LinkTransformer retrieval exited {process.returncode}; "
                         f"see {directory / 'process.log'}")
    result = pd.read_csv(directory / "candidates.csv", dtype={c: str for c in PAIR_COLUMNS},
                         keep_default_na=False)
    validate_pairs(result, left, right, "LinkTransformer candidates")
    return result, json.loads((directory / "metadata.json").read_text())


def run(dataset, *, out, model=DEFAULT_MODEL, revision=None, k=10, seed=1729, data_dir=DATA,
        judge_pairs=False, live=False, budget=0.0, lt_python=None, encoder=None, transport=None):
    validate_budget(budget)
    if not live and budget != 0:
        raise ValueError("a positive budget requires --live; offline runs never buy judgments")
    if live and (budget is None or budget <= 0):
        raise ValueError("--live requires a finite positive total budget")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} already contains files; choose a new output directory")
    left, right, truth, meta = load_dataset(dataset, data_dir)
    left, right = left.assign(id=left.id.astype(str)), right.assign(id=right.id.astype(str))
    truth = truth.astype({c: str for c in PAIR_COLUMNS})
    fields = [(lc, rc) for _, lc, rc in parse_on(meta["on"])]
    semantic = block.embeddings(*fields, k=k, model=model, revision=revision, encoder=encoder)
    splits = entity_split(left, right, truth, on=meta["on"], seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out / "split.csv", index=False)
    report = {"schema_version": 1, "dataset": dataset, "seed": seed, "k": k,
              "split": "30% development / 70% test; disjoint observed entities",
              "prior_exposure": "existing public benchmark; not an untouched external test",
              "label_policy": "positive-only crosswalk; unlisted pairs unknown" if dataset == "nber-firms"
              else "supplied benchmark truth treated as complete",
              "definition": RULES[dataset], "exact_shortcut": False,
              "judge_status": "live" if live else "cache_only" if judge_pairs else "not_run",
              "linktransformer": "retrieval only; its LLM judge was not run" if lt_python else "not_run",
              "budget_usd": budget, "new_api_calls": 0, "new_api_cost_usd": 0.0,
              "source_hashes": {name: file_hash(Path(data_dir) / dataset / name) for name in
                                ("left.parquet", "right.parquet", "truth.parquet", "meta.json")},
              "partitions": {}}
    for partition in ("dev", "test"):
        a, b, gold = subset(left, right, truth, splits, partition)
        if a.empty or b.empty or gold.empty:
            raise ValueError(f"{partition} has no records or known links; use a different split seed")
        records = {"left": len(a), "right": len(b), "truth": len(gold)}
        pools, timings = {}, {}
        for name, passes in (("lexical", [block.ngrams(*fields, k=k)]), ("semantic", [semantic])):
            start = time.perf_counter()
            pools[name] = block.candidates(a, b, on=meta["on"], blockers=passes, left_id="id", right_id="id")
            timings[name] = time.perf_counter() - start
        pools["hybrid"] = union_candidates(pools["lexical"], pools["semantic"])
        entry = {"records": records, "retrieval_seconds": timings, "retrieval": {}}
        if lt_python:
            external, metadata = linktransformer_candidates(
                lt_python, out / f"{partition}-linktransformer", a, b, meta["on"], model, revision, k)
            # Reuse jlink's candidate schema and lexical score for all downstream resolvers.
            class External(block.Blocker):
                name = "linktransformer"

                def pairs(self, left, right):
                    ia, ib = pd.Index(left.id), pd.Index(right.id)
                    import numpy as np
                    return np.column_stack((ia.get_indexer(external.left_id), ib.get_indexer(external.right_id)))

            pools["linktransformer"] = block.candidates(a, b, on=meta["on"], blockers=[External()],
                                                       left_id="id", right_id="id")
            entry["linktransformer_metadata"] = metadata
        for name, candidates in pools.items():
            entry["retrieval"][name] = retrieval_metrics(candidates, gold, pools["lexical"])
            candidates.to_csv(out / f"{partition}-{name}-candidates.csv", index=False)
        if judge_pairs or live:
            combined = union_candidates(*pools.values())
            scores, meter = judge(combined, a, b, on=meta["on"], entity=RULES[dataset]["entity"],
                                  definition=RULES[dataset]["definition"], left_id="id", right_id="id",
                                  budget=max(0, budget - report["new_api_cost_usd"]), concurrency=8,
                                  progress=False, transport=transport, exact_shortcut=False)
            scores.to_csv(out / f"{partition}-union-scores.csv", index=False)
            report["new_api_calls"] += meter.calls
            report["new_api_cost_usd"] += meter.cost
            entry["judgments"] = {"calls": meter.calls, "cached": meter.cached, "cost": meter.cost,
                                   "resolved_models": meter.resolved_models,
                                   "unjudged": int(scores.p.isna().sum())}
            entry["same_judge_comparison"] = {}
            for name, candidates in pools.items():
                selected = candidates.merge(scores.drop(columns=["sim", "block"]), on=PAIR_COLUMNS,
                                             validate="one_to_one")
                metrics, links = evaluate_stages(candidates, selected, gold, how=RULES[dataset]["how"],
                                                threshold=.5, complete=dataset != "nber-firms")
                entry["same_judge_comparison"][name] = metrics
                links.to_csv(out / f"{partition}-{name}-links.csv", index=False)
        report["partitions"][partition] = entry
        write_json(out / "progress.json", report)
    report["embedding_config"] = semantic.to_config()
    report["environment"] = {"python": sys.version, "packages": {
        p: importlib.metadata.version(p) for p in ("jlink", "numpy", "pandas", "scikit-learn")}}
    if encoder is None:
        report["environment"]["packages"]["sentence-transformers"] = importlib.metadata.version(
            "sentence-transformers")
    report["code_hashes"] = {str(p.relative_to(ROOT.parent)): file_hash(p) for p in
                             [Path(__file__), ROOT / "linktransformer_retrieval.py",
                              ROOT / "evaluation.py", ROOT.parent / "src/jlink/embeddings.py",
                              ROOT.parent / "src/jlink/block.py", ROOT.parent / "src/jlink/judge.py"]}
    report["artifacts"] = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*"))
                           if p.is_file() and p.name != "progress.json"}
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=NAMES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True, help="pin the embedding Hub commit or tag")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--judge", action="store_true", dest="judge_pairs")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--budget", type=float, default=0)
    parser.add_argument("--lt-python", type=Path, help="Python with LinkTransformer and pyarrow installed")
    args = vars(parser.parse_args())
    report = run(**args)
    print(json.dumps({"output": str(args["out"]), "judge_status": report["judge_status"],
                      "test_retrieval": report["partitions"]["test"]["retrieval"]}, indent=2))


if __name__ == "__main__":
    main()

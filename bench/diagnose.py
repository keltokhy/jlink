"""Audit a saved hybrid experiment offline; never request new model scores.

Usage: python -m bench.diagnose abt-buy --run bench/out/hybrid-20260920-final/abt-buy --out report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bench.data import DATA, PAIR_COLUMNS, load_dataset, pair_set
from bench.evaluation import choose_threshold, pair_metrics, subset, validate_pairs, validate_scores
from bench.heldout import file_hash, write_json
from jlink.fields import parse_on
from jlink.judge import _records
from jlink.resolve import resolve


def diagnose_pairs(scores, truth, left, right, *, on, threshold=.5):
    """Separate retrieval/judgment errors and find identical inputs with conflicting gold labels.

    Requires complete benchmark labels. Conflicts are relative to the exact fields supplied
    to the pair judge, not a claim that the source data or real entities are identical.
    """
    validate_pairs(scores, left, right, "scores")
    validate_pairs(truth, left, right, "truth")
    validate_scores(scores)
    gold, proposed = pair_set(truth), pair_set(scores)
    selected = pair_set(scores.loc[scores.p.ge(threshold)])
    fields = parse_on(on)
    signatures = []
    for frame, position in ((left, 1), (right, 2)):
        records = _records(frame, pd.Index(frame.id), [(f[0], f[position]) for f in fields]).record
        signatures.append(records.map(lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False)).to_dict())
    groups = {}
    for a, b in scores[PAIR_COLUMNS].itertuples(index=False, name=None):
        groups.setdefault((signatures[0][a], signatures[1][b]), []).append(
            {"left_id": a, "right_id": b, "gold_match": (a, b) in gold})
    mixed = [rows for rows in groups.values() if len({r["gold_match"] for r in rows}) > 1]
    missed = gold - selected
    unjudged = pair_set(scores.loc[scores.p.isna()])
    return {
        "gold_links": len(gold), "candidate_pairs": len(proposed),
        "retrieval_misses": len(gold - proposed),
        "unjudged_gold_candidates": len(gold & unjudged),
        "judged_false_negatives": len((missed & proposed) - unjudged),
        "false_positives": len(selected - gold), "final_missed_links": len(missed),
        "identical_judge_inputs_with_mixed_labels": len(mixed),
        "pairs_in_mixed_groups": sum(map(len, mixed)),
        "mixed_groups": mixed,
    }


def run(dataset, *, saved_run, out, data_dir=DATA):
    if dataset == "nber-firms":
        raise ValueError("NBER has positive-only labels; use retrieval recall, not this complete-label audit")
    saved_run, out = Path(saved_run), Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; choose a new output file")
    left, right, truth, meta = load_dataset(dataset, data_dir)
    left, right = left.assign(id=left.id.astype(str)), right.assign(id=right.id.astype(str))
    truth = truth.astype({c: str for c in PAIR_COLUMNS})
    if meta["how"] != "many-to-many":
        raise ValueError("this pair-decision audit requires a many-to-many benchmark")
    split = pd.read_csv(saved_run / "split.csv", dtype=str, keep_default_na=False)
    inputs = [saved_run / "split.csv"]
    partitions = {}
    for part in ("dev", "test"):
        a, b, gold = subset(left, right, truth, split, part)
        path = saved_run / f"{part}-union-scores.csv"
        inputs.append(path)
        scores = pd.read_csv(path, dtype={c: str for c in PAIR_COLUMNS}, keep_default_na=False,
                             na_values={"p": [""]}, float_precision="round_trip")
        partitions[part] = (a, b, gold, scores)
    report = {"dataset": dataset, "new_api_calls": 0, "new_api_cost_usd": 0,
              "label_policy": "supplied benchmark truth treated as complete",
              "prior_exposure": "exploratory; test results previously inspected",
              "threshold_policy": "maximize development F1 on 0.05 to 0.95 by 0.05; frozen on test",
              "methods": {}}
    for method in ("lexical", "hybrid"):
        tables = {}
        for part, (a, b, gold, scores) in partitions.items():
            path = saved_run / f"{part}-{method}-candidates.csv"
            inputs.append(path)
            pool = pd.read_csv(path, dtype={c: str for c in PAIR_COLUMNS}, keep_default_na=False)
            validate_pairs(pool, a, b)
            table = pool[PAIR_COLUMNS].merge(scores, on=PAIR_COLUMNS, how="left", validate="one_to_one",
                                            indicator=True)
            if not table._merge.eq("both").all() or table.p.isna().any():
                raise ValueError("complete stored judgments required for threshold comparison")
            tables[part] = table.drop(columns="_merge")
        threshold, trials = choose_threshold(tables["dev"], partitions["dev"][2],
                                            how="many-to-many", grid=[i / 100 for i in range(5, 100, 5)])
        entry = {"selected_threshold": threshold, "development_trials": trials, "partitions": {}}
        for part, (a, b, gold, _) in partitions.items():
            scores = tables[part]
            baseline = resolve(scores, how="many-to-many", threshold=.5)
            tuned = scores.iloc[:0] if threshold is None else resolve(
                scores, how="many-to-many", threshold=threshold)
            entry["partitions"][part] = {
                "at_0.5": pair_metrics(baseline, gold, complete=True),
                "at_development_threshold": pair_metrics(tuned, gold, complete=True),
                "diagnosis_at_0.5": diagnose_pairs(scores, gold, a, b, on=meta["on"]),
            }
        report["methods"][method] = entry
    report["input_sha256"] = {str(p): file_hash(p) for p in inputs}
    report["data_sha256"] = {name: file_hash(Path(data_dir) / dataset / name) for name in
                             ("left.parquet", "right.parquet", "truth.parquet", "meta.json")}
    report["code_sha256"] = {str(p): file_hash(p) for p in
                             (Path(__file__), Path(__file__).with_name("evaluation.py"))}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--run", type=Path, required=True, dest="saved_run")
    parser.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args())
    run(**args)
    print(args["out"])

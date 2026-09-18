"""The live benchmark pass: real Jev calls, capped per dataset, scored against known matches.

    uv run --group bench python bench/live.py nber-firms --budget 1.00 [--sample 300] [--tag pilot]

Writes bench/out/live/<dataset>[-tag]/ (links.csv, scores.csv, settings.json, result.json).
Only the project lead runs this; it spends money.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import jlink

HERE = Path(__file__).parent

# Match rules written for the judge: what a careful research assistant would be told.
# meta.json's definitions describe each benchmark; these say how to decide from the fields.
RULES = {
    "nber-firms": dict(
        entity="firm", how="many-to-one",
        # Revised once after reading the errors of a 300-record pilot, as a user would: the first draft did
        # not say that CPY means Company in these data, and made the judge too wary of Inc versus Corp.
        definition="Ignore legal-form suffixes (Inc, Corp, Co, CPY, Ltd, GmbH, N V). A subsidiary or division "
                   "counts as its parent company."),
    "febrl4": dict(
        entity="person", how="one-to-one",
        definition="Both records come from forms filled in by hand, so expect typing errors, swapped or "
                   "missing fields and abbreviations. Sharing only a surname or only an address does not "
                   "establish a match."),
    "dblp-acm": dict(
        entity="publication", how="one-to-one",
        definition="The same paper listed in two bibliographic databases. Author lists may be abbreviated "
                   "and venue names may differ. A different paper by the same authors is not a match."),
    "abt-buy": dict(
        entity="product", how="many-to-many",
        definition="Two listings match when they offer the same product model. The same product line in a "
                   "different size, color, capacity, software version or license bundle is not a match."),
    "amazon-google": dict(
        entity="product", how="many-to-many",
        definition="Two listings match when they offer the same software product and edition. A different "
                   "version, platform, license type or bundle is not a match."),
}


def load(name: str, sample: int | None):
    d = HERE / "data" / name
    left, right, truth = (pd.read_parquet(d / f"{t}.parquet") for t in ("left", "right", "truth"))
    meta = json.loads((d / "meta.json").read_text())
    if sample and sample < len(left):
        left = left.sample(sample, random_state=0).sort_index()
        truth = truth[truth["left_id"].isin(left["id"])]
    return left, right, truth, meta


def sweep(scores: pd.DataFrame, truth: pd.DataFrame, how: str) -> pd.DataFrame:
    rows = []
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            links = jlink.resolve(scores, how=how, threshold=t)
        s = jlink.score_against_truth(links, truth)
        rows.append({"threshold": t, "links": len(links), **{k: round(s[k], 4) for k in ("precision", "recall", "f1")}})
    return pd.DataFrame(rows)


def calibration(scores: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    judged = scores[scores["source"] == "jev"].copy()
    true = set(zip(truth["left_id"], truth["right_id"]))
    judged["match"] = [pair in true for pair in zip(judged["left_id"], judged["right_id"])]
    bins = pd.cut(judged["p"], [0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0], include_lowest=True)
    out = judged.groupby(bins, observed=True).agg(pairs=("p", "size"), mean_p=("p", "mean"), match_rate=("match", "mean"))
    return out.round(3).reset_index(names="bin").astype({"bin": str})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(RULES))
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--definition", help="override the judge's rule (for wording experiments)")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("-j", type=int, default=64)
    a = ap.parse_args()

    left, right, truth, meta = load(a.dataset, a.sample)
    rule = dict(RULES[a.dataset])
    if a.definition is not None:
        rule["definition"] = a.definition
    on = meta["on"]
    blockers = [jlink.block.ngrams(*on, k=a.k)]
    linker = jlink.Linker(rule["entity"], on, rule["definition"], blockers, concurrency=a.j)
    t0 = time.perf_counter()
    result = linker.link(left, right, left_id="id", right_id="id", how=rule["how"], budget=a.budget)
    seconds = time.perf_counter() - t0

    out = HERE / "out" / "live" / (a.dataset + (f"-{a.tag}" if a.tag else ""))
    result.save(out)
    at_half = jlink.score_against_truth(result.links, truth, result.candidates)
    table = sweep(result.scores, truth, rule["how"])
    best = table.loc[table["f1"].idxmax()]
    lat = np.array(result.meter.latencies) * 1000 if result.meter.latencies else np.array([np.nan])
    summary = {
        "dataset": a.dataset, "sample": a.sample, "rule": rule, "on": on, "k": a.k,
        "left": len(left), "right": len(right), "truth": len(truth), "pairs": len(result.scores),
        "sources": result.scores["source"].value_counts().to_dict(),
        "at_0.5": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in at_half.items()},
        "best_f1_in_sample": best.to_dict(), "sweep": table.to_dict("records"),
        "calibration": calibration(result.scores, truth).to_dict("records"),
        "calls": result.meter.calls, "cached": result.meter.cached, "retries": result.meter.retries,
        "input_tokens": result.meter.input_tokens, "dollars": round(result.meter.cost, 4),
        "tokens_per_call": round(result.meter.input_tokens / max(result.meter.calls, 1), 1),
        "seconds": round(seconds, 1), "median_latency_ms": round(float(np.median(lat)), 0),
        "model": result.meter.model,
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2, default=str))
    print(result.report())
    print(f"\nagainst truth at 0.5: {summary['at_0.5']}")
    print(table.to_string(index=False))
    print("\ncalibration (judged pairs):\n" + calibration(result.scores, truth).to_string(index=False))


if __name__ == "__main__":
    main()

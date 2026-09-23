"""The live benchmark pass: real Jev calls, capped per dataset, scored against known matches.

    uv run --group bench python bench/live.py nber-firms --budget 1.00 [--sample 300] [--tag pilot]

Writes bench/out/live/<dataset>[-tag]/ (links.csv, scores.csv, settings.json, result.json).
Only the project lead runs this against a hosted provider; it spends money.

A local decision server, on a sample of left records whose candidate pairs are exactly the full
run's (blocking sees every left record, then keeps the sampled ones), with the full run's exact
shortcut, a per-request deadline and an empty answer cache:

    XDG_CACHE_HOME=bench/out/local-2026-09-22/cache-diffusiongemma-nber-firms \\
      uv run --group bench python bench/live.py nber-firms --api diffusiongemma --model openjev-0.1 \\
      --budget 0.01 --sample 300 --block-all-left --exact-shortcut -j 4 --timeout 300 \\
      --require-empty-cache --tag diffusiongemma-s300

    XDG_CACHE_HOME=bench/out/local-2026-09-22/cache-laya-nber-firms \\
      uv run --group bench python bench/live.py nber-firms --api laya --model laya-421m \\
      --budget 0.01 --sample 300 --block-all-left --exact-shortcut -j 4 --timeout 120 \\
      --require-empty-cache --laya-audit /path/to/laya-audit.jsonl --tag laya-s300

Local servers cost $0, so the budget never binds; it is a tripwire, not a cap. Do not pass
--budget 0: a zero budget allows no new requests at all and every pair is left unjudged.
bench/local_models.py then puts Jev's saved scores for the same pairs beside each local run.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import jlink
from jevkit_runtime import Client

HERE = Path(__file__).parent
NEW_YORK = ZoneInfo("America/New_York")


class AllLeft(jlink.block.Blocker):
    """Block with every left record, then keep the sampled ones' pairs.

    N-gram TF-IDF is fitted on the tables it is given, so blocking a sample alone proposes a few
    percent different pairs than the full run did. This pass proposes exactly the full run's pairs.
    """

    def __init__(self, inner: jlink.block.Blocker, full_left: pd.DataFrame):
        self.inner, self.full_left, self.name = inner, full_left, inner.name

    def pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
        pairs = self.inner.pairs(self.full_left, right)
        where = pd.Index(left["id"]).get_indexer(self.full_left["id"].to_numpy()[pairs[:, 0]])
        keep = where >= 0
        return np.column_stack((where[keep], pairs[keep, 1])).astype(np.int64)

    def to_config(self) -> dict:
        return self.inner.to_config() | {"blocked_left_records": len(self.full_left), "kept": "sampled left ids"}


def audit_count(path: Path | None) -> dict:
    """Lines in the Laya server's audit log, by status. Diff two counts to get one run's requests."""
    if path is None or not path.exists():
        return {}
    counts: dict = {"lines": 0}
    for line in path.read_text().splitlines():
        if line.strip():
            status = json.loads(line).get("status", "unknown")
            counts["lines"] += 1
            counts[status] = counts.get(status, 0) + 1
    return counts


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


def load(name: str, sample: int | None, *, full: bool = False):
    d = HERE / "data" / name
    left, right, truth = (pd.read_parquet(d / f"{t}.parquet") for t in ("left", "right", "truth"))
    meta = json.loads((d / "meta.json").read_text())
    everyone = left
    if sample and sample < len(left):
        left = left.sample(sample, random_state=0).sort_index()
        truth = truth[truth["left_id"].isin(left["id"])]
    return (left, right, truth, meta, everyone) if full else (left, right, truth, meta)


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
    ap.add_argument("--api", help="provider name (typesafe, openrouter, diffusiongemma, laya); default: JEV_API, "
                                  "then the first hosted provider with a key")
    ap.add_argument("--model", help="model name sent to the provider; default: JEV_MODEL, then the provider's")
    ap.add_argument("--timeout", type=float, help="deadline per request in seconds, retries included "
                                                  "(the runtime's default is 15)")
    ap.add_argument("--exact-shortcut", action="store_true",
                    help="accept pairs whose fields are all equal without asking, as jlink 0.1.0 always did "
                         "(the 2026-09-18 full runs)")
    ap.add_argument("--block-all-left", action="store_true",
                    help="with --sample: block with every left record, so the sample's pairs are the full run's")
    ap.add_argument("--require-empty-cache", action="store_true",
                    help="refuse to start unless XDG_CACHE_HOME is set and holds no answer cache yet")
    ap.add_argument("--laya-audit", type=Path, help="the Laya server's --audit file; its new lines are counted")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing result directory")
    a = ap.parse_args()

    out = HERE / "out" / "live" / (a.dataset + (f"-{a.tag}" if a.tag else ""))
    if (out / "result.json").exists() and not a.overwrite:
        sys.exit(f"{out} already holds a result; choose another --tag or pass --overwrite")
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if a.require_empty_cache and (not cache_home or (Path(cache_home) / "jev").exists()):
        sys.exit(f"--require-empty-cache: set XDG_CACHE_HOME to a new directory (now {cache_home!r})")
    if a.block_all_left and not a.sample:
        sys.exit("--block-all-left only means something with --sample")
    if a.timeout is not None:
        # jlink does not expose the request deadline, so its judge gets a client with this one.
        sys.modules["jlink.judge"].Client = functools.partial(Client, timeout=a.timeout)

    left, right, truth, meta, everyone = load(a.dataset, a.sample, full=True)
    rule = dict(RULES[a.dataset])
    if a.definition is not None:
        rule["definition"] = a.definition
    on = meta["on"]
    blockers = [jlink.block.ngrams(*on, k=a.k)]
    if a.block_all_left:
        blockers = [AllLeft(blockers[0], everyone)]
    linker = jlink.Linker(rule["entity"], on, rule["definition"], blockers, concurrency=a.j,
                          api=a.api, model=a.model, exact_shortcut=a.exact_shortcut)
    audit_before = audit_count(a.laya_audit)
    started = datetime.now(NEW_YORK)
    t0 = time.perf_counter()
    result = linker.link(left, right, left_id="id", right_id="id", how=rule["how"], budget=a.budget)
    seconds = time.perf_counter() - t0
    finished = datetime.now(NEW_YORK)
    audit_after = audit_count(a.laya_audit)

    result.save(out)
    at_half = jlink.score_against_truth(result.links, truth, result.candidates)
    table = sweep(result.scores, truth, rule["how"])
    best = table.loc[table["f1"].idxmax()]
    # jevkit-runtime's Meter no longer records latencies; older runs had them.
    latencies = getattr(result.meter, "latencies", None)
    lat = np.array(latencies) * 1000 if latencies else np.array([np.nan])
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
        "seconds": round(seconds, 1), "median_latency_ms": None if np.isnan(lat).all() else round(float(np.median(lat)), 0),
        "model": result.meter.model,
    }
    errors = result.scores.loc[result.scores["source"] == "error", "error"].astype(str)
    summary |= {
        "api": result.settings["provider"], "requested_model": result.settings["request_model"],
        "resolved_models": result.settings["resolved_models"], "sample_seed": 0 if a.sample else None,
        "block_all_left": a.block_all_left, "exact_shortcut": a.exact_shortcut, "concurrency": a.j,
        "timeout": a.timeout, "budget": a.budget, "cache_home": cache_home,
        "started_at": started.isoformat(timespec="seconds"), "finished_at": finished.isoformat(timespec="seconds"),
        "failed_pairs": len(errors), "rejected_422": int(errors.str.startswith("HTTP 422").sum()),
        "unjudged_pairs": int((result.scores["source"] == "unjudged").sum()),
        "error_examples": errors.value_counts().head(5).to_dict(),
    }
    if a.laya_audit is not None:
        summary["laya_audit"] = {"path": a.laya_audit.name, "before": audit_before, "after": audit_after,
                                 **{k: audit_after.get(k, 0) - audit_before.get(k, 0)
                                    for k in ("lines", "ok", "context_rejected")}}
    (out / "result.json").write_text(json.dumps(summary, indent=2, default=str))
    print(result.report())
    print(f"\nagainst truth at 0.5: {summary['at_0.5']}")
    print(table.to_string(index=False))
    print("\ncalibration (judged pairs):\n" + calibration(result.scores, truth).to_string(index=False))
    print(f"\nprovider {summary['api']}, models {summary['resolved_models']}, ${summary['dollars']:.4f}; "
          f"{summary['seconds']}s from {summary['started_at']} to {summary['finished_at']}; "
          f"failed {summary['failed_pairs']} (HTTP 422: {summary['rejected_422']}), "
          f"unjudged {summary['unjudged_pairs']}"
          + (f"; Laya audit +{summary['laya_audit']['lines']} lines, "
             f"{summary['laya_audit']['context_rejected']} context_rejected" if "laya_audit" in summary else ""))


if __name__ == "__main__":
    main()

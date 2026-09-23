"""Local decision models beside Jev on the same candidate pairs: the Jev column and the frozen results.

    uv run --group bench python bench/local_models.py \\
        [--tags diffusiongemma-s300 diffusiongemma-s200 laya-s300 laya-s200] \\
        [--out docs/benchmarks/local-models-2026-09-22.json]

For every bench/out/live/<dataset>-<tag>/ that `live.py --sample N --block-all-left` wrote, this
takes the 2026-09-18 full Jev run of that dataset (bench/out/live/<dataset>/), keeps the pairs of
the same sampled left records, checks that they are the pairs the local model was asked about,
and resolves and scores Jev's saved probabilities with the code the local run used. It then
counts how often the two models decide a pair the same way. No model is called: Jev's numbers
are the ones it gave on 2026-09-18.

Writes compare.json into each run directory and the frozen summary to --out.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import jlink
from live import HERE, RULES, load, sweep

ROOT = HERE.parent
TAGS = ["diffusiongemma-s300", "diffusiongemma-s200", "laya-s300", "laya-s200"]
SERVERS = {
    "diffusiongemma": {"api": "diffusiongemma", "model": "openjev-0.1", "url": "http://127.0.0.1:8080/v1/systemone",
                       "server": "OpenJev, commit e04794a",
                       "weights": "mlx-community/diffusiongemma-26B-A4B-it-4bit, revision a7a8140"},
    "laya": {"api": "laya", "model": "laya-421m", "url": "http://127.0.0.1:8081/v1/systemone",
             "server": "jevkit-core scripts/laya_server.py; laya-mlx commit fc1df62",
             "weights": "aac6fef/laya-mlx, revision 0476785",
             "context": "512 tokens including the question; longer states are refused with HTTP 422"},
}
KEY = ["left_id", "right_id"]


def metrics(scores: pd.DataFrame, truth: pd.DataFrame, how: str) -> dict:
    """Precision, recall and F1 at 0.5 and the best threshold in the sample, as live.py reports them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        links = jlink.resolve(scores, how=how, threshold=0.5)
    at_half = jlink.score_against_truth(links, truth, scores[KEY])
    table = sweep(scores, truth, how)
    best = table.loc[table["f1"].idxmax()]
    return {"at_0.5": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in at_half.items()},
            "best_f1_in_sample": best.to_dict(), "sweep": table.to_dict("records"), "links": links}


def pairs_of(frame: pd.DataFrame) -> set:
    return set(zip(frame["left_id"], frame["right_id"]))


def agreement(local: pd.DataFrame, jev: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Pair decisions at 0.5 where both models gave a probability; and what Jev said where the local model gave none."""
    both = local[KEY + ["p", "source"]].merge(jev[KEY + ["p", "source"]], on=KEY, suffixes=("_local", "_jev"))
    true = pairs_of(truth)
    both["match"] = [pair in true for pair in zip(both["left_id"], both["right_id"])]
    judged = both[both["p_local"].notna() & both["p_jev"].notna()]
    a, b = judged["p_local"] >= 0.5, judged["p_jev"] >= 0.5
    n = len(judged)
    agree = float((a == b).mean()) if n else float("nan")
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())) if n else float("nan")
    kappa = (agree - expected) / (1 - expected) if n and expected < 1 else float("nan")
    missing = both[both["p_local"].isna()]
    return {
        "pairs_both_scored": n, "decision_agreement": round(agree, 4), "cohen_kappa": round(kappa, 4),
        "both_yes": int((a & b).sum()), "local_yes_jev_no": int((a & ~b).sum()),
        "local_no_jev_yes": int((~a & b).sum()), "both_no": int((~a & ~b).sum()),
        "spearman_p": round(float(judged["p_local"].corr(judged["p_jev"], method="spearman")), 4) if n > 2 else None,
        "local_unscored": {"pairs": len(missing), "jev_yes": int((missing["p_jev"] >= 0.5).sum()),
                           "true_matches": int(missing["match"].sum())},
        "source_differs": both.loc[both["source_local"] != both["source_jev"],
                                   KEY + ["source_local", "source_jev"]].head(20).to_dict("records"),
        "source_differs_count": int((both["source_local"] != both["source_jev"]).sum()),
    }


def compare(dataset: str, run: Path) -> dict:
    result = json.loads((run / "result.json").read_text())
    if not result.get("block_all_left"):
        raise SystemExit(f"{run}: blocked on the sample alone, so its pairs are not the full run's; "
                         "rerun it with --block-all-left")
    how = RULES[dataset]["how"]
    left, _, truth, _ = load(dataset, result["sample"])
    local = jlink.load(run)
    full = jlink.load(HERE / "out" / "live" / dataset)
    sampled = set(left["id"])
    jev = full.scores[full.scores["left_id"].isin(sampled)].reset_index(drop=True)
    only_local, only_jev = pairs_of(local.scores) - pairs_of(jev), pairs_of(jev) - pairs_of(local.scores)
    jev_metrics, local_metrics = metrics(jev, truth, how), metrics(local.scores, truth, how)
    jev_links, local_links = pairs_of(jev_metrics.pop("links")), pairs_of(local_metrics.pop("links"))
    full_links = full.links[full.links["left_id"].isin(sampled)]
    full_result = json.loads((HERE / "out" / "live" / dataset / "result.json").read_text())
    rel = lambda p: str(p.relative_to(ROOT))  # noqa: E731
    log = run.with_name(run.name + ".log")
    wrapper = next((ln for ln in log.read_text().splitlines() if ln.startswith("WRAPPER")), None) if log.exists() else None
    out = {
        "dataset": dataset, "tag": run.name.removeprefix(dataset + "-"), "model": result["api"],
        "resolved_models": result["resolved_models"], "sample": result["sample"], "left": len(left),
        "truth": len(truth), "how": how, "rule": result["rule"],
        "pairs": {"local": len(local.scores), "jev": len(jev), "only_local": len(only_local),
                  "only_jev": len(only_jev), "only_local_examples": sorted(only_local)[:20],
                  "only_jev_examples": sorted(only_jev)[:20], "same": not only_local and not only_jev},
        "local": {"at_0.5": result["at_0.5"], "best_f1_in_sample": result["best_f1_in_sample"],
                  "check_at_0.5": local_metrics["at_0.5"], "sweep": result["sweep"],
                  "calibration": result["calibration"], "sources": result["sources"],
                  "failed_pairs": result["failed_pairs"], "rejected_422": result["rejected_422"],
                  "unjudged_pairs": result["unjudged_pairs"], "error_examples": result["error_examples"],
                  "laya_audit": result.get("laya_audit") and {**result["laya_audit"], "path": Path(result["laya_audit"]["path"]).name},
                  "calls": result["calls"], "cached": result["cached"],
                  "retries": result["retries"], "input_tokens": result["input_tokens"],
                  "dollars": result["dollars"], "seconds": result["seconds"],
                  "median_latency_ms": result["median_latency_ms"], "concurrency": result["concurrency"],
                  "timeout": result["timeout"], "started_at": result["started_at"],
                  "finished_at": result["finished_at"], "exact_shortcut": result["exact_shortcut"],
                  "wrapper": wrapper,
                  "raw": {"dir": rel(run), "result": rel(run / "result.json"), "scores": rel(run / "scores.csv"),
                          "links": rel(run / "links.csv"), "settings": rel(run / "settings.json"),
                          "log": rel(log) if log.exists() else None, "compare": rel(run / "compare.json")}},
        "jev": {"model": full.settings["model"], "run": f"bench/out/live/{dataset}", "date": full.settings["date"],
                "jlink": full.settings.get("jlink"),
                "provenance": ("Jev's probabilities are the 'p' column of "
                               f"bench/out/live/{dataset}/scores.csv (full run of {full.settings['date']}, "
                               f"{full.settings['model']}, jlink {full.settings.get('jlink')}), restricted to the "
                               "sampled left ids and resolved and scored here; full_run_links_restricted uses "
                               f"bench/out/live/{dataset}/links.csv. No Jev call was made on 2026-09-22."),
                "full_run": {"file": f"bench/out/live/{dataset}/result.json", "pairs": full_result["pairs"],
                             "at_0.5": full_result["at_0.5"], "dollars": full_result.get("dollars"),
                             "seconds": full_result.get("seconds")},
                **jev_metrics, "sources": jev["source"].value_counts().to_dict(),
                "full_run_links_restricted": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in
                                              jlink.score_against_truth(full_links, truth).items()}},
        "agreement": agreement(local.scores, jev, truth) | {
            "links_both": len(local_links & jev_links), "links_local_only": len(local_links - jev_links),
            "links_jev_only": len(jev_links - local_links),
            "links_jaccard": round(len(local_links & jev_links) / max(len(local_links | jev_links), 1), 4)},
    }
    (run / "compare.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def line(c: dict) -> str:
    local, jev, agree = c["local"]["at_0.5"], c["jev"]["at_0.5"], c["agreement"]
    return (f"{c['dataset']:<14} {c['tag']:<20} n={c['sample']:<4} pairs={c['pairs']['local']:<5} "
            f"same_pairs={c['pairs']['same']!s:<5} F1 local {local['f1']:.3f} (P {local['precision']:.3f} "
            f"R {local['recall']:.3f}) jev {jev['f1']:.3f}  failed={c['local']['failed_pairs']} "
            f"agree={agree['decision_agreement']:.3f} {c['local']['seconds']:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=TAGS)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "benchmarks" / "local-models-2026-09-22.json")
    a = ap.parse_args()
    runs = []
    for dataset in ("nber-firms", "abt-buy", "dblp-acm", "amazon-google", "febrl4"):
        for tag in a.tags:
            run = HERE / "out" / "live" / f"{dataset}-{tag}"
            if (run / "result.json").exists():
                runs.append(compare(dataset, run))
                print(line(runs[-1]))
    if not runs:
        raise SystemExit("no local runs found under bench/out/live/")
    frozen = {
        "date": "2026-09-22", "tool": "jlink", "jlink": jlink.__version__,
        "machine": "Apple M3 Ultra, 96 GiB unified memory; one benchmark on one model at a time",
        "servers": SERVERS,
        "conditions": {
            "concurrency": sorted({r["local"]["concurrency"] for r in runs}),
            "deadline_seconds": {m: sorted({r["local"]["timeout"] for r in runs if r["model"] == m})
                                 for m in sorted({r["model"] for r in runs})},
            "samples": {f"{r['dataset']}/{r['model']}": r["sample"] for r in runs},
            "order": "every DiffusionGemma run, then every Laya run, one at a time; the other server stayed "
                     "loaded but idle during each run and no other benchmark process was running",
            "budget": "--budget 0.01 as a tripwire (local calls metered at $0; every run reported $0.0000)",
            "timezone": "America/New_York",
        },
        "method": ("live.py --sample N (left records drawn with random_state=0, truth kept for them) "
                   "--block-all-left (n-gram TF-IDF blocking, k=10, fitted on every left record, so the pairs are "
                   "the full run's) --exact-shortcut (as jlink 0.1.0 always did), threshold 0.5, the final rules "
                   "in bench/live.py, an empty answer cache per run. The Jev column is the 2026-09-18 full run "
                   "(jlink 0.1.0, Jev 1.13) restricted to the same left records and resolved again with the "
                   "same code; Jev was not called again. Wall-clock seconds include blocking and resolution."),
        "runs": runs,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(frozen, indent=2, default=str))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

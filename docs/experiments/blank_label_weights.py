"""Simulation: blank audit labels spread unevenly across probability bins.

Run from the repository root with PYTHONPATH=src. Offline: no API client, no data files.
See docs/evaluation.md ("Blank labels") for the interpretation and the recorded result.

A population of judged pairs has perfectly calibrated scores and known truth. Each replicate
draws a stratified audit, then leaves labels blank completely at random WITHIN a bin, but
only in the three bins below p = 0.5: a labeler who starts with the likely matches and runs
out of time. "dropped" evaluates the labeled rows alone, which is what `evaluate` did before
it reweighted blank labels (with no blank rows in the table there is nothing to adjust).
"reweighted" evaluates the same audit with its blank rows kept.
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from jlink.audit import audit_sample, evaluate


def run(*, pairs: int, n: int, blank_rate: float, replicates: int, n_boot: int) -> dict:
    rng = np.random.default_rng(1)
    low = int(pairs * .9)
    p = np.concatenate([rng.beta(.3, 6, low), rng.beta(8, 1, pairs - low)])
    truth = rng.random(pairs) < p
    scores = pd.DataFrame({"left_id": np.arange(pairs), "right_id": np.arange(pairs), "p": p})
    predicted = p >= .5
    target = {"precision": float((predicted & truth).sum() / predicted.sum()),
              "recall": float((predicted & truth).sum() / truth.sum())}
    arms = {"all labeled": [], "dropped": [], "reweighted": []}
    for replicate in range(replicates):
        sample = audit_sample(scores, n=n, seed=replicate)
        labels = truth[sample.left_id.to_numpy()].astype(float)
        blank = ((np.random.default_rng(10_000 + replicate).random(len(sample)) < blank_rate)
                 & (sample.p < .5).to_numpy())
        partial = sample.assign(is_match=np.where(blank, np.nan, labels))
        for arm, frame in (("all labeled", sample.assign(is_match=labels)),
                           ("dropped", partial.loc[~blank]), ("reweighted", partial)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = evaluate(frame, threshold=.5, n_boot=n_boot, seed=0)
            arms[arm].append([*result.precision, *result.recall])
    report = {"population_pairs": pairs, "audit_n": n, "blank_rate_below_half": blank_rate,
              "replicates": replicates, "n_boot": n_boot, "truth": target, "arms": {}}
    for arm, rows in arms.items():
        values = np.asarray(rows)
        report["arms"][arm] = {
            metric: {"mean_estimate": float(values[:, 3 * i].mean()),
                     "interval_coverage": float(((values[:, 3 * i + 1] <= target[metric])
                                                 & (target[metric] <= values[:, 3 * i + 2])).mean())}
            for i, metric in enumerate(("precision", "recall"))}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=int, default=60_000)
    parser.add_argument("-n", type=int, default=600)
    parser.add_argument("--blank-rate", type=float, default=.7)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--out", help="also write the report as JSON")
    args = parser.parse_args()
    report = run(pairs=args.pairs, n=args.n, blank_rate=args.blank_rate, replicates=args.replicates,
                 n_boot=args.n_boot)
    print(f"truth: precision {report['truth']['precision']:.4f}, recall {report['truth']['recall']:.4f}")
    for arm, metrics in report["arms"].items():
        print(f"{arm:12} " + "; ".join(
            f"{metric} {m['mean_estimate']:.4f} (95% interval covers truth {m['interval_coverage']:.0%})"
            for metric, m in metrics.items()))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()

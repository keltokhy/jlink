"""LinkTransformer control for the firm-rule fixtures: one cosine per pair, whatever the rule.

Run in LinkTransformer's own environment with a supplied JSON fixture. Each pair has
record_a, record_b, a dev/test split, and Boolean labels keyed by research definition.
A similarity score cannot depend on the definition, so its decisions are bounded by
the rule-blind ceiling; this measures how close it gets.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

FIELDS = ["name", "as_of", "record_scope", "site", "facts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import linktransformer as lt

    pairs = json.loads(args.fixture.read_text())["pairs"]
    table = pd.DataFrame([{"pair_id": p["pair_id"], "split": p["split"],
                           **{f"{side}_{f}": p[f"record_{side}"].get(f) or "" for side in "ab" for f in FIELDS}}
                          for p in pairs])
    grid = [i / 100 for i in range(101)] + [None]
    report = {"fixture": args.fixture.name, "arms": {}}
    for name in args.models:
        model = lt.LinkTransformer(name, device="cpu")
        # Compare name and date only with every supplied fact.
        for arm, fields in (("names_and_dates", ["name", "as_of"]), ("rich", FIELDS)):
            scored = lt.evaluate_pairs(table, model=model, left_on=[f"a_{f}" for f in fields],
                                       right_on=[f"b_{f}" for f in fields])
            cosine = dict(zip(scored.pair_id, scored.score.astype(float)))
            rows = [{"split": p["split"], "gold": value, "sim": cosine[p["pair_id"]]}
                    for p in pairs for value in p["labels"].values() if value is not None]
            # A single threshold chosen on dev must remain identical across research definitions.
            trials = [(sum((t is not None and r["sim"] >= t) == r["gold"] for r in rows if r["split"] == "dev"), t)
                      for t in grid]
            chosen = max(reversed(trials), key=lambda trial: trial[0])[1]
            result = {"threshold": chosen}
            for split in ("dev", "test"):
                selected = [r for r in rows if r["split"] == split]
                result[split] = {"correct": sum((chosen is not None and r["sim"] >= chosen) == r["gold"]
                                                for r in selected), "n": len(selected),
                                 "best_possible_threshold_correct": max(
                                     sum((t is not None and r["sim"] >= t) == r["gold"] for r in selected)
                                     for t in grid)}
            report["arms"][f"{name}|{arm}"] = result
    for split in ("dev", "test"):
        labels = [[v for v in p["labels"].values() if v is not None] for p in pairs if p["split"] == split]
        report[f"{split}_rule_blind_ceiling"] = sum(max(l.count(True), l.count(False)) for l in labels)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

"""Zero-shot Jev judgments on the DeepMatcher test pairs behind LinkTransformer's Table A-5.

Default is an offline estimate; --live needs a finite budget. Every labelled test pair is judged
once through jlink's public judge() with a definition frozen in configs/deepmatcher.json, and
scored as pairwise classification at 0.5, the task and metric of the published table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import urllib.request

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data import DATA, ROOT
from bench.evaluation import digest
from bench.heldout import file_hash, write_json
from jlink.core import PRICE_PER_MTOK
from jlink.judge import _records, judge, question, validate_budget

CONFIG = ROOT / "configs" / "deepmatcher.json"
FIELD = re.compile(r"COL (\S+) VAL (.*?)(?=\s+COL \S+ VAL |\s*$)")
OVERHEAD_TOKENS = 270  # fixed tokens per decision observed in earlier runs; estimate only


def load(config: dict, dataset: dict, data_dir: Path):
    """One left and one right record per labelled pair; IDs are the pair's line number."""
    path = Path(data_dir) / "_downloads" / "er_magellan" / dataset["name"] / "test.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config["source"].format(path=dataset["name"]), path)
    sides, labels = ([], []), []
    for line in path.read_text().splitlines():
        a, b, label = line.split("\t")
        for side, text in zip(sides, (a, b)):
            side.append(dict(FIELD.findall(text)))
        labels.append(int(label))
    fields = list(sides[0][0])
    if any(list(record) != fields for side in sides for record in side):
        raise ValueError(f"{dataset['name']} has records whose fields differ from {fields}")
    left, right = (pd.DataFrame(side).assign(id=[f"{tag}{i}" for i in range(len(side))])
                   for tag, side in zip("LR", sides))
    pairs = pd.DataFrame({"left_id": left.id, "right_id": right.id, "sim": 0.0, "label": labels})
    return left, right, pairs, fields, path


def definition(config: dict, dataset: dict) -> str:
    return dataset["definition"] + (" " + config["dirty_note"] if dataset.get("dirty") else "")


def estimate(config: dict, data_dir: Path) -> dict:
    """Request bytes over four plus a fixed overhead per call, at the list price. No API calls."""
    rows = {}
    for dataset in config["datasets"]:
        left, right, pairs, fields, _ = load(config, dataset, data_dir)
        ask = question(dataset["entity"], definition(config, dataset))
        a, b = (_records(frame, frame.id, [(c, c) for c in fields]).record for frame in (left, right))
        size = sum(len(json.dumps({"record_a": x, "record_b": y, "q": ask}, ensure_ascii=False).encode())
                   for x, y in zip(a, b))
        tokens = size / 4 + OVERHEAD_TOKENS * len(pairs)
        rows[dataset["name"]] = {"pairs": len(pairs), "positives": int(pairs.label.sum()),
                                 "estimated_tokens": int(tokens),
                                 "estimated_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4)}
    return {"datasets": rows, "pairs": sum(r["pairs"] for r in rows.values()),
            "estimated_usd": round(sum(r["estimated_usd"] for r in rows.values()), 4),
            "note": "bytes/4 plus 270 tokens per call at the list price; not a provider quote"}


def metrics(scores: pd.DataFrame, threshold: float) -> dict:
    """Unjudged and failed pairs count as predicted nonmatches and are reported, never dropped."""
    predicted, actual = scores.p.ge(threshold), scores.label.eq(1)
    tp, fp, fn = int((predicted & actual).sum()), int((predicted & ~actual).sum()), int((~predicted & actual).sum())
    precision, recall = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    return {"pairs": len(scores), "positives": int(actual.sum()), "judged": int(scores.p.notna().sum()),
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0}


def run(*, out: Path, data_dir: Path, budget: float, only: list[str] | None = None) -> dict:
    validate_budget(budget)
    if budget is None or budget <= 0:
        raise ValueError("paid collection requires a finite positive budget")
    config = json.loads(CONFIG.read_text())
    settings, out = config["model_settings"], Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "schema_version": 1, "config_sha256": digest(config), "datasets": {}, "spent_usd": 0.0,
        "code_sha256": {name: file_hash(ROOT.parent / name) for name in (
            "bench/deepmatcher_pairs.py", "src/jlink/judge.py", "src/jlink/core.py")}}
    if report["config_sha256"] != digest(config):
        raise ValueError("the frozen configuration changed since this output directory was started")
    for dataset in config["datasets"]:
        name = dataset["name"]
        if (only and name not in only) or report["datasets"].get(name, {}).get("complete"):
            continue
        remaining = budget - report["spent_usd"]
        if remaining <= 0:
            break
        left, right, pairs, fields, path = load(config, dataset, data_dir)
        scores, meter = judge(pairs, left, right, on=fields, entity=dataset["entity"],
                              definition=definition(config, dataset), left_id="id", right_id="id",
                              api=settings["api"], model=settings["model"], budget=remaining,
                              exact_shortcut=settings["exact_shortcut"], progress=False)
        directory = out / name.replace("/", "-")
        directory.mkdir(exist_ok=True)
        scores.to_csv(directory / "scores.csv", index=False)
        result = metrics(scores, settings["threshold"])
        report["spent_usd"] += meter.cost
        report["datasets"][name] = dict(
            result, table_row=dataset["table_row"], complete=result["judged"] == result["pairs"],
            published=dict(zip(config["published"]["columns"], dataset["published"])),
            models=sorted(scores.model.dropna().unique()), new_cost_usd=meter.cost,
            source_sha256=file_hash(path), scores_sha256=file_hash(directory / "scores.csv"))
        write_json(report_path, report)
        print(f"{name}: F1 {result['f1']:.4f} (P {result['precision']:.3f}, R {result['recall']:.3f}); "
              f"{result['judged']:,}/{result['pairs']:,} judged; ${meter.cost:.4f}", flush=True)
    return report


def table(out: Path) -> str:
    """The published row beside jlink's zero-shot F1, in the table's own units (percent)."""
    report = json.loads((Path(out) / "report.json").read_text())
    lines = ["| Table A-5 row | Test pairs | Matches | jlink, zero-shot | LT zero-shot | LT fine-tuned | "
             "Magellan | DeepMatcher | Ditto | REMS |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["datasets"].values():
        lines.append(f"| {row['table_row']} | {row['pairs']:,} | {row['positives']:,} | **{100 * row['f1']:.2f}** | "
                     + " | ".join(f"{value:g}" for value in row["published"].values()) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "deepmatcher-pairs")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--budget", type=float, default=0)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--table", action="store_true", help="print the saved results as markdown")
    args = parser.parse_args(argv)
    try:
        if args.table:
            print(table(args.out))
            return 0
        if not args.live:
            print(json.dumps(estimate(json.loads(CONFIG.read_text()), args.data_dir), indent=1))
            return 0
        report = run(out=args.out, data_dir=args.data_dir, budget=args.budget, only=args.only)
    except (ValueError, OSError) as exc:
        print(f"bench deepmatcher_pairs: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"spent_usd": round(report["spent_usd"], 4),
                      "complete": all(d["complete"] for d in report["datasets"].values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

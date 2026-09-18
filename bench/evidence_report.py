"""Collect frozen offline reports and render readable evidence without executing experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data import NAMES
from bench.heldout import file_hash


def collect(cached_dir: Path, unmatched_dir: Path, rules_dir: Path, legacy_dir: Path) -> dict:
    """Preserve complete run reports and hash legacy inputs without copying source records."""
    def read(path):
        return {"report_sha256": file_hash(path), "report": json.loads(path.read_text())}
    return {"schema_version": 1, "collected_at": datetime.now(timezone.utc).isoformat(),
            "cached_replays": {name: read(cached_dir / name / "report.json") for name in NAMES},
            "explicit_unmatched": read(unmatched_dir / "febrl4" / "report.json"),
            "rule_sensitivity": read(rules_dir / "report.json"),
            "legacy_inputs": {str(path.relative_to(legacy_dir)): file_hash(path)
                              for name in NAMES for path in
                              (legacy_dir / f"{name}.json", legacy_dir / "live" / name / "result.json",
                               legacy_dir / "live" / name / "settings.json",
                               legacy_dir / "live" / name / "scores.csv")}}


def render(bundle: dict) -> str:
    """Render measured values while keeping unverified or incomplete evidence explicit."""
    lines = ["# Offline benchmark evidence", "", "Measured on 2026-09-18 from prepared datasets and existing "
             "cached model scores. **Zero new API calls and zero new API cost.** "
             "[evidence/2026-09-18.json](evidence/2026-09-18.json) retains complete reports, settings, "
             "parameters, data/code/artifact hashes and threshold trials. "
             "[EVALUATION.md](EVALUATION.md) specifies how to reproduce and interpret them.", "",
             "## Matched retrospective replay", "",
             "Observed entities are split 30% development / 70% test, seed 1729. "
             "Every method uses the same cached candidate pool filtered within each partition, "
             "and the same dataset cardinality. String/ECM thresholds use development final F1; "
             "Jev stays at 0.5. These retrospective partitions do not undo earlier dataset or prompt "
             "inspection. Cosines retain the historical full-corpus fit. They are not untouched "
             "prospective results or comparable to the original in-sample oracle table.", "",
             "| Dataset | Test left/right | Test truth | Candidate pairs | Candidate recall | Exact F1 | "
             "Jaro F1 | TF-IDF F1 | ECM F1 | Cached Jev F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, entry in bundle["cached_replays"].items():
        d = entry["report"]
        counts, results = d["split"]["counts"]["test"], d["results"]
        candidate = results["exact"]["test"]["candidates"]
        values = []
        for method in ("exact", "jaro_winkler", "tfidf", "ecm", "cached_jev"):
            value = results.get(method, {}).get("test", {}).get("final_assignment", {}).get("f1")
            values.append("—" if value is None else f"{value:.4f}")
        lines.append(f"| {name} | {counts['left']:,} / {counts['right']:,} | {counts['truth']:,} | "
                     f"{candidate['pairs']:,} | {candidate['recall']:.4f} | " + " | ".join(values) + " |")
    lines += ["", "Leipzig metrics use the supplied mappings' closed-world convention; they do not "
              "independently verify omitted links. Product cardinality is many-to-many for all methods. "
              "The old string oracles chose only one right record per left. The difference in product "
              "F1 is therefore not evidence of a regression in the underlying string algorithms.", "",
              "ECM is competitive on people/publications but weak under the fixed product feature "
              "specification. It is inapplicable to name-only NBER. A more complex method does not "
              "guarantee a better result. On FEBRL's complete test, selected Jaro and ECM thresholds "
              "triggered the existing resolver's >2,000-node greedy fallback; the report preserves "
              "those warnings. No new resolver optimization is claimed here.", "",
              "### Pair judgments versus final assignments", "",
              "All values below are from cached Jev probabilities at 0.5. Pair metrics precede "
              "resolution and include exact shortcuts; per-source detail is in JSON. Final recall "
              "uses all test truth, including matches missed by blocking.", "",
              "| Dataset | Pair precision among candidates | Pair recall among candidate truth | "
              "Final precision | Final recall against all test truth |",
              "|---|---:|---:|---:|---:|"]
    for name, entry in bundle["cached_replays"].items():
        stages = entry["report"]["results"]["cached_jev"]["test"]
        pair, final = stages["judge_conditional_on_candidates"], stages["final_assignment"]
        values = [pair["precision"], pair["recall"], final["precision"], final["recall"]]
        lines.append(f"| {name} | " + " | ".join("—" if v is None else f"{v:.4f}" for v in values) + " |")
    nber = bundle["cached_replays"]["nber-firms"]["report"]["results"]["cached_jev"]["test"]
    lines += ["", f"NBER: {nber['final_assignment']['listed_tp']:,} listed test links selected; "
              f"{nber['final_assignment']['unlisted_predictions_unknown']:,} "
              "unlisted predictions remain unknown. "
              "No NBER precision, F1, calibration, false-positive claim or F1 tuning "
              "is made in this protocol. "
              "NBER historical oracle precision/F1 in BASELINES.md is retained only as crosswalk agreement "
              "under its original negative assumption.", "", "## Explicit unmatched FEBRL records", ""]
    unmatched = bundle["explicit_unmatched"]["report"]
    counts = unmatched["split"]["counts"]["test"]
    lines += [f"The test universe has {counts['left']:,} left and {counts['right']:,} right records, "
              f"{counts['truth']:,} known matches, and {counts['explicit_unmatched'] // 2:,} explicitly "
              "unmatched records on each side after removing known counterparts. All methods use "
              "the same filtered cached candidate pool. "
              "No previously unknown entity is declared unmatched.", "",
              "| Method | Dev-selected/fixed threshold | Test F1 | Unjudged candidates | "
              "Unmatched left falsely linked | Unmatched right falsely linked |",
              "|---|---:|---:|---:|---:|---:|"]
    for method, result in unmatched["results"].items():
        stages = result["test"]
        final = stages["final_assignment"]
        abstain = final["explicit_unmatched"]
        lines.append(f"| {method} | {result['threshold']} | {final['f1']:.4f} | "
                     f"{stages['judge_conditional_on_candidates']['unjudged_pairs']:,} | "
                     f"{abstain['left']['false_linked']:,} | {abstain['right']['false_linked']:,} |")
    rules = bundle["rule_sensitivity"]["report"]
    lines += ["", "## English-definition sensitivity", "",
              f"Status: **{rules['status']}**. The committed synthetic fixture creates "
              f"{rules['requests']} exact requests for 18 pairs and two definitions. "
              f"There are {rules['responses']} suitable cached model responses. "
              "Offline fake tests verify changed-case sensitivity, invariant cases, missing responses, "
              "stale hashes and evidence separation. They are not live model results. "
              "Real-company ownership knowledge is not tested by this fixture.", "",
              "## Provenance and remaining experiments", "",
              "Inputs were read from `/Volumes/K3/GitHub/jlink/bench/data` and `bench/out` without "
              "modification. Replays save source hashes, explicit split definitions, scores, links and "
              "unmatched removals. The JSON bundle preserves reports and hashes; per-pair CSVs remain "
              "in the worktree's ignored `bench/out/final-cached`, `final-unmatched`, and `final-rules` "
              "directories and are reproducible with the documented commands. Legacy caches lack raw "
              "provider responses and original dataset hashes; current ID/field validation cannot prove "
              "which historical source values were submitted. "
              "These limitations are disclosed in each report.", "",
              "Pending work: obtain independently reviewed legal-entity/family labels for real firms; "
              "run both definitions on the frozen synthetic requests with explicit spending authorization; "
              "collect additional independently labeled NBER negatives before precision/F1 claims; "
              "evaluate new blockers and unjudged pairs after integration; and repeat across prespecified "
              "splits or new untouched datasets before generalizing. No result from those experiments is "
              "implied by this offline report.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-dir", type=Path, required=True)
    parser.add_argument("--unmatched-dir", type=Path, required=True)
    parser.add_argument("--rules-dir", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = collect(args.cached_dir, args.unmatched_dir, args.rules_dir, args.legacy_dir)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    # Keep the evidence bundle compact; the Markdown is the reading view.
    args.json.write_text(json.dumps(bundle, ensure_ascii=False, allow_nan=False) + "\n")
    args.markdown.write_text(render(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Freeze and evaluate paired plain-English definitions entirely offline.

The fixture is deliberately synthetic. This program writes exact requests but cannot send them.
Cache rows must bind to those requests and preserve the provider's parsed response unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.evaluation import digest
from bench.heldout import file_hash, write_json
from jlink.judge import question

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "rule_sensitivity.json"
RULE_IDS = {"same_legal_entity", "corporate_family"}


def load_fixture(path: Path = FIXTURE) -> dict:
    """Validate explicit labels, source provenance, and group-disjoint fixture partitions."""
    fixture = json.loads(path.read_text())
    if fixture.get("schema_version") != 1 or set(fixture.get("rules", {})) != RULE_IDS:
        raise ValueError("fixture needs schema_version 1 and both legal-entity and corporate-family rules")
    if any(not isinstance(fixture.get(key), str) or not fixture[key].strip()
           for key in ("title", "provenance")):
        raise ValueError("fixture needs a nonempty title and provenance")
    if any(not isinstance(rule, str) or not rule.strip() for rule in fixture["rules"].values()):
        raise ValueError("fixture rules must be nonempty strings")
    settings = fixture.get("model_settings", {})
    if not settings.get("model") or not settings.get("api"):
        raise ValueError("fixture must pin model_settings.model and api")
    if settings.get("exact_shortcut") is not False:
        raise ValueError("rule sensitivity requires exact_shortcut=false")
    seen, groups = set(), {}
    for row in fixture.get("pairs", []):
        if not row.get("pair_id") or row["pair_id"] in seen:
            raise ValueError("fixture pair_id must be nonempty and unique")
        seen.add(row["pair_id"])
        if row.get("split") not in {"dev", "test"} or not row.get("group_id"):
            raise ValueError("each fixture pair needs dev/test split and group_id")
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("fixture groups must be disjoint between development and test")
        labels = row.get("labels", {})
        if set(labels) != RULE_IDS or any(type(value) is not bool for value in labels.values()):
            raise ValueError("both labels must be explicit JSON booleans")
        if labels["same_legal_entity"] and not labels["corporate_family"]:
            raise ValueError("same legal entity must also count as corporate family under these definitions")
        source = row.get("source", {})
        if source.get("kind") not in {"synthetic", "primary_source"} or not source.get("reference"):
            raise ValueError("each pair needs a synthetic or primary_source reference")
        if (not source.get("rationale") or any(not isinstance(row.get(key), dict) or not row[key]
                                                for key in ("record_a", "record_b"))):
            raise ValueError("each pair needs two record objects and a label rationale")
    if not seen or set(groups.values()) != {"dev", "test"}:
        raise ValueError("fixture must include nonempty development and test groups")
    return fixture


def requests(fixture: dict) -> list[dict]:
    """Bind every comparison to exact question, state, model settings, and fixture content."""
    rows = []
    for pair in fixture["pairs"]:
        for rule_id, definition in fixture["rules"].items():
            payload = {"model": fixture["model_settings"]["model"],
                       "state": {"record_a": pair["record_a"], "record_b": pair["record_b"]},
                       "questions": {"match": question("firm", definition)}}
            item = {"pair_id": pair["pair_id"], "rule_id": rule_id, "split": pair["split"],
                    "fixture_sha256": digest(fixture), "model_settings": fixture["model_settings"],
                    "payload": payload}
            rows.append(dict(item, request_sha256=digest(item)))
    return rows


def evaluate(fixture: dict, cached: list[dict], *, threshold: float = .5) -> dict:
    """Evaluate complete paired responses; report coverage and synthetic fake runs explicitly."""
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite probability")
    expected = {row["request_sha256"]: row for row in requests(fixture)}
    results, evidence = {}, set()
    for row in cached:
        key = row.get("request_sha256")
        if key not in expected or key in results:
            raise ValueError("cached response has an unknown or duplicate request_sha256")
        if row.get("evidence_kind") not in {"live_model", "offline_fake"}:
            raise ValueError("cache evidence_kind must distinguish live_model from offline_fake")
        if not row.get("model") or row["model"] != fixture["model_settings"]["model"]:
            raise ValueError("cached response model differs from the frozen request")
        if not row.get("recorded_at"):
            raise ValueError("cached responses must preserve recorded_at")
        if row["evidence_kind"] == "live_model" and not row.get("provider_response"):
            raise ValueError("live_model cache must preserve the original provider_response")
        try:
            probability = row["response"]["match"]["noul"]
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise ValueError
            probability = float(probability)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("cached response needs response.match.noul between 0 and 1") from exc
        if row["evidence_kind"] == "live_model":
            provider = row["provider_response"]
            if not isinstance(provider, dict) or provider.get("answers") != row["response"]:
                raise ValueError("parsed response must equal provider_response.answers")
        results[key] = probability
        evidence.add(row["evidence_kind"])
    if len(evidence) > 1:
        raise ValueError("do not pool offline_fake and live_model evidence")
    by_pair = {}
    for key, value in results.items():
        row = expected[key]
        by_pair.setdefault(row["pair_id"], {})[row["rule_id"]] = value
    metrics = {}
    for split in ("dev", "test"):
        pairs = [row for row in fixture["pairs"] if row["split"] == split]
        stats = {"pairs": len(pairs), "complete_paired_responses": 0, "by_rule": {},
                 "changed_definition": {"n": 0, "both_correct": 0, "correct_direction": 0},
                 "invariant_definition": {"n": 0, "both_correct": 0}}
        for rule in sorted(RULE_IDS):
            present = [row for row in pairs if rule in by_pair.get(row["pair_id"], {})]
            correct = sum((by_pair[row["pair_id"]][rule] >= threshold) == row["labels"][rule]
                          for row in present)
            stats["by_rule"][rule] = {"responses": len(present), "correct": correct,
                                       "accuracy": correct / len(present) if present else None}
        for row in pairs:
            probabilities = by_pair.get(row["pair_id"], {})
            if set(probabilities) != RULE_IDS:
                continue
            stats["complete_paired_responses"] += 1
            changing = row["labels"]["same_legal_entity"] != row["labels"]["corporate_family"]
            target = stats["changed_definition" if changing else "invariant_definition"]
            target["n"] += 1
            target["both_correct"] += int(all((probabilities[rule] >= threshold) == row["labels"][rule]
                                              for rule in RULE_IDS))
            if changing:
                target["correct_direction"] += int(probabilities["corporate_family"] >
                                                    probabilities["same_legal_entity"])
        metrics[split] = stats
    return {"status": "live_model_pending" if not results else
                      "offline_fake_only" if evidence == {"offline_fake"} else
                      "cached_live_complete" if len(results) == len(expected) else "cached_live_partial",
            "fixture_sha256": digest(fixture), "requests": len(expected), "responses": len(results),
            "missing_responses": len(expected) - len(results), "threshold": threshold,
            "threshold_policy": "fixed before model evaluation", "metrics": metrics,
            "new_api_calls": 0, "new_api_cost_usd": 0}


def run(*, fixture_path: Path = FIXTURE, cache_path: Path | None = None, out: Path) -> dict:
    fixture = load_fixture(fixture_path)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} already contains artifacts; choose a new --out directory")
    cached = ([json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
              if cache_path else [])
    report = evaluate(fixture, cached)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "fixture.json", fixture)
    (out / "requests.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                for row in requests(fixture)))
    if cache_path:
        # Preserve exact cache bytes, not a reformatted or lossy response projection.
        (out / "responses.jsonl").write_bytes(cache_path.read_bytes())
    report["code_sha256"] = {name: file_hash(ROOT.parent / name) for name in (
        "bench/rule_sensitivity.py", "bench/evaluation.py", "src/jlink/judge.py", "src/jlink/core.py")}
    report["artifacts"] = {path.name: file_hash(path) for path in sorted(out.iterdir()) if path.is_file()}
    write_json(out / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "rule-sensitivity")
    args = parser.parse_args(argv)
    try:
        result = run(fixture_path=args.fixture, cache_path=args.cache, out=args.out)
    except (ValueError, OSError) as exc:
        print(f"bench rules: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Source-backed rule-following pilot. Default is offline; --live needs a finite budget.

Runs prescribed pairs through jlink's existing question builder, record serializer and Jev
client. This isolates judging; it does not measure retrieval, assignment or source extraction.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from bench.evaluation import digest
from bench.heldout import file_hash, write_json
from jlink.core import Jev, JevError, JevFatal, resolve_backend
from jlink.judge import _records, question, validate_budget

FIXTURE = Path(__file__).with_name("fixtures") / "firm_rules.json"
RULES = {"legal_entity", "corporate_group", "physical_site", "operating_business"}
ARMS = ("rich", "names_and_dates")


def load_fixture(path=FIXTURE):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1 or set(data.get("rules", {})) != RULES:
        raise ValueError("fixture requires schema_version 1 and the four firm rules")
    settings = data.get("model_settings", {})
    if (settings.get("threshold") != .5 or settings.get("exact_shortcut") is not False
            or not all(settings.get(k) for k in ("api", "model", "endpoint"))):
        raise ValueError("fixture must pin API, endpoint, model, threshold 0.5 and disabled exact shortcut")
    if not data.get("provenance") or set(data.get("arms", {})) != set(ARMS):
        raise ValueError("fixture needs provenance and both comparison arms")
    if any(not isinstance(v, str) or not v.strip() for v in data["rules"].values()):
        raise ValueError("rules must be nonempty definitions")
    ids, groups = set(), {}
    for pair in data.get("pairs", []):
        if not pair.get("pair_id") or pair["pair_id"] in ids:
            raise ValueError("pair_id must be nonempty and unique")
        ids.add(pair["pair_id"])
        group, split = pair.get("group_id"), pair.get("split")
        if not group or split not in {"dev", "test"} or groups.setdefault(group, split) != split:
            raise ValueError("corporate groups must be disjoint across dev/test")
        labels = pair.get("labels", {})
        if (set(labels) != RULES or any(v is not None and type(v) is not bool for v in labels.values())
                or not any(v is not None for v in labels.values())):
            raise ValueError("labels must contain four explicit booleans or null, with a known label")
        if not pair.get("rationale") or not pair.get("source_ids"):
            raise ValueError("each pair needs source references and a label rationale")
        for sid in pair["source_ids"]:
            source = data.get("sources", {}).get(sid, {})
            if not source.get("url", "").startswith("https://") or not source.get("evidence_summary"):
                raise ValueError("each source needs an HTTPS reference and evidence summary")
        for side in ("record_a", "record_b"):
            record = pair.get(side, {})
            if not isinstance(record, dict) or not record.get("name") or not record.get("as_of"):
                raise ValueError("records need name and as_of fields")
            if set(record) & {"labels", "rationale", "pair_id", "group_id", "split"}:
                raise ValueError("records must not contain evaluation metadata")
    if not ids or set(groups.values()) != {"dev", "test"}:
        raise ValueError("fixture needs nonempty development and test groups")
    return data


def requests(fixture):
    """Preserve the production question and serialization; keep gold/source notes out of payloads."""
    rows = []
    for pair in fixture["pairs"]:
        for arm in ARMS:
            source = [pair["record_a"], pair["record_b"]]
            columns = (list(dict.fromkeys(k for record in source for k in record))
                       if arm == "rich" else ["name", "as_of"])
            frame = pd.DataFrame(source).reindex(columns=columns)
            records = _records(frame, frame.index, [(c, c) for c in columns]).record.tolist()
            for rule, definition in fixture["rules"].items():
                if pair["labels"][rule] is None:
                    continue
                payload = {"model": fixture["model_settings"]["model"],
                           "state": dict(zip(("record_a", "record_b"), records)),
                           "questions": {"match": question("firm", definition)}}
                item = {"pair_id": pair["pair_id"], "arm": arm, "rule_id": rule,
                        "fixture_sha256": digest(fixture), "payload": payload}
                rows.append(dict(item, request_sha256=digest(item)))
    return rows


def _metrics(rows):
    """Missing responses reduce coverage and overall correctness, never disappear silently."""
    answered = [r for r in rows if r["p"] is not None]
    correct = sum((r["p"] >= .5) == r["gold"] for r in answered)
    tp = sum(r["gold"] and r["p"] >= .5 for r in answered)
    fp = sum(not r["gold"] and r["p"] >= .5 for r in answered)
    fn = sum(r["gold"] and (r["p"] is None or r["p"] < .5) for r in rows)
    return {"n": len(rows), "responses": len(answered), "correct": correct,
            "accuracy": correct / len(answered) if answered else None,
            "correct_fraction_of_expected": correct / len(rows) if rows else None,
            "coverage": len(answered) / len(rows) if rows else None,
            "tp": tp, "fp": fp, "fn_including_missing": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def evaluate(fixture, responses):
    frozen = {r["request_sha256"]: r for r in requests(fixture)}
    answers, evidence = {}, set()
    for row in responses:
        key = row.get("request_sha256")
        if key not in frozen or key in answers:
            raise ValueError("response has unknown or duplicate request hash")
        if row.get("evidence_kind") not in {"live_model", "offline_fake"}:
            raise ValueError("response must identify live or fake evidence")
        if row.get("model") != fixture["model_settings"]["model"] or not row.get("recorded_at"):
            raise ValueError("response must preserve requested model and time")
        if row.get("provider_response", {}).get("answers") != row.get("response"):
            raise ValueError("parsed answers must equal the preserved provider response")
        try:
            p = row["response"]["match"]["noul"]
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("response needs a finite match probability between 0 and 1") from exc
        answers[key] = p
        evidence.add(row["evidence_kind"])
    if len(evidence) > 1:
        raise ValueError("cannot pool fake and live evidence")
    pairs = {p["pair_id"]: p for p in fixture["pairs"]}
    decisions = []
    for key, req in frozen.items():
        pair = pairs[req["pair_id"]]
        decisions.append({"pair_id": pair["pair_id"], "group_id": pair["group_id"],
                          "split": pair["split"], "arm": req["arm"], "rule_id": req["rule_id"],
                          "gold": pair["labels"][req["rule_id"]], "p": answers.get(key)})
    metrics = {}
    for split in ("dev", "test"):
        metrics[split] = {}
        for arm in ARMS:
            rows = [r for r in decisions if r["split"] == split and r["arm"] == arm]
            changing = [p for p in fixture["pairs"] if p["split"] == split and
                        len({v for v in p["labels"].values() if v is not None}) > 1]
            fully_correct = 0
            for pair in changing:
                values = [r for r in rows if r["pair_id"] == pair["pair_id"]]
                fully_correct += int(all(r["p"] is not None and (r["p"] >= .5) == r["gold"]
                                         for r in values))
            metrics[split][arm] = {"overall": _metrics(rows),
                "by_rule": {rule: _metrics([r for r in rows if r["rule_id"] == rule]) for rule in RULES},
                "by_group": {group: _metrics([r for r in rows if r["group_id"] == group])
                             for group in sorted({r["group_id"] for r in rows})},
                "rule_changing_pairs": {"n": len(changing), "all_known_rules_correct": fully_correct}}
    status = "pending" if not answers else "offline_fake" if evidence == {"offline_fake"} else (
        "live_complete" if len(answers) == len(frozen) else "live_partial")
    return {"status": status, "fixture_sha256": digest(fixture), "requests": len(frozen),
            "responses": len(answers), "missing_responses": len(frozen) - len(answers),
            "threshold": .5, "metrics": metrics, "decisions": decisions,
            "unknown_labels_excluded_per_arm": sum(v is None for p in pairs.values()
                                                    for v in p["labels"].values())}


def baselines(fixture):
    """A rule-blind name baseline and its oracle ceiling; no labels enter the text features."""
    pairs = fixture["pairs"]
    names = [p[side]["name"] for p in pairs for side in ("record_a", "record_b")]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(names)
    similarities = matrix[::2].multiply(matrix[1::2]).sum(axis=1).A1
    rows = [{"pair_id": pair["pair_id"], "split": pair["split"], "gold": value,
             "sim": float(sim)} for pair, sim in zip(pairs, similarities)
            for value in pair["labels"].values() if value is not None]
    grid = [i / 20 for i in range(21)] + [None]
    # A single threshold chosen on dev must remain identical across research definitions.
    trials = [{"threshold": t, "correct": sum((t is not None and r["sim"] >= t) == r["gold"]
                                               for r in rows if r["split"] == "dev")} for t in grid]
    chosen = max(reversed(trials), key=lambda r: r["correct"])["threshold"]
    result = {"threshold": chosen, "development_trials": trials, "partitions": {}}
    for split in ("dev", "test"):
        selected = [r for r in rows if r["split"] == split]
        ceiling = 0
        for pair in pairs:
            if pair["split"] == split:
                labels = [v for v in pair["labels"].values() if v is not None]
                ceiling += max(labels.count(True), labels.count(False))
        result["partitions"][split] = {
            "name_tfidf": _metrics([dict(r, p=float(chosen is not None and r["sim"] >= chosen))
                                    for r in selected]),
            "rule_blind_oracle_correct": ceiling, "n": len(selected),
            "oracle_note": "Diagnostic upper bound for one fixed binary decision per pair across all rules; uses gold labels, not an executable model."}
    return result


def run(*, out, fixture_path=FIXTURE, live=False, budget=0, transport=None):
    validate_budget(budget)
    if (live and (budget is None or budget <= 0)) or (not live and budget != 0):
        raise ValueError("paid collection requires --live and a finite positive budget")
    fixture, out = load_fixture(fixture_path), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory contains artifacts; choose a new directory")
    frozen = requests(fixture)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "fixture.json", fixture)
    (out / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in frozen))
    manifest = {"frozen_at": datetime.now(timezone.utc).isoformat(), "fixture_sha256": digest(fixture),
                "code_sha256": {str(p): file_hash(p) for p in
                                (Path(__file__), Path("src/jlink/judge.py"), Path("src/jlink/core.py"))}}
    write_json(out / "frozen.json", manifest)
    rows, errors = [], []

    async def collect():
        backend, key = resolve_backend(fixture["model_settings"]["api"])
        client = Jev(key, backend, model=fixture["model_settings"]["model"], concurrency=1,
                     cache=None, transport=transport)
        raw = []

        async def capture(response):
            await response.aread()
            try:
                raw.append(response.json())
            except ValueError:
                raw.append({"http_status": response.status_code})

        client.http.event_hooks["response"] = [capture]
        try:
            if client.url != fixture["model_settings"]["endpoint"]:
                raise ValueError("configured endpoint differs from the frozen fixture")
            for req in frozen:
                if client.meter.cost >= budget:
                    break
                raw.clear()
                try:
                    result = await client.ask(req["payload"]["state"], req["payload"]["questions"])
                except (JevError, JevFatal) as exc:
                    errors.append({"request_sha256": req["request_sha256"], "error": str(exc),
                                   "provider_responses": list(raw)})
                    if isinstance(exc, JevFatal):
                        break
                    continue
                row = {"request_sha256": req["request_sha256"], "model": req["payload"]["model"],
                       "recorded_at": datetime.now(timezone.utc).isoformat(), "response": result,
                       "provider_response": raw[-1],
                       "evidence_kind": "offline_fake" if transport is not None else "live_model"}
                # Validate binding and parsed probability before treating a response as evidence.
                evaluate(fixture, [row])
                rows.append(row)
                with (out / "responses.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            await client.close()
        return client.meter

    meter = asyncio.run(collect()) if live else None
    report = evaluate(fixture, rows)
    report.update(baselines=baselines(fixture), budget_usd=budget, errors=errors,
                  new_api_calls=meter.calls if meter and transport is None else 0,
                  new_api_cost_usd=meter.cost if meter and transport is None else 0,
                  resolved_models=meter.resolved_models if meter else [],
                  cost_sources=meter.cost_sources if meter else {},
                  limitation="Curated, source-backed judging pilot; not independently adjudicated, not raw data or end-to-end linkage.")
    pd.DataFrame(report["decisions"]).to_csv(out / "decisions.csv", index=False)
    report["artifacts"] = {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE, dest="fixture_path")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--budget", type=float, default=0)
    report = run(**vars(parser.parse_args()))
    print(json.dumps({k: report[k] for k in ("status", "responses", "missing_responses",
                                            "new_api_cost_usd", "metrics")}, indent=2))

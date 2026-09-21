"""Run the frozen synthetic rule experiment with explicit spending and raw response preservation."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.heldout import file_hash, write_json
from bench.rule_sensitivity import FIXTURE, evaluate, load_fixture, requests
from jlink.core import Jev, JevError, JevFatal, resolve_backend
from jlink.judge import validate_budget


def run(*, out: Path, live: bool = False, budget: float = 0, fixture_path: Path = FIXTURE,
        transport=None):
    validate_budget(budget)
    if not live or budget is None or budget <= 0:
        raise ValueError("rule collection requires --live and a finite positive --budget")
    fixture = load_fixture(fixture_path)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} already contains artifacts; choose a new output directory")
    backend, key = resolve_backend(fixture["model_settings"]["api"])
    frozen = requests(fixture)
    rows, errors = [], []
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "fixture.json", fixture)
    (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in frozen))

    async def collect():
        client = Jev(key, backend, model=fixture["model_settings"]["model"],
                     concurrency=1, transport=transport)
        raw = []

        async def capture(response):
            await response.aread()
            try:
                raw.append(response.json())
            except ValueError:
                raw.append({"status": response.status_code, "body": response.text})

        client.http.event_hooks["response"] = [capture]
        try:
            if client.url != fixture["model_settings"]["endpoint"]:
                raise ValueError("configured endpoint differs from frozen fixture; remove JEV_URL override")
            for request in frozen:
                if client.meter.cost >= budget:
                    break
                raw.clear()
                payload = request["payload"]
                try:
                    answers = await client.ask(payload["state"], payload["questions"])
                except (JevError, JevFatal) as exc:
                    errors.append({"request_sha256": request["request_sha256"], "error": str(exc),
                                   "provider_responses": list(raw)})
                    if isinstance(exc, JevFatal):
                        break
                    continue
                row = {"request_sha256": request["request_sha256"], "model": payload["model"],
                       "recorded_at": datetime.now(timezone.utc).isoformat(), "response": answers,
                       "provider_response": raw[-1],
                       "evidence_kind": "offline_fake" if transport is not None else "live_model"}
                rows.append(row)
                with (out / "responses.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            await client.close()
        return client.meter

    meter = asyncio.run(collect())
    report = evaluate(fixture, rows)
    report.update(new_api_calls=meter.calls if transport is None else 0,
                  new_api_cost_usd=meter.cost if transport is None else 0,
                  budget_usd=budget, errors=errors, resolved_models=meter.resolved_models,
                  synthetic_fixture=True, cost_sources=meter.cost_sources)
    report["code_hashes"] = {str(p): file_hash(p) for p in
                             (Path(__file__), Path(__file__).with_name("rule_sensitivity.py"))}
    report["artifacts"] = {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--budget", type=float, default=0)
    parser.add_argument("--fixture", type=Path, default=FIXTURE, dest="fixture_path")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()

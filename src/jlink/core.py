"""jlink compatibility adapter for the shared JevKit implementation.

Prompts, cache identity, answer reuse, and budget policy retain their existing contracts.
Transport, configuration, storage, validation, and usage parsing come from jevkit_core.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from jevkit_core import (
    FATAL,
    RETRYABLE,
    PRICE_PER_MTOK,
    AnswerCache,
    Backend as _Backend,
    DecisionClient,
    JevBudgetExceeded,
    JevError,
    JevFatal,
    Meter as _Meter,
    answer_provenance,
    backend_catalog,
    cache_path,
    config_dir,
    digest,
    parse_usage,
    resolve_backend as _resolve_backend,
)


Backend = _Backend


BACKENDS = backend_catalog("typesafe", "openrouter")


def resolve_backend(name: str | None = None, *, require_key: bool = True):
    return _resolve_backend(BACKENDS, name, require_key=require_key)


class Cache(AnswerCache):
    """Keep the existing cache identity while sharing its storage implementation."""

    metadata = True

    @staticmethod
    def key(model: str, state, question: dict) -> str:
        return digest([model, state, question])


@dataclass
class Meter(_Meter):
    provider: str = ""

    requested_model: str = ""

    answer_provenance: list[dict] = field(default_factory=list)

    cost_sources: dict[str, int] = field(default_factory=dict)

    @property
    def resolved_models(self) -> list[str]:
        return sorted(
            {p["resolved_model"] for p in self.answer_provenance if p["resolved_model"]}
        )

    @property
    def unknown_model_answers(self) -> int:
        return sum(
            p["count"] for p in self.answer_provenance if not p["resolved_model"]
        )

    def note_answer(self, metadata: dict, source: str) -> None:
        """Summarize actual answer origins, including cache entries from older model aliases."""
        item = {
            k: metadata.get(k)
            for k in ("provider", "requested_model", "resolved_model")
        }
        item["source"] = source
        for previous in self.answer_provenance:
            if all(previous[k] == value for k, value in item.items()):
                previous["count"] += 1
                break
        else:
            self.answer_provenance.append(item | {"count": 1})
        models = self.resolved_models
        self.model = (
            models[0] if len(models) == 1 and not self.unknown_model_answers else ""
        )


class Jev(DecisionClient):
    def __init__(
        self,
        key: str,
        backend: Backend | str = "openrouter",
        *,
        model: str | None = None,
        timeout: float = 15.0,
        attempts: int = 4,
        concurrency: int = 32,
        cache: Cache | None = None,
        transport=None,
    ):
        backend = BACKENDS[backend] if isinstance(backend, str) else backend
        meter = Meter(
            provider=backend.name,
            requested_model=model or os.environ.get("JEV_MODEL") or backend.model,
        )
        super().__init__(
            key,
            backend,
            model=model,
            timeout=timeout,
            attempts=attempts,
            concurrency=concurrency,
            cache=cache,
            transport=transport,
            meter=meter,
        )

    async def ask(
        self,
        state,
        questions: dict[str, dict],
        *,
        allow_paid: bool = True,
        provenance: dict | None = None,
    ) -> dict[str, dict]:
        """Answer questions, optionally cache-only, keeping the historical answer return shape.

        `allow_paid=False` still permits a cache hit or sharing a request already in flight.
        The optional `provenance` dictionary receives metadata per question, without changing answers.
        """
        keys = {qid: Cache.key(self.model, state, q) for qid, q in questions.items()}
        answers, origins = {}, {}
        if self.cache:
            for qid, k in keys.items():
                if (hit := self.cache.get_entry(k)) is not None:
                    answers[qid], metadata = hit
                    origins[qid] = metadata | {"source": "cache"}
        misses = {qid: q for qid, q in questions.items() if qid not in answers}
        if not misses:
            self.meter.cached += 1
        else:
            def start():
                if not allow_paid:
                    raise JevBudgetExceeded(
                        "a new paid request is not allowed by the budget"
                    )
                return self._call(state, misses)

            task, started = self.share_request(
                (keys[qid] for qid in misses), start
            )
            source = "api" if started else "shared"
            by_key, metadata = await task
            answers.update({qid: by_key[keys[qid]] for qid in misses})
            origins.update({qid: metadata | {"source": source} for qid in misses})
        for item in origins.values():
            self.meter.note_answer(item, item["source"])
        if provenance is not None:
            provenance.update(origins)
        return answers

    def _record(
        self, state, questions: dict, data: dict, seconds: float
    ) -> tuple[dict[str, dict], dict]:
        usage = parse_usage(data.get("usage"), price_per_mtok=PRICE_PER_MTOK)
        self.meter.record(usage, seconds)
        cost_source = usage.source
        self.meter.cost_sources[cost_source] = (
            self.meter.cost_sources.get(cost_source, 0) + 1
        )
        metadata = answer_provenance(
            provider=self.backend.name,
            requested_model=self.model,
            resolved_model=data.get("model"),
        )
        out = {}
        for qid, q in questions.items():
            if qid not in data["answers"]:
                raise JevError(f"no answer returned for question {qid!r}")
            k = Cache.key(self.model, state, q)
            out[k] = data["answers"][qid]
            if self.cache:
                self.cache.put(k, out[k], metadata=metadata)
        return out, metadata


__all__ = [
    "BACKENDS",
    "Backend",
    "Cache",
    "Meter",
    "Jev",
    "JevError",
    "JevFatal",
    "JevBudgetExceeded",
    "PRICE_PER_MTOK",
    "RETRYABLE",
    "FATAL",
    "cache_path",
    "config_dir",
    "resolve_backend",
]

"""Client for TypeSafe's Jev decision model: two backends, an answer cache and a cost meter.

Jev can be reached through TypeSafe's own API or through OpenRouter. Both take one state and any
number of questions per call and return one typed answer per question. Answers are cached per
(model, state, question), so packing questions into a call and rerunning a command are both cheap.

Historically shared verbatim with jgrep. jlink's additive provenance and cache-only request
extensions are documented in docs/run-provenance.md; the old client/cache interfaces still work.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

RETRYABLE = {408, 429, 500, 502, 503, 504, 529}
FATAL = {401, 402, 403}
# TypeSafe's API reports tokens but not cost; OpenRouter reports both.
PRICE_PER_MTOK = float(os.environ.get("JEV_PRICE_PER_MTOK", 0.042))


class JevError(Exception):
    """One request failed; the rest of the run can continue."""


class JevFatal(Exception):
    """Nothing will work until the user fixes something, such as a bad key or no credits."""


class JevBudgetExceeded(Exception):
    """An answer was not cached or in flight, and a new paid request was not allowed."""


@dataclass(frozen=True)
class Backend:
    name: str
    url: str
    model: str
    key_env: str

    @property
    def key_file(self) -> Path:
        return config_dir() / f"{self.name}.key"

    def key(self) -> str | None:
        if os.environ.get(self.key_env):
            return os.environ[self.key_env].strip()
        return self.key_file.read_text().strip() if self.key_file.exists() else None


# Order matters: with keys for both, TypeSafe's own API is used.
BACKENDS = {
    "typesafe": Backend("typesafe", "https://api.typesafe.ai/v1/systemone", "jev-latest", "TYPESAFE_API_KEY"),
    "openrouter": Backend("openrouter", "https://openrouter.ai/api/alpha/decisions", "~typesafe/jev-latest",
                          "OPENROUTER_API_KEY"),
}


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "jev"


def cache_path() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "jev" / "answers.sqlite"


def resolve_backend(name: str | None = None, *, require_key: bool = True) -> tuple[Backend, str]:
    """The API to use and its key. A name (or JEV_API) wins; otherwise the first backend with a key."""
    name = name or os.environ.get("JEV_API")
    if name:
        if name not in BACKENDS:
            raise JevFatal(f"unknown API {name!r}; choose from {', '.join(BACKENDS)}")
        backend = BACKENDS[name]
        if not (key := backend.key()) and require_key:
            raise JevFatal(f"no key for {name}. Set {backend.key_env} or put the key in {backend.key_file}")
        return backend, key or ""
    for backend in BACKENDS.values():
        if key := backend.key():
            return backend, key
    if not require_key:
        return next(iter(BACKENDS.values())), ""
    options = " or ".join(b.key_env for b in BACKENDS.values())
    raise JevFatal(f"no API key. Set {options}, or put a key in {config_dir()}/<api>.key")


class Cache:
    """Answers on disk, keyed on the exact model, state and question."""

    def __init__(self, path: Path | None = None):
        path = path or cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
        self._lock = threading.RLock()
        # WAL lets two tools in one pipeline share the file.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS answers "
                        "(key TEXT PRIMARY KEY, answer TEXT NOT NULL, at REAL NOT NULL) WITHOUT ROWID")
        # A side table keeps the three-column answers table readable/writable by older clients.
        self.db.execute("CREATE TABLE IF NOT EXISTS answer_metadata "
                        "(key TEXT PRIMARY KEY, at REAL NOT NULL, metadata TEXT NOT NULL) WITHOUT ROWID")

    @staticmethod
    def key(model: str, state, question: dict) -> str:
        blob = json.dumps([model, state, question], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self.db.execute("SELECT answer FROM answers WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_entry(self, key: str) -> tuple[dict, dict] | None:
        """Return an answer and its recorded provenance; legacy answers have empty metadata."""
        with self._lock:
            row = self.db.execute(
                "SELECT a.answer, m.metadata FROM answers a LEFT JOIN answer_metadata m "
                "ON a.key = m.key AND a.at = m.at WHERE a.key = ?", (key,)).fetchone()
        return (json.loads(row[0]), json.loads(row[1]) if row[1] else {}) if row else None

    def put(self, key: str, answer: dict, *, metadata: dict | None = None) -> None:
        encoded = json.dumps(answer)
        encoded_metadata = json.dumps(metadata) if metadata is not None else None
        with self._lock:
            at = time.time()
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?)", (key, encoded, at))
                if encoded_metadata is not None:
                    self.db.execute("INSERT OR REPLACE INTO answer_metadata VALUES (?, ?, ?)",
                                    (key, at, encoded_metadata))
                else:
                    self.db.execute("DELETE FROM answer_metadata WHERE key = ?", (key,))
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise


@dataclass
class Meter:
    calls: int = 0
    cached: int = 0
    retries: int = 0
    input_tokens: int = 0
    cost: float = 0.0
    model: str = ""  # the model the API says answered, which resolves aliases like jev-latest
    latencies: list[float] = field(default_factory=list)
    provider: str = ""
    requested_model: str = ""
    answer_provenance: list[dict] = field(default_factory=list)
    cost_sources: dict[str, int] = field(default_factory=dict)

    @property
    def resolved_models(self) -> list[str]:
        return sorted({p["resolved_model"] for p in self.answer_provenance if p["resolved_model"]})

    @property
    def unknown_model_answers(self) -> int:
        return sum(p["count"] for p in self.answer_provenance if not p["resolved_model"])

    def note_answer(self, metadata: dict, source: str) -> None:
        """Summarize actual answer origins, including cache entries from older model aliases."""
        item = {k: metadata.get(k) for k in ("provider", "requested_model", "resolved_model")}
        item["source"] = source
        for previous in self.answer_provenance:
            if all(previous[k] == value for k, value in item.items()):
                previous["count"] += 1
                break
        else:
            self.answer_provenance.append(item | {"count": 1})
        models = self.resolved_models
        self.model = models[0] if len(models) == 1 and not self.unknown_model_answers else ""

    def summary(self) -> str:
        parts = [f"{self.calls:,} calls, {self.cached:,} cached"]
        if self.retries:
            parts.append(f"{self.retries:,} retries")
        if self.calls:
            parts.append(f"{self.input_tokens:,} tokens")
            parts.append(f"${self.cost:.4f}")
        return "; ".join(parts)


class Jev:
    def __init__(self, key: str, backend: Backend | str = "openrouter", *, model: str | None = None,
                 timeout: float = 15.0, attempts: int = 4, concurrency: int = 32, cache: Cache | None = None,
                 transport=None):
        self.backend = BACKENDS[backend] if isinstance(backend, str) else backend
        self.model = model or os.environ.get("JEV_MODEL") or self.backend.model
        self.url = os.environ.get("JEV_URL") or self.backend.url
        self.timeout, self.attempts, self.cache = timeout, attempts, cache
        self.meter = Meter(provider=self.backend.name, requested_model=self.model)
        self._flights: dict[str, asyncio.Task] = {}
        self.http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}", "X-Title": "jev tools"},
            limits=httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4),
            transport=transport,
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def ask(self, state, questions: dict[str, dict], *, allow_paid: bool = True,
                  provenance: dict | None = None) -> dict[str, dict]:
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
            # Identical requests already in the air share one call; logs repeat themselves a lot.
            flight = "|".join(sorted(keys[qid] for qid in misses))
            task = self._flights.get(flight)
            source = "shared" if task is not None else "api"
            if task is None:
                if not allow_paid:
                    raise JevBudgetExceeded("a new paid request is not allowed by the budget")
                task = asyncio.ensure_future(self._call(state, misses))
                self._flights[flight] = task
                task.add_done_callback(lambda _: self._flights.pop(flight, None))
            else:
                self.meter.cached += 1
            by_key, metadata = await task
            answers.update({qid: by_key[keys[qid]] for qid in misses})
            origins.update({qid: metadata | {"source": source} for qid in misses})
        for item in origins.values():
            self.meter.note_answer(item, item["source"])
        if provenance is not None:
            provenance.update(origins)
        return answers

    async def _call(self, state, questions: dict[str, dict]) -> tuple[dict[str, dict], dict]:
        """One request, retried inside a time budget. Returns answers by cache key and provenance."""
        body = {"model": self.model, "state": state, "questions": questions}
        deadline = time.monotonic() + self.timeout
        last = "no attempt made"
        for attempt in range(self.attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            t0 = time.perf_counter()
            try:
                r = await self.http.post(self.url, json=body, timeout=remaining)
            except httpx.TransportError as e:
                last = type(e).__name__
            else:
                data = _json(r)
                if r.status_code == 200 and "answers" in data:
                    return self._record(state, questions, data, time.perf_counter() - t0)
                detail = _detail(data) or r.text[:200]
                if r.status_code in FATAL:
                    raise JevFatal(f"{self.backend.name} said {r.status_code}: {detail}")
                if r.status_code != 200 and r.status_code not in RETRYABLE:
                    raise JevError(f"HTTP {r.status_code}: {detail}")
                last = f"HTTP {r.status_code}"
            if attempt + 1 < self.attempts:
                self.meter.retries += 1
                pause = 0.2 * 2 ** attempt + random.random() * 0.1
                await asyncio.sleep(max(0.0, min(pause, deadline - time.monotonic())))
        raise JevError(f"gave up after {self.timeout:g}s ({last})")

    def _record(self, state, questions: dict, data: dict, seconds: float) -> tuple[dict[str, dict], dict]:
        usage = data.get("usage") or {}
        tokens = usage.get("input_tokens") or 0
        cost = usage.get("cost")
        self.meter.calls += 1
        self.meter.input_tokens += tokens
        self.meter.cost += tokens * PRICE_PER_MTOK / 1e6 if cost is None else cost
        cost_source = "estimated_from_tokens" if cost is None else "reported_by_api"
        self.meter.cost_sources[cost_source] = self.meter.cost_sources.get(cost_source, 0) + 1
        self.meter.latencies.append(seconds)
        resolved = data.get("model")
        metadata = {"version": 1, "provider": self.backend.name, "requested_model": self.model,
                    "resolved_model": resolved if isinstance(resolved, str) and resolved.strip() else None,
                    "answered_at": time.time()}
        out = {}
        for qid, q in questions.items():
            if qid not in data["answers"]:
                raise JevError(f"no answer returned for question {qid!r}")
            k = Cache.key(self.model, state, q)
            out[k] = data["answers"][qid]
            if self.cache:
                self.cache.put(k, out[k], metadata=metadata)
        return out, metadata


def _json(r: httpx.Response) -> dict:
    try:
        data = r.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _detail(data: dict) -> str:
    """The human-readable part of an error body. OpenRouter nests it under `error`, TypeSafe under `detail`,
    and `detail` may itself be a string, an object or a list of validation problems."""
    found = data.get("error", data.get("detail"))
    if isinstance(found, list):
        found = "; ".join(_detail({"detail": item}) for item in found)
    elif isinstance(found, dict):
        found = found.get("message") or found.get("msg") or json.dumps(found)
    return " ".join(str(found or "").split())[:200]

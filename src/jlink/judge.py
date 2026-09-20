"""Pair probabilities from Jev: one yes/no question per candidate pair.

The question is the user's match rule. The state is the two records, field by field. Pairs are
judged in descending cheap similarity, so a budget is spent on the likeliest pairs first.
Exact-text acceptance is opt-in: equal names need not mean equal entities.
"""

from __future__ import annotations

import asyncio
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .core import Cache, Jev, JevBudgetExceeded, JevError, JevFatal, Meter, resolve_backend
from .fields import check_columns, ids, normalize, parse_on

SCORE_COLUMNS = ["p", "source", "error"]
EXACT_POLICY = "all_fields_nonempty_and_equal_v1"


def validate_budget(budget: float | None) -> None:
    """None is unlimited; zero allows only exact, cached, or already in-flight answers."""
    if budget is not None and (isinstance(budget, (bool, np.bool_)) or not isinstance(budget, Real)
                               or not math.isfinite(budget) or budget < 0):
        raise ValueError("`budget` must be a finite nonnegative number of dollars, or None for unlimited")


def question(entity: str, definition: str = "") -> dict:
    text = f"Record A and record B refer to the same {entity.strip()}."
    if definition and definition.strip():
        text += " " + definition.strip()
    return {"type": "noul", "instructions": text}


def judge(candidates: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, *, on, entity: str,
          definition: str = "", left_id: str | None = None, right_id: str | None = None,
          api: str | None = None, model: str | None = None, concurrency: int = 32,
          budget: float | None = 5.0, cache: bool | str | Path | Cache = True, exact_shortcut: bool = False,
          progress: bool = True, transport=None) -> tuple[pd.DataFrame, Meter]:
    """Score every candidate pair. Returns the scores table (candidates plus p, source, error) and the meter."""
    validate_budget(budget)
    if not isinstance(exact_shortcut, (bool, np.bool_)):
        raise ValueError("`exact_shortcut` must be a boolean; enable only when equal fields establish identity")
    if isinstance(concurrency, (bool, np.bool_)) or not isinstance(concurrency, Integral) or concurrency < 1:
        raise ValueError("`concurrency` must be a positive integer")
    if not entity or not entity.strip():
        raise ValueError('`entity` says what a record is, for example "firm" or "person"; it cannot be empty')
    for column in ("left_id", "right_id", "sim"):
        if column not in candidates.columns:
            raise ValueError(f"candidates must have a {column!r} column; build them with jlink.block.candidates")
    fields = parse_on(on)
    check_columns(left, [lc for _, lc, _ in fields], "left")
    check_columns(right, [rc for _, _, rc in fields], "right")

    a = _records(left, ids(left, left_id, "left"), [(label, lc) for label, lc, _ in fields])
    b = _records(right, ids(right, right_id, "right"), [(label, rc) for label, _, rc in fields])
    for side, known, column in (("left", a, "left_id"), ("right", b, "right_id")):
        unknown = ~candidates[column].isin(known.index)
        if unknown.any():
            raise ValueError(f"candidates name a {column} that is not in the {side} data: "
                             f"{candidates.loc[unknown, column].iloc[0]!r}")

    scores = candidates.reset_index(drop=True).copy()
    left_ids, right_ids = scores["left_id"].to_numpy(), scores["right_id"].to_numpy()
    n = len(scores)
    p, source, error = np.full(n, np.nan), np.full(n, "unjudged", dtype=object), np.full(n, pd.NA, dtype=object)
    models, providers, origins = (np.full(n, pd.NA, dtype=object) for _ in range(3))
    answered_at = np.full(n, np.nan)

    if exact_shortcut and n:
        keys_a, keys_b = a["exact_key"].reindex(left_ids), b["exact_key"].reindex(right_ids)
        same = np.array([ka is not None and ka == kb for ka, kb in zip(keys_a, keys_b)], dtype=bool)
        p[same], source[same] = 1.0, "exact"

    todo = np.flatnonzero(source == "unjudged")
    todo = todo[np.argsort(-scores["sim"].to_numpy()[todo], kind="stable")]
    record_a, record_b = a["record"].to_dict(), b["record"].to_dict()
    backend, key = resolve_backend(api, require_key=bool(len(todo)) and budget != 0)
    store = cache if isinstance(cache, Cache) else Cache(Path(cache)) if isinstance(cache, (str, Path)) \
        else Cache() if cache else None
    jev = Jev(key, backend, model=model, concurrency=concurrency, cache=store, transport=transport)
    ask = question(entity, definition)

    async def work() -> None:
        sem = asyncio.Semaphore(concurrency)
        bar = tqdm(total=len(todo), desc="judging pairs", unit="pair", disable=not progress or not len(todo))
        tasks: set[asyncio.Task] = set()
        fatal: list[JevFatal] = []

        async def one(i: int) -> None:
            try:
                state = {"record_a": record_a[left_ids[i]], "record_b": record_b[right_ids[i]]}
                provenance = {}
                answer = await jev.ask(state, {"match": ask}, provenance=provenance,
                                       allow_paid=budget is None or jev.meter.cost < budget)
                p[i], source[i] = float(answer["match"]["noul"]), "jev"
                meta = provenance["match"]
                models[i], providers[i] = meta.get("resolved_model") or pd.NA, meta.get("provider") or pd.NA
                origins[i], answered_at[i] = meta["source"], meta.get("answered_at", np.nan)
            except JevBudgetExceeded:
                pass  # Leave this pair unjudged; later pairs may still be cached.
            except JevError as e:
                source[i], error[i] = "error", str(e)
            except JevFatal as e:  # bad key or no credits: stop launching calls
                fatal.append(e)
            finally:
                sem.release()
                bar.update(1)

        try:
            for i in todo:
                await sem.acquire()
                if fatal:
                    sem.release()
                    break
                task = asyncio.create_task(one(int(i)))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            await asyncio.gather(*tasks)
        finally:
            for task in list(tasks):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            bar.close()
            await jev.close()
        if fatal:
            raise fatal[0]

    _run(work())
    scores["p"], scores["source"], scores["error"] = p, source, error
    scores["model"], scores["provider"], scores["score_origin"] = models, providers, origins
    scores["answered_at"] = answered_at
    left_over = int((source == "unjudged").sum())
    if left_over:
        warnings.warn(f"the ${budget:.2f} budget ran out with {left_over:,} of {n:,} pairs unjudged; "
                      "new requests were prioritized by similarity; cached and exact scores were retained. "
                      "Raise `budget` to judge more pairs.", stacklevel=2)
    errors = int((source == "error").sum())
    if errors:
        warnings.warn(f"{errors:,} pairs failed and have no probability; the first error was: "
                      f"{error[source == 'error'][0]}", stacklevel=2)
    return scores, jev.meter


def _records(frame: pd.DataFrame, index: pd.Index, fields: list[tuple[str, str]]) -> pd.DataFrame:
    """Per ID: the judge's record and a fieldwise key, absent if any field normalizes to empty."""
    columns = [c for _, c in fields]
    rows = frame[columns].to_dict("records")
    record = [{label: v for (label, c) in fields if (v := _clean(row[c])) is not None} for row in rows]
    keys = [tuple(normalize(_clean(row[c])) for c in columns) for row in rows]
    return pd.DataFrame({"record": record, "exact_key": [key if all(key) else None for key in keys]}, index=index)


def _clean(value):
    """A JSON-ready value, or None if missing. Whole floats become ints: a year read as 1985.0 is 1985."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value) if value.is_integer() and abs(value) < 1e15 else value
    if isinstance(value, (bool, int)):
        return value
    text = str(value).strip()
    return text or None


def _run(coro):
    """Run a coroutine to completion, including inside Jupyter, where a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()

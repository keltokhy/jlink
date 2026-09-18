"""Pair probabilities from Jev: one yes/no question per candidate pair.

The question is the user's match rule. The state is the two records, field by field. Pairs are
judged in descending cheap similarity, so a budget is spent on the likeliest pairs first, and
pairs whose fields are identical after normalization are settled without a call.
"""

from __future__ import annotations

import asyncio
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .core import Cache, Jev, JevError, JevFatal, Meter, resolve_backend
from .fields import check_columns, ids, parse_on, record_text

SCORE_COLUMNS = ["p", "source", "error"]


def question(entity: str, definition: str = "") -> dict:
    text = f"Record A and record B refer to the same {entity.strip()}."
    if definition and definition.strip():
        text += " " + definition.strip()
    return {"type": "noul", "instructions": text}


def judge(candidates: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, *, on, entity: str,
          definition: str = "", left_id: str | None = None, right_id: str | None = None,
          api: str | None = None, model: str | None = None, concurrency: int = 32,
          budget: float | None = 5.0, cache: bool | str | Path | Cache = True, exact_shortcut: bool = True,
          progress: bool = True, transport=None) -> tuple[pd.DataFrame, Meter]:
    """Score every candidate pair. Returns the scores table (candidates plus p, source, error) and the meter."""
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

    if exact_shortcut and n:
        text_a, text_b = a["text"].reindex(left_ids).to_numpy(), b["text"].reindex(right_ids).to_numpy()
        same = (text_a == text_b) & (text_a != "")
        p[same], source[same] = 1.0, "exact"

    todo = np.flatnonzero(source == "unjudged")
    todo = todo[np.argsort(-scores["sim"].to_numpy()[todo], kind="stable")]
    record_a, record_b = a["record"].to_dict(), b["record"].to_dict()
    backend, key = resolve_backend(api)
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
                answer = await jev.ask(state, {"match": ask})
                p[i], source[i] = float(answer["match"]["noul"]), "jev"
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
                if fatal or (budget and jev.meter.cost >= budget):
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
    left_over = int((source == "unjudged").sum())
    if left_over:
        warnings.warn(f"the ${budget:.2f} budget ran out with {left_over:,} of {n:,} pairs unjudged; "
                      "the least similar pairs were left. Raise `budget`, or rerun: judged pairs are cached "
                      "and cost nothing the second time.", stacklevel=2)
    errors = int((source == "error").sum())
    if errors:
        warnings.warn(f"{errors:,} pairs failed and have no probability; the first error was: "
                      f"{error[source == 'error'][0]}", stacklevel=2)
    return scores, jev.meter


def _records(frame: pd.DataFrame, index: pd.Index, fields: list[tuple[str, str]]) -> pd.DataFrame:
    """Per ID: the record as the judge sees it, and its normalized text for the exact-match shortcut."""
    columns = [c for _, c in fields]
    rows = frame[columns].to_dict("records")
    record = [{label: v for (label, c) in fields if (v := _clean(row[c])) is not None} for row in rows]
    return pd.DataFrame({"record": record, "text": record_text(frame, columns).to_numpy()}, index=index)


def _clean(value):
    """A JSON-ready value, or None if missing. Whole floats become ints: a year read as 1985.0 is 1985."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
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

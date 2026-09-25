"""Pair probabilities from Jev: one yes/no question per candidate pair.

The question is the user's match rule. The state is the two records, field by field. Pairs are
judged in descending cheap similarity, so a budget is spent on the likeliest pairs first.
Exact-text acceptance is opt-in: equal names need not mean equal entities.
"""

from __future__ import annotations

import asyncio
import math
import warnings
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from jevkit_runtime import (
    AnswerStore,
    Budget,
    Client,
    JevBudgetExceeded,
    JevError,
    JevFatal,
    Meter,
    Noul,
    Run,
    estimate_tokens,
    request_body,
    resolve,
)
from jevkit_runtime.cli import run_sync

from . import __version__
from .core import PROVIDERS
from .fields import check_columns, clean, ids, normalize, parse_on, side_fields

SCORE_COLUMNS = ["p", "source", "error"]
EXACT_POLICY = "all_fields_nonempty_and_equal_v1"
STYLES = ("identity", "rule")


def validate_budget(budget: float | Budget | None) -> None:
    """None is unlimited; zero allows only exact, cached, or already in-flight answers."""
    if isinstance(budget, Budget):
        return
    if budget is not None and (isinstance(budget, (bool, np.bool_)) or not isinstance(budget, Real)
                               or not math.isfinite(budget) or budget < 0):
        raise ValueError("`budget` must be a finite nonnegative number of dollars, None for unlimited, "
                         "or a jevkit_runtime.Budget")


def as_budget(budget: float | Budget | None) -> Budget:
    """The runtime budget a `budget=` argument stands for: dollars, None for no limit, or a Budget."""
    validate_budget(budget)
    if isinstance(budget, Budget):
        return budget
    return Budget(math.inf if budget is None else float(budget))


def question(entity: str, definition: str = "", *, style: str = "identity") -> Noul:
    """Build the match proposition; rule style lets the definition name the relation."""
    if style not in {"identity", "rule"}:
        raise ValueError("question style must be 'identity' or 'rule'")
    if style == "rule" and definition and definition.strip():
        return Noul("Record A and record B satisfy the following match rule. " + definition.strip())
    text = f"Record A and record B refer to the same {entity.strip()}."
    if definition and definition.strip():
        text += " " + definition.strip()
    return Noul(text)


def validate_question(entity: str | None, definition: str, style: str) -> None:
    """Identity questions need an entity; rule questions need the definition that states the relation."""
    if style not in STYLES:
        raise ValueError(f"`style` must be one of {', '.join(STYLES)}; got {style!r}")
    if style == "rule":
        if not definition or not definition.strip():
            raise ValueError('`style="rule"` asks whether two records satisfy your `definition`, '
                             "so the definition cannot be empty")
    elif not entity or not entity.strip():
        raise ValueError('`entity` says what a record is, for example "firm" or "person"; it cannot be empty')


def judge(candidates: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, *, on,
          entity: str | None = None, definition: str = "", style: str = "identity",
          left_id: str | None = None, right_id: str | None = None,
          api: str | None = None, model: str | None = None, concurrency: int = 32,
          budget: float | Budget | None = 5.0, cache: bool | str | Path | AnswerStore = True,
          exact_shortcut: bool = False,
          progress: bool = True, transport=None) -> tuple[pd.DataFrame, Meter]:
    """Score every candidate pair. Returns the scores table (candidates plus p, source, error) and the meter.

    ``scores.attrs["question"]`` holds the proposition exactly as it was put to the model, and
    ``scores.attrs["run"]`` the runtime's record of the judging: who answered, what it cost, the budget.
    ``budget`` is dollars, None for no limit, or a jevkit_runtime Budget; requests go out in similarity
    order, each setting its estimated price aside first, so the budget is spent on the likeliest pairs.
    ``style="rule"`` asks whether the pair satisfies ``definition``, a relation that need not be
    identity; ``entity`` is then not part of the question. One-sided ``on`` fields, ``(left, None)``
    or ``(None, right)``, appear only in that side's record.
    """
    budget = as_budget(budget)
    if not isinstance(exact_shortcut, (bool, np.bool_)):
        raise ValueError("`exact_shortcut` must be a boolean; enable only when equal fields establish identity")
    if isinstance(concurrency, (bool, np.bool_)) or not isinstance(concurrency, Integral) or concurrency < 1:
        raise ValueError("`concurrency` must be a positive integer")
    validate_question(entity, definition, style)
    for column in ("left_id", "right_id", "sim"):
        if column not in candidates.columns:
            raise ValueError(f"candidates must have a {column!r} column; build them with jlink.block.candidates")
    fields = parse_on(on, unpaired=True)
    shown_left, shown_right = side_fields(fields, "left"), side_fields(fields, "right")
    if exact_shortcut and any(lc is None or rc is None for _, lc, rc in fields):
        raise ValueError("`exact_shortcut` accepts pairs whose fields are all equal, so every `on` field "
                         "must exist on both sides; remove the one-sided fields or the shortcut")
    check_columns(left, [c for _, c in shown_left], "left")
    check_columns(right, [c for _, c in shown_right], "right")

    a = _records(left, ids(left, left_id, "left"), shown_left)
    b = _records(right, ids(right, right_id, "right"), shown_right)
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
    backend = resolve(PROVIDERS, api, model=model, require_key=bool(len(todo)) and budget.limit != 0)
    store = cache if isinstance(cache, AnswerStore) else AnswerStore(Path(cache)) if isinstance(cache, (str, Path)) \
        else AnswerStore() if cache else None
    jev = Client(backend, concurrency=concurrency, store=store, budget=budget, transport=transport)
    run = Run("jlink", __version__)
    ask = question(entity or "", definition, style=style)

    async def work() -> None:
        sem = asyncio.Semaphore(concurrency)
        bar = tqdm(total=len(todo), desc="judging pairs", unit="pair", disable=not progress or not len(todo))
        tasks: set[asyncio.Task] = set()
        fatal: list[JevFatal] = []

        async def one(i: int) -> None:
            try:
                state = {"record_a": record_a[left_ids[i]], "record_b": record_b[right_ids[i]]}
                answer = await jev.ask(state, {"match": ask})
                p[i], source[i] = ask.value(answer["match"]), "jev"
                meta = answer.origins["match"]
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

    run_sync(work())
    scores["p"], scores["source"], scores["error"] = p, source, error
    scores["model"], scores["provider"], scores["score_origin"] = models, providers, origins
    scores["answered_at"] = answered_at
    # The one place the proposition is built. Callers quote this text, never a rebuilt copy, so a
    # report or methods paragraph cannot name a sentence the model did not see.
    scores.attrs["question"] = ask.text
    scores.attrs["run"] = run.record(jev)
    left_over = int((source == "unjudged").sum())
    if left_over:
        warnings.warn(f"the ${budget.limit:.2f} budget ran out with {left_over:,} of {n:,} pairs unjudged; "
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
    record = [{label: v for (label, c) in fields if (v := clean(row[c])) is not None} for row in rows]
    keys = [tuple(normalize(clean(row[c])) for c in columns) for row in rows]
    return pd.DataFrame({"record": record, "exact_key": [key if all(key) else None for key in keys]}, index=index)


def pair_tokens(candidates: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, *, on, entity: str | None,
                definition: str, style: str, left_id: str | None, right_id: str | None, model: str,
                sample: int = 200, seed: int = 0) -> float | None:
    """The runtime's token estimate for a judged pair, averaged over a sample of these candidates' own
    records; None without candidates."""
    if not len(candidates):
        return None
    fields = parse_on(on, unpaired=True)
    a = _records(left, ids(left, left_id, "left"), side_fields(fields, "left"))["record"].to_dict()
    b = _records(right, ids(right, right_id, "right"), side_fields(fields, "right"))["record"].to_dict()
    pairs = candidates.sample(n=min(sample, len(candidates)), random_state=seed)
    ask = {"match": question(entity or "", definition, style=style)}
    sizes = [estimate_tokens(request_body(model, {"record_a": a[l], "record_b": b[r]}, ask))
             for l, r in zip(pairs["left_id"], pairs["right_id"]) if l in a and r in b]
    return float(np.mean(sizes)) if sizes else None

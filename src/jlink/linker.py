"""The front door: describe the match, link two tables, inspect and document the result."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from . import __version__, audit, block
from .core import Meter
from .fields import ids, parse_on
from .judge import judge, question
from .resolve import resolve

_KEEP = object()  # relink: "leave this setting as it is", distinct from None, which turns the margin off
HOWS = ("one-to-one", "many-to-one", "one-to-many", "many-to-many")
# Measured on OpenRouter in September 2026; used only for estimates.
TOKENS_PER_PAIR, PAIRS_PER_SECOND, PRICE_PER_MTOK = 330, 200, 0.042


class Linker:
    """A match rule and how to find candidates for it.

        linker = jlink.Linker(entity="firm", on=["name", "state"],
                              definition="A parent company and its subsidiary are different firms.")
        result = linker.link(compustat, patents, left_id="gvkey", right_id="assignee_id")
    """

    def __init__(self, entity: str, on, definition: str = "", blockers: list | None = None, *,
                 api: str | None = None, model: str | None = None, concurrency: int = 32,
                 cache=True, exact_shortcut: bool = True):
        if not entity or not entity.strip():
            raise ValueError('`entity` says what a record is, for example "firm" or "person"; it cannot be empty')
        self.entity, self.definition, self.on = entity.strip(), (definition or "").strip(), on
        self.fields = parse_on(on)
        self.blockers = blockers
        self.api, self.model, self.concurrency = api, model, concurrency
        self.cache, self.exact_shortcut = cache, exact_shortcut

    def candidates(self, left: pd.DataFrame, right: pd.DataFrame, *, left_id=None, right_id=None,
                   max_pairs: int | None = 5_000_000) -> pd.DataFrame:
        return block.candidates(left, right, on=self.on, blockers=self.blockers, left_id=left_id,
                                right_id=right_id, max_pairs=max_pairs)

    def estimate(self, left: pd.DataFrame, right: pd.DataFrame, *, left_id=None, right_id=None) -> dict:
        """Run blocking only and say what judging would cost. No API calls."""
        pairs = len(self.candidates(left, right, left_id=left_id, right_id=right_id))
        return {"left": len(left), "right": len(right), "pairs": pairs,
                "dollars": round(pairs * TOKENS_PER_PAIR * PRICE_PER_MTOK / 1e6, 4),
                "seconds": round(pairs / PAIRS_PER_SECOND, 1)}

    def link(self, left: pd.DataFrame, right: pd.DataFrame, *, left_id: str | None = None,
             right_id: str | None = None, how: str = "one-to-one", threshold: float = 0.5,
             min_margin: float | None = None, budget: float | None = 5.0, max_pairs: int | None = 5_000_000,
             progress: bool = True, transport=None) -> "Result":
        _check_how(how, threshold)
        ids(left, left_id, "left"), ids(right, right_id, "right")
        t0 = time.perf_counter()
        cands = self.candidates(left, right, left_id=left_id, right_id=right_id, max_pairs=max_pairs)
        scores, meter = judge(cands, left, right, on=self.on, entity=self.entity, definition=self.definition,
                              left_id=left_id, right_id=right_id, api=self.api, model=self.model,
                              concurrency=self.concurrency, budget=budget, cache=self.cache,
                              exact_shortcut=self.exact_shortcut, progress=progress, transport=transport)
        links = resolve(scores, how=how, threshold=threshold, min_margin=min_margin)
        settings = {
            "jlink": __version__, "date": date.today().isoformat(), "entity": self.entity,
            "definition": self.definition, "question": question(self.entity, self.definition)["instructions"],
            "on": [[lc, rc] for _, lc, rc in self.fields], "left_id": left_id, "right_id": right_id,
            "blockers": [b.name for b in self.blockers] if self.blockers else ["ngrams (default, k=10)"],
            "how": how, "threshold": threshold, "min_margin": min_margin, "budget": budget,
            "n_left": len(left), "n_right": len(right), "model": meter.model or self.model or "",
            "calls": meter.calls, "cached": meter.cached, "input_tokens": meter.input_tokens,
            "dollars": round(meter.cost, 6), "seconds": round(time.perf_counter() - t0, 1),
        }
        return Result(links, scores, settings, meter, _left=left, _right=right)


def link(left: pd.DataFrame, right: pd.DataFrame, *, entity: str, on, definition: str = "", blockers=None,
         **kwargs) -> "Result":
    """One call for the common case. Keyword arguments are those of `Linker.link`."""
    return Linker(entity, on, definition, blockers).link(left, right, **kwargs)


@dataclass
class Result:
    links: pd.DataFrame
    scores: pd.DataFrame
    settings: dict
    meter: Meter = field(default_factory=Meter)
    _left: pd.DataFrame | None = field(default=None, repr=False)
    _right: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def candidates(self) -> pd.DataFrame:
        return self.scores[["left_id", "right_id", "block", "sim"]]

    def relink(self, *, how: str | None = None, threshold: float | None = None,
               min_margin: float | None = _KEEP) -> "Result":
        """Choose links again under different rules. Uses the probabilities already paid for.

        `min_margin=None` removes a margin requirement; leave it out to keep the current one.
        """
        s = dict(self.settings)
        s["how"] = how or s["how"]
        s["threshold"] = s["threshold"] if threshold is None else threshold
        s["min_margin"] = s["min_margin"] if min_margin is _KEEP else min_margin
        _check_how(s["how"], s["threshold"])
        links = resolve(self.scores, how=s["how"], threshold=s["threshold"], min_margin=s["min_margin"])
        return Result(links, self.scores, s, self.meter, self._left, self._right)

    def merged(self, left: pd.DataFrame | None = None, right: pd.DataFrame | None = None,
               suffixes: tuple[str, str] = ("_left", "_right")) -> pd.DataFrame:
        """The linked rows side by side: every left and right column, plus p and margin."""
        left, right = self._frames(left, right)
        a = _with_id(left, self.settings["left_id"], "left_id")
        b = _with_id(right, self.settings["right_id"], "right_id")
        out = self.links[["left_id", "right_id", "p", "margin"]].merge(a, on="left_id").merge(
            b, on="right_id", suffixes=suffixes)
        return out

    def audit_sample(self, n: int = 200, *, seed: int = 0, left=None, right=None, **kwargs) -> pd.DataFrame:
        """A stratified sample of judged pairs to label by hand. Pass the labeled file to `jlink.evaluate`."""
        try:
            left, right = self._frames(left, right)
        except ValueError:
            left = right = None
        on = [(lc, rc) for lc, rc in self.settings["on"]]
        return audit.audit_sample(self.scores, n=n, seed=seed, left=left, right=right, on=on if left is not None else None,
                                  left_id=self.settings["left_id"], right_id=self.settings["right_id"], **kwargs)

    def report(self) -> str:
        s, sc = self.settings, self.scores
        by_source = sc["source"].value_counts()
        by_block = sc["block"].value_counts()
        lines = [
            f"jlink {s['jlink']}, {s['date']}",
            f"Rule: {s['question']}",
            f"Compared on: {', '.join(lc if lc == rc else f'{lc} = {rc}' for lc, rc in s['on'])}",
            f"Records: {s['n_left']:,} left, {s['n_right']:,} right",
            f"Candidate pairs: {len(sc):,} ({'; '.join(f'{k}: {v:,}' for k, v in by_block.items())})",
            "Judged: " + ", ".join(f"{by_source.get(k, 0):,} {label}" for k, label in (
                ("jev", "by Jev"), ("exact", "identical after normalization"), ("error", "failed"),
                ("unjudged", "left unjudged by the budget")) if by_source.get(k, 0)),
            f"Links: {len(self.links):,} ({s['how']}, p >= {s['threshold']:g}"
            + (f", margin >= {s['min_margin']:g}" if s["min_margin"] is not None else "") + ")",
        ]
        if len(self.links):
            p = self.links["p"]
            lines.append(f"Link probabilities: median {p.median():.2f}; {int((p >= 0.95).sum()):,} at 0.95 or above; "
                         f"{int((p < 0.8).sum()):,} below 0.8")
            lines.append(f"Left records linked: {self.links['left_id'].nunique():,} of {s['n_left']:,} "
                         f"({self.links['left_id'].nunique() / max(s['n_left'], 1):.1%})")
        lines.append(f"Model: {s['model'] or 'unknown'}; {s['calls']:,} calls, {s['cached']:,} from cache, "
                     f"{s['input_tokens']:,} tokens, ${s['dollars']:.4f}, {s['seconds']:g}s")
        return "\n".join(lines)

    def methods(self) -> str:
        """A paragraph for a data appendix. Edit it; it states only what this run did."""
        s, sc = self.settings, self.scores
        fields_ = ", ".join(lc if lc == rc else f"{lc} ({rc} in the second source)" for lc, rc in s["on"])
        judged, exact = int((sc["source"] == "jev").sum()), int((sc["source"] == "exact").sum())
        rules = {"one-to-one": "We then chose the set of links with the highest total probability such that no "
                               "record is linked twice",
                 "many-to-one": "We then kept, for each record in the first source, its most probable match",
                 "one-to-many": "We then kept, for each record in the second source, its most probable match",
                 "many-to-many": "We then kept every pair"}
        text = (
            f"We linked {s['n_left']:,} records to {s['n_right']:,} records using jlink {s['jlink']}. "
            f"Candidate pairs were generated by blocking ({'; '.join(s['blockers'])}), which produced "
            f"{len(sc):,} pairs. "
            + (f"{exact:,} pairs whose compared fields ({fields_}) were identical after normalizing case, accents "
               f"and punctuation were accepted directly. " if exact else "")
            + f"Each of the remaining {judged:,} pairs was shown to the {s['model'] or 'Jev'} decision model "
            f"(TypeSafe), which sees the compared fields ({fields_}) of both records and returns a probability "
            f"that the following statement is true: \"{s['question']}\" "
            f"{rules[s['how']]}, among pairs with probability of at least {s['threshold']:g}"
            + (f", and dropped links whose probability exceeded that of the best competing pair by less than "
               f"{s['min_margin']:g}" if s["min_margin"] is not None else "")
            + f". This yielded {len(self.links):,} links, covering "
            f"{self.links['left_id'].nunique() / max(s['n_left'], 1):.1%} of records in the first source. "
            "The model's probabilities are stored with the replication files, so these links can be reproduced "
            "exactly without calling the model again; repeated calls to the model return nearly but not exactly "
            "the same probability."
        )
        unjudged = int((sc["source"] == "unjudged").sum())
        if unjudged:
            text += f" {unjudged:,} low-similarity candidate pairs were not judged."
        return text

    def save(self, directory: str | Path) -> Path:
        """Write links.csv, scores.csv and settings.json. Everything needed to reproduce and relink."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.links.to_csv(d / "links.csv", index=False)
        self.scores.to_csv(d / "scores.csv", index=False)
        kinds = {c: "int" if pd.api.types.is_integer_dtype(self.scores[c]) else "str" for c in ("left_id", "right_id")}
        (d / "settings.json").write_text(json.dumps(self.settings | {"id_kinds": kinds}, indent=2))
        return d

    def _frames(self, left, right):
        left = self._left if left is None else left
        right = self._right if right is None else right
        if left is None or right is None:
            raise ValueError("this result was loaded from disk, so pass the original `left` and `right` tables")
        return left, right


def load(directory: str | Path) -> Result:
    """Read a result written by `Result.save`. IDs come back as they were: integers or strings."""
    d = Path(directory)
    settings = json.loads((d / "settings.json").read_text())
    kinds = settings.pop("id_kinds", {})
    dtype = {c: "int64" if kinds.get(c) == "int" else str for c in ("left_id", "right_id")}
    scores = pd.read_csv(d / "scores.csv", dtype=dtype, keep_default_na=True)
    links = pd.read_csv(d / "links.csv", dtype=dtype, keep_default_na=True)
    return Result(links, scores, settings)


def _check_how(how: str, threshold: float) -> None:
    if how not in HOWS:
        raise ValueError(f"`how` must be one of {', '.join(HOWS)}; got {how!r}")
    if not 0 <= threshold <= 1:
        raise ValueError(f"`threshold` is a probability between 0 and 1; got {threshold!r}")


def _with_id(frame: pd.DataFrame, id_column: str | None, name: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, name, frame.index if id_column is None else frame[id_column].to_numpy())
    return out.reset_index(drop=True)

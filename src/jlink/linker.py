"""The front door: describe the match, link two tables, inspect and document the result."""

from __future__ import annotations

import json
import math
import platform
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from . import __version__, audit, block
from .cluster import LINKAGES, UNPROPOSED, cluster
from jevkit_runtime import Budget, Meter, Settings
from jevkit_runtime import resolve as resolve_backend

from .core import PROVIDERS
from .fields import ids, parse_on
from .judge import EXACT_POLICY, judge, pair_tokens, question, validate_budget, validate_question
from .provenance import blocker_config, frame_fingerprint, input_fingerprints
from .resolve import resolve

_KEEP = object()  # relink: "leave this setting as it is", distinct from None, which turns the margin off
HOWS = ("one-to-one", "many-to-one", "one-to-many", "many-to-many")
PAIRS_PER_SECOND = 200  # measured on OpenRouter with short records in September 2026; only for estimates


class Linker:
    """A match rule and how to find candidates for it.

        linker = jlink.Linker(entity="firm", on=["name", "state"],
                              definition="A parent company and its subsidiary are different firms.")
        result = linker.link(compustat, patents, left_id="gvkey", right_id="assignee_id")

    ``style="identity"`` (the default) asks whether two records are the same ``entity`` and appends
    the definition. ``style="rule"`` asks whether the pair satisfies the definition, which may state
    any relation ("the article reports this incident"); ``entity`` is then optional and not part of
    the question. An ``on`` item may be one-sided, ``("text", None)`` or ``(None, "precinct")``: it
    is shown to the judge on that side only and never used to pair columns.
    """

    def __init__(self, entity: str | None = None, on=None, definition: str = "",
                 blockers: list | None = None, *, style: str = "identity", api: str | None = None,
                 model: str | None = None, concurrency: int = 32, cache=True, exact_shortcut: bool = False):
        self.entity, self.definition, self.on = (entity or "").strip() or None, (definition or "").strip(), on
        validate_question(self.entity, self.definition, style)
        self.style = style
        self.fields = parse_on(on, unpaired=True)
        if not isinstance(exact_shortcut, bool):
            raise ValueError("`exact_shortcut` must be a boolean; equal names alone do not establish identity")
        if exact_shortcut and any(lc is None or rc is None for _, lc, rc in self.fields):
            raise ValueError("`exact_shortcut` accepts pairs whose fields are all equal, so every `on` field "
                             "must exist on both sides; remove the one-sided fields or the shortcut")
        self.blockers = blockers
        self.api, self.model, self.concurrency = api, model, concurrency
        self.cache, self.exact_shortcut = cache, exact_shortcut

    def candidates(self, left: pd.DataFrame, right: pd.DataFrame, *, left_id=None, right_id=None,
                   max_pairs: int | None = 5_000_000) -> pd.DataFrame:
        return block.candidates(left, right, on=self.on, blockers=self.blockers, left_id=left_id,
                                right_id=right_id, max_pairs=max_pairs)

    def estimate(self, left: pd.DataFrame, right: pd.DataFrame | None = None, *, left_id=None,
                 right_id=None, tokens_per_pair: float | None = None) -> dict:
        """Run blocking and return a cost/time scenario with its assumptions. No API calls.

        With one table the estimate is for `dedupe`, and `left_id` names its ID column. Tokens per pair
        are the runtime's estimate over a sample of these candidates' own records, at the provider's list
        price; `tokens_per_pair` overrides the token figure. Throughput remains a short-record scenario.
        """
        if tokens_per_pair is not None and (isinstance(tokens_per_pair, bool)
                                            or not isinstance(tokens_per_pair, (int, float))
                                            or not math.isfinite(tokens_per_pair) or tokens_per_pair <= 0):
            raise ValueError("`tokens_per_pair` must be a finite positive number")
        if right is None:
            cands = block.self_candidates(left, on=self.on, blockers=self.blockers, id=left_id)
            sizes, other, other_id = {"records": len(left)}, left, left_id
        else:
            cands = self.candidates(left, right, left_id=left_id, right_id=right_id)
            sizes, other, other_id = {"left": len(left), "right": len(right)}, right, right_id
        backend = resolve_backend(PROVIDERS, self.api, model=self.model, require_key=False)
        measured = pair_tokens(cands, left, other, on=self.on, entity=self.entity, definition=self.definition,
                               style=self.style, left_id=left_id, right_id=other_id, model=backend.model)
        tokens = tokens_per_pair if tokens_per_pair is not None else (measured or 0.0)
        price, pairs = backend.price_per_mtok, len(cands)
        return sizes | {"pairs": pairs, "dollars": round(pairs * tokens * price / 1e6, 4),
                        "seconds": round(pairs / PAIRS_PER_SECOND, 1),
                        "assumptions": {"tokens_per_pair": round(tokens, 1), "price_per_million_tokens": price,
                                        "pairs_per_second": PAIRS_PER_SECOND,
                                        "token_basis": "sampled_records" if tokens_per_pair is None
                                        else "caller_supplied",
                                        "throughput_basis": "short_records"}}

    def link(self, left: pd.DataFrame, right: pd.DataFrame, *, left_id: str | None = None,
             right_id: str | None = None, how: str = "one-to-one", threshold: float = 0.5,
             min_margin: float | None = None, budget: float | None = 5.0, max_pairs: int | None = 5_000_000,
             progress: bool = True, transport=None) -> "Result":
        _check_how(how, threshold)
        validate_budget(budget)
        ids(left, left_id, "left"), ids(right, right_id, "right")
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        cands = self.candidates(left, right, left_id=left_id, right_id=right_id, max_pairs=max_pairs)
        # Reached only when candidates() accepted the same default, so a paired field exists.
        passes = self.blockers if self.blockers is not None else block.default_passes(self.fields)
        configs = [blocker_config(b) for b in passes]
        inputs = input_fingerprints(left, right, fields=self.fields, left_id=left_id, right_id=right_id)
        scores, meter = judge(cands, left, right, on=self.on, entity=self.entity, definition=self.definition,
                              style=self.style, left_id=left_id, right_id=right_id, api=self.api, model=self.model,
                              concurrency=self.concurrency, budget=budget, cache=self.cache,
                              exact_shortcut=self.exact_shortcut, progress=progress, transport=transport)
        links = resolve(scores, how=how, threshold=threshold, min_margin=min_margin)
        settings = self._settings(
            {"left_id": left_id, "right_id": right_id, "how": how, "threshold": threshold,
             "min_margin": min_margin, "n_left": len(left), "n_right": len(right)},
            passes=passes, configs=configs, cands=cands, scores=scores, meter=meter, t0=t0, started_at=started_at,
            inputs=inputs, budget=budget, max_pairs=max_pairs)
        return Result(links, scores, settings, meter, _left=left, _right=right)

    def dedupe(self, frame: pd.DataFrame, *, id: str | None = None, threshold: float = 0.5,
               linkage: str = "average", unproposed: str = "nonmatch", budget: float | None = 5.0,
               max_pairs: int | None = 5_000_000, progress: bool = True, transport=None) -> "DedupeResult":
        """Link one table to itself and group its records into clusters.

        Each unordered pair of records is a candidate at most once and is judged once, with the
        record from the earlier row as record A. `linkage="average"` (the default) merges two
        clusters while the mean probability between them is at least `threshold`, counting pairs
        that blocking never proposed as non-matches unless `unproposed="ignore"`;
        `linkage="components"` joins everything reachable through pairs at or above the threshold,
        so one wrong pair can chain two groups together. See `jlink.cluster`.
        """
        _check_clustering(linkage, threshold, unproposed)
        validate_budget(budget)
        record_ids = ids(frame, id, "deduplicated")
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        cands = block.self_candidates(frame, on=self.on, blockers=self.blockers, id=id, max_pairs=max_pairs)
        passes = self.blockers if self.blockers is not None else [
            block.ngrams(*[lc for _, lc, _ in self.fields], k=11)]
        configs = [blocker_config(b) for b in passes]
        columns = [lc for _, lc, _ in self.fields]
        inputs = {"records": {
            "compared": frame_fingerprint(frame, id_column=id, columns=columns, side="deduplicated"),
            "full": frame_fingerprint(frame, id_column=id, columns=list(frame.columns), side="deduplicated")}}
        scores, meter = judge(cands, frame, frame, on=self.on, entity=self.entity, definition=self.definition,
                              style=self.style, left_id=id, right_id=id, api=self.api, model=self.model,
                              concurrency=self.concurrency, budget=budget, cache=self.cache,
                              exact_shortcut=self.exact_shortcut, progress=progress, transport=transport)
        clusters = cluster(scores, ids=record_ids, threshold=threshold, linkage=linkage,
                           unproposed=unproposed)
        settings = self._settings(
            {"task": "dedupe", "id": id, "left_id": id, "right_id": id, "linkage": linkage,
             "unproposed": unproposed, "threshold": threshold, "pair_order": "earlier_row_is_record_a_v1",
             "n_records": len(frame), "n_left": len(frame), "n_right": len(frame)},
            passes=passes, configs=configs, cands=cands, scores=scores, meter=meter, t0=t0, started_at=started_at,
            inputs=inputs, budget=budget, max_pairs=max_pairs)
        return DedupeResult(clusters, scores, settings, meter, _frame=frame)

    def _settings(self, specific: dict, *, passes, configs, cands, scores, meter, t0, started_at, inputs, budget,
                  max_pairs) -> dict:
        """A saved run's settings. `specific` holds IDs, sizes and how pairs became links or clusters."""
        # Quote the question judge() sent. Rebuilding it here is only a fallback for a replaced judge.
        asked = scores.attrs.get("question") or question(
            self.entity or "", self.definition, style=self.style).text
        settings = {
            "jlink": __version__, "date": date.today().isoformat(), "entity": self.entity,
            "definition": self.definition, "question": asked,
            "on": [[lc, rc] for _, lc, rc in self.fields], **specific,
            "blockers": [b.name for b in passes], "blocker_configs": configs,
            "budget": _limit(budget), "model": meter.model,
            "calls": meter.calls, "cached": meter.cached, "input_tokens": meter.input_tokens,
            "dollars": round(meter.cost, 6), "seconds": round(time.perf_counter() - t0, 1),
            "provenance_version": 1, "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
            "requested_api": self.api, "requested_model": self.model,
            "provider": meter.provider, "request_model": meter.requested_model,
            "resolved_models": meter.resolved_models, "unknown_model_answers": meter.unknown_model_answers,
            "answer_provenance": meter.answer_provenance,
            "exact_shortcut": self.exact_shortcut, "exact_policy": EXACT_POLICY,
            "normalization": "jlink.fields.normalize_v1", "concurrency": int(self.concurrency),
            "cache_enabled": bool(self.cache), "max_pairs": None if max_pairs is None else int(max_pairs),
            "budget_policy": "stop_new_requests_at_observed_cost_v1",
            "cost_sources": meter.cost_sources, "estimated_price_per_million_tokens": Settings.from_env().list_price,
            "retries": meter.retries, "cost": meter.cost,
            "runtime": {"python": platform.python_version(), **{p: version(p) for p in (
                "numpy", "pandas", "scipy", "scikit-learn", "httpx")}},
        }
        # Missing style already means identity in saved runs. Keep ordinary link settings
        # compatible with main; rule runs and the new dedupe format name their style explicitly.
        if self.style != "identity" or specific.get("task") == "dedupe":
            settings["style"] = self.style
        if "blocking" in cands.attrs:
            settings["blocking"] = deepcopy(cands.attrs["blocking"])
        if "run" in scores.attrs:
            settings["run"] = deepcopy(scores.attrs["run"])  # the runtime's record: who answered, cost, budget
        return settings


def link(left: pd.DataFrame, right: pd.DataFrame, *, entity: str | None = None, on, definition: str = "",
         blockers=None, style: str = "identity", exact_shortcut: bool = False, **kwargs) -> "Result":
    """One call for the common case. Keyword arguments are those of `Linker.link`."""
    return Linker(entity, on, definition, blockers, style=style,
                  exact_shortcut=exact_shortcut).link(left, right, **kwargs)


def dedupe(frame: pd.DataFrame, *, entity: str | None = None, on, definition: str = "", blockers=None,
           style: str = "identity", exact_shortcut: bool = False, **kwargs) -> "DedupeResult":
    """Find the records in one table that match each other. Keyword arguments are those of `Linker.dedupe`."""
    return Linker(entity, on, definition, blockers, style=style,
                  exact_shortcut=exact_shortcut).dedupe(frame, **kwargs)


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

    def resume(self, left: pd.DataFrame | None = None, right: pd.DataFrame | None = None, *,
               budget: float | None = 5.0, api: str | None = None, model: str | None = None,
               concurrency: int = 32, cache=True, progress: bool = True, transport=None) -> "Result":
        """Judge the pairs this run left unjudged (budget) or failed (errors), then choose links again.

        Only those pairs are sent, with the saved question, fields and model; the candidate pairs are
        the saved ones, so blocking does not run again and a changed default cannot alter them. Pairs
        already judged keep their saved probabilities. The answer cache already makes a fresh
        `link` free for pairs it answered; `resume` adds the fixed candidate set, a run that no
        longer needs that cache, and one folder of provenance. A loaded run needs `left` and
        `right`, which must match the saved input fingerprints.
        """
        validate_budget(budget)
        left, right = self._frames(left, right)
        s = deepcopy(self.settings)
        on = [(lc, rc) for lc, rc in s["on"]]
        if "inputs" in s:
            now = input_fingerprints(left, right, fields=parse_on(on, unpaired=True),
                                     left_id=s["left_id"], right_id=s["right_id"])
            for side in ("left", "right"):
                if now[side]["compared"]["sha256"] != s["inputs"][side]["compared"]["sha256"]:
                    raise ValueError(f"the {side} table differs from the one this run was made from (its IDs "
                                     "or compared fields changed); pass the original table")
        style = s.get("style", "identity")
        if question(s["entity"] or "", s["definition"], style=style).text != s["question"]:
            raise ValueError("this jlink version would word the question differently from the saved run, "
                             "so resuming would mix two questions; run `link` again instead")
        todo = self.scores["source"].isin(["unjudged", "error"]).to_numpy()
        if not todo.any():
            return self.relink()
        t0, started_at = time.perf_counter(), datetime.now(timezone.utc).isoformat()
        api = api or s.get("provider") or s.get("requested_api")
        model = model or s.get("request_model") or s.get("requested_model")
        fresh, meter = judge(self.scores.loc[todo, ["left_id", "right_id", "block", "sim"]], left, right, on=on,
                             entity=s["entity"], definition=s["definition"], style=style,
                             left_id=s["left_id"], right_id=s["right_id"], api=api, model=model,
                             concurrency=concurrency, budget=budget, cache=cache, progress=progress,
                             transport=transport)
        scores = self.scores.copy()
        for column in ("p", "answered_at", "source", "error", "model", "provider", "score_origin"):
            numeric = column in ("p", "answered_at")
            values = scores[column] if column in scores else pd.Series(pd.NA, index=scores.index)
            values = pd.to_numeric(values) if numeric else values.astype(object)
            values[todo] = fresh[column].to_numpy()
            scores[column] = values
        total = Meter(provider=meter.provider or self.meter.provider,
                      requested_model=meter.requested_model or self.meter.requested_model,
                      calls=self.meter.calls + meter.calls, cached=self.meter.cached + meter.cached,
                      retries=self.meter.retries + meter.retries, input_tokens=self.meter.input_tokens
                      + meter.input_tokens, cost=self.meter.cost + meter.cost,
                      cost_sources={k: self.meter.cost_sources.get(k, 0) + meter.cost_sources.get(k, 0)
                                    for k in {*self.meter.cost_sources, *meter.cost_sources}},
                      answer_provenance=deepcopy(self.meter.answer_provenance))
        for item in meter.answer_provenance:
            for _ in range(item["count"]):
                total.note_answer(item)
        s.update({"calls": total.calls, "cached": total.cached, "retries": total.retries,
                  "input_tokens": total.input_tokens, "cost": total.cost, "dollars": round(total.cost, 6),
                  "model": total.model, "resolved_models": total.resolved_models,
                  "unknown_model_answers": total.unknown_model_answers,
                  "answer_provenance": total.answer_provenance, "cost_sources": total.cost_sources,
                  "seconds": round(s.get("seconds", 0) + time.perf_counter() - t0, 1)})
        s.setdefault("resumes", []).append({
            "jlink": __version__, "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
            "pairs": int(todo.sum()), "budget": _limit(budget), "calls": meter.calls,
            "cached": meter.cached, "dollars": round(meter.cost, 6),
            "left_unjudged": int((fresh["source"] == "unjudged").sum()),
            "failed": int((fresh["source"] == "error").sum())})
        links = resolve(scores, how=s["how"], threshold=s["threshold"], min_margin=s["min_margin"])
        return Result(links, scores, s, total, left, right)

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
        """Sample judged pairs, with actual link membership for evaluate(..., mode="selected")."""
        try:
            left, right = self._frames(left, right)
        except ValueError:
            left = right = None
        on = [(lc, rc) for lc, rc in self.settings["on"]]
        return audit.audit_sample(self.scores, n=n, seed=seed, left=left, right=right, on=on if left is not None else None,
                                  left_id=self.settings["left_id"], right_id=self.settings["right_id"],
                                  links=self.links, **kwargs)

    def report(self) -> str:
        s, sc = self.settings, self.scores
        by_source = sc["source"].value_counts()
        by_block = sc["block"].value_counts()
        lines = [
            f"jlink {s['jlink']}, {s['date']}",
            f"Rule: {s['question']}",
            *(["Question style: rule (the definition states the relation; no entity is named)"]
              if s.get("style") == "rule" else []),
            f"Compared on: {_field_list(s['on'], ' = ', '{} (left only)', '{} (right only)')}",
            f"Records: {s['n_left']:,} left, {s['n_right']:,} right",
            f"Candidate pairs: {len(sc):,} ({'; '.join(f'{k}: {v:,}' for k, v in by_block.items())})",
            _unproposed(s["n_left"], sc["left_id"], s["n_right"], sc["right_id"]),
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
        lines.append(f"Model: {_model_description(s)}; {s['calls']:,} calls, {s['cached']:,} from cache, "
                     f"{s['input_tokens']:,} tokens, ${s['dollars']:.4f}, {s['seconds']:g}s")
        return "\n".join(lines)

    def methods(self) -> str:
        """A paragraph for a data appendix. Edit it; it states only what this run did."""
        s, sc = self.settings, self.scores
        fields_ = _field_list(s["on"], None, "{} (first source only)", "{} (second source only)")
        one_sided = any(lc is None or rc is None for lc, rc in s["on"])
        shown = (f"the listed fields of each record ({fields_})" if one_sided
                 else f"the compared fields ({fields_}) of both records")
        if s.get("style") == "rule":
            # The proposition is the user's relation, so the paragraph must not claim identity.
            relation = ("A link here is a relation between two records that a written match rule defines, "
                        "not a claim that both records describe the same entity. ")
            asked = "returns a probability that the pair satisfies the rule, put to the model as"
        else:
            relation, asked = "", "returns a probability that the following statement is true"
        judged, exact = int((sc["source"] == "jev").sum()), int((sc["source"] == "exact").sum())
        rules = {"one-to-one": "We then chose the set of links with the highest total probability such that no "
                               "record is linked twice",
                 "many-to-one": "We then kept, for each record in the first source, its most probable match",
                 "one-to-many": "We then kept, for each record in the second source, its most probable match",
                 "many-to-many": "We then kept every pair"}
        exact_text = ""
        if exact:
            if s.get("exact_policy") == EXACT_POLICY:
                exact_text = (f"{exact:,} pairs whose compared fields ({fields_}) were all nonempty and "
                              "individually identical after normalizing case, accents and punctuation "
                              "were accepted directly. ")
            else:
                exact_text = (f"{exact:,} pairs were accepted directly by the saved run's exact shortcut; "
                              "its fieldwise and missing-value policy was not recorded. ")
        text = (
            f"We linked {s['n_left']:,} records to {s['n_right']:,} records using jlink {s['jlink']}. "
            + relation
            + f"Candidate pairs were generated by blocking ({'; '.join(s['blockers'])}), which produced "
            f"{len(sc):,} pairs. "
            + exact_text
            + f"The {judged:,} model-scored pairs used Jev ({_model_description(s)}) "
            f"(TypeSafe), which sees {shown} and {asked}: \"{s['question']}\" "
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
            text += f" {unjudged:,} candidate pairs were not judged; new calls were prioritized by similarity."
        return text

    def save(self, directory: str | Path) -> Path:
        """Write links.csv, scores.csv and settings.json. Everything needed to reproduce and relink."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.links.to_csv(d / "links.csv", index=False)
        self.scores.to_csv(d / "scores.csv", index=False)
        kinds = {c: _id_kind(self.scores[c]) for c in ("left_id", "right_id")}
        (d / "settings.json").write_text(json.dumps(
            self.settings | {"result_format_version": 2, "id_kinds": kinds}, indent=2), encoding="utf-8")
        return d

    def _frames(self, left, right):
        left = self._left if left is None else left
        right = self._right if right is None else right
        if left is None or right is None:
            raise ValueError("this result was loaded from disk, so pass the original `left` and `right` tables")
        return left, right


@dataclass
class DedupeResult:
    """Clusters of records within one table, with the pair scores they came from.

    `clusters` has one row per record, in table order: `id`, `cluster_id`, `cluster_size`.
    `scores` has one row per candidate pair; `left_id` is the record from the earlier row, which
    the judge saw as record A.
    """

    clusters: pd.DataFrame
    scores: pd.DataFrame
    settings: dict
    meter: Meter = field(default_factory=Meter)
    _frame: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def candidates(self) -> pd.DataFrame:
        return self.scores[["left_id", "right_id", "block", "sim"]]

    @property
    def links(self) -> pd.DataFrame:
        """The judged pairs whose two records share a cluster, most probable first.

        Records can share a cluster without a judged pair between them; those pairs are not here.
        """
        member = self.clusters.set_index("id")["cluster_id"]
        together = (member.reindex(self.scores["left_id"]).to_numpy()
                    == member.reindex(self.scores["right_id"]).to_numpy())
        kept = self.scores.loc[together & self.scores["p"].notna().to_numpy()]
        return kept.sort_values("p", ascending=False, kind="stable").reset_index(drop=True)

    def recluster(self, *, threshold: float | None = None, linkage: str | None = None,
                  unproposed: str | None = None) -> "DedupeResult":
        """Group the records again under another threshold or rule. No new API calls."""
        s = dict(self.settings)
        s["threshold"] = s["threshold"] if threshold is None else threshold
        s["linkage"], s["unproposed"] = linkage or s["linkage"], unproposed or s["unproposed"]
        _check_clustering(s["linkage"], s["threshold"], s["unproposed"])
        clusters = cluster(self.scores, ids=self.clusters["id"], threshold=s["threshold"],
                           linkage=s["linkage"], unproposed=s["unproposed"])
        return DedupeResult(clusters, self.scores, s, self.meter, self._frame)

    def split_pairs(self) -> pd.DataFrame:
        """Judged pairs at or above the threshold whose records ended in different clusters.

        Components never leave any. Under average linkage they are where the rule overrode a
        single high probability: a wrong pair it resisted, or a true pair of a group that
        blocking covered too thinly. They are the first pairs to read when checking the clusters.
        """
        member = self.clusters.set_index("id")["cluster_id"]
        apart = (member.reindex(self.scores["left_id"]).to_numpy()
                 != member.reindex(self.scores["right_id"]).to_numpy())
        strong = self.scores["p"].ge(self.settings["threshold"]).to_numpy()
        split = self.scores.loc[apart & strong]
        return split.sort_values("p", ascending=False, kind="stable").reset_index(drop=True)

    def labeled(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """The table with `cluster_id` and `cluster_size` appended to every record."""
        frame = self._table(frame)
        for column in ("cluster_id", "cluster_size"):
            if column in frame.columns:
                raise ValueError(f"the table already has a {column!r} column; rename it before labeling")
        out = frame.copy()
        out["cluster_id"] = self.clusters["cluster_id"].to_numpy()
        out["cluster_size"] = self.clusters["cluster_size"].to_numpy()
        return out

    def audit_sample(self, n: int = 200, *, seed: int = 0, frame: pd.DataFrame | None = None,
                     **kwargs) -> pd.DataFrame:
        """Sample judged pairs for labeling. `selected` means the two records share a cluster.

        `evaluate(..., mode="selected")` then measures the clusters pair by pair, among judged
        pairs only: records joined through other records, with no judged pair of their own, are
        outside the sample.
        """
        try:
            frame = self._table(frame)
        except ValueError:
            if frame is not None:
                raise
        on = [(lc, rc) for lc, rc in self.settings["on"]]
        column = self.settings["id"]
        return audit.audit_sample(self.scores, n=n, seed=seed, left=frame, right=frame,
                                  on=on if frame is not None else None, left_id=column, right_id=column,
                                  links=self.links, **kwargs)

    def report(self) -> str:
        s, sc, sizes = self.settings, self.scores, self._sizes()
        by_source = sc["source"].value_counts()
        by_block = sc["block"].value_counts()
        rule = ("connected components of pairs with p" if s["linkage"] == "components"
                else "mean p of the judged pairs between two clusters" if s["unproposed"] == "ignore"
                else "mean p between two clusters, unproposed pairs as non-matches,")
        lines = [
            f"jlink {s['jlink']}, {s['date']} (dedupe)",
            f"Rule: {s['question']}",
            *(["Question style: rule (the definition states the relation; no entity is named)"]
              if s.get("style") == "rule" else []),
            f"Compared on: {_field_list(s['on'], ' = ', '{}', '{}')}",
            f"Records: {s['n_records']:,} in one table",
            f"Candidate pairs: {len(sc):,} unordered pairs, earlier row as record A "
            f"({'; '.join(f'{k}: {v:,}' for k, v in by_block.items())})",
            "Judged: " + ", ".join(f"{by_source.get(k, 0):,} {label}" for k, label in (
                ("jev", "by Jev"), ("exact", "identical after normalization"), ("error", "failed"),
                ("unjudged", "left unjudged by the budget")) if by_source.get(k, 0)),
            f"Clusters: {int((sizes > 1).sum()):,} of two or more records, holding "
            f"{int(sizes[sizes > 1].sum()):,} records; {int((sizes == 1).sum()):,} records alone "
            f"({s['linkage']} linkage: {rule} >= {s['threshold']:g})",
        ]
        if s["linkage"] == "average":
            lines.append("Pairs at or above the threshold left in different clusters: "
                         f"{len(self.split_pairs()):,}")
        if (sizes > 1).any():
            lines.append(f"Cluster sizes: largest {int(sizes.max()):,}; {int((sizes == 2).sum()):,} clusters "
                         f"of exactly two records; {int((sizes > 2).sum()):,} larger")
        lines.append(f"Model: {_model_description(s)}; {s['calls']:,} calls, {s['cached']:,} from cache, "
                     f"{s['input_tokens']:,} tokens, ${s['dollars']:.4f}, {s['seconds']:g}s")
        return "\n".join(lines)

    def methods(self) -> str:
        """A paragraph for a data appendix. Edit it; it states only what this run did."""
        s, sc, sizes = self.settings, self.scores, self._sizes()
        fields_ = _field_list(s["on"], None, "{}", "{}")
        judged, exact = int((sc["source"] == "jev").sum()), int((sc["source"] == "exact").sum())
        if s.get("style") == "rule":
            aim = ("that a written match rule relates to each other; the rule defines a relation between two "
                   "records and need not mean that they describe the same entity")
            asked = "returns a probability that the pair satisfies the rule, put to the model as"
        else:
            aim = f"that refer to the same {s['entity']}"
            asked = "returns a probability that the following statement is true"
        if s["linkage"] == "average":
            counted = ("The mean was taken over every pair of records between the two groups: a pair that "
                       "blocking never proposed counted as zero, and a candidate pair that was never judged "
                       "was left out" if s["unproposed"] == "nonmatch" else
                       "The mean was taken over the judged pairs between the two groups; pairs that were never "
                       "proposed or never judged counted neither for nor against a merge")
            grouping = ("We then grouped records by average-linkage agglomeration: starting from single "
                        "records, the two groups with the highest mean probability between them were merged, "
                        f"repeatedly, while that mean was at least {s['threshold']:g}. {counted}")
        else:
            grouping = ("We then grouped records into the connected components of the graph whose edges are "
                        f"the pairs with probability of at least {s['threshold']:g}, so records can share a "
                        "group through a chain of such pairs")
        exact_text = ""
        if exact:
            exact_text = (f"{exact:,} pairs whose compared fields ({fields_}) were all nonempty and individually "
                          "identical after normalizing case, accents and punctuation were accepted directly. ")
        text = (
            f"We searched {s['n_records']:,} records for groups of records {aim}, using jlink {s['jlink']}. "
            f"Candidate pairs were generated by blocking the table against itself ({'; '.join(s['blockers'])}); "
            f"no record was paired with itself and each unordered pair was kept once, which produced "
            f"{len(sc):,} pairs. "
            + exact_text
            + f"The {judged:,} model-scored pairs used Jev ({_model_description(s)}) (TypeSafe), which sees the "
            f"compared fields ({fields_}) of both records, the record from the earlier row presented first, "
            f"and {asked}: \"{s['question']}\" Each pair was judged once, in that order. "
            f"{grouping}. This yielded {int((sizes > 1).sum()):,} groups of two or more records, holding "
            f"{int(sizes[sizes > 1].sum()):,} of the {s['n_records']:,} records. "
            "The model's probabilities are stored with the replication files, so these groups can be reproduced "
            "exactly without calling the model again; repeated calls to the model return nearly but not exactly "
            "the same probability."
        )
        unjudged = int((sc["source"] == "unjudged").sum())
        if unjudged:
            text += f" {unjudged:,} candidate pairs were not judged; new calls were prioritized by similarity."
        return text

    def save(self, directory: str | Path) -> Path:
        """Write clusters.csv, scores.csv and settings.json. Everything needed to reproduce and recluster."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.clusters.to_csv(d / "clusters.csv", index=False)
        self.scores.to_csv(d / "scores.csv", index=False)
        kinds = {c: _id_kind(frame[c])
                 for frame, columns in ((self.scores, ("left_id", "right_id")), (self.clusters, ("id",)))
                 for c in columns}
        (d / "settings.json").write_text(json.dumps(
            self.settings | {"result_format_version": 2, "id_kinds": kinds}, indent=2), encoding="utf-8")
        return d

    def _sizes(self):
        return self.clusters.drop_duplicates("cluster_id")["cluster_size"].to_numpy()

    def _table(self, frame):
        frame = self._frame if frame is None else frame
        if frame is None:
            raise ValueError("this result was loaded from disk, so pass the original table")
        records = pd.Index(ids(frame, self.settings["id"], "deduplicated"))
        if not records.equals(pd.Index(self.clusters["id"])):
            raise ValueError("the table's IDs differ from the saved clusters, in value or in order; "
                             "pass the table that was deduplicated")
        return frame


def load(directory: str | Path) -> "Result | DedupeResult":
    """Read a result written by `save`. IDs come back as they were: integers, floats or strings.

    A directory written by `DedupeResult.save` comes back as a `DedupeResult`.
    """
    d = Path(directory)
    settings = json.loads((d / "settings.json").read_text(encoding="utf-8"))
    settings.pop("result_format_version", None)
    kinds = settings.pop("id_kinds", {})
    # Runs saved before float IDs were recorded call them "str", and still load as text.
    dtype = {c: _ID_DTYPES.get(kinds.get(c), str) for c in ("left_id", "right_id")}
    # Missing numeric scores/errors are separate from literal IDs such as NA, NULL and 001.
    numeric = ("sim", "p", "margin", "answered_at")
    optional_text = ("error", "model", "provider", "score_origin")
    dtype.update({c: float for c in numeric})
    dtype.update({c: object for c in optional_text})
    na_values = {c: [""] for c in numeric + optional_text}
    scores = pd.read_csv(d / "scores.csv", dtype=dtype, keep_default_na=False, na_values=na_values,
                         float_precision="round_trip")
    dedupe_run = settings.get("task") == "dedupe"
    if dedupe_run:
        links = pd.read_csv(d / "clusters.csv", keep_default_na=False, na_values={},
                            dtype={"id": _ID_DTYPES.get(kinds.get("id"), str),
                                   "cluster_id": "int64", "cluster_size": "int64"},
                            float_precision="round_trip")
    else:
        links = pd.read_csv(d / "links.csv", dtype=dtype, keep_default_na=False, na_values=na_values,
                            float_precision="round_trip")
    if "blocking" in settings:
        scores.attrs["blocking"] = deepcopy(settings["blocking"])
    if "provenance_version" not in settings:
        settings.setdefault("model_identity_status", "legacy_unverified")
    meter = Meter(calls=settings.get("calls", 0), cached=settings.get("cached", 0),
                  retries=settings.get("retries", 0), input_tokens=settings.get("input_tokens", 0),
                  cost=settings.get("cost", settings.get("dollars", 0.0)),
                  provider=settings.get("provider", ""), requested_model=settings.get("request_model", ""),
                  answer_provenance=settings.get("answer_provenance", []),
                  cost_sources=settings.get("cost_sources", {}))
    return (DedupeResult if dedupe_run else Result)(links, scores, settings, meter)


_ID_DTYPES = {"int": "int64", "float": "float64"}


def _id_kind(ids_: pd.Series) -> str:
    """Stata often stores numeric IDs as doubles; as text they would no longer merge with the source."""
    if pd.api.types.is_integer_dtype(ids_):
        return "int"
    return "float" if pd.api.types.is_float_dtype(ids_) else "str"


def _unproposed(n_left: int, left_ids: pd.Series, n_right: int, right_ids: pd.Series) -> str:
    """Records blocking never paired with anything: no threshold or rule can link them."""
    left, right = n_left - left_ids.nunique(), n_right - right_ids.nunique()
    return (f"Records with no candidate pair: {left:,} of {n_left:,} left, {right:,} of {n_right:,} right"
            + (" (these cannot be linked; add a blocking pass to reach them)" if left or right else ""))


def _field_list(on: list, joiner: str | None, left_only: str, right_only: str) -> str:
    """Name the fields in saved `on` pairs; a null column marks a field that one side lacks."""
    names = []
    for lc, rc in on:
        if lc is None or rc is None:
            names.append((left_only if rc is None else right_only).format(lc or rc))
        elif lc == rc:
            names.append(lc)
        else:
            names.append(f"{lc}{joiner}{rc}" if joiner else f"{lc} ({rc} in the second source)")
    return ", ".join(names)


def _model_description(settings: dict) -> str:
    if settings.get("model_identity_status") == "legacy_unverified":
        return f"{settings.get('model') or 'unknown'} (legacy model identity unverified)"
    models = settings.get("resolved_models", [])
    unknown = settings.get("unknown_model_answers", 0)
    if not models and not unknown and settings.get("provenance_version"):
        return "not used (no model-scored pairs)"
    description = ", ".join(models) or "unknown"
    if unknown:
        description += f"; {unknown:,} answers with unknown model identity"
    return description


def _limit(budget) -> float | None:
    """A budget as saved settings record it: its dollar limit, or None for no limit."""
    if isinstance(budget, Budget):
        return None if budget.unlimited else budget.limit
    return None if budget is None else float(budget)


def _check_clustering(linkage: str, threshold: float, unproposed: str) -> None:
    if linkage not in LINKAGES:
        raise ValueError(f"`linkage` must be one of {', '.join(LINKAGES)}; got {linkage!r}")
    if unproposed not in UNPROPOSED:
        raise ValueError(f"`unproposed` must be one of {', '.join(UNPROPOSED)}; got {unproposed!r}")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError(f"`threshold` is a probability between 0 and 1; got {threshold!r}")


def _check_how(how: str, threshold: float) -> None:
    if how not in HOWS:
        raise ValueError(f"`how` must be one of {', '.join(HOWS)}; got {how!r}")
    if not 0 <= threshold <= 1:
        raise ValueError(f"`threshold` is a probability between 0 and 1; got {threshold!r}")


def _with_id(frame: pd.DataFrame, id_column: str | None, name: str) -> pd.DataFrame:
    out = frame.copy()
    if id_column != name:
        out.insert(0, name, frame.index if id_column is None else frame[id_column].to_numpy())
    return out.reset_index(drop=True)

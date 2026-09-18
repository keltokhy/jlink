"""Local review snapshots, append-only decisions, and offline constrained resolution."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .fields import ids
from .resolve import _margin, _pairs, _probabilities, resolve

if TYPE_CHECKING:
    from .linker import Result

FORMAT = "jlink-review"
VERSION = 1
_KEEP = object()
_MODES = ("one-to-one", "many-to-one", "one-to-many", "many-to-many")
_METADATA = {"review_decision", "review_event_id", "selection_basis", "original_selected", "review_run_id"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _value(value):
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if abs(value) > 2**53 - 1:
            return repr(value)
        if value.is_integer():
            return int(value)
    # JavaScript cannot losslessly round-trip large integers as JSON numbers.
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _id(value) -> dict:
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, (bool,)) or value is None or value is pd.NA:
        raise ValueError("review IDs must be strings, integers or finite floats, with no missing IDs")
    if isinstance(value, numbers.Integral):
        return {"type": "int", "value": str(int(value))}
    if isinstance(value, numbers.Real) and math.isfinite(float(value)):
        return {"type": "float", "value": repr(float(value))}
    raise ValueError(f"review ID {value!r}: expected a string, integer or finite float")


def _decode_id(value):
    if not isinstance(value, dict) or set(value) != {"type", "value"} or not isinstance(value["value"], str):
        raise ValueError("review IDs need a type and a lossless text value")
    try:
        decoded = {"str": str, "int": int, "float": float}[value["type"]](value["value"])
    except (KeyError, ValueError, TypeError, OverflowError):
        raise ValueError("invalid typed review ID") from None
    if _id(decoded) != value:
        raise ValueError("review ID is not in canonical form")
    return decoded


def _key(a, b) -> str:
    return _json([a, b])


def _table(frame: pd.DataFrame) -> dict:
    if frame.columns.has_duplicates or not all(isinstance(c, str) for c in frame.columns):
        raise ValueError("review table columns must have unique string names")
    return {"columns": list(frame.columns), "rows": [
        {c: _id(v) if c in ("left_id", "right_id") else _value(v) for c, v in zip(frame.columns, row)}
        for row in frame.itertuples(index=False, name=None)]}


def _frame(table: dict) -> pd.DataFrame:
    if not isinstance(table, dict) or set(table) != {"columns", "rows"}:
        raise ValueError("review tables need columns and rows")
    cols, rows = table["columns"], table["rows"]
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols) or len(cols) != len(set(cols)):
        raise ValueError("review table columns must have unique string names")
    if not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != set(cols) for r in rows):
        raise ValueError("review table rows must match the declared columns")
    result = pd.DataFrame(rows, columns=cols)
    for side in ("left_id", "right_id"):
        if side not in cols:
            raise ValueError(f"review table needs {side}")
        # Object dtype prevents a mixed integer/float ID column from rounding large integers.
        result[side] = pd.Series([_decode_id(v) for v in result[side]], dtype=object)
    for name in ("p", "sim", "margin"):
        if name in result:
            result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
    return result


def _settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("review settings must be an object")
    how, threshold, margin = (settings.get(k) for k in ("how", "threshold", "min_margin"))
    if how not in _MODES:
        raise ValueError(f"review how must be one of {', '.join(_MODES)}")
    for name, value in (("threshold", threshold), ("min_margin", margin)):
        if name == "min_margin" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"review {name} must be a finite number")
        if not (0 if name == "threshold" else -1) <= value <= 1:
            raise ValueError(f"review {name} is outside its permitted range")
    return {"how": how, "threshold": threshold, "min_margin": margin}


def _timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise ValueError("review timestamps must be ISO 8601 with a timezone") from None


def _history(events: list, pairs: set[str]) -> tuple[list, dict]:
    """Validate causal chains; clock times never silently choose a winning amendment."""
    if not isinstance(events, list):
        raise ValueError("review history must be a list")
    by_id, successors = {}, {}
    required = {"event_id", "previous_event", "left_id", "right_id", "decision", "reviewer",
                "timestamp", "note"}
    for event in events:
        if not isinstance(event, dict) or set(event) != required:
            raise ValueError("review history event has missing or unexpected fields")
        for side in ("left_id", "right_id"):
            _decode_id(event[side])
        pair = _key(event["left_id"], event["right_id"])
        if pair not in pairs:
            raise ValueError("review decision names a pair absent from the source candidates")
        if event["decision"] not in ("accept", "reject", "unsure"):
            raise ValueError("review decision must be accept, reject or unsure")
        if not isinstance(event["reviewer"], str) or not event["reviewer"].strip():
            raise ValueError("every review decision needs a reviewer identity")
        if not isinstance(event["note"], str):
            raise ValueError("review note must be text")
        _timestamp(event["timestamp"])
        eid, previous = event["event_id"], event["previous_event"]
        if not isinstance(eid, str) or not eid or (previous is not None and not isinstance(previous, str)):
            raise ValueError("review events need an event_id and a previous_event ID or null")
        if eid in by_id:
            raise ValueError(f"duplicate review event_id {eid!r}")
        by_id[eid] = event
        branch = (pair, previous)
        if branch in successors:
            raise ValueError("conflicting amendments for the same pair; import its shared history, "
                             "then record an explicit amendment")
        successors[branch] = eid
    # Resolve each pair's causal chain independently, retaining deterministic merge order.
    ordered, latest = [], {}
    roots = sorted((e for e in events if e["previous_event"] is None),
                   key=lambda e: (e["timestamp"], e["event_id"]))
    for event in roots:
        pair = _key(event["left_id"], event["right_id"])
        while event is not None:
            ordered.append(event)
            latest[pair] = event
            next_id = successors.get((pair, event["event_id"]))
            event = by_id.get(next_id)
    if len(ordered) != len(events):
        raise ValueError("review history has a missing predecessor, wrong pair predecessor, or cycle")
    return ordered, latest


def _source_id(source: dict) -> str:
    return "sha256:" + hashlib.sha256(_json(source).encode("utf-8")).hexdigest()


class Review:
    """A self-contained, versioned review artifact. IDs are typed strings in JSON.

    Use create_review(result), then decide / write_html / save / merge / apply.
    Public readers return copies; changes must go through an explicit history event.
    """

    def __init__(self, document: dict):
        self._doc = copy.deepcopy(document)
        doc = self._doc
        if not isinstance(doc, dict) or set(doc) != {
                "format", "version", "run_id", "created_at", "source", "history"}:
            raise ValueError("expected a complete jlink review artifact")
        if doc["format"] != FORMAT or type(doc["version"]) is not int or doc["version"] != VERSION:
            raise ValueError("unsupported review format or version")
        _timestamp(doc["created_at"])
        source = doc["source"]
        if not isinstance(source, dict) or set(source) != {
                "settings", "scores", "links", "records", "close_margin"}:
            raise ValueError("review source snapshot is incomplete")
        if doc["run_id"] != _source_id(source):
            raise ValueError("review source run identity does not match its snapshot; source data changed")
        _settings(source["settings"])
        fields = source["settings"].get("on", [])
        if not isinstance(fields, list) or any(
                not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(c, str) for c in pair)
                for pair in fields):
            raise ValueError("review on fields must be a list of [left column, right column] pairs")
        close = source["close_margin"]
        if isinstance(close, bool) or not isinstance(close, numbers.Real) or not 0 <= close <= 1:
            raise ValueError("close_margin must be between 0 and 1")
        scores, links = _frame(source["scores"]), _frame(source["links"])
        _pairs(scores, "review scores")
        _pairs(links, "original links")
        _probabilities(scores)
        _probabilities(scores, "sim", missing=False)
        if _METADATA.intersection(scores.columns):
            raise ValueError("source scores contain reserved review metadata columns")
        self._pairs = {_key(r["left_id"], r["right_id"]) for r in source["scores"]["rows"]}
        if any(_key(r["left_id"], r["right_id"]) not in self._pairs for r in source["links"]["rows"]):
            raise ValueError("an original link is absent from the source candidates")
        records = source["records"]
        if not isinstance(records, dict) or set(records) != {"left", "right"}:
            raise ValueError("review needs original records from both sides")
        for side in ("left", "right"):
            if not isinstance(records[side], list):
                raise ValueError(f"review {side} records must be a list")
            values = []
            for record in records[side]:
                if not isinstance(record, dict) or set(record) != {"id", "fields"}:
                    raise ValueError("review records need id and fields")
                values.append(_decode_id(record["id"]))
                if not isinstance(record["fields"], dict):
                    raise ValueError("review record fields must be an object")
            if pd.Index(values).has_duplicates:
                raise ValueError(f"review {side} records have duplicate IDs")
            known = {_json(r["id"]) for r in records[side]}
            if any(_json(r[f"{side}_id"]) not in known for r in source["scores"]["rows"]):
                raise ValueError(f"review candidate has an ID absent from the original {side} data")
        doc["history"], self._latest = _history(doc["history"], self._pairs)

    @property
    def run_id(self) -> str:
        return self._doc["run_id"]

    @property
    def history(self) -> list[dict]:
        return copy.deepcopy(self._doc["history"])

    @property
    def candidates(self) -> pd.DataFrame:
        """Copy of original scores with selected derived from actual original links membership."""
        source = self._doc["source"]
        frame = _frame(source["scores"])
        selected = {_key(r["left_id"], r["right_id"]) for r in source["links"]["rows"]}
        frame["selected"] = [_key(r["left_id"], r["right_id"]) in selected for r in source["scores"]["rows"]]
        return frame

    def to_dict(self) -> dict:
        return copy.deepcopy(self._doc)

    def decide(self, left_id, right_id, decision: str, *, reviewer: str, note: str = "",
               timestamp: str | None = None) -> "Review":
        """Append an amendment. Unsure releases a previous accept/reject constraint."""
        a, b = _id(left_id), _id(right_id)
        previous = self._latest.get(_key(a, b))
        event = dict(event_id=str(uuid.uuid4()), previous_event=previous["event_id"] if previous else None,
                     left_id=a, right_id=b, decision=decision, reviewer=reviewer,
                     timestamp=_now() if timestamp is None else timestamp, note=note)
        history, latest = _history([*self._doc["history"], event], self._pairs)
        self._doc["history"], self._latest = history, latest
        return self

    def merge(self, other: "Review") -> "Review":
        """Import a same-run history without discarding local work or resolving branches silently."""
        if not isinstance(other, Review):
            raise ValueError("import requires a Review artifact; load it with read_review(path)")
        if self.run_id != other.run_id or self._doc["source"] != other._doc["source"]:
            raise ValueError("cannot import decisions from a different source run")
        events = {event["event_id"]: event for event in self.history}
        for event in other.history:
            if event["event_id"] in events and events[event["event_id"]] != event:
                raise ValueError("import changes an existing review event")
            events[event["event_id"]] = event
        history, latest = _history(list(events.values()), self._pairs)
        self._doc["history"], self._latest = history, latest
        return self

    def save(self, path: str | Path) -> Path:
        """Save the durable JSON artifact, including the original snapshot and every amendment."""
        path = Path(path)
        path.write_text(_json(self._doc) + "\n", encoding="utf-8")
        return path

    def write_html(self, path: str | Path) -> Path:
        """Write a standalone local page. No remote resources, service, or API key required."""
        assets = Path(__file__).with_name("assets")
        data = _json(self._doc).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        template = (assets / "review.html").read_text(encoding="utf-8")
        page = template.replace("/* REVIEW_CSS */", (assets / "review.css").read_text(encoding="utf-8"))
        page = page.replace("/* REVIEW_JS */", (assets / "review.js").read_text(encoding="utf-8"))
        page = page.replace("REVIEW_DATA", data)
        path = Path(path)
        path.write_text(page, encoding="utf-8")
        return path

    def apply(self, *, how: str | None = None, threshold: float | None = None,
              min_margin: float | None = _KEEP) -> "ReviewedLinks":
        """Pin accepts, remove rejects, resolve remaining candidates with unchanged probabilities.

        Accepted pairs bypass threshold/margin explicitly. Automatic margins use only the
        remaining feasible candidates. Conflicting accepts raise ValueError before resolution.
        """
        source = self._doc["source"]
        settings = _settings(source["settings"])
        if how is not None:
            settings["how"] = how
        if threshold is not None:
            settings["threshold"] = threshold
        if min_margin is not _KEEP:
            settings["min_margin"] = min_margin
        settings = _settings(settings)
        scores, original = _frame(source["scores"]), _frame(source["links"])
        keys = [_key(r["left_id"], r["right_id"]) for r in source["scores"]["rows"]]
        events = [self._latest.get(key, {}) for key in keys]
        decisions = pd.Series([e.get("decision", "") for e in events], index=scores.index, dtype=object)
        accepted = scores.loc[decisions.eq("accept")].copy()
        mode = settings["how"]
        unique_sides = (["left_id"] if mode == "many-to-one" else ["right_id"] if mode == "one-to-many"
                        else ["left_id", "right_id"] if mode == "one-to-one" else [])
        for side in unique_sides:
            conflicting = accepted.loc[accepted[side].duplicated(keep=False), side].tolist()
            if conflicting:
                raise ValueError(f"conflicting manual acceptances for {side} {conflicting!r} under {mode}; "
                                 "amend a decision to reject/unsure or explicitly change how")
        feasible = ~decisions.isin(["accept", "reject"])
        for side in unique_sides:
            feasible &= ~scores[side].isin(accepted[side])
        automatic = resolve(scores.loc[feasible].copy(), **settings)
        # Keep an accepted pair's original competitor margin, including NaN for unjudged pairs.
        accepted["margin"] = _margin(scores)[accepted.index.to_numpy(dtype=int)]
        original_keys = {_key(r["left_id"], r["right_id"]) for r in source["links"]["rows"]}
        event_by_key = dict(zip(keys, events))
        annotated = []
        for frame, basis in ((accepted, "manual_accept"), (automatic, "automatic")):
            frame = frame.copy()
            selected_keys = [_key(_id(a), _id(b)) for a, b in zip(frame.left_id, frame.right_id)]
            frame["selection_basis"] = basis
            frame["original_selected"] = [key in original_keys for key in selected_keys]
            frame["review_decision"] = [event_by_key[key].get("decision", "") for key in selected_keys]
            frame["review_event_id"] = [event_by_key[key].get("event_id", "") for key in selected_keys]
            frame["review_run_id"] = self.run_id
            annotated.append(frame)
        links = pd.concat(annotated, ignore_index=True)
        settings["manual_policy"] = "accepts override threshold/margin; rejects removed; unsure unconstrained"
        settings["margin_policy"] = "automatic: feasible candidates; manual: original candidates"
        return ReviewedLinks(links, scores, original, settings, self.to_dict())


@dataclass
class ReviewedLinks:
    """Separate reviewed output; never presented as an unmodified model Result."""

    links: pd.DataFrame
    scores: pd.DataFrame
    original_links: pd.DataFrame
    settings: dict
    review: dict

    def selected_scores(self) -> pd.DataFrame:
        """Copy of original probabilities with Boolean membership in the reviewed links."""
        selected = {_key(_id(a), _id(b)) for a, b in zip(self.links.left_id, self.links.right_id)}
        frame = self.scores.copy(deep=True)
        frame["selected"] = [_key(_id(a), _id(b)) in selected for a, b in zip(frame.left_id, frame.right_id)]
        return frame

    def save(self, path: str | Path) -> Path:
        """Save authoritative typed JSON with settings, decisions and original source snapshot."""
        document = dict(format="jlink-reviewed-links", version=VERSION, applied_at=_now(),
                        source_run_id=self.review["run_id"], settings=self.settings,
                        links=_table(self.links), review=self.review)
        path = Path(path)
        path.write_text(_json(document) + "\n", encoding="utf-8")
        return path

    def write_csv(self, path: str | Path) -> Path:
        """Write a spreadsheet-safe display copy. Typed JSON remains the lossless ID authority."""
        def safe(value):
            if isinstance(value, str) and (value.lstrip().startswith(("=", "+", "-", "@"))
                                           or value.startswith(("\t", "\r", "\n"))):
                return "'" + value
            return value

        frame = self.links.copy()
        for column in frame:
            frame[column] = frame[column].map(safe)
        frame.columns = [safe(c) for c in frame.columns]
        path = Path(path)
        frame.to_csv(path, index=False)
        return path


def create_review(result: "Result", *, left: pd.DataFrame | None = None, right: pd.DataFrame | None = None,
                  close_margin: float = 0.1) -> Review:
    """Snapshot Result scores, actual links, settings and original records for local review.

    Only string, integer and finite float IDs are supported, preserving their types exactly.
    Explicit frames override Result's attached frames; loaded Results need both frames.
    """
    left = result._left if left is None else left
    right = result._right if right is None else right
    if left is None or right is None:
        raise ValueError("review needs the original left and right tables; pass left= and right=")
    # Reject invalid probabilities before JSON's representation of missing values could hide them.
    _pairs(result.scores, "review scores")
    _probabilities(result.scores)
    _probabilities(result.scores, "sim", missing=False)
    records = {}
    for side, frame in (("left", left), ("right", right)):
        expected = result.settings.get(f"n_{side}")
        if expected is not None and expected != len(frame):
            raise ValueError(
                f"review needs all original {side} records: expected {expected}, got {len(frame)}")
        values = ids(frame, result.settings.get(f"{side}_id"), side)
        if frame.columns.has_duplicates or not all(isinstance(c, str) for c in frame.columns):
            raise ValueError("original record columns must have unique string names")
        records[side] = [{"id": _id(value), "fields": {c: _value(v) for c, v in zip(frame.columns, row)}}
                         for value, row in zip(values, frame.itertuples(index=False, name=None))]
    source = _value(dict(settings=result.settings, scores=_table(result.scores), links=_table(result.links),
                         records=records, close_margin=close_margin))
    return Review(dict(format=FORMAT, version=VERSION, run_id=_source_id(source), created_at=_now(),
                       source=source, history=[]))


def read_review(path: str | Path) -> Review:
    """Load and validate a JSON review artifact from the page or Python."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"invalid review JSON: {exc}") from None
    return Review(document)

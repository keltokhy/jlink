"""Small, versioned descriptions of inputs and blocking configuration for saved runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from .block import Blocker
from .fields import ids


def blocker_config(blocker: Blocker) -> dict:
    """Prefer the public to_config protocol, with a dataclass fallback for older built-ins.

    Configurations describe passes; loading a result never imports or executes custom code.
    """
    method = getattr(blocker, "to_config", None)
    if callable(method):
        config = method()
        if not isinstance(config, dict):
            raise ValueError(f"blocker {blocker.name!r}: to_config() must return a JSON-compatible dictionary")
        return _json_value(config)
    config = {"type": f"{type(blocker).__module__}.{type(blocker).__qualname__}", "name": blocker.name,
              "serialization": "dataclass" if is_dataclass(blocker) else "opaque",
              "configuration_complete": is_dataclass(blocker), "reconstructable": False}
    if is_dataclass(blocker):
        parameters, unavailable = {}, {}
        for f in fields(blocker):
            value = getattr(blocker, f.name)
            try:
                parameters[f.name] = _json_value(value)
            except ValueError:
                # Custom dataclasses can contain functions or fitted objects; do not break a run
                # or pretend those opaque values were captured as reproducible configuration.
                unavailable[f.name] = f"{type(value).__module__}.{type(value).__qualname__}"
        config["parameters"] = parameters
        if unavailable:
            config.update(configuration_complete=False, unserialized_fields=unavailable)
    return config


def _json_value(value):
    if isinstance(value, Blocker):
        return blocker_config(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"blocker configuration contains a value that cannot be saved as JSON: {type(value).__name__}")


def _scalar(value):
    """Canonical typed values: distinguish numeric IDs from strings and preserve field boundaries."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return ["missing"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["missing"] if math.isnan(value) else ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (datetime, date)):
        return [type(value).__name__, value.isoformat()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_scalar(v) for v in value]]
    # Other scalar values are shown to Jev as text; retain their type and text here too.
    return [f"{type(value).__module__}.{type(value).__qualname__}", str(value)]


def frame_fingerprint(frame: pd.DataFrame, *, id_column: str | None, columns: list, side: str) -> dict:
    """SHA-256 of ordered IDs, field labels and raw typed values, without saving their contents."""
    columns = list(dict.fromkeys(columns))
    digest = hashlib.sha256()

    def add(value):
        digest.update(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")

    add(["jlink-input-v1", [_scalar(c) for c in columns]])
    index = ids(frame, id_column, side)
    for identity, values in zip(index, frame[columns].itertuples(index=False, name=None)):
        add([_scalar(identity), [_scalar(v) for v in values]])
    return {"algorithm": "sha256", "format": "jlink-input-v1", "sha256": digest.hexdigest(),
            "rows": len(frame), "columns": columns, "id_column": id_column, "row_order_matters": True}


def input_fingerprints(left, right, *, fields, left_id, right_id) -> dict:
    result = {}
    for side, frame, id_column, columns in (
        ("left", left, left_id, [lc for _, lc, _ in fields if lc is not None]),
        ("right", right, right_id, [rc for _, _, rc in fields if rc is not None]),
    ):
        result[side] = {
            "compared": frame_fingerprint(frame, id_column=id_column, columns=columns, side=side),
            # Including all columns covers grouping and custom blockers without guessing their internals.
            "full": frame_fingerprint(frame, id_column=id_column, columns=list(frame.columns), side=side),
        }
    return result

"""Optional local semantic retrieval with bounded exact cosine search."""

from __future__ import annotations

import json
from numbers import Real

import numpy as np
import pandas as pd

from .block import (_StreamingBlocker, _columns, _field_config, _pack_rows, _pass_fields,
                    _positive_int, _top_k)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_QUERY_ROWS = 128
_INDEX_ROWS = 4096


def serialize(frame: pd.DataFrame, fields: list, side: int) -> list[str]:
    """Preserve text and field boundaries; paired column names use the same left labels."""
    rows = []
    for values in frame[[f[side] for f in fields]].itertuples(index=False, name=None):
        record = {f[0]: str(v).strip() for f, v in zip(fields, values)
                  if not pd.isna(v) and str(v).strip()}
        rows.append((next(iter(record.values())) if len(fields) == 1 else
                     json.dumps(record, ensure_ascii=False)) if record else "")
    return rows


def embeddings(*columns: str | tuple[str, str], k: int = 10, model: str = DEFAULT_MODEL,
               revision: str | None = None, min_sim: float = 0.0, reverse: bool = False,
               batch_size: int = 64, device: str = "cpu", encoder=None,
               name: str | None = None) -> _StreamingBlocker:
    """Retrieve semantic neighbors locally; union with ngrams for hybrid candidates.

    Install ``jlink[embeddings]`` to load a SentenceTransformer lazily. A custom
    ``encoder`` may instead implement ``encode(list[str], **kwargs)``. Blank records
    and zero vectors never propose pairs. Pin ``revision`` for reproducible runs.
    ``sim`` in the final candidate table remains lexical similarity, not cosine
    from this model. Search is exact and bounded in memory, but still quadratic
    in comparisons; use ``within`` to restrict large searches when keys are reliable.
    """
    fields, name = _pass_fields(columns, name, "embeddings-reverse" if reverse else "embeddings")
    _positive_int(k, "k")
    _positive_int(batch_size, "batch_size")
    if not isinstance(reverse, (bool, np.bool_)):
        raise ValueError("`reverse` must be a boolean")
    if (isinstance(min_sim, (bool, np.bool_)) or not isinstance(min_sim, Real)
            or not np.isfinite(min_sim) or not -1 <= min_sim <= 1):
        raise ValueError("`min_sim` must be a finite cosine between -1 and 1")
    for label, value in (("model", model), ("device", device)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`{label}` must be a nonempty string")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("`revision` must be a nonempty model revision or None")
    if encoder is not None and not callable(getattr(encoder, "encode", None)):
        raise ValueError("`encoder` must provide encode(texts, **kwargs)")
    return _Embeddings(fields, name, int(k), model, revision, float(min_sim), bool(reverse),
                       int(batch_size), device, encoder)


class _Embeddings(_StreamingBlocker):
    def __init__(self, fields, name, k, model, revision, min_sim, reverse, batch_size, device, encoder):
        self.fields, self.name, self.k = fields, name, k
        self.model, self.revision, self.min_sim = model, revision, min_sim
        self.reverse, self.batch_size, self.device = reverse, batch_size, device
        self._encoder, self._custom = encoder, encoder is not None
        self._resolved_revision = None

    def to_config(self) -> dict:
        return _field_config("embeddings", self) | {
            "k": self.k, "model": self.model, "revision": self.revision,
            "resolved_revision": self._resolved_revision, "min_sim": self.min_sim,
            "reverse": self.reverse, "batch_size": self.batch_size, "device": self.device,
            "serialization": "single_text_or_labeled_json_v1", "search": "chunked_exact_cosine_v1",
            "custom_encoder": self._custom, "reconstructable": not self._custom,
        }

    def _load(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ValueError("embedding retrieval needs the optional dependency; "
                                 "install 'jlink[embeddings]' or supply encoder=") from exc
            self._encoder = SentenceTransformer(self.model, revision=self.revision,
                                                device=self.device, trust_remote_code=False)
            first = self._encoder[0]
            config = getattr(getattr(first, "auto_model", None), "config", None)
            self._resolved_revision = getattr(config, "_commit_hash", None)
        return self._encoder

    def vectors(self, left: pd.DataFrame, right: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Encode each distinct nonempty serialized record once for this invocation."""
        self._validate(left, right)
        texts = serialize(left, self.fields, 1) + serialize(right, self.fields, 2)
        unique = list(dict.fromkeys(t for t in texts if t))
        if not unique:
            return np.zeros((len(left), 0)), np.zeros((len(right), 0))
        values = np.asarray(self._load().encode(unique, batch_size=self.batch_size,
                                               show_progress_bar=False, convert_to_numpy=True), dtype=float)
        if (values.ndim != 2 or values.shape[0] != len(unique) or values.shape[1] == 0
                or not np.isfinite(values).all()):
            raise ValueError("encoder must return a finite 2D matrix with one vector per record")
        # Scale first to avoid overflow when a custom encoder returns large finite vectors.
        scale = np.max(np.abs(values), axis=1, keepdims=True)
        values = np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)
        lookup = {text: i for i, text in enumerate(unique)}
        values = np.vstack((values, np.zeros((1, values.shape[1]))))
        matrix = values[[lookup.get(text, len(unique)) for text in texts]]
        return matrix[:len(left)], matrix[len(left):]

    def iter_pairs(self, left: pd.DataFrame, right: pd.DataFrame):
        _columns(left, right, self.fields)
        if left.empty or right.empty:
            return
        x, y = self.vectors(left, right)
        if self.reverse:
            x, y = y, x
        for pairs in _pack_rows(self._neighbors(x, y)):
            yield pairs[:, ::-1].copy() if self.reverse else pairs

    def _neighbors(self, x, y):
        valid_y = np.flatnonzero(np.any(y != 0, axis=1))
        for start in range(0, len(x), _QUERY_ROWS):
            q = x[start:start + _QUERY_ROWS]
            best_pos = [np.empty(0, dtype=np.int64) for _ in q]
            best_sim = [np.empty(0, dtype=float) for _ in q]
            for offset in range(0, len(valid_y), _INDEX_ROWS):
                positions = valid_y[offset:offset + _INDEX_ROWS]
                product = np.clip(q @ y[positions].T, -1, 1)
                for i, scores in enumerate(product):
                    if not np.any(q[i]):
                        continue
                    keep = scores >= self.min_sim
                    pos = np.concatenate((best_pos[i], positions[keep]))
                    sim = np.concatenate((best_sim[i], scores[keep]))
                    # Resolve cosine ties by original neighbor position across chunks.
                    order = np.argsort(pos, kind="stable")
                    chosen = order[_top_k(np.arange(len(order)), sim[order], self.k)]
                    best_pos[i], best_sim[i] = pos[chosen], sim[chosen]
            for i, positions in enumerate(best_pos):
                yield start + i, positions

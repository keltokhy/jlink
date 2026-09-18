"""Optional multi-field Fellegi-Sunter ECM baseline, using the bench-only recordlinkage package."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jlink.fields import normalize, parse_on
from bench.evaluation import validate_pairs


def comparison_vectors(left, right, candidates, *, on) -> pd.DataFrame:
    """One binary Jaro-Winkler >= .9 agreement per field; missing values disagree.

    Keep fields separate so ECM can learn match/nonmatch agreement probabilities. This is a
    fixed generic specification, not a claim that binary agreement is best for every field.
    """
    import jellyfish
    validate_pairs(candidates, left, right)
    fields = parse_on(on)
    values = {}
    a, b = left.set_index("id"), right.set_index("id")
    for label, lc, rc in fields:
        x, y = a[lc].map(normalize).to_dict(), b[rc].map(normalize).to_dict()
        values[label] = [int(bool(x[i] and y[j]) and jellyfish.jaro_winkler_similarity(x[i], y[j]) >= .9)
                         for i, j in candidates[["left_id", "right_id"]].itertuples(index=False, name=None)]
    return pd.DataFrame(values, index=pd.MultiIndex.from_frame(candidates[["left_id", "right_id"]]))


class InsufficientComparisons(ValueError):
    """The observed development features cannot identify a multi-field model."""


class ECMBaseline:
    """Fit unsupervised on development candidates only; freeze parameters for test scoring."""

    def fit(self, features: pd.DataFrame) -> "ECMBaseline":
        try:
            import recordlinkage
        except ImportError as exc:
            raise ValueError("ECM needs the optional bench dependencies: uv sync --group bench") from exc
        if features.shape[1] < 2:
            raise ValueError("ECM needs multiple comparison fields; it is unsuitable for name-only NBER")
        self.input_columns = list(features.columns)
        self.columns = [col for col in features if features[col].nunique() == 2]
        if len(self.columns) < 2:
            raise InsufficientComparisons("ECM needs at least two varying development comparison fields")
        features = features[self.columns]
        self.model = recordlinkage.ECMClassifier(init="jaro", max_iter=100, atol=1e-5)
        self.model.fit(features)
        self.settings = {"implementation": "recordlinkage.ECMClassifier", "init": "jaro",
                         "max_iter": 100, "atol": 1e-5, "feature": "per-field Jaro-Winkler >= 0.9",
                         "missing": "disagreement", "training": "development candidates, no labels",
                         "fields": self.columns,
                         "excluded_constant_fields": [c for c in self.input_columns if c not in self.columns],
                         "m": {field: {str(level): float(p) for level, p in levels.items()}
                               for field, levels in self.model.m_probs.items()},
                         "u": {field: {str(level): float(p) for level, p in levels.items()}
                               for field, levels in self.model.u_probs.items()}, "prior": float(self.model.p)}
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        if list(features.columns) != self.input_columns:
            raise ValueError("ECM test fields differ from development fields")
        values = (self.model.prob(features[self.columns]).to_numpy(dtype=float)
                  if len(features) else np.array([]))
        if not np.isfinite(values).all():
            raise ValueError("ECM returned nonfinite probabilities; inspect degenerate agreement patterns")
        return values

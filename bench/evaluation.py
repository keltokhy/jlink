"""Offline benchmark primitives: entity splits, candidate adapters, and stage-specific metrics."""

from __future__ import annotations

import hashlib
import json
import math
import warnings

import numpy as np
import pandas as pd

from bench.baselines import score_against_truth
from bench.data import PAIR_COLUMNS, pair_set
from jlink import block
from jlink.fields import normalize, parse_on, record_text
from jlink.resolve import resolve


def digest(value) -> str:
    """Hash canonical JSON; reject non-finite values rather than silently changing evidence."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_pairs(frame: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                   label: str = "candidates") -> None:
    """Require unique, in-universe pairs for comparable candidate and score tables."""
    if len(pair_set(frame, label)) != len(frame):
        raise ValueError(f"{label} contains duplicate pairs")
    for column, source in (("left_id", left), ("right_id", right)):
        if not frame[column].isin(source.id).all():
            raise ValueError(f"{label} has {column} values absent from the records")


def entity_split(left: pd.DataFrame, right: pd.DataFrame, truth: pd.DataFrame, *, on,
                 dev_fraction: float = .3, seed: int = 1729) -> pd.DataFrame:
    """Keep truth-connected entities and identical nonempty record signatures in one partition.

    This is disjoint in *observed* entities, not proof that incomplete labels reveal every alias.
    Names alone do not merge records unless they are the full declared comparison signature.
    """
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must lie strictly between 0 and 1")
    validate_pairs(truth, left, right, "truth")
    nodes = [(side, str(value)) for side, frame in (("left", left), ("right", right)) for value in frame.id]
    if len(set(nodes)) != len(nodes):
        raise ValueError("record IDs must be unique within each source after string conversion")
    parent = {node: node for node in nodes}

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)

    for a, b in truth[PAIR_COLUMNS].itertuples(index=False, name=None):
        union(("left", str(a)), ("right", str(b)))
    signatures = {}
    fields = parse_on(on)
    for side, frame, pos in (("left", left, 1), ("right", right, 2)):
        for row in frame.to_dict("records"):
            signature = tuple(normalize(row[f[pos]]) for f in fields)
            if not any(signature):
                continue
            node = (side, str(row["id"]))
            if signature in signatures:
                union(node, signatures[signature])
            else:
                signatures[signature] = node
    components = {}
    for node in nodes:
        components.setdefault(root(node), []).append(node)
    groups = {key: digest(sorted(value)) for key, value in components.items()}
    ordered = sorted(groups, key=lambda key: digest([seed, groups[key]]))
    if len(ordered) < 2:
        raise ValueError("entity-disjoint splitting needs at least two observed entity groups")
    ndev = max(1, min(len(ordered) - 1, int(len(ordered) * dev_fraction)))
    development = set(ordered[:ndev])
    return pd.DataFrame([{"side": side, "id": value, "group": groups[root((side, value))],
                          "split": "dev" if root((side, value)) in development else "test"}
                         for side, value in sorted(nodes)])


def subset(left, right, truth, split, name):
    """Restrict both search universes; no truth-connected entity crosses the split."""
    chosen = split.loc[split.split.eq(name)]
    a = left.loc[left.id.isin(chosen.loc[chosen.side.eq("left"), "id"])].copy()
    b = right.loc[right.id.isin(chosen.loc[chosen.side.eq("right"), "id"])].copy()
    gold = truth.loc[truth.left_id.isin(a.id) & truth.right_id.isin(b.id)].copy()
    return a, b, gold


def make_unmatched(left, right, truth, *, fraction: float, seed: int = 1729):
    """Remove opposite counterparts of disjoint, completely labeled one-to-one entities.

    Only the FEBRL caller may use this transform: its underlying identities are known.
    The unmatched labels describe the resulting universe, never the original full source.
    """
    if not 0 <= fraction < .5:
        raise ValueError("unmatched fraction must be at least 0 and below .5")
    validate_pairs(truth, left, right, "truth")
    if (truth.left_id.duplicated().any() or truth.right_id.duplicated().any()
            or set(truth.left_id) != set(left.id) or set(truth.right_id) != set(right.id)):
        raise ValueError("unmatched construction needs complete one-to-one identity labels for all records")
    pairs = sorted(pair_set(truth), key=lambda pair: digest([seed, *pair]))
    n = int(len(pairs) * fraction)
    keep_left, keep_right = pairs[:n], pairs[n:2*n]
    remove_left, remove_right = {x for x, _ in keep_right}, {y for _, y in keep_left}
    a, b = left.loc[~left.id.isin(remove_left)].copy(), right.loc[~right.id.isin(remove_right)].copy()
    gold = truth.loc[truth.left_id.isin(a.id) & truth.right_id.isin(b.id)].copy()
    labels = pd.DataFrame([{"side": "left", "id": x, "removed_counterpart": y} for x, y in keep_left]
                          + [{"side": "right", "id": y, "removed_counterpart": x} for x, y in keep_right],
                          columns=["side", "id", "removed_counterpart"])
    return a, b, gold, labels


def propose(left, right, *, on, specs: list[dict], registry: dict | None = None):
    """Build existing public blockers from JSON; registry is the hook for future blockers."""
    factories = {"ngrams": block.ngrams, "exact": block.exact, "initials": block.initials}
    factories.update(registry or {})
    passes = []
    for spec in specs:
        unknown = set(spec) - {"kind", "columns", "kwargs"}
        if unknown or spec.get("kind") not in factories:
            raise ValueError(f"unknown blocker configuration: {spec}")
        columns = spec.get("columns", on)
        columns = [tuple(c) if isinstance(c, list) else c for c in columns]
        passes.append(factories[spec["kind"]](*columns, **spec.get("kwargs", {})))
    if not passes:
        raise ValueError("configure at least one blocker")
    return block.candidates(left, right, on=on, blockers=passes, left_id="id", right_id="id")


def string_scores(left, right, candidates, *, on, method: str) -> pd.DataFrame:
    """Score exactly the shared candidate table, with no private blocking or best-match step."""
    validate_pairs(candidates, left, right)
    fields = parse_on(on)
    a, b = left.set_index("id"), right.set_index("id")
    if method == "exact":
        values = np.ones(len(candidates), dtype=bool)
        for _, lc, rc in fields:
            x = a[lc].map(normalize).reindex(candidates.left_id).to_numpy()
            y = b[rc].map(normalize).reindex(candidates.right_id).to_numpy()
            values &= (x == y) & (x != "") & (y != "")
    elif method == "tfidf":
        # block.candidates already computes this on every candidate using all on fields.
        values = candidates.sim.to_numpy(dtype=float)
    elif method == "jaro_winkler":
        import jellyfish
        x = dict(zip(left.id, record_text(left, [f[1] for f in fields])))
        y = dict(zip(right.id, record_text(right, [f[2] for f in fields])))
        values = [jellyfish.jaro_winkler_similarity(x[i], y[j]) if x[i] and y[j] else 0.0
                  for i, j in candidates[PAIR_COLUMNS].itertuples(index=False, name=None)]
    else:
        raise ValueError(f"unknown baseline {method!r}")
    return candidates.assign(p=np.asarray(values, dtype=float), source=method)


def pair_metrics(predicted, truth, *, complete: bool) -> dict:
    """For incomplete crosswalks report listed-positive recall; unlisted predictions stay unknown."""
    predicted, actual = pair_set(predicted), pair_set(truth)
    tp = len(predicted & actual)
    result = {"listed_tp": tp, "listed_fn": len(actual - predicted), "predicted_pairs": len(predicted),
              "listed_positive_recall": tp / len(actual) if actual else None}
    if not complete:
        return dict(result, unlisted_predictions_unknown=len(predicted - actual),
                    precision=None, recall=result["listed_positive_recall"], f1=None, fp=None)
    return dict(result, **score_against_truth(pd.DataFrame(sorted(predicted), columns=PAIR_COLUMNS), truth))


def validate_scores(scores: pd.DataFrame) -> None:
    if len(pair_set(scores, "scores")) != len(scores):
        raise ValueError("scores contains duplicate pairs")
    if "p" not in scores:
        raise ValueError("scores needs column 'p'")
    values = pd.to_numeric(scores.p, errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("scores p must be between 0 and 1, or missing for unjudged pairs")


def choose_threshold(scores, truth, *, how: str, grid: list[float]) -> tuple[float | None, list[dict]]:
    """Select on development final-assignment F1 only; None is the explicit reject-all action."""
    validate_scores(scores)
    if not len(truth):
        raise ValueError("development truth is empty; threshold selection is not supported")
    if not grid or any(isinstance(t, bool) or not math.isfinite(t) or not 0 <= t <= 1 for t in grid):
        raise ValueError("threshold grid must contain finite numbers between 0 and 1")
    scores = scores.assign(p=pd.to_numeric(scores.p))
    trials = [{"threshold": None, "f1": 0.0, "links": 0, "resolver_warnings": []}]
    for threshold in sorted(set(grid), reverse=True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            links = resolve(scores, how=how, threshold=threshold, min_margin=None)
        trials.append({"threshold": threshold, "f1": score_against_truth(links, truth)["f1"],
                       "links": len(links), "resolver_warnings": sorted({str(w.message) for w in caught})})
    winner = max(trials, key=lambda row: (row["f1"], -row["links"]))
    return winner["threshold"], trials


def evaluate_stages(candidates, scores, truth, *, how: str, threshold: float | None,
                    complete: bool, unmatched: pd.DataFrame | None = None,
                    score_is_probability: bool = True) -> tuple[dict, pd.DataFrame]:
    """Separate blocking loss, conditional pair judgments, and final resolved links.

    Unjudged candidate positives remain in conditional recall's denominator; Brier uses only
    scored pairs and only labels declared exhaustive. Scores must cover the candidate table.
    """
    validate_scores(scores)
    scores = scores.assign(p=pd.to_numeric(scores.p))
    proposed, gold = pair_set(candidates), pair_set(truth)
    if pair_set(scores) != proposed or len(candidates) != len(proposed):
        raise ValueError("scores must contain each candidate exactly once (use missing p for unjudged)")
    covered = pd.DataFrame(sorted(proposed & gold), columns=PAIR_COLUMNS)
    judged = scores.loc[scores.p.notna()]
    decisions = judged.iloc[:0] if threshold is None else judged.loc[judged.p.ge(threshold)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        links = scores.iloc[:0].copy() if threshold is None else resolve(
            scores, how=how, threshold=threshold, min_margin=None)
    conditional = pair_metrics(decisions, covered, complete=complete)
    conditional.update(scored_pairs=len(judged), unjudged_pairs=len(scores) - len(judged),
                       brier=None, calibration=[])
    if complete and len(judged) and score_is_probability:
        labels = np.array([pair in gold for pair in judged[PAIR_COLUMNS].itertuples(index=False, name=None)])
        conditional["brier"] = float(np.mean((judged.p.to_numpy() - labels) ** 2))
        table = pd.DataFrame({"p": judged.p.to_numpy(), "label": labels})
        table["bin"] = pd.cut(table.p, [0, .05, .2, .5, .8, .95, 1], include_lowest=True)
        conditional["calibration"] = [{"bin": str(key), "n": len(group), "mean_p": float(group.p.mean()),
                                        "match_rate": float(group.label.mean())}
                                       for key, group in table.groupby("bin", observed=True)]
    conditional["score_is_probability"] = score_is_probability
    conditional["by_source"] = {}
    if "source" in scores:
        for source, group in scores.groupby(scores.source.fillna("unjudged"), sort=True):
            source_gold = pd.DataFrame(sorted(pair_set(group) & gold), columns=PAIR_COLUMNS)
            accepted = group.iloc[:0] if threshold is None else group.loc[group.p.ge(threshold)]
            conditional["by_source"][source] = dict(pair_metrics(accepted, source_gold, complete=complete),
                                                    pairs=len(group),
                                                    unjudged_pairs=int(group.p.isna().sum()))
    final = pair_metrics(links, truth, complete=complete)
    final["resolver_warnings"] = sorted({str(w.message) for w in caught})
    if unmatched is not None:
        final["explicit_unmatched"] = {}
        for side in ("left", "right"):
            known = set(unmatched.loc[unmatched.side.eq(side), "id"])
            linked = set(links[f"{side}_id"])
            false_linked = len(known & linked)
            final["explicit_unmatched"][side] = {"records": len(known), "false_linked": false_linked,
                                                "correct_abstention_rate": 1 - false_linked / len(known)
                                                if known else None}
    return {"candidates": {"pairs": len(proposed), "truth_pairs": len(gold),
                            "covered_truth_pairs": len(proposed & gold),
                            "recall": len(proposed & gold) / len(gold) if gold else None},
            "judge_conditional_on_candidates": conditional, "final_assignment": final}, links

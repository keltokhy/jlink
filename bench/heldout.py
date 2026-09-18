"""Offline, shared-candidate development/test evaluation. No API client or live option."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data import DATA, NAMES, ROOT, PAIR_COLUMNS, load_dataset
from bench.evaluation import (choose_threshold, digest, entity_split, evaluate_stages, make_unmatched,
                              propose, string_scores, subset, validate_pairs, validate_scores)

DEFAULT_CONFIG = {"seed": 1729, "dev_fraction": .3, "candidate_source": "configured",
                  "blockers": [{"kind": "ngrams", "kwargs": {"k": 10}}],
                  "threshold_grid": [i / 20 for i in range(21)], "unmatched_fraction": 0.0}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(dataset: str, *, data_dir: Path = DATA, out_dir: Path = ROOT / "out" / "heldout",
        config: dict | None = None, cached_run: Path | None = None, ecm: bool = False) -> dict:
    """Freeze observed-entity splits, select baseline thresholds on dev, and score test once.

    Cached results are retrospective: a new split cannot undo earlier prompt or method tuning.
    Cached candidate sets retain their original corpus fit/top-k, filtered to each partition.
    """
    settings = dict(DEFAULT_CONFIG, **(config or {}))
    unknown = set(settings) - set(DEFAULT_CONFIG) - {"how"}
    if unknown:
        raise ValueError(f"unknown evaluation settings: {sorted(unknown)}")
    if settings["candidate_source"] not in {"configured", "cached"}:
        raise ValueError("candidate_source must be configured or cached")
    if settings["candidate_source"] == "cached" and cached_run is None:
        raise ValueError("cached candidates require --cached-run")
    left, right, truth, meta = load_dataset(dataset, data_dir)
    # All benchmark IDs are strings; prevent CSV inference from destroying leading zeroes.
    left, right = left.assign(id=left.id.astype(str)), right.assign(id=right.id.astype(str))
    truth = truth.astype({"left_id": str, "right_id": str})
    source_hashes = {name: file_hash(Path(data_dir) / dataset / name)
                     for name in ("left.parquet", "right.parquet", "truth.parquet", "meta.json")}
    how = settings.get("how", meta["how"])
    complete = dataset != "nber-firms"
    label_policy = ("complete synthetic identity labels" if dataset == "febrl4" else
                    "positive-only historical crosswalk; unlisted pairs unknown" if not complete else
                    ("closed-world supplied perfectMapping; absences are benchmark negatives, "
                     "not independently verified"))
    split = entity_split(left, right, truth, on=meta["on"],
                         dev_fraction=settings["dev_fraction"], seed=settings["seed"])
    cache, provenance = None, None
    if cached_run is not None:
        cached_run = Path(cached_run)
        cache = pd.read_csv(cached_run / "scores.csv", dtype={"left_id": str, "right_id": str})
        validate_pairs(cache, left, right, "cached scores")
        validate_scores(cache)
        saved_settings = json.loads((cached_run / "settings.json").read_text())
        cached_fields = [tuple(item) if isinstance(item, list) else (item, item)
                         for item in saved_settings["on"]]
        expected_fields = [tuple(item) if isinstance(item, (list, tuple)) else (item, item)
                           for item in meta["on"]]
        if cached_fields != expected_fields or saved_settings["entity"] != meta["entity"]:
            raise ValueError("cached settings fields/entity differ from this dataset")
        provenance = {"files": {name: file_hash(cached_run / name)
                                for name in ("scores.csv", "settings.json", "result.json")
                                if (cached_run / name).exists()}, "settings": saved_settings,
                      "status": "retrospective cached replay; no new model calls",
                      "raw_provider_responses": "not in legacy saved run; scores are parsed probabilities",
                      "dataset_binding": "legacy cache has no dataset hashes; current ID/field checks only",
                      "prior_tuning": ("NBER rule revised after a 300-record pilot; this split cannot undo that"
                                       if dataset == "nber-firms" else
                                       "no new prompt tuning; prior exposure not excluded")}
    directory = Path(out_dir) / dataset
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} already contains artifacts; choose a new --out directory")
    directory.mkdir(parents=True, exist_ok=True)
    split.to_csv(directory / "split.csv", index=False)
    partitions, candidate_diagnostics = {}, {}
    for name in ("dev", "test"):
        a, b, gold = subset(left, right, truth, split, name)
        if not len(a) or not len(b) or not len(gold):
            raise ValueError(f"{name} needs nonempty left/right records and truth; choose a different split")
        unmatched = pd.DataFrame(columns=["side", "id", "removed_counterpart"])
        if settings["unmatched_fraction"]:
            if dataset != "febrl4":
                raise ValueError("explicit unmatched construction is supported only for FEBRL's "
                                 "known identities")
            a, b, gold, unmatched = make_unmatched(a, b, gold, fraction=settings["unmatched_fraction"],
                                                  seed=settings["seed"])
        if settings["candidate_source"] == "cached":
            candidates = cache.loc[cache.left_id.isin(a.id) & cache.right_id.isin(b.id),
                                   [*PAIR_COLUMNS, "sim", "block"]].reset_index(drop=True)
        else:
            candidates = propose(a, b, on=meta["on"], specs=settings["blockers"])
        candidate_diagnostics[name] = candidates.attrs.get("blocking")
        partitions[name] = (a, b, gold, candidates, unmatched)
        candidates.to_csv(directory / f"{name}-candidates.csv", index=False)
        unmatched.to_csv(directory / f"{name}-unmatched.csv", index=False)
    method_scores = {}
    for method in ("exact", "jaro_winkler", "tfidf"):
        method_scores[method] = {name: string_scores(a, b, candidates, on=meta["on"], method=method)
                                for name, (a, b, _, candidates, _) in partitions.items()}
    optional = {"ecm": {"status": "not_requested"}}
    if ecm:
        from bench.probabilistic import ECMBaseline, InsufficientComparisons, comparison_vectors
        if len(meta["on"]) < 2:
            optional["ecm"] = {"status": "not_applicable", "reason": "single comparison field"}
        else:
            features = {name: comparison_vectors(a, b, c, on=meta["on"])
                        for name, (a, b, _, c, _) in partitions.items()}
            try:
                model = ECMBaseline().fit(features["dev"])
            except InsufficientComparisons as exc:
                optional["ecm"] = {"status": "not_applicable", "reason": str(exc)}
            else:
                method_scores["ecm"] = {
                    name: partitions[name][3].assign(p=model.score(features[name]), source="ecm")
                    for name in partitions}
                optional["ecm"] = {"status": "measured", "settings": model.settings}
    if cache is not None:
        method_scores["cached_jev"] = {}
        for name, (_, _, _, candidates, _) in partitions.items():
            # Missing pairs remain unjudged; candidate coverage is reported instead of fabricated scores.
            method_scores["cached_jev"][name] = candidates.merge(
                cache[[*PAIR_COLUMNS, "p", "source"]], on=PAIR_COLUMNS, how="left", validate="one_to_one")
    results = {}
    for method, scores in method_scores.items():
        trials = []
        if method == "exact":
            threshold, policy = 1.0, "fixed normalized exact"
        elif method == "cached_jev":
            threshold, policy = .5, "fixed 0.5; cached historical model decisions"
        elif not complete:
            threshold, policy = .5, "fixed 0.5; F1 selection disabled without negative gold labels"
        else:
            threshold, trials = choose_threshold(scores["dev"], partitions["dev"][2], how=how,
                                                  grid=settings["threshold_grid"])
            policy = "development final-assignment F1 on prespecified grid; ties prefer fewer links"
        results[method] = {"threshold": threshold, "threshold_policy": policy, "dev_trials": trials}
        for name, (_, _, gold, candidates, unmatched) in partitions.items():
            metrics, links = evaluate_stages(candidates, scores[name], gold, how=how, threshold=threshold,
                                             complete=complete, unmatched=unmatched,
                                             score_is_probability=method in {"ecm", "cached_jev"})
            results[method][name] = metrics
            scores[name].to_csv(directory / f"{name}-{method}-scores.csv", index=False)
            links.to_csv(directory / f"{name}-{method}-links.csv", index=False)
    report = {"schema_version": 1, "dataset": dataset, "config": settings, "how": how,
              "min_margin": None, "label_policy": label_policy, "source_meta": meta,
              "source_sha256": source_hashes, "dataset_sha256": digest(source_hashes),
              "split": {"method": "truth-connected components plus identical nonempty on-field signatures",
                        "qualification": ("disjoint observed entities; "
                                          "missing aliases may remain undiscovered"),
                        "sha256": file_hash(directory / "split.csv"),
                        "definition": ("sort group SHA256 by SHA256([seed, group]); "
                                       "floor(dev_fraction * groups) to dev"),
                        "counts": {name: {"left": len(a), "right": len(b), "truth": len(gold),
                                         "explicit_unmatched": len(unmatched)}
                                   for name, (a, b, gold, _, unmatched) in partitions.items()}},
              "candidate_policy": ("same pairs, same how and min_margin for every method; "
                                   "cached top-k filtered within split (not reretrieved)"
                                   if settings["candidate_source"] == "cached" else
                                   "same configured candidates, same how and min_margin for every method"),
              "candidate_diagnostics": candidate_diagnostics,
              "cached_run": provenance, "optional_baselines": optional, "results": results,
              "code_sha256": {name: file_hash(ROOT.parent / name) for name in (
                  "bench/heldout.py", "bench/evaluation.py", "bench/probabilistic.py", "bench/data.py",
                  "bench/baselines.py", "src/jlink/resolve.py", "src/jlink/block.py", "src/jlink/fields.py")},
              "new_api_calls": 0, "new_api_cost_usd": 0,
              "environment": {"python": platform.python_version(), "packages": {
                  name: importlib.metadata.version(name) for name in
                  ("jlink", "pandas", "numpy", "scipy", "scikit-learn", "jellyfish")}},
              "artifacts": {path.name: file_hash(path) for path in sorted(directory.glob("*.csv"))}}
    if ecm:
        report["environment"]["packages"]["recordlinkage"] = importlib.metadata.version("recordlinkage")
    write_json(directory / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=NAMES)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "heldout")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cached-run", type=Path)
    parser.add_argument("--ecm", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text()) if args.config else None
        result = run(args.dataset, data_dir=args.data_dir, out_dir=args.out, config=config,
                     cached_run=args.cached_run, ecm=args.ecm)
    except (ValueError, OSError, ImportError) as exc:
        print(f"bench heldout: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"dataset": result["dataset"], "how": result["how"], "new_api_calls": 0,
                      "report": str(args.out / args.dataset / "report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

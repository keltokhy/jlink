"""Compare LinkTransformer's own link decisions with cached Jev judgments on the frozen splits.

`prepare` writes partitions, the shared candidate pool and development training pairs.
`evaluate` reads cosines exported by bench/linktransformer_scores.py (run in LinkTransformer's
own environment) and scores every arm with the shared resolver. No API client, no live option.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data import DATA, NAMES, ROOT, PAIR_COLUMNS, load_dataset, pair_set
from bench.evaluation import (choose_threshold, digest, entity_split, evaluate_stages, string_scores,
                              subset, validate_pairs, validate_scores)
from bench.heldout import file_hash, write_json
from jlink.resolve import resolve

SEED, DEV_FRACTION, TUNE_FRACTION, NEGATIVES_PER_LEFT = 1729, .3, .3, 3
COARSE_GRID = [i / 20 for i in range(21)]      # the grid the string baselines have always used
COSINE_GRID = [i / 100 for i in range(101)]    # finer for cosines, which cluster near the top


def columns(meta) -> list[str]:
    fields = [c if isinstance(c, str) else c[0] for c in meta["on"]]
    if any(not isinstance(c, str) and c[0] != c[1] for c in meta["on"]):
        raise ValueError("this comparison expects the same field names on both sides")
    return fields


def partitions(dataset: str, data_dir: Path, cached_run: Path):
    """The same observed-entity split and cached candidate pool as bench.heldout's cached replay."""
    left, right, truth, meta = load_dataset(dataset, data_dir)
    left, right = left.assign(id=left.id.astype(str)), right.assign(id=right.id.astype(str))
    truth = truth.astype({"left_id": str, "right_id": str})
    cache = pd.read_csv(cached_run / "scores.csv", dtype={"left_id": str, "right_id": str})
    validate_pairs(cache, left, right, "cached scores")
    validate_scores(cache)
    split = entity_split(left, right, truth, on=meta["on"], dev_fraction=DEV_FRACTION, seed=SEED)
    # Development entities are divided again: most train LinkTransformer, the rest pick its threshold.
    groups = sorted(set(split.loc[split.split.eq("dev"), "group"]), key=lambda g: digest([SEED, "tune", g]))
    tune = set(groups[:max(1, int(len(groups) * TUNE_FRACTION))])
    split["dev_role"] = [None if s != "dev" else "tune" if g in tune else "train"
                         for s, g in zip(split.split, split.group)]
    parts = {}
    for name in ("dev", "test"):
        a, b, gold = subset(left, right, truth, split, name)
        candidates = cache.loc[cache.left_id.isin(a.id) & cache.right_id.isin(b.id)].reset_index(drop=True)
        parts[name] = (a, b, gold, candidates)
    return parts, split, meta


def training_pairs(parts, split, fields, *, complete: bool) -> pd.DataFrame:
    """All development-train matches, plus the hardest shared-candidate nonmatches where labels allow."""
    a, b, gold, candidates = parts["dev"]
    role = split.loc[split.dev_role.eq("train")]
    lefts, rights = set(role.loc[role.side.eq("left"), "id"]), set(role.loc[role.side.eq("right"), "id"])
    positives = gold.loc[gold.left_id.isin(lefts) & gold.right_id.isin(rights), PAIR_COLUMNS].assign(label=1)
    frames = [positives]
    if complete:
        known = pair_set(gold)
        pool = candidates.loc[candidates.left_id.isin(lefts) & candidates.right_id.isin(rights)]
        pool = pool.loc[[pair not in known for pair in pool[PAIR_COLUMNS].itertuples(index=False, name=None)]]
        hardest = pool.sort_values(["left_id", "sim"], ascending=[True, False]).groupby("left_id").head(
            NEGATIVES_PER_LEFT)
        frames.append(hardest[PAIR_COLUMNS].assign(label=0))
    pairs = pd.concat(frames, ignore_index=True)
    out = pairs.rename(columns={"left_id": "l_id", "right_id": "r_id"})
    for prefix, frame, key in (("l_", a, "l_id"), ("r_", b, "r_id")):
        out = out.merge(frame[["id", *fields]].rename(columns={"id": key, **{c: prefix + c for c in fields}}),
                        on=key, how="left", validate="many_to_one")
    return out


def prepare(dataset: str, *, data_dir: Path, cached_run: Path, out_dir: Path) -> dict:
    directory = out_dir / dataset
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} already contains artifacts; choose a new --out directory")
    directory.mkdir(parents=True, exist_ok=True)
    parts, split, meta = partitions(dataset, data_dir, cached_run)
    fields, complete = columns(meta), dataset != "nber-firms"
    split.to_csv(directory / "split.csv", index=False)
    for name, (a, b, gold, candidates) in parts.items():
        a[["id", *fields]].to_parquet(directory / f"{name}-left.parquet", index=False)
        b[["id", *fields]].to_parquet(directory / f"{name}-right.parquet", index=False)
        gold.to_csv(directory / f"{name}-truth.csv", index=False)
        candidates.to_csv(directory / f"{name}-candidates.csv", index=False)
    train = training_pairs(parts, split, fields, complete=complete)
    train.to_csv(directory / "train-pairs.csv", index=False)
    request = {"dataset": dataset, "on": fields, "how": meta["how"], "k": 10, "complete_labels": complete,
               "seed": SEED, "dev_fraction": DEV_FRACTION, "tune_fraction": TUNE_FRACTION,
               "negatives_per_left": NEGATIVES_PER_LEFT if complete else 0,
               "training_pairs": {"positives": int(train.label.eq(1).sum()),
                                  "negatives": int(train.label.eq(0).sum())},
               "counts": {name: {"left": len(a), "right": len(b), "truth": len(gold), "candidates": len(c)}
                          for name, (a, b, gold, c) in parts.items()},
               "cached_run": str(cached_run), "cached_scores_sha256": file_hash(cached_run / "scores.csv")}
    write_json(directory / "request.json", request)
    return request


def cosine_scores(path: Path, left, right) -> pd.DataFrame:
    """Cosines are similarities, not probabilities; negative values cannot pass any threshold here."""
    frame = pd.read_csv(path, dtype={"left_id": str, "right_id": str})
    validate_pairs(frame, left, right, path.name)
    cosine = frame.cosine.clip(0, 1)
    return frame.assign(p=cosine, sim=cosine, block="linktransformer", source=path.stem)


def final(metrics: dict) -> dict:
    stage = metrics["final_assignment"]
    return {"precision": stage["precision"], "recall": stage["recall"], "f1": stage["f1"],
            "links": stage["predicted_pairs"], "candidate_recall": metrics["candidates"]["recall"],
            "resolver_warnings": stage["resolver_warnings"]}


def restrict(frame: pd.DataFrame, split: pd.DataFrame, role: str) -> pd.DataFrame:
    chosen = split.loc[split.dev_role.eq(role)]
    lefts, rights = set(chosen.loc[chosen.side.eq("left"), "id"]), set(chosen.loc[chosen.side.eq("right"), "id"])
    return frame.loc[frame.left_id.isin(lefts) & frame.right_id.isin(rights)].reset_index(drop=True)


def evaluate(dataset: str, *, data_dir: Path, cached_run: Path, out_dir: Path) -> dict:
    directory = out_dir / dataset
    request = json.loads((directory / "request.json").read_text())
    if file_hash(cached_run / "scores.csv") != request["cached_scores_sha256"]:
        raise ValueError("cached scores changed since prepare")
    parts, split, meta = partitions(dataset, data_dir, cached_run)
    split = split.astype({"id": str})
    how, complete = meta["how"], request["complete_labels"]
    exported = json.loads((directory / "linktransformer.json").read_text())
    arms = {}

    def add(name, scores, *, threshold_policy, grid=None, fixed=None, tuned_on="dev", candidates=None,
            how=meta["how"], **extra):
        """Score one arm; tuned arms pick a threshold on development labels only, then also report
        the threshold that is best on the test answers as an upper bound no tuning could exceed."""
        a, b, gold, shared = parts["test"]
        pool = {part: (candidates or {}).get(part, parts[part][3])[PAIR_COLUMNS] for part in ("dev", "test")}
        # Fine-tuned models are tuned and compared on development entities their training never saw.
        dev_scores, dev_gold, dev_pool = scores["dev"], parts["dev"][2], pool["dev"]
        if tuned_on == "tune":
            dev_scores, dev_gold, dev_pool = (restrict(frame, split, "tune")
                                              for frame in (dev_scores, dev_gold, dev_pool))
        threshold = fixed
        if fixed is None:
            threshold, _ = choose_threshold(dev_scores, dev_gold, how=how, grid=grid)
        dev_metrics, _ = evaluate_stages(dev_pool, dev_scores, dev_gold, how=how, threshold=threshold,
                                         complete=True, score_is_probability=False)
        metrics, links = evaluate_stages(pool["test"], scores["test"], gold, how=how,
                                         threshold=threshold, complete=True, score_is_probability=False)
        arm = dict(extra, how=how, threshold=threshold, threshold_policy=threshold_policy, test=final(metrics),
                   dev_f1=dev_metrics["final_assignment"]["f1"], dev_entities=tuned_on)
        if fixed is None:
            best, _ = choose_threshold(scores["test"], gold, how=how, grid=grid)
            oracle, _ = evaluate_stages(pool["test"], scores["test"], gold, how=how,
                                        threshold=best, complete=True, score_is_probability=False)
            arm["test_oracle_threshold"] = dict(final(oracle), threshold=best,
                                                note="threshold chosen on test answers; upper bound only")
        links.to_csv(directory / f"test-{name}-links.csv", index=False)
        arms[name] = arm

    # Every method is scored under the dataset's own cardinality and, where that allows several links
    # per record, also as best match per left record and as a one-to-one assignment. jlink's unlabelled
    # headline is its shipped rule; choosing among rules on development F1 is a labelled choice.
    if complete:
        jev = {part: parts[part][3][[*PAIR_COLUMNS, "sim", "block", "p", "source"]] for part in parts}
        modes = ({"": how, "_top1": "many-to-one", "_1to1": "one-to-one"} if how == "many-to-many"
                 else {"": how})
        for suffix, mode in modes.items():
            add("jlink_jev" + suffix, jev, fixed=.5, how=mode, labels_used=0,
                threshold_policy="fixed 0.5; the dataset's shipped link rule" if not suffix
                else "fixed 0.5; an alternative link rule")
        tfidf = {part: string_scores(*parts[part][:2], parts[part][3][[*PAIR_COLUMNS, "sim", "block"]],
                                     on=meta["on"], method="tfidf") for part in parts}
        for suffix, mode in modes.items():
            add("tfidf" + suffix, tfidf, grid=COARSE_GRID, threshold_policy="development F1, 0.05 grid",
                how=mode, labels_used=len(parts["dev"][2]))
        for model in exported["arms"]:
            tuned_on = "tune" if model["finetuned"] else "dev"
            policy = ("held-out development entities (not used in training) F1, 0.01 grid" if model["finetuned"]
                      else "development F1, 0.01 grid")
            for source in ("shared", "native"):
                scores, pools = {}, {}
                for part, (a, b, _, shared) in parts.items():
                    frame = cosine_scores(directory / f"{part}-{model['name']}-{source}.csv", a, b)
                    if source == "shared" and pair_set(frame) != pair_set(shared):
                        raise ValueError(f"{model['name']} did not score exactly the shared candidates")
                    scores[part], pools[part] = frame, frame[PAIR_COLUMNS]
                for suffix, mode in modes.items():
                    add(f"lt_{model['name']}_{source}{suffix}", scores, grid=COSINE_GRID, tuned_on=tuned_on,
                        threshold_policy=policy, candidates=pools, model=model, how=mode,
                        candidate_pool="jlink's cached n-gram candidates" if source == "shared"
                        else "LinkTransformer merge_knn, k=10, within partition",
                        labels_used=len(parts["dev"][2]) + (model.get("training") or {}).get("negatives", 0))
    else:
        arms = equal_links(directory, parts, exported, how, cached_run)
    report = {"schema_version": 1, "dataset": dataset, "how": how, "request": request, "arms": arms,
              "linktransformer": exported, "new_api_calls": 0, "new_api_cost_usd": 0,
              "protocol": ("entity-disjoint 30/70 development/test split, seed 1729; every arm uses the same "
                           "resolver and cardinality; final recall counts matches lost in retrieval"),
              "code_sha256": {name: file_hash(ROOT.parent / name) for name in (
                  "bench/lt_compare.py", "bench/linktransformer_scores.py", "bench/evaluation.py",
                  "src/jlink/resolve.py")}}
    write_json(directory / "report.json", report)
    return report


def equal_links(directory: Path, parts, exported, how, cached_run: Path) -> dict:
    """Positive-only labels: listed matches recovered at the number of links jlink made.

    Scored twice: all pairs over the cut (many-to-many, the dataset's rule) and each left record's
    best match only. Development counts choose each family's variant; test counts are reported, also
    without the left records of jlink's earlier rule-wording pilots, which did see these labels.
    """
    if how != "many-to-many":
        raise ValueError("the equal-link comparison assumes the dataset's many-to-many rule")
    pilots = set()
    for run in sorted(cached_run.parent.glob(cached_run.name + "-pilot*")):
        pilots |= set(pd.read_csv(run / "scores.csv", dtype={"left_id": str}).left_id)

    def best_per_left(frame, column):
        return frame.sort_values(["left_id", column], ascending=[True, False], kind="stable").drop_duplicates(
            "left_id")

    def count(frame, column, n, listed, top1):
        top = (best_per_left(frame, column) if top1 else frame).nlargest(n, column)
        return len(pair_set(top) & listed)

    arms = {}
    for top1, suffix in ((False, ""), (True, "_top1")):
        sizes, frames = {}, {}
        for part, (a, b, gold, shared) in parts.items():
            links = resolve(shared, how="many-to-one" if top1 else how, threshold=.5, min_margin=None)
            for label, keep in (("", None), ("_excluding_pilot", pilots)):
                kept = links if keep is None else links.loc[~links.left_id.isin(keep)]
                sizes[part + label] = (len(kept), len(pair_set(kept) & pair_set(gold)))
        arms["jlink_jev" + suffix] = {
            "links": sizes["test"][0], "listed_matches": sizes["test"][1],
            "dev_listed_matches": sizes["dev"][1], "excluding_pilot": dict(zip(
                ("links", "listed_matches"), sizes["test_excluding_pilot"])),
            "threshold_policy": "fixed 0.5; rule wording was revised on a labelled 300-record pilot"}
        sources = {"tfidf": lambda part: parts[part][3].assign(cosine=parts[part][3].sim)}
        for model in exported["arms"]:
            for source in ("shared", "native"):
                sources[f"lt_{model['name']}_{source}"] = (
                    lambda part, m=model, s=source: cosine_scores(
                        directory / f"{part}-{m['name']}-{s}.csv", *parts[part][:2]))
        for name, read in sources.items():
            result = {}
            for part, (a, b, gold, shared) in parts.items():
                frame, listed = read(part), pair_set(gold)
                result[part] = count(frame, "cosine", sizes[part][0], listed, top1)
                if part == "test":
                    rest = frame.loc[~frame.left_id.isin(pilots)]
                    outside = pair_set(gold.loc[~gold.left_id.isin(pilots)])
                    result["excluding_pilot"] = count(rest, "cosine", sizes["test_excluding_pilot"][0],
                                                      outside, top1)
            arms[name + suffix] = {"links": sizes["test"][0], "listed_matches": result["test"],
                                   "dev_listed_matches": result["dev"],
                                   "excluding_pilot": {"links": sizes["test_excluding_pilot"][0],
                                                       "listed_matches": result["excluding_pilot"]},
                                   "threshold_policy": "highest similarities, as many links as jlink made"}
    listed = len(parts["test"][2])
    for arm in arms.values():
        arm["listed_recall"] = arm["listed_matches"] / listed
        arm["dev_f1"] = arm["dev_listed_matches"]  # the development quantity families are chosen on
    arms["_note"] = (f"positive-only crosswalk: precision and F1 are undefined, so every arm predicts as many "
                     f"links as jlink and is scored on listed matches; {len(pilots)} left records were in "
                     "jlink's rule-wording pilots")
    return arms


FAMILIES = {"jlink, shipped rule, no labels": "jlink_jev|",
            "jlink, link rule chosen on development": "jlink_jev",
            "character TF-IDF": "tfidf",
            "LinkTransformer, pretrained": ("lt_minilm", "lt_mpnet", "lt_company"),
            "LinkTransformer, fine-tuned on development labels": "lt_ft_"}


def chosen(arms: dict, prefix) -> tuple[str, dict, list[float]] | None:
    """The family's best development variant. Ties prefer LinkTransformer's own retrieval, then the
    package-default configuration, then the name; the test scores of every tied variant come back too."""
    if isinstance(prefix, str) and prefix.endswith("|"):
        members = {k: v for k, v in arms.items() if k == prefix[:-1]}
    else:
        members = {k: v for k, v in arms.items() if k.startswith(prefix)}
    if not members:
        return None
    best = max(v["dev_f1"] for v in members.values())
    tied = {k: v for k, v in members.items() if abs(v["dev_f1"] - best) < 1e-9}
    name = min(tied, key=lambda k: ("_native" not in k, "_paper" in k or "onlinecontrastive" in k, k))
    return name, tied[name], sorted(v["test"]["f1"] if "test" in v else v["listed_matches"]
                                    for v in tied.values())


def cell(pick, complete: bool) -> str:
    """The chosen variant's test score; a range follows when variants tied on development."""
    if not pick:
        return ""
    name, arm, tied = pick
    if complete:
        text = f"{arm['test']['f1']:.4f}"
        return text + (f" (tied variants {tied[0]:.4f}-{tied[-1]:.4f})" if len(tied) > 1 else "")
    return f"{arm['listed_matches']:,}" + (f" (tied {tied[0]:,}-{tied[-1]:,})" if len(tied) > 1 else "")


def headline(out_dir: Path) -> list[str]:
    """Each family's development-chosen variant on test: F1, or listed matches for positive-only labels."""
    lines = ["| Dataset | " + " | ".join(FAMILIES) + " |", "|---|" + "---:|" * len(FAMILIES)]
    for dataset in NAMES:
        path = out_dir / dataset / "report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        arms = {k: v for k, v in report["arms"].items() if not k.startswith("_")}
        if report["request"]["complete_labels"]:
            lines.append(f"| {dataset} | " + " | ".join(
                cell(chosen(arms, prefix), True) for prefix in FAMILIES.values()) + " |")
            continue
        # Positive-only labels: counts are comparable only at one link rule and one number of links.
        for label, top1 in (("all pairs over the cut", False), ("best match per left record", True)):
            rule = {k.removesuffix("_top1"): v for k, v in arms.items() if k.endswith("_top1") == top1}
            links = rule["jlink_jev"]["links"]
            lines.append(f"| {dataset}: listed matches among {links:,} links, {label} | " + " | ".join(
                cell(chosen(rule, prefix), False) for prefix in FAMILIES.values()) + " |")
    return lines


def summary(out_dir: Path) -> str:
    """One row per method family: the variant with the best development score, never the best test score."""
    lines = headline(out_dir)
    for dataset in NAMES:
        path = out_dir / dataset / "report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        arms = {k: v for k, v in report["arms"].items() if not k.startswith("_")}
        counts, training = report["request"]["counts"]["test"], report["request"]["training_pairs"]
        lines += ["", f"### {dataset}", "", f"Test: {counts['left']:,} x {counts['right']:,} records, "
                  f"{counts['truth']:,} listed matches. LinkTransformer fine-tuning saw {training['positives']} "
                  f"matched development pairs (and {training['negatives']} nonmatches for its labelled-pair loss).", ""]
        if not report["request"]["complete_labels"]:
            lines += [report["arms"]["_note"] + ".", "",
                      "| Variant | Links predicted | Listed matches | Listed recall | Development listed matches | "
                      "Outside pilot records: links | Outside pilot records: listed matches |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            lines += [f"| `{k}` | {v['links']:,} | {v['listed_matches']:,} | {v['listed_recall']:.4f} | "
                      f"{v['dev_listed_matches']:,} | {v['excluding_pilot']['links']:,} | "
                      f"{v['excluding_pilot']['listed_matches']:,} |" for k, v in arms.items()]
            continue
        lines += ["| Method | Variant chosen on development | Labels used | Precision | Recall | F1 | "
                  "F1 with threshold picked on test | Test F1 of variants tied on development |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
        for family, prefix in FAMILIES.items():
            if not (pick := chosen(arms, prefix)):
                continue
            name, arm, tied = pick
            test, oracle = arm["test"], arm.get("test_oracle_threshold")
            labels = len(report["arms"]) and arm["labels_used"]
            if family.endswith("chosen on development"):
                labels = report["request"]["counts"]["dev"]["truth"]
            lines.append(f"| {family} | `{name}`, {arm['how']}, threshold {arm['threshold']} | {labels} | "
                         f"{test['precision']:.3f} | {test['recall']:.3f} | **{test['f1']:.4f}** | "
                         + (f"{oracle['f1']:.4f}" if oracle else "n/a (fixed)") + " | "
                         + (f"{len(tied)} tied: {tied[0]:.4f}-{tied[-1]:.4f}" if len(tied) > 1 else "") + " |")
        lines += ["", "<details><summary>Every variant</summary>", "",
                  "| Variant | Link rule | Threshold | Dev F1 | Test P | Test R | Test F1 | Oracle F1 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for name, arm in arms.items():
            test, oracle = arm["test"], arm.get("test_oracle_threshold")
            lines.append(f"| `{name}` | {arm['how']} | {arm['threshold']} | {arm['dev_f1']:.4f} | "
                         f"{test['precision']:.3f} | {test['recall']:.3f} | {test['f1']:.4f} | "
                         + (f"{oracle['f1']:.4f}" if oracle else "") + " |")
        lines += ["", "</details>"]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "summary"))
    parser.add_argument("dataset", choices=NAMES, nargs="?")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--cached-run", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "lt-compare")
    args = parser.parse_args(argv)
    if args.command == "summary":
        print(summary(args.out))
        return 0
    if args.dataset is None:
        parser.error("prepare and evaluate need a dataset")
    cached = args.cached_run or ROOT / "out" / "live" / args.dataset
    try:
        run = prepare if args.command == "prepare" else evaluate
        result = run(args.dataset, data_dir=args.data_dir, cached_run=cached, out_dir=args.out)
    except (ValueError, OSError) as exc:
        print(f"bench lt_compare: {exc}", file=sys.stderr)
        return 2
    if args.command == "prepare":
        print(json.dumps(result))
    else:
        for name, arm in result["arms"].items():
            if name.startswith("_"):
                continue
            row = arm.get("test", arm)
            print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                    if k not in {"resolver_warnings", "model"}}),
                  "oracle_f1=%s" % round(arm["test_oracle_threshold"]["f1"], 4)
                  if "test_oracle_threshold" in arm else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

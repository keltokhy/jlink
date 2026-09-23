"""LinkTransformer on the 300-record samples of the local-model comparison, beside Jev and the local models.

    uv run --group bench python bench/lt_sample.py [--datasets nber-firms ...] \\
        [--out docs/benchmarks/local-models-2026-09-22.json]

For each benchmark this takes the left records live.py --sample 300 drew (random_state=0) and runs
LinkTransformer's own retrieval (merge_knn, k=10) for them against the full right table, in
LinkTransformer's own environment (bench/out/lt-env; it is GPL-3.0, so jlink never imports it), for
two arms of the 2026-09-20 comparison in bench/out/lt-compare/: a pretrained model (the company model
on firms; elsewhere MiniLM or MPNet, whichever had the higher development F1 under the dataset's link
rule) and ft_supcon, fine-tuned on that comparison's development labels. Back here each arm's cosines
are resolved with jlink.resolve under the dataset's own link rule, at the threshold that comparison
tuned on development labels, and scored as bench/local_models.py scores Jev, DiffusionGemma and Laya.

The fine-tuned model was trained on development records, and some sampled records are among them, so
every method is also scored on the sampled records in that comparison's test split, from the saved
scores at the same thresholds. Writes raw outputs to bench/out/local-2026-09-22/lt/<dataset>/ and adds
a "linktransformer" block to the frozen JSON, leaving the rest of it unchanged. bench/local_models.py
rewrites that file without the block, so run this after it.

`lt_sample.py retrieve <dir> <arm>` is the part that runs inside LinkTransformer's environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LT_PYTHON = HERE / "out" / "lt-env" / "bin" / "python"
COMPARED = HERE / "out" / "lt-compare"
OUT = HERE / "out" / "local-2026-09-22" / "lt"
DATASETS = ("nber-firms", "abt-buy", "dblp-acm", "amazon-google", "febrl4")
LOCAL = {"diffusiongemma": "diffusiongemma-s300", "laya": "laya-s300"}
KEY = ["left_id", "right_id"]
# torch, faiss and scikit-learn each ship their own OpenMP runtime in LinkTransformer's environment;
# with more than one OpenMP thread, MPNet-based models crash (SIGSEGV) on their first batch, on MPS or CPU.
THREADS = {"OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}

try:  # absent from LinkTransformer's environment, which only runs retrieve()
    import jlink
    from live import RULES, load
    sys.path.insert(0, str(ROOT))
    from bench.evaluation import choose_threshold
    from bench.lt_compare import COSINE_GRID, cosine_scores, restrict
except ImportError:
    jlink = None


def retrieve(directory: Path, name: str) -> None:
    """In LinkTransformer's environment: load one arm's model, then time merge_knn (encode and search) alone."""
    import functools
    import importlib.metadata

    import linktransformer as lt
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from linktransformer_scores import records

    request = json.loads((directory / "request.json").read_text())
    arm = next(a for a in request["arms"] if a["name"] == name)
    fields, k = request["on"], request["k"]
    left, right = (records(directory / f"{side}.parquet", fields) for side in ("left", "right"))
    start, downloaded = time.perf_counter(), False
    if arm.get("path"):
        path = ROOT / arm["path"]
    else:
        fetch = functools.partial(snapshot_download, arm["model"], revision=arm["revision"], allow_patterns=[
            "*.json", "*.txt", "*.safetensors", "pytorch_model.bin", "*.model"],
            ignore_patterns=["onnx/*", "openvino/*"])
        try:  # the pinned revision is normally cached from the 2026-09-20 comparison
            path = Path(fetch(local_files_only=True))
        except LocalEntryNotFoundError:
            path, downloaded = Path(fetch()), True
    model = lt.LinkTransformer(str(path))
    loaded = time.perf_counter()
    merged = lt.merge_knn(left, right, left_on=fields, right_on=fields, model=model, k=min(k, len(right)))
    done = time.perf_counter()
    merged[["id_x", "id_y", "score"]].rename(
        columns={"id_x": "left_id", "id_y": "right_id", "score": "cosine"}).drop_duplicates(KEY).to_csv(
        directory / f"{name}-native.csv", index=False)
    (directory / f"{name}-retrieval.json").write_text(json.dumps({
        "arm": name, "model_path": str(path), "downloaded": downloaded, "device": str(model.device),
        "load_seconds": round(loaded - start, 2), "encode_retrieve_seconds": round(done - loaded, 2),
        "left": len(left), "right": len(right), "k": k,
        "packages": {p: importlib.metadata.version(p) for p in (
            "linktransformer", "sentence-transformers", "torch", "transformers", "faiss-cpu", "pandas")},
    }, indent=2) + "\n")


def rounded(result: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}


def scored(scores: pd.DataFrame, truth: pd.DataFrame, how: str, threshold: float) -> tuple[dict, pd.DataFrame]:
    """local_models.metrics' at_0.5, at any threshold: jlink.resolve, then jlink.score_against_truth."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        links = jlink.resolve(scores, how=how, threshold=threshold)
    return rounded(jlink.score_against_truth(links, truth, scores[KEY])) | {"links": len(links)}, links


def arms_of(dataset: str, how: str) -> dict:
    """The two arms of the 2026-09-20 comparison, with the development-tuned threshold for this link rule."""
    report = json.loads((COMPARED / dataset / "report.json").read_text())
    exported = {a["name"]: a for a in json.loads((COMPARED / dataset / "linktransformer.json").read_text())["arms"]}
    if dataset == "nber-firms":
        pretrained = "company"
    else:
        # Every dataset but firms has the variant under its own link rule, unsuffixed, in report.json.
        pretrained = max(("minilm", "mpnet"), key=lambda n: report["arms"][f"lt_{n}_native"]["dev_f1"])
    arms = {}
    for role, name in (("zero_shot", pretrained), ("fine_tuned", "ft_supcon")):
        model = exported[name]
        tuned, dev_f1 = tune(dataset, name, finetuned=model["finetuned"], how=how)
        recorded = report["arms"].get(f"lt_{name}_native", {})
        if "threshold" in recorded:
            if recorded["how"] != how or recorded["threshold"] != tuned:
                raise SystemExit(f"{dataset} {name}: retuning gave {tuned} under {how}, report.json has "
                                 f"{recorded['threshold']} under {recorded['how']}")
            dev_f1 = recorded["dev_f1"]
            source = (f"bench/out/lt-compare/{dataset}/report.json, arms.lt_{name}_native.threshold "
                      f"({recorded['threshold_policy']}; retuned here and equal)")
        else:
            source = ("tuned here as bench/lt_compare.py tunes the other datasets ("
                      + ("held-out development entities, not used in training" if model["finetuned"]
                         else "development") + f" F1 under {how}, 0.01 grid) on bench/out/lt-compare/{dataset}"
                      "/dev-*; report.json has no threshold for firms, whose comparison held the number of "
                      "links equal instead, and scoring treats the listed matches as complete, as the jlink "
                      "columns do")
        base = exported.get(next((n for n, a in exported.items() if not a["finetuned"]
                                  and a["model"] == (model.get("training") or {}).get("base")), None), {})
        arms[role] = {
            "arm": name, "model": model["model"], "revision": model.get("revision"),
            "finetuned": model["finetuned"], "path": model["model"] if model["finetuned"] else None,
            "weights_sha256": sha256(ROOT / model["model"] / "model.safetensors") if model["finetuned"] else None,
            "base": ({"model": model["training"]["base"], "pretrained_arm_revision": base.get("revision"),
                      "loss": model["training"]["loss"], "epochs": model["training"]["epochs"],
                      "training_pairs": model["training"]["rows"]} if model["finetuned"] else None),
            "how": how, "threshold": tuned, "threshold_source": source, "dev_f1": round(dev_f1, 4),
            "dev_entities": "tune" if model["finetuned"] else "dev",
        }
    return arms


def tune(dataset: str, name: str, *, finetuned: bool, how: str) -> tuple[float, float]:
    """bench/lt_compare.py's threshold choice, from its saved development cosines and labels."""
    directory = COMPARED / dataset
    split = pd.read_csv(directory / "split.csv", dtype={"id": str})
    left, right = (pd.read_parquet(directory / f"dev-{side}.parquet").astype({"id": str}) for side in ("left", "right"))
    scores = cosine_scores(directory / f"dev-{name}-native.csv", left, right)
    gold = pd.read_csv(directory / "dev-truth.csv", dtype=str)
    if finetuned:
        scores, gold = restrict(scores, split, "tune"), restrict(gold, split, "tune")
    threshold, trials = choose_threshold(scores, gold, how=how, grid=COSINE_GRID)
    return threshold, next(t["f1"] for t in trials if t["threshold"] == threshold)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(dataset: str) -> dict:
    how = RULES[dataset]["how"]
    left, right, truth, meta = load(dataset, 300)
    request = json.loads((COMPARED / dataset / "request.json").read_text())
    fields = request["on"]
    if fields != meta["on"]:
        raise SystemExit(f"{dataset}: the 2026-09-20 comparison compared {fields}, live.py {meta['on']}")
    split = pd.read_csv(COMPARED / dataset / "split.csv", dtype={"id": str})
    test_left = set(split.loc[split.side.eq("left") & split.split.eq("test"), "id"])
    sampled = set(left["id"])
    tested = sampled & test_left
    bases = {"sample": (sampled, truth), "test_split": (tested, truth[truth["left_id"].isin(tested)])}

    directory = OUT / dataset
    directory.mkdir(parents=True, exist_ok=True)
    left[["id", *fields]].to_parquet(directory / "left.parquet", index=False)
    right[["id", *fields]].to_parquet(directory / "right.parquet", index=False)
    truth.to_csv(directory / "truth.csv", index=False)
    arms = arms_of(dataset, how)
    (directory / "request.json").write_text(json.dumps(
        {"dataset": dataset, "on": fields, "k": request["k"], "how": how,
         "arms": [{"name": a["arm"], "model": a["model"], "revision": a["revision"], "path": a["path"]}
                  for a in arms.values()]}, indent=2) + "\n")

    rel = lambda p: str(p.relative_to(ROOT))  # noqa: E731
    judged = {}
    for role, arm in arms.items():
        name = arm["arm"]
        started = time.perf_counter()
        with open(directory / f"{name}-retrieval.log", "w") as log:
            subprocess.run([str(LT_PYTHON), str(HERE / "lt_sample.py"), "retrieve", str(directory), name],
                           check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=os.environ | THREADS)
        process = time.perf_counter() - started
        retrieval = json.loads((directory / f"{name}-retrieval.json").read_text())
        cosines = pd.read_csv(directory / f"{name}-native.csv", dtype={"left_id": str, "right_id": str})
        if not (set(cosines["left_id"]) <= sampled and cosines["right_id"].isin(right["id"]).all()):
            raise SystemExit(f"{dataset} {name}: retrieval returned ids outside the sample or the right table")
        cosine = cosines["cosine"].clip(0, 1)  # as bench/lt_compare.py: cosines are not probabilities
        scores = judged[role] = cosines.assign(p=cosine, sim=cosine, block="linktransformer:merge_knn",
                                               source=f"lt_{name}")
        started = time.perf_counter()
        results = {}
        for basis, (ids, gold) in bases.items():
            results[basis], links = scored(scores[scores["left_id"].isin(ids)], gold, how, arm["threshold"])
            links[KEY + ["p"]].rename(columns={"p": "cosine"}).to_csv(directory / f"{name}-links-{basis}.csv",
                                                                       index=False)
        resolve_seconds = time.perf_counter() - started
        # The same model scored these pairs on 2026-09-20 wherever both records are in the test split.
        before = pd.read_csv(COMPARED / dataset / f"test-{name}-native.csv", dtype={"left_id": str, "right_id": str})
        both = cosines.merge(before, on=KEY, suffixes=("", "_before"))
        arm |= {
            "pairs": len(cosines), **results,
            "contaminated_on_sample": arm["finetuned"],
            "seconds": {"load": retrieval["load_seconds"], "encode_retrieve": retrieval["encode_retrieve_seconds"],
                        "resolve_and_score": round(resolve_seconds, 2), "process": round(process, 1)},
            "device": retrieval["device"], "downloaded": retrieval["downloaded"], "packages": retrieval["packages"],
            "check_against_2026_09_20": {
                "pairs_in_both": len(both),
                "max_abs_cosine_difference": float(f"{(both.cosine - both.cosine_before).abs().max():.2g}")},
            "raw": {"cosines": rel(directory / f"{name}-native.csv"),
                    "links": {b: rel(directory / f"{name}-links-{b}.csv") for b in bases},
                    "retrieval": rel(directory / f"{name}-retrieval.json"),
                    "log": rel(directory / f"{name}-retrieval.log")},
        }
        if arm["finetuned"]:
            arm["contamination"] = (f"{len(sampled - tested)} of the {len(sampled)} sampled left records are in the "
                                    "development split this model was trained and its threshold tuned on; only "
                                    "test_split is a fair basis for it")

    # Jev, DiffusionGemma and Laya on the test-split records, from their saved scores at 0.5.
    frozen = {"jev": HERE / "out" / "live" / dataset,
              **{m: HERE / "out" / "live" / f"{dataset}-{tag}" for m, tag in LOCAL.items()}}
    others, check = {}, {}
    for method, path in frozen.items():
        saved = jlink.load(path).scores
        saved = judged[method] = saved[saved["left_id"].isin(sampled)].reset_index(drop=True)
        check[method], _ = scored(saved, truth, how, 0.5)
        others[method], _ = scored(saved[saved["left_id"].isin(tested)], bases["test_split"][1], how, 0.5)
        others[method]["scores"] = rel(path / "scores.csv")
    result = {
        "dataset": dataset, "how": how, "on": fields, "k": request["k"],
        "overlap": {"sampled_left": len(sampled), "in_development_split": len(sampled - tested),
                    "in_test_split": len(tested), "known_links_sample": len(truth),
                    "known_links_test_split": len(bases["test_split"][1]),
                    "split": rel(COMPARED / dataset / "split.csv")},
        "arms": arms, "others_test_split": others, "others_sample_recomputed": check,
    }
    if how == "many-to-many":
        # That rule keeps every candidate over the threshold, and LinkTransformer proposes ten per record.
        report = json.loads((COMPARED / dataset / "report.json").read_text())["arms"]
        thresholds = {role: report[f"lt_{arm['arm']}_native_1to1"]["threshold"] for role, arm in arms.items()}
        result["one_to_one"] = {"note": (
            "Supplementary: every method again under one-to-one assignment instead of the dataset's many-to-many "
            "rule. LinkTransformer at the thresholds report.json tuned for that rule on development labels "
            "(arms.lt_<arm>_native_1to1.threshold), the others at 0.5; the headline columns use many-to-many.")}
        for method, scores in judged.items():
            threshold = thresholds.get(method, 0.5)
            result["one_to_one"][method] = {"threshold": threshold, **{
                basis: scored(scores[scores["left_id"].isin(ids)], gold, "one-to-one", threshold)[0]
                for basis, (ids, gold) in bases.items()}}
    (directory / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def verify(result: dict, frozen: dict) -> None:
    """The recomputed sample scores of Jev and the local models must be the frozen ones."""
    for r in (r for r in frozen["runs"] if r["dataset"] == result["dataset"] and r["tag"] in LOCAL.values()):
        for method, recorded in (("jev", r["jev"]["at_0.5"]), (r["model"], r["local"]["at_0.5"])):
            mine = {k: v for k, v in result["others_sample_recomputed"][method].items() if k != "links"}
            if mine != recorded:
                raise SystemExit(f"{result['dataset']} {method}: recomputed {mine}, frozen {recorded}")


def line(r: dict) -> str:
    z, f, o = r["arms"]["zero_shot"], r["arms"]["fine_tuned"], r["others_test_split"]
    return (f"{r['dataset']:<14} zero-shot {z['arm']} t={z['threshold']} F1 {z['sample']['f1']:.3f} / "
            f"{z['test_split']['f1']:.3f}  ft_supcon t={f['threshold']} F1 {f['sample']['f1']:.3f} / "
            f"{f['test_split']['f1']:.3f}  test split: jev {o['jev']['f1']:.3f} dg {o['diffusiongemma']['f1']:.3f} "
            f"laya {o['laya']['f1']:.3f}  {z['seconds']['encode_retrieve']:.1f}s + "
            f"{f['seconds']['encode_retrieve']:.1f}s")


def main() -> None:
    if sys.argv[1:2] == ["retrieve"]:
        retrieve(Path(sys.argv[2]), sys.argv[3])
        return
    if jlink is None:
        raise SystemExit("run with uv run --group bench python bench/lt_sample.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "benchmarks" / "local-models-2026-09-22.json")
    a = ap.parse_args()
    frozen = json.loads(a.out.read_text())
    results = {}
    for dataset in a.datasets:
        results[dataset] = run(dataset)
        verify(results[dataset], frozen)
        print(line(results[dataset]), flush=True)

    def total(role: str) -> float:
        return round(sum(r["arms"][role]["seconds"]["encode_retrieve"] for r in results.values()), 1)

    frozen["linktransformer"] = {
        "date": time.strftime("%Y-%m-%d"), "script": "bench/lt_sample.py",
        "machine": "Apple M3 Ultra, 96 GiB unified memory; one arm at a time, nothing else benchmarking",
        "environment": ("bench/out/lt-env: LinkTransformer's own environment, run as a subprocess; jlink never "
                        "imports it (GPL-3.0). Run with OMP_NUM_THREADS=1: that environment loads three OpenMP "
                        "runtimes (torch, faiss, scikit-learn), and with more threads MPNet-based models crashed "
                        "(SIGSEGV) on their first batch, on MPS and on CPU. The models ran on the GPU (MPS), so "
                        "the setting mainly limits the faiss search."),
        "comparison": ("bench/out/lt-compare/<dataset>/: the 2026-09-20 comparison whose development split trained "
                       "the fine-tuned models and chose every threshold (report.json, linktransformer.json, "
                       "split.csv)"),
        "method": ("The 300 left records live.py --sample 300 drew (random_state=0) against the full right table. "
                   "LinkTransformer's own retrieval, merge_knn with k=10, proposes each arm's candidate pairs, "
                   "scored by cosine similarity (clipped to 0..1, as bench/lt_compare.py does). jlink.resolve links "
                   "them under the link rule the jlink columns use, at the arm's development-tuned threshold, and "
                   "jlink.score_against_truth scores the links against the sample's known matches: the code "
                   "bench/local_models.py uses for Jev, DiffusionGemma and Laya at 0.5. Seconds: load is reading "
                   "the model, encode_retrieve is merge_knn alone (encoding both tables, indexing, searching), "
                   "each arm in a fresh process with no warm-up; process includes starting Python. On the two "
                   "product sets, datasets.<dataset>.one_to_one repeats every method under one-to-one assignment."),
        "bases": {"sample": "all 300 sampled left records and their known matches",
                  "test_split": ("the sampled left records in the test split of the 2026-09-20 comparison, and "
                                 "their known matches; the right table stays whole. Jev, DiffusionGemma and Laya "
                                 "are rescored from their saved scores.csv on these records at 0.5")},
        "roles": {"zero_shot": ("pretrained, not fine-tuned; the company model on firms, elsewhere MiniLM or MPNet, "
                                "whichever had the higher development F1 in report.json under the dataset's link "
                                "rule; its threshold used the development labels"),
                  "fine_tuned": ("ft_supcon: the dataset's pretrained base fine-tuned by LinkTransformer's "
                                 "train_model on development matches (supervised contrastive loss, the package "
                                 "default, 10 epochs)")},
        "encode_retrieve_seconds_total": {"datasets": list(results), "zero_shot": total("zero_shot"),
                                          "fine_tuned": total("fine_tuned")},
        "api_cost_usd": 0, "model_api_calls": 0,
        "datasets": {d: {k: v for k, v in r.items() if k != "others_sample_recomputed"}
                     for d, r in results.items()},
        "check": ("For every dataset, Jev's, DiffusionGemma's and Laya's F1, precision, recall and counts on the "
                  "300 records, recomputed here with the same code, equal runs[].jev.at_0.5 and runs[].local.at_0.5."),
    }
    before = json.loads(a.out.read_text())
    text = json.dumps(frozen, indent=2, default=str)
    if {k: v for k, v in json.loads(text).items() if k != "linktransformer"} != {
            k: v for k, v in before.items() if k != "linktransformer"}:
        raise SystemExit("refusing to write: an existing entry of the frozen JSON would change")
    a.out.write_text(text)
    print(f"wrote the linktransformer block of {a.out}")


if __name__ == "__main__":
    main()

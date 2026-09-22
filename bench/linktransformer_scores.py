"""Isolated LinkTransformer arms for bench/lt_compare.py: its scoring, retrieval and training APIs.

Run with the interpreter of a separate environment that has linktransformer and pyarrow. Pretrained
arms are `name=hub-model[@revision]`; `--finetune base` trains on the prepared development pairs with
LinkTransformer's own train_model, once per loss. No paid API calls and no jlink imports.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import pandas as pd

LOSSES = {"supcon": "matched pairs only (the package default)",
          "onlinecontrastive": "matched pairs and hard nonmatches (its labelled-pair option)"}


def records(path: Path, fields: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame.assign(**{c: frame[c].map(lambda v: "" if pd.isna(v) else str(v)) for c in fields})


def export(lt, model, name: str, directory: Path, fields: list[str], k: int) -> dict:
    """Cosines for the shared candidates via evaluate_pairs, then LinkTransformer's own merge_knn."""
    seconds = {}
    for part in ("dev", "test"):
        left, right = (records(directory / f"{part}-{side}.parquet", fields) for side in ("left", "right"))
        pairs = pd.read_csv(directory / f"{part}-candidates.csv", dtype={"left_id": str, "right_id": str})
        table = pairs[["left_id", "right_id"]]
        for prefix, frame, key in (("l_", left, "left_id"), ("r_", right, "right_id")):
            table = table.merge(frame.rename(columns={"id": key, **{c: prefix + c for c in fields}}),
                                on=key, how="left", validate="many_to_one")
        start = time.perf_counter()
        scored = lt.evaluate_pairs(table, model=model, left_on=["l_" + c for c in fields],
                                   right_on=["r_" + c for c in fields])
        scored[["left_id", "right_id"]].assign(cosine=scored.score.astype(float)).to_csv(
            directory / f"{part}-{name}-shared.csv", index=False)
        merged = lt.merge_knn(left, right, left_on=fields, right_on=fields, model=model,
                              k=min(k, len(right)))
        merged[["id_x", "id_y", "score"]].rename(
            columns={"id_x": "left_id", "id_y": "right_id", "score": "cosine"}).drop_duplicates(
            ["left_id", "right_id"]).to_csv(directory / f"{part}-{name}-native.csv", index=False)
        seconds[part] = round(time.perf_counter() - start, 1)
    return seconds


def finetune(lt, directory: Path, base: str, loss: str, fields: list[str], epochs: int,
             name: str) -> tuple[str, dict]:
    # Everything as text: inferred dtypes would turn a price of "499" into "499.0" and drop a
    # postcode's leading zero, so training would see strings that inference never does.
    train = pd.read_csv(directory / "train-pairs.csv", dtype=str, keep_default_na=False)
    left, right = ["l_" + c for c in fields], ["r_" + c for c in fields]
    train = train.assign(label=train.label.astype(int))
    labelled = loss == "onlinecontrastive"
    if labelled and not train.label.eq(0).any():
        raise ValueError("labelled-pair training needs nonmatches; this dataset has positive-only labels")
    data = train if labelled else train.loc[train.label.eq(1)].drop(columns="label")
    start = time.perf_counter()
    best = lt.train_model(data=data.reset_index(drop=True), model_path=base, left_col_names=left,
                          right_col_names=right, left_id_name=["l_id"], right_id_name=["r_id"],
                          label_col_name="label" if labelled else None, log_wandb=False,
                          training_args={"num_epochs": epochs, "loss_type": loss,
                                         "model_save_dir": str(directory / "models"),
                                         "model_save_name": name})
    return best, {"base": base, "loss": loss, "loss_data": LOSSES[loss], "epochs": epochs,
                  "rows": len(data), "positives": int(train.label.eq(1).sum()),
                  "negatives": int(train.label.eq(0).sum()) if labelled else 0,
                  "other_settings": ("LinkTransformer linkage.json defaults: batch 64, lr 2e-5, and its own "
                                     "80/10/10 train/validation/test division of these pairs"),
                  "seconds": round(time.perf_counter() - start, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pretrained", nargs="*", default=[], metavar="NAME=MODEL[@REVISION]")
    parser.add_argument("--finetune", metavar="BASE_MODEL")
    parser.add_argument("--losses", nargs="*", default=list(LOSSES), choices=list(LOSSES))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--tag", default="", help="suffix naming a second fine-tuning configuration")
    parser.add_argument("--redo", action="store_true", help="retrain and replace arms already recorded")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, metavar="N",
                        help="train and embed on CPU with N torch threads (PyTorch's MPS backend asserts "
                             "on the empty selections online contrastive loss can produce)")
    args = parser.parse_args()

    if args.cpu_threads:
        import torch
        torch.backends.mps.is_available = lambda: False
        torch.set_num_threads(args.cpu_threads)
        args.device = "cpu"
    import linktransformer as lt
    from huggingface_hub import snapshot_download

    request = json.loads((args.directory / "request.json").read_text())
    fields, k = request["on"], request["k"]
    report = args.directory / "linktransformer.json"
    result = json.loads(report.read_text()) if report.exists() else {"arms": []}
    done = set() if args.redo else {arm["name"] for arm in result["arms"]}

    def record(arm):
        # Reread first: a CPU and a GPU run may be adding arms to the same directory.
        result = json.loads(report.read_text()) if report.exists() else {"arms": []}
        result["arms"] = [a for a in result["arms"] if a["name"] != arm["name"]] + [arm]
        result["packages"] = {p: importlib.metadata.version(p) for p in (
            "linktransformer", "sentence-transformers", "torch", "transformers", "faiss-cpu", "pandas")}
        report.write_text(json.dumps(result, indent=2) + "\n")

    for spec in args.pretrained:
        name, _, model = spec.partition("=")
        model, _, revision = model.partition("@")
        if name in done:
            continue
        path = snapshot_download(model, revision=revision or None, allow_patterns=[
            "*.json", "*.txt", "*.safetensors", "pytorch_model.bin", "*.model"],
            ignore_patterns=["onnx/*", "openvino/*"])
        seconds = export(lt, lt.LinkTransformer(path, device=args.device), name, args.directory, fields, k)
        record({"name": name, "model": model, "revision": Path(path).name, "finetuned": False,
                "seconds": seconds})
    if args.finetune:
        for loss in args.losses:
            name = f"ft_{loss}{args.tag}"
            if name in done or (loss == "onlinecontrastive" and not request["complete_labels"]):
                continue
            best, training = finetune(lt, args.directory, args.finetune, loss, fields, args.epochs, name)
            seconds = export(lt, lt.LinkTransformer(best, device=args.device), name, args.directory, fields, k)
            record({"name": name, "model": best, "finetuned": True, "training": training, "seconds": seconds})


if __name__ == "__main__":
    main()

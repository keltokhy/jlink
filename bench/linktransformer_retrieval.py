"""Isolated adapter for the real LinkTransformer merge_knn implementation; no paid API calls."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd


def main():
    import linktransformer as lt
    from huggingface_hub import snapshot_download

    directory = Path(sys.argv[1])
    request = json.loads((directory / "request.json").read_text())
    left, right = (pd.read_parquet(directory / f"{side}.parquet") for side in ("left", "right"))
    fields = [(c, c) if isinstance(c, str) else tuple(c) for c in request["on"]]
    path = snapshot_download(request["model"], revision=request["revision"],
                             allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin",
                                             "*.model"], ignore_patterns=["onnx/*", "openvino/*"])
    start = time.perf_counter()
    model = lt.LinkTransformer(path, device="cpu")
    result = lt.merge_knn(left, right, left_on=[c[0] for c in fields], right_on=[c[1] for c in fields],
                          model=model, k=min(request["k"], len(right)))
    result[["id_x", "id_y", "score"]].rename(columns={"id_x": "left_id", "id_y": "right_id",
                                                       "score": "retrieval_cosine"}).to_csv(
        directory / "candidates.csv", index=False)
    import linktransformer.infer as infer
    metadata = {"api": "linktransformer.merge_knn", "seconds_including_model_load": time.perf_counter() - start,
                "version": importlib.metadata.version("linktransformer"), "model": request["model"],
                "requested_revision": request["revision"], "resolved_snapshot": Path(path).name,
                "infer_sha256": hashlib.sha256(Path(infer.__file__).read_bytes()).hexdigest(),
                "score_semantics": "embedding cosine, not match probability", "new_api_calls": 0,
                "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
                "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM"),
                "packages": {p: importlib.metadata.version(p) for p in
                             ("sentence-transformers", "torch", "transformers", "numpy", "pandas", "faiss-cpu")}}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

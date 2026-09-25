# Hybrid retrieval and comparison protocol

`block.embeddings` adds local semantic neighbors to jlink's existing candidate pipeline.
It can be unioned with `ngrams`, `initials` and other passes, used with `reverse=True`, or
wrapped in `block.within`. No change to the default lexical retrieval is made without evidence.

## Install and use

```sh
uv sync --extra embeddings --group bench
jev-link estimate left.csv right.csv --on name \
  --block ngrams:name:10 --block embeddings:name:10 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

The same flags work with `jev-link link`. Embeddings run locally on CPU by default; set
`--embedding-device` or Python `device=` explicitly to use another device. The first call
downloads public model weights, unless already cached. Custom encoders can implement
`encode(texts, **kwargs)` and return a finite NumPy matrix; they do not need the optional
dependency. Their provenance explicitly says they are not reconstructable from configuration.

Single-column records retain their text. Multiple columns become JSON with the left field
names as shared labels, so field boundaries and surrounding context are preserved. Empty
records and zero embeddings never propose pairs, even at a permissive cosine threshold.
Each distinct serialized record is encoded once per invocation. Vectors are normalized and
searched using bounded blocks of exact cosine products; ties prefer original neighbor row
position. This bounds temporary search memory, not time: global search still does O(n*m*d)
work. It is not an approximate nearest-neighbor index or a million-record scaling claim.

Candidate `sim` continues to mean character TF-IDF similarity. It is not overwritten with
embedding cosine or misrepresented as a probability. Existing budget prioritization still
uses this lexical `sim`; low-lexical-similarity semantic candidates may be left unjudged under
tight budgets. Per-pass diagnostics retain model/revision, serialization and search settings.
Pin a full Hub commit for reproducibility. Floating-point behavior can still vary by runtime
and device; a requested floating tag or an unpinned local directory is not immutable provenance.

## Exact-text migration

`exact_shortcut=False` is now the default in Python and CLI. Equal names no longer silently
receive `p=1`. Opt in with `exact_shortcut=True` or `--exact-shortcut` only after establishing
that all compared fields jointly identify an entity under the requested definition. The old
fieldwise nonmissing normalization rules still apply when enabled. Old saved runs load and
relink without changing their historical judgments. With `budget=0`, uncached pairs remain
`unjudged` with missing `p`, including identical names; they are not evidence of nonmatches.

## Reproducible retrieval experiment

```sh
uv run --extra embeddings --group bench python bench/hybrid.py nber-firms \
  --out bench/out/hybrid-firms \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

The runner freezes observed-entity development/test splits, reretrieves inside each partition,
and compares lexical, semantic and their union. Reports include candidate recall, newly
recovered known matches, lost matches, pair counts, runtimes, source/code/artifact hashes,
model settings and package versions. These public datasets have been inspected previously;
a deterministic split does not make them untouched prospective evidence. Incomplete NBER
crosswalk labels support listed-positive recall, not precision or F1.

To run the actual LinkTransformer retrieval API, install its pinned checkout in a separate
environment with pyarrow and pass that interpreter as `--lt-python /path/to/env/bin/python`.
The adapter calls `linktransformer.merge_knn` with the same embedding checkpoint and `k`.
It keeps LinkTransformer's native serialization and search, records its source hash and
dependency versions, and preserves its raw candidate export. Each library uses its own
dependencies. This is retrieval comparison, not a test of LinkTransformer's LLM judge.
The child uses one OpenMP thread and disables tokenizer parallelism for native-runtime
compatibility; its timing is not an equal-thread speed comparison with jlink.

Add `--judge` for cache-only Jev scoring or `--live --budget 0.50` for an explicitly paid run.
The positive budget covers both partitions, each request setting its estimated price aside
before it goes out. Each partition's union is scored once, then identical scores are projected
back to every candidate pool. All use the same rule, cardinality, fixed threshold 0.5 and
disabled exact shortcut. This isolates retrieval effects on final links. Unjudged pairs stay
missing and are counted; partial coverage must not be presented as a complete judge comparison.

The first milestone does not establish superiority over LinkTransformer or the historical
entity-linking paper. Remaining experiments include native LinkTransformer LLM adjudication,
rule-sensitive accuracy on independently labeled cases, calibration on development labels,
precision/coverage/cost curves, and mention clustering for historical-text comparisons.

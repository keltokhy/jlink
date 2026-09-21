# Benchmarks

For entity-disjoint development/test comparisons, use `bench/heldout.py`.
The commands below retain the original in-sample oracle comparison.

Prepare public sources, run offline comparisons, and regenerate the measured report:

```sh
uv sync --group bench
uv run python bench/prepare.py
for name in febrl4 dblp-acm abt-buy amazon-google nber-firms; do
  uv run python bench/run.py --dataset "$name" --budget 0.10
done
uv run python bench/report.py
uv run pytest
```

Preparation accepts any subset of those names, for example
`uv run python bench/prepare.py nber-firms abt-buy`. It continues after a source failure
and returns a nonzero exit status if anything failed. It never substitutes fabricated data.
The commands also work as `uv run python -m bench.prepare` and `-m bench.run` from the repository root.

Each `bench/data/NAME/` contains `left.parquet`, `right.parquet`, `truth.parquet`, and
`meta.json`. Source frames use a unique `id` column; truth uses `left_id` and `right_id`.
IDs from CSV are strings, preserving leading zeros. Metadata supplies the entity, a plain-language
match definition, fields, cardinality, source links, terms, dates, checksums, counts and validation
diagnostics. Source ZIPs and download receipts are cached under `bench/data/_downloads/`.
Content changes fail checksum verification rather than silently changing a benchmark. FEBRL4
comes from the installed, locked `recordlinkage` package; its bundled CSVs are also hash-checked.
The NBER legacy ZIP's malformed extra metadata is handled using only the Python standard library,
with source SHA-256 and member size/CRC checks.

Generated data and JSON reports are git-ignored. Downloaded sources have their own terms;
the package's license does not relicense them. Leipzig explicitly provides CC-BY-4.0; the NBER
page provides research downloads without an explicit redistribution license. See
[FIRM_DATA.md](FIRM_DATA.md) for the firm-source investigation and [BASELINES.md](BASELINES.md)
for actual measured results and limitations.

The legacy baseline functions use `jlink.fields` and an independent pair scorer. Blocking and
the full pipeline are now implemented; `bench/run.py` runs the character n-gram blocker with
`k=10`. Its import guard remains for older/incomplete checkouts. Internal import errors surface.

Normal runs initialize no API client and perform no API calls. The legacy `--live` flag enables
real API calls with its historical settings; `bench/live.py` was used for the saved live runs
summarized in [BASELINES.md](BASELINES.md). `bench/heldout.py`
has no live flag and cannot send requests. It replays saved scores.

For a small pass, `--sample 1000` samples left records with seed 0, restricts truth to those
left IDs, and keeps all right records as competitors. The sample IDs have a recorded digest.
At `k=10`, 1,000 left records permit at most 10,000 proposed pairs (approximately $0.139 at
330 input tokens and $0.042 per million input tokens, before exact shortcuts or caching).
A $0.10 cap can therefore leave pairs unjudged. The estimate is not measured spend.
Use `--sample 500` to bring that simple estimate to approximately $0.069.

Every run replaces `bench/out/NAME.json`, which records sampling, definitions, software versions,
all baseline metrics, thresholds, blocking recall and elapsed time. Regenerate `BASELINES.md`
after changes; its tables explicitly identify sampled runs. Similarity thresholds see the truth
and maximize in-sample F1, so these results are upper bounds for the fixed methods rather than
held-out performance. A missing threshold in JSON means the best decision was to reject all pairs.

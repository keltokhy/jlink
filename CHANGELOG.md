# Changelog

## 0.3.0

- Add `--api diffusiongemma` and `--api laya` for System One servers running on your own machine,
  through `jevkit-runtime` 0.3: chosen only by name, no key needed, and metered at zero API fees
  unless `JEV_PRICE_PER_MTOK` is set. The runtime's `docs/` explain how to run the servers.

## 0.2.0

- Move transport, configuration, the answer cache and metering to the shared `jevkit-runtime` 0.2.
  Answers are keyed by provider, endpoint and model, so an answer from one provider is never reused
  for another; the cache written by earlier versions is reset on first use and re-asked. Saved runs
  from earlier versions still load.

- Keep ordinary identity link settings compatible with main by omitting the implicit identity
  style; saved runs with an explicit identity style remain readable.

- Qualify API estimates with token, price and throughput assumptions and accept a caller-supplied
  `tokens_per_pair`. Label CLI cost/time figures as short-record scenarios; no article calibration is claimed.

- Keep date windows within their declared bounds at fractional-nanosecond and timestamp-range
  boundaries, using directed integer bounds and excluding out-of-range search intervals.

- Validate and capture blocker provenance before judging in both link and dedupe runs, restoring
  the identity link validation order and preventing requests for unsaveable configurations.

- Reject `--save` directories that equal an input or output path, and reserved run members
  that are directories or special files, before blocking or judging.

- Detect UTC awareness from parsed custom date formats in `window`, including `%z` and `%Z`,
  so equivalent aware times compare and aware/naive comparisons are rejected.

- Read saved probabilities at full float precision in `cluster`, preserving exact threshold
  and average-linkage decisions without model calls.

- Preserve full float precision when loading dedupe cluster IDs, including adjacent float IDs.
  Saved version 2 runs keep the existing integer, float and string ID metadata.

- Add `style="rule"` to `Linker`, `jlink.link`, `jlink.judge`, the `link` command (`--style rule`)
  and the Stata and R wrappers. The proposition becomes "Record A and record B satisfy the
  following match rule. <definition>", so a link can be a relation that is not identity. `entity`
  is optional under this style and a definition is required. Settings record the style;
  `report()` names it and `methods()` describes a relation. Identity remains the default, and
  identity and rule answers never share a cache entry.
- Let an `on` item be one-sided: `("text", None)` or `(None, "neighborhood")` in Python, `--on text=`
  or `--on "=neighborhood"` on the command line. Such a field is shown to the judge on that side
  only and enters that side's text for `sim`. Blocking passes and the exact shortcut need paired
  columns and say so. Saved settings, fingerprints, audit samples and review snapshots accept them.
- Add `jlink.block.window`: candidate pairs whose left value minus right value lies within a
  tolerance or a one-sided range, for numbers and for dates and times. It sorts once and uses
  binary search, streams bounded batches under `max_pairs`, composes with `within` and unions
  with other passes. Missing and unreadable values are dropped and counted, never guessed;
  non-ISO dates need `date_format`. Blocking diagnostics record the counts as `dropped_values`.
- Add `--block window:COLUMN:TOLERANCE`, `--block window:LEFT=RIGHT:LOW..HIGH` with `w/d/h/m/s`
  units, `--block within:COLUMNS:RULE`, and `--date-format`. `estimate` reports records a window
  cannot use.
- Add dedupe, linking a table to itself: `jlink.dedupe`, `Linker.dedupe`, `DedupeResult` and the
  `dedupe` command. No record is paired with itself, each unordered pair is proposed and judged
  once with the earlier row as record A, and every record gets a `cluster_id`. `recluster`, the
  `cluster` command and `jlink.cluster` regroup saved scores under another threshold or rule
  without API calls. `save`, `load`, `report()`, `methods()`, `audit_sample` and `evaluate` work
  for dedupe runs; the review page and the Stata and R wrappers do not yet.
- Cluster by average linkage in which a pair that blocking never proposed counts as a non-match,
  so that one wrong pair does not chain two groups together. `unproposed="ignore"` averages over
  judged pairs only and `linkage="components"` joins everything reachable; `split_pairs()` lists
  the high-probability pairs a rule left apart. Both failure modes are measured on synthetic
  scores in `docs/dedupe.md`.
- Add `block.self_candidates`, `pairs_completeness(..., unordered=True)` and one-table
  `Linker.estimate(table)`.
- Add `--save DIR` to `link` and `dedupe`: the run directory that `Result.save()` writes, which
  `review create` and `jlink.load` read. The review workflow was unreachable from the command
  line alone before. Stata's `rundir()` and R's `run_dir =` forward it.
- Settings, `report()` and `methods()` now quote the question exactly as `judge()` sent it;
  `judge()` records it in `scores.attrs["question"]` and is the only place it is built.

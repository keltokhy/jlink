# Candidate search: groups, reverse neighbors, and limits

`block.candidates` unions its passes. An added pass can recover candidates; it cannot
remove candidates or make an earlier search cheaper. The default remains a forward
character n-gram pass with `k=10`, `n=(2, 4)`, and `min_sim=0.1`.

## Restrict comparisons with `within`

```python
from jlink import block

grouped = block.within(
    block.ngrams(("conm", "assignee"), k=10),
    ("state", "region"), "year",
)
candidates = block.candidates(
    left, right, on=[("conm", "assignee")],
    blockers=[grouped], left_id="gvkey", right_id="assignee_id",
)
```

`within(blocker, *columns, missing="drop", name=None)` runs any `Blocker` separately
on each matching exact group, then maps its local positions back to the original
tables. Group keys use the same normalization as `exact`: case, punctuation, accents,
and whitespace are normalized. Every component must agree; mapped column names use
`("left_column", "right_column")`. A group with no counterpart produces no candidates.

Whole numbers agree however they are stored. pandas turns an integer column into floats as
soon as one value is missing, and a CSV written from that column says `1985.0`, so integer
`1985`, float `1985.0` and the texts `"1985"` and `"1985.0"` are the same key. Other
spellings are not reconciled: `"02139"` and `2139`, or `"FY85"` and `1985`, remain
different keys. If an `exact` or `within` pass proposes no pairs because its key columns
have no value in common between the two tables, `candidates` warns and shows one key from
each side; the pass's `proposed_pairs` diagnostic is 0.

N-gram TF-IDF is fitted on the left and right records **within each group**. Neither
vector fitting nor nearest-neighbor matrix multiplication compares different groups.
This searches for the best available neighbors inside a group, rather than filtering
the globally chosen top-k afterward. The candidate table's final `sim` remains the
existing full-table, all-`on`-field score, so retrieval scores and displayed scores can
differ. Grouping fields need not appear in `on`.

Missing-group behavior is explicit:

| `missing` | Behavior |
|---|---|
| `"drop"` (default) | Exclude rows whose normalized group key has any empty component. |
| `"match"` | Empty components match other empty components; all remaining components must agree. |

For example, with `missing="match"`, `(missing state, 2020)` can match `(missing state,
2020)`, but cannot match `(NY, 2020)` or `(missing state, 2021)`. Missing includes nulls,
empty strings, and text normalized to empty, such as punctuation. Empty is never a
wildcard. Matching missing keys can create a large group. There is no implicit global
fallback: add a separate ungrouped pass if that is intended.

Grouping is appropriate only when the grouping fields reliably agree for true matches.
It can lose true pairs with missing, stale, or inconsistent keys. A nested wrapper is
supported, as are `exact`, `initials`, and custom child passes.

### Why adding `exact("state")` does not partition n-gram search

```python
import pandas as pd

left = pd.DataFrame({"name": ["Acme"], "state": ["NY"]})
right = pd.DataFrame({"name": ["Acme", "Acme New York", "Acme NY"],
                      "state": ["CA", "NY", "NY"]})
```

For these records:

| Passes | Candidate positions |
|---|---|
| `ngrams("name", k=1)` | `(0, 0)` |
| `ngrams("name", k=1), exact("state")` | `(0, 0), (0, 1), (0, 2)` |
| `within(ngrams("name", k=1), "state")` | `(0, 2)` |

The separate exact pass adds every NY pair and retains the CA n-gram candidate. The
wrapper searches only NY and returns its best neighbor. This example is an executable
regression in `tests/test_block_grouped.py`.

## Recover candidates from the reverse direction

```python
passes = [
    block.ngrams(("conm", "assignee"), k=10),
    block.ngrams(("conm", "assignee"), k=10, reverse=True),
]
# Optional restriction applies to each pass:
passes = [block.within(p, ("state", "region")) for p in passes]
```

`reverse=True` selects up to k **left** neighbors for each **right** record. It still
returns `(left_position, right_position)` and accepts the same left/right column
mapping. Unioning forward and reverse passes gives symmetric candidates. Using only
the reverse pass replaces the forward search and can lose forward candidates.

The reverse default name is `ngrams-reverse:<columns>`, distinct from the forward
`ngrams:<columns>`. Names do not encode all parameters; use `to_config()` for provenance.
The symmetric union has at most `k * (len(left) + len(right))` proposals before overlap
and similarity filtering. It fits and searches each direction separately, so it costs
more local computation as well as potentially more judging calls. No defaults changed.

Ordering is deterministic for fixed inputs and settings. Forward passes traverse left
rows; reverse passes traverse right rows. Similarity ties use the neighbor's original
row position. Groups are traversed in first-left-occurrence order, preserving each
group's source row order. The final candidate table retains the existing order: left
row position, descending full-record similarity, then right row position. Source ID
values, dtypes, and nonconsecutive DataFrame indexes remain mapped to original rows.

## Pair limits and memory

Built-in `exact`, `initials`, `ngrams`, and `within` stream pair batches through
`Blocker.iter_pairs`. `candidates(max_pairs=...)` checks the unique union as it grows
and raises at the first excess pair, before scoring candidates. Duplicates and overlaps
do not consume the limit. The error reports an **at least** count; it intentionally
does not finish an oversized Cartesian block to count all its pairs.

For example, a group with 100,000 records on each side would emit 10 billion exact
pairs. With `max_pairs=1000`, the union stops at 1,001; it never constructs the complete
list. Reducing `k` helps n-gram passes, but has no effect on broad exact passes. Constrain
or remove those passes, or replace them with grouped nearest neighbors. Adding another
exact pass cannot reduce the union.

The limit is a pair-count guard, **not a process-memory budget**. Built-in pair batches
contain at most 8,192 pairs. Sparse products retain their existing row-chunk target of
about 64 MiB (a single very wide row can exceed that target); vector storage, group
indexes, and the unique union still scale with input and output size. Final scoring
uses candidate-pair chunks and never builds a dense all-pairs matrix. Very uneven
groups can still be expensive, and many tiny groups incur repeated vectorizer fits.

Calling `blocker.pairs(...)` directly materializes its complete result and has no pair
limit; use `candidates` for the guard or consume `iter_pairs` yourself. Existing custom
passes implementing only `pairs` remain supported, but their allocation happens before
the guard sees their output. A custom `iter_pairs` implementation can yield bounded
integer arrays of shape `(m, 2)`. Each batch is validated before it is used, including
inside a group. Custom passes receive sliced DataFrames and must return **local row
positions**, not index labels or IDs.

## Configuration and contribution diagnostics

Every built-in exposes a JSON-safe `to_config()` dictionary. Common fields are `type`,
`name`, and `columns` (a list of explicit left/right name pairs). Additional fields are:

| `type` | Additional fields |
|---|---|
| `exact` | None |
| `initials` | `min_len` |
| `ngrams` | `k`, `n` (two-element list), `min_sim`, `reverse` |
| `within` | `missing`, `blocker` (recursive child configuration) |

For example:

```json
{
  "type": "within",
  "name": "within:state=region[drop](ngrams-reverse:name)",
  "columns": [["state", "region"]],
  "missing": "drop",
  "blocker": {
    "type": "ngrams",
    "name": "ngrams-reverse:name",
    "columns": [["name", "name"]],
    "k": 10,
    "n": [2, 4],
    "min_sim": 0.1,
    "reverse": true
  }
}
```

Nested wrappers retain all grouping and reverse parameters. An opaque custom pass
inherits `{"type": "custom", "name": ..., "class": ..., "reconstructable": false}`.
That record explicitly does not claim to preserve custom constructor parameters. A
custom blocker can override `to_config()` with a JSON-safe parameter description.
This API describes configuration; it does not deserialize or execute saved classes.

`candidates.attrs["blocking"]` is a JSON-safe object with `schema_version: 1`,
`max_pairs`, `pair_count`, and `passes` in the submitted order. Each pass has:

| Field | Meaning |
|---|---|
| `pass` | Zero-based pass index; remains unambiguous even if names repeat. |
| `name`, `config` | Display name and full nested `to_config()` output. |
| `proposed_pairs` | Raw proposals, including duplicates within that pass. |
| `unique_pairs` | Distinct pairs from that pass. |
| `added_pairs` | Pairs absent from earlier passes; order-dependent. |
| `overlapping_pairs` | Distinct pairs already proposed by earlier passes. |
| `union_pairs` | Cumulative union size after this pass. |
| `exclusive_pairs` | Pairs supplied only by this pass after all passes finish. |

Counts are independent of ambiguous joined `block` labels and need no additional
pair sets. `added_pairs` sum to `pair_count`; `exclusive_pairs` quantify what removing
a pass would lose from the final union. Counts describe candidates, not true matches.
The diagnostics deliberately omit timings so repeated candidate results are stable.
DataFrame attributes are in-memory metadata, not CSV columns; save the object as JSON
if needed. For nested wrappers, diagnostics summarize the top-level grouped pass.

## Offline firm comparison (2026-09-18)

These measurements read the existing local NBER benchmark (4,585 assignees, 2,488
Compustat firms, 4,591 distinct truth pairs). Inputs were read-only; no paid judging or
other API calls were made. The prepared benchmark and saved `bench/out/nber-firms.json`
were inspected. [Recorded results](experiments/blocking-firms.json) include input file
SHA-256 values, full pass configurations, diagnostics, and package versions.

All rows below use `min_sim=0.1`; times include candidate union and final similarity
scoring, but exclude loading the data and measuring recall. These are single-run local
timings, not cross-machine throughput guarantees.

| Search | Pairs | True pairs found | Candidate recall | Seconds |
|---|---:|---:|---:|---:|
| Forward k=10 (unchanged default) | 45,567 | 3,061 | 66.67% | 0.463 |
| Reverse k=10 only | 24,848 | 2,992 | 65.17% | 0.416 |
| Forward + reverse k=10 | 54,976 | 3,080 | 67.09% | 0.767 |
| Forward k=20 | 89,465 | 3,089 | 67.28% | 0.531 |
| Forward k=50 | 197,350 | 3,126 | 68.09% | 0.673 |
| Forward k=100 | 300,842 | 3,141 | 68.42% | 0.832 |
| Forward k=10 + initials | 45,568 | 3,062 | 66.70% | 0.476 |
| Within first name letter, forward k=10 | 35,956 | 2,969 | 64.67% | 0.342 |
| Within first name letter, forward + reverse k=10 | 40,113 | 2,975 | 64.80% | 0.541 |

The symmetric union adds 9,409 candidates (20.6%) and finds 19 more true links than the
default, losing none. Increasing forward k to 20 adds 43,898 candidates and finds 28
more true links. These modest improvements do not establish a universally best setting
and do not justify changing the default. Judging precision, final-link recall, and
cost for the new candidates were **not measured**.

The benchmark has no state/year fields. The grouped rows derive only the first
normalized name character, with no use of identifiers or truth. Group-local fitting
and ranking recover some candidates missed globally, while the strict prefix condition
also loses matches: grouped forward k=10 gains 39 true pairs but loses 131 relative to
the default. These measurements do not establish real-world state-group recall.

The old 67% is an observed result for one blocker, not a theoretical ceiling for
name-based candidate generation. Larger k already exceeds it on these same data.
The cached baseline's much broader positive-cosine search reported 95.62% candidate
recall among 9,777,784 pairs; that is a different search policy and a cached measurement,
not a new final-link evaluation. Exhaustive pairing would include all labeled pairs
within these tables, at the cost of 11,407,480 candidates. This says nothing about the
ability to identify the correct matches among those pairs.

The complete default candidate table was also compared with baseline commit `2c10353`:
all 45,567 candidate ID pairs, labels, ordering, and floating-point similarity values
were exactly equal. The new diagnostics add metadata without changing those values.

## Scaling and early-stop measurements

Fresh processes on this Mac (Python 3.12.14, macOS arm64) produced:

| Rows on each side | Search | Pairs | Seconds | Peak RSS | Planted-match recall |
|---|---|---:|---:|---:|---:|
| 20,000 | Global forward k=10 | 200,000 | 11.30 | 706 MiB | 99.99% |
| 20,000 | Forward k=10 within 50 equal groups | 200,000 | 2.04 | 383 MiB | 100.00% |
| 100,000 | Forward k=10 within 50 equal groups | 1,000,000 | 14.10 | 921 MiB | 100.00% |

These full `candidates` timings include final scoring and metadata. They should not be
compared directly with older timings that called only `ngrams(...).pairs(...)`.
The synthetic group keys agree by construction; the recall figures are not predictions
for real state/year data. Raw [20k global](experiments/blocking-scale-20k-global.json)
and [100k grouped](experiments/blocking-scale-100k-grouped.json) outputs are retained.

A single 100,000-by-100,000 exact group has 10 billion possible pairs. The
[`max_pairs=1000` guard check](experiments/blocking-guard-100k.json) stopped at 1,001
unique pairs in 0.300 seconds, with 194 MiB peak RSS for the entire process. It did not
materialize the Cartesian output. Regression tests additionally intercept iteration
to prove that oversized exact, grouped exact, initials, and zero-vocabulary n-gram
passes stop after their first bounded batch.

The full offline suite passed **308 tests**. Focused regressions cover group-local
matrix dimensions, sparse-product bounds, chunk-independent ties, original position
and ID mapping, mapped columns, nested/custom passes, explicit missing keys, reverse
recovery, JSON-safe nested configurations, and overlap-aware diagnostics.

## Reproduce without judging

Run these commands from the repository root after installing the development extras.
The script is isolated from the benchmark harness and writes only the requested output.

```bash
PYTHONPATH=src uv run python docs/experiments/blocking.py firms \
  --data-root /Volumes/K3/GitHub/jlink/bench/data/nber-firms \
  --output docs/experiments/blocking-firms.json
PYTHONPATH=src uv run python docs/experiments/blocking.py scale \
  --rows 100000 --groups 50 --output docs/experiments/blocking-scale-100k-grouped.json
PYTHONPATH=src uv run python docs/experiments/blocking.py scale \
  --rows 20000 --groups 1 --output docs/experiments/blocking-scale-20k-global.json
PYTHONPATH=src uv run python docs/experiments/blocking.py scale \
  --rows 20000 --groups 50 --output docs/experiments/blocking-scale-20k-grouped.json
PYTHONPATH=src uv run python docs/experiments/blocking.py guard \
  --rows 100000 --output docs/experiments/blocking-guard-100k.json
PYTHONPATH=src uv run pytest
```

The synthetic scale corpus uses common company-name tokens, duplicate names, changed
legal suffixes, and abbreviated first words; both sides have the same fixed group per
true pair and the right side is shuffled. It is a controlled scaling check, not
empirical evidence that grouping fields agree in real records. Each scale/guard command
should run in its own process to measure peak RSS independently. Timing includes union
and final scoring; RSS includes Python, imports, input frames, vectors, and outputs.

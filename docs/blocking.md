# Candidate search: groups, windows, reverse neighbors, and limits

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

## Windows on dates and numbers

```python
after = block.window(("published", "occurred"), between=(0, 3), unit="days")
nearby = block.window("year", 1)                       # numbers: -1 <= left - right <= 1
in_borough = block.within(after, ("boro", "BORO"))     # compose like any other pass
```

`window(column, tolerance=None, *, between=None, unit=None, date_format=None, name=None)`
proposes every pair whose **left value minus right value** lies inside the window. Give
exactly one of:

| Argument | Pairs kept |
|---|---|
| `tolerance=t` (`t >= 0`) | `-t <= left - right <= t` |
| `between=(low, high)` | `low <= left - right <= high` |

`between` may exclude one direction. With articles on the left and incidents on the right,
`between=(0, 3), unit="days"` keeps an article published on the day of the incident or up to
three days after it, and never one published before it. Swapping the tables flips the sign:
the same window is then `between=(-3, 0)`.

Without `unit` the values are numbers. With `unit` (`"weeks"`, `"days"`, `"hours"`,
`"minutes"` or `"seconds"`) they are dates or times and the bounds are in that unit; bounds
may be fractions. The two cases are never inferred from the data: a date column without a
unit, or a number column with one, is an error that names the column. The window takes one
column, or one `(left, right)` pair; repeat the pass or wrap it in `within` for more.

**How values are read.** Nothing is repaired or guessed.

- Numbers: numeric columns as they are; text through `pd.to_numeric`. `"1,000"`, `"abc"`
  and infinities do not parse. Values are compared as float64, so integers above 2^53 lose
  precision. The test is `left - high <= right <= left - low`, evaluated in floating point.
- Dates: datetime columns as they are; Python `date` and `datetime` objects; text in ISO
  8601 (`2024-03-01`, `2024-03-01T14:30`). Any other text needs
  `date_format="%m/%d/%Y"`, or `date_format=(left_format, right_format)` with `None` for an
  ISO side. Without it `03/02/2024` is unreadable, not a guess between March and February.
  Dates are compared as whole nanoseconds, exactly; they must fall in the years 1677 to 2262.
- A date-only value is midnight. `between=(0, 3), unit="days"` therefore pairs an article
  stamped `2024-03-05T09:00` with an incident on `2024-03-02` only if the incident has a
  time of day at or after 09:00. Truncate timestamps to dates first when you mean calendar
  days, or widen the window.
- Times with UTC offsets are compared in UTC. A column that mixes offset and offset-free
  times, or one side with offsets and the other without, is an error and not an assumption.
- Missing values (nulls, empty or blank text) and unreadable values never pair.
  `blocker.dropped(left, right)` returns the count of each per side, and `candidates` stores the
  same object as `dropped_values` in that pass's diagnostics. Unreadable values also raise a
  warning, because they usually mean a wrong format; missing values alone do not. Inside
  `within`, the counts still cover the whole tables.

**How it searches.** The right values are sorted once. Each left value finds the start and
end of its range by binary search, and the pairs in that range are streamed in batches of at
most 8,192 like every built-in pass, so `max_pairs` stops an oversized window after one
batch. No left-by-right comparison is made. Output order is left row order, then right row
position. Inside `within`, the sort and search run separately in each group.

A window is only as selective as the data are sparse. A three-day window over a register with
four incidents a day proposes about sixteen incidents for each article. Group it with
`within(...)` on a field that reliably agrees (borough, state), and remember that grouping loses
pairs whose keys disagree or are missing. Union it with other passes like any blocker: a union
adds pairs and never narrows the window.

Measured offline on this Mac (Python 3.13.15, macOS arm64; single runs, synthetic data with
uniformly spread dates and five groups, each article planted 0 to 3 days after one incident):

| Articles x incidents | Possible pairs | Pass | Pairs | Seconds | Peak RSS |
|---|---:|---|---:|---:|---:|
| 48,000 x 24,000 | 1.15 billion | full `candidates`, grouped 0 to 3 days | 188,280 | 1.08 | 264 MiB |
| 2,000,000 x 1,000,000 | 2 trillion | pass only, grouped 0 to 3 days | 17,997,740 | 6.80 | 1,343 MiB |

The first row includes the union and `sim` scoring and found every planted pair, which the
construction guarantees; it says nothing about recall on real dates. The second row times
`iter_pairs` alone, without the union, whose memory grows with the number of pairs. Neither
row involved judging. Raw outputs: [register scale](experiments/blocking-window-register.json)
and [one million incidents](experiments/blocking-window-1m-pass.json).

On the command line the rule is `window:COLUMN:TOLERANCE` or `window:LEFT=RIGHT:LOW..HIGH`,
with a `w`, `d`, `h`, `m` or `s` suffix for dates and none for numbers, and
`within:COLUMNS:RULE` wraps any rule:

```bash
jev-link estimate articles.csv incidents.csv --on text= --on "=neighborhood" \
    --block within:boro=BORO:window:published=OCCUR_DATE:0..3d --date-format "=%m/%d/%Y"
```

`--date-format FORMAT` applies to both files and `LEFT=RIGHT` gives each its own, where an
empty side stays ISO 8601. `estimate` prints, for each window pass, how many records on each
side have a missing or unreadable value. `within` on the command line always uses
`missing="drop"`.

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

Built-in `exact`, `initials`, `ngrams`, `window`, and `within` stream pair batches through
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
| `window` | `low`, `high`, `unit` (null for numbers), `difference` (`left_minus_right`), `date_format` (two-element list, or null for numbers), `values` (how values were compared) |

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
| `dropped_values` | Only for passes that drop unusable values (`window`, or `within` around one): `kind`, and per side the `column`, `missing` and `unparseable` counts and the `time_zone` convention. |

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
  --data-root bench/data/nber-firms \
  --output docs/experiments/blocking-firms.json
PYTHONPATH=src uv run python docs/experiments/blocking.py scale \
  --rows 100000 --groups 50 --output docs/experiments/blocking-scale-100k-grouped.json
PYTHONPATH=src uv run python docs/experiments/blocking.py scale \
  --rows 20000 --groups 1 --output docs/experiments/blocking-scale-20k-global.json
PYTHONPATH=src uv run python docs/experiments/blocking.py scale \
  --rows 20000 --groups 50 --output docs/experiments/blocking-scale-20k-grouped.json
PYTHONPATH=src uv run python docs/experiments/blocking.py guard \
  --rows 100000 --output docs/experiments/blocking-guard-100k.json
PYTHONPATH=src uv run python docs/experiments/blocking.py window \
  --rows 24000 --groups 5 --output docs/experiments/blocking-window-register.json
PYTHONPATH=src uv run python docs/experiments/blocking.py window \
  --rows 1000000 --groups 5 --pass-only --output docs/experiments/blocking-window-1m-pass.json
PYTHONPATH=src uv run pytest
```

The synthetic scale corpus uses common company-name tokens, duplicate names, changed
legal suffixes, and abbreviated first words; both sides have the same fixed group per
true pair and the right side is shuffled. It is a controlled scaling check, not
empirical evidence that grouping fields agree in real records. Each scale/guard command
should run in its own process to measure peak RSS independently. Timing includes union
and final scoring; RSS includes Python, imports, input frames, vectors, and outputs.

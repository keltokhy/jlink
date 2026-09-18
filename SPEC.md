# jlink design contract

jlink links records across two datasets. The user writes the match rule in plain English. A
cheap blocking step proposes candidate pairs, TypeSafe's Jev model returns a probability that
each pair is a match, and a resolve step turns probabilities into links. An audit step draws a
sample for hand-checking and reports precision, recall and calibration for a data appendix.

The audience is applied economists: people who link firm names, person records, places and
products across sources, who work in pandas, Stata or R, and who must describe and defend the
linkage in a paper.

Modules are built in parallel by different people. This file is the contract between them.
Do not change a signature or a column name here without raising it; note the problem in your
DONE.md instead.

## Pipeline and modules

```
left, right ──block──▶ candidates ──judge──▶ scores ──resolve──▶ links
                                                 └──audit──▶ sample ──(human labels)──▶ evaluate
```

| Module | Owner | Contents |
|---|---|---|
| `core.py` | done | Jev client, cache, cost meter. Shared verbatim with jgrep. Do not edit. |
| `fields.py` | done | `parse_on`, `record_text`, `normalize`. Do not edit. |
| `block.py` | worker `block` | candidate generation |
| `judge.py` | lead | pair probabilities from Jev |
| `resolve.py` | worker `resolve` | probabilities to links |
| `audit.py` | worker `resolve` | audit sample, evaluation, scoring against known truth |
| `linker.py` | lead | `Linker`, `Result`, methods paragraph |
| `io.py`, `cli.py` | worker `cli` | file formats and the `jlink` command |
| `stata/`, `r/` | worker `cli` | thin wrappers around the command |
| `bench/` | worker `bench` | benchmark datasets, baselines, harness |

## Conventions

- Python 3.10+. Dependencies are fixed in `pyproject.toml`: httpx, numpy, pandas, scipy,
  scikit-learn, tqdm. Do not add runtime dependencies. `bench/` may use the `bench` group.
- Tests use pytest, run offline, and never call a real API. `tests/fakes.py` has a fake Jev.
- Line length up to 110. Type hints on public functions. Docstrings say what and why, briefly.
  Match the style of `core.py`.
- Public functions validate their inputs and raise `ValueError` with a message an economist can
  act on (name the column, say what was expected).
- Nothing prints except the CLI and progress bars. Library code may use `warnings.warn`.

## Fields: the `on` argument

`on` lists the fields shown to the judge and used for similarity. Each item is a column name
present in both frames, or a `(left_column, right_column)` pair. `fields.parse_on(on)` returns
`[(label, left_column, right_column), ...]`, where the label is the left column name.

`fields.record_text(frame, columns)` returns a `pd.Series` of normalized text: the listed
columns joined by a space, after `fields.normalize`, which casefolds, strips accents, turns `&`
into ` and `, drops punctuation and collapses whitespace. Missing values become empty strings.

## IDs

`left_id` and `right_id` name the ID column in each frame. `None` means the frame's index.
IDs must be unique within a frame; raise `ValueError` if not. Every table below carries IDs in
columns literally named `left_id` and `right_id`, whatever the source columns were called.

## Tables

**candidates**: one row per proposed pair, no duplicate `(left_id, right_id)`.

| column | type | meaning |
|---|---|---|
| `left_id`, `right_id` | as in source | the pair |
| `block` | str | names of the passes that proposed it, joined by `+`, in the order the passes were given |
| `sim` | float in [0, 1] | cosine similarity of character n-gram TF-IDF vectors of `record_text` over all `on` fields, vectorizer fit on left and right together. Present for every pair, whichever pass proposed it. |

**scores**: candidates plus

| column | type | meaning |
|---|---|---|
| `p` | float | probability the pair is a match; NaN if not judged |
| `source` | str | `exact` (all `on` fields identical after normalization; no API call), `jev`, `error`, or `unjudged` (budget ran out) |
| `error` | str or NA | message when `source == "error"` |

**links**: the chosen pairs, with every `scores` column plus

| column | type | meaning |
|---|---|---|
| `margin` | float | `p` minus the highest `p` among all other scored pairs that share this pair's `left_id` or `right_id`. NaN when there is no such competitor. Competitors are all scored pairs, not only those above the threshold. |

Sorted by `p` descending, then `left_id`; index reset.

## block.py

```python
class Blocker:
    name: str
    def pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray: ...
        # shape (m, 2), dtype int64: positions (not IDs) of left and right rows
    def iter_pairs(self, left: pd.DataFrame, right: pd.DataFrame) -> Iterator[np.ndarray]: ...
        # bounded batches for built-ins; fallback calls pairs() for existing custom passes
    def to_config(self) -> dict: ...  # JSON-safe configuration; recursive for within

def exact(*columns: str | tuple[str, str], name: str | None = None) -> Blocker
def ngrams(*columns: str | tuple[str, str], k: int = 10, n: tuple[int, int] = (2, 4),
           min_sim: float = 0.1, name: str | None = None, reverse: bool = False) -> Blocker
def initials(column: str | tuple[str, str], min_len: int = 2, name: str | None = None) -> Blocker
def within(blocker: Blocker, *columns: str | tuple[str, str], missing: str = "drop",
           name: str | None = None) -> Blocker
def candidates(left, right, *, on, blockers: list[Blocker] | None = None, left_id=None, right_id=None,
               max_pairs: int | None = 5_000_000) -> pd.DataFrame
def pairs_completeness(candidates: pd.DataFrame, truth: pd.DataFrame) -> float
```

- `exact`: pairs whose listed columns are all equal after `normalize`. Rows with an empty key
  never pair. Default name `exact:<columns>`.
- `ngrams`: for each left row, the `k` right rows with the highest character n-gram TF-IDF
  cosine similarity on the listed columns (joined as in `record_text`), keeping only
  similarity >= `min_sim`. Must scale: 100,000 by 100,000 rows in a few minutes and under
  4 GB, so multiply sparse matrices in row chunks and take top-k per chunk. Never build a
  dense n-by-m matrix. Default name `ngrams:<columns>`.
  With `reverse=True`, select up to `k` left neighbors per right row, still returning
  `(left_position, right_position)`; default name `ngrams-reverse:<columns>`. Union forward
  and reverse passes for symmetric search. The default remains forward only.
- `within`: run its child blocker separately in each matching normalized exact group,
  supporting mapped left/right columns and restoring original positions. N-gram TF-IDF is
  fit within each group. `missing="drop"` omits incomplete keys; `missing="match"` allows
  identical incomplete keys (empty components are equal, not wildcards).
- `initials`: pairs where one side's normalized text, read as one token of at least `min_len`
  letters, equals the initials of the other side's tokens, in either direction ("IBM" and
  "International Business Machines"). Ignore the stop words `and`, `of`, `the`, `for` and
  common corporate suffixes (`inc`, `corp`, `co`, `ltd`, `llc`, `plc`, `company`,
  `corporation`, `incorporated`, `limited`) when forming initials. Default name
  `initials:<column>`.
- `candidates`: run each blocker, union the pairs, fill `block` and `sim`, map positions to
  IDs. `blockers=None` means `[ngrams(*all on fields, k=10)]`. If the union exceeds
  `max_pairs`, raise `ValueError` at the first excess unique pair, reporting an "at least"
  count and suggesting smaller `k`, `within(...)`, or constraining/removing broad passes.
  Built-ins stream bounded pair batches via `iter_pairs`; custom `pairs` implementations
  remain supported. Adding `exact` cannot constrain another pass because passes are unioned.
  `Blocker.to_config()` and `candidates.attrs["blocking"]` expose nested configurations and
  per-pass contributions; see [the schema and ordering contract](docs/blocking.md).
- `pairs_completeness`: share of `truth` pairs (columns `left_id`, `right_id`) present in
  `candidates`. This is blocking recall.

## judge.py (lead)

```python
def judge(candidates, left, right, *, on, entity: str, definition: str = "", left_id=None, right_id=None,
          api=None, model=None, concurrency=32, budget: float | None = 5.0, cache=True,
          exact_shortcut=True, progress=True, transport=None) -> tuple[pd.DataFrame, Meter]
```

One call per pair. The state is `{"record_a": {label: value, ...}, "record_b": {...}}` with
missing fields dropped. Pairs are judged in descending `sim`, so a budget is spent on the
likeliest pairs first. Returns the scores table and the cost meter.

## resolve.py

```python
def resolve(scores: pd.DataFrame, *, how: str = "one-to-one", threshold: float = 0.5,
            min_margin: float | None = None) -> pd.DataFrame
```

Rows with NaN `p` are ignored. `how` is one of:

- `many-to-many`: every pair with `p >= threshold`.
- `many-to-one`: each `left_id` keeps its single best pair (several left rows may share a
  right row). `one-to-many` is the mirror image.
- `one-to-one`: a maximum-total-`p` matching among pairs with `p >= threshold`, so no ID
  appears twice. Solve per connected component with `scipy.optimize.linear_sum_assignment`;
  for a component with more than 2,000 nodes on a side, fall back to greedy by descending `p`
  and warn. Ties break by higher `sim`, then by ID order, so output is deterministic.

Then compute `margin`. When `min_margin` is given, drop links with `margin < min_margin` (NaN margins
pass). The default is no margin filter: a filter at 0 would silently remove every link that has a
higher-scoring competitor, which guts `many-to-many`.

## audit.py

```python
def audit_sample(scores, *, n=200, bins=(0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0), seed=0,
                 left=None, right=None, on=None, left_id=None, right_id=None) -> pd.DataFrame
def evaluate(labeled: pd.DataFrame, *, threshold=0.5, n_boot=2000, seed=0) -> Evaluation
def score_against_truth(links, truth, candidates=None) -> dict
```

- `audit_sample`: a stratified random sample of judged pairs (links and non-links) for a
  person to label. Strata are `p` bins (right-closed, first bin includes 0). Allocate `n`
  equally across non-empty bins, capped at bin size, and redistribute the remainder. Columns:
  `left_id`, `right_id`, `p`, `bin`, `weight` (bin population divided by bin sample size), an
  empty `is_match` column, and, when `left`, `right` and `on` are given, the fields side by
  side as `a_<label>` and `b_<label>` so the labeler needs nothing else. Shuffled, so bins are
  not labeled in order.
- `evaluate`: reads `is_match` as 1/0, True/False, y/n or yes/no in any case. Blank rows are
  dropped and counted. Using `weight`, estimate precision, recall and F1 at `threshold`, each
  with a 95% interval from a stratified bootstrap, plus a weighted Brier score and a
  calibration table (`bin`, `n`, `mean_p`, `match_rate`). Recall here is recall among candidate
  pairs; say so in the summary, because pairs lost in blocking are invisible to it.
- `Evaluation`: attributes `precision`, `recall`, `f1` (each a `(estimate, low, high)`
  tuple), `brier`, `calibration`, `n_labeled`, `n_unlabeled`, `threshold`; methods
  `summary() -> str` and `to_markdown() -> str` (a table fit for a data appendix).
- `score_against_truth`: for benchmarks with known matches. `truth` has `left_id`, `right_id`.
  Returns `precision`, `recall`, `f1`, `tp`, `fp`, `fn`, and `pairs_completeness` when
  `candidates` is given.

## linker.py (lead)

```python
linker = jlink.Linker(entity="firm", definition="...", on=["name", ("city", "town")],
                      blockers=[jlink.block.ngrams("name", k=10), jlink.block.initials("name")])
result = linker.link(left, right, left_id="gvkey", right_id="id", how="one-to-one",
                     threshold=0.5, min_margin=None, budget=5.0)
result.links, result.scores, result.candidates, result.meter, result.settings
result.relink(how=..., threshold=..., min_margin=...)   # no new API calls
result.merged(left, right)                             # left and right columns side by side, plus p
result.audit_sample(n=200), result.report(), result.methods()
result.save(directory); jlink.load(directory)
```

## io.py and cli.py

`io.read_table(path)` and `io.write_table(frame, path)` choose the format from the extension:
`.csv`, `.tsv`, `.dta`, `.parquet`. Read every ID column as it is stored; do not coerce
strings to numbers (`dtype=str` for delimited files, then leave conversion to the caller).

```
jlink link LEFT RIGHT --on name [--on city=town] --entity firm [--define "..."]
           [--left-id COL] [--right-id COL] [--block ngrams:name:10] [--block exact:state]
           [--block initials:name] [--how one-to-one] [--threshold 0.5] [--min-margin M]
           [--budget 5] [-o links.csv] [--scores scores.csv] [--report report.md]
           [--api typesafe|openrouter] [--model ID] [--no-cache] [-j 32]
jlink estimate LEFT RIGHT --on ...      # blocking only: pair count, cost and time estimate, no API calls
jlink audit SCORES [-n 200] [--left LEFT --right RIGHT --on ...] -o audit.csv
jlink evaluate LABELED [--threshold 0.5] [--markdown]
jlink --version
```

`--on city=town` means left column `city`, right column `town`. Cost estimate: about 330 input
tokens per pair at $0.042 per million tokens; time estimate: about 200 pairs a second. Exit
status 0 on success, 2 on any error, with a one-line message on stderr prefixed `jlink:`.
Console scripts `jlink` and `jev-link` are the same program; macOS ships a Java tool at
`/usr/bin/jlink`, so the docs must mention `jev-link` and `python -m jlink`.

The CLI builds a `jlink.Linker` and calls `link`. Until `linker.py` lands, code against the
signatures above.

## Stata and R wrappers

Thin shims that write the data in memory to a temporary file, call the command, and read the
links back. Stata: `jlink using right.dta, on(name city) entity(firm) [define() leftid()
rightid() how() threshold() budget() saving()]`, with a `.sthlp` help file. R: a single
`jlink()` function in `r/jlink.R` using `system2`, returning a data frame. Find the executable
as `jev-link` first, then `python3 -m jlink`. State plainly in each file's header whether it
was run against a real Stata or R on this machine.

## bench/

Datasets with known matches, prepared into a common layout so one harness runs them all:
`bench/data/<name>/left.parquet`, `right.parquet`, `truth.parquet` (`left_id`, `right_id`),
and `meta.json` (`entity`, `definition`, `on`, `how`, source URL, license, row counts).
`bench/data/` is git-ignored; `bench/prepare.py` downloads and builds it reproducibly.

- Febrl4 (people; via the `recordlinkage` package).
- DBLP-ACM, Abt-Buy, Amazon-Google (the Leipzig entity-resolution benchmarks).
- A firm-name benchmark that an economist would recognize. Investigate, in this order, and
  document what you find: the NBER Patent Data Project's assignee-to-Compustat name match;
  any other public, labeled company-name matching set. Time-box this to about an hour.

`bench/baselines.py`: exact match after normalization; best-match Jaro-Winkler with the
threshold tuned on the truth (an upper bound for the method); best-match TF-IDF cosine, same
tuning. `bench/run.py --dataset NAME --budget DOLLARS` runs blocking, the baselines and, only
when `--live` is passed, jlink itself, and writes `bench/out/<name>.json`. The `bench` worker
does not pass `--live`; the lead runs the live pass and controls the spend.

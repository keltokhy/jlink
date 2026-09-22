# jlink

Record linkage where you write the match rule in plain English.

```python
import jlink

result = jlink.link(
    compustat, patents,
    entity="firm",
    on=[("conm", "assignee"), "state"],
    definition="A parent company and its subsidiary are different firms. "
               "A firm that changed its legal form (Inc. to LLC) is the same firm.",
    left_id="gvkey", right_id="assignee_id",
)

result.links             # gvkey, assignee_id, p (probability of a match), margin, ...
print(result.report())   # pairs compared, links made, what it cost
print(result.methods())  # a paragraph for your data appendix
```

String distance links "Acme Widgets Inc." to "ACME WIDGETS, INC". It does not link "IBM" to
"International Business Machines", and it cannot know that you want a subsidiary kept apart
from its parent. A research assistant can do both, slowly. jlink asks
[Jev](https://docs.typesafe.ai), a decision model from TypeSafe, the question a research
assistant would answer: given these two records and this rule, are they the same firm? Jev
returns a probability in about a fifth of a second for about a thousandth of a cent. In the
benchmarks below, 146,119 candidate pairs cost $2.49 in total, at about 250 pairs a second.

jlink is built for the way applied economists link data:

- **The rule is part of the method.** You state what counts as a match, and that sentence goes
  in your appendix. Change the sentence and you change the linkage.
- **Every pair gets a probability**, so you can set a threshold, require a margin over the
  runner-up, or carry the uncertainty into estimation.
- **You can check it.** jlink draws a stratified sample for hand-labeling and turns your labels
  into precision, recall and a calibration table with confidence intervals.
- **It reproduces.** Every probability is stored. Rerunning costs nothing and returns the same
  links.
- **It works from Python, the command line, Stata and R**, on `.csv`, `.dta` and `.parquet`.

## Before you start: where your data goes

jlink sends the fields you list in `on`, for each candidate pair, to an outside API (TypeSafe
or OpenRouter). Do not use it on confidential or restricted-use data, such as Census RDC
files, identified administrative records or anything under a data use agreement, unless that
agreement allows it. Only the `on` fields leave your machine; blocking runs locally. With a
local server (`--api laya` or `--api diffusiongemma`, below) nothing leaves the machine at all.

## Install

```bash
uv add jlink            # in a project, for `import jlink`
uv tool install jlink   # the command line, which Stata and R also use
```

Both need [uv](https://docs.astral.sh/uv/). Add `pyarrow` if you read or write `.parquet`.

You need a key for one of two APIs. With keys for both, jlink uses TypeSafe's.

| API | Environment variable | Get a key |
|---|---|---|
| TypeSafe | `TYPESAFE_API_KEY` | [console.typesafe.ai](https://console.typesafe.ai/settings/keys) |
| OpenRouter | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |

A key can also live in `~/.config/jev/typesafe.key` or `~/.config/jev/openrouter.key`.

### Local servers (experimental)

`--api diffusiongemma` and `--api laya`, or `JEV_API=laya` for the Python, Stata and R entry
points, send the same pair questions to a System One server on your own machine, an
[OpenJev](https://github.com/razorback16/openjev) or [laya-mlx](https://github.com/mizorewww/laya-mlx)
process that you run separately. They are never chosen automatically, need no key, and count as $0
in the cost meter and a saved run's settings unless `JEV_PRICE_PER_MTOK` is set. A run records the
provider and model that answered, so results from a local model are attributed like any other.
The runtime's [DiffusionGemma](https://github.com/keltokhy/jevkit-core/blob/main/docs/diffusiongemma.md)
and [Laya](https://github.com/keltokhy/jevkit-core/blob/main/docs/laya.md) guides explain the setup; keep concurrency low while a local model warms up.

## Try it

The repository ships a toy example: twelve Compustat-style firms and sixteen patent assignees.

```bash
jev-link link examples/compustat_sample.csv examples/patent_assignees.csv \
    --on conm=assignee --entity firm --left-id gvkey --right-id assignee_id --how many-to-many \
    --define "A subsidiary or division counts as its parent company. Different companies that merely share a word are not a match." \
    --block ngrams:conm=assignee:5 --block initials:conm=assignee --block exact:state -o links.csv
```

```
0.97  INTL BUSINESS MACHINES CORP   <- IBM                                      [initials]
0.98  INTL BUSINESS MACHINES CORP   <- International Business Machines Corporation
0.97  MINNESOTA MINING & MFG CO     <- 3M Company                               [exact:state]
0.94  UNITED TECHNOLOGIES CORP      <- Pratt & Whitney (United Technologies)
0.87  CORNING INC                   <- Corning Glass Works
0.80  EXXON CORP                    <- Exxon Research and Engineering Co.
0.69  FORD MOTOR CO                 <- Ford Global Technologies
      ... and five plain matches (Abbott Labs, The Boeing Company, ...)
not linked: Westinghouse Air Brake Company, Abbott Ball Company, General Electrodynamics Corp, Rockwell Automation
```

Fifty pairs, $0.0007. Jev knows that 3M is Minnesota Mining and that Westinghouse Air Brake is
not Westinghouse Electric. It can only judge pairs that blocking proposes, though: "3M Company"
shares no letters with "Minnesota Mining & Mfg", so it was found only because the
`exact:state` pass paired firms within a state.

## Benchmarks

Five public datasets with known matches, run end to end (blocking, judging, resolving) on
2026-09-18 with Jev 1.13 through OpenRouter. jlink used its default threshold of 0.5 everywhere.
Nothing was tuned on the answers. Each string baseline, by contrast, was given the single
threshold that maximizes its F1 on the answers, so the baseline column is an upper bound on
what that method can do.

| Dataset | Records | jlink precision / recall | jlink F1 | Best string baseline F1 | Pairs judged | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Firms: NBER patent assignees to Compustat | 4,585 x 2,488 | 0.90 / 0.62 | **0.73** | 0.69 | 45,567 | $0.62 |
| Publications: DBLP to ACM | 2,616 x 2,294 | 0.99 / 1.00 | **0.996** | 0.975 | 26,146 | $0.45 |
| Products: Abt to Buy | 1,081 x 1,092 | 0.90 / 0.93 | **0.92** | 0.86 | 10,796 | $0.22 |
| Software: Amazon to Google | 1,363 x 3,226 | 0.60 / 0.74 | **0.66** | 0.64 | 13,610 | $0.22 |
| People: FEBRL4, synthetic typos | 5,000 x 5,000 | 1.00 / 0.92 | 0.96 | **0.998** | 50,000 | $0.97 |

The string baselines are best-match Jaro-Winkler and best-match TF-IDF cosine; the table shows
the better of the two. Exact matching after normalization scores 0.26, 0.41, 0.00, 0.00 and
0.22.

What the table says:

- **Where names carry meaning, jlink wins without tuning.** On firms, products and publications
  it beats string similarity that was handed its best threshold.
- **Where records differ only by typos, you do not need it.** FEBRL4 is synthetic person data
  with character-level corruption, and TF-IDF cosine is nearly perfect there and free. jlink
  made no false links (precision 1.00) but was too cautious at 0.5; at a threshold of 0.3 its
  F1 is 0.98.
- **The firm run was limited by its candidate search.** The default forward top-10 n-gram
  pass proposed 67% of known links; this is measured blocking recall, not a ceiling for
  name-based methods. Ownership links such as "Homogeneous Metals Inc" to "United Technologies
  Corp" can be difficult to retrieve from names. jlink found 93% of the proposed true links.
  [Candidate search](docs/blocking.md) describes reverse search, larger `k` and their pair-count cost. (A sliver of its error is the benchmark's: 11 Compustat names appear
  under two IDs, which accounts for 14 of jlink's 331 false links.)
- **Amazon to Google is hard for everyone**, because listings differ in version and edition
  details that the records often omit.

The firm rule was revised once, after reading the errors of a 300-record pilot, as a user
would: the first draft did not tell Jev that `CPY` means Company in these data. That changed F1
by two points. The final rule is two sentences: *"Ignore legal-form suffixes (Inc, Corp, Co,
CPY, Ltd, GmbH, N V). A subsidiary or division counts as its parent company."*

**Are the probabilities calibrated?** Roughly, and it depends on the data. On the firm
benchmark, pairs scored above 0.8 were true matches 95 to 98% of the time and pairs scored
below 0.2 were true 0.1% of the time, but the 0.5 to 0.8 band was overconfident (mean 0.64,
true 39% of the time). On FEBRL4 Jev was
underconfident: pairs in the 0.2 to 0.5 band were true matches 66% of the time. Treat `p` as a
strong ranking and check the middle band with an audit sample before using it as a literal
probability.

**Speed.** The firm run judged 45,567 pairs in 176 seconds (259 pairs a second, 64 calls in
flight, median latency 214 ms, one retry). Blocking 100,000 by 100,000 records takes about four
minutes and 1 GB on an M3 Ultra; blocking time grows roughly with the square of the data, so
beyond that size, a reliable shared field such as state or year can restrict the search with
`jlink.block.within(jlink.block.ngrams("name"), "state")`. This searches separately inside
matching groups. Adding a separate `exact("state")` pass unions more pairs and does not split
the existing search. Grouping can lose matches when group values disagree or are missing;
see [grouping, reverse search, and pair limits](docs/blocking.md).

Reproduce everything: `uv sync --group bench`, `uv run python bench/prepare.py`, then
`uv run python bench/live.py nber-firms --budget 1.00`. Baselines and data provenance are in
`bench/BASELINES.md` and `bench/FIRM_DATA.md`.

## How it works

1. **Block.** Comparing every record with every other is wasteful, so jlink first proposes
   candidate pairs on your machine, at no cost. By default each left record is paired with its
   ten nearest right records by character n-grams. Add passes for what n-grams miss:
   `jlink.block.initials("name")` pairs "IBM" with "International Business Machines",
   `jlink.block.exact("state")` pairs everything within a state, and
   `jlink.block.window("year", 1)` pairs records whose years differ by at most one.
2. **Judge.** Each candidate pair goes to Jev with your rule, including equal names: identical
   text need not identify the same entity. If equal compared fields establish identity in your
   data, explicitly enable `Linker(..., exact_shortcut=True)` to accept complete normalized
   equalities without a call. Likelier pairs are judged first; cached scores remain available after the budget runs out.
3. **Resolve.** Choose links from the probabilities: `one-to-one` (the default; the best
   overall assignment with no record used twice), `many-to-one`, `one-to-many` or
   `many-to-many`, with a probability threshold and an optional margin over the runner-up.
   `result.relink(...)` tries other rules without paying again.
4. **Audit.** `result.audit_sample(200)` draws pairs across the probability range, links and
   non-links alike, with both records side by side. Label them in a spreadsheet, then
   `jlink.evaluate(labeled, mode="selected")` evaluates the delivered links and pair-score calibration.

```python
linker = jlink.Linker(
    entity="firm", on=[("conm", "assignee"), "state"], definition="...",
    blockers=[jlink.block.ngrams(("conm", "assignee"), k=10), jlink.block.initials(("conm", "assignee"))],
)
linker.estimate(compustat, patents, left_id="gvkey", right_id="assignee_id")   # blocking only, no API calls
# {'left': 4585, 'right': 2488, 'pairs': 45567, 'dollars': 0.6316, 'seconds': 227.8,
#  'assumptions': {'tokens_per_pair': 330, 'price_per_million_tokens': 0.042,
#                  'pairs_per_second': 200, 'token_basis': 'short_records',
#                  'throughput_basis': 'short_records'}}
# (the real run on these data cost $0.62 and took 176 seconds)

result = linker.link(compustat, patents, left_id="gvkey", right_id="assignee_id", budget=2.00)
strict = result.relink(threshold=0.9, min_margin=0.3)     # no new calls
panel = strict.merged()                                   # both tables side by side, plus p
result.save("linkage/")                                   # links.csv, scores.csv, settings.json
```

`budget=0` allows cache hits and explicitly enabled exact shortcuts only; `budget=None` is unlimited. A positive
budget stops new requests at the observed cost, but calls already in flight can overshoot it.
Saved runs retain input fingerprints, blocker parameters, and model identities, including
cached answers. See [budget semantics and run provenance](docs/run-provenance.md).

### Relations, not only identity

By default the question put to Jev is "Record A and record B refer to the same firm", followed
by your definition. Some linkages are not identity. A news article is not a police incident,
but it can report one. `style="rule"` makes your definition the whole proposition: "Record A and
record B satisfy the following match rule. ..." The two tables then rarely share columns, so an
`on` item may be one-sided: `("text", None)` is shown on the left record only and
`(None, "neighborhood")` on the right record only.

```python
published_soon_after = jlink.block.window(          # article 0 to 3 days after the incident,
    ("published", "occurred"), between=(0, 3), unit="days")           # never before it
result = jlink.link(
    articles, incidents, style="rule",
    definition="Record A is a news article that reports the shooting incident in record B.",
    on=[("text", None), ("published", None),
        (None, "occurred"), (None, "neighborhood"), (None, "victim_age_group"), (None, "fatal")],
    blockers=[jlink.block.within(published_soon_after, "borough")],   # if both tables have one
    left_id="article_id", right_id="incident_id", how="many-to-one",
)
```

This is a sketch of a design, not a result: it has been run against a fake model in the tests
and never against Jev, so nothing is known yet about how well Jev judges this relation. The
date logic sits in blocking on purpose. `jlink.block.window` compares dates and numbers exactly
on your machine, so the rule need not ask Jev to do arithmetic, and only pairs inside the
window are paid for. `how="many-to-one"` lets several articles report one incident.

`entity` is optional under `style="rule"` because the question no longer names one.
`result.methods()` then describes a relation defined by your rule and does not say the records
are the same entity. Identity and rule answers are cached under different questions and never
mix. One-sided fields are shown to the judge only; a blocking pass needs a column on each side
and says so if given one. See [relation linking](docs/relation-linking.md) and
[windows on dates and numbers](docs/blocking.md#windows-on-dates-and-numbers).

### Dedupe: one table against itself

```python
events = jlink.dedupe(
    articles, style="rule", on=["text", "published"], id="article_id",
    definition="Both news articles report the same shooting incident.",
    blockers=[jlink.block.within(jlink.block.window("published", 3, unit="days"), "borough")],
)
events.clusters                          # id, cluster_id, cluster_size: one row per article
events.labeled()                         # the articles, with cluster_id appended
looser = events.recluster(threshold=0.4) # no new calls, like relink
```

`jlink.dedupe` blocks a table against itself, never pairs a record with itself, judges each
unordered pair once (earlier row as record A), and groups records into clusters. Joining every
pair above the threshold lets one wrong pair chain two unrelated groups together, so the default
is average linkage in which every pair between two clusters votes, a pair that blocking never
proposed counting as a non-match. That resists chaining and can split a true group that
blocking covered only in part; `unproposed="ignore"` and `linkage="components"` are the other
two rules, and switching is free. `report()` counts the high-probability pairs the rule left
apart, and `result.split_pairs()` lists them. [Dedupe](docs/dedupe.md) measures both failure
modes on synthetic scores, and states what is not known: nothing here measures Jev on a dedupe
task, or whether it answers (A, B) and (B, A) alike.

### Optional semantic candidate search

Install `uv add 'jlink[embeddings]'` (or `uv sync --extra embeddings` in this checkout), then
combine local embeddings with character matching:

```python
passes = [
    jlink.block.ngrams(("conm", "assignee"), k=10),
    jlink.block.embeddings(("conm", "assignee"), k=10,
        model="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
]
linker = jlink.Linker("firm", [("conm", "assignee")], definition="...", blockers=passes)
```

The embedding model runs locally; its weights download on first use. The union can recover
aliases that character similarity misses, at the cost of more candidate pairs. This is an
optional retrieval method, not a claim that any particular encoder beats other systems.
See [semantic retrieval and benchmark instructions](docs/hybrid-linkage.md).

## Checking the links

```python
sample = result.audit_sample(n=200)
sample.to_csv("audit.csv", index=False)      # fill in is_match with 1 or 0, then:

labeled = pd.read_csv("audit.csv", dtype={"left_id": str, "right_id": str})
ev = jlink.evaluate(labeled, mode="selected")
print(ev.summary())
print(ev.to_markdown())                      # a table for the appendix
```

The sample is stratified by probability, so the uncertain middle is covered and not only the
easy ends, and the estimates are weighted back to all judged pairs. Selected mode uses the
sample's saved membership in the final links; the default `mode="threshold"` instead assesses
`p >= threshold`, before assignment and margin filtering. Recall covers judged candidates
only, excluding unjudged pairs and true matches lost in blocking. Brier and calibration always
assess pair scores. You may leave labels blank, but keep those rows in the file: the labeled
pairs of each probability bin are reweighted to stand for the whole bin, so skipping most of
the unlikely pairs does not inflate recall. Blanks that fall on the hard pairs within a bin
can still bias the result, and a bin with no label at all leaves the estimates undefined. See
[evaluation modes and limitations](docs/evaluation.md) for blank labels, bootstrap assumptions
and the exported table contract.

For a local side-by-side review page with accept/reject/unsure decisions, durable history,
and offline recomputation, see [Local human review](docs/review.md). Start with
`jlink.create_review(result).write_html("review.html")` or `jev-link review --help`.

## Command line, Stata and R

```bash
jlink estimate compustat.dta patents.csv --on conm=assignee --on state
jlink link compustat.dta patents.csv --on conm=assignee --on state --entity firm \
      --define "A parent company and its subsidiary are different firms." \
      --left-id gvkey --right-id assignee_id --block ngrams:conm=assignee:10 --block initials:conm=assignee \
      -o links.csv --scores scores.csv --report report.md --save linkage/
jlink audit scores.csv --links links.csv --left compustat.dta --right patents.csv --on conm=assignee \
      --left-id gvkey --right-id assignee_id -n 200 -o audit.csv
jlink evaluate audit.csv --mode selected --markdown
```

`--save linkage/` writes what `result.save("linkage/")` writes: the links, every candidate score,
and `settings.json` with the question, blocking and provenance. `jlink review create linkage/ ...`,
`jlink.load` and a replication package all read that folder, so the review page is reachable
from the command line alone. Stata's `rundir()` and R's `run_dir =` forward it.
The save folder must be separate from input and output paths. Existing reserved run members
must be files; these checks run before blocking or judging.

`jlink dedupe firms.dta --on name --entity firm --id gvkey -o clusters.csv --scores scores.csv`
groups the records of one file, and `jlink cluster scores.csv --threshold 0.8 -o strict.csv`
regroups saved scores without API calls.

`--style rule` asks the relation in `--define` and makes `--entity` optional; `--on text=` and
`--on "=neighborhood"` are the one-sided fields (quote a leading `=`: zsh, the macOS default
shell, otherwise reads `=word` as a command lookup and stops before jlink runs). Stata's `style(rule)` and R's `style = "rule"`
forward the same option. `--block window:published=occurred:0..3d` is the date window above,
`--block window:year:1` a numeric one, and `--block within:borough:RULE` runs any rule inside
groups; `--date-format` reads dates that are not ISO 8601.

The Stata and R wrappers are single files in this repository, not part of the Python package: copy
`stata/jlink.ado` and `stata/jlink.sthlp` to your personal ado directory (`sysdir` shows it), and
`source()` `r/jlink.R`. Both call the installed command, so install the package first.

macOS ships a Java tool at `/usr/bin/jlink`. If `jlink` opens a Java prompt, use `jev-link`,
which is the same program, or `python -m jlink`.

```stata
use compustat, clear
jlink using patents.dta, on(conm=assignee state) entity(firm) leftid(gvkey) rightid(assignee_id) ///
    define("A parent company and its subsidiary are different firms.") saving(links.dta)
```

```r
source("r/jlink.R")
links <- jlink(compustat, patents, on = c("conm=assignee", "state"), entity = "firm",
               left_id = "gvkey", right_id = "assignee_id")
```

## Limits

- These are a model's judgments. Audit a sample before you rely on the links.
- Jev can only judge pairs that blocking proposes. If the names share nothing, add a pass that
  brings the pair together some other way (`exact` on state, year or industry).
- Jev reads the fields you give it and nothing else. It does not look anything up, and what it
  knows about firms stops at its training data.
- Repeated calls return nearly but not exactly the same probability (within 0.03 in our tests).
  Saved scores make results exact: keep `scores.csv` with your replication files.
- The default model ID is an alias for the latest Jev. Pin one with `model=` or `--model`
  (for example `typesafe/jev-1.13` on OpenRouter) and report it; `result.methods()` does.
- The Stata and R wrappers were run on macOS against Stata 19.5 and R 4.5.1. They do not
  support Windows yet, and they call `link` only: dedupe and the review page need the command
  line or Python.
- Rule-style relation linking, the window blocker and dedupe are tested against a fake model
  and synthetic data only. No benchmark in this README covers them.

## Development

```bash
uv sync --group bench && uv run pytest   # offline, no key; Stata and R tests skip if absent
```

`SPEC.md` is the design contract the modules were built against. `src/jlink/core.py` names the
providers jlink offers; the client, retries, answer cache and cost meter are the shared
[`jevkit-runtime`](https://github.com/keltokhy/jevkit-core), which [jgrep](https://github.com/keltokhy/jgrep)
and the other JevKit tools use too. jlink's provenance records are documented in
[run provenance](docs/run-provenance.md#backward-compatibility-and-unknown-provenance).

MIT license. The benchmark datasets keep their own terms; see `bench/FIRM_DATA.md`.

## Shared JevKit development

This tool uses [`jevkit-runtime`](https://github.com/keltokhy/jevkit-core), imported
as `jevkit_runtime`. Clone that repository beside this one as `../jevkit-core`, then
run `uv sync`. Core Python edits apply on the next invocation of this tool;
restart long-lived Python processes after editing.

The distribution name is `jevkit-runtime` because `jevkit-core` on PyPI belongs
to a different project. The runtime is [available on PyPI](https://pypi.org/project/jevkit-runtime/).
Use the sibling checkout for shared development, or `uv sync --no-sources` for a
standalone source checkout. Existing published versions of this tool are
unaffected by this source migration.

From the core checkout, `python scripts/dev.py setup`, `check`, and `wheel-check`
set up and validate all five consumers in separate environments.
CI checks out core tag `v0.3.0`. Prompts, question construction, and budget policies
remain in this repository; answer identity, the answer store, transport, and metering
are the runtime's. Runtime 0.2 keys and stores answers differently from 0.1, so a cache
written by an earlier version is re-asked once after upgrading.

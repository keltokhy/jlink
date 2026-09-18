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
agreement allows it. Only the `on` fields leave your machine; blocking runs locally.

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
- **The firm benchmark has a ceiling that no name-based method can pass.** A third of the NBER
  crosswalk's links are ownership facts with nothing in common in the names ("Homogeneous
  Metals Inc" to "United Technologies Corp"), so blocking can propose only 67% of true links.
  jlink found 93% of those. (A sliver of its error is the benchmark's: 11 Compustat names appear
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
beyond that size add an `exact` pass on a field such as state or year to split the problem.

Reproduce everything: `uv sync --group bench`, `uv run python bench/prepare.py`, then
`uv run python bench/live.py nber-firms --budget 1.00`. Baselines and data provenance are in
`bench/BASELINES.md` and `bench/FIRM_DATA.md`.

## How it works

1. **Block.** Comparing every record with every other is wasteful, so jlink first proposes
   candidate pairs on your machine, at no cost. By default each left record is paired with its
   ten nearest right records by character n-grams. Add passes for what n-grams miss:
   `jlink.block.initials("name")` pairs "IBM" with "International Business Machines", and
   `jlink.block.exact("state")` pairs everything within a state.
2. **Judge.** Each candidate pair goes to Jev with your rule. Pairs whose compared fields are all
   nonempty and individually equal after normalizing case, accents and punctuation are accepted
   without a call. Likelier pairs are judged first; cached scores remain available after the budget runs out.
3. **Resolve.** Choose links from the probabilities: `one-to-one` (the default; the best
   overall assignment with no record used twice), `many-to-one`, `one-to-many` or
   `many-to-many`, with a probability threshold and an optional margin over the runner-up.
   `result.relink(...)` tries other rules without paying again.
4. **Audit.** `result.audit_sample(200)` draws pairs across the probability range, links and
   non-links alike, with both records side by side. Label them in a spreadsheet, then
   `jlink.evaluate(labeled)` reports precision, recall and calibration.

```python
linker = jlink.Linker(
    entity="firm", on=[("conm", "assignee"), "state"], definition="...",
    blockers=[jlink.block.ngrams(("conm", "assignee"), k=10), jlink.block.initials(("conm", "assignee"))],
)
linker.estimate(compustat, patents, left_id="gvkey", right_id="assignee_id")   # blocking only, no API calls
# {'left': 4585, 'right': 2488, 'pairs': 45567, 'dollars': 0.6316, 'seconds': 227.8}
# (the real run on these data cost $0.62 and took 176 seconds)

result = linker.link(compustat, patents, left_id="gvkey", right_id="assignee_id", budget=2.00)
strict = result.relink(threshold=0.9, min_margin=0.3)     # no new calls
panel = strict.merged()                                   # both tables side by side, plus p
result.save("linkage/")                                   # links.csv, scores.csv, settings.json
```

`budget=0` allows exact matches and cache hits only; `budget=None` is unlimited. A positive
budget stops new requests at the observed cost, but calls already in flight can overshoot it.
Saved runs retain input fingerprints, blocker parameters, and model identities, including
cached answers. See [budget semantics and run provenance](docs/run-provenance.md).

## Checking the links

```python
sample = result.audit_sample(n=200)
sample.to_csv("audit.csv", index=False)      # fill in is_match with 1 or 0, then:

ev = jlink.evaluate(pd.read_csv("audit.csv"))
print(ev.summary())
print(ev.to_markdown())                      # a table for the appendix
```

The sample is stratified by probability, so the uncertain middle is covered and not only the
easy ends, and the estimates are weighted back to all judged pairs. Recall is measured among
candidate pairs. A true match that blocking never proposed is invisible to the audit, so widen
blocking (a larger `k`, an extra pass) and see whether new links appear.

## Command line, Stata and R

```bash
jlink estimate compustat.dta patents.csv --on conm=assignee --on state
jlink link compustat.dta patents.csv --on conm=assignee --on state --entity firm \
      --define "A parent company and its subsidiary are different firms." \
      --left-id gvkey --right-id assignee_id --block ngrams:conm=assignee:10 --block initials:conm=assignee \
      -o links.csv --scores scores.csv --report report.md
jlink audit scores.csv --left compustat.dta --right patents.csv --on conm=assignee -n 200 -o audit.csv
jlink evaluate audit.csv --markdown
```

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
  support Windows yet.

## Development

```bash
uv sync --group bench && uv run pytest   # 279 tests, offline, no key; Stata and R tests skip if absent
```

`SPEC.md` is the design contract the modules were built against. `src/jlink/core.py` is the
Jev client (two backends, retries, cache, cost meter), historically shared with
[jgrep](https://github.com/keltokhy/jgrep), which is grep with a description in place of a
pattern. jlink's additive cache/provenance extensions are documented in
[run provenance](docs/run-provenance.md#backward-compatibility-and-unknown-provenance).

MIT license. The benchmark datasets keep their own terms; see `bench/FIRM_DATA.md`.

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
returns a probability in about a fifth of a second for about a thousandth of a cent, so
comparing a hundred thousand candidate pairs costs about a dollar and a half and takes about
ten minutes.

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
pip install git+https://github.com/keltokhy/jlink
```

You need a key for one of two APIs. With keys for both, jlink uses TypeSafe's.

| API | Environment variable | Get a key |
|---|---|---|
| TypeSafe | `TYPESAFE_API_KEY` | [console.typesafe.ai](https://console.typesafe.ai/settings/keys) |
| OpenRouter | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |

A key can also live in `~/.config/jev/typesafe.key` or `~/.config/jev/openrouter.key`.

## How it works

1. **Block.** Comparing every record with every other is wasteful, so jlink first proposes
   candidate pairs on your machine, at no cost. By default each left record is paired with its
   ten nearest right records by character n-grams. Add passes for what n-grams miss:
   `jlink.block.initials("name")` pairs "IBM" with "International Business Machines", and
   `jlink.block.exact("state")` pairs everything within a state.
2. **Judge.** Each candidate pair goes to Jev with your rule. Pairs whose fields are identical
   after normalizing case, accents and punctuation are accepted without a call. Likelier pairs
   are judged first, so if a budget runs out it is the long shots that go unjudged.
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
linker.estimate(compustat, patents, left_id="gvkey", right_id="assignee_id")
# {'left': 9000, 'right': 40000, 'pairs': 93112, 'dollars': 1.29, 'seconds': 465.6}

result = linker.link(compustat, patents, left_id="gvkey", right_id="assignee_id", budget=2.00)
strict = result.relink(threshold=0.9, min_margin=0.3)     # no new calls
panel = strict.merged()                                   # both tables side by side, plus p
result.save("linkage/")                                   # links.csv, scores.csv, settings.json
```

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

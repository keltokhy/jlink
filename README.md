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
benchmarks below, 171,354 pairs cost $2.95 in total, at about 250 pairs a second. With no
labels, jlink beats LinkTransformer's zero-shot models on eleven of twelve standard benchmarks
and its fine-tuned ones on most of the product data ([below](#against-linktransformer)).

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

`--api diffusiongemma`, `--api laya` and `--api gliner`, or `JEV_API` set to one of those names for the Python, Stata
and R entry points, send the same pair questions to a System One server on your own machine, an
[OpenJev](https://github.com/razorback16/openjev), [laya-mlx](https://github.com/mizorewww/laya-mlx) or [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)
process that you run separately. They are never chosen automatically, need no key, and count as $0
in the cost meter and a saved run's settings unless `JEV_PRICE_PER_MTOK` is set. The report still
counts a local model's answers "by Jev" and `scores.csv` marks them `source=jev`; the report's
`Model:` line, the `model` and `provider` columns of `scores.csv` and `settings.json` name the
model that answered. The runtime's [DiffusionGemma](https://github.com/keltokhy/jevkit-core/blob/main/docs/diffusiongemma.md),
[Laya](https://github.com/keltokhy/jevkit-core/blob/main/docs/laya.md) and [GLiNER](https://github.com/keltokhy/jevkit-core/blob/main/docs/gliner.md) guides explain the
setup; keep concurrency low while a local model warms up.

[On a local server](#on-a-local-server) compares both with Jev on the five benchmarks, and the
[local-model results](https://github.com/keltokhy/jlink/blob/main/docs/benchmarks/local-models-2026-09-22.md) have the full record.

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
- **The firm run was limited by its candidate search.** The forward top-10 n-gram pass, which
  was the default until 0.4.0, proposed 67% of known links, and jlink found 93% of the true links it proposed. The 67% is
  measured blocking recall, not a ceiling for name-based methods, though ownership links such as
  "Homogeneous Metals Inc" to "United Technologies Corp" can be difficult to retrieve from names.
  jlink now also searches in reverse by default, from each right record to its nearest left
  records, and the report counts records that no pass paired with anything.
  [Candidate search](https://github.com/keltokhy/jlink/blob/main/docs/blocking.md) describes reverse search, larger `k` and their pair-count
  cost. A sliver of the error is the benchmark's: 11 Compustat names appear under two IDs, which
  accounts for 14 of jlink's 331 false links.
- **Amazon to Google is hard for everyone**, because listings differ in version and edition
  details that the records often omit.

The firm rule was revised once, after reading the errors of a 300-record pilot, as a user
would: the first draft did not tell Jev that `CPY` means Company in these data. That changed F1
by two points. The final rule is two sentences: *"Ignore legal-form suffixes (Inc, Corp, Co,
CPY, Ltd, GmbH, N V). A subsidiary or division counts as its parent company."*

**Are the probabilities calibrated?** Roughly, and it depends on the data. On the firm
benchmark, pairs scored above 0.8 were true matches 95 to 98% of the time and pairs scored
below 0.2 were true 0.1% of the time, but the 0.5 to 0.8 band was overconfident (mean 0.64,
true 39% of the time). On FEBRL4 Jev was underconfident: pairs in the 0.2 to 0.5 band were true
matches 66% of the time. Treat `p` as a strong ranking and check the middle band with an audit
sample before using it as a literal probability.

**Speed.** The firm run judged 45,567 pairs in 176 seconds (259 pairs a second, 64 calls in
flight, median latency 214 ms, one retry). Blocking 100,000 by 100,000 records takes about four
minutes and 1 GB on an M3 Ultra. Blocking time grows roughly with the square of the data, so
beyond that size a reliable shared field such as state or year can restrict the search:
`jlink.block.within(jlink.block.ngrams("name"), "state")` searches separately inside each group.
A separate `exact("state")` pass is different: it adds pairs to the union and does not split the
existing search. Grouping can lose matches when group values disagree or are missing; see
[grouping, reverse search, and pair limits](https://github.com/keltokhy/jlink/blob/main/docs/blocking.md).

Reproduce everything: `uv sync --group bench`, `uv run python bench/prepare.py`, then
`uv run python bench/live.py nber-firms --budget 1.00`. Baselines and data provenance are in
`bench/BASELINES.md` and `bench/FIRM_DATA.md`.

### On a local server

On 2026-09-22 the two [local servers](#local-servers-experimental), on an Apple M3 Ultra, judged
the candidate pairs of 300 left records from each benchmark, the same pairs Jev judged in the full
runs above. The [local-model results](https://github.com/keltokhy/jlink/blob/main/docs/benchmarks/local-models-2026-09-22.md) have every
number, dataset by dataset.

| F1, 300 left records | Jev 1.13 (OpenRouter) | DiffusionGemma (`openjev-0.1`, local) | Laya (`laya-421m`, local) | LinkTransformer, zero-shot (local) |
|---|---:|---:|---:|---:|
| Firms: NBER patent assignees to Compustat | 0.71 | 0.69 | 0.22 | 0.67 |
| Publications: DBLP to ACM | 0.99 | 0.98 | 0.69 | 0.97 |
| Products: Abt to Buy | 0.94 | 0.91 | 0.18 | 0.32 |
| Software: Amazon to Google | 0.67 | 0.65 | 0.16 | 0.40 |
| People: FEBRL4, synthetic typos | 0.97 | 0.84 | 0.69 | 0.91 |
| Wall-clock for all five, API cost | | 82 min, $0 | 6 min, $0 | 7 s, $0 |

DiffusionGemma comes within 0.03 of Jev's F1 on firms, publications, products and software and
decides 94 to 99% of candidate pairs the way Jev does, at 14 to 18 minutes for each 3,000 pairs.
On FEBRL4 it is stricter about typos: at precision 1.0 its recall is 0.72, against Jev's 0.93.
Laya puts 90 to 93% of the candidate pairs for firms, products and software at 0.5 or above and is
not a substitute for Jev in record linkage. LinkTransformer's pretrained models, with a threshold
tuned on labels, match DiffusionGemma within 0.03 on firms and publications, beat it on FEBRL4 and
fall far behind on the two product sets, where a cosine cutoff keeps many near-duplicates.
Fine-tuned on labels, it links FEBRL4 perfectly and still trails Jev on the other four under the
same link rules ([Against LinkTransformer](#against-linktransformer)). Jev remains the default;
DiffusionGemma is the option when records may not leave your machine.

### Against LinkTransformer

[LinkTransformer](https://github.com/dell-research-harvard/linktransformer) (Arora and Dell) is
the embedding linker economists reach for: it encodes each record with a sentence transformer,
links nearest neighbours by cosine similarity, runs locally and costs nothing per pair. On
2026-09-20 jlink was compared with it three ways. jlink used its default threshold of 0.5
everywhere and saw no labels; LinkTransformer ran locally, pretrained and fine-tuned.

**With no labels, jlink beats LinkTransformer's zero-shot models on eleven of the twelve
DeepMatcher benchmarks** and is level on the twelfth (Beer, where 14 matches make one pair worth
three points of F1). These are the 25,235 labelled test pairs behind Table A-5 of the
LinkTransformer paper ([arXiv:2309.00789](https://arxiv.org/abs/2309.00789)), each judged once at
0.5 with a one- or two-sentence rule, for $0.46. Fine-tuned on each benchmark's training
labels, LinkTransformer pulls ahead on eight rows, by up to seven points; jlink stays ahead on
four, three of them product data, where it also beats Ditto, a fine-tuned language-model
matcher. F1 in percent; the published columns are as printed in the paper, and bold marks where
jlink beats the fine-tuned LinkTransformer.

| DeepMatcher test set | jlink, no labels | LinkTransformer, zero-shot | LinkTransformer, fine-tuned | Ditto, fine-tuned |
|---|---:|---:|---:|---:|
| Products: Abt-Buy | **91.5** | 28.8 | 84.0 | 88.9 |
| Products: Walmart-Amazon | **89.5** | 45.0 | 73.8 | 85.8 |
| Products: Walmart-Amazon, dirty | **85.4** | 45.0 | 71.0 | 82.6 |
| Songs: iTunes-Amazon, dirty | **92.0** | 68.8 | 84.0 | 92.9 |
| Software: Amazon-Google | 68.6 | 47.1 | 74.0 | 74.1 |
| Songs: iTunes-Amazon | 82.6 | 60.6 | 90.0 | 92.3 |
| Beers: BeerAdvo-RateBeer | 83.3 | 83.4 | 90.3 | 84.6 |
| Restaurants: Fodors-Zagats | 97.7 | 75.0 | 98.0 | 98.1 |
| Publications: DBLP-ACM | 96.1 | 95.0 | 98.0 | 99.0 |
| Publications: DBLP-ACM, dirty | 95.2 | 89.8 | 98.0 | 98.9 |
| Publications: DBLP-Scholar | 90.6 | 80.0 | 92.0 | 95.6 |
| Publications: DBLP-Scholar, dirty | 89.3 | 87.5 | 92.6 | 95.4 |

The publication rows are pairwise classification: every pair is judged alone, two versions of
one paper look like a match, and jlink's precision there is 0.83 to 0.93. Linked end to end with
one-to-one assignment, the same judgments score 0.9965 on DBLP to ACM (above).

**On firm names, jlink beats LinkTransformer's purpose-built company model.** On the NBER to
Compustat split, each method was allowed the 2,537 links jlink made and scored on listed matches
recovered: jlink 2,034, LinkTransformer's Wikidata company model 1,960, LinkTransformer
fine-tuned on the development labels 1,931, character TF-IDF 1,889. The lead holds on the
records outside jlink's rule-wording pilot (1,912 against 1,841). The product splits are the
closer contest. With development labels to fine-tune on and a threshold tuned on them,
LinkTransformer edges past jlink on Abt to Buy (0.954 against 0.942 with one-to-one assignment,
where character TF-IDF also reaches 0.951) and on Amazon to Google (0.736 against 0.721), and on
FEBRL4 typos it reaches 0.999 where jlink stops at 0.960, as the string baseline above already
does.

**Only jlink changes its answer when the definition changes.** Two small fixtures, hand-labelled
from public filings by this repository's author (HP, IBM and Kyndryl, Meta, Kellogg, Kraft,
Alphabet and others), ask the same pairs under four definitions: legal entity, corporate group at
the record dates, physical site, operating business. jlink answered 30 of 33 and 39 of 40 test
decisions correctly. A similarity score is one number per pair whatever the definition, so no
threshold on any score can exceed 23 of 33 and 30 of 40. LinkTransformer's company model and
`all-mpnet-base-v2`, with a threshold chosen on development pairs, reached 18 and 23. Pairs
within one corporate family are correlated, so the gap is the finding, not the decimals.

These are public benchmarks that Jev, like LinkTransformer's encoders, may have met in training,
and each comparison is one split with one seed, so differences of a point or two are noise.
Reproduce it with `uv run --group bench python -m bench.deepmatcher_pairs --live --budget 0.60`
(the Table A-5 pairs) and `bench/lt_compare.py` with `bench/linktransformer_scores.py` (the
splits). LinkTransformer is GPL-3.0 and runs in its own environment; jlink never imports it.

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
   equalities without a call. Likelier pairs are judged first; cached scores remain available
   after the budget runs out.
3. **Resolve.** Choose links from the probabilities: `one-to-one` (the default; the best
   overall assignment with no record used twice), `many-to-one`, `one-to-many` or
   `many-to-many`, with a probability threshold and an optional margin over the runner-up.
   `result.relink(...)` tries other rules without paying again.
4. **Audit.** `result.audit_sample(200)` draws pairs across the probability range, links and
   non-links alike, with both records side by side. Label them in a spreadsheet, then
   `jlink.evaluate(labeled, mode="selected")` evaluates the delivered links and pair-score
   calibration.

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

`budget=0` allows cache hits and explicitly enabled exact shortcuts only; `budget=None` is
unlimited. A positive budget stops new requests at the observed cost, but calls already in
flight can overshoot it. Saved runs retain input fingerprints, blocker parameters, and model
identities, including cached answers. See [budget semantics and run provenance](https://github.com/keltokhy/jlink/blob/main/docs/run-provenance.md).

When the budget runs out or some calls fail, `result.resume(budget=5.00)` (for a loaded run,
`jlink.load("linkage/").resume(compustat, patents)`, or `jev-link resume linkage/ LEFT RIGHT`)
judges only the unjudged and failed pairs and chooses links again. It keeps the saved candidate
pairs, question and model and skips blocking. It does not save API spend over running `link`
again: the answer cache on your machine already makes pairs it answered free. It matters when
blocking is slow, when the cache is gone or was off, or when the candidate pairs must stay exactly
those of the saved run.

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
and says so if given one. See [relation linking](https://github.com/keltokhy/jlink/blob/main/docs/relation-linking.md) and
[windows on dates and numbers](https://github.com/keltokhy/jlink/blob/main/docs/blocking.md#windows-on-dates-and-numbers).

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
apart, and `result.split_pairs()` lists them. [Dedupe](https://github.com/keltokhy/jlink/blob/main/docs/dedupe.md) measures both failure
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
See [semantic retrieval and benchmark instructions](https://github.com/keltokhy/jlink/blob/main/docs/hybrid-linkage.md).

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
[evaluation modes and limitations](https://github.com/keltokhy/jlink/blob/main/docs/evaluation.md) for blank labels, bootstrap assumptions
and the exported table contract.

For a local side-by-side review page with accept/reject/unsure decisions, durable history,
and offline recomputation, see [Local human review](https://github.com/keltokhy/jlink/blob/main/docs/review.md). Start with
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
from the command line alone. Stata's `rundir()` and R's `run_dir =` forward it. The save folder
must be separate from input and output paths, and existing reserved run members must be files;
both are checked before blocking or judging.

`jlink dedupe firms.dta --on name --entity firm --id gvkey -o clusters.csv --scores scores.csv`
groups the records of one file, and `jlink cluster scores.csv --threshold 0.8 -o strict.csv`
regroups saved scores without API calls.

`--style rule` asks the relation in `--define` and makes `--entity` optional; `--on text=` and
`--on "=neighborhood"` are the one-sided fields (quote a leading `=`: zsh, the macOS default
shell, otherwise reads `=word` as a command lookup and stops before jlink runs). Stata's
`style(rule)` and R's `style = "rule"` forward the same option.
`--block window:published=occurred:0..3d` is the date window above, `--block window:year:1` a
numeric one, and `--block within:borough:RULE` runs any rule inside groups; `--date-format`
reads dates that are not ISO 8601.

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
[run provenance](https://github.com/keltokhy/jlink/blob/main/docs/run-provenance.md#backward-compatibility-and-unknown-provenance).

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

From the core checkout, `python scripts/dev.py setup`, `check`, and `wheel-check` set up and
validate all five consumers in separate environments. CI checks out core tag `v0.3.1`. Prompts,
question construction, and budget policies remain in this repository; answer identity, the
answer store, transport, and metering are the runtime's. Runtime 0.2 keys and stores answers
differently from 0.1, so a cache written by an earlier version is re-asked once after upgrading.

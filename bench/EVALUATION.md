# Offline evaluation protocol

`heldout.py` separates three questions: which labeled matches did blocking retrieve, how well
did pair scores classify those candidates, and which links survived the actual assignment
rule? `rule_sensitivity.py` tests whether changing the English definition changes the right
judgments. Neither program has a live-call option. Results and limitations from the first
run are in [EVIDENCE.md](EVIDENCE.md).

## Reproduce without API access

Use prepared data and cached runs already on disk. No downloads, credential reads, or model
requests are needed after installing the optional benchmark dependencies:

```sh
uv sync --group bench
uv run python -m bench.heldout febrl4 \
  --data-dir bench/data \
  --cached-run bench/out/live/febrl4 \
  --config bench/configs/cached.json --ecm --out bench/out/replay
uv run python -m bench.heldout febrl4 \
  --data-dir bench/data \
  --cached-run bench/out/live/febrl4 \
  --config bench/configs/febrl-unmatched.json --ecm --out bench/out/unmatched
uv run python -m bench.rule_sensitivity --out bench/out/rules
uv run pytest
```

The checked-in report is collected from the final run directories with:

```sh
uv run python -m bench.evidence_report \
  --cached-dir bench/out/final-cached --unmatched-dir bench/out/final-unmatched \
  --rules-dir bench/out/final-rules --legacy-dir bench/out \
  --json bench/evidence/2026-09-18.json --markdown bench/EVIDENCE.md
```

Change the dataset and cached-run directory together to replay the other four datasets.
Omit `--cached-run` for an entirely local baseline comparison. Input directories are read
only. Choose a fresh output directory for each experiment: existing artifacts are protected
from overwrite. Data remain subject to their original terms; no NBER source records are
redistributed by the checked-in evidence reports.

## Partitions and labels

The deterministic split builds a graph of left/right record IDs connected by truth pairs.
It also groups identical nonempty signatures across all declared comparison fields, so
obvious duplicate records do not straddle partitions. Component membership is hashed, groups
are sorted by a seed-dependent hash, and 30% of groups go to development (seed 1729).
Both record universes are restricted to their partition. `split.csv` records every ID,
component and partition. Many-to-many truth stays together. A single connected entity or an
empty partition raises an error; it does not silently fall back to a leaking record split.

This is disjointness in **observed entities**. Incomplete labels can miss aliases and corporate
relationships. Neither these partitions nor any new threshold sweep can erase a model's
training exposure or earlier benchmark inspection. In particular, the historical NBER prompt
was revised after a 300-record pilot. Cached replays are retrospective diagnostics, not a
new untouched prospective test. A single fixed split provides no uncertainty interval.

Label policy is explicit:

- FEBRL has synthetic underlying identity labels. To test abstention, the optional transform
  chooses two disjoint sets of known one-to-one identities in each partition. It removes the
  right counterpart of one set and the left counterpart of the other. At fraction 0.1, each
  set contains 10% of that partition's original entities. Retained records are then truly
  unmatched **within that reduced universe**. Removed counterpart IDs are recorded. The
  transform refuses incomplete labels and is enabled only for FEBRL.
- Leipzig uses its supplied `perfectMapping` as a closed-world benchmark. Unlisted pairs are
  benchmark negatives under that convention, not independently verified claims. No new
  unmatched labels are manufactured from mapping omissions. Existing unlinked records remain
  in the search universe.
- NBER is a positive-only historical association crosswalk. The new harness reports recall
  of listed positives and counts unlisted predictions as **unknown**. Precision, F1, Brier,
  calibration and false-positive counts are unavailable. It disables F1 threshold tuning and
  uses a clearly labeled fixed 0.5 similarity cutoff for non-exact string methods. No claim
  that the methods are comparable on gold-standard NBER F1 follows from this run.

## Shared candidates and final assignment

All methods within a run score the same pairs and use the same `how`, threshold convention
(`>=`), and `min_margin=None`. Exact matching uses all nonempty normalized comparison fields;
Jaro-Winkler scores concatenated normalized fields on the shared pairs; TF-IDF uses the
candidate table's character 2–4-gram cosine. Neither similarity baseline privately reduces
the table to one best right record. This differs deliberately from the historical oracle
methods in [BASELINES.md](BASELINES.md).

`candidate_source="cached"` filters the original candidate pool to each partition. Its
cosines retain the original full-corpus fit and its top-k choices are **not recomputed** after
removing cross-partition competitors. This is a matched comparison of existing cached
judgments, not a fresh blocking experiment. `candidate_source="configured"` recomputes
blocking separately within each partition. A configured pair absent from the cache stays
unjudged for the cached model; it is not given a fabricated score. Judged counts and missing
scores are reported explicitly. Legacy caches have no binding to dataset hashes: the loader
checks IDs and comparison fields, records current input hashes, and discloses that remaining
provenance limitation.

Default cardinality comes from the dataset: one-to-one for people/publications,
many-to-many for products, many-to-one for NBER. A JSON `how` override applies to **every**
method. Consequently the product baseline results should not be compared as if they were the
older one-best-right oracle results. Resolver fallback warnings are saved in every threshold
trial and final metric, including the existing greedy fallback above 2,000 nodes per side.

The three metric groups are:

1. `candidates`: candidate count, listed truth count, retrieved truth count, and recall before
   any judgment or assignment.
2. `judge_conditional_on_candidates`: pair-threshold decisions before resolution, with recall
   denominator restricted to retrieved truth. Unjudged positives still count as missed. Brier
   and calibration use only scored pairs, only exhaustive benchmark labels, and only methods
   producing probabilities. They are omitted for raw string similarities. `by_source` separates
   legacy `exact` shortcuts from actual `jev` judgments and missing decisions.
3. `final_assignment`: actual resolved membership scored against all truth in the partition,
   including truths missed in blocking. For the explicit unmatched experiment it also reports
   distinct records falsely linked and correct abstention by side. This uses actual selection,
   consistent with the audit API's `mode="selected"`; it does not infer links by
   thresholding probabilities again.

Undefined candidate recall and explicit-unmatched rates are null. The legacy pair scorer's
empty-denominator precision/recall/F1 convention is zero for complete labels. Count denominators
are always included, so a zero rate cannot be mistaken for evidence from nonempty gold.

## Development-only thresholds and optional probabilistic baseline

For complete benchmark labels, select Jaro-Winkler, TF-IDF and ECM thresholds by final-assignment
F1 on development data. Use the prespecified grid 0, 0.05, ..., 1 plus a reject-all action;
keep score ties together and prefer fewer links when F1 ties. Save every trial. The selected
threshold is then frozen for test. Exact stays at 1. Cached Jev stays at its historical 0.5,
with no fresh tuning on either partition. These grid-selected baseline scores are not
in-sample oracle upper bounds.

The optional `--ecm` method uses `recordlinkage.ECMClassifier`, a probabilistic Fellegi–Sunter
method documented by the [Python Record Linkage Toolkit](https://recordlinkage.readthedocs.io/en/latest/ref-classifiers.html#recordlinkage.ECMClassifier).
It fits an unsupervised mixture on development candidates; no test labels or test features are
used to estimate its parameters. Each field contributes a separate binary Jaro-Winkler >= 0.9
agreement (missing values disagree). Constant development fields are excluded and named in the
report; fewer than two varying fields makes the method inapplicable. Initialization is `jaro`,
maximum iterations 100, tolerance 1e-5. Fitted m/u probabilities and prior are saved.

This is a stronger *method class* than a single concatenated similarity, not a promise of
higher accuracy. Its conditional-independence approximation and generic binary features are
particularly restrictive for correlated descriptions, manufacturer names, and prices.
The measured product results are weak. Field-specific comparators or a richer Splink model
would need a prespecified development-only experiment. Splink's own [project documentation](https://github.com/moj-analytical-services/splink)
recommends multiple informative columns rather than a single bag-of-words field. This is also
why no multi-field probabilistic result is claimed for name-only NBER. `recordlinkage` stays in
the optional `bench` dependency group; runtime dependencies are unchanged.

## Rule-sensitivity fixture and response cache

[fixtures/rule_sensitivity.json](fixtures/rule_sensitivity.json) declares 18 invented pairs in
three disjoint synthetic corporate worlds: one development world (6 pairs), two test worlds
(12 pairs). Each world has an alias, rename, parent/subsidiary, sister-company, unrelated
namesake and divestiture case. Parent/subsidiary and sister-company labels change from false
under legal identity to true under corporate family; the remaining cases retain their labels.
Relationship and date facts are visible in the input records. This is controlled instruction
following, not a test of model knowledge of real ownership or a representative firm sample.

The [JSON Schema](fixtures/rule_sensitivity.schema.json) defines the fixture format; the loader
also enforces unique IDs, group-disjoint splits and legal-entity/family label consistency.
Each pair has a source kind (`synthetic` or `primary_source`), reference, and rationale. The
bundled pairs use explicit synthetic-world references, never invented real-source citations.
Curated real examples can be added only with primary-source facts and an as-of date when needed.

The request manifest saves the exact model, state and questions sent by the current Jev
protocol, the definition, provider settings, fixture hash, split and request hash. The fixed
model ID is `typesafe/jev-1.13-20260917`; temperature is not sent. Exact shortcuts are disabled,
so identical names can still receive different judgments under different definitions.

An external, separately authorized live runner can populate `responses.jsonl`; this harness
only reads it. Each row has:

```json
{"request_sha256":"<from requests.jsonl>","model":"typesafe/jev-1.13-20260917",
 "recorded_at":"<UTC timestamp>","evidence_kind":"live_model",
 "response":{"match":{"noul":0.82}},
 "provider_response":{"answers":{"match":{"noul":0.82}},"model":"<returned model>","usage":{}}}
```

Preserve the actual full provider response; do not construct the example placeholder as
measurement. Fakes use `evidence_kind="offline_fake"`. The validator rejects stale request
hashes, duplicate responses, model mismatches, missing live provider responses, invalid
probabilities and mixed fake/live evidence. It copies cache bytes unchanged into the output.
Partial caches report missing responses and score only available decisions; paired sensitivity
requires both rule responses. Metrics separate per-rule accuracy, both-correct judgments on
changing cases, correct probability direction, and both-correct judgments on invariant cases.
Threshold 0.5 is fixed. With no suitable cache, status is `live_model_pending` and all measured
accuracy fields are null. Passing fake tests is not model performance.

## Extending blocking and preserving evidence

Configuration exposes `blockers=[{"kind":"ngrams","columns":["name"],"kwargs":{"k":10}}]`.
Current factories are `ngrams`, `exact`, `initials`. `evaluation.propose(..., registry={...})`
accepts a factory returning a public `Blocker`, so new retrieval experiments need no changes to
the metric code. `candidate.attrs['blocking']`, when provided by the blocker, is saved.

`kwargs={"k":10,"reverse":true}` forms a reverse
pass alongside a forward pass. A registry factory can wrap a pass with
`block.within(block.ngrams(*columns, **kwargs), "state", missing="drop")`. Nested serialization
uses `Blocker.to_config()`. The integrated registry supports these configurations; the
measured results in this report still use the recorded historical candidate pools.

Each held-out run saves source/dataset hashes, source metadata, split definitions and CSV,
explicit unmatched/removal labels, candidates, all method scores, final links, settings,
software versions, code hashes, optional model parameters and artifact hashes. Cached-run
settings and hashes are included. Legacy saved scores preserve parsed probabilities but do
not include full raw provider responses; that gap is disclosed, not reconstructed as fact.
Keep the output directory with replication materials. The checked-in evidence JSON records
summaries and hashes; local CSVs can be regenerated from the frozen inputs. Different resolver
or blocker code after integration may legitimately change the replay and must get a new report.

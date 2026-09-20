# What to improve after hybrid retrieval

The next experiment should improve the information and decision given to the matcher.
The current evidence does not support making larger hybrid candidate pools the default.

## What the saved errors show

An offline audit of the existing Abt–Buy hybrid test run separates its 50 missed gold links:

| Cause | Missed links |
|---|---:|
| Correct pair absent from candidates | 3 |
| Correct pair retrieved but rejected by Jev at 0.5 | 47 |
| Correct pair retrieved but not judged | 0 |

There are also 67 false positives under the benchmark's complete-label convention.
Eight groups, comprising 34 candidate pairs, have identical records in every field actually
shown to the pair judge but mixed positive/negative gold labels. Nine such groups occur in
development. This is an information limitation relative to those inputs, not proof that the
gold labels are wrong. Record IDs are not matching evidence. A larger model cannot reliably
distinguish identical inputs without additional evidence.

Development examples include a generic microwave listing without a model number, paired
against stainless-steel, black and white versions. Multiple right IDs have identical judge
inputs while the supplied mapping associates them with different variants. Some cases need
more source attributes or an explicit unresolved decision.

As a cheap control, select a threshold using development F1 over 0.05–0.95 in steps of 0.05,
then freeze it for test. Both lexical and hybrid select 0.35:

| Retrieval | Test F1 at 0.5 | Test F1 at development-selected threshold |
|---|---:|---:|
| Lexical | 0.9238 | 0.9219 |
| Hybrid | 0.9245 | 0.9220 |

Threshold tuning slightly hurts generalization here. Do not adopt it as an improvement.
The audit spends nothing and changes no production defaults. This is exploratory analysis
of a previously inspected test partition, not fresh evidence from an untouched test.

Firm retrieval has a different limitation. The NBER development gold contains name pairs
such as `CHEMPRENE INC` → `WITCO CORP` and `CITIES SERVICE CPY` → `OCCIDENTAL PETROLEUM CORP`.
These are historical crosswalk associations; this audit does not independently verify their
ownership or dates. Name similarity alone does not supply the historical relationship.
NBER's unlisted pairs remain unknown and must not be used as negative labels in this audit.

## Proposed experiments, not implemented performance claims

1. **Give the judge the actual relation and usable evidence.** Separate legal identity,
   corporate control at a date, and product-model equality. For firms, ingest independently
   sourced aliases and dated ownership links with provenance. For products, retain model,
   size and variant attributes when the source supplies them; mark missing evidence explicitly.
   Never populate these fields from held-out crosswalk labels or invented model facts.
2. **Show competing candidates together for ambiguous cases.** Test whether comparing a
   short candidate list resolves mistakes that independent yes/no questions make. Allow no
   match, multiple matches when the relation permits them, and insufficient evidence. Evaluate
   order sensitivity and record the complete list in request/cache provenance. Existing
   assignment constraints already operate on pair scores; this experiment changes the
   information available while judging.
3. **Escalate selectively.** Use a stronger judge or human review on candidate disagreement,
   missing distinguishing attributes and close alternatives. Report accuracy and automatic
   coverage together so abstention cannot masquerade as a recall improvement. Compare equal
   total token/dollar budgets, not request counts alone.
4. **Learn from reviewed errors.** Train a small matcher or retrieval model on reviewed
   examples and difficult nonmatches, preserving entity-disjoint evaluation. Start a new
   external test after developing these changes on the already-inspected benchmarks.

Candidate interaction is an established research direction: [Wang et al., COLING 2025](
https://aclanthology.org/2025.coling-main.8/) compare matching, comparing and selecting strategies.
Their results motivate an experiment; they do not establish a gain for jlink. Likewise, the
[historical entity-linking paper](https://aclanthology.org/2024.emnlp-main.355/) trains contrastive
models with difficult negatives and contextual evidence. A generic embedding swap does not
reproduce that training setup or establish superiority over it.

## Reproduce the product audit

```sh
.venv/bin/python -m bench.diagnose abt-buy \
  --run bench/out/hybrid-20260920-final/abt-buy \
  --out /tmp/jlink-product-diagnosis.json
```

The saved input files can be restored from the existing
`bench/evidence/hybrid-2026-09-20/runs.tar.gz` archive. The new report is retained at
[`bench/evidence/diagnosis-2026-09-20/products.json`](../bench/evidence/diagnosis-2026-09-20/products.json),
with input/data/code hashes, all threshold trials and IDs for mixed-label groups.

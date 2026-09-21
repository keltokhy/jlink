# Evaluating pair scores and final links

An audit can answer two different questions. Choose the mode explicitly when reporting
results:

| Mode | Prediction used for precision, recall and F1 | Question |
|:--|:--|:--|
| `threshold` (default) | `p >= threshold` | How well do the pair scores classify candidate pairs? |
| `selected` | Saved `selected` membership in the actual final links | How accurate are the links delivered after assignment and margin filtering? |

For example, suppose A–X has probability 0.9 and is a true match, while A–Y has probability
0.8 and is not. One-to-one assignment delivers A–X alone. Threshold evaluation at 0.5 has
precision 0.5; final-link evaluation has precision 1.0. Both have recall 1.0 within these
judged candidates. Many-to-one, one-to-many, one-to-one and margin filtering can all make
the predictions differ from a probability cutoff.

## Python

```python
import pandas as pd
import jlink

strict = result.relink(threshold=0.8, min_margin=0.2)
sample = strict.audit_sample(n=200, seed=42)
sample.to_csv("audit.csv", index=False)
# Fill is_match with 1 or 0, retaining every other column and every sampled row.
labeled = pd.read_csv("audit.csv", dtype={"left_id": str, "right_id": str})

final = jlink.evaluate(labeled, mode="selected", n_boot=2000, seed=42)
print(final.summary())
print(final.to_markdown())

pair_scores = jlink.evaluate(labeled, mode="threshold", threshold=0.8)
```

`Result.audit_sample()` derives `selected` from actual membership in `Result.links`, not
from the threshold, settings or a new assignment on the sample. This also works after
`relink()` and with a saved result loaded using `jlink.load()`.

For separate tables, use `jlink.audit_sample(scores, links=links, n=200, seed=42)` with the
complete final links table. IDs must have matching values and types. Duplicate pairs,
links absent from scores and links with missing probabilities in scores are errors. An
empty links table is valid and records that no pair was selected. Omitting `links` means
selection is unknown: the sample has no `selected` column, even if the scores input happens
to contain a column with that name. Scores alone cannot establish assignment or margin
decisions.

Selected mode never reapplies `threshold`; even an explicitly supplied threshold has no
effect on its predictions. `Evaluation.mode` identifies the mode, and
`Evaluation.threshold` is `None` in selected mode. To evaluate another decision rule,
relink the complete scores and derive membership from that result. Do not resolve just
the audit sample: absent competitors can change assignment and margins. Keep the final
links and run settings alongside the labeled audit to identify which result was evaluated.

## Command line

```bash
jev-link audit scores.csv --links links.csv -n 200 -o audit.csv
# Fill is_match, keeping selected and the sampling columns unchanged.
jev-link evaluate audit.csv --mode selected --markdown
jev-link evaluate audit.csv --mode threshold --threshold 0.8 --markdown
```

The existing `--left`, `--right`, `--on`, `--left-id` and `--right-id` options add source
fields beside each other for labeling. When source tables are provided, both score and
link IDs are aligned to those originals. Without originals, supply score/link tables with
matching ID types, such as two CSV files written by the same result. CSV readers preserve
leading zeros in IDs. CLI sampling and bootstrap evaluation use fixed default seeds of 0.

Old audit files without `selected` continue to work in the default threshold mode. Selected
mode rejects them with instructions for obtaining actual membership; it never guesses from
`p`. Old files that retain pair IDs can be augmented by an exact membership join against
the complete links table from the same run, preserving labels, bins and weights.

## Exported table contract

These ordinary columns are sufficient for an independent labeling or review interface;
selection does not depend on pandas attributes or categorical metadata:

| Column | Contract |
|:--|:--|
| `left_id`, `right_id` | Nonmissing, unique pair key; preserve values and types, including leading zeros. Required for selected-mode evaluation. |
| `p` | Pair probability between 0 and 1. Sampled pairs have finite probabilities. |
| `selected` | Optional boolean: true exactly when this pair belongs to the complete final links table. Absence means unknown, not false. |
| `is_match` | Human truth label: 1/0, True/False, y/n or yes/no, ignoring case and surrounding spaces. A blank is an incomplete label. |
| `bin` | Original sampling stratum; do not recompute it after labeling or filtering. In a `.dta` audit it is a coded value label; `jev-link evaluate` names the codes from those labels for display, which changes no stratum and no number. |
| `weight` | Positive inverse inclusion weight: bin population divided by bin sample size. Required on every row, including rows whose label is blank. |
| `a_<field>`, `b_<field>` | Optional fields displayed for human comparison; not used in the metrics. |

The exporter writes boolean `selected` values. The evaluator accepts booleans, numeric
0/1 (including 0.0/1.0), or their text forms and `True`/`False` strings, ignoring case and
surrounding spaces. Blank, missing, yes/no and other values are errors in this machine-made
column, including on rows without human labels. `"False"` is never treated as a truthy
string. Threshold mode ignores `selected`. Human labels and machine selection have distinct
roles; filling `is_match` must not change `selected`.

## What the estimates cover

Sampling includes every candidate with finite `p`, including exact-match shortcuts, and
excludes candidates left unjudged by a budget or a failed call (missing `p`). Both modes
estimate accuracy over this **judged candidate population**. Recall's denominator is true
matches among judged candidates, not all true matches in the original data. It excludes
true matches that blocking never proposed and those in candidates without probabilities.
Selected mode changes the predictions, not the population. A manually labeled row with
missing `p` is rejected by evaluation, rather than counted as a negative prediction.

For labels `y`, predictions `z` and sampling weights `w`, the weighted totals are
`TP = sum(w*y*z)`, predicted matches `sum(w*z)` and true matches `sum(w*y)`. Precision is
`TP / sum(w*z)`, recall is `TP / sum(w*y)`, and F1 is `2*TP / (sum(w*z) + sum(w*y))`.
Zero denominators produce NaN and an explanation that speaks of labeled pairs, since a
predicted or selected link on a row with a blank label enters no total. Calibration and the
weighted Brier score always compare **pair probabilities** to human truth, over labeled
pairs, across selected and unselected pairs alike. They do not measure the calibration of
final-link membership.

## Blank labels

A row whose `is_match` is blank is counted and left out of the totals, but it still says
how much of the population its bin stands for. Within each bin, the labeled pairs are
reweighted to the total weight of all the bin's sampled rows: each labeled weight is
multiplied by (weight of all sampled rows in the bin) / (weight of its labeled rows). A
bin in which 30 of 100 sampled pairs were labeled therefore counts as much as a bin in
which all 100 were. Without this, bins with more blanks count for less. If a labeler works
down from the likely matches and leaves most low-probability pairs blank, the true matches
in those bins, which the links miss, are undercounted and recall is overstated.

The factor applies to precision, recall, F1 and the weighted Brier score, in both modes,
and to every bootstrap replicate. The summary lists the adjusted bins and their factors. **If no
label is blank, every factor is exactly 1 and the results are identical to those of earlier
versions.** If blanks are equally frequent in every bin, the factors are equal and the
estimates do not move.

What is adjusted is an unequal *rate* of blanks across bins. What is not adjusted is *which*
pairs within a bin are blank. The adjustment treats a bin's labeled pairs as representative
of its blank ones. If the pairs left blank within a bin are the hard ones, where the model
is more often wrong, the labeled pairs flatter the bin and no reweighting can recover that.
Label those pairs, or report the blank counts beside the estimates.

Each calibration row describes the bin's **labeled** pairs. `n` counts them, and `mean_p`
and `match_rate` are both weighted means over those same pairs; the reweighting factor is
common to a bin and cancels, so a row is the same with or without it. `mean_p` is therefore
not the mean probability of every pair sampled in the bin, even though the blank rows'
probabilities are usually known. Calibration is the gap between the two columns, and that
gap means something only if both describe the same pairs. If blanks within a bin depend on
`p`, say a labeler skips the lower-scored pairs of a bin, the labeled pairs still show
whether scores near theirs come true, while the mean `p` of all sampled rows set against
the match rate of the labeled ones would show a gap that is not there. It also lets a
blank row carry no usable `p` at all. A bin without labels has no calibration point:
both columns are NaN.

A bin with sampled rows but no label at all cannot be estimated, and its weight is not
redistributed to other bins, whose pairs have different probabilities. Population-wide
metrics, Brier and intervals are then NaN. The summary names those bins and states the
share of the sampled weight they hold; bins that do have labels keep their calibration
rows. Label at least a few pairs in every bin.

Keep blank rows in the file, with their weights: they are what the adjustment is computed
from, and deleting them returns the old, unadjusted estimate without any sign of it. A
blank row needs a valid `weight` but not a valid `p`, in Python and in `jev-link evaluate`
alike. CSV does not preserve unused
categorical bins: choose a sample size large enough to include every populated bin before
exporting, and do not delete rows or bins. Once omitted rows or bins have been removed from
a plain file, the evaluator cannot reconstruct them or verify the sample's coverage.

## Intervals

The 95% intervals use seeded within-bin resampling of labeled rows with their weights,
including the blank-label factor, and fixed predictions. A replicate draws as many labeled
rows in a bin as were labeled. With the one weight per bin that `audit_sample` writes, this
equals rescaling each replicate to its bin's total weight; with weights that vary inside a
bin, the factor is fixed rather than recomputed per replicate, which keeps fully labeled
results unchanged. The intervals condition on the observed candidate pool, the selection
and the number of labels in each bin. They do **not** estimate blocking uncertainty, error
from unjudged candidates, assignment changes in another dataset, model-call variability or
labels left blank for reasons related to the truth. They do not apply
a finite-population correction; a fully labeled census can still have bootstrap intervals.
A bin with only one label cannot reveal within-bin variability, and undefined bootstrap
replicates are excluded and reported. Treat sparse-bin intervals cautiously: a bin whose few
labels show no match contributes no spread at all, so intervals for recall run short when
true matches are rare in the low bins, with or without blank labels.

## Changes to reported numbers

Estimates from **partially labeled** audits change with the blank-label reweighting above:
precision, recall, F1, Brier and their intervals, whenever blanks are more frequent in some
bins than in others. Rerun `evaluate` on the labeled file, with its blank rows, and report
the new numbers. Estimates from **fully labeled** audits do not change at all; a regression
test compares them bit for bit with the previous evaluator. Two things are stricter: a row
with a blank label must carry a valid positive `weight`, and the note for a bin without
labels now states the weight it holds.

When complete benchmark truth is available, `jlink.score_against_truth(links, truth,
candidates)` compares final links with the full truth and separately reports blocking
completeness. Its full-truth recall can be lower than audit recall because their
denominators differ.

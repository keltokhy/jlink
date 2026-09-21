# Dedupe: linking a table to itself

`jlink.dedupe` finds the records of one table that match each other and returns a `cluster_id`
for every record, with the pair scores behind it. The same steps run as for two tables (block,
judge, then decide), except that the last step builds clusters where `link` chooses links.

```python
result = jlink.dedupe(
    firms, entity="firm", on=["name", "city"], id="firm_id",
    definition="A subsidiary or division counts as its parent company.",
)
result.clusters              # firm_id as `id`, cluster_id, cluster_size: one row per record
result.labeled()             # the table itself, with cluster_id and cluster_size appended
result.scores                # every candidate pair with p, as for a two-table run
print(result.report()); print(result.methods())

loose = result.recluster(threshold=0.4)                 # no new calls
whole = result.recluster(unproposed="ignore")           # another rule, still no calls
result.save("dedupe/"); again = jlink.load("dedupe/")   # clusters.csv, scores.csv, settings.json
```

`jlink.Linker(...).dedupe(table, id=...)` is the same call on a configured linker, and
`linker.estimate(table, left_id=...)` runs blocking only. Rule-style questions work as they do
for two tables: `style="rule"` with "Both news articles report the same shooting incident."
collapses articles into events.

## Pairs

Every pass runs with the table on both sides. `jlink.block.self_candidates(table, on=..., id=...)`
then drops each record's pair with itself and keeps (i, j) and (j, i) as one pair, so an
unordered pair is a candidate once and is judged once. `max_pairs` counts unordered pairs.

- `on` names plain columns. A `(left, right)` pair or a one-sided field has no meaning in one
  table and is an error.
- A record is its own nearest n-gram neighbor. The default pass is therefore `ngrams` over the
  `on` fields with `k=11`, which leaves ten others. A pass you supply is used as given:
  `ngrams("name", k=10)` finds nine others.
- A one-sided window works in both directions, because either order can propose the pair:
  `window("year", between=(0, 1))` keeps pairs at most one year apart.
- Pass diagnostics count `proposed_pairs` before self-pairs and mirror images are removed, and
  `unique_pairs` after. `candidates.attrs["blocking"]["unordered"]` is true.
- `pairs_completeness(candidates, truth, unordered=True)` treats (a, b) and (b, a) alike.
- For two tables, `candidates` warns when an `exact` or `within` pass proposes nothing because
  the two sides share no key. One table cannot disagree with itself, so here a keyed pass warns
  only when no record has a complete key. Keys that are all different propose no pairs and do
  not warn: that is an answer, not a fault. `within(..., missing="match")` groups the records
  that lack the key, as it does for two tables.
- Saved runs record the ID kind, float IDs included, for `left_id`, `right_id` and the
  clusters' `id`, so `jlink.load` returns them as they were.

`left_id` is always the record from the earlier row and `right_id` the later one, whichever
direction a pass proposed. Tables keep the column names `left_id` and `right_id` so that
`audit_sample`, `evaluate` and saved scores work unchanged.

### Presentation order

The judge sees the earlier row as record A and the later row as record B, once. jlink has
**not measured** whether Jev answers (A, B) and (B, A) alike. What the code does establish:

- The answer cache is keyed on the state, which names `record_a` and `record_b`. The reverse
  presentation is a different question to the cache. Sorting the table differently therefore
  asks, and pays for, pairs that an earlier run already judged the other way round.
- Settings record `pair_order: "earlier_row_is_record_a_v1"`, and `methods()` says that each
  pair was judged once with the earlier row first.
- Keep the row order fixed between runs, or reuse saved scores with `recluster`, to stay on
  one presentation. Judging both orders and comparing them would double the cost and is left
  to the user; jlink does not do it silently.

## From pair probabilities to clusters

`jlink.cluster(scores, ids=..., threshold=0.5, linkage="average", unproposed="nonmatch")` is the
whole last step, usable on any saved scores. Three rules are offered.

| Rule | A merge needs | Fails by |
|---|---|---|
| `linkage="components"` | one pair with `p >= threshold` | chaining: one wrong pair joins two groups of any size |
| `linkage="average", unproposed="ignore"` | mean `p` of the **judged** pairs between two clusters `>= threshold` | joining two groups whose only judged pair is wrong |
| `linkage="average", unproposed="nonmatch"` (default) | mean `p` over **every** pair between two clusters `>= threshold`, a pair blocking never proposed counting as 0 | splitting a true group that blocking covered only in part |

Average linkage starts from single records and repeatedly merges the two clusters with the
highest mean between them, while that mean is at least the threshold. A candidate pair with
no probability (left unjudged by the budget, or failed) is never evidence under any rule: it
leaves the mean, numerator and denominator both.

**Why the default counts unproposed pairs.** Blocking is a similarity search. When it proposes
one pair between two groups of three and none of the other eight, those eight are not unknown:
blocking found them too unlike to propose. Counting them as non-matches means one high
probability cannot join two groups (0.9 over nine pairs is 0.1), while a group whose members
were all compared merges exactly as before. The price is the third row of the table: a record
joins a cluster only on evidence against enough of its members, at the default threshold at
least half of them. If blocking reaches only some members of a true group, the group is split.

Isolated wrong pairs can join otherwise separate groups when only judged pairs count.
Counting unproposed pairs as non-matches resists that failure but can split true groups
when blocking covers too few of their members. Check both failure modes on your own data.

The default was chosen because chaining is the failure that is hard to see afterwards: a false
merge hides inside a large cluster, while a false split leaves evidence behind. Every judged
pair at or above the threshold whose records ended in different clusters is listed by
`result.split_pairs()` and counted in `report()`. Read those pairs first. If they are true
matches, blocking is too thin for the rule: raise `k`, add a pass, or recluster with
`unproposed="ignore"`. Reclustering costs nothing, so compare the rules on your own scores.
Passes that propose every pair inside a block (`exact`, `window`, `within` around either) leave
no pair unproposed inside it, so they split a true group only when it spans blocks: articles a
week apart under a three-day window, for example.

Other properties:

- Greedy merging takes the best pair of clusters first. A wrong pair that outranks every true
  pair merges first; the damage then stays with the records it touches (they split off, or
  one is pulled across), and the two groups still do not chain into one. `tests/test_dedupe.py`
  pins both outcomes.
- Means are compared in exact integer arithmetic on the stored double-precision values, and
  ties go to the clusters whose members appear earliest in the table. The result does not
  depend on the row order of `scores`. One consequence: 0.95 and 0.05 are stored as binary
  fractions whose exact mean is a hair under 0.5, although `0.95 + 0.05 == 1.0` in floating
  point. The threshold is inclusive.
- The heap-based implementation is checked against a direct restatement of the definition
  (rescan every pair of clusters after every merge) on random graphs with ties, under both
  `unproposed` settings.
- Clusters always lie inside the connected components of the pairs at or above the threshold,
  so components are an upper bound on merging. Work is done per component; a component with
  nothing to decide is skipped.
- `cluster_id` numbers clusters by the first appearance of any member in the table.

## Checking the clusters

```python
sample = result.audit_sample(n=200)        # a_<field>, b_<field>, p, selected, ...
# fill is_match, then:
jlink.evaluate(labeled, mode="selected")   # are the pairs inside clusters true matches?
jlink.evaluate(labeled, mode="threshold")  # how well do the pair scores classify?
```

`selected` is true when the two records share a cluster. This measures the clusters pair by
pair **among judged pairs**. Two records can share a cluster through other records without a
judged pair of their own; such pairs are outside the sample, so the precision estimate does
not cover them. Recall, as for two tables, covers judged candidates only and says nothing
about matches that blocking never proposed.

## Command line

```bash
jev-link dedupe firms.dta --on name --on city --entity firm --id firm_id --estimate
jev-link dedupe firms.dta --on name --on city --entity firm --id firm_id \
    -o clusters.csv --scores scores.csv --links links.csv --report report.md --save dedupe/
jev-link cluster scores.csv --records firms.dta --id firm_id --threshold 0.8 -o strict.csv
jev-link cluster scores.csv --records firms.dta --id firm_id --unproposed ignore -o whole.csv
jev-link audit scores.csv --links links.csv --left firms.dta --right firms.dta --on name \
    --left-id firm_id --right-id firm_id -n 200 -o audit.csv
jev-link evaluate audit.csv --mode selected --markdown
```

`dedupe` accepts the blocking, question, budget and cache options of `link`, plus `--threshold`,
`--linkage` and `--unproposed`. `--estimate` runs blocking only. `--save DIR` writes the folder that `result.save(DIR)` writes, for
`jlink.load` and replication. `--links` saves the judged pairs
that share a cluster, which is what `audit --links` expects. `cluster` regroups saved scores
without API calls; without `--records` it can only list records that some pair mentions.

Saved runs retain integer, float and string ID kinds, including literal `001` and `NA` strings.
Loading preserves float IDs and probabilities exactly, so `recluster` reproduces the in-memory
clusters with the same settings and record order, without calling the model.
The `cluster` command also reads probabilities at full float precision. Use
`cluster dedupe/scores.csv --records firms.dta --id firm_id` to retain the source ID types,
records without candidates, and source order. CSV and TSV source IDs are strings; Stata and
Parquet retain numeric ID types. Without `--records`, score order determines the ID order
and can affect tie-breaking.

## What does not work yet

- The local review page and `review apply` resolve links between two tables. They refuse a
  dedupe result and point to `audit_sample`.
- The Stata and R wrappers call `link` only. Use the command line for dedupe.
- `score_against_truth` compares ordered `(left_id, right_id)` pairs. For dedupe, write the
  truth pairs with the earlier row first, or use `pairs_completeness(..., unordered=True)` for
  blocking recall. There is no cluster-level metric against known truth.
- No measurement of Jev on a dedupe task exists in this repository. The tests use a fake model.

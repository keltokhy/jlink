# Hybrid linkage experiments, 2026-09-20

These are exploratory measurements on previously inspected public datasets. They do not
establish that jlink beats LinkTransformer end to end or the historical entity-linking paper.
No default retrieval change is justified by the firm results below.

The [follow-up error audit](../docs/linkage-next-experiment.md) isolates product decision
errors and missing distinguishing fields, and records an unsuccessful development-threshold
control. It proposes the next experiments without changing the results below.

## Firm retrieval

Both model experiments use the same observed-entity split: 3,251 left records, 1,744 right
records and 3,256 listed test links. Each retrieval pass uses k=10; hybrid takes the union.
Thresholds and k were not tuned on test labels. The company encoder was added after inspecting
the general encoder's results, so the model comparison is exploratory, not model selection on
an untouched test. No firm judgments were purchased in these experiments.

| Retrieval | Encoder | Pairs | Listed matches retrieved | Candidate recall |
|---|---|---:|---:|---:|
| Lexical | character TF-IDF | 32,101 | 2,157 | 66.25% |
| Lexical, k=20 control | character TF-IDF | 61,969 | 2,176 | 66.83% |
| Semantic / native LinkTransformer | all-MiniLM-L6-v2 | 32,510 | 2,144 | 65.85% |
| Hybrid | all-MiniLM-L6-v2 | 56,131 | 2,187 | 67.17% |
| Semantic / native LinkTransformer | lt-wikidata-comp-en | 32,510 | 2,140 | 65.72% |
| Hybrid | lt-wikidata-comp-en | 59,408 | 2,207 | 67.78% |

For these single-name inputs, jlink's semantic retrieval and the actual LinkTransformer
`merge_knn` call found the same number of listed links with each encoder. The company hybrid
adds 27,307 pairs to recover 50 additional listed matches relative to lexical retrieval.
It also retrieves 31 more listed matches than lexical k=20 while proposing 2,561 fewer
pairs. This comparison controls approximately for candidate workload, not runtime or final
judgment quality.
This modest gain at much greater judging workload points to missing information in name-only
parent/subsidiary linkage. Richer dated context or independently learned aliases should be
tested before simply enlarging candidate pools. NBER's unlisted links remain unknown; these
figures are not precision/F1 or an estimate of all real matches.

## Product linkage with a shared judge

On the Abt-Buy test partition (755 left records, 762 right records, 766 supplied links),
all methods use exactly the same stored Jev probability for any shared candidate, the same
many-to-many rule, threshold 0.5 and disabled exact shortcut. LinkTransformer supplies its
native retrieval; **its LLM judge is not evaluated here**. The general MiniLM encoder is used.

| Retrieval followed by Jev | Candidates | Candidate recall | Final precision | Final recall | Final F1 |
|---|---:|---:|---:|---:|---:|
| Lexical k=10 | 7,533 | 99.09% | 0.9143 | 0.9334 | 0.9238 |
| Semantic k=10 | 7,550 | 95.56% | 0.9244 | 0.8943 | 0.9091 |
| Hybrid union | 10,698 | 99.61% | 0.9144 | 0.9347 | 0.9245 |
| Native LinkTransformer k=10 | 7,550 | 96.87% | 0.9231 | 0.9086 | 0.9158 |

Hybrid recovers four candidate positives missed by lexical retrieval, but only one additional
final true link. Its 0.0007 F1 gain is small. A lexical k=20 retrieval-only control finds 765 of
766 listed links (99.87%) with 14,934 pairs; its additional pairs were not judged. These are
not matched-precision or equal-cost end-to-end superiority results.

The original dev/test union run made 16,886 new calls at reported cost $0.351708546 and reused
161 cached answers. Three test requests timed out. A fresh replay reused 17,047 cached answers
and completed those three at $0.000060312, leaving zero unjudged pairs. Final metrics use that
complete replay. The total new product-experiment cost was $0.351768858; this is the cost of
judging a shared union, not a separate standalone cost for each method. Including the synthetic
rule experiment, total recorded new API cost was $0.352386342. Unmetered failed-request charges,
if any, are not included. The experiment resolves to `typesafe/jev-1.13-20260917`.

## Rule sensitivity

The frozen synthetic fixture contains 6 development and 12 test pairs. Each pair is judged
under strict legal identity and common corporate control at the stated date. The supplied
facts are invented; no real-company ownership knowledge is tested. All 36 requests completed
with `typesafe/jev-1.13-20260917`, using a fixed 0.5 threshold. Reported API cost: $0.000617484.

| Test measure | Result |
|---|---:|
| Legal-entity decisions correct | 12 / 12 |
| Corporate-family decisions correct | 10 / 12 |
| Rule-dependent pairs: both decisions correct | 2 / 4 |
| Rule-dependent pairs: score changed in expected direction | 4 / 4 |
| Rule-invariant pairs: both decisions correct | 8 / 8 |

The development fixture also misses the corporate-family match between explicitly stated
sister companies (score 0.25). Score movement alone does not establish correct rule following.
These data identify a concrete gap to address with development-only prompt/model experiments
and a new independently labeled evaluation set. The current match question and threshold were
not changed after inspecting these results.

## Provenance and reproduction

The comparison executed LinkTransformer commit `584562c9755d987f47e4a4a04d0100e4bcf7c044`
(package version 0.1.18) in a separate environment using its declared dependencies. Its
native LLM judge was not run. Model revisions:

- `sentence-transformers/all-MiniLM-L6-v2`: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- `dell-research-harvard/lt-wikidata-comp-en`: `af65d96525b7cb3e125a52b89928ae78b39c6d21`

Use the commands in [the protocol](../docs/hybrid-linkage.md). Detailed reports preserve
input, split, source-code and artifact hashes, per-stage counts and dependency versions.
Local full artifacts reside in `bench/out/hybrid-20260920-v2`,
`bench/out/hybrid-20260920-company`, `bench/out/hybrid-20260920-final` and
`bench/out/rules-live-20260920`.
Frozen reports and the small synthetic request/response bundle are retained under
`bench/evidence/hybrid-2026-09-20/`. Split CSVs are gzip-compressed without changing their
decompressed bytes. `runs.tar.gz` preserves the candidate tables, final scores and links,
external retrieval exports and metadata for independent inspection. Its manifest records
the archive checksum and exact measured spend. The first retrieval reports' code hashes refer
to commit `586c91a`; the final replay adds a cosine-one floating-point boundary repair.

Validation: 510 offline tests pass; source and wheel builds pass; a fresh base-only wheel
install verifies the safer identity default, optional-dependency behavior and embedding
cosine boundary; the real hybrid CLI estimate runs successfully on the bundled example.

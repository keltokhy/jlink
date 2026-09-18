# Offline benchmark evidence

Measured on 2026-09-18 from prepared datasets and existing cached model scores. **Zero new API calls and zero new API cost.** [evidence/2026-09-18.json](evidence/2026-09-18.json) retains complete reports, settings, parameters, data/code/artifact hashes and threshold trials. [EVALUATION.md](EVALUATION.md) specifies how to reproduce and interpret them.

## Matched retrospective replay

Observed entities are split 30% development / 70% test, seed 1729. Every method uses the same cached candidate pool filtered within each partition, and the same dataset cardinality. String/ECM thresholds use development final F1; Jev stays at 0.5. These retrospective partitions do not undo earlier dataset or prompt inspection. Cosines retain the historical full-corpus fit. They are not untouched prospective results or comparable to the original in-sample oracle table.

| Dataset | Test left/right | Test truth | Candidate pairs | Candidate recall | Exact F1 | Jaro F1 | TF-IDF F1 | ECM F1 | Cached Jev F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| febrl4 | 3,500 / 3,500 | 3,500 | 25,711 | 1.0000 | 0.2166 | 0.9983 | 0.9996 | 1.0000 | 0.9596 |
| dblp-acm | 1,836 / 1,606 | 1,565 | 13,255 | 0.9994 | 0.4110 | 0.9891 | 0.9885 | 0.9880 | 0.9965 |
| abt-buy | 755 / 762 | 766 | 5,561 | 0.9883 | 0.0000 | 0.2620 | 0.6877 | 0.2393 | 0.9238 |
| amazon-google | 939 / 2,254 | 898 | 6,774 | 0.9699 | 0.0000 | 0.3923 | 0.5279 | 0.2271 | 0.7213 |
| nber-firms | 3,251 / 1,744 | 3,256 | 23,250 | 0.6579 | — | — | — | — | — |

Leipzig metrics use the supplied mappings' closed-world convention; they do not independently verify omitted links. Product cardinality is many-to-many for all methods. The old string oracles chose only one right record per left. The difference in product F1 is therefore not evidence of a regression in the underlying string algorithms.

ECM is competitive on people/publications but weak under the fixed product feature specification. It is inapplicable to name-only NBER. A more complex method does not guarantee a better result. On FEBRL's complete test, selected Jaro and ECM thresholds triggered the existing resolver's >2,000-node greedy fallback; the report preserves those warnings. No new resolver optimization is claimed here.

### Pair judgments versus final assignments

All values below are from cached Jev probabilities at 0.5. Pair metrics precede resolution and include exact shortcuts; per-source detail is in JSON. Final recall uses all test truth, including matches missed by blocking.

| Dataset | Pair precision among candidates | Pair recall among candidate truth | Final precision | Final recall against all test truth |
|---|---:|---:|---:|---:|
| febrl4 | 0.9994 | 0.9223 | 1.0000 | 0.9223 |
| dblp-acm | 0.9304 | 1.0000 | 0.9955 | 0.9974 |
| abt-buy | 0.9143 | 0.9445 | 0.9143 | 0.9334 |
| amazon-google | 0.6952 | 0.7727 | 0.6952 | 0.7494 |
| nber-firms | — | 0.9496 | — | 0.6247 |

NBER: 2,034 listed test links selected; 503 unlisted predictions remain unknown. No NBER precision, F1, calibration, false-positive claim or F1 tuning is made in this protocol. NBER historical oracle precision/F1 in BASELINES.md is retained only as crosswalk agreement under its original negative assumption.

## Explicit unmatched FEBRL records

The test universe has 3,150 left and 3,150 right records, 2,800 known matches, and 350 explicitly unmatched records on each side after removing known counterparts. All methods use the same filtered cached candidate pool. No previously unknown entity is declared unmatched.

| Method | Dev-selected/fixed threshold | Test F1 | Unjudged candidates | Unmatched left falsely linked | Unmatched right falsely linked |
|---|---:|---:|---:|---:|---:|
| exact | 1.0 | 0.2217 | 0 | 0 | 0 |
| jaro_winkler | 0.7 | 0.9689 | 0 | 62 | 64 |
| tfidf | 0.35 | 0.9980 | 0 | 8 | 8 |
| ecm | 0.1 | 0.9973 | 0 | 0 | 0 |
| cached_jev | 0.5 | 0.9616 | 0 | 0 | 0 |

## English-definition sensitivity

Status: **live_model_pending**. The committed synthetic fixture creates 36 exact requests for 18 pairs and two definitions. There are 0 suitable cached model responses. Offline fake tests verify changed-case sensitivity, invariant cases, missing responses, stale hashes and evidence separation. They are not live model results. Real-company ownership knowledge is not tested by this fixture.

## Provenance and remaining experiments

Inputs were read from `/Volumes/K3/GitHub/jlink/bench/data` and `bench/out` without modification. Replays save source hashes, explicit split definitions, scores, links and unmatched removals. The JSON bundle preserves reports and hashes; per-pair CSVs remain in the worktree's ignored `bench/out/final-cached`, `final-unmatched`, and `final-rules` directories and are reproducible with the documented commands. Legacy caches lack raw provider responses and original dataset hashes; current ID/field validation cannot prove which historical source values were submitted. These limitations are disclosed in each report.

Pending work: obtain independently reviewed legal-entity/family labels for real firms; run both definitions on the frozen synthetic requests with explicit spending authorization; collect additional independently labeled NBER negatives before precision/F1 claims; evaluate new blockers and unjudged pairs after integration; and repeat across prespecified splits or new untouched datasets before generalizing. No result from those experiments is implied by this offline report.

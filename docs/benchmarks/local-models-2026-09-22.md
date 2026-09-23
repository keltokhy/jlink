# Jev, DiffusionGemma and Laya on jlink's benchmarks

Run from 22:36 on September 22, 2026 to 00:06 on September 23, America/New_York, on an Apple M3
Ultra with 96 GiB of unified memory, on the five benchmarks in the
[README](../../README.md#benchmarks). For each, `bench/live.py --sample 300` drew 300 left records
with `random_state=0` and kept their known matches, blocked with every left record (n-gram TF-IDF,
`k=10`) and kept the sampled records' candidate pairs, so each local model judged exactly the
pairs Jev judged in the full run. Each pair
went to the model with the final rule in `bench/live.py`, the exact shortcut on as in jlink 0.1.0,
an empty answer cache and 4 requests in flight, and the links were resolved at 0.5 as in the full
run. Jev 1.13 was not called again: its column is its saved probabilities from the full runs of
September 18, restricted to the same pairs, then resolved and scored by the same code.
`bench/local_models.py` builds that column and writes
[local-models-2026-09-22.json](local-models-2026-09-22.json), the frozen output every number here
comes from. The [last section](#linktransformer) adds LinkTransformer, run on the same samples on
September 23.

"Pairs scored 0.5 or above" and "pair decisions that match Jev's" compare probabilities pair by
pair, before resolution. "Links made" are after it. Precision, recall and F1 are rounded to three
places from the counts in the JSON, which stores them to four.

## Firms: NBER patent assignees to Compustat

300 left records with 300 known links, 2,981 candidate pairs, many-to-one. The rule: *"Ignore
legal-form suffixes (Inc, Corp, Co, CPY, Ltd, GmbH, N V). A subsidiary or division counts as its
parent company."*

| | Jev 1.13 (OpenRouter, recorded run) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| Precision / recall at 0.5 | 0.874 / 0.600 | 0.818 / 0.600 | 0.223 / 0.223 |
| F1 at 0.5 | **0.711** | 0.692 | 0.223 |
| Best F1 in the sample (threshold) | 0.717 (0.6) | 0.711 (0.95) | 0.234 (0.95) |
| Links made at 0.5, of them true | 206, 180 | 220, 180 | 300, 67 |
| Pairs scored 0.5 or above | 259 | 350 | 2,679 |
| Pair decisions that match Jev's at 0.5 (Cohen's κ) | | 93.9% (0.67) | 17.2% (0.00) |
| Jev's links it also made | | 184 of 206 | 69 of 206 |
| Model calls; failed or refused pairs | | 2,935; 0 | 2,935; 0 |
| Wall-clock; API cost | | 831.9 s; $0 | 65.9 s; $0 |

Jev's full run judged all 45,567 candidate pairs of the table in 175.8 s for $0.62, with F1 0.732
at 0.5.

## Publications: DBLP to ACM

300 left records with 262 known links, 3,000 candidate pairs, one-to-one. The rule: *"The same
paper listed in two bibliographic databases. Author lists may be abbreviated and venue names may
differ. A different paper by the same authors is not a match."*

| | Jev 1.13 (OpenRouter, recorded run) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| Precision / recall at 0.5 | 0.981 / 1.000 | 0.970 / 0.981 | 0.653 / 0.740 |
| F1 at 0.5 | **0.991** | 0.975 | 0.694 |
| Best F1 in the sample (threshold) | 0.996 (0.7) | 0.979 (0.9) | 0.717 (0.9) |
| Links made at 0.5, of them true | 267, 262 | 265, 257 | 297, 194 |
| Pairs scored 0.5 or above | 281 | 284 | 1,609 |
| Pair decisions that match Jev's at 0.5 (Cohen's κ) | | 99.2% (0.96) | 55.3% (0.16) |
| Jev's links it also made | | 260 of 267 | 198 of 267 |
| Model calls; failed or refused pairs | | 2,912; 0 | 2,912; 0 |
| Wall-clock; API cost | | 1,034.3 s; $0 | 48.6 s; $0 |

Jev's full run judged all 26,146 candidate pairs in 111.6 s for $0.45, with F1 0.996 at 0.5.

## Products: Abt to Buy

300 left records with 306 known links, 3,000 candidate pairs, many-to-many. The rule: *"Two
listings match when they offer the same product model. The same product line in a different size,
color, capacity, software version or license bundle is not a match."*

| | Jev 1.13 (OpenRouter, recorded run) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| Precision / recall at 0.5 | 0.938 / 0.941 | 0.881 / 0.948 | 0.102 / 0.908 |
| F1 at 0.5 | **0.940** | 0.913 | 0.183 |
| Best F1 in the sample (threshold) | 0.942 (0.4) | 0.936 (0.9) | 0.185 (0.3) |
| Links made at 0.5, of them true | 307, 288 | 329, 290 | 2,728, 278 |
| Pairs scored 0.5 or above | 307 | 329 | 2,728 |
| Pair decisions that match Jev's at 0.5 (Cohen's κ) | | 98.1% (0.90) | 17.7% (0.00) |
| Jev's links it also made | | 290 of 307 | 283 of 307 |
| Model calls; failed or refused pairs | | 2,973; 0 | 2,973; 0 |
| Wall-clock; API cost | | 1,098.7 s; $0 | 158.7 s; $0 |

Jev's full run judged all 10,796 candidate pairs in 57.8 s for $0.22, with F1 0.915 at 0.5.

## Software: Amazon to Google

300 left records with 285 known links, 2,997 candidate pairs, many-to-many. The rule: *"Two
listings match when they offer the same software product and edition. A different version,
platform, license type or bundle is not a match."*

| | Jev 1.13 (OpenRouter, recorded run) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| Precision / recall at 0.5 | 0.598 / 0.768 | 0.540 / 0.807 | 0.088 / 0.856 |
| F1 at 0.5 | **0.673** | 0.647 | 0.159 |
| Best F1 in the sample (threshold) | 0.673 (0.5) | 0.652 (0.3) | 0.163 (0.8) |
| Links made at 0.5, of them true | 366, 219 | 426, 230 | 2,787, 244 |
| Pairs scored 0.5 or above | 366 | 426 | 2,787 |
| Pair decisions that match Jev's at 0.5 (Cohen's κ) | | 95.7% (0.81) | 16.7% (−0.01) |
| Jev's links it also made | | 331 of 366 | 328 of 366 |
| Model calls; failed or refused pairs | | 2,904; 0 | 2,904; 0 |
| Wall-clock; API cost | | 927.1 s; $0 | 44.8 s; $0 |

Jev's full run judged all 13,610 candidate pairs in 60.0 s for $0.22, with F1 0.664 at 0.5.

## People: FEBRL4, synthetic typos

300 left records with 300 known links, 3,000 candidate pairs, one-to-one. The rule: *"Both records
come from forms filled in by hand, so expect typing errors, swapped or missing fields and
abbreviations. Sharing only a surname or only an address does not establish a match."*

| | Jev 1.13 (OpenRouter, recorded run) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| Precision / recall at 0.5 | 1.000 / 0.933 | 1.000 / 0.723 | 0.783 / 0.613 |
| F1 at 0.5 | **0.966** | 0.839 | 0.688 |
| Best F1 in the sample (threshold) | 0.997 (0.1) | 0.879 (0.1) | 0.721 (0.8) |
| Links made at 0.5, of them true | 280, 280 | 217, 217 | 235, 184 |
| Pairs scored 0.5 or above | 280 | 217 | 370 |
| Pair decisions that match Jev's at 0.5 (Cohen's κ) | | 97.9% (0.86) | 90.9% (0.53) |
| Jev's links it also made | | 217 of 280 | 183 of 280 |
| Model calls; failed or refused pairs | | 2,972; 0 | 2,972; 0 |
| Wall-clock; API cost | | 1,054.4 s; $0 | 57.8 s; $0 |

Jev's full run judged all 50,000 candidate pairs in 207.6 s for $0.97, with F1 0.959 at 0.5.

## What the tables say

- **DiffusionGemma comes within 0.03 of Jev's F1 on firms, publications and products.** At 0.5 it
  scores 0.692 against Jev's 0.711 on firms, 0.975 against 0.991 on DBLP to ACM, 0.913 against
  0.940 on Abt to Buy and 0.647 against 0.673 on Amazon to Google, and it decides 93.9 to 99.2% of
  candidate pairs the way Jev did. It scores more pairs 0.5 or above than Jev (350 against 259 on
  firms, 426 against 366 on Amazon to Google), which costs it precision: 0.818 against 0.874 and
  0.540 against 0.598. Its recall is level with Jev's or higher on firms and products, and 0.981
  against 1.000 on publications.
- **Its probabilities sit near 0 and 1, so a higher threshold costs it little.** On Abt to Buy it
  put 2,559 of 3,000 pairs at 0.05 or below and 295 above 0.95. Its F1 there rises from 0.913 at
  0.5 to 0.936 at 0.9, where Jev's falls from 0.940 to 0.795. On firms, none of the 71 pairs it
  scored above 0.5 and up to 0.8 was a true match. Its best thresholds on firms, Abt to Buy and
  DBLP to ACM are 0.95, 0.9 and 0.9, where Jev's are 0.6, 0.4 and 0.7, so a threshold chosen on
  Jev's scores does not carry over. `result.relink` tries another without new calls.
- **On FEBRL4 DiffusionGemma made no false links and missed more true ones than Jev.** Its
  precision is 1.000 at every threshold from 0.1 to 0.95, and every pair it scored above 0.05 was
  a true match. It left 83 of the 300 true pairs below 0.5, so its recall is 0.723 against Jev's
  0.933 and its F1 0.839 against 0.966; at 0.1 its F1 is 0.879 and Jev's 0.997. The rule tells
  the judge to expect typing errors and missing fields, and DiffusionGemma still holds them
  against a pair more often than Jev does. As the README notes, string similarity is the better
  tool for these data.
- **Laya scores most candidate pairs on firms and products 0.5 or above.** It did so for 2,679 of
  2,981 firm pairs, 2,728 of 3,000 Abt to Buy pairs and 2,787 of 2,997 Amazon to Google pairs,
  where Jev did for 259, 307 and 366. Its decisions match Jev's on 16.7 to 17.7% of those pairs,
  with κ between −0.01 and 0.00, and its F1 is 0.223, 0.183 and 0.159. On firms, many-to-one
  resolution then gives each of the 300 left records its best candidate, so precision and recall
  are both 0.223. No threshold in the sweep lifts its F1 above 0.234, 0.185 and 0.163.
- **On publications and people Laya's F1 is about 0.7.** It is 0.694 on DBLP to ACM and 0.688 on
  FEBRL4 at 0.5, and at best 0.717 (at 0.9) and 0.721 (at 0.8), against Jev's 0.991 and 0.966. On
  DBLP to ACM it scored 1,609 of 3,000 pairs 0.5 or above, where Jev scored 281. On FEBRL4 its
  most confident answers are right: all 107 pairs it scored above 0.95 are true matches.
- **Laya read every pair whole.** None of its 14,696 requests was refused: jlink recorded no HTTP
  422, and the adapter's audit log gained 14,696 lines, none with status `context_rejected`. The
  compared fields of these benchmarks fit its 512-token window, so the results above are its
  answers, not an effect of the window.
- **Speed and cost.** DiffusionGemma took 831.9 to 1,098.7 s per sample (14 to 18 minutes for about
  3,000 pairs) and 4,946.4 s (82 minutes) for all five, about three pairs a second at 4 requests in
  flight, on a server that runs model work one call at a time. At that rate the full firm table,
  45,567 pairs, would take about three and a half hours, where Jev's full run took 175.8 s and cost
  $0.62. Laya took 44.8 to 158.7 s per sample and 375.8 s (6 minutes) for all five. Neither local
  model cost anything in API fees.

## Provenance and caveats

- **Jev column.** Jev 1.13 (`typesafe/jev-1.13-20260917` through OpenRouter) made no calls for this
  comparison. For each dataset its probabilities are the `p` column of
  `bench/out/live/<dataset>/scores.csv`, the full run of September 18 with jlink 0.1.0 at threshold
  0.5, restricted to the sampled left records. They were resolved again with `jlink.resolve` (the
  dataset's resolution, at 0.5) and scored with `jlink.score_against_truth` and `live.sweep`, the
  code the local runs used. Each entry of the JSON names its source file under `jev.provenance` and
  `jev.full_run.file`.
- **Sample against full table.** Jev's F1 at 0.5 is 0.732, 0.996, 0.915, 0.664 and 0.959 on the
  full tables (firms, DBLP to ACM, Abt to Buy, Amazon to Google, FEBRL4; each full run's
  `result.json`) and 0.711, 0.991, 0.940, 0.673 and 0.966 on the samples. That is the variance of
  a 300-record sample. The full run's own links cut to the sample give 0.711, 0.998, 0.940, 0.673
  and 0.966. On DBLP to ACM the re-resolved number is lower because one-to-one assignment within
  the sample can hand a sampled record a right record that the full run gave to its true match
  outside the sample. The local models were resolved the same way, so the tables use the
  re-resolved number.
- **Same pairs.** In all ten runs the local model judged exactly the candidate pairs of Jev's full
  run for the sampled records: `only_local` and `only_jev` are 0. N-gram TF-IDF is fitted on the
  tables it is given, so `--block-all-left` blocks with every left record and then keeps the
  sampled records' pairs.
- **Pairs and calls.** Pairs judged are model calls, plus pairs accepted by the exact shortcut,
  plus pairs whose question an earlier pair in the same run had already asked, answered from the
  run's own cache: 2,935 + 32 + 14 = 2,981 on firms, 2,912 + 69 + 19 = 3,000 on DBLP to ACM,
  2,973 + 0 + 27 = 3,000 on Abt to Buy, 2,904 + 0 + 93 = 2,997 on Amazon to Google and
  2,972 + 28 + 0 = 3,000 on FEBRL4, the same for both local models.
- **Exact shortcut.** The runs passed `--exact-shortcut`, as jlink 0.1.0 always did. jlink 0.3.0
  accepts a pair without a call only when every compared field is nonempty and equal, so on
  FEBRL4 4 sampled pairs that Jev's run had accepted as exact (32 exact there, 28 in the local
  runs) went to the local models: `rec-810`, `rec-1485`, `rec-3891` and `rec-268` against their
  `dup-0`. Both local models scored all four as matches. Firms had 32 exact pairs and DBLP to ACM
  69 in every column.
- **Local servers.** DiffusionGemma: OpenJev at commit `e04794a`, serving `openjev-0.1` from
  `mlx-community/diffusiongemma-26B-A4B-it-4bit` revision `a7a8140` at
  `http://127.0.0.1:8080/v1/systemone`, which runs model work one call at a time on the GPU. Laya:
  laya-mlx at commit `fc1df62` with `aac6fef/laya-mlx` revision `0476785`, served as `laya-421m` at
  `http://127.0.0.1:8081/v1/systemone` through jevkit-core's `scripts/laya_server.py`, which refuses
  with HTTP 422 any request whose state would be cropped to fit its 512-token window, question
  included; jlink records a refusal as a failed pair, not a non-match. Both on an Apple M3 Ultra
  with 96 GiB of unified memory and loaded throughout.
- **Order.** Every DiffusionGemma run came first, 22:36 to 23:59 EDT on September 22, then every
  Laya run, 23:59 on September 22 to 00:06 EDT on September 23, one at a time. During each run
  the other server was loaded but idle and no other benchmark process was running.
- **Concurrency, deadlines, budget and cache.** `-j 4` on every run, with a deadline of 300 s per
  request for DiffusionGemma and 120 s for Laya. There were no deadline errors and no retries, so
  concurrency was never lowered. jlink has no deadline argument yet; `bench/live.py --timeout`
  sets one by handing jlink's judge a runtime client with that timeout. Each run had a new, empty
  `XDG_CACHE_HOME` (`bench/out/local-2026-09-22/cache-<model>-<dataset>`, checked by
  `--require-empty-cache`) and `--budget 0.01`: in jlink a zero budget sends no request at all,
  and local calls are metered at $0, so the budget was a tripwire that never bound. Every run
  reported $0.0000 and named `openjev-0.1` or `laya-421m` as the model that answered. No hosted
  model was called.
- **Who answered.** jlink's `report()` says "Judged: N by Jev" and `scores.csv` marks model
  answers `source=jev` whichever model gave them. The report's `Model:` line (on FEBRL4,
  `Model: openjev-0.1; 2,972 calls, ...`), `settings.json`'s `provider` and `resolved_models` and the
  `provider` and `model` columns of `scores.csv` name the local model. The Laya runs crossed
  midnight, so their report headers say September 23.
- **Laya refusals** are counted two ways: HTTP 422 errors in `scores.csv`, and the lines each run
  added to the adapter's audit log. Both are 0 on every dataset, and the lines added equal the
  model calls (2,935, 2,912, 2,973, 2,904 and 2,972), so nothing else called Laya during a run.
  The 1,581 `context_rejected` lines already in the log come from other tools' runs.
- **Samples.** 300 left records per dataset for both models, with 300, 262, 306, 285 and 300
  known links (firms, DBLP to ACM, Abt to Buy, Amazon to Google, FEBRL4). No run fell back to a
  smaller sample.
- **Nothing was tuned.** Rules, `k=10` and the threshold of 0.5 are those of the September 18 runs,
  and the rules are the final ones in `bench/live.py`. Each dataset ran once on each model. These
  are public benchmarks that any of the three models may have met in training.
- **Raw outputs** are in `bench/out/live/<dataset>-<model>-s300/` (`links.csv`, `scores.csv`,
  `settings.json`, `result.json`, `compare.json`) with a log beside each, and
  `bench/out/local-2026-09-22/RESULTS-NOTES.md` records the session; `bench/out/` is not
  committed. To reproduce, run the two commands at the top of `bench/live.py` for each dataset,
  then `uv run --group bench python bench/local_models.py --tags diffusiongemma-s300 laya-s300`.

## LinkTransformer

On September 23, from 00:53 to 00:54 EDT, on the same machine,
[LinkTransformer](https://github.com/dell-research-harvard/linktransformer) (Arora and Dell), the
embedding linker of the README's [comparison](../../README.md#against-linktransformer), linked the
same 300 left records of each benchmark. It ran as it did there, in its own environment, which
jlink never imports because LinkTransformer is GPL-3.0. Its own retrieval, `merge_knn` with
`k=10`, searched the full right table for each sampled record, so it scored 3,000 candidate pairs
of its own rather than jlink's. Back in jlink, `jlink.resolve` linked its cosine similarities
under the link rule the other columns use, at a threshold tuned on the development labels of the
September 20 comparison, and `jlink.score_against_truth` scored the links, the code that scored
Jev, DiffusionGemma and Laya. Two models ran on each dataset: a pretrained one (zero-shot) and one
fine-tuned on those development labels. `bench/lt_sample.py` ran them and added the
`linktransformer` block to the JSON, where LinkTransformer's numbers below come from; nothing
else in the JSON changed.

| F1, 300 left records | Link rule | Jev 1.13 | DiffusionGemma | Laya | LinkTransformer, zero-shot |
|---|---|---:|---:|---:|---:|
| Firms | many-to-one | **0.711** | 0.692 | 0.223 | 0.667 |
| DBLP to ACM | one-to-one | **0.991** | 0.975 | 0.694 | 0.972 |
| Abt to Buy | many-to-many | **0.940** | 0.913 | 0.183 | 0.323 |
| Amazon to Google | many-to-many | **0.673** | 0.647 | 0.159 | 0.403 |
| FEBRL4 | one-to-one | **0.966** | 0.839 | 0.688 | 0.912 |
| Threshold | | 0.5 | 0.5 | 0.5 | 0.66 to 0.81, tuned on development labels |
| Wall-clock for all five; API cost | | | 4,946.4 s; $0 | 375.8 s; $0 | 6.8 s; $0 |

| F1, sampled records in the test split | Records (known links) | Jev 1.13 | DiffusionGemma | Laya | LinkTransformer, zero-shot | LinkTransformer, fine-tuned |
|---|---:|---:|---:|---:|---:|---:|
| Firms | 199 (199) | **0.708** | 0.698 | 0.226 | 0.657 | 0.655 |
| DBLP to ACM | 210 (179) | **0.986** | 0.972 | 0.715 | 0.970 | 0.965 |
| Abt to Buy | 211 (216) | **0.930** | 0.906 | 0.186 | 0.328 | 0.568 |
| Amazon to Google | 214 (204) | **0.701** | 0.665 | 0.159 | 0.409 | 0.425 |
| FEBRL4 | 207 (207) | 0.957 | 0.834 | 0.701 | 0.906 | **1.000** |

The test split is the September 20 comparison's. Its development split trained the fine-tuned
models and chose every LinkTransformer threshold, so the second table is the fair one for the
fine-tuned model.

- **With no labels, jlink with DiffusionGemma is ahead of LinkTransformer's pretrained models on
  four samples of five.** It scores 0.692 against 0.667 on firms, 0.975 against 0.972 on DBLP to
  ACM, 0.913 against 0.323 on Abt to Buy and 0.647 against 0.403 on Amazon to Google.
  LinkTransformer leads on FEBRL4, 0.912 against 0.839. Jev is ahead of it on all five.
  LinkTransformer's thresholds used the development labels; jlink's 0.5 used none.
- **On the product sets much of the gap is the link rule.** Many-to-many linking keeps every
  candidate over the threshold, and LinkTransformer proposes ten for each record: on Abt to Buy its
  pretrained model made 1,008 links for 306 known matches, at precision 0.210. Resolved one-to-one
  in every column instead, with LinkTransformer's threshold tuned for that rule on the same labels
  (0.52 and 0.61), the pretrained model scores 0.673 on Abt to Buy and 0.583 on Amazon to Google,
  against DiffusionGemma's 0.963 and 0.636 and Jev's 0.949 and 0.631 (`one_to_one` in the JSON).
- **Fine-tuning pays on FEBRL4 and not on firms.** On the test-split records the fine-tuned model
  links FEBRL4 perfectly, 1.000 against Jev's 0.957, and scores 0.655 on firms, where the
  pretrained company model scores 0.657 and Jev 0.708. On DBLP to ACM it scores 0.965, against
  0.970 pretrained. On the product sets it scores 0.568 and 0.425 under many-to-many, and 0.927
  and 0.701 one-to-one, where Jev scores 0.939 and 0.670: with labels and one-to-one assignment it
  passes Jev on Amazon to Google, as in the README's comparison.
- **Retrieval sets a ceiling.** The pretrained models' ten candidates per record held 64.3% of the
  known firm links, the same share as jlink's n-gram blocking, and 97.1%, 100%, 95.4% and 95.7% on
  Abt to Buy, DBLP to ACM, Amazon to Google and FEBRL4, where jlink's held 98.7%, 100%, 97.5% and
  100%.

| Models and thresholds | Zero-shot model, revision | Threshold | Fine-tuned from, on matched development pairs | Threshold | Encoding and search, zero-shot / fine-tuned |
|---|---|---:|---|---:|---:|
| Firms | `dell-research-harvard/lt-wikidata-comp-en`, `af65d96` | 0.66 | `lt-wikidata-comp-en`, 880 | 0.75 | 1.0 s / 1.0 s |
| DBLP to ACM | `sentence-transformers/all-mpnet-base-v2`, `e8c3b32` | 0.81 | `all-mpnet-base-v2`, 456 | 0.67 | 2.6 s / 2.7 s |
| Abt to Buy | `sentence-transformers/all-MiniLM-L6-v2`, `1110a24` | 0.73 | `all-mpnet-base-v2`, 236 | 0.67 | 0.7 s / 1.9 s |
| Amazon to Google | `sentence-transformers/all-MiniLM-L6-v2`, `1110a24` | 0.73 | `all-mpnet-base-v2`, 277 | 0.63 | 1.2 s / 2.7 s |
| FEBRL4 | `sentence-transformers/all-MiniLM-L6-v2`, `1110a24` | 0.72 | `all-mpnet-base-v2`, 1,050 | 0.37 | 1.3 s / 3.1 s |

- **Models.** The revisions are those the September 20 comparison recorded in
  `bench/out/lt-compare/<dataset>/linktransformer.json`, loaded from the local Hugging Face cache;
  nothing was downloaded. The zero-shot model is LinkTransformer's company-name model on firms
  and, elsewhere, whichever of MiniLM and MPNet had the higher development F1 in that comparison's
  `report.json` under the dataset's link rule. The fine-tuned model is that comparison's
  `ft_supcon`: LinkTransformer's `train_model` with its default supervised contrastive loss, 10
  epochs, on the matched development pairs counted in the table. The JSON has the SHA-256 of its
  weights. Records became text exactly as on September 20: on the pairs both runs scored, the
  cosines differ by less than 0.000001.
- **Thresholds.** Each is the threshold the September 20 comparison chose for that model and link
  rule by F1 on its development labels, over a 0.01 grid, and for the fine-tuned model on
  development entities held out of its training. `bench/lt_sample.py` chooses them again from that
  comparison's saved development cosines and stops if it gets a different number. On firms
  `report.json` holds none: the firm labels list some matches but not all, so that comparison
  held the number of links equal instead. There the script chooses one the same way, under
  many-to-one, counting unlisted pairs as false links as the F1 of the jlink columns does: 0.66
  for the pretrained model and 0.75 for the fine-tuned one. The jlink columns stay at 0.5 and used
  no labels.
- **Why the test split.** The September 20 comparison divided each dataset 30/70 by entity into
  development and test (seed 1729). Of the 300 sampled left records, 101, 90, 89, 86 and 93
  (firms, DBLP to ACM, Abt to Buy, Amazon to Google, FEBRL4) are in its development split, so the
  fine-tuned model was trained on some of them. On all 300 it scores 0.672, 0.972, 0.591, 0.428
  and 1.000; the JSON keeps these, marked `contaminated_on_sample`. The second table keeps the
  other records and their known links, still searches the whole right table, and rescores Jev,
  DiffusionGemma and Laya from their saved `scores.csv` on those records at 0.5 with the same
  code. The pretrained models' F1 moves by 0.01 or less between the tables.
- **Firm pilot.** The 300 sampled firm records are the ones jlink's rule-wording pilot read
  (`bench/out/live/nber-firms-pilot*`), whose errors led to the one revision of the firm rule that
  the README describes. On firms the jlink columns in both tables have had that one look at these
  records' labels; LinkTransformer's thresholds saw none of the test-split records.
- **Retrieval and resolution.** `merge_knn` with `k=10` matched each sampled record against the
  full right table (2,488, 2,294, 1,092, 3,226 and 5,000 records), 3,000 pairs per model, where
  jlink's n-gram blocking proposed 2,981 to 3,000. Cosines were clipped to 0 to 1, as on September
  20, and resolved by the dataset's rule: many-to-one on firms, one-to-one on DBLP to ACM and
  FEBRL4, many-to-many on the product sets.
- **Environment and timing.** LinkTransformer 0.1.18 with sentence-transformers 5.2.2, torch
  2.10.0, transformers 5.1.0 and faiss-cpu 1.13.2 in `bench/out/lt-env`, on the GPU (MPS).
  `bench/lt_sample.py` started one process per model, one at a time, and read back only the
  cosines and timings. The times are `merge_knn` alone: encoding the 300 records and the right table, building
  the index and searching it, once, with no warm-up. Loading a model took 0.16 to 0.25 s more, a
  whole process with Python and torch starting up 4.8 to 7.3 s, and resolving and scoring 0.03 s
  or less. DiffusionGemma's and Laya's times cover blocking, judging and resolving on servers
  already loaded. The processes ran with `OMP_NUM_THREADS=1`: this environment loads three OpenMP
  runtimes (torch, faiss, scikit-learn), and with more threads the MPNet-based models crashed on
  their first batch, on the GPU and on the CPU.
- **Raw outputs** are in `bench/out/local-2026-09-22/lt/<dataset>/`: the sample and right table
  (`left.parquet`, `right.parquet`, `truth.csv`), each model's cosines (`<model>-native.csv`), its
  links on both bases, its timing and log, and `result.json`, with `lt_sample.log` beside them. To
  reproduce, run `uv run --group bench python bench/lt_sample.py` after `bench/local_models.py`,
  which rewrites the JSON without this block. No Jev, hosted model or paid API was called.

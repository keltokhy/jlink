# Firm linkage under different research definitions

## Observed results, 2026-09-20

All 96 live requests completed with `typesafe/jev-1.13-20260917`. The fixed test set contains
33 supported decisions across 14 pairs in four corporate families. No labels, definitions,
thresholds or inputs were revised after observing these responses.

| Test rule | Rich records correct | Names and dates correct |
|---|---:|---:|
| Legal entity | 8 / 8 | 7 / 8 |
| Corporate group at the record dates | 12 / 14 | 10 / 14 |
| Physical site | 2 / 2 | 2 / 2 |
| Operating business continuity | 8 / 9 | 8 / 9 |
| **All supported decisions** | **30 / 33 (90.9%)** | **27 / 33 (81.8%)** |

Of the nine test pairs requiring different answers under different rules, rich inputs got
**all known rules right for 6 / 9**, versus **4 / 9** with names and dates. Development results
were 15 / 15 and 11 / 15 respectively. The rich arm corrects five test decisions that the
names-and-dates arm misses but introduces two new errors; its net gain is three decisions.
The raw counts are small and correlated, so these percentages are descriptive only.

The name TF–IDF control, using the development-selected threshold 0.6, gets 16 / 33 correct.
A rule-blind method returning one fixed answer per pair can achieve at most 23 / 33 on this
test's conflicting definitions, even with oracle access to the labels. These are limited
controls, not evidence that jlink beats a rule-aware competing system.

### Concrete decision changes

Kyndryl Holdings, Inc. before and after its separation is correctly accepted as the same
legal corporation (p=0.84) and rejected as the same ultimate group at the respective dates
(p=0.33). With names and dates alone, the group question is incorrectly accepted (p=0.96).
The source facts about ownership, rather than name similarity, supply the missing distinction.

For IBM and its pre-separation Kyndryl subsidiary, the rich arm correctly rejects legal
identity (p=0.04) and accepts common corporate group (p=0.82). For the Lordstown factory
transfer, it rejects legal identity (p=0.03) and common group (p=0.18), while accepting the
same physical site (p=0.85).

### All three rich-input test errors

| Pair | Rule | Gold | Jev p | Decision at 0.5 |
|---|---|---|---:|---|
| HP enterprise business before / HPE after separation | Corporate group | No | 0.50 | Incorrect yes |
| HP personal systems and printing / enterprise divisions before separation | Corporate group | Yes | 0.45 | Incorrect no |
| Lordstown / Foxconn factory transfer | Operating business continuity | Yes | 0.46 | Incorrect no |

The Lordstown operating-business error is introduced by richer input; the simpler arm answers
it correctly at p=0.58. The model separates the factory from its corporate owner successfully
but misses continuation of the Endurance manufacturing program. These near-threshold errors
remain errors under the frozen protocol; the threshold was not adjusted to repair them.

The experiment made 96 new calls, no cached decisions, at **$0.001855476 API-reported cost**.
The full repository suite passed **523 offline tests**. Exact saved responses re-evaluate to
the reported metrics, and all artifact hashes have been verified. The pre-run fixture and
protocol are committed at `9f119b5`; [the evidence bundle](../bench/evidence/firm-rules-2026-09-20/)
contains the fixture, requests, provider responses, decisions CSV and report.

## Protocol frozen before collecting model responses

This pilot asks whether jlink can use a research definition to change decisions on the
same information-rich records. It uses 20 curated pairs drawn from primary corporate
disclosures, with 48 supported pair–rule labels. The other 32 labels are explicitly unknown
or inapplicable and excluded, never converted into nonmatches.

Four definitions are frozen in [the fixture](../bench/fixtures/firm_rules.json): legal entity,
corporate group at each record's date, physical site, and continuity of the specified
operating business. Legal identity can persist while corporate control changes across dates.
An operating segment is not automatically an identified legal corporation; an entire company
is not automatically the same operating business as one division.

The same prescribed pairs are evaluated with two inputs:

- **Rich:** record name, date, scope and short source-backed facts, plus a site when established.
- **Names and dates:** only the record name and date, with the same rule and model.

Development contains GE and Tesla/NUMMI (6 pairs); test contains HP, IBM/Kyndryl, Meta and
Lordstown/Foxconn (14 pairs). Model, rules, labels, threshold 0.5 and both arms are fixed before
any responses. No development or test prompt tuning is planned for this run. Every request
contains one pair and one rule. Exact shortcuts and response caching are disabled.

The runner uses the existing `jlink.judge.question`, `_records` serializer and `Jev` client;
an offline test verifies payload equality with the public `judge()` API. It measures the
judgment step on prescribed pairs. It does not evaluate candidate retrieval, assignment or
automated extraction of facts from sources, and it does not change the production matcher.

## Endpoints and controls

Report correct/expected decisions and coverage, broken down by rule and corporate family.
For pairs whose known labels differ across rules, report the number for which **every known
rule** is correctly decided. Merely changing the probability is insufficient.

A character TF–IDF name baseline selects one threshold on development accuracy and applies
that threshold unchanged to all test definitions. Vocabulary/IDF use unlabeled names from
both splits, a transductive retrieval convention. Also report the oracle upper bound for
any method assigning a fixed binary decision per pair across all definitions. That bound
uses gold labels for diagnosis; it is not an executable competing model. Neither control
replaces a comparison against a trained or explicitly rule-aware competing system.

The sequential live run has a $0.25 budget. One final call may cross the budget because its
cost is known after completion. Preserve the exact fixture, timestamped request hashes,
raw provider responses, parsed decisions, resolved model and API-reported costs. Fake
responses used in tests are explicitly marked and cannot be pooled with live evidence.

## Sources and limitations

Each pair lists source references and a rationale separately from the model payload. The
records are factual paraphrases prepared here, not raw registry data. Labels are derived
from disclosures independently of the evaluated model, but have **not** been independently
adjudicated by human annotators. Sources and record facts may be clearer than production data;
the model may also know these public events from training. Cases within a company group are
correlated, and the physical-site test has only two labeled comparisons. Do not treat
individual decisions as independent observations or this pilot as a general accuracy claim.

Examples of the primary evidence:

- [GE's separation filing](https://www.sec.gov/Archives/edgar/data/40545/000004054524000088/ge-20240402.htm)
  distinguishes the retained corporation from the spun-off energy company.
- [HP's results announcement](https://investor.hp.com/news-events/news/news-details/2015/HP-Inc-Reports-Hewlett-Packard-Company-Fiscal-2015-Full-Year-and-Fourth-Quarter-Results/default.aspx)
  distinguishes renaming the retained corporation from separating enterprise operations.
- [IBM's completion notice](https://newsroom.ibm.com/2021-11-03-IBM-Completes-the-Separation-of-Kyndryl)
  documents independent Kyndryl and IBM's retained minority stake.
- [Meta's name-change filing](https://www.sec.gov/Archives/edgar/data/1326801/000132680121000071/fb-20211028.htm)
  identifies the continuing corporation and its two reporting segments.
- [Lordstown's transaction filing](https://www.sec.gov/Archives/edgar/data/1759546/000110465922058868/tm2215185d1_8k.htm)
  identifies the factory transfer and continuation of the Endurance manufacturing program.

## Reproduction

```sh
# Freeze requests and score the offline controls; makes no API calls.
.venv/bin/python -m bench.firm_rules --out /tmp/firm-rules-offline

# Explicit live collection, using a fresh output directory.
.venv/bin/python -m bench.firm_rules --out /tmp/firm-rules-live --live --budget 0.25
```

Offline re-evaluation of a saved run uses `evaluate(fixture, responses)` from
`bench.firm_rules`; the saved responses are JSONL. Do not overwrite evidence directories.

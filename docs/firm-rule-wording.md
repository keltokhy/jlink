# Explicit rule wording experiment

## Result: no improvement on the fresh cases

The frozen comparison completed all 80 fresh-test requests on 2026-09-20. Both wordings
made exactly the same binary decisions: **39 / 40 correct (97.5%)**. For the ten pairs whose
answers differ across research definitions, both got every known rule right on **9 / 10**.

| Fresh-test rule | Current identity wording | Explicit rule wording |
|---|---:|---:|
| Legal entity | 8 / 8 | 8 / 8 |
| Corporate group at the record dates | 16 / 17 | 16 / 17 |
| Physical site | 3 / 3 | 3 / 3 |
| Operating business continuity | 12 / 12 | 12 / 12 |
| **All supported decisions** | **39 / 40** | **39 / 40** |

The alternative **fails the predeclared adoption criterion**, which required a higher correct
count. It remains a benchmark experiment; no new public `judge()`, `Linker`, `link()` or CLI
option is introduced, and production wording stays unchanged. A tie in this small sample
does not establish equivalence on other data.

> **Later note.** `style="rule"` has since been exposed as an opt-in, for a reason this
> experiment did not test: linking by a relation that is not identity, such as a news article
> to the incident it reports, where "refer to the same firm" is the wrong proposition whatever
> its accuracy. See [relation linking](relation-linking.md). The result above stands as
> recorded: no measured improvement, and identity wording remains the default.

Both errors concern `linkedin-acquired`: LinkedIn before and after the Microsoft acquisition
is the same legal entity and operating business, but has different ultimate controllers at
the two record dates. Both styles correctly identify the first two relations yet incorrectly
accept common corporate group: p=0.92 with identity wording and p=0.82 with rule wording.
The payload already states independence before acquisition and Microsoft control afterward;
the failure is not simply missing ownership facts. The threshold stays at the frozen 0.5.

These manually prepared records support a promising use case for rule-dependent linkage,
but neither the 97.5% figure nor this prompt comparison establishes performance on raw data
or superiority over another linking system. A subsequent experiment could extract each
record's dated controller separately and compare normalized controller identities. That is
a hypothesis, not an implemented or validated improvement, and would need its own fresh
cases including renamings, sales, and unresolved ownership.

The fresh run cost **$0.001693104**, with no cached answers, no failed requests and the
resolved model `typesafe/jev-1.13-20260917`. Including the development comparison, this
experiment made 110 calls for **$0.002322600**. The protocol and fresh fixture were committed
at `400fc86` before fresh-test collection. Raw requests, responses, hashes and decisions
are preserved in the [fresh evidence bundle](../bench/evidence/firm-wording-fresh-2026-09-20/)
and [development bundle](../bench/evidence/firm-wording-dev-2026-09-20/).

The fresh report's `collection_complete: true` covers all 80 requested test calls. Its
overall `status: live_partial` and 30 missing responses refer to deliberately uncalled
development requests retained in the same manifest, not failed or omitted test decisions.

Validation: **525 offline tests passed**. All three firm-pilot evidence bundles reconstruct
their frozen requests and replay to their saved metrics; artifact hashes verify, and the
fresh collection's code files still match its frozen code hashes.

## Frozen protocol

The current question begins with `Record A and record B refer to the same firm.` even when
the supplied definition concerns a factory, corporate control or a transferred operation.
The earlier pilot's errors motivated a single alternative prefix:

> Record A and record B satisfy the following match rule.

Both styles append exactly the same definition. Both see identical rich records, use
`typesafe/jev-1.13-20260917`, disable exact shortcuts and cache, and decide at threshold 0.5.
No examples, extra source facts, changed labels, thresholds or rule-specific prompt variants
are introduced. The identity wording remains the default.

On the original six development pairs, both styles correctly answered 15 / 15 supported
decisions, including every rule on all three rule-changing pairs. That satisfies the
development non-regression condition; it does not demonstrate improvement. Thirty calls
cost $0.000629496. The old test set has already been inspected and is not fresh validation.

The fresh confirmation fixture has 17 test pairs and 40 supported labels from five new
families: Kellogg/Kellanova, Kraft/Mondelez, Microsoft/LinkedIn, Hambach/Rastatt factories,
and Google/Alphabet. None of those companies appeared in earlier model runs. Labels and
records are committed before collecting responses. The development records remain in the
fixture for provenance but are not called again in the fresh-test run.

The predeclared decision is to expose the alternative as an opt-in public option only if
it improves fresh-test correct counts and does not reduce the count of rule-changing pairs
with every known rule correct. Retain the production default regardless: a small curated
pilot is insufficient to change it for all users. If the variant fails the condition, retain
it only as a recorded experiment. No retries with revised wording or labels are planned.

## Scope

This remains a judging experiment on manually prepared records, with evidence-derived labels
authored here and no independent human adjudication. Fresh corporate families reuse some
event patterns from development. Outcomes within a corporate family are correlated. Public
events may have appeared in model training. Neither source retrieval nor information
extraction is measured. The three physical-site comparisons are descriptive examples.

New evidence includes [Microsoft's acquisition filing](https://www.sec.gov/Archives/edgar/data/789019/000119312516788575/d259484d8k.htm),
[Mondelez's separation record](https://www.mondelezinternational.com/investors/stock/spin-off-information/),
[Google's reorganization filing](https://www.sec.gov/Archives/edgar/data/1288776/000119312515336550/d56649d8k.htm),
and [INEOS's factory acquisition announcement](https://ineosgrenadier.com/en/se/news/ineos-automotive-confirms-acquisition-of-hambach-production-site-from-mercedes-benz).
All primary references and per-pair rationales are in the fixture. The Hambach records bracket
November 2020 and February 2021 because primary INEOS accounts describe a December announcement
and January acquisition; the pilot does not assume December 8 was the legal closing date.

## Reproduce

```sh
.venv/bin/python -m bench.firm_rules \
  --fixture bench/fixtures/firm_rules_wording_dev.json --partition dev \
  --out /tmp/firm-wording-dev --live --budget 0.10

.venv/bin/python -m bench.firm_rules \
  --fixture bench/fixtures/firm_rules_wording_fresh.json --partition test \
  --out /tmp/firm-wording-fresh --live --budget 0.10
```

Each run freezes requests for both partitions but collects only the requested partition;
check partition-specific coverage rather than interpreting the unused partition as failures.

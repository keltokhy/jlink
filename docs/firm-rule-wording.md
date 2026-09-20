# Explicit rule wording experiment

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

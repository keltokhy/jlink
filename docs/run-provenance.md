# Correctness, budgets, and saved-run provenance

`Result.save(directory)` writes `scores.csv`, `links.csv`, and `settings.json`. On the command
line, `link --save DIR` and `dedupe --save DIR` write the same directory; Stata's `rundir()` and
R's `run_dir =` forward it.
`jlink.load(directory)` can relink these scores without an API call; pass the original input
frames to `merged(left, right)`. Saving does not embed those input frames or API credentials.

## Exact matches

**Exact-text acceptance is disabled by default.** Equal names can refer to different people
or firms. `Linker(..., exact_shortcut=True)`, `jlink.link(..., exact_shortcut=True)`, or CLI
`--exact-shortcut` explicitly asserts that equality on all compared fields establishes identity
under your rule. This changes the earlier default; historical saved scores are unchanged.

When explicitly enabled, the shortcut compares each `on` field separately after cleaning missing values and
normalizing case, accents, punctuation, and whitespace. Every field must normalize to a
nonempty value on both sides. For example:

| Left fields | Right fields | Shortcut |
|---|---|---|
| `Ácme & Sons, Inc.`, `1985.0` | `ACME and SONS INC`, `1985` | Exact, `p=1` |
| `Mary Ann`, `Smith` | `Mary`, `Ann Smith` | Sent to the judge |
| `Mary`, missing | `MARY`, missing | Sent to the judge |
| missing, missing | missing, missing | Sent to the judge |

Whitespace-only and punctuation-only fields count as empty. Missing on both sides is not
evidence of a match. Keep the default when equal text still requires model judgment.
The settings record `exact_policy="all_fields_nonempty_and_equal_v1"`.

## Budget semantics

- `budget=0` prohibits new paid requests. Cached answers and explicitly enabled exact shortcuts remain available.
- `budget=None` allows unlimited new requests.
- A positive budget stops launching new requests once the observed cost reaches the budget.
  New requests are prioritized by descending candidate similarity; cache hits are still read
  after the budget is exhausted, including hits appearing after uncached pairs.
- Negative, infinite, NaN, Boolean, and nonnumeric budgets raise `ValueError` before blocking
  or API setup. `concurrency` must be a positive integer.

**How a positive budget holds.** Each request sets its estimated price aside before it goes
out (`jevkit_runtime.Budget`): the runtime's token estimate for the request, at the dearest rate
the provider has charged so far, or 1.5 times the list price before the first charge. The first
request goes alone, so a price far from the estimate is learned from one call. A request that
does not fit waits for money held by others to come back and is refused only when nothing is in
the air, so requests in flight together cannot pass the limit. Spending can still pass it when a
price rises while requests are in the air, and failed requests may incur provider charges absent
from the local meter. A zero-budget run starts no paid request at all.

Cost is the API-reported amount when present; otherwise jlink estimates it from input tokens
using `JEV_PRICE_PER_MTOK` (default `$0.042` per million). Settings retain the unrounded local
`cost`, the rounded display `dollars`, `cost_sources`, and the estimation rate.
Cached answers contribute no new cost. `cached` retains its historical meaning: cache hits
plus identical requests sharing a call already in flight. Per-score `score_origin`
distinguishes those cases.

Zero-budget runs do not require credentials. If there is no configured key, specify the
original `api` and `model` to select the correct cache namespace. With no explicit provider,
environment setting, or available key, backend selection falls back to TypeSafe's default.
An entirely exact run also needs no key.

## What a new run records

Settings with `provenance_version=1` retain:

- The exact question, its `style` (`identity` or `rule`), definition, field mappings, source ID
  column names, resolution options, record counts, UTC start/end times, duration, and
  jlink/Python/dependency versions. A one-sided field is saved as `["text", null]` or
  `[null, "neighborhood"]`. Runs saved before `style` existed omit it and read as identity.
- Human-readable `blockers` and full structured `blocker_configs`. An omitted blocker list
  records the actual default n-gram parameters; an empty list records no passes.
  When the blocking module supplies `candidates.attrs['blocking']` diagnostics, settings
  preserve them under `blocking` and loading restores them on scores/candidates.
- `inputs.left` and `inputs.right` fingerprints. `compared` covers ordered IDs and raw `on`
  fields; `full` covers ordered IDs and every input column, including grouping fields or
  fields used by custom blockers. These are SHA-256 hashes, not saved input data.
- `exact_shortcut`, its policy version, normalization version, `max_pairs`, concurrency,
  cache use, budget policy, calls, cached answers, retries, tokens, and cost accounting.
- Explicit `requested_api` and `requested_model` arguments (null when omitted), effective
  `provider` and `request_model` after defaults/environment selection, all `resolved_models`,
  `unknown_model_answers`, and grouped `answer_provenance` counts.

A dedupe run writes `clusters.csv` in place of `links.csv`. Its settings add `task: "dedupe"`,
`id`, `n_records`, `linkage`, `unproposed`, `threshold`, `pair_order` (which record the judge saw
as record A) and `blocking.unordered`, and fingerprint the one table under `inputs.records`.
`jlink.load` returns a `DedupeResult` for such a directory; `recluster` reuses its scores.

The input hash format `jlink-input-v1` uses canonical JSON with type tags and field boundaries.
String `"001"` differs from integer `1`; moving words between fields changes the hash; row
order matters because blocking ties can depend on it. Ordinary tabular scalar values retain
their types and values. Missing representations (`None`, NaN, `pd.NA`, `pd.NaT`) share one
missing marker. Other scalar objects use their type and string representation, so custom
objects require a stable string representation for reproducible hashes. Changing unrelated
columns changes `full` but leaves `compared` unchanged. Fingerprints are recorded for comparison;
loading and merging do not automatically verify source frames against them.

For each model-scored pair, the scores table also retains:

| Column | Meaning |
|---|---|
| `model` | Model identifier reported by the API that originally answered; missing if unknown |
| `provider` | Original answering API provider, including for cache hits; missing for legacy entries |
| `score_origin` | `api`, `cache`, or `shared` (another request already in flight) |
| `answered_at` | Original response time as Unix seconds; missing if unknown |

These columns are empty for exact, failed, and unjudged pairs. API-reported identities are
preserved literally: if a provider returns an alias, jlink cannot prove the underlying model
revision. Cache hits retain the original response identity and time. A run can contain several
model versions under one requested alias; `resolved_models`, per-score identities, and the
methods/report text show them all. The compatibility `model` setting and `Meter.model` contain
a single identity only when every returned model answer has that same known identity; otherwise
they are empty. Exact-only runs have no model answer identity to report.

## Blocker serialization protocol

A blocker may implement `to_config() -> dict` returning a JSON-compatible configuration.
jlink preserves this representation, including nested blockers and options such as grouping,
missing-group policy, and reverse direction. Custom implementations should explicitly state
whether their configuration is complete and reconstructable.

For older built-ins without `to_config`, jlink recursively records declared dataclass fields
under `parameters`. This includes `fields`, `name`, `k`, `n`, `min_sim`, and `min_len`, without
guessing constructor arguments from a blocker's display name. Nested blockers use the same
protocol. An opaque custom blocker records its class/name with `configuration_complete=false`
and `reconstructable=false`. This fallback describes a pass; it does not claim a constructor
can replay arbitrary custom code. Custom dataclass fields containing opaque objects are listed
under `unserialized_fields`, with `configuration_complete=false`. Loading results never imports
or runs serialized blockers.

## Backward compatibility and unknown provenance

Old result directories remain loadable. String, integer and floating-point ID types are
retained, so a loaded result still merges with source tables whose numeric IDs are stored as
doubles, as Stata often stores them. Float IDs saved before this was recorded load as text,
as they did then. Literal IDs `"NA"`, `"NULL"`, `"nan"`, empty strings, and leading-zero
strings are not treated as missing scores. Numeric score columns and optional error/provenance columns handle empty CSV cells
separately; literal error messages `"NA"` and `"NULL"` survive. Floating-point scores use the
round-trip CSV parser so reloading does not shift a threshold boundary by a rounding unit.
The saved format records `result_format_version=2` alongside the historical `id_kinds`.

Legacy settings cannot establish whether their single model label describes every cached
answer. Loading flags `model_identity_status="legacy_unverified"` and retains the original
label without creating requested-provider, resolved-model, blocker-parameter, or input-hash
metadata that was never recorded. Legacy exact decisions also remain as saved; loading does
not retrospectively apply the corrected exact policy. Relinking only reuses saved scores.

Answers are stored by `jevkit-runtime` (`~/.cache/jev/answers.v3.sqlite` from runtime 0.4), keyed on
the provider, endpoint, requested model and exact question and records, with each answer's provider,
resolved model and time in the same row. `settings["run"]` holds the runtime's record of the judging:
the backends, the models that answered and how often, the question as asked, calls, tokens, cost
and the budget.

## Verification scope

Offline tests cover field-boundary collisions, missing fields, normalized exact matches,
zero/unlimited/invalid budgets, post-budget cache hits, a first charge dearer than the budget, mixed model versions, old saved results, blocker serialization, input fingerprints,
and saved IDs/probabilities/errors through relinking and merging. Existing saved results for
NBER firms, DBLP–ACM, Abt–Buy, Amazon–Google, and FEBRL4 were also loaded, saved, relinked, and
merged without API calls: all 146,119 score rows and the existing link sets were preserved.
This checks compatibility with frozen artifacts; it is not a new live-provider test or a new
measurement of linkage accuracy, API latency, or actual provider billing.

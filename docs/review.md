# Local human review

jlink can generate a standalone review page from a `Result` and the original records.
It opens as a local HTML file, uses no hosted service or remote assets, and makes no API
calls. The review artifact contains the original records, scores, links and settings,
plus the decision history. Keep the JSON file with your replication materials.

## Start from Python

```python
import jlink

# result = linker.link(...), or jlink.load("linkage/")
review = jlink.create_review(result)  # uses the records attached to a live Result
# For a loaded Result:
# review = jlink.create_review(result, left=original_left, right=original_right)
review.write_html("review.html")
review.save("review.json")           # optional initial copy
```

Open `review.html` in a browser. Enter a reviewer name or researcher ID, inspect the
side-by-side fields, and select **Accept**, **Reject** or **Unsure**. Add a note before
clicking a decision to explain the evidence or amendment. Each click is a new event;
changing a decision preserves the earlier decisions, identities, notes and timestamps.

The queue includes every record on both sides, including those with no proposed pair.
Switch the **Record table** control to review right-side records and competitors.
Close competitors (a top-two probability gap at most `close_margin`, default 0.1) come
first, followed by originally unmatched records. Search and filters narrow the queue.
Large queues and candidate lists have “show more” controls.

The page distinguishes:

- **Original link:** actual membership in `Result.links`, irrespective of threshold.
- **Not originally linked:** a candidate excluded from that result's assignment or margin,
  or below its threshold. An above-threshold pair is not automatically a selected link.
- **No candidate:** blocking proposed no pair for the record. There is nothing to judge on
  this page; changing candidates requires a separate run.
- **Unjudged candidate / judgment error:** probability is missing. This is not a rejection.
- **Manually rejected:** the latest human event explicitly rejects the pair.
- **Accepted constraint / unsure:** pending manual instructions, separate from original selection.

The page does not run the assignment algorithm. It shows the original selection and
pending constraints; export and apply to compute a new assignment.

## Export, import and amend

**Export review JSON** downloads a complete `review.json`, with the source snapshot and
all history. **Import review JSON** merges another artifact from the same snapshot.
Existing events cannot be silently changed; independent decisions on different pairs
merge, while incompatible branches of the same pair's history are refused. To reconcile
such a branch, keep both exports, load the shared history and record an explicit new
amendment with a note explaining the disagreement. Timestamp order never automatically
chooses a reviewer whose decision wins.

Browser storage is keyed by the source snapshot ID. Reloading the same page restores its
history when storage is available; writes merge compatible history from other open tabs.
Storage failures and conflicts are visible. Storage availability for local files depends
on the browser, so the exported JSON is the durable artifact. A fresh page can be generated
from that JSON:

```bash
jev-link review page review.json -o reopened-review.html
```

Programmatic decisions use exactly the same schema and validation:

```python
review = jlink.read_review("review.json")
review.decide("001", "A", "accept", reviewer="researcher-17", note="Confirmed in registry")
review.decide("001", "A", "unsure", reviewer="researcher-17", note="Registry entry ambiguous")
review.merge(jlink.read_review("colleague-review.json"))
review.save("amended-review.json")
```

Decisions must reference an existing candidate. The page cannot invent new pairs.
Reviewer names are self-reported, and timestamps use the reviewer's device clock.

## Apply constraints offline

```python
reviewed = jlink.read_review("amended-review.json").apply()
# Optional overrides; omitted values retain the source run's settings:
reviewed = review.apply(how="one-to-one", threshold=0.7, min_margin=0.1)
# min_margin=None explicitly removes the margin requirement.
reviewed.save("reviewed-links.json")
reviewed.write_csv("reviewed-links.csv")  # optional spreadsheet-safe display copy

reviewed.links           # recomputed links, unchanged p, and explicit review metadata
reviewed.scores          # original candidate scores
reviewed.original_links  # original links
reviewed.review          # full source snapshot and decision history
reviewed.settings        # applied cardinality, threshold, margin, and manual policies
```

Application follows these rules in order:

1. The latest event in each pair's amendment chain is authoritative. **Unsure** releases
   the previous accept/reject constraint, so the pair again participates automatically.
2. **Accept** pins the pair. A pinned pair bypasses the threshold and margin, including
   when its original probability is missing. Its `p` and `source` remain unchanged.
3. Conflicting pinned pairs raise `ValueError` before producing output: `one-to-one`
   requires unique IDs on both sides, `many-to-one` unique left IDs, `one-to-many` unique
   right IDs, and `many-to-many` permits either side to repeat. No accepted pair is silently
   discarded. Amend decisions or explicitly select different cardinality.
4. Remove rejected pairs and candidates made infeasible by pinned pairs. Run the existing
   resolver on the remaining candidates with the declared settings, without judging again.
5. Combine pinned and automatically selected pairs. Automatic margins use all judged
   **remaining feasible candidates**, including below-threshold competitors. Pinned pairs
   retain their margin against **original candidates**, solely as diagnostic information.

The existing resolver's behavior remains in effect: its margin filter runs after assignment
and does not refill links removed by that filter. Its documented large-component greedy
fallback and warnings also remain in effect. Review does not change statistical evaluation.

Each output link has `selection_basis` (`manual_accept` or `automatic`),
`original_selected` (Boolean), `review_decision`, `review_event_id`, and `review_run_id`.
An automatic pair whose last decision was “unsure” retains that event ID. The JSON output
also includes the complete rejected/unsure history, so excluded pairs remain traceable.
`ReviewedLinks` is separate from `Result`: the original model's methods/report text is not
reused to describe a manually constrained assignment.

For integrations requiring pair-level selection, `review.candidates` returns a copy of
original scores with a Boolean `selected` column derived from original link membership.
`reviewed.selected_scores()` instead derives Boolean `selected` from the **reviewed** links.
Both use `(left_id, right_id)` membership, never `p >= threshold`; neither mutates the
original scores. To audit reviewed links with model scores, draw a sample with
`jlink.audit_sample(reviewed.scores, links=reviewed.links)`, fill its `is_match` labels,
then call `jlink.evaluate(labeled, mode="selected")`. Review itself neither samples nor
estimates accuracy. The score-stratified audit requires a probability for every selected
link and rejects manually accepted unjudged pairs. Such pairs need a separate audit that
includes links without model scores; do not assign them fabricated probabilities.

## Command line

`jev-link` and `python -m jlink` avoid the macOS Java executable named `jlink`.

```bash
# A directory previously written with Result.save(), plus the original tables:
jev-link review create linkage/ --left firms.csv --right registry.parquet \
  -o review.html --artifact review.json

# After reviewing and exporting in the browser:
jev-link review apply exported-review.json -o reviewed-links.json --csv reviewed-links.csv

# Explicit settings overrides, without any API calls:
jev-link review apply exported-review.json -o strict-reviewed-links.json \
  --how one-to-one --threshold 0.8 --min-margin 0.2
# Use --no-margin to remove the original margin requirement.
```

The CLI create adapter uses `jlink.load` to preserve saved probabilities exactly, together
with literal `NA`/`NULL`, leading-zero string IDs, and recorded provenance. It honors saved
integer ID metadata and checks candidate IDs against the original records. JSON-only
`page` and `apply` do not require the saved-run directory or the original table files.

Output paths cannot replace input files. Invalid artifacts and acceptance conflicts exit
with status 2 and an actionable message. A conflict is detected before output is written.

## Artifact contract and data safety

`jlink-review`, version 1, has these top-level fields:

| Field | Contents |
|---|---|
| `format`, `version` | Explicit format and schema version |
| `run_id` | SHA-256 fingerprint of the complete embedded source snapshot |
| `created_at` | Artifact creation time with timezone |
| `source` | Original settings, score/link tables, both original record tables, queue gap |
| `history` | Immutable events linked through `previous_event` within each pair |

Each event contains `event_id`, `previous_event`, `left_id`, `right_id`, `decision`,
`reviewer`, `timestamp`, and `note`. Every timestamp must include a timezone. Each ID is
encoded as `{"type":"str|int|float","value":"lossless text"}`. This preserves leading zeros,
literal `NA`/`NULL`, numeric types, and integers larger than JavaScript's exact numeric range.
Missing, Boolean and unsupported ID types are rejected. Non-ID record values are embedded
for display; non-JSON scalar values and very large numeric display values become strings.

The run ID identifies this exact review snapshot, including original link membership and
record fields. It does not assert a new saved-run provenance contract, distinguish two
otherwise identical executions, or act as a digital signature. Editing an artifact and
recomputing its fingerprint is not prevented. Reviewer authentication and tamper-evident
signing would require a separate system.

All record text, IDs, field names, settings, notes and reviewer identities are rendered as
text nodes. Embedded JSON escapes HTML-significant characters and Unicode separators.
The page has a restrictive content security policy and loads no remote resources. Exported
JSON is authoritative and preserves IDs. Optional CSV output quotes fields normally and
prefixes formula-like strings/headers with an apostrophe to prevent spreadsheet formula
execution. Thus a dangerous string ID's display representation can differ in CSV; use the
JSON artifact when exact IDs matter. Reopening CSV in a spreadsheet can also strip leading
zeros unless its columns are imported as text.

The HTML/JSON snapshot contains **all columns of both original record tables**, not just
`on` fields. Treat it as a copy of those records. This implementation loads the snapshot
in browser memory; very large tables may require smaller review snapshots or a future
paged data store. No hosted service, runtime dependency or frontend framework is added.

## Offline demo and verification

```bash
uv run python examples/review_demo.py /tmp/jlink-review
# Open /tmp/jlink-review/review.html, review pairs, export JSON, then:
uv run jev-link review apply /path/to/exported/review.json -o /tmp/jlink-review/applied.json
uv run pytest tests/test_review.py tests/test_review_cli.py
```

The synthetic demo includes a globally selected lower-probability competitor, an unjudged
candidate, a below-threshold candidate, and records with no candidate on both sides. Its
probabilities are explicitly example data, not measured model performance.

Verification on 2026-09-18 used Chromium via Playwright at 1440×1100 and 390×844. The full
flow covered all three decisions, amendment history, surfaced/resolved acceptance conflicts,
no-candidate and unjudged states, JSON download, fresh-session import, reload, different-run
import refusal, script-injection text, and offline application. There were no application
console errors or remote requests. Browser storage is a convenience; other browsers and
very large review histories have not been tested.

A read-only check of the existing cached Abt-Buy run used 2,173 original records and
10,796 candidate pairs. No-decision recomputation reproduced all 1,134 original links.
On the development machine, snapshot creation plus HTML/JSON writing took 0.330 seconds,
application took 0.117 seconds, and the HTML file was 3,012,770 bytes. These are one-run
local measurements of this review implementation, not general performance guarantees.
No new model calls were made.

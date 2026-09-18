# Firm-name benchmark investigation

Checked 2026-09-18, in the requested order: NBER first, then other company-matching sources.
The source search and archive inspection stayed within the one-hour limit. A real labeled NBER
file is downloadable, so `nber-firms` uses it. No company names or labels were invented.

## 1. NBER Patent Data Project and its predecessor

The [PDP assignee-to-Compustat page](https://users.nber.org/~jbessen/match.htm) links
`pdpcohdr.dta` and `dynass.dta`. Both file URLs returned HTTP 404 during this run. The
[current PDP downloads page](https://sites.google.com/site/patentdataproject/Home/downloads)
was reachable, but its converted embedded-files section did not expose downloadable files.
That page describes community research access and encourages error reports and sharing
supplemental links; it does not specify a standard open-data license.

The [PDP user documentation](https://users.nber.org/~jbessen/matchdoc.pdf) explains why a firm
definition matters here: assignments include subsidiaries, reorganizations, and changes of
corporate owner. The later project uses company identifiers and year intervals to connect
assignees to Compustat. A name-only identity benchmark cannot treat every such association as
the same legal entity. We did not reconstruct the newer dynamic project from the older file.

The predecessor [NBER U.S. Patents, 1975–1999 page](https://www.nber.org/research/data/us-patents-1975-1999)
does supply a public ASCII archive, [amatch.zip](https://data.nber.org/patents/amatch.zip).
It downloaded successfully without authentication. Its [data dictionary](https://data.nber.org/patents/match.txt)
identifies assignee names, Compustat names, six-character CUSIPs, and parent/subsidiary fields,
and attributes the matching to Bronwyn Hall and associates. The archive contains `match.csv`
with 4,906 rows. These are directly observed source-file counts, not the later PDP release.

**Terms:** no explicit redistribution license was found on this older data page or its
dictionary. The listed underlying sources include USPTO and Compustat. Public download is
established; unrestricted redistribution is not. We provide a source downloader and metadata,
and keep the downloaded and derived data git-ignored. Cite Hall, Jaffe and Trajtenberg (2001),
*The NBER Patent Citation Data File: Lessons, Insights and Methodological Tools*, NBER WP 8498.

## What was built

`uv run python bench/prepare.py nber-firms` pins the original archive's SHA-256 to
`98d89982c9d09c3b07cb5f92b98e669d4afc22b9b1d71d8d57e1c7acaa017cfe`.
The ZIP dates its CSV member to 2002-08-29; this is an archive timestamp, not a claim about
publication or the date of any ownership relation. Its malformed ZIP extra metadata is
decoded through a checksum-guarded standard-library fallback; the CSV length and CRC are verified.

The transformation keeps rows with nonempty assignee ID/name and CUSIP/company name, retains
one record per assignee and CUSIP, and uses the most frequent source spelling, with lexical
tie-breaking. All observed spellings remain in `source_names` for inspection. Only `name` is
used for matching; identifier and parent/ownership information are excluded from `on`.

| Construction check | Count |
|---|---:|
| Original crosswalk rows | 4,906 |
| Completely repeated source rows | 2 |
| Repeated assignee-ID rows beyond the first | 9 |
| Incomplete rows excluded | 312 |
| Complete rows before deduplicating pairs | 4,594 |
| Repeated complete assignee/CUSIP pairs removed | 3 |
| Unique left assignees | 4,585 |
| Unique right CUSIPs | 2,488 |
| Distinct truth pairs | 4,591 |
| Left IDs with more than one labeled right match | 6 |
| Right IDs with more than one labeled left match | 686 |
| Retained assignees with multiple observed names | 3 |
| Retained CUSIPs with multiple observed names | 41 |

Every truth endpoint exists in the corresponding prepared source; source IDs are unique.
The metadata definition is:

> A patent assignee matches a Compustat company when the historical NBER assignee-to-Compustat
> crosswalk assigns it to that company, including subsidiary-to-parent links. This is the
> crosswalk's historical corporate association, not necessarily the same legal entity or its present owner.

`how` is `many-to-many`, preserving the source's multiple associations. Similarity baselines
still select one right per left, as required; this restricts attainable recall slightly.
The entire left side fits within 45,850 potential `k=10` pairs. A seed-0 sample of 500 left
records is more suitable for a cents-scale first live run.

The main weaknesses are historical ownership ambiguity, incomplete labels, selection into
patenting and Compustat, and a right universe derived from the labeled crosswalk. Missing
links are not verified negative labels. Scoring assumes unlisted pairs are negative, and the
prepared set contains no known unmatched left records. Choosing names from the crosswalk
also makes this a benchmark of published associations rather than independent source cleaning.
A subsidiary can have no words in common with its parent; names alone cannot reliably recover
every labeled link. These are substantive limitations, not model errors that threshold tuning fixes.

## 2. Other public candidates inspected

| Source | What exists and terms | Decision |
|---|---|---|
| [Dyèvre–Seager Compustat–patent links](https://github.com/arnauddyevre/compustat-patents) | Static and dynamic patent-to-GVKEY outputs under CC-BY-4.0; source Compustat annual data described as proprietary. | Useful economic crosswalk, but published outputs do not supply a standalone pair of company-name tables. |
| [OpenSanctions matching judgments](https://www.opensanctions.org/docs/opensource/pairs/) | Human positive/negative/unsure judgments across entity types. Grouped data require a delivery token; the page describes an older flat release and links CC-BY-NC-4.0 terms. | A credible future company subset, but the current delivery is credentialed, the old file is over 2 GB, and sparse pair labels do not establish complete cross-table truth. No token was sought or read. |
| [ING EntityMatchingModel](https://entitymatchingmodel.readthedocs.io/en/latest/emm.data.html) | Helpers retrieve Dutch KVK/LEI names and generate noised matching examples. | Useful synthetic robustness tests, not independently labeled cross-source company matches. Source-data terms were not established here, so none were downloaded or repackaged. |
| [TurnToData ER-7](https://turntodata.ai/benchmarks/er7) | Describes GLEIF-derived alias/damaged-name pairs, claims CC0 and publishes a generator command and seed. | The inspected page exposed no direct artifact/repository download link. Availability was not verified beyond the documentation, and these are registry-derived labels rather than independently adjudicated economic links. |

## Best next construction for legal-entity identity

This is a proposal, not another delivered or scored dataset. Use a dated
[GLEIF Golden Copy](https://www.gleif.org/en/lei-data/gleif-golden-copy/download-the-golden-copy)
and its canonical and other entity names to form two name tables, linked by LEI. The
[GLEIF open-data listing](https://registry.opendata.aws/lei/) specifies CC0. Keep true aliases,
retain same-country and similar-name distractors, exclude duplicate/merged registrations,
withhold identifiers from the matching fields, and split by entity before developing any model.
Review a stratified sample by hand before treating different LEIs as verified negatives.

Such a construction has clearer legal-entity semantics and easier redistribution than this NBER
file. It remains a registry-derived silver standard: alias coverage is selective, names share a
source, registration defects can corrupt labels, and firms with LEIs are not representative of
all businesses. Parent-subsidiary links would be a separate task with a separate definition.

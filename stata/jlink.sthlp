{smcl}
{* Tested with real Stata/SE 19.5 on macOS and a fake CLI; no real Linker or API call.}
{title:jlink -- Link firm, person, or product records using a plain-English rule}

{p 8 12 2}
{cmd:jlink using} {it:right.dta}{cmd:, on(}{it:name city}{cmd:) entity(}{it:firm}{cmd:)}
[{cmd:define(}{it:text}{cmd:)} {cmd:style(}{it:identity|rule}{cmd:)} {cmd:leftid(}{it:varname}{cmd:)}
{cmd:rightid(}{it:column}{cmd:)} {cmd:how(}{it:mode}{cmd:)}
{cmd:threshold(}{it:number}{cmd:)} {cmd:budget(}{it:dollars}{cmd:)}
{cmd:saving(}{it:links.dta}{cmd:)} {cmd:replace} {cmd:merge} {cmd:rundir(}{it:folder}{cmd:)}]

{title:Setup}
{pstd}Requires Stata 14 or later on macOS or Unix, and the Python jlink package.
Place jlink.ado and jlink.sthlp together on your adopath. The command looks for
{cmd:jev-link} on PATH, then {cmd:python3 -m jlink}. On macOS, {cmd:/usr/bin/jlink}
is an unrelated Java tool; use {cmd:jev-link} or {cmd:python -m jlink} in a terminal.
Configure the chosen API provider before running real links.{p_end}

{title:Inputs and options}
{pstd}The data in memory are the left dataset. {cmd:using} accepts a CSV, TSV,
Stata (.dta), or Parquet file; Parquet requires Python's pyarrow package.
{cmd:on()} lists fields separated by spaces. Use {cmd:city=town} when names differ.
A field that only one dataset has is written {cmd:text=} (data in memory only) or
{cmd:=place} (using dataset only); it is shown to the model and never compared.
{cmd:entity()} describes the kind of record. {cmd:define()} adds your match rule.
{cmd:style(rule)} asks whether a pair satisfies {cmd:define()}, which may state any
relation between the two records; {cmd:entity()} is then optional and {cmd:define()}
is required. The default, {cmd:style(identity)}, asks whether both records are the
same {cmd:entity()}.
Quote paths with spaces; use Stata compound double quotes around definitions
that contain quotation marks.{p_end}
{pstd}{cmd:leftid()} and {cmd:rightid()} name unique IDs. Without them the CLI uses
zero-based row numbers. String IDs retain leading zeros; Stata value labels do
not replace the underlying numeric IDs.{p_end}
{pstd}{cmd:how()} defaults to {cmd:one-to-one}, unique on both sides.
{cmd:many-to-one} allows several left records to share a right record;
{cmd:one-to-many} is its mirror; {cmd:many-to-many} keeps all qualifying pairs.
{cmd:threshold()} is a minimum probability from 0 to 1 (default 0.5).
{cmd:budget()} is the maximum API spend in US dollars (default 5).{p_end}
{pstd}{cmd:saving()} writes a Stata links dataset, including {cmd:left_id},
{cmd:right_id}, {cmd:p}, and any score columns. Add {cmd:replace} to overwrite it.
Links are read with text IDs, preserving leading zeros.
Without {cmd:saving()} or {cmd:merge}, the wrapper only reports the number of links.{p_end}
{pstd}{cmd:rundir()} saves the whole run in a folder: links.csv, scores.csv with every
candidate pair's probability, and settings.json with the question, blocking and
provenance. Keep it with your replication files. In a terminal,
{cmd:jev-link review create} {it:folder} {cmd:--left} {it:left.dta} {cmd:--right} {it:right.dta} {cmd:-o review.html}
then builds the local review page from it.{p_end}
{pstd}Your data in memory remain unchanged unless you request {cmd:merge}.
{cmd:merge} requires {cmd:leftid()} and {cmd:how(one-to-one)} or
{cmd:how(many-to-one)}. It attaches link columns to the left data and retains
unmatched left records. It refuses to overwrite existing column names.
No right-side source fields are attached. Failed commands restore the original data;
the CLI's diagnostic is shown in Stata.{p_end}

{title:Examples}
{phang2}{cmd:. use "survey firms.dta", clear}{p_end}
{phang2}{cmd:. jlink using "company registry.dta", on(name city=town) entity(firm) leftid(gvkey) rightid(id) saving("firm links.dta")}{p_end}
{phang2}{cmd:. jlink using "company registry.dta", on(name) entity(firm) define(`"The same firm, including "Inc." abbreviations."') leftid(gvkey) rightid(id) merge}{p_end}

{title:Returned results and auditing}
{pstd}{cmd:r(N_links)} contains the number of links; {cmd:r(saving)} contains the
requested output path. To save all scores, specify custom blocking, or perform
an audit, use {cmd:jev-link --help} in a terminal. {cmd:jev-link audit} draws a
sample for hand checking; {cmd:jev-link evaluate} reports accuracy. Audit recall
covers candidate pairs and cannot detect matches lost during blocking.{p_end}

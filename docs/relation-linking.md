# Relation linking: rule-style questions and one-sided fields

jlink was built for identity: two records, one firm. Some linkages ask a different question.
A news article is not a police incident, a patent is not a product, a parent is not its
subsidiary, yet each pair can stand in a relation that a sentence defines. This page covers
the two options that make such a linkage expressible: the question style and one-sided fields.

## The question

| `style` | Proposition sent to Jev | `entity` | `definition` |
|---|---|---|---|
| `"identity"` (default) | `Record A and record B refer to the same <entity>. <definition>` | required | optional |
| `"rule"` | `Record A and record B satisfy the following match rule. <definition>` | optional, not in the question | required |

```python
linker = jlink.Linker(
    style="rule",
    definition="Record A is a news article that reports the shooting incident in record B.",
    on=[("text", None), ("published", None), (None, "occurred"), (None, "neighborhood")],
    blockers=[...],
)
```

```bash
jev-link link articles.csv incidents.csv --style rule \
    --define "Record A is a news article that reports the shooting incident in record B." \
    --on text= --on published= --on =occurred --on =neighborhood \
    --left-id article_id --right-id incident_id --how many-to-one --block ... -o links.csv
```

`Linker`, `jlink.link`, `jlink.judge`, the `link` command, Stata's `style()` and R's `style =`
all accept the option. Under rule style an empty definition is an error: there would be nothing
to ask. Under identity style an empty entity is an error, as before. Record A is always the
left record and record B the right one, so a rule can refer to them by those names.

The style is part of the question text, and the answer cache is keyed on the model, the state
and the question. An identity answer therefore never serves a rule-style run on the same
records, or the reverse. `tests/test_relation.py` checks this with one shared cache.

## One-sided fields

Each `on` item is one of:

| Python | Command line | Shown to the judge |
|---|---|---|
| `"state"` | `--on state` | on both records, labeled `state` |
| `("conm", "assignee")` | `--on conm=assignee` | on both records, labeled `conm` |
| `("text", None)` | `--on text=` | on the left record only, labeled `text` |
| `(None, "neighborhood")` | `--on =neighborhood` | on the right record only, labeled `neighborhood` |

A paired field takes the left column's name on both records. That suits identity, where both
columns hold the same kind of thing. For a relation the two columns often mean different
things: an article's `published` date and an incident's `occurred` date. List those as two
one-sided fields so that each keeps its own name in front of the judge, and pair them only
inside a blocking pass, which never shows anything to the judge.

Rules that follow from this:

- Every record needs at least one field, and a record cannot show two fields with one label.
- Blocking passes compare a left column with a right column. `exact`, `ngrams`, `initials`,
  `embeddings` and `within` reject a one-sided field and say why. `--block` keeps the stricter
  `name` or `left=right` grammar.
- With `blockers=None`, the default n-gram pass searches the paired fields only. If every field
  is one-sided there is no default; choose the passes yourself.
- Candidate `sim` is still character TF-IDF cosine between the two records' text. Each side's
  text is now its own `on` columns, paired or one-sided. With paired fields only, nothing
  changes. `sim` orders judging under a budget; it is not evidence of a match.
- `exact_shortcut=True` needs every field on both sides and refuses one-sided fields.
- Saved settings write a one-sided field as `["text", null]` or `[null, "neighborhood"]`.
  Input fingerprints, `audit_sample` (which adds `a_text` without a `b_text`), `evaluate` and the
  local review page all accept them.

## What the run records

`settings.json` gains `style`; `question` holds the exact proposition; `entity` is null when
none was given. Runs saved before this option have no `style` and read as identity. `report()`
prints the question as before and adds a `Question style: rule` line. `methods()` says that a
link is a relation defined by a written rule and not a claim that both records describe the
same entity, lists which source each one-sided field came from, and quotes the proposition.
`relink`, `save` and `load` carry the style along.

## Limits

- Do not rely on Jev to compare numbers or dates. jlink treats that as a known weak spot of the
  model and has not measured it. A rule such as "published within three days of the incident"
  belongs in blocking, where the arithmetic is exact and only plausible pairs are asked about.
- Jev reads the fields it is given and returns a probability. It does not extract a name, an
  address or a date from the text, and it never generates text.
- The `estimate` figures assume about 330 input tokens per pair, which was measured on short
  firm records. Pairs that carry article text use more tokens and cost more than estimated.
- Nothing here measures Jev on article-to-incident pairs or any other relation. Whether it
  judges yours well is an empirical question: audit a sample before relying on the links.

# Changelog

## Unreleased

- Add `style="rule"` to `Linker`, `jlink.link`, `jlink.judge`, the `link` command (`--style rule`)
  and the Stata and R wrappers. The proposition becomes "Record A and record B satisfy the
  following match rule. <definition>", so a link can be a relation that is not identity. `entity`
  is optional under this style and a definition is required. Settings record the style;
  `report()` names it and `methods()` describes a relation. Identity remains the default, and
  identity and rule answers never share a cache entry.
- Let an `on` item be one-sided: `("text", None)` or `(None, "neighborhood")` in Python, `--on text=`
  or `--on =neighborhood` on the command line. Such a field is shown to the judge on that side
  only and enters that side's text for `sim`. Blocking passes and the exact shortcut need paired
  columns and say so. Saved settings, fingerprints, audit samples and review snapshots accept them.
- Add `jlink.block.window`: candidate pairs whose left value minus right value lies within a
  tolerance or a one-sided range, for numbers and for dates and times. It sorts once and uses
  binary search, streams bounded batches under `max_pairs`, composes with `within` and unions
  with other passes. Missing and unreadable values are dropped and counted, never guessed;
  non-ISO dates need `date_format`. Blocking diagnostics record the counts as `dropped_values`.
- Add `--block window:COLUMN:TOLERANCE`, `--block window:LEFT=RIGHT:LOW..HIGH` with `w/d/h/m/s`
  units, `--block within:COLUMNS:RULE`, and `--date-format`. `estimate` reports records a window
  cannot use.

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

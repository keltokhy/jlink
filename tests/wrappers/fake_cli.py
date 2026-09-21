"""Offline CLI stand-in; launched only by run.py's temporary executable shims."""

import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd


def main():
    args = sys.argv[1:]
    assert args and args[0] == "link", args
    left, right = map(Path, args[1:3])
    options = {}
    for key, value in zip(args[3::2], args[4::2]):
        options.setdefault(key, []).append(value)
    assert len(args[3:]) % 2 == 0, args
    definition = options.get("--define", [""])[0]
    record = {"args": args, "left": str(left), "right": str(right), "output": options["-o"][0]}
    with open(os.environ["JLINK_TEST_LOG"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    if definition == "FAIL":
        print("jlink: registry.dta: column 'name' is missing (offline failure)", file=sys.stderr)
        raise SystemExit(2)
    if definition and definition != "EMPTY":
        assert definition == os.environ["JLINK_TEST_DEFINE"], repr(definition)
    assert left.exists() and right.exists()
    if left.suffix == ".dta":
        frame = pd.read_stata(left, convert_categoricals=False)
    else:
        frame = pd.read_csv(left, dtype=str)
    assert frame.firm_id.astype(str).tolist() in (["00123", "00456"], ["123", "456"])
    assert frame.name.tolist() == ["Acme & Sons", "Other firm"]
    other = (pd.read_stata(right, convert_categoricals=False) if right.suffix == ".dta"
             else pd.read_csv(right, dtype=str))
    assert other.record_id.iloc[0] == "00007"
    assert options["--on"] == ["name", "city=town"]
    assert options["--left-id"] == ["firm_id"] and options["--right-id"] == ["record_id"]
    style = options.get("--style", ["identity"])
    assert style in (["identity"], ["rule"]), style
    # A rule-style call states the relation in --define and may omit --entity.
    assert options.get("--entity") == (None if style == ["rule"] else ["firm"]), options
    assert style == ["identity"] or definition
    with open(options["-o"][0], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left_id", "right_id", "block", "sim", "p", "source", "error", "margin"])
        if definition != "EMPTY":
            writer.writerow([frame.firm_id.iloc[0], "00007", "ngrams:name", "0.9", "0.98", "jev", "", "0.6"])
    print("jlink: 2 left records, 2 right records; 2 candidate pairs; 1 links; 0 calls; $0.0000",
          file=sys.stderr)


if __name__ == "__main__":
    main()

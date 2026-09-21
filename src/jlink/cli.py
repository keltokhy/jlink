"""Commands for linking records and reporting how well the links work."""

from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

from . import __version__
from .fields import check_columns, ids, parse_on
from .io import FORMATS, read_table, stata_value_labels, write_table

_BLOCK_FORMS = ("ngrams:name:10, embeddings:name:10, ngrams:name+city:20, exact:state, initials:name, "
                "window:year:1, window:published=occurred:0..3d, within:state:ngrams:name:10")
_HOW = ("one-to-one", "many-to-one", "one-to-many", "many-to-many")
_NUMBER = r"[+-]?\d+(?:\.\d+)?"
_WINDOW = re.compile(rf"(?:(?P<low>{_NUMBER})\.\.(?P<high>{_NUMBER})|(?P<tolerance>{_NUMBER}))"
                     r"(?P<unit>[wdhms])?")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _column(value: str, *, unpaired: bool = False) -> str | tuple[str | None, str | None]:
    """name, left=right, or for --on only: left= (left records only) and =right (right records only)."""
    parts = value.split("=")
    blank = [not part.strip() for part in parts]
    if len(parts) == 2 and unpaired and sum(blank) == 1:
        return (None, parts[1]) if blank[0] else (parts[0], None)
    if len(parts) not in (1, 2) or any(blank):
        forms = ("name, left_column=right_column, left_column= or =right_column" if unpaired
                 else "name or left_column=right_column")
        raise ValueError(f"column {value!r}: use {forms}")
    return parts[0] if len(parts) == 1 else (parts[0], parts[1])


def _block_spec(value: str) -> tuple[str, list, dict]:
    """(kind, columns, options). A within rule wraps the rule that follows its columns."""
    parts = value.split(":")
    try:
        kind, options = parts[0], {}
        if kind in ("ngrams", "embeddings") and len(parts) == 3:
            if not parts[2].isascii() or not parts[2].isdigit() or int(parts[2]) < 1:
                raise ValueError
            options["k"] = int(parts[2])
        elif kind == "window" and len(parts) == 3:
            found = _WINDOW.fullmatch(parts[2])
            if not found or not parts[2].isascii():
                raise ValueError
            if found["tolerance"] is not None:
                options["tolerance"] = float(found["tolerance"])
                if options["tolerance"] < 0:
                    raise ValueError
            else:
                options["between"] = (float(found["low"]), float(found["high"]))
                if options["between"][0] > options["between"][1]:
                    raise ValueError
            options["unit"] = found["unit"]
        elif kind == "within" and len(parts) >= 4:
            options["child"] = _block_spec(":".join(parts[2:]))
        elif not (kind in ("exact", "initials") and len(parts) == 2):
            raise ValueError
        columns = [_column(item) for item in parts[1].split("+")]
        if kind in ("initials", "window") and len(columns) != 1:
            raise ValueError
        return kind, columns, options
    except ValueError:
        raise ValueError(f"--block {value!r}: accepted forms are {_BLOCK_FORMS}") from None


def _number(value: str, *, low: float, high: float | None = None) -> float:
    number = float(value)
    if not math.isfinite(number) or number < low or (high is not None and number > high):
        limit = f"between {low:g} and {high:g}" if high is not None else f"at least {low:g}"
        raise argparse.ArgumentTypeError(f"expected a finite number {limit}")
    return number


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive whole number")
    return number


def _probability(value: str) -> float:
    return _number(value, low=0, high=1)


def _nonnegative(value: str) -> float:
    return _number(value, low=0)


def _margin(value: str) -> float:
    return _number(value, low=-1, high=1)


def _add_fields(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--on", action="append", required=required, metavar="COL[=COL]",
                        help="field to compare; repeat for more fields (city=town uses different names; "
                             "text= shows a field only the left records have, =place only the right)")
    parser.add_argument("--left-id", metavar="COL",
                        help="unique left record ID; default: zero-based row number")
    parser.add_argument("--right-id", metavar="COL",
                        help="unique right record ID; default: zero-based row number")


def _add_pair_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("left", metavar="LEFT", help="left .csv, .tsv, .dta or .parquet file")
    parser.add_argument("right", metavar="RIGHT", help="right .csv, .tsv, .dta or .parquet file")
    _add_fields(parser)
    parser.add_argument("--block", action="append", metavar="RULE",
                        help="ngrams finds similar text, exact requires equal fields, initials finds "
                             "abbreviations, window keeps left minus right within a tolerance (1) or a "
                             "range (0..3), in weeks, days, hours, minutes or seconds with a w/d/h/m/s "
                             "suffix and as plain numbers without; within:COLS:RULE runs RULE inside "
                             f"groups equal on COLS; repeat to combine passes. Forms: {_BLOCK_FORMS}. "
                             "Default: 10 nearest text matches across all paired --on fields")
    parser.add_argument("--date-format", metavar="FORMAT",
                        help="strptime format of text dates in window passes, such as %%m/%%d/%%Y; "
                             "LEFT=RIGHT gives each file its own and an empty side stays ISO 8601 "
                             "(default: ISO 8601, for example 2024-03-01 or 2024-03-01T14:30)")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="local SentenceTransformer for embeddings passes (optional install)")
    parser.add_argument("--embedding-revision", help="pin the embedding model to a Hub commit")
    parser.add_argument("--embedding-device", default="cpu", help="embedding device (default: cpu)")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="jlink", description="Link records across two datasets using a match rule in plain English.",
        epilog="Example: jev-link link firms.dta registry.dta --on name --entity firm -o links.csv\n"
               "On macOS /usr/bin/jlink is Java's tool. Use jev-link or python -m jlink instead.\n"
               "Tables: CSV, TSV, Stata (.dta), Parquet (requires pyarrow). Errors exit with status 2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"jlink {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, title="commands")
    link = sub.add_parser(
        "link", help="choose links between two datasets",
        description="Propose similar pairs, judge your match rule, and choose links.",
        epilog='Example: jev-link link firms.dta registry.dta --on name --on city=town '
               '--entity firm --define "The same firm, despite abbreviations." --left-id gvkey '
               '--right-id id -o links.csv',
    )
    _add_pair_inputs(link)
    link.add_argument("--entity", help="kind of record, for example firm, person or product; "
                                       "required unless --style rule")
    link.add_argument("--define", dest="definition", default="", metavar="TEXT",
                      help="your match rule in plain English (default: no additional rule)")
    link.add_argument("--style", choices=("identity", "rule"), default="identity",
                      help="identity asks whether both records are the same --entity; rule asks whether "
                           "the pair satisfies --define, which may state any relation (default: identity)")
    link.add_argument("--how", choices=_HOW, default="one-to-one",
                      help="one-to-one: unique on both sides; many-to-one: one right per left; "
                           "one-to-many: one left per right; many-to-many: all qualifying pairs "
                           "(default: one-to-one)")
    link.add_argument("--threshold", type=_probability, default=0.5,
                      help="minimum match probability, 0 to 1 (default: 0.5)")
    link.add_argument("--min-margin", type=_margin, default=None,
                      help="keep a link only if its probability leads every competing pair by this much, "
                           "-1 to 1 (default: no such requirement)")
    link.add_argument("--budget", type=_nonnegative, default=5.0,
                      help="stop new requests at this observed USD cost; in-flight calls may overshoot "
                           "(default: 5; 0: cached/exact only)")
    link.add_argument("-o", "--output", metavar="FILE", help="links table; default: CSV on standard output")
    link.add_argument("--scores", metavar="FILE", help="save all candidate scores for a later audit")
    link.add_argument("--report", metavar="FILE", help="save the linkage report as Markdown")
    link.add_argument("--api", choices=("typesafe", "openrouter"),
                      help="API provider (default: configured provider)")
    link.add_argument("--model", metavar="ID", help="model identifier (default: provider's Jev model)")
    link.add_argument("--no-cache", action="store_true", help="do not reuse or save cached judgments")
    link.add_argument("--exact-shortcut", action="store_true",
                      help="accept equal nonempty normalized fields without judging; opt in only "
                           "when those fields establish identity (default: judge equal text too)")
    link.add_argument("-j", type=_positive, default=32, dest="concurrency", metavar="N",
                      help="maximum simultaneous API requests (default: 32)")
    link.set_defaults(run=_link)
    estimate = sub.add_parser(
        "estimate", help="count candidates and estimate cost without API calls",
        description="Run blocking only; estimate judging cost and time without making API calls.",
        epilog="Example: jev-link estimate firms.dta registry.dta --on name --block ngrams:name:10",
    )
    _add_pair_inputs(estimate)
    estimate.set_defaults(run=_estimate)
    audit = sub.add_parser(
        "audit", help="draw a sample of scored pairs for hand checking",
        description="Sample judged pairs across probability groups. Fill is_match with 1/0 or yes/no.",
        epilog="Example: jev-link audit scores.csv -n 200 --left firms.dta --right registry.dta "
               "--on name --left-id gvkey --right-id id -o audit.csv",
    )
    audit.add_argument("scores", metavar="SCORES", help="saved candidate scores table")
    audit.add_argument("--links", metavar="LINKS", help="actual final links table; adds selected membership")
    audit.add_argument("-n", type=_positive, default=200, help="number of pairs to sample (default: 200)")
    audit.add_argument("--left", metavar="LEFT", help="original left data, for side-by-side fields")
    audit.add_argument("--right", metavar="RIGHT", help="original right data, for side-by-side fields")
    _add_fields(audit, required=False)
    audit.add_argument("-o", "--output", required=True, metavar="FILE", help="table to label by hand")
    audit.set_defaults(run=_audit)
    evaluate = sub.add_parser(
        "evaluate", help="report accuracy from the hand-labeled audit",
        description="Evaluate thresholded pair scores or saved final-link membership from a labeled audit. "
                    "Recall covers judged candidates only; Brier and calibration always assess pair scores.",
        epilog="Example: jev-link evaluate labeled_firms.dta --threshold 0.8 --markdown",
    )
    evaluate.add_argument("labeled", metavar="LABELED", help="audit table with completed is_match labels")
    evaluate.add_argument("--mode", choices=("threshold", "selected"), default="threshold",
                          help="threshold assesses pair scores (default); "
                               "selected assesses saved final links")
    evaluate.add_argument("--threshold", type=_probability, default=0.5,
                          help="pair probability cutoff, used only in threshold mode (default: 0.5)")
    evaluate.add_argument("--markdown", action="store_true",
                          help="print a Markdown table for a data appendix")
    evaluate.set_defaults(run=_evaluate)
    from .review_cli import add_parser as add_review_parser

    add_review_parser(sub)
    return parser


def _fields(args: argparse.Namespace, left: pd.DataFrame, right: pd.DataFrame) -> list:
    on = [_column(item, unpaired=True) for item in args.on]
    for _, a, b in parse_on(on, unpaired=True):
        check_columns(left, [a] if a is not None else [], args.left)
        check_columns(right, [b] if b is not None else [], args.right)
    ids(left, args.left_id, args.left)
    ids(right, args.right_id, args.right)
    return on


def _inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list, list | None]:
    # Parse rules before importing the independently developed blocking module.
    specs = [_block_spec(rule) for rule in args.block or []]
    left, right = read_table(args.left), read_table(args.right)
    on = _fields(args, left, right)
    blockers = None
    if specs:
        from . import block

        def build(spec):
            kind, columns, options = spec
            for _, a, b in parse_on(columns):
                check_columns(left, [a], args.left)
                check_columns(right, [b], args.right)
            kwargs = dict(options)
            if kind == "within":
                return block.within(build(kwargs.pop("child")), *columns)
            if kind == "embeddings":
                kwargs.update(model=args.embedding_model, revision=args.embedding_revision,
                              device=args.embedding_device)
            if kind == "window" and kwargs["unit"] is not None:
                kwargs["date_format"] = _date_formats(args.date_format)
            return getattr(block, kind)(*columns, **kwargs)

        blockers = [build(spec) for spec in specs]
    return left, right, on, blockers


def _date_formats(value: str | None) -> tuple[str | None, str | None] | None:
    """FORMAT for both files, or LEFT=RIGHT, where an empty side keeps ISO 8601."""
    if value is None:
        return None
    parts = value.split("=")
    if len(parts) > 2 or not value.strip("= "):
        raise ValueError(f"--date-format {value!r}: use FORMAT, or LEFT_FORMAT=RIGHT_FORMAT with an "
                         "empty side for ISO 8601")
    left, right = (parts[0], parts[-1])
    return (left.strip() or None, right.strip() or None)


def _outputs(inputs: list[str], tables: list[str | None], report: str | None = None) -> None:
    seen = {Path(path).expanduser().resolve() for path in inputs}
    for value in [*tables, report]:
        if value is None:
            continue
        path = Path(value).expanduser()
        if value in tables and path.suffix.lower() not in FORMATS:
            raise ValueError(f"{value!r}: expected a .csv, .tsv, .dta or .parquet output file")
        if path.resolve() in seen:
            raise ValueError(f"{value!r}: choose a separate output file; it repeats an input or output path")
        if not path.parent.is_dir() or path.is_dir():
            raise ValueError(f"{value!r}: output needs an existing folder and a file name")
        if path.suffix.lower() == ".parquet" and value in tables:
            from .io import _require_arrow

            _require_arrow(path)
        seen.add(path.resolve())


def _link(args: argparse.Namespace) -> None:
    _outputs([args.left, args.right], [args.output, args.scores], args.report)
    if args.style == "rule" and not args.definition.strip():
        raise ValueError("--style rule asks whether a pair satisfies --define, so --define cannot be empty")
    if args.style == "identity" and not (args.entity or "").strip():
        raise ValueError("--entity must name a kind of record, such as firm (only --style rule can omit it)")
    left, right, on, blockers = _inputs(args)
    from .linker import Linker

    linker = Linker(entity=args.entity, definition=args.definition, on=on, blockers=blockers,
                    style=args.style, api=args.api, model=args.model, cache=not args.no_cache,
                    concurrency=args.concurrency, exact_shortcut=args.exact_shortcut)
    result = linker.link(left, right, left_id=args.left_id, right_id=args.right_id, how=args.how,
                         threshold=args.threshold, min_margin=args.min_margin, budget=args.budget,
                         progress=sys.stderr.isatty())  # no progress bar in Stata logs, R output or pipes
    if args.output:
        write_table(result.links, args.output)
    else:
        result.links.to_csv(sys.stdout, index=False)
    if args.scores:
        write_table(result.scores, args.scores)
    if args.report:
        Path(args.report).expanduser().write_text(result.report(), encoding="utf-8")
    print(f"jlink: {len(left):,} left records, {len(right):,} right records; "
          f"{len(result.candidates):,} candidate pairs; {len(result.links):,} links; "
          f"{result.meter.calls:,} calls; ${result.meter.cost:.4f}", file=sys.stderr)


def _estimate(args: argparse.Namespace) -> None:
    left, right, on, blockers = _inputs(args)
    from .block import candidates

    pairs = candidates(left, right, on=on, blockers=blockers, left_id=args.left_id, right_id=args.right_id)
    n = len(pairs)
    for item in pairs.attrs.get("blocking", {}).get("passes", []):
        lost = item.get("dropped_values")
        counts = lost and {side: lost[side]["missing"] + lost[side]["unparseable"]
                           for side in ("left", "right")}
        if counts and any(counts.values()):
            print(f"{item['name']}: {counts['left']:,} left and {counts['right']:,} right records have a "
                  f"missing or unreadable value and cannot pair in this pass")
    print(f"{len(left):,} left records; {len(right):,} right records; {n:,} candidate pairs\n"
          f"Estimated judging cost: ${n * 330 * 0.042 / 1_000_000:.6f}\n"
          f"Estimated judging time: {n / 200:.1f} seconds\n"
          "Assumes 330 input tokens per pair at $0.042 per million and 200 pairs/second. "
          "Actual cost and time depend on caching, exact matches, record length and the provider. "
          "No API calls.")


def _numeric(frame: pd.DataFrame, columns: list[str], path: str, *, every_row: str = "") -> None:
    """Parse numeric columns. `every_row` says how to recover a column that may have no empty cells."""
    check_columns(frame, columns, path)
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError):
            allowed = f"a number in every row; {every_row}" if every_row else "numbers or empty cells"
            raise ValueError(f"{path!r}: column {column!r} must contain {allowed}") from None


def _align_ids(scores: pd.DataFrame, frame: pd.DataFrame, column: str | None, side: str) -> None:
    values = ids(frame, column, side)
    # CSV scores contain text even when the original Stata IDs were numeric.
    # Translate through the source IDs, without guessing numeric types for text IDs.
    mapping = {str(value): value for value in values}
    if len(mapping) != len(values):
        raise ValueError(f"{side!r}: IDs must also be unique when written as text")
    key = f"{side}_id"
    converted = scores[key].map(lambda value: mapping.get(str(value), value))
    if not converted.isin(values).all():
        raise ValueError(f"scores column {key!r} contains an ID absent from the original {side} data")
    scores[key] = converted


def _audit(args: argparse.Namespace) -> None:
    enrich = any((args.left, args.right, args.on, args.left_id, args.right_id))
    if enrich and not all((args.left, args.right, args.on)):
        raise ValueError("audit needs --left, --right and --on together to show record fields")
    _outputs([args.scores] + ([args.links] if args.links else [])
             + ([args.left, args.right] if enrich else []), [args.output])
    scores = read_table(args.scores)
    check_columns(scores, ["left_id", "right_id"], args.scores)
    _numeric(scores, ["p"], args.scores)
    links = read_table(args.links) if args.links else None
    if links is not None:
        check_columns(links, ["left_id", "right_id"], args.links)
    kwargs = {}
    if enrich:
        left, right = read_table(args.left), read_table(args.right)
        on = _fields(args, left, right)
        _align_ids(scores, left, args.left_id, "left")
        _align_ids(scores, right, args.right_id, "right")
        if links is not None:
            _align_ids(links, left, args.left_id, "left")
            _align_ids(links, right, args.right_id, "right")
        kwargs = dict(left=left, right=right, on=on, left_id=args.left_id, right_id=args.right_id)
    if links is not None:
        kwargs["links"] = links
    from .audit import audit_sample

    write_table(audit_sample(scores, n=args.n, **kwargs), args.output)


def _evaluate(args: argparse.Namespace) -> None:
    labeled = read_table(args.labeled)
    check_columns(labeled, ["is_match", "bin", "p"], args.labeled)
    # Stata stores the bins as coded value labels. Naming them is a one-to-one relabeling: the
    # strata, their order and every number stay as they are, and the report reads "(0.2, 0.5]", not "2".
    names = stata_value_labels(args.labeled, "bin")
    if len(set(names.values())) == len(names) and labeled["bin"].isin(names).all():
        labeled["bin"] = labeled["bin"].map(names)
    from .audit import _blank, evaluate

    # As in the Python API, a row without a label need not have a usable probability.
    labeled["p"] = labeled["p"].mask(_blank(labeled["is_match"]))
    _numeric(labeled, ["p"], args.labeled)
    _numeric(labeled, ["weight"], args.labeled,
             every_row="rows with a blank label need theirs too, so restore it from the table `audit` wrote")
    result = evaluate(labeled, threshold=args.threshold, mode=args.mode)
    print(result.to_markdown() if args.markdown else result.summary())


def cli(argv: list[str] | None = None) -> None:
    """Run a command, keeping expected failures short and suitable for shell scripts."""
    try:
        args = _parser().parse_args(argv)
        with warnings.catch_warnings():
            # Keep name-repair and resolver warnings off CSV standard output.
            warnings.showwarning = lambda message, *a, **kw: print(
                "jlink: warning: " + " ".join(str(message).split()), file=sys.stderr
            )
            args.run(args)
    except (Exception, KeyboardInterrupt) as exc:
        message = "interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc)
        print("jlink: " + (" ".join(message.split()) or type(exc).__name__), file=sys.stderr)
        raise SystemExit(2) from None

"""Commands for linking records and reporting how well the links work."""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import pandas as pd

from . import __version__
from .fields import check_columns, ids, parse_on
from .io import FORMATS, read_table, write_table

_BLOCK_FORMS = "ngrams:name:10, ngrams:name+city:20, exact:state, exact:state=st, initials:name"
_HOW = ("one-to-one", "many-to-one", "one-to-many", "many-to-many")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _column(value: str) -> str | tuple[str, str]:
    parts = value.split("=")
    if len(parts) not in (1, 2) or any(not part.strip() for part in parts):
        raise ValueError(f"column {value!r}: use name or left_column=right_column")
    return parts[0] if len(parts) == 1 else (parts[0], parts[1])


def _block_spec(value: str) -> tuple[str, list, int | None]:
    parts = value.split(":")
    try:
        kind = parts[0]
        if kind == "ngrams" and len(parts) == 3:
            if not parts[2].isascii() or not parts[2].isdigit() or int(parts[2]) < 1:
                raise ValueError
            k = int(parts[2])
        elif kind in ("exact", "initials") and len(parts) == 2:
            k = None
        else:
            raise ValueError
        columns = [_column(item) for item in parts[1].split("+")]
        if kind == "initials" and len(columns) != 1:
            raise ValueError
        return kind, columns, k
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
                        help="field to compare; repeat for more fields (city=town uses different names)")
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
                             f"abbreviations; repeat to combine passes. Forms: {_BLOCK_FORMS}. "
                             "Default: 10 nearest text matches across all --on fields")


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
    link.add_argument("--entity", required=True, help="kind of record, for example firm, person or product")
    link.add_argument("--define", dest="definition", default="", metavar="TEXT",
                      help="your match rule in plain English (default: no additional rule)")
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
                      help="maximum API spending in US dollars (default: 5)")
    link.add_argument("-o", "--output", metavar="FILE", help="links table; default: CSV on standard output")
    link.add_argument("--scores", metavar="FILE", help="save all candidate scores for a later audit")
    link.add_argument("--report", metavar="FILE", help="save the linkage report as Markdown")
    link.add_argument("--api", choices=("typesafe", "openrouter"),
                      help="API provider (default: configured provider)")
    link.add_argument("--model", metavar="ID", help="model identifier (default: provider's Jev model)")
    link.add_argument("--no-cache", action="store_true", help="do not reuse or save cached judgments")
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
    return parser


def _fields(args: argparse.Namespace, left: pd.DataFrame, right: pd.DataFrame) -> list:
    on = [_column(item) for item in args.on]
    for _, a, b in parse_on(on):
        check_columns(left, [a], args.left)
        check_columns(right, [b], args.right)
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

        blockers = []
        for kind, columns, k in specs:
            for _, a, b in parse_on(columns):
                check_columns(left, [a], args.left)
                check_columns(right, [b], args.right)
            factory = getattr(block, kind)
            blockers.append(factory(*columns, **({"k": k} if kind == "ngrams" else {})))
    return left, right, on, blockers


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
    if not args.entity.strip():
        raise ValueError("--entity must name a kind of record, such as firm")
    left, right, on, blockers = _inputs(args)
    from .linker import Linker

    linker = Linker(entity=args.entity, definition=args.definition, on=on, blockers=blockers,
                    api=args.api, model=args.model, cache=not args.no_cache, concurrency=args.concurrency)
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
    print(f"{len(left):,} left records; {len(right):,} right records; {n:,} candidate pairs\n"
          f"Estimated judging cost: ${n * 330 * 0.042 / 1_000_000:.6f}\n"
          f"Estimated judging time: {n / 200:.1f} seconds\n"
          "Assumes 330 input tokens per pair at $0.042 per million and 200 pairs/second. "
          "Actual cost and time depend on caching, exact matches, record length and the provider. "
          "No API calls.")


def _numeric(frame: pd.DataFrame, columns: list[str], path: str) -> None:
    check_columns(frame, columns, path)
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError):
            raise ValueError(f"{path!r}: column {column!r} must contain numbers or empty cells") from None


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
    check_columns(labeled, ["is_match", "bin"], args.labeled)
    _numeric(labeled, ["p", "weight"], args.labeled)
    from .audit import evaluate

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

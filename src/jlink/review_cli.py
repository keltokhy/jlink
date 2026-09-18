"""Small CLI adapter for the independent local review workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .review import create_review, read_review


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "review", help="create a local review page or apply human decisions offline",
        description="Review original records and candidate links locally, with durable JSON history.")
    commands = parser.add_subparsers(dest="review_command", required=True)
    create = commands.add_parser("create", help="make a page from a saved Result and original tables")
    create.add_argument("run_directory", metavar="RUN", help="directory written by Result.save()")
    create.add_argument("--left", required=True, help="original left .csv, .tsv, .dta or .parquet table")
    create.add_argument("--right", required=True, help="original right table")
    create.add_argument("-o", "--output", required=True, help="standalone .html page")
    create.add_argument("--artifact", help="also save the initial .json review artifact")
    create.add_argument("--close-margin", type=float, default=0.1,
                        help="queue gap for close competitors (default: 0.1)")
    create.set_defaults(run_review=_create, run=_dispatch)
    page = commands.add_parser("page", help="reopen an exported review artifact as a local page")
    page.add_argument("artifact", help="review JSON exported by the page or Python")
    page.add_argument("-o", "--output", required=True, help="standalone .html page")
    page.set_defaults(run_review=_page, run=_dispatch)
    apply = commands.add_parser("apply", help="apply constraints and recompute links without API calls")
    apply.add_argument("artifact", help="review JSON exported by the page or Python")
    apply.add_argument("-o", "--output", required=True,
                       help="authoritative .json output, including review history")
    apply.add_argument("--csv", help="optional spreadsheet-safe .csv display copy (JSON preserves exact IDs)")
    apply.add_argument("--how", choices=("one-to-one", "many-to-one", "one-to-many", "many-to-many"))
    apply.add_argument("--threshold", type=float)
    margins = apply.add_mutually_exclusive_group()
    margins.add_argument("--min-margin", type=float, default=argparse.SUPPRESS)
    margins.add_argument("--no-margin", action="store_true",
                         help="explicitly remove the original margin requirement")
    apply.set_defaults(run_review=_apply, run=_dispatch)


def _dispatch(args) -> None:
    args.run_review(args)


def _outputs(inputs: list, outputs: list[tuple[str | None, str]]) -> None:
    seen = {Path(p).expanduser().resolve() for p in inputs}
    for name, suffix in outputs:
        if not name:
            continue
        path = Path(name).expanduser()
        if path.suffix.lower() != suffix:
            raise ValueError(f"review output {name!r} must end in {suffix}")
        if path.resolve() in seen:
            raise ValueError("review output repeats an input or output path; choose a separate file")
        if not path.parent.is_dir() or path.is_dir():
            raise ValueError("review output needs an existing folder and a file name")
        seen.add(path.resolve())


def _create(args) -> None:
    from .fields import ids
    from .io import read_table
    from .linker import Result
    from .cli import _align_ids, _numeric

    directory = Path(args.run_directory).expanduser()
    run_files = [directory / name for name in ("settings.json", "scores.csv", "links.csv")]
    _outputs([*run_files, args.left, args.right], [(args.output, ".html"), (args.artifact, ".json")])
    settings = json.loads(run_files[0].read_text(encoding="utf-8"))
    left, right = read_table(args.left), read_table(args.right)
    scores, links = read_table(run_files[1]), read_table(run_files[2])
    # Read CSV with the existing lossless table reader: NA/NULL and leading zeros are IDs.
    # Saved integer IDs are restored before aligning both tables to the original records.
    for side, frame in (("left", left), ("right", right)):
        column = settings.get(f"{side}_id")
        kind = settings.get("id_kinds", {}).get(f"{side}_id")
        if kind == "int" and column is not None:
            values = ids(frame, column, side)
            try:
                converted = [int(v) for v in values]
                if any(str(v) != str(i) for v, i in zip(values, converted)):
                    raise ValueError
                frame[column] = converted
            except (ValueError, TypeError, OverflowError):
                raise ValueError(
                    f"original {side} ID column {column!r} disagrees with saved integer IDs") from None
        for table in (scores, links):
            _align_ids(table, frame, column, side)
    _numeric(scores, ["p", "sim"], str(run_files[1]))
    _numeric(links, ["p", "sim", "margin"], str(run_files[2]))
    settings.pop("id_kinds", None)
    review = create_review(Result(links, scores, settings), left=left, right=right,
                           close_margin=args.close_margin)
    review.write_html(Path(args.output).expanduser())
    if args.artifact:
        review.save(Path(args.artifact).expanduser())
    print(f"jlink: wrote local review page {args.output}; source {review.run_id}; no API calls",
          file=sys.stderr)


def _page(args) -> None:
    _outputs([args.artifact], [(args.output, ".html")])
    read_review(Path(args.artifact).expanduser()).write_html(Path(args.output).expanduser())
    print(f"jlink: wrote local review page {args.output}; no API calls", file=sys.stderr)


def _apply(args) -> None:
    _outputs([args.artifact], [(args.output, ".json"), (args.csv, ".csv")])
    kwargs = dict(how=args.how, threshold=args.threshold)
    if args.no_margin:
        kwargs["min_margin"] = None
    elif hasattr(args, "min_margin"):
        kwargs["min_margin"] = args.min_margin
    applied = read_review(Path(args.artifact).expanduser()).apply(**kwargs)
    applied.save(Path(args.output).expanduser())
    if args.csv:
        applied.write_csv(Path(args.csv).expanduser())
    manual = int(applied.links.selection_basis.eq("manual_accept").sum())
    print(f"jlink: {len(applied.links)} reviewed links ({manual} manual accepts); "
          f"{applied.settings['how']}; no API calls; authoritative output {args.output}", file=sys.stderr)

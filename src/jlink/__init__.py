"""jlink: record linkage with match rules in plain English, judged by TypeSafe's Jev model.

    import jlink
    result = jlink.link(left, right, entity="firm", on=["name", "state"],
                        definition="A parent company and its subsidiary are different firms.")
    result.links            # the chosen pairs, with probabilities
    print(result.methods()) # a paragraph for the data appendix
"""

__version__ = "0.2.0"

from . import block  # noqa: E402
from .audit import Evaluation, audit_sample, evaluate, score_against_truth  # noqa: E402
from .cluster import cluster  # noqa: E402
from .judge import judge  # noqa: E402
from .linker import DedupeResult, Linker, Result, dedupe, link, load  # noqa: E402
from .resolve import resolve  # noqa: E402
from .review import Review, ReviewedLinks, create_review, read_review  # noqa: E402

__all__ = ["Linker", "Result", "link", "load", "block", "judge", "resolve", "audit_sample", "evaluate",
           "score_against_truth", "Evaluation", "__version__", "Review", "ReviewedLinks", "create_review",
           "read_review", "dedupe", "DedupeResult", "cluster"]

"""Create an entirely offline review example: python examples/review_demo.py /tmp/jlink-review."""
from pathlib import Path
import argparse

import pandas as pd

import jlink


def demo_result() -> jlink.Result:
    left = pd.DataFrame({
        "id": ["001", "002", "003", "NA", "005", "006"],
        "name": ["Acme Laboratories", "Acme Research", "Northstar Ltd", "River Trading",
                 "No Candidate Inc", "Juniper Works"],
        "city": ["Boston", "Boston", "Seattle", "New York", "Chicago", "Portland"],
    })
    right = pd.DataFrame({
        "id": ["A", "B", "C", "D", "E", "NULL"],
        "name": ["Acme Research Laboratories", "Acme Labs", "North Star", "River Traders",
                 "Juniper Works LLC", "Orphan record"],
        "city": ["Boston", "Cambridge", "Seattle", "New York", "Portland", "Denver"],
    })
    scores = pd.DataFrame({
        "left_id": ["001", "001", "002", "002", "003", "NA", "006"],
        "right_id": ["A", "B", "A", "B", "C", "D", "E"],
        "p": [.82, .79, .81, .2, float("nan"), .15, .96],
        "sim": [.9, .85, .87, .4, .8, .65, .98],
        "source": ["jev", "jev", "jev", "jev", "unjudged", "jev", "jev"],
        "block": ["demo:ngrams"] * 7, "error": [None] * 7,
    })
    settings = dict(how="one-to-one", threshold=.5, min_margin=None, left_id="id", right_id="id",
                    on=[["name", "name"], ["city", "city"]], entity="firm", n_left=6, n_right=6,
                    definition="The same firm, despite abbreviations. Subsidiaries are different firms.",
                    question="Do these records identify the same firm? Subsidiaries are different firms.",
                    model="offline-demo", calls=0, dollars=0)
    return jlink.Result(jlink.resolve(scores), scores, settings, _left=left, _right=right)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    destination = parser.parse_args().directory
    destination.mkdir(parents=True, exist_ok=True)
    result = demo_result()
    result.save(destination / "run")
    result._left.to_csv(destination / "left.csv", index=False)
    result._right.to_csv(destination / "right.csv", index=False)
    review = jlink.create_review(result)
    review.write_html(destination / "review.html")
    review.save(destination / "review.json")
    print(destination / "review.html")

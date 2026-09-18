import pandas as pd
import pytest

from jlink.fields import ids, normalize, parse_on, record_text


def test_normalize():
    assert normalize("  Société Générale & Co., S.A. ") == "societe generale and co s a"
    assert normalize(None) == "" and normalize(float("nan")) == "" and normalize(pd.NA) == ""


def test_parse_on():
    assert parse_on(["name", ("city", "town")]) == [("name", "name", "name"), ("city", "city", "town")]
    assert parse_on("name") == [("name", "name", "name")]
    with pytest.raises(ValueError):
        parse_on([])


def test_record_text_and_ids():
    df = pd.DataFrame({"id": [1, 2], "name": ["A & B", None], "city": ["New York", "Cairo"]})
    assert list(record_text(df, ["name", "city"])) == ["a and b new york", "cairo"]
    assert list(ids(df, "id", "left")) == [1, 2]
    with pytest.raises(ValueError, match="duplicate"):
        ids(pd.DataFrame({"id": [1, 1]}), "id", "left")

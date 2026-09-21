"""The window blocker against a brute-force oracle, plus its limits, diagnostics and CLI grammar."""

import importlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import jlink
from fakes import FakeJev
from jlink import block

cli = importlib.import_module("jlink.cli").cli


def oracle(left_values, right_values, low, high, same_group=None):
    """Every (i, j) with low <= left - right <= high, by comparing all pairs."""
    pairs = []
    for i, a in enumerate(left_values):
        for j, b in enumerate(right_values):
            if pd.isna(a) or pd.isna(b) or (same_group is not None and not same_group(i, j)):
                continue
            if low <= a - b <= high:
                pairs.append([i, j])
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


@pytest.mark.parametrize("kwargs,low,high", [
    ({"tolerance": 3}, -3, 3), ({"tolerance": 0}, 0, 0), ({"between": (0, 3)}, 0, 3),
    ({"between": (-5, -2)}, -5, -2), ({"between": (2, 2)}, 2, 2), ({"between": (-0.5, 1.25)}, -0.5, 1.25)])
@pytest.mark.parametrize("seed", [0, 1])
def test_numbers_match_the_oracle(kwargs, low, high, seed):
    rng = np.random.default_rng(seed)
    # Quarter steps are exact in binary, so the oracle's subtraction has no rounding to disagree about.
    a = rng.integers(-40, 40, 70) / 4
    b = rng.integers(-40, 40, 90) / 4
    a[rng.integers(0, 70, 6)] = np.nan
    b[rng.integers(0, 90, 6)] = np.nan
    left, right = pd.DataFrame({"x": a}), pd.DataFrame({"y": b})
    got = block.window(("x", "y"), **kwargs).pairs(left, right)
    assert_array_equal(got, oracle(a, b, low, high))  # left order, then right position
    assert got.dtype == np.int64


@pytest.mark.parametrize("kwargs,low,high", [
    ({"tolerance": 2, "unit": "days"}, -48, 48),
    ({"between": (0, 3), "unit": "days"}, 0, 72),          # published 0 to 3 days after, never before
    ({"between": (-36, 12), "unit": "hours"}, -36, 12),
    ({"between": (-90, 45.5), "unit": "minutes"}, -1.5, 45.5 / 60),
    ({"between": (1, 2), "unit": "weeks"}, 7 * 24, 14 * 24)])
@pytest.mark.parametrize("form", ["text", "datetime64", "objects"])
def test_dates_match_the_oracle_in_every_representation(kwargs, low, high, form):
    rng = np.random.default_rng(7)
    start = pd.Timestamp("2024-02-20")
    a = pd.Series(start + pd.to_timedelta(rng.integers(0, 24 * 30, 60), unit="h"))
    b = pd.Series(start + pd.to_timedelta(rng.integers(0, 30, 80), unit="D"))
    a[[3, 17]], b[[0, 5, 9]] = pd.NaT, pd.NaT
    # The oracle works in whole nanoseconds, as Python integers; low and high are in hours.
    stamps = [[None if pd.isna(v) else int(v.value) for v in values] for values in (a, b)]
    expected = oracle(*stamps, round(low * 3_600 * 10**9), round(high * 3_600 * 10**9))
    if form == "text":
        a, b = a.dt.strftime("%Y-%m-%dT%H:%M:%S"), b.dt.strftime("%Y-%m-%d")
    elif form == "objects":
        a = a.astype(object).where(a.notna(), None)
        b = pd.Series([None if pd.isna(v) else v.date() for v in b], dtype=object)
    got = block.window(("published", "occurred"), **kwargs).pairs(
        pd.DataFrame({"published": a}), pd.DataFrame({"occurred": b}))
    assert len(expected)
    assert_array_equal(got, expected)


def test_asymmetric_window_never_pairs_an_article_with_a_later_incident():
    articles = pd.DataFrame({"published": ["2024-03-04", "2024-03-01"]})
    incidents = pd.DataFrame({"occurred": ["2024-03-01", "2024-03-02", "2024-03-04", "2024-03-05"]})
    after = block.window(("published", "occurred"), between=(0, 3), unit="days")
    assert_array_equal(after.pairs(articles, incidents), [[0, 0], [0, 1], [0, 2], [1, 0]])
    assert after.name == "window:published=occurred[0..3d]"
    # A date-only value is midnight: three days and nine hours is outside a three-day window.
    stamped = pd.DataFrame({"published": ["2024-03-05T09:00", "2024-03-05T00:00"]})
    assert_array_equal(after.pairs(stamped, incidents.iloc[[1, 2]]), [[0, 1], [1, 0], [1, 1]])
    assert block.window("year", 1).name == "window:year[-1..1]"


def test_missing_and_unreadable_values_are_dropped_and_counted_never_guessed():
    left = pd.DataFrame({"when": ["2024-03-02", None, "  ", "03/02/2024", "2024-02-30", "not a date"]})
    right = pd.DataFrame({"when": ["2024-03-02", "", "2024-13-01", pd.NA]})
    passed = block.window("when", 0, unit="days")
    assert_array_equal(passed.pairs(left, right), [[0, 0]])
    assert passed.dropped(left, right) == {
        "kind": "dates",
        "left": {"column": "when", "missing": 2, "unparseable": 3, "time_zone": "none"},
        "right": {"column": "when", "missing": 2, "unparseable": 1, "time_zone": "none"}}
    with pytest.warns(UserWarning, match=r"could not read 3 left values in 'when' and 1 right values in "
                                         r"'when' as dates.*not guessed.*ISO 8601"):
        table = block.candidates(left, right, on="when", blockers=[passed])
    info = table.attrs["blocking"]["passes"][0]
    assert info["dropped_values"]["left"]["unparseable"] == 3 and info["proposed_pairs"] == 1
    assert json.loads(json.dumps(table.attrs["blocking"])) == table.attrs["blocking"]
    numbers = block.window("n", 1)
    frame = pd.DataFrame({"n": ["1", "1,000", None, "inf", "2.5", "abc"]})
    assert_array_equal(numbers.pairs(frame, frame), [[0, 0], [4, 4]])
    assert numbers.dropped(frame, frame)["left"] == {
        "column": "n", "missing": 1, "unparseable": 3, "time_zone": None}
    # Missing values alone are ordinary and do not warn.
    quiet = pd.DataFrame({"n": [1.0, np.nan]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(block.candidates(quiet, quiet, on="n", blockers=[numbers])) == 1


def test_values_are_compared_as_numbers_whatever_their_storage():
    # A numeric column with one missing value is float64; its neighbor without one is int64.
    # Compared as text, 1985 and 1985.0 would never meet.
    whole = pd.DataFrame({"year": pd.Series([1985, 1990, 2001], dtype="int64")})
    gappy = pd.DataFrame({"year": pd.Series([1985.0, np.nan, 2001.0, 1991.0], dtype="float64")})
    assert_array_equal(block.window("year", 0).pairs(whole, gappy), [[0, 0], [2, 2]])
    assert_array_equal(block.window("year", 1).pairs(whole, gappy), [[0, 0], [1, 3], [2, 2]])
    assert block.window("year", 0).dropped(whole, gappy)["right"] == {
        "column": "year", "missing": 1, "unparseable": 0, "time_zone": None}
    for stored, expected in [
            (pd.Series(["1985", "1985.0", " 2001 ", None]), [[0, 0], [0, 1], [2, 2]]),          # text
            (pd.Series([1985, None, 2001.0, "x"], dtype=object), [[0, 0], [2, 2]]),             # mixed objects
            (pd.Series([1985, pd.NA, 2001, 7], dtype="Int64"), [[0, 0], [2, 2]])]:              # nullable integers
        assert_array_equal(block.window("year", 0).pairs(whole, pd.DataFrame({"year": stored})), expected)


def test_one_column_with_mixed_date_formats_keeps_only_what_the_stated_format_reads():
    mixed = pd.DataFrame({"when": ["2024-03-01", "03/01/2024", "1 March 2024", "2024-03-01T08:00", None]})
    iso = pd.DataFrame({"when": ["2024-03-01"]})
    default = block.window("when", 1, unit="days")
    assert_array_equal(default.pairs(mixed, iso), [[0, 0], [3, 0]])
    assert default.dropped(mixed, iso)["left"] == {
        "column": "when", "missing": 1, "unparseable": 2, "time_zone": "none"}
    american = block.window("when", 1, unit="days", date_format=("%m/%d/%Y", None))
    assert_array_equal(american.pairs(mixed, iso), [[1, 0]])  # 03/01 is March 1st because the format says so
    assert american.dropped(mixed, iso)["left"]["unparseable"] == 3
    with pytest.warns(UserWarning, match="could not read 3 left values in 'when' as dates"):
        block.candidates(mixed, iso, on="when", blockers=[american])


def numeric_groups(right_dtype):
    left = pd.DataFrame({"precinct": pd.Series([75, 75, 40, 40], dtype="int64"), "day": [1, 5, 1, 9]})
    right = pd.DataFrame({"precinct": pd.Series([75, 40, 40, None] if right_dtype == "float64" else [75, 40, 40, 1],
                                                dtype=right_dtype), "day": [2.0, 1.0, np.nan, 1.0]})
    return left, right


def test_window_inside_within_on_a_numeric_group_key():
    left, right = numeric_groups("int64")
    grouped = block.within(block.window("day", 1), "precinct")
    assert_array_equal(grouped.pairs(left, right), [[0, 0], [2, 1]])
    assert grouped.dropped(left, right)["right"]["missing"] == 1


def test_window_inside_within_when_one_numeric_group_key_is_float_because_of_a_missing_value():
    # One missing precinct makes the whole column float64. Group keys read 75 and 75.0 as one key.
    left, right = numeric_groups("float64")
    assert right.precinct.dtype == "float64" and left.precinct.dtype == "int64"
    grouped = block.within(block.window("day", 1), "precinct")
    assert_array_equal(grouped.pairs(left, right), [[0, 0], [2, 1]])
    text = right.assign(precinct=["75.0", "40", "40.0", None])  # the same column after a trip through CSV
    assert_array_equal(grouped.pairs(left, text), [[0, 0], [2, 1]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = block.candidates(left, right, on="day", blockers=[grouped])
    assert len(table) == 2 and table.attrs["blocking"]["passes"][0]["dropped_values"]["right"]["missing"] == 1


def test_a_grouped_window_warns_when_group_keys_disagree_and_only_then():
    articles = pd.DataFrame({"boro": ["BROOKLYN", "BRONX"], "day": [1, 1]})
    register = pd.DataFrame({"boro": ["K", "X"], "day": [1, 2]})            # coded differently
    grouped = block.within(block.window("day", 1), "boro")
    with pytest.warns(UserWarning, match=r"within:boro\[drop\]\(window:day\[-1\.\.1\]\)' proposed no pairs: its key "
                                         r"columns have no value in common.*'brooklyn'.*'k'") as caught:
        assert block.candidates(articles, register, on="day", blockers=[grouped]).empty
    assert caught[0].filename == __file__  # reported at the caller, not inside jlink
    # Keys agree and the window is simply empty, or the pass has no keys at all: an answer, not a fault.
    far = register.assign(boro=["BROOKLYN", "BRONX"], day=[50, 60])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert block.candidates(articles, far, on="day", blockers=[grouped]).empty
        assert block.candidates(articles, far, on="day", blockers=[block.window("day", 1)]).empty


def test_a_wrong_date_format_is_reported_and_an_explicit_one_is_honored():
    register = pd.DataFrame({"OCCUR_DATE": ["08/27/2006", "03/01/2024", "13/45/2024"]})
    articles = pd.DataFrame({"published": ["2024-03-02", "2006-08-27"]})
    unread = block.window(("published", "OCCUR_DATE"), between=(0, 3), unit="days")
    assert unread.pairs(articles, register).shape == (0, 2)
    assert unread.dropped(articles, register)["right"]["unparseable"] == 3
    read = block.window(("published", "OCCUR_DATE"), between=(0, 3), unit="days",
                        date_format=(None, "%m/%d/%Y"))
    assert_array_equal(read.pairs(articles, register), [[0, 1], [1, 0]])
    assert read.dropped(articles, register)["right"]["unparseable"] == 1
    assert read.to_config()["date_format"] == [None, "%m/%d/%Y"]
    both = block.window("d", 0, unit="days", date_format="%d.%m.%Y")
    frame = pd.DataFrame({"d": ["01.03.2024", "2024-03-01"]})
    assert_array_equal(both.pairs(frame, frame), [[0, 0]])


def test_offsets_compare_on_one_clock_and_mixtures_are_refused():
    left = pd.DataFrame({"t": ["2024-03-01T23:30:00-05:00", "2024-03-02T10:00:00Z"]})
    right = pd.DataFrame({"t": ["2024-03-02T04:30:00+00:00", "2024-03-02T06:00:00-04:00"]})
    assert_array_equal(block.window("t", 0, unit="seconds").pairs(left, right), [[0, 0], [1, 1]])
    assert block.window("t", 0, unit="seconds").dropped(left, right)["left"]["time_zone"] == "utc_offsets"
    aware = pd.DataFrame({"t": pd.to_datetime(["2024-03-02T04:30:00"]).tz_localize("UTC")})
    assert_array_equal(block.window("t", 0, unit="hours").pairs(left, aware), [[0, 0]])
    naive = pd.DataFrame({"t": ["2024-03-02"]})
    with pytest.raises(ValueError, match="one side's times carry UTC offsets and the other's do not"):
        block.window("t", 1, unit="days").pairs(left, naive)
    mixed = pd.DataFrame({"t": ["2024-03-02T04:30:00Z", "2024-03-02T04:30:00"]})
    with pytest.raises(ValueError, match="left column 't' mixes times with and without a UTC offset"):
        block.window("t", 1, unit="days").pairs(mixed, mixed)
    # A bare date ends in "-01" but names no offset.
    dates = pd.DataFrame({"t": ["2024-03-01", "2024-03-02T08:00"]})
    assert block.window("t", 1, unit="days").dropped(dates, dates)["left"]["time_zone"] == "none"


@pytest.mark.parametrize("missing", ["drop", "match"])
def test_window_inside_within_matches_a_grouped_oracle(missing):
    rng = np.random.default_rng(3)
    boros = np.array(["BRONX", "Brooklyn", "QUEENS", None], dtype=object)
    left = pd.DataFrame({"day": rng.integers(0, 15, 80).astype(float), "boro": boros[rng.integers(0, 4, 80)]},
                        index=rng.permutation(80) + 100)
    right = pd.DataFrame({"day": rng.integers(0, 15, 60).astype(float),
                          "borough": [b.lower() if b else b for b in boros[rng.integers(0, 4, 60)]]})
    left.loc[left.index[:4], "day"] = np.nan

    def same_group(i, j):
        a, b = left.boro.iloc[i], right.borough.iloc[j]
        return (a is None and b is None and missing == "match") or (
            a is not None and b is not None and a.lower() == b)

    grouped = block.within(block.window("day", between=(0, 2)), ("boro", "borough"), missing=missing)
    got = grouped.pairs(left, right)
    expected = oracle(left.day.to_numpy(), right.day.to_numpy(), 0, 2, same_group)
    assert len(expected) and set(map(tuple, got)) == set(map(tuple, expected)) and len(got) == len(expected)
    assert grouped.dropped(left, right)["left"]["missing"] == 4
    config = grouped.to_config()
    assert config["type"] == "within" and config["blocker"]["type"] == "window"
    assert config["blocker"] | {"name": None} == {
        "type": "window", "name": None, "columns": [["day", "day"]], "low": 0.0, "high": 2.0, "unit": None,
        "difference": "left_minus_right", "date_format": None, "values": "float64_v1"}
    table = block.candidates(left, right, on="day", blockers=[grouped])
    assert table.attrs["blocking"]["passes"][0]["dropped_values"]["left"]["missing"] == 4
    assert set(table.left_id) <= set(left.index)  # positions map back to the original index labels


def test_unions_with_other_passes_and_labels_each_pair():
    left = pd.DataFrame({"name": ["Acme", "Beta"], "year": [2001, 2010]})
    right = pd.DataFrame({"name": ["Acme", "Gamma"], "year": [2002, 2030]})
    table = block.candidates(left, right, on="name",
                             blockers=[block.exact("name"), block.window("year", 1)])
    assert table.block.tolist() == ["exact:name+window:year[-1..1]"]
    counts = [(p["added_pairs"], p["overlapping_pairs"]) for p in table.attrs["blocking"]["passes"]]
    assert counts == [(1, 0), (0, 1)]
    truth = pd.DataFrame({"left_id": [0, 1], "right_id": [0, 1]})
    assert block.pairs_completeness(table, truth) == 0.5


def test_an_oversized_window_stops_after_one_bounded_batch(monkeypatch):
    frame = pd.DataFrame({"day": np.zeros(20_000), "boro": ["BRONX"] * 20_000})
    for blocker in (block.window("day", 1), block.within(block.window("day", 1), "boro")):
        original, batches = blocker.iter_pairs, []

        def batches_only(left, right, original=original, batches=batches):
            for batch in original(left, right):
                batches.append(len(batch))
                yield batch
                raise AssertionError("guard must stop before requesting another batch")

        def forbidden(*args, **kwargs):
            raise AssertionError("candidates must stream rather than call materializing pairs()")

        monkeypatch.setattr(block, "_PAIR_CHUNK_ROWS", 7)
        monkeypatch.setattr(blocker, "iter_pairs", batches_only)
        monkeypatch.setattr(blocker, "pairs", forbidden)
        with pytest.raises(ValueError, match=r"at least 4 pairs.*max_pairs=3"):
            block.candidates(frame, frame, on="day", blockers=[blocker], max_pairs=3)
        assert batches == [7]


def test_large_tables_are_searched_without_an_all_pairs_comparison():
    # 100,000 by 100,000 is ten billion comparisons; sorting and binary search need none of them.
    rng = np.random.default_rng(11)
    left = pd.DataFrame({"v": rng.integers(0, 10**9, 100_000)})
    right = pd.DataFrame({"v": np.concatenate([left.v.to_numpy()[:500], rng.integers(0, 10**9, 99_500)])})
    got = block.window("v", 0).pairs(left, right)
    merged = left.reset_index().merge(right.reset_index(), on="v")
    assert len(got) == len(merged) >= 500
    assert set(map(tuple, got)) == set(zip(merged.index_x, merged.index_y))


def test_saturating_bounds_near_the_ends_of_the_timestamp_range():
    edge = pd.DataFrame({"t": pd.to_datetime(["1677-09-22", "2262-04-11", "2000-01-01", "1990-06-01"])})
    days = 290 * 365
    wide = block.window("t", days, unit="days")
    # Python integers cannot wrap, so they are the reference for sums that leave the int64 range.
    stamps = [int(value) for value in edge.t.astype("int64")]
    expected = [[i, j] for i, a in enumerate(stamps) for j, b in enumerate(stamps)
                if abs(a - b) <= days * 86_400 * 10**9]
    assert [0, 1] not in expected and [1, 2] in expected and [0, 3] not in expected
    assert_array_equal(wide.pairs(edge, edge), expected)
    assert_array_equal(block.window("t", between=(-days, 0), unit="days").pairs(edge, edge),
                       [pair for pair in expected if stamps[pair[0]] <= stamps[pair[1]]])
    with pytest.raises(ValueError, match="about 292 years"):
        block.window("t", 200_000, unit="weeks").pairs(edge, edge)


def test_invalid_windows_say_what_to_change():
    numbers = pd.DataFrame({"year": [2001], "flag": [True], "day": pd.to_datetime(["2024-03-01"])})
    for kwargs, message in [
            ({}, "either a `tolerance` or `between"), ({"tolerance": 1, "between": (0, 1)}, "not both"),
            ({"tolerance": -1}, "zero or positive"), ({"between": (3, 0)}, "low <= high"),
            ({"between": (0, float("inf"))}, "finite numbers"), ({"tolerance": True}, "finite numbers"),
            ({"between": 3}, "finite numbers"), ({"tolerance": 1, "unit": "months"}, "`unit` must be one of"),
            ({"tolerance": 1, "date_format": "%Y"}, "give `unit` as well"),
            ({"tolerance": 1, "unit": "days", "date_format": ""}, "strptime format"),
            ({"tolerance": 1, "name": " "}, "nonempty string")]:
        with pytest.raises(ValueError, match=message):
            block.window("year", **kwargs)
    with pytest.raises(ValueError, match="one-sided field"):
        block.window(("published", None), 1)
    with pytest.raises(ValueError, match="left column 'year' holds numbers, not dates"):
        block.window("year", 1, unit="days").pairs(numbers, numbers)
    with pytest.raises(ValueError, match="left column 'day' holds dates; give `window` a `unit`"):
        block.window("day", 1).pairs(numbers, numbers)
    with pytest.raises(ValueError, match="holds booleans"):
        block.window("flag", 1).pairs(numbers, numbers)
    with pytest.raises(ValueError, match="left data has no column 'month'"):
        block.window("month", 1).pairs(numbers, numbers)
    assert block.window("year", 1, unit="d").unit == "days"
    empty = numbers.iloc[:0]
    assert block.window("year", 1).pairs(empty, numbers).shape == (0, 2)


ARTICLES = pd.DataFrame({
    "article_id": ["a1", "a2", "a3", "a4"], "boro": ["BROOKLYN", "BRONX", "BROOKLYN", "QUEENS"],
    "published": ["2024-03-03", "2024-03-02", "2024-02-28", "not dated"],
    "text": ["Man shot on Fulton St in Bedford-Stuyvesant", "Teen wounded in Mott Haven",
             "Shots fired in Bedford-Stuyvesant, no injuries", "Shooting in Jamaica"]})
INCIDENTS = pd.DataFrame({
    "incident_id": [501, 502, 503, 504], "BORO": ["BROOKLYN", "BRONX", "BROOKLYN", "QUEENS"],
    "OCCUR_DATE": ["03/02/2024", "03/02/2024", "03/05/2024", "03/02/2024"],
    "neighborhood": ["Bedford-Stuyvesant", "Mott Haven", "Bedford-Stuyvesant", "Jamaica"]})
RULE = "Record A is a news article that reports the shooting incident in record B."


def test_relation_linking_with_a_grouped_window_records_its_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(name, raising=False)
    passes = [block.within(block.window(("published", "OCCUR_DATE"), between=(0, 3), unit="days",
                                        date_format=(None, "%m/%d/%Y")), ("boro", "BORO"))]
    linker = jlink.Linker(style="rule", definition=RULE, blockers=passes, on=[
        ("text", None), ("published", None), (None, "OCCUR_DATE"), (None, "neighborhood")])
    with pytest.warns(UserWarning, match="could not read 1 left values in 'published'"):
        estimate = linker.estimate(ARTICLES, INCIDENTS, left_id="article_id", right_id="incident_id")
    assert estimate["pairs"] == 2  # a3 precedes both Brooklyn incidents; a4 has no readable date
    fake = FakeJev(lambda state, asked: 0.9 if state["record_b"]["neighborhood"]
                   in state["record_a"]["text"] else 0.1)
    with pytest.warns(UserWarning, match="could not read"):
        result = linker.link(ARTICLES, INCIDENTS, left_id="article_id", right_id="incident_id",
                             how="many-to-one", progress=False, transport=fake.transport)
    assert set(zip(result.links.left_id, result.links.right_id)) == {("a1", 501), ("a2", 502)}
    assert all("OCCUR_DATE" in body["state"]["record_b"] for body in fake.bodies)
    name = "within:boro=BORO[drop](window:published=OCCUR_DATE[0..3d])"
    assert result.settings["blockers"] == [name] and name in result.methods()
    assert result.settings["blocker_configs"] == [passes[0].to_config()]
    lost = result.settings["blocking"]["passes"][0]["dropped_values"]
    assert lost["left"]["unparseable"] == 1 and lost["right"]["unparseable"] == 0
    back = jlink.load(result.save(tmp_path / "run"))
    assert back.settings["blocking"] == result.settings["blocking"]
    truth = pd.DataFrame({"left_id": ["a1", "a2", "a4"], "right_id": [501, 502, 504]})
    assert block.pairs_completeness(result.candidates, truth) == pytest.approx(2 / 3)


def test_cli_window_and_within_rules(tmp_path, capsys):
    left, right = tmp_path / "articles.csv", tmp_path / "incidents.csv"
    ARTICLES.to_csv(left, index=False)
    INCIDENTS.to_csv(right, index=False)
    base = ["estimate", str(left), str(right), "--on", "text=", "--on", "=neighborhood",
            "--left-id", "article_id", "--right-id", "incident_id"]
    cli([*base, "--block", "within:boro=BORO:window:published=OCCUR_DATE:0..3d",
         "--date-format", "=%m/%d/%Y"])
    captured = capsys.readouterr()
    assert "4 left records; 4 right records; 2 candidate pairs" in captured.out
    assert ("within:boro=BORO[drop](window:published=OCCUR_DATE[0..3d]): 1 left and 0 right records have a "
            "missing or unreadable value") in captured.out
    assert "jlink: warning: blocker" in captured.err and "could not read 1 left values" in captured.err
    # Without the format the register's dates are unreadable: reported, never guessed.
    cli([*base, "--block", "window:published=OCCUR_DATE:0..3d"])
    captured = capsys.readouterr()
    assert "0 candidate pairs" in captured.out and "1 left and 4 right records" in captured.out
    # Symmetric and ungrouped: a1 and a2 are within three days of all four incidents, a3 of three.
    cli([*base, "--block", "window:published=OCCUR_DATE:3d", "--date-format", "=%m/%d/%Y"])
    assert "11 candidate pairs" in capsys.readouterr().out


@pytest.mark.parametrize("rule", ["window:year", "window:year:", "window:year:abc", "window:year:3..1",
                                  "window:year:-1", "window:year:1x", "window:year:1..2..3", "window:a+b:1",
                                  "window:year:1e3", "window:year=:1", "within:state", "within:state:ngrams",
                                  "within:state:window:year", "within::exact:name",
                                  "within:state:bogus:name"])
def test_cli_rejects_malformed_window_and_within_rules(tmp_path, rule, capsys):
    with pytest.raises(SystemExit) as exc:
        cli(["estimate", "left.csv", "right.csv", "--on", "name", "--block", rule])
    assert exc.value.code == 2 and "accepted forms are" in capsys.readouterr().err


def test_cli_window_spec_forms():
    spec = importlib.import_module("jlink.cli")._block_spec
    assert spec("window:year:1") == ("window", ["year"], {"tolerance": 1.0, "unit": None})
    assert spec("window:a=b:-1.5..0.25h") == ("window", [("a", "b")], {"between": (-1.5, 0.25), "unit": "h"})
    assert spec("within:st=state+year:within:zip:exact:name") == (
        "within", [("st", "state"), "year"],
        {"child": ("within", ["zip"], {"child": ("exact", ["name"], {})})})
    dates = importlib.import_module("jlink.cli")._date_formats
    assert dates(None) is None and dates("%d.%m.%Y") == ("%d.%m.%Y", "%d.%m.%Y")
    assert dates("=%m/%d/%Y") == (None, "%m/%d/%Y") and dates("%m/%d/%Y=") == ("%m/%d/%Y", None)
    for bad in ("=", "a=b=c", " "):
        with pytest.raises(ValueError, match="--date-format"):
            dates(bad)


@pytest.mark.parametrize("value,date_format,utc", [
    ("03/01/2024 12:00 PM -0500", "%m/%d/%Y %I:%M %p %z", "2024-03-01T17:00:00Z"),
    ("20240301120000-0500", "%Y%m%d%H%M%S%z", "2024-03-01T17:00:00Z"),
    ("03/01/2024 05:00 PM UTC", "%m/%d/%Y %I:%M %p %Z", "2024-03-01T17:00:00Z")])
def test_custom_date_formats_preserve_offset_awareness(value, date_format, utc):
    left = pd.DataFrame({"date": [value, "unreadable", None]})
    right = pd.DataFrame({"date": [utc]})
    window = block.window("date", 0, unit="seconds", date_format=(date_format, None))
    assert_array_equal(window.pairs(left, right), [[0, 0]])
    assert window.dropped(left, right)["left"] == {
        "column": "date", "missing": 1, "unparseable": 1, "time_zone": "utc_offsets"}
    with pytest.raises(ValueError, match="one side's times carry UTC offsets"):
        window.pairs(left, pd.DataFrame({"date": [utc[:-1]]}))


def test_custom_date_format_handles_different_offsets_and_literal_directives():
    left = pd.DataFrame({"date": ["03/01/2024 12:00 PM -0500", "03/01/2024 06:00 PM +0100"]})
    right = pd.DataFrame({"date": ["2024-03-01T17:00:00Z"]})
    window = block.window("date", 0, unit="seconds", date_format=("%m/%d/%Y %I:%M %p %z", None))
    assert_array_equal(window.pairs(left, right), [[0, 0], [1, 0]])
    literal = block.window("date", 0, unit="seconds", date_format=("%Y-%m-%d %%z", None))
    naive = pd.DataFrame({"date": ["2024-03-01 %z"]})
    assert_array_equal(literal.pairs(naive, pd.DataFrame({"date": ["2024-03-01"]})), [[0, 0]])
    assert literal.dropped(naive, pd.DataFrame({"date": ["2024-03-01"]}))["left"]["time_zone"] == "none"


@pytest.mark.parametrize("bounds", [(-0.6e-9, 0.6e-9), (0.6e-9, 0.6e-9),
                                    (0.6e-9, 1.6e-9), (-1.6e-9, -0.6e-9)])
def test_fractional_nanosecond_bounds_do_not_broaden_the_window(bounds):
    from fractions import Fraction

    values = [pd.Timestamp("2024-01-01") + pd.Timedelta(n, unit="ns") for n in range(4)]
    frame = pd.DataFrame({"date": values})
    low, high = (Fraction(v) * 10**9 for v in bounds)
    expected = np.array([[i, j] for i, a in enumerate(values) for j, b in enumerate(values)
                         if low <= a.value - b.value <= high], dtype=np.int64).reshape(-1, 2)
    window = block.window("date", between=bounds, unit="seconds")
    assert_array_equal(window.pairs(frame, frame), expected)
    if bounds == (-0.6e-9, 0.6e-9):
        assert_array_equal(block.window("date", 0.6e-9, unit="seconds").pairs(frame, frame), expected)


@pytest.mark.parametrize("bounds", [(-1, -1), (1, 1), (-2, -1), (1, 2), (0, 0), (-1, 1)])
def test_date_search_overflow_never_matches_a_saturated_endpoint(bounds):
    values = [pd.Timestamp.min, pd.Timestamp.min + pd.Timedelta(1, unit="s"),
              pd.Timestamp.max - pd.Timedelta(1, unit="s"), pd.Timestamp.max]
    frame = pd.DataFrame({"date": values})
    low, high = (v * 10**9 for v in bounds)
    expected = np.array([[i, j] for i, a in enumerate(values) for j, b in enumerate(values)
                         if low <= a.value - b.value <= high], dtype=np.int64).reshape(-1, 2)
    assert_array_equal(block.window("date", between=bounds, unit="seconds").pairs(frame, frame), expected)

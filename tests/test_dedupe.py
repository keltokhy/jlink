"""Dedupe: pairs within one table, clusters that resist a bad edge, and reclustering without calls."""

import importlib
import itertools
import json
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import jlink
from fakes import FakeJev
from jlink import block

cli = importlib.import_module("jlink.cli").cli

FIRMS = pd.DataFrame({
    "rid": ["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
    "name": ["Acme Widgets Inc", "ACME WIDGETS, INC.", "Acme Widget Co", "Zeta Holdings", "Zeta Holdings LLC",
             "Northwind Traders", "Acme Widgets Inc"],
    "year": [2001, 2001, 2002, 2010, 2011, 1999, 2001]})
SAME = [{"r1", "r2", "r3", "r7"}, {"r4", "r5"}]


def oracle(state, asked):
    a, b = state["record_a"]["name"], state["record_b"]["name"]
    return 0.95 if a[:4].casefold() == b[:4].casefold() else 0.03


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(name, raising=False)


def partition(clusters):
    return {frozenset(group) for _, group in clusters.groupby("cluster_id")["id"]}


def two_groups_and_a_bad_edge(bad=0.9, cross=0.05, within=0.95):
    a, b = ["a1", "a2", "a3"], ["b1", "b2", "b3"]
    rows = [(x, y, within) for group in (a, b) for x, y in itertools.combinations(group, 2)]
    rows += [(x, y, bad if (x, y) == ("a3", "b1") else cross) for x in a for y in b]
    return pd.DataFrame(rows, columns=["left_id", "right_id", "p"]), a + b + ["alone"]


# Pair generation

@pytest.mark.parametrize("passes", [
    None, [block.ngrams("name", k=4)], [block.ngrams("name", k=3, reverse=True)], [block.exact("year")],
    [block.window("year", 1)], [block.within(block.ngrams("name", k=7, min_sim=0), "year")],
    [block.exact("year"), block.ngrams("name", k=7), block.window("year", between=(0, 1))]])
def test_pairs_hold_no_record_twice_and_no_pair_twice(passes):
    table = block.self_candidates(FIRMS, on=["name", "year"], blockers=passes, id="rid")
    position = {rid: i for i, rid in enumerate(FIRMS.rid)}
    left, right = table.left_id.map(position), table.right_id.map(position)
    assert len(table) and (left < right).all()  # never a record with itself; earlier row first
    assert not pd.MultiIndex.from_arrays([left, right]).has_duplicates
    assert table.attrs["blocking"]["unordered"] is True and table.sim.between(0, 1).all()


def test_pairs_equal_the_unordered_union_of_a_two_table_run():
    passes = [block.ngrams("name", k=3), block.window("year", between=(0, 1))]
    both = block.candidates(FIRMS, FIRMS, on="name", blockers=passes, left_id="rid", right_id="rid")
    expected = {frozenset(pair) for pair in zip(both.left_id, both.right_id) if pair[0] != pair[1]}
    table = block.self_candidates(FIRMS, on="name", blockers=passes, id="rid")
    assert {frozenset(pair) for pair in zip(table.left_id, table.right_id)} == expected
    assert len(table) == len(expected) < len(both)
    # One direction of a one-sided window is enough to propose the pair: years at most one apart.
    years = block.self_candidates(FIRMS, on="name", blockers=[block.window("year", between=(0, 1))], id="rid")
    assert {frozenset(pair) for pair in zip(years.left_id, years.right_id)} == {
        frozenset(pair) for pair in itertools.combinations(FIRMS.rid, 2)
        if abs(FIRMS.set_index("rid").year[pair[0]] - FIRMS.set_index("rid").year[pair[1]]) <= 1}


def test_the_earlier_row_is_always_the_left_record():
    class Backward(block.Blocker):
        name = "backward"

        def pairs(self, left, right):
            return np.array([[5, 1], [1, 5], [2, 2], [6, 0]], dtype=np.int64)

    table = block.self_candidates(FIRMS, on="name", blockers=[Backward()], id="rid")
    assert list(zip(table.left_id, table.right_id)) == [("r1", "r7"), ("r2", "r6")]
    info = table.attrs["blocking"]["passes"][0]
    assert info["proposed_pairs"] == 4 and info["unique_pairs"] == 2
    unlabeled = block.self_candidates(FIRMS.set_index("rid"), on="name", blockers=[Backward()])
    assert list(zip(unlabeled.left_id, unlabeled.right_id)) == [("r1", "r7"), ("r2", "r6")]


def test_default_pass_leaves_ten_other_neighbors_and_limits_count_unordered_pairs():
    table = block.self_candidates(FIRMS, on="name", id="rid")
    config = table.attrs["blocking"]["passes"][0]["config"]
    assert config["type"] == "ngrams" and config["k"] == 11
    same_year = pd.DataFrame({"name": ["Acme"] * 5, "year": [2000] * 5})
    assert len(block.self_candidates(same_year, on="name", blockers=[block.exact("year")], max_pairs=10)) == 10
    with pytest.raises(ValueError, match=r"at least 10 pairs.*max_pairs=9"):
        block.self_candidates(same_year, on="name", blockers=[block.exact("year")], max_pairs=9)
    with pytest.raises(ValueError, match="each `on` field is one column name.*\\('name', 'year'\\)"):
        block.self_candidates(FIRMS, on=[("name", "year")], id="rid")
    with pytest.raises(ValueError, match="one side only"):
        block.self_candidates(FIRMS, on=[("name", None)], id="rid")
    with pytest.raises(ValueError, match="duplicate IDs"):
        block.self_candidates(FIRMS, on="name", id="year")


def test_keyed_pass_on_one_table_warns_only_when_no_record_has_a_key():
    # One table cannot disagree with itself about how a key is coded, so the two-table warning
    # ("no value in common between left and right") has no meaning here.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Every key different: no record matches another. That is an answer, not a fault.
        unique = block.self_candidates(FIRMS, on="name", blockers=[block.exact("rid")], id="rid")
        assert unique.empty and unique.attrs["blocking"]["passes"][0]["proposed_pairs"] == 7  # itself, 7 times
        grouped = block.self_candidates(FIRMS, on="name", id="rid",
                                        blockers=[block.within(block.window("year", 0), "rid")])
        assert grouped.empty
    blank = FIRMS.assign(state=[None, "", "  ", pd.NA, np.nan, "!!", None])
    for blocker in (block.exact("state"), block.within(block.ngrams("name"), "state")):
        with pytest.warns(UserWarning, match="proposed no pairs: none of the 7 records has a complete key") as caught:
            assert block.self_candidates(blank, on="name", blockers=[blocker], id="rid").empty
        assert "left" not in str(caught[0].message) and caught[0].filename == __file__
    # With missing="match", records that all lack the key form one group, as they do for two tables.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        together = block.self_candidates(blank, on="name", id="rid", blockers=[
            block.within(block.ngrams("name", k=7, min_sim=0), "state", missing="match")])
    assert len(together) == 21


def test_unordered_pairs_completeness():
    table = block.self_candidates(FIRMS, on="name", blockers=[block.exact("year")], id="rid")
    truth = pd.DataFrame({"left_id": ["r2", "r1", "r5"], "right_id": ["r1", "r7", "r4"]})
    assert block.pairs_completeness(table, truth) == pytest.approx(1 / 3)  # only (r1, r7) is in that order
    assert block.pairs_completeness(table, truth, unordered=True) == pytest.approx(2 / 3)
    assert np.isnan(block.pairs_completeness(table, truth.iloc[:0], unordered=True))


# Clustering

def test_average_linkage_resists_a_single_bad_edge_that_components_chain_through():
    scores, ids = two_groups_and_a_bad_edge()
    average = jlink.cluster(scores, ids=ids)
    assert partition(average) == {frozenset({"a1", "a2", "a3"}), frozenset({"b1", "b2", "b3"}),
                                  frozenset({"alone"})}
    assert average.cluster_id.tolist() == [0, 0, 0, 1, 1, 1, 2]
    assert average.cluster_size.tolist() == [3, 3, 3, 3, 3, 3, 1]
    chained = jlink.cluster(scores, ids=ids, linkage="components")
    assert partition(chained) == {frozenset({"a1", "a2", "a3", "b1", "b2", "b3"}), frozenset({"alone"})}


def test_a_lone_wrong_pair_joins_two_groups_only_when_unproposed_pairs_are_ignored():
    scores, ids = two_groups_and_a_bad_edge()
    two = {frozenset({"a1", "a2", "a3"}), frozenset({"b1", "b2", "b3"}), frozenset({"alone"})}
    only = scores[(scores.p > 0.5)]  # blocking never proposed the other pairs between the groups
    assert partition(jlink.cluster(only, ids=ids)) == two
    ignoring = jlink.cluster(only, ids=ids, unproposed="ignore")
    assert partition(ignoring) == partition(jlink.cluster(only, ids=ids, linkage="components")) != two
    # A candidate pair that was proposed but never judged is no evidence under either setting.
    unjudged = scores.assign(p=scores.p.where(scores.p > 0.5))
    assert partition(jlink.cluster(unjudged, ids=ids)) == partition(ignoring)
    assert partition(jlink.cluster(unjudged, ids=ids, unproposed="ignore")) == partition(ignoring)


def test_counting_unproposed_pairs_splits_a_group_that_blocking_covered_thinly():
    ring = pd.DataFrame({"left_id": ["a", "b", "c", "d", "e"], "right_id": ["b", "c", "d", "e", "a"],
                         "p": [0.9, 0.92, 0.94, 0.96, 0.98]})  # five records, each compared with two
    assert partition(jlink.cluster(ring, unproposed="ignore")) == {frozenset("abcde")}
    assert partition(jlink.cluster(ring, linkage="components")) == {frozenset("abcde")}
    # e and a merge first at 0.98; d then matches one of those two and no more, a mean of 0.48.
    assert partition(jlink.cluster(ring)) == {frozenset("ae"), frozenset("cd"), frozenset("b")}
    clique = pd.DataFrame(list(itertools.combinations("abcde", 2)), columns=["left_id", "right_id"]).assign(p=0.9)
    assert partition(jlink.cluster(clique)) == {frozenset("abcde")}


def test_limits_of_average_linkage_are_the_documented_ones():
    scores, ids = two_groups_and_a_bad_edge()
    # A bad edge that outranks every true pair merges first. The damage stays with the records it
    # touches: depending on the other scores they split off or one is pulled across, but the two
    # groups never chain into one.
    worst, ids = two_groups_and_a_bad_edge(bad=0.99, within=0.9)
    assert partition(jlink.cluster(worst, ids=ids)) == {
        frozenset({"a1", "a2"}), frozenset({"a3", "b1"}), frozenset({"b2", "b3"}), frozenset({"alone"})}
    pulled, ids = two_groups_and_a_bad_edge(bad=0.99, cross=0.2)
    assert partition(jlink.cluster(pulled, ids=ids)) == {
        frozenset({"a1", "a2", "a3", "b1"}), frozenset({"b2", "b3"}), frozenset({"alone"})}


def test_threshold_boundary_is_exact_and_inclusive():
    scores = pd.DataFrame({"left_id": ["a", "a", "b"], "right_id": ["b", "c", "c"], "p": [1.0, 0.75, 0.25]})
    # a and b merge at 1.0; their judged pairs with c average exactly 0.5.
    assert partition(jlink.cluster(scores, threshold=0.5)) == {frozenset("abc")}
    assert partition(jlink.cluster(scores, threshold=0.5000000001)) == {frozenset("ab"), frozenset("c")}
    # Means are exact, not rounded: 0.95 and 0.05 are stored as binary fractions whose mean is a
    # hair under one half, although 0.95 + 0.05 == 1.0 in floating point.
    assert Fraction(0.95) + Fraction(0.05) < 1 and 0.95 + 0.05 == 1.0
    tenths = pd.DataFrame({"left_id": ["a", "a", "b"], "right_id": ["b", "c", "c"], "p": [1.0, 0.95, 0.05]})
    assert partition(jlink.cluster(tenths, threshold=0.5)) == {frozenset("ab"), frozenset("c")}
    assert jlink.cluster(tenths, threshold=1.0).cluster_size.tolist() == [2, 2, 1]
    assert partition(jlink.cluster(tenths, threshold=0.0, linkage="components")) == {frozenset("abc")}


def reference_average_linkage(scores, ids, threshold, unproposed):
    """The definition, slowly: rescan every pair of clusters after every merge, in exact arithmetic."""
    position = {record: i for i, record in enumerate(ids)}
    weight = {frozenset((position[a], position[b])): None if np.isnan(p) else Fraction(p)
              for a, b, p in zip(scores.left_id, scores.right_id, scores.p)}
    clusters = [[i] for i in range(len(ids))]
    while True:
        best = None
        for x, y in itertools.combinations(sorted(clusters), 2):
            pairs = [frozenset((i, j)) for i in x for j in y]
            between = [weight[pair] for pair in pairs if weight.get(pair) is not None]
            voters = len(between) if unproposed == "ignore" else sum(weight.get(pair, 0) is not None for pair in pairs)
            if between:
                mean = sum(between) / voters
                if mean >= Fraction(threshold) and (best is None or mean > best[0]):
                    best = (mean, x, y)  # sorted clusters, strict '>': ties go to the earliest members
        if best is None:
            return {frozenset(ids[i] for i in members) for members in clusters}
        clusters = [c for c in clusters if c is not best[1] and c is not best[2]] + [sorted(best[1] + best[2])]


@pytest.mark.parametrize("unproposed", ["nonmatch", "ignore"])
@pytest.mark.parametrize("seed", range(12))
def test_average_linkage_equals_the_slow_definition_and_ignores_row_order(seed, unproposed):
    rng = np.random.default_rng(seed)
    ids = [f"n{i:02d}" for i in range(14)]
    pairs = [pair for pair in itertools.combinations(ids, 2) if rng.random() < 0.45]
    # Few distinct values, so that equal means occur and the tie rule is exercised.
    p = rng.choice([0.05, 0.25, 0.5, 0.75, 0.95, np.nan], size=len(pairs))
    scores = pd.DataFrame(pairs, columns=["left_id", "right_id"]).assign(p=p)
    threshold = [0.5, 0.6, 0.3][seed % 3] if unproposed == "ignore" else [0.25, 0.4, 0.15][seed % 3]
    expected = reference_average_linkage(scores, ids, threshold, unproposed)
    first = jlink.cluster(scores, ids=ids, threshold=threshold, unproposed=unproposed)
    assert partition(first) == expected and len(expected) < len(ids)
    shuffled = scores.sample(frac=1, random_state=seed)
    flip = rng.random(len(shuffled)) < 0.5  # which record is written first does not matter
    mirrored = shuffled.assign(left_id=np.where(flip, shuffled.right_id, shuffled.left_id),
                               right_id=np.where(flip, shuffled.left_id, shuffled.right_id))
    for table in (mirrored, scores):
        pd.testing.assert_frame_equal(
            jlink.cluster(table, ids=ids, threshold=threshold, unproposed=unproposed), first)
    components = jlink.cluster(scores, ids=ids, threshold=threshold, linkage="components")
    assert all(any(group <= whole for whole in partition(components)) for group in partition(first))


def test_cluster_validates_its_input():
    scores, ids = two_groups_and_a_bad_edge()
    assert jlink.cluster(scores).id.tolist() == ids[:-1]  # without ids: records the pairs mention, in order
    assert jlink.cluster(scores.iloc[:0]).empty and len(jlink.cluster(scores.iloc[:0], ids=ids)) == 7
    for bad, message in [
            (dict(linkage="complete"), "`linkage` must be one of average, components"),
            (dict(unproposed="zero"), "`unproposed` must be one of ignore, nonmatch"),
            (dict(threshold=1.5), "threshold must be between 0 and 1"),
            (dict(ids=ids[:-2]), "not in `ids`: 'b3'"), (dict(ids=ids + ["a1"]), "each record once")]:
        with pytest.raises(ValueError, match=message):
            jlink.cluster(scores, **bad)
    with pytest.raises(ValueError, match="pair the record 'a1' with itself"):
        jlink.cluster(pd.concat([scores, pd.DataFrame({"left_id": ["a1"], "right_id": ["a1"], "p": [1.0]})]))
    with pytest.raises(ValueError, match="in both orders"):
        jlink.cluster(pd.concat([scores, scores.iloc[:1].rename(
            columns={"left_id": "right_id", "right_id": "left_id"})]))
    with pytest.raises(ValueError, match="needs a 'p' column"):
        jlink.cluster(scores.drop(columns="p"))


# End to end

def run(fake=None, **kwargs):
    fake = fake or FakeJev(oracle)
    options = {"entity": "firm", "on": "name", "id": "rid", "progress": False, "transport": fake.transport}
    return jlink.dedupe(FIRMS, **(options | kwargs)), fake


def test_each_unordered_pair_is_asked_once_with_the_earlier_row_as_record_a():
    result, fake = run()
    asked = [(body["state"]["record_a"]["name"], body["state"]["record_b"]["name"]) for body in fake.bodies]
    assert len(asked) == len(result.scores) == 7
    assert len(set(map(frozenset, zip(result.scores.left_id, result.scores.right_id)))) == 7
    names, rows = FIRMS.set_index("rid").name, {rid: i for i, rid in enumerate(FIRMS.rid)}
    for left, right in zip(result.scores.left_id, result.scores.right_id):
        assert rows[left] < rows[right] and (names[left], names[right]) in asked
    assert partition(result.clusters) == {frozenset(group) for group in SAME} | {frozenset({"r6"})}
    assert result.clusters.id.tolist() == FIRMS.rid.tolist()
    labeled = result.labeled()
    assert labeled.columns.tolist() == ["rid", "name", "year", "cluster_id", "cluster_size"]
    assert labeled.cluster_id.tolist() == [0, 0, 0, 1, 1, 2, 0]
    together = set(map(frozenset, zip(result.links.left_id, result.links.right_id)))
    assert together and all(any(pair <= group for group in SAME) for pair in together)
    assert result.links.p.is_monotonic_decreasing


def test_reclustering_from_saved_scores_makes_no_calls(tmp_path):
    chain = {("Acme Widget Co", "Zeta Holdings"): 0.9}  # one wrong pair between the two groups
    fake = FakeJev(lambda state, asked: chain.get(
        (state["record_a"]["name"], state["record_b"]["name"]), oracle(state, asked)))
    result, _ = run(fake, blockers=[block.ngrams("name", k=7, min_sim=0)])
    calls = len(fake.bodies)
    assert calls == len(result.scores) == 21 and result.meter.calls == calls
    assert partition(result.clusters) == {frozenset(group) for group in SAME} | {frozenset({"r6"})}
    back = jlink.load(result.save(tmp_path / "run"))
    assert isinstance(back, jlink.DedupeResult) and back.settings == result.settings
    pd.testing.assert_frame_equal(back.clusters, result.clusters)
    pd.testing.assert_series_equal(back.scores.p, result.scores.p)
    chained = back.recluster(linkage="components")
    assert partition(chained.clusters) == {frozenset(set().union(*SAME)), frozenset({"r6"})}
    assert chained.settings["linkage"] == "components" and result.settings["linkage"] == "average"
    strict = chained.recluster(threshold=0.99, linkage="average")
    assert strict.clusters.cluster_size.eq(1).all() and strict.settings["threshold"] == 0.99
    pd.testing.assert_frame_equal(strict.recluster(threshold=0.5).clusters, result.clusters)
    assert len(fake.bodies) == calls and strict.meter is back.meter and back.meter.calls == calls
    assert "connected components" in chained.methods() and "average-linkage" in back.methods()
    with pytest.raises(ValueError, match="loaded from disk"):
        back.labeled()
    assert back.labeled(FIRMS).cluster_id.tolist() == result.labeled().cluster_id.tolist()
    with pytest.raises(ValueError, match="IDs differ from the saved clusters"):
        back.labeled(FIRMS.iloc[::-1])
    with pytest.raises(ValueError, match="`linkage` must be one of"):
        back.recluster(linkage="single")


def test_float_ids_survive_a_saved_dedupe_run(tmp_path):
    # Stata often stores numeric IDs as doubles; main records that kind for link runs, and so does dedupe.
    table = FIRMS.assign(rid=[1001.0, 1002.0, 1003.5, 2.0 ** 60, 5.0, 6.0, 7.0])
    result = jlink.dedupe(table, entity="firm", on="name", id="rid", progress=False,
                          transport=FakeJev(oracle).transport)
    directory = result.save(tmp_path / "floats")
    assert json.loads((directory / "settings.json").read_text())["id_kinds"] == {
        "left_id": "float", "right_id": "float", "id": "float"}
    back = jlink.load(directory)
    assert back.clusters["id"].dtype == back.scores["left_id"].dtype == "float64"
    pd.testing.assert_frame_equal(back.clusters, result.clusters)
    pd.testing.assert_frame_equal(back.labeled(table), result.labeled())
    pd.testing.assert_frame_equal(back.recluster(linkage="components").clusters,
                                  result.recluster(linkage="components").clusters)


def test_settings_report_methods_and_cached_rerun(tmp_path):
    result, fake = run()
    s = result.settings
    assert s["task"] == "dedupe" and s["id"] == s["left_id"] == s["right_id"] == "rid"
    assert s["linkage"] == "average" and s["threshold"] == 0.5 and s["n_records"] == 7
    assert s["pair_order"] == "earlier_row_is_record_a_v1" and s["blocking"]["unordered"] is True
    assert s["blockers"] == ["ngrams:name"] and s["blocker_configs"][0]["k"] == 11
    assert s["inputs"]["records"]["compared"]["columns"] == ["name"]
    assert s["inputs"]["records"]["full"]["columns"] == ["rid", "name", "year"]
    json.dumps(s, allow_nan=False)
    report, methods = result.report(), result.methods()
    assert "(dedupe)" in report and "Records: 7 in one table" in report
    assert "Clusters: 2 of two or more records, holding 6 records; 1 records alone (average linkage" in report
    assert "unordered pairs, earlier row as record A" in report
    assert "We searched 7 records for groups of records that refer to the same firm" in methods
    assert "each unordered pair was kept once" in methods and "Each pair was judged once" in methods
    assert "the record from the earlier row presented first" in methods
    assert "2 groups of two or more records, holding 6 of the 7 records" in methods
    again, quiet = run(budget=0)
    assert not quiet.bodies and again.meter.cached == len(result.scores)
    pd.testing.assert_frame_equal(again.clusters, result.clusters)


def test_presentation_order_follows_row_order_and_is_part_of_the_cached_question():
    distinct = FIRMS.iloc[:6]  # no two records share a name, so every state names its order
    asked = FakeJev(oracle)
    first = jlink.dedupe(distinct, entity="firm", on="name", id="rid", progress=False,
                         transport=asked.transport)
    assert len(asked.bodies) == len(first.scores) == 4
    # The same table again is answered from the cache.
    again = FakeJev(oracle)
    jlink.dedupe(distinct, entity="firm", on="name", id="rid", progress=False, transport=again.transport)
    assert not again.bodies
    # Reversed rows present every pair the other way round: a different state, asked and paid anew.
    flipped, swapped = distinct.iloc[::-1].reset_index(drop=True), FakeJev(oracle)
    second = jlink.dedupe(flipped, entity="firm", on="name", id="rid", progress=False,
                          transport=swapped.transport)
    assert len(swapped.bodies) == 4
    assert {(b["state"]["record_b"]["name"], b["state"]["record_a"]["name"]) for b in swapped.bodies} == {
        (b["state"]["record_a"]["name"], b["state"]["record_b"]["name"]) for b in asked.bodies}
    # This fake answers both orders alike, so the clusters agree; nothing here measures whether Jev does.
    assert partition(second.clusters) == partition(first.clusters)


def test_unproposed_setting_is_recorded_reported_and_changeable_without_calls():
    lone = {("Acme Widget Co", "Zeta Holdings"): 0.9}  # the only pair blocking proposes between the groups
    fake = FakeJev(lambda state, asked: lone.get(
        (state["record_a"]["name"], state["record_b"]["name"]), oracle(state, asked)))

    class Extra(block.Blocker):
        name = "extra"

        def pairs(self, left, right):
            return np.array([[2, 3]], dtype=np.int64)

    result, _ = run(fake, blockers=[block.ngrams("name", k=11), Extra()])
    calls = len(fake.bodies)
    assert result.settings["unproposed"] == "nonmatch"
    assert partition(result.clusters) == {frozenset(group) for group in SAME} | {frozenset({"r6"})}
    split = result.split_pairs()
    assert list(zip(split.left_id, split.right_id, split.p)) == [("r3", "r4", 0.9)]
    assert "Pairs at or above the threshold left in different clusters: 1" in result.report()
    assert "unproposed pairs as non-matches" in result.report()
    assert "a pair that blocking never proposed counted as zero" in result.methods()
    ignoring = result.recluster(unproposed="ignore")
    assert partition(ignoring.clusters) == {frozenset(set().union(*SAME)), frozenset({"r6"})}
    assert ignoring.settings["unproposed"] == "ignore" and ignoring.split_pairs().empty
    assert "mean p of the judged pairs between two clusters" in ignoring.report()
    assert "counted neither for nor against a merge" in ignoring.methods()
    chained = ignoring.recluster(linkage="components")
    assert "left in different clusters" not in chained.report() and chained.split_pairs().empty
    assert len(fake.bodies) == calls
    with pytest.raises(ValueError, match="`unproposed` must be one of"):
        result.recluster(unproposed="zero")


def test_audit_and_evaluate_measure_clusters_pair_by_pair():
    result, _ = run(blockers=[block.ngrams("name", k=7, min_sim=0)])
    sample = result.audit_sample(n=len(result.scores))
    assert {"a_name", "b_name", "selected", "is_match", "weight"} <= set(sample.columns)
    expected = set(map(frozenset, zip(result.links.left_id, result.links.right_id)))
    assert {frozenset(pair) for pair in zip(sample.left_id[sample.selected], sample.right_id[sample.selected])} \
        == expected
    sample["is_match"] = [int(any({a, b} <= group for group in SAME)) for a, b in zip(sample.left_id, sample.right_id)]
    evaluation = jlink.evaluate(sample, mode="selected", n_boot=20)
    assert evaluation.precision[0] == 1.0 and evaluation.recall[0] == 1.0
    loose = result.recluster(threshold=0.01, linkage="components").audit_sample(n=len(result.scores))
    loose["is_match"] = [int(any({a, b} <= group for group in SAME)) for a, b in zip(loose.left_id, loose.right_id)]
    assert jlink.evaluate(loose, mode="selected", n_boot=20).precision[0] < 1.0
    with pytest.raises(ValueError, match="do not handle dedupe clusters yet"):
        jlink.create_review(result)


def test_rule_style_dedupe_collapses_articles_into_events():
    articles = pd.DataFrame({
        "published": ["2024-03-02", "2024-03-03", "2024-03-09", "2024-03-03", None],
        "text": ["Man shot on Fulton St", "Victim of Fulton St shooting identified", "Fulton St vigil",
                 "Teen wounded in Mott Haven", "Undated Fulton St brief"]})
    rule = "Both news articles report the same shooting incident."
    fake = FakeJev(lambda state, asked: 0.9 if "Fulton" in state["record_a"]["text"]
                   and "Fulton" in state["record_b"]["text"] else 0.1)
    linker = jlink.Linker(style="rule", definition=rule, on=["text", "published"],
                          blockers=[block.window("published", 3, unit="days")])
    estimate = linker.estimate(articles)
    assert {k: estimate[k] for k in ("records", "pairs", "seconds")} == {"records": 5, "pairs": 3, "seconds": 0.0}
    assert 0 <= estimate["dollars"] < 0.001 and estimate["assumptions"]["tokens_per_pair"] > 270
    assert estimate["assumptions"]["token_basis"] == "sampled_records"
    result = linker.dedupe(articles, progress=False, transport=fake.transport)
    # The vigil is six days later and the brief has no date: the window never proposes them.
    assert result.clusters.cluster_id.tolist() == [0, 0, 1, 2, 3] and len(fake.bodies) == 3
    assert {body["questions"]["match"]["instructions"] for body in fake.bodies} == {
        "Record A and record B satisfy the following match rule. " + rule}
    assert result.settings["blocking"]["passes"][0]["dropped_values"]["left"]["missing"] == 1
    methods = result.methods()
    assert "that a written match rule relates to each other" in methods and "refer to the same" not in methods
    assert "need not mean that they describe the same entity" in methods and rule in methods
    assert "Question style: rule" in result.report()


def test_invalid_dedupe_arguments():
    with pytest.raises(ValueError, match="`linkage` must be one of"):
        run(linkage="single")
    with pytest.raises(ValueError, match="`threshold` is a probability"):
        run(threshold=2)
    with pytest.raises(ValueError, match="budget"):
        run(budget=-1)
    with pytest.raises(ValueError, match="duplicate IDs"):
        run(id="year")
    with pytest.raises(ValueError, match="already has a 'cluster_id' column"):
        run()[0].labeled(FIRMS.assign(cluster_id=0))


# Command line

def test_cli_round_trip_dedupe_cluster_audit_evaluate(tmp_path, monkeypatch, capsys):
    fake = FakeJev(oracle)
    linker_module = importlib.import_module("jlink.linker")
    real = linker_module.judge
    monkeypatch.setattr(linker_module, "judge",
                        lambda *args, **kwargs: real(*args, **{**kwargs, "transport": fake.transport}))
    table, clusters, scores, links, report = [tmp_path / name for name in (
        "firms.csv", "clusters.csv", "scores.csv", "links.csv", "report.md")]
    FIRMS.assign(rid=["001", "002", "003", "004", "005", "006", "007"]).to_csv(table, index=False)
    base = ["dedupe", str(table), "--on", "name", "--id", "rid", "--block", "ngrams:name:7"]
    cli([*base, "--estimate"])
    captured = capsys.readouterr()
    assert "7 records, paired with each other; " in captured.out and "No API calls" in captured.out
    assert not fake.bodies
    cli([*base, "--entity", "firm", "--no-cache", "-o", str(clusters), "--scores", str(scores),
         "--links", str(links), "--report", str(report), "--save", str(tmp_path / "run")])
    saved = jlink.load(tmp_path / "run")
    assert isinstance(saved, jlink.DedupeResult) and saved.settings["task"] == "dedupe"
    assert saved.clusters.id.tolist() == ["001", "002", "003", "004", "005", "006", "007"]
    assert saved.recluster(threshold=0.99).clusters.cluster_size.eq(1).all()
    captured = capsys.readouterr()
    assert captured.out == "" and "2 clusters of two or more records, holding 6" in captured.err
    calls = len(fake.bodies)
    assert calls == len(pd.read_csv(scores)) > 0
    saved = pd.read_csv(clusters, dtype=str)
    assert saved.columns.tolist() == ["id", "cluster_id", "cluster_size"]
    assert saved.id.tolist() == ["001", "002", "003", "004", "005", "006", "007"]  # leading zeros survive
    assert saved.cluster_id.tolist() == ["0", "0", "0", "1", "1", "2", "0"]
    assert "(dedupe)" in report.read_text()

    # Another rule from the saved scores: no calls, and records without any pair are still listed.
    again = tmp_path / "again.csv"
    cli(["cluster", str(scores), "--records", str(table), "--id", "rid", "--threshold", "0.99", "-o", str(again)])
    assert "no API calls" in capsys.readouterr().err and len(fake.bodies) == calls
    assert pd.read_csv(again).cluster_size.eq(1).all() and len(pd.read_csv(again)) == 7
    cli(["cluster", str(scores), "--records", str(table), "--id", "rid", "--unproposed", "ignore",
         "-o", str(again)])
    assert pd.read_csv(again).cluster_id.tolist() == [0, 0, 0, 1, 1, 2, 0]
    cli(["cluster", str(scores), "--linkage", "components"])
    captured = capsys.readouterr()
    assert captured.out.startswith("id,cluster_id,cluster_size") and "components linkage at 0.5" in captured.err
    pd.testing.assert_frame_equal(
        pd.read_csv(clusters), jlink.cluster(pd.read_csv(scores, dtype={"left_id": str, "right_id": str}),
                                             ids=saved.id).assign(id=lambda f: f.id.astype(int)))

    # The ordinary audit and evaluate commands accept the dedupe outputs unchanged.
    audit = tmp_path / "audit.csv"
    cli(["audit", str(scores), "--links", str(links), "--left", str(table), "--right", str(table),
         "--on", "name", "--left-id", "rid", "--right-id", "rid", "-n", "50", "-o", str(audit)])
    labeled = pd.read_csv(audit, dtype=str)
    labeled["is_match"] = [int(a[:4].casefold() == b[:4].casefold()) for a, b in zip(labeled.a_name, labeled.b_name)]
    labeled.to_csv(audit, index=False)
    cli(["evaluate", str(audit), "--mode", "selected"])
    assert "Precision 1.0000" in capsys.readouterr().out
    assert len(fake.bodies) == calls


@pytest.mark.parametrize("args,message", [
    (["--on", "name"], "--entity must name a kind of record"),
    (["--on", "name", "--style", "rule"], "--define cannot be empty"),
    (["--on", "name=alias", "--entity", "firm"], "column 'alias'"),
    (["--on", "missing", "--entity", "firm"], "column 'missing'"),
    (["--on", "name", "--entity", "firm", "--id", "year"], "duplicate IDs"),
    (["--on", "name", "--entity", "firm", "--linkage", "single"], "--linkage"),
    (["--on", "name", "--entity", "firm", "--block", "exact:nope"], "column 'nope'")])
def test_cli_dedupe_errors_are_one_line(tmp_path, capsys, args, message):
    table = tmp_path / "firms.csv"
    FIRMS.to_csv(table, index=False)
    with pytest.raises(SystemExit) as exc:
        cli(["dedupe", str(table), *args])
    captured = capsys.readouterr()
    assert exc.value.code == 2 and message in captured.err and captured.err.count("\n") == 1


def test_cli_cluster_errors(tmp_path, capsys):
    scores = tmp_path / "scores.csv"
    pd.DataFrame({"left_id": ["r1"], "right_id": ["zz"], "p": [0.9]}).to_csv(scores, index=False)
    table = tmp_path / "firms.csv"
    FIRMS.to_csv(table, index=False)
    for args, message in [(["--id", "rid"], "--id names a column of --records"),
                          (["--records", str(table), "--id", "rid"], "absent from the original"),
                          (["-o", str(scores)], "separate output file")]:
        with pytest.raises(SystemExit):
            cli(["cluster", str(scores), *args])
        assert message in capsys.readouterr().err


@pytest.mark.parametrize("record_ids", [
    [1, 2, 2**53 + 1], [0.30000000000000004, 1.0, 2.0],
    [0.3, 0.30000000000000004, 1.0], ["001", "NA", "NULL"]])
def test_saved_dedupe_preserves_ids_scores_and_reclustering_exactly(tmp_path, record_ids):
    frame = pd.DataFrame({"rid": record_ids, "name": ["alpha", "beta", "gamma"], "group": [1, 1, 1]})
    probabilities = {("alpha", "beta"): 0.99, ("alpha", "gamma"): 0.30000000000000004,
                     ("beta", "gamma"): 0.7}
    fake = FakeJev(lambda state, asked: probabilities[(state["record_a"]["name"], state["record_b"]["name"])])
    result = jlink.Linker("firm", "name", blockers=[block.exact("group")], cache=False).dedupe(
        frame, id="rid", progress=False, transport=fake.transport)
    assert result.clusters.cluster_size.tolist() == [3, 3, 3]
    directory = result.save(tmp_path / "run")
    assert json.loads((directory / "settings.json").read_text())["result_format_version"] == 2
    back = jlink.load(directory)
    pd.testing.assert_frame_equal(back.clusters, result.clusters, check_exact=True)
    columns = ["left_id", "right_id", "p", "sim", "answered_at"]
    pd.testing.assert_frame_equal(back.scores[columns], result.scores[columns], check_exact=True)
    pd.testing.assert_frame_equal(back.links[columns], result.links[columns], check_exact=True)
    pd.testing.assert_frame_equal(back.labeled(frame), result.labeled(), check_exact=True)
    for linkage, unproposed, threshold in itertools.product(
            ["average", "components"], ["ignore", "nonmatch"], [0.5, 0.99]):
        options = dict(linkage=linkage, unproposed=unproposed, threshold=threshold)
        pd.testing.assert_frame_equal(back.recluster(**options).clusters,
                                      result.recluster(**options).clusters, check_exact=True)
    assert len(fake.bodies) == 3  # saving, loading and reclustering never judge again


@pytest.mark.parametrize("record_ids", [[1, 2, 2**53 + 1, 4],
                                       [0.3, 0.30000000000000004, 1.0, 2.0],
                                       ["001", "NA", "NULL", "alone"]])
def test_cli_dedupe_save_then_cluster_preserves_exact_scores_and_ids(tmp_path, monkeypatch, record_ids):
    probabilities = {("alpha", "beta"): 0.99, ("alpha", "gamma"): 0.30000000000000004,
                     ("beta", "gamma"): 0.7}
    fake = FakeJev(lambda state, asked: probabilities[(state["record_a"]["name"], state["record_b"]["name"])])
    linker_module = importlib.import_module("jlink.linker")
    real = linker_module.judge
    monkeypatch.setattr(linker_module, "judge",
                        lambda *args, **kwargs: real(*args, **{**kwargs, "transport": fake.transport}))
    frame = pd.DataFrame({"rid": record_ids, "name": ["alpha", "beta", "gamma", "alone"],
                          "group": [1, 1, 1, 2]})
    records, directory = tmp_path / "records.parquet", tmp_path / "run"
    frame.to_parquet(records, index=False)
    captured, original = [], jlink.Linker.dedupe

    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(jlink.Linker, "dedupe", capture)
    cli(["dedupe", str(records), "--id", "rid", "--entity", "firm", "--on", "name", "--block", "exact:group",
         "--no-cache", "--save", str(directory), "-o", str(tmp_path / "first.parquet")])
    result, = captured
    back = jlink.load(directory)
    assert result.clusters.cluster_size.tolist() == [3, 3, 3, 1]
    columns = ["left_id", "right_id", "p", "sim", "answered_at"]
    pd.testing.assert_frame_equal(back.scores[columns], result.scores[columns], check_exact=True)
    pd.testing.assert_frame_equal(back.clusters, result.clusters, check_exact=True)
    assert len(fake.bodies) == 3
    for linkage in ("average", "components"):
        for policy in ("nonmatch", "ignore"):
            output, links = tmp_path / "again.parquet", tmp_path / "links.parquet"
            cli(["cluster", str(directory / "scores.csv"), "--records", str(records), "--id", "rid",
                 "--linkage", linkage, "--unproposed", policy, "-o", str(output), "--links", str(links)])
            expected = result.recluster(linkage=linkage, unproposed=policy)
            pd.testing.assert_frame_equal(pd.read_parquet(output), expected.clusters, check_exact=True)
            pd.testing.assert_frame_equal(pd.read_parquet(links)[["left_id", "right_id", "p"]],
                                          expected.links[["left_id", "right_id", "p"]], check_exact=True)
    assert len(fake.bodies) == 3


def test_cli_cluster_preserves_probability_threshold_and_missing_values(tmp_path):
    scores, output = tmp_path / "scores.csv", tmp_path / "clusters.parquet"
    pd.DataFrame({"left_id": ["001", "NA"], "right_id": ["NA", "NULL"],
                  "p": [0.30000000000000004, np.nan]}).to_csv(scores, index=False)
    cli(["cluster", str(scores), "--threshold", "0.30000000000000004", "-o", str(output)])
    assert pd.read_parquet(output).cluster_size.tolist() == [2, 2, 1]


@pytest.mark.parametrize("value", ["invalid", "inf", "1.01"])
def test_cli_cluster_rejects_invalid_probability_text(tmp_path, capsys, value):
    scores = tmp_path / "scores.csv"
    scores.write_text(f"left_id,right_id,p\na,b,{value}\n")
    with pytest.raises(SystemExit) as exc:
        cli(["cluster", str(scores)])
    assert exc.value.code == 2 and "p" in capsys.readouterr().err


@pytest.mark.parametrize("task", ["link", "dedupe"])
@pytest.mark.parametrize("collision", ["output", "scores", "report", "first", "scores.csv", "settings.json"])
def test_cli_save_path_errors_precede_model_calls(tmp_path, monkeypatch, capsys, task, collision):
    fake = FakeJev(lambda state, asked: 0.9)
    linker_module = importlib.import_module("jlink.linker")
    real = linker_module.judge
    monkeypatch.setattr(linker_module, "judge",
                        lambda *args, **kwargs: real(*args, **{**kwargs, "transport": fake.transport}))
    table = tmp_path / "records.csv"
    pd.DataFrame({"name": ["Alpha", "Beta"], "group": [1, 1]}).to_csv(table, index=False)
    directory = tmp_path / "run.csv"
    args = [task, str(table), *([str(table)] if task == "link" else []), "--entity", "firm",
            "--on", "name", "--block", "exact:group", "--no-cache", "--save", str(directory)]
    if collision in ("output", "scores", "report"):
        flag = {"output": "-o", "scores": "--scores", "report": "--report"}[collision]
        args += [flag, str(directory)]
    else:
        member = ("links.csv" if task == "link" else "clusters.csv") if collision == "first" else collision
        (directory / member).mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        cli(args)
    assert exc.value.code == 2
    assert len(fake.bodies) == 0
    assert "--save" in capsys.readouterr().err
    if collision in ("output", "scores", "report"):
        assert not directory.exists()

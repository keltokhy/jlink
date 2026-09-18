"""Review snapshots, amendment chains, conflicts and offline constrained assignment."""
import copy
import json

import pandas as pd
import pytest

import jlink
from jlink.review import Review, _source_id


@pytest.fixture
def result():
    left = pd.DataFrame({"id": ["001", "002", "NA", "004", "005"],
                         "name": ["Acme", "Acme Labs", "Unjudged", "Low probability", "No candidate"]})
    right = pd.DataFrame({"id": ["A", "B", "C", "D", "NULL"], "name": list("abcde")})
    scores = pd.DataFrame({"left_id": ["001", "001", "002", "002", "NA", "004"],
                           "right_id": ["A", "B", "A", "B", "C", "D"],
                           "p": [.82, .79, .81, .2, float("nan"), .15],
                           "sim": [.9, .8, .9, .3, .5, .6], "block": "fake", "error": None,
                           "source": ["jev"] * 4 + ["unjudged", "jev"]})
    settings = dict(how="one-to-one", threshold=.5, min_margin=None, left_id="id", right_id="id",
                    on=[["name", "name"]], question="Same firm?", model="offline fake")
    return jlink.Result(jlink.resolve(scores), scores, settings, _left=left, _right=right)


def pairs(frame):
    return set(zip(frame.left_id, frame.right_id))


def test_round_trip_preserves_ids_original_scores_links_and_all_records(result, tmp_path):
    original_scores, original_links = result.scores.copy(deep=True), result.links.copy(deep=True)
    review = jlink.create_review(result)
    result.scores.loc[0, "p"] = .01  # A snapshot is independent of the caller's frames.
    loaded = jlink.read_review(review.save(tmp_path / "review.json"))
    applied = loaded.apply()
    assert pairs(applied.links) == {("001", "B"), ("002", "A")}
    pd.testing.assert_frame_equal(applied.scores, original_scores)
    pd.testing.assert_frame_equal(applied.original_links, original_links)
    assert loaded.run_id == review.run_id
    records = loaded.to_dict()["source"]["records"]
    assert {r["id"]["value"] for r in records["left"]} == {"001", "002", "NA", "004", "005"}
    assert records["right"][-1]["id"]["value"] == "NULL"
    assert applied.links.original_selected.all()


def test_actual_link_membership_not_probability_threshold(result):
    result.scores["selected"] = True  # A stale annotation must never decide membership.
    review = jlink.create_review(result)
    original = review.to_dict()["source"]["links"]["rows"]
    assert {(r["left_id"]["value"], r["right_id"]["value"]) for r in original} == {("001", "B"), ("002", "A")}
    review.decide("001", "A", "accept", reviewer="Alice")
    link = review.apply().links.iloc[0]
    assert not link.original_selected and link.p == .82


@pytest.mark.parametrize("values", [["001", "NA", "NULL"], [2**53 + 1, 2**53 + 3, 2**53 + 7],
                                   [1.25, 2.5, 3.75]])
def test_typed_ids_survive_browser_compatible_json(result, values, tmp_path):
    mapping = dict(zip(["001", "002", "NA"], values))
    result._left = result._left.iloc[:3].copy()
    result._left["id"] = values
    result.scores = result.scores.loc[result.scores.left_id.isin(mapping)].copy()
    result.scores["left_id"] = result.scores.left_id.map(mapping)
    result.links = jlink.resolve(result.scores)
    review = jlink.create_review(result).decide(values[2], "C", "accept", reviewer="R")
    back = jlink.read_review(review.save(tmp_path / "review.json"))
    accepted = back.apply().links.query("selection_basis == 'manual_accept'").iloc[0]
    assert accepted.left_id == values[2]
    assert type(accepted.left_id) is type(values[2])
    assert pd.isna(accepted.p)
    assert back.history[0]["left_id"]["value"] == str(values[2])


def test_accept_reject_unsure_history_is_explicit_and_releases_constraint(result):
    review = jlink.create_review(result)
    for decision, reviewer in [("accept", "Alice"), ("reject", "Bob"), ("unsure", "Alice")]:
        review.decide("001", "A", decision, reviewer=reviewer, note=f"amended to {decision}")
    events = review.history
    assert [e["decision"] for e in events] == ["accept", "reject", "unsure"]
    assert events[0]["previous_event"] is None
    assert events[2]["previous_event"] == events[1]["event_id"]
    assert all(e["timestamp"].endswith("Z") for e in events)
    assert pairs(review.apply().links) == pairs(result.links)
    events.clear()
    assert len(review.history) == 3  # Readers cannot erase the durable history.


def test_manual_accept_overrides_threshold_and_margin_without_fabricating_p(result):
    review = jlink.create_review(result).decide("004", "D", "accept", reviewer="R")
    review.decide("NA", "C", "accept", reviewer="R")
    applied = review.apply(threshold=.99, min_margin=.99)
    assert pairs(applied.links) == {("004", "D"), ("NA", "C")}
    assert applied.links.selection_basis.eq("manual_accept").all()
    assert applied.links.loc[applied.links.left_id.eq("004"), "p"].item() == .15
    assert pd.isna(applied.links.loc[applied.links.left_id.eq("NA"), "p"].item())
    pd.testing.assert_frame_equal(applied.scores, result.scores)
    pd.testing.assert_frame_equal(applied.original_links, result.links)
    assert "override" in applied.settings["manual_policy"]


def test_reject_recomputes_assignment_and_margin(result):
    review = jlink.create_review(result).decide("001", "B", "reject", reviewer="R")
    review.decide("002", "A", "reject", reviewer="R")
    applied = review.apply(min_margin=.5)
    assert pairs(applied.links) == {("001", "A")}
    assert pd.isna(applied.links.iloc[0].margin)  # Rejected competitors no longer suppress it.
    assert not applied.links.iloc[0].original_selected
    assert review.apply(min_margin=None).settings["min_margin"] is None


@pytest.mark.parametrize("how,accepts,conflicts", [
    ("one-to-one", [("001", "A"), ("001", "B")], True),
    ("one-to-one", [("001", "A"), ("002", "A")], True),
    ("many-to-one", [("001", "A"), ("001", "B")], True),
    ("many-to-one", [("001", "A"), ("002", "A")], False),
    ("one-to-many", [("001", "A"), ("002", "A")], True),
    ("one-to-many", [("001", "A"), ("001", "B")], False),
    ("many-to-many", [("001", "A"), ("001", "B"), ("002", "A")], False),
])
def test_conflicting_acceptances_by_cardinality(result, how, accepts, conflicts):
    review = jlink.create_review(result)
    for a, b in accepts:
        review.decide(a, b, "accept", reviewer="R")
    if conflicts:
        with pytest.raises(ValueError, match="conflicting manual acceptances"):
            review.apply(how=how)
    else:
        links = review.apply(how=how).links
        assert set(accepts) <= pairs(links)
        if how in ("one-to-one", "many-to-one"):
            assert links.left_id.is_unique
        if how in ("one-to-one", "one-to-many"):
            assert links.right_id.is_unique


@pytest.mark.parametrize("how", ["one-to-one", "many-to-one", "one-to-many", "many-to-many"])
def test_unreviewed_application_equals_existing_resolver(result, how):
    applied = jlink.create_review(result).apply(how=how, threshold=.1, min_margin=-.5)
    expected = jlink.resolve(result.scores, how=how, threshold=.1, min_margin=-.5)
    pd.testing.assert_frame_equal(applied.links[expected.columns].reset_index(drop=True), expected)


def test_history_import_merges_independent_pairs_and_extensions(result):
    original = jlink.create_review(result).decide("001", "A", "accept", reviewer="A")
    other = Review(original.to_dict()).decide("001", "A", "unsure", reviewer="B")
    original.decide("004", "D", "reject", reviewer="A")
    original.merge(other).merge(other)
    assert len(original.history) == 3
    assert pairs(original.apply().links) == pairs(result.links)


def test_import_refuses_branches_changed_events_and_other_runs_atomically(result):
    base = jlink.create_review(result).decide("001", "A", "accept", reviewer="A")
    branch = Review(base.to_dict()).decide("001", "A", "reject", reviewer="B")
    base.decide("001", "A", "unsure", reviewer="A")
    before = base.to_dict()
    with pytest.raises(ValueError, match="conflicting amendments"):
        base.merge(branch)
    assert before == base.to_dict()
    modified = copy.deepcopy(before)
    modified["history"][0]["note"] = "rewritten"
    with pytest.raises(ValueError, match="existing review event"):
        base.merge(Review(modified))
    result.settings["threshold"] = .8
    with pytest.raises(ValueError, match="different source run"):
        base.merge(jlink.create_review(result))


@pytest.mark.parametrize("edit,error", [
    (lambda d: d["source"]["scores"]["rows"][0].update(p=.01), "identity"),
    (lambda d: d.update(version=99), "version"),
    (lambda d: d["history"][0].update(reviewer=" "), "reviewer"),
    (lambda d: d["history"][0].update(timestamp="2026-09-18"), "timezone"),
    (lambda d: d["history"][0].update(decision="yes"), "decision"),
    (lambda d: d["history"][0].update(previous_event="missing"), "predecessor"),
    (lambda d: d["history"][0].update(left_id={"type": "str", "value": "absent"}), "absent"),
    (lambda d: d["history"].append(copy.deepcopy(d["history"][0])), "duplicate"),
])
def test_artifact_validation(result, edit, error):
    document = jlink.create_review(result).decide("001", "A", "accept", reviewer="A").to_dict()
    edit(document)
    with pytest.raises(ValueError, match=error):
        Review(document)


@pytest.mark.parametrize("bad", [None, float("nan"), True, (1, 2)])
def test_unsupported_or_missing_record_ids_rejected(result, bad):
    result._left["id"] = result._left.id.astype(object)
    result._left.at[4, "id"] = bad
    with pytest.raises(ValueError, match="ID"):
        jlink.create_review(result)


def test_missing_duplicate_records_and_unknown_links_fail(result):
    with pytest.raises(ValueError, match="duplicate IDs"):
        jlink.create_review(result, left=pd.concat([result._left, result._left]))
    with pytest.raises(ValueError, match="absent"):
        jlink.create_review(result, left=result._left.iloc[1:])
    result.links.loc[0, "right_id"] = "NOT A CANDIDATE"
    with pytest.raises(ValueError, match="absent"):
        jlink.create_review(result)


@pytest.mark.parametrize("kwargs", [{"how": "bad"}, {"threshold": float("nan")}, {"threshold": 2},
                                    {"min_margin": float("inf")}, {"min_margin": -2}])
def test_invalid_resolution_settings(result, kwargs):
    with pytest.raises(ValueError):
        jlink.create_review(result).apply(**kwargs)


def test_empty_candidates_still_review_all_records(result, tmp_path):
    result.scores = result.scores.iloc[:0]
    result.links = result.links.iloc[:0]
    review = jlink.create_review(result)
    assert len(review.to_dict()["source"]["records"]["left"]) == 5
    assert review.apply().links.empty
    assert "No candidate was proposed" in review.write_html(tmp_path / "review.html").read_text()


def test_no_api_calls(result, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("review must not call a model or make a network request")
    import httpx
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    review = jlink.create_review(result).decide("001", "B", "reject", reviewer="R")
    assert review.apply().links.shape[0] == 1


def test_script_and_csv_injection_are_inert_and_json_ids_remain_exact(result, tmp_path):
    payload = '</script><script>window.PWNED=1</script>&\u2028'
    result._left.loc[0, "name"] = payload
    result.settings["question"] = payload
    review = jlink.create_review(result).decide("001", "A", "accept", reviewer=payload, note=payload)
    html = review.write_html(tmp_path / "review.html").read_text()
    assert payload not in html
    assert "\\u003c/script\\u003e" in html
    assert "innerHTML" not in html
    assert "connect-src 'none'" in html
    result._left.loc[0, "id"] = "=1+2"
    result.scores.loc[result.scores.left_id.eq("001"), "left_id"] = "=1+2"
    result.links.loc[result.links.left_id.eq("001"), "left_id"] = "=1+2"
    applied = jlink.create_review(result).apply()
    applied.links["\t=HEADER"] = " \t@SUM(1,1)"
    csv = applied.write_csv(tmp_path / "links.csv").read_text()
    assert "'=1+2" in csv and "' \t@SUM" in csv and "'\t=HEADER" in csv
    saved = json.loads(applied.save(tmp_path / "links.json").read_text())
    assert "=1+2" in {r["left_id"]["value"] for r in saved["links"]["rows"]}
    assert saved["review"]["source"]["links"]  # original links travel with the applied output


def test_source_identity_changes_for_selection_settings_or_record_changes(result):
    run_id = jlink.create_review(result).run_id
    result.links = result.links.iloc[:1]
    assert jlink.create_review(result).run_id != run_id
    next_id = jlink.create_review(result).run_id
    result._left.loc[0, "name"] = "Amended original record"
    assert jlink.create_review(result).run_id != next_id


def test_mixed_numeric_id_types_never_round_large_integers(result):
    source_ids = pd.Series([2**53 + 1, 1.25, 2**53 + 3], dtype=object)
    result._left = pd.DataFrame({"id": source_ids, "name": ["A", "B", "C"]})
    result.scores = result.scores.iloc[:3].copy()
    result.scores["left_id"] = source_ids
    result.links = jlink.resolve(result.scores)
    review = jlink.create_review(result)
    decoded = review.apply().scores.left_id.tolist()
    assert decoded == source_ids.tolist()
    assert [type(value) for value in decoded] == [int, float, int]


def test_boolean_selection_projection_is_explicit_and_independent(result):
    review = jlink.create_review(result)
    assert review.candidates.selected.tolist() == [False, True, True, False, False, False]
    review.decide("NA", "C", "accept", reviewer="R")
    applied = review.apply()
    assert applied.selected_scores().selected.tolist() == [False, True, True, False, True, False]
    assert applied.selected_scores().selected.dtype == bool
    assert "selected" not in applied.scores


def test_malformed_source_is_rejected_even_with_a_recomputed_hash(result):
    document = jlink.create_review(result).to_dict()
    document["source"]["settings"]["on"] = "not a field mapping"
    document["run_id"] = _source_id(document["source"])
    with pytest.raises(ValueError, match="on fields"):
        Review(document)


def test_integral_floats_in_json_have_a_browser_stable_identity(result):
    result.scores.loc[0, ["p", "sim"]] = 1.0
    result._left["count"] = 1.0
    document = jlink.create_review(result).to_dict()
    assert type(document["source"]["scores"]["rows"][0]["p"]) is int
    assert Review(json.loads(json.dumps(document))).run_id == document["run_id"]


def test_missing_records_without_candidates_are_not_silently_dropped(result):
    result.settings['n_left'] = len(result._left)
    with pytest.raises(ValueError, match="all original left records"):
        jlink.create_review(result, left=result._left.iloc[:-1])


@pytest.mark.parametrize('column,value', [('p', float('inf')), ('p', -0.1), ('sim', float('inf'))])
def test_invalid_scores_are_rejected_before_json_conversion(result, column, value):
    result.scores.loc[0, column] = value
    with pytest.raises(ValueError, match="finite|probabilities"):
        jlink.create_review(result)

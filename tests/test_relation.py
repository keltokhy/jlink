"""Rule-style questions and one-sided fields, end to end against a fake Jev."""

import importlib
import json

import pandas as pd
import pytest

import jlink
from fakes import FakeJev
from jlink.fields import parse_on
from jlink.judge import question

ARTICLES = pd.DataFrame({
    "article_id": ["a1", "a2", "a3"],
    "published": ["2024-03-02", "2024-03-02", "2024-03-09"],
    "text": ["Man, 31, shot on Fulton St in Bedford-Stuyvesant", "Teen wounded in Mott Haven shooting",
             None]})
INCIDENTS = pd.DataFrame({
    "incident_id": [501, 502, 503],
    "occurred": ["2024-03-02", "2024-03-02", "2024-03-09"],
    "neighborhood": ["Bedford-Stuyvesant", "Mott Haven", "Jamaica"],
    "victim_age_group": ["25-44", "<18", None]})
RULE = "Record A is a news article that reports the shooting incident in record B."
ON = [("published", "occurred"), ("text", None), (None, "neighborhood"), (None, "victim_age_group")]
REPORTED = {("a1", 501), ("a2", 502)}


def oracle(state, asked):
    """Higher under the rule question, so a collision between styles would be visible in p."""
    text, place = state["record_a"].get("text", ""), state["record_b"].get("neighborhood", "")
    hit = bool(place) and place in text
    rule = asked["instructions"].startswith("Record A and record B satisfy the following match rule.")
    return (0.9 if rule else 0.7) if hit else (0.1 if rule else 0.2)


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(name, raising=False)


def linker(style="rule", **kwargs):
    entity = None if style == "rule" else "event"
    return jlink.Linker(entity, ON, RULE, blockers=[jlink.block.exact(("published", "occurred"))],
                        style=style, **kwargs)


def run(fake=None, style="rule", **kwargs):
    fake = fake or FakeJev(oracle)
    result = linker(style).link(ARTICLES, INCIDENTS, left_id="article_id", right_id="incident_id",
                                how="many-to-one", progress=False, transport=fake.transport, **kwargs)
    return result, fake


def test_rule_question_and_one_sided_state_reach_jev():
    result, fake = run()
    asked = {body["questions"]["match"]["instructions"] for body in fake.bodies}
    assert asked == {"Record A and record B satisfy the following match rule. " + RULE}
    assert "refer to the same" not in asked.pop()
    states = {(s["record_a"].get("text"), s["record_b"].get("neighborhood")): s
              for s in (body["state"] for body in fake.bodies)}
    state = states[("Man, 31, shot on Fulton St in Bedford-Stuyvesant", "Bedford-Stuyvesant")]
    # Paired fields take the left column's label on both sides; one-sided fields keep their own.
    assert state == {"record_a": {"published": "2024-03-02", "text": ARTICLES.text[0]},
                     "record_b": {"published": "2024-03-02", "neighborhood": "Bedford-Stuyvesant",
                                  "victim_age_group": "25-44"}}
    # Missing one-sided values are dropped from the record, like any other missing field.
    assert states[(None, "Jamaica")] == {"record_a": {"published": "2024-03-09"},
                                         "record_b": {"published": "2024-03-09", "neighborhood": "Jamaica"}}
    assert set(zip(result.links.left_id, result.links.right_id)) == REPORTED


def test_identity_and_rule_answers_never_share_a_cache_entry():
    first, fake = run(style="identity")
    pairs = len(first.scores)
    assert len(fake.bodies) == pairs and first.meter.cached == 0
    second, fake = run(style="rule")
    assert len(fake.bodies) == pairs and second.meter.cached == 0  # same records, new proposition
    assert set(first.scores.p) == {0.7, 0.2} and set(second.scores.p) == {0.9, 0.1}
    for style, expected in (("identity", first), ("rule", second)):
        again, fake = run(style=style, budget=0)
        assert not fake.bodies and again.meter.cached == pairs
        pd.testing.assert_series_equal(again.scores.p, expected.scores.p)
    assert question("event", RULE)["instructions"] != question("", RULE, style="rule")["instructions"]


@pytest.mark.parametrize("style", ["identity", "rule"])
def test_saved_and_quoted_question_is_the_question_that_was_sent(style, tmp_path, monkeypatch):
    result, fake = run(style=style)
    sent = {body["questions"]["match"]["instructions"] for body in fake.bodies}
    assert sent == {result.settings["question"]} == {result.scores.attrs["question"]}
    (asked,) = sent
    assert f'"{asked}"' in result.methods() and f"Rule: {asked}" in result.report()
    assert jlink.load(result.save(tmp_path / "run")).settings["question"] == asked
    deduped = linker(style).dedupe  # the same linker's question, on one table
    with pytest.raises(ValueError, match="one side only"):
        deduped(ARTICLES)
    same = FakeJev(oracle)
    clusters = jlink.dedupe(ARTICLES, entity="article", on="published", definition=RULE, style=style,
                            blockers=[jlink.block.exact("published")], progress=False,
                            transport=same.transport)
    assert {body["questions"]["match"]["instructions"] for body in same.bodies} == {
        clusters.settings["question"]}
    assert f'"{clusters.settings["question"]}"' in clusters.methods()
    # Settings follow what judge() sent even if the two ever diverge: there is one builder.
    module = importlib.import_module("jlink.linker")
    real = module.judge

    def reworded(*args, **kwargs):
        scores, meter = real(*args, **kwargs)
        scores.attrs["question"] = "A sentence only this judge sent."
        return scores, meter

    monkeypatch.setattr(module, "judge", reworded)
    assert run(style=style)[0].settings["question"] == "A sentence only this judge sent."


def test_settings_report_and_methods_describe_a_relation(tmp_path):
    result, _ = run()
    s = result.settings
    assert s["style"] == "rule" and s["entity"] is None
    assert s["question"] == "Record A and record B satisfy the following match rule. " + RULE
    assert s["on"] == [["published", "occurred"], ["text", None], [None, "neighborhood"],
                       [None, "victim_age_group"]]
    assert s["inputs"]["left"]["compared"]["columns"] == ["published", "text"]
    assert s["inputs"]["right"]["compared"]["columns"] == ["occurred", "neighborhood", "victim_age_group"]
    json.dumps(s, allow_nan=False)
    report, methods = result.report(), result.methods()
    assert "Rule: Record A and record B satisfy the following match rule." in report
    assert "Question style: rule" in report
    assert "published = occurred, text (left only), neighborhood (right only)" in report
    assert "relation between two records" in methods and "pair satisfies the rule" in methods
    assert "refer to the same" not in methods and "same entity. Candidate" in methods
    assert "text (first source only)" in methods and "neighborhood (second source only)" in methods
    assert "the listed fields of each record" in methods and RULE in methods
    back = jlink.load(result.save(tmp_path / "run"))
    assert back.settings == s and back.methods() == methods
    strict = back.relink(threshold=0.95)
    assert strict.settings["style"] == "rule" and strict.links.empty
    assert "relation between two records" in strict.methods()


def test_identity_methods_and_old_saved_runs_are_unchanged(tmp_path):
    result, _ = run(style="identity")
    assert "style" not in result.settings and result.settings["entity"] == "event"
    assert "Question style" not in result.report()
    assert "returns a probability that the following statement is true" in result.methods()
    assert "relation between" not in result.methods()
    # Runs saved before `style` existed carry no such setting and read as identity.
    directory = result.save(tmp_path / "old")
    settings = json.loads((directory / "settings.json").read_text())
    settings["style"] = "identity"  # explicitly saved identity styles also remain readable
    (directory / "settings.json").write_text(json.dumps(settings))
    assert jlink.load(directory).methods() == result.methods()


def test_audit_evaluate_and_review_accept_one_sided_fields():
    result, _ = run()
    sample = result.audit_sample(n=len(result.scores))
    assert {"a_published", "b_published", "a_text", "b_neighborhood", "b_victim_age_group",
            "selected"} <= set(sample.columns)
    assert "b_text" not in sample and "a_neighborhood" not in sample
    sample["is_match"] = [int(pair in REPORTED) for pair in zip(sample.left_id, sample.right_id)]
    evaluation = jlink.evaluate(sample, mode="selected", n_boot=20)
    assert evaluation.precision[0] == 1.0 and evaluation.recall[0] == 1.0
    review = jlink.create_review(result)
    assert review.to_dict()["source"]["settings"]["on"][1] == ["text", None]
    review.decide("a1", 501, "reject", reviewer="test")
    assert set(zip(*[review.apply().links[c] for c in ("left_id", "right_id")])) == {("a2", 502)}


def test_estimate_and_one_line_link_accept_rule_style():
    estimate = linker().estimate(ARTICLES, INCIDENTS, left_id="article_id", right_id="incident_id")
    assert estimate["pairs"] == 5
    fake = FakeJev(oracle)
    result = jlink.link(ARTICLES, INCIDENTS, on=ON, definition=RULE, style="rule", how="many-to-one",
                        blockers=[jlink.block.exact(("published", "occurred"))], left_id="article_id",
                        right_id="incident_id", progress=False, transport=fake.transport)
    assert result.settings["style"] == "rule" and len(result.links) == 2


def test_one_sided_text_enters_each_side_of_sim():
    on = [("text", None), (None, "neighborhood")]
    passes = [jlink.block.exact(("published", "occurred"))]
    table = jlink.block.candidates(ARTICLES, INCIDENTS, on=on, blockers=passes,
                                   left_id="article_id", right_id="incident_id").set_index(
                                       ["left_id", "right_id"])
    # The article that names the neighborhood ranks first, so a budget is spent on it first.
    assert table.sim[("a1", 501)] > 0.5 > table.sim[("a1", 502)]
    assert table.sim[("a2", 502)] > 0.5 > table.sim[("a2", 501)]
    assert table.sim[("a3", 503)] == 0  # a3 has no text


def test_question_requirements_are_stated():
    with pytest.raises(ValueError, match="definition cannot be empty"):
        jlink.Linker(on=ON, style="rule")
    with pytest.raises(ValueError, match="`entity` says what a record is"):
        jlink.Linker(on="name", definition=RULE)
    with pytest.raises(ValueError, match="`style` must be one of identity, rule"):
        jlink.Linker("firm", "name", style="relation")
    with pytest.raises(ValueError, match="at least one field"):
        jlink.Linker(definition=RULE, style="rule")
    candidates = pd.DataFrame({"left_id": ["a1"], "right_id": [501], "sim": [0.0], "block": ["given"]})
    with pytest.raises(ValueError, match="definition cannot be empty"):
        jlink.judge(candidates, ARTICLES, INCIDENTS, on=ON, style="rule", left_id="article_id",
                    right_id="incident_id")


def test_one_sided_fields_are_refused_where_columns_must_pair():
    for factory in (jlink.block.exact, jlink.block.ngrams, jlink.block.initials):
        with pytest.raises(ValueError, match=r"compares a left column with a right column.*\('text', None\)"):
            factory(("text", None))
    with pytest.raises(ValueError, match="within.*one-sided"):
        jlink.block.within(jlink.block.exact("published"), (None, "neighborhood"))
    with pytest.raises(ValueError, match="one side only"):
        parse_on([("text", None)])
    with pytest.raises(ValueError, match="default n-gram pass needs an `on` field that both sides have"):
        jlink.block.candidates(ARTICLES, INCIDENTS, on=[("text", None), (None, "neighborhood")])
    with pytest.raises(ValueError, match="`exact_shortcut`.*both sides"):
        linker(exact_shortcut=True)
    with pytest.raises(ValueError, match="right records no field"):
        parse_on([("text", None)], unpaired=True)
    with pytest.raises(ValueError, match="two fields labeled 'published'"):
        parse_on([("published", "occurred"), (None, "published")], unpaired=True)
    for bad in [(None, None), ("a", "b", "c"), ("a", 1), 7]:
        with pytest.raises(ValueError, match="each `on` item"):
            parse_on([bad], unpaired=True)
    assert parse_on([["text", None], [None, "place"], "date"], unpaired=True) == [
        ("text", "text", None), ("place", None, "place"), ("date", "date", "date")]
    with pytest.raises(ValueError, match="left data has no column 'headline'"):
        jlink.Linker(on=[("headline", None), (None, "neighborhood")], definition=RULE, style="rule",
                     blockers=[jlink.block.exact(("published", "occurred"))]).link(ARTICLES, INCIDENTS)


def test_cli_rule_style_reaches_the_question_and_saved_report(tmp_path, monkeypatch, capsys):
    fake = FakeJev(oracle)
    linker_module = importlib.import_module("jlink.linker")
    real = linker_module.judge
    monkeypatch.setattr(linker_module, "judge",
                        lambda *args, **kwargs: real(*args, **{**kwargs, "transport": fake.transport}))
    left, right = tmp_path / "articles.csv", tmp_path / "incidents.csv"
    ARTICLES.to_csv(left, index=False)
    INCIDENTS.to_csv(right, index=False)
    importlib.import_module("jlink.cli").cli([
        "link", str(left), str(right), "--style", "rule", "--define", RULE, "--on", "published=occurred",
        "--on", "text=", "--on", "=neighborhood", "--left-id", "article_id", "--right-id", "incident_id",
        "--block", "exact:published=occurred", "--how", "many-to-one", "--no-cache",
        "-o", str(tmp_path / "links.csv"), "--report", str(tmp_path / "report.md")])
    assert "2 links" in capsys.readouterr().err
    assert {body["questions"]["match"]["instructions"] for body in fake.bodies} == {
        "Record A and record B satisfy the following match rule. " + RULE}
    assert all("victim_age_group" not in body["state"]["record_b"] for body in fake.bodies)
    assert any(body["state"]["record_a"].get("text") for body in fake.bodies)
    links = pd.read_csv(tmp_path / "links.csv", dtype=str)
    assert set(zip(links.left_id, links.right_id)) == {("a1", "501"), ("a2", "502")}
    assert "Question style: rule" in (tmp_path / "report.md").read_text()

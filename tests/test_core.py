"""Offline compatibility and provenance checks for the shared Jev client interfaces."""

import asyncio
import json
import sqlite3

import httpx
import pytest

from fakes import FakeJev
from jlink.core import Cache, Jev, JevBudgetExceeded

QUESTION = {"match": {"type": "noul", "instructions": "Same firm?"}}
STATE = {"name": "Acme"}


def test_cache_reads_legacy_schema_and_keeps_answer_interface(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE answers (key TEXT PRIMARY KEY, answer TEXT NOT NULL, at REAL NOT NULL) WITHOUT ROWID")
        db.execute("INSERT INTO answers VALUES (?, ?, ?)", ("old", json.dumps({"noul": 0.7}), 1.0))
    cache = Cache(path)
    assert cache.get("old") == {"noul": 0.7}
    assert cache.get_entry("old") == ({"noul": 0.7}, {})
    cache.put("new", {"noul": 0.8}, metadata={"resolved_model": "jev-1"})
    assert cache.get("new") == {"noul": 0.8}
    assert cache.db.execute("SELECT answer FROM answers WHERE key='new'").fetchone()[0] == '{"noul": 0.8}'
    cache.db.close()


def test_old_writers_cannot_leave_stale_model_metadata(tmp_path):
    cache = Cache(tmp_path / "answers.sqlite")
    cache.put("key", {"noul": 0.8}, metadata={"resolved_model": "jev-1"})
    cache.db.execute("UPDATE answers SET at=at+1, answer=? WHERE key='key'", (json.dumps({"noul": 0.9}),))
    assert cache.get_entry("key") == ({"noul": 0.9}, {})
    cache.put("key", {"noul": 0.1})
    assert cache.get_entry("key") == ({"noul": 0.1}, {})
    cache.db.close()


def test_cache_answer_and_metadata_replace_atomically(tmp_path):
    cache = Cache(tmp_path / "answers.sqlite")
    cache.put("key", {"noul": 0.8}, metadata={"resolved_model": "jev-1"})
    # A failed metadata write must not leave a newer answer paired with older/absent provenance.
    cache.db.execute("CREATE TRIGGER reject_metadata BEFORE INSERT ON answer_metadata "
                     "BEGIN SELECT RAISE(ABORT, 'metadata rejected'); END")
    with pytest.raises(sqlite3.IntegrityError, match="metadata rejected"):
        cache.put("key", {"noul": 0.9}, metadata={"resolved_model": "jev-2"})
    assert cache.get_entry("key") == ({"noul": 0.8}, {"resolved_model": "jev-1"})
    cache.db.close()


def test_legacy_cache_hit_does_not_invent_actual_model_or_provider(tmp_path):
    cache = Cache(tmp_path / "answers.sqlite")
    cache.put(Cache.key("requested-alias", STATE, QUESTION["match"]), {"noul": 0.7})

    async def check():
        jev = Jev("", model="requested-alias", cache=cache, transport=FakeJev().transport)
        provenance = {}
        answer = await jev.ask(STATE, QUESTION, allow_paid=False, provenance=provenance)
        assert answer == {"match": {"noul": 0.7}}
        assert provenance == {"match": {"source": "cache"}}
        assert jev.meter.model == "" and jev.meter.unknown_model_answers == 1
        assert jev.meter.answer_provenance[0]["provider"] is None
        await jev.close()

    asyncio.run(check())
    cache.db.close()


def test_no_reported_model_stays_unknown_even_with_an_explicit_request(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"answers": {"match": {"noul": 0.8}}, "usage": {"input_tokens": 100}})

    async def check():
        jev = Jev("test", model="pinned-request", transport=httpx.MockTransport(handler))
        await jev.ask(STATE, QUESTION)
        assert jev.meter.model == "" and jev.meter.unknown_model_answers == 1
        assert jev.meter.requested_model == "pinned-request"
        assert jev.meter.cost_sources == {"estimated_from_tokens": 1}
        await jev.close()

    asyncio.run(check())


def test_cache_only_request_miss_never_calls_the_transport():
    fake = FakeJev()

    async def check():
        jev = Jev("", transport=fake.transport)
        with pytest.raises(JevBudgetExceeded):
            await jev.ask(STATE, QUESTION, allow_paid=False)
        assert not fake.bodies
        await jev.close()

    asyncio.run(check())


def test_cache_only_can_share_existing_flight_and_keeps_original_ask_shape():
    fake = FakeJev()

    async def check():
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            entered.set()
            await release.wait()
            return fake(request)

        jev = Jev("test", model="jev-1", transport=httpx.MockTransport(handler))
        first = asyncio.create_task(jev.ask(STATE, QUESTION))
        await entered.wait()
        provenance = {}
        second = asyncio.create_task(jev.ask(STATE, QUESTION, allow_paid=False, provenance=provenance))
        await asyncio.sleep(0)
        release.set()
        assert (await first) == (await second) == {"match": {"type": "noul", "noul": 0.5}}
        assert len(fake.bodies) == 1 and jev.meter.cached == 1
        assert provenance["match"]["source"] == "shared"
        assert jev.meter.model == "jev-1"
        await jev.close()

    asyncio.run(check())


def test_cache_provenance_retains_origin_provider_across_backends(tmp_path):
    cache = Cache(tmp_path / "answers.sqlite")

    async def check():
        first = Jev("test", "typesafe", model="shared-name", cache=cache, transport=FakeJev().transport)
        await first.ask(STATE, QUESTION)
        await first.close()
        second = Jev("", "openrouter", model="shared-name", cache=cache, transport=FakeJev().transport)
        provenance = {}
        await second.ask(STATE, QUESTION, allow_paid=False, provenance=provenance)
        assert second.meter.provider == "openrouter"  # current backend selection
        assert provenance["match"]["provider"] == "typesafe"  # actual answer origin
        assert second.meter.calls == 0
        await second.close()

    asyncio.run(check())
    cache.db.close()


def test_multi_question_client_keeps_cached_and_fresh_answers_separate(tmp_path):
    cache = Cache(tmp_path / "answers.sqlite")
    cache.put(Cache.key("jev-1", STATE, QUESTION["match"]), {"noul": 0.7})
    questions = QUESTION | {"other": {"type": "noul", "instructions": "Same state?"}}
    fake = FakeJev()

    async def check():
        jev = Jev("test", model="jev-1", cache=cache, transport=fake.transport)
        provenance = {}
        answer = await jev.ask(STATE, questions, provenance=provenance)
        assert answer == {"match": {"noul": 0.7}, "other": {"type": "noul", "noul": 0.5}}
        assert list(fake.bodies[0]["questions"]) == ["other"]
        assert provenance["match"] == {"source": "cache"}
        assert provenance["other"]["resolved_model"] == "jev-1"
        assert jev.meter.resolved_models == ["jev-1"] and jev.meter.unknown_model_answers == 1
        assert jev.meter.model == ""
        assert await jev.ask(STATE, questions, allow_paid=False) == answer
        assert len(fake.bodies) == 1
        await jev.close()

    asyncio.run(check())
    cache.db.close()

"""Offline checks of jlink's use of the shared client: the store, provenance, and cache-only budgets."""

import asyncio

import httpx
import pytest

from fakes import FakeJev
from jevkit_runtime import AnswerStore, Backend, Client, JevBudgetExceeded
from jlink.core import PROVIDERS

QUESTION = {"match": {"type": "noul", "instructions": "Same firm?"}}
STATE = {"name": "Acme"}


def backend(name="openrouter", model="jev-alias", key="test"):
    return Backend(name, PROVIDERS[name].url, model, key=key)


def test_store_keeps_each_answer_with_its_origin(tmp_path):
    cache = AnswerStore(tmp_path / "answers.sqlite")
    cache.put("new", {"noul": 0.8}, {"resolved_model": "jev-1"})
    assert cache.get("new") == {"noul": 0.8}
    assert cache.entry("new").metadata == {"resolved_model": "jev-1"}
    cache.put("new", {"noul": 0.1})
    assert cache.entry("new").metadata is None
    cache.close()


def test_unattributed_cache_hit_does_not_invent_a_model_or_provider(tmp_path):
    cache = AnswerStore(tmp_path / "answers.sqlite")
    jev = Client(backend(key=""), store=cache, transport=FakeJev().transport)
    cache.put(jev.key(STATE, QUESTION["match"]), {"noul": 0.7})

    async def check():
        answer = await jev.ask(STATE, QUESTION, allow_paid=False)
        assert answer == {"match": {"noul": 0.7}}
        assert answer.origins == {"match": {"source": "cache"}}
        assert jev.meter.model == "" and jev.meter.unknown_model_answers == 1
        assert jev.meter.answer_provenance[0]["provider"] is None
        await jev.close()

    asyncio.run(check())
    cache.close()


def test_no_reported_model_stays_unknown_even_with_an_explicit_request():
    def handler(request):
        return httpx.Response(200, json={"answers": {"match": {"noul": 0.8}}, "usage": {"input_tokens": 100}})

    async def check():
        jev = Client(backend(model="pinned-request"), transport=httpx.MockTransport(handler))
        await jev.ask(STATE, QUESTION)
        assert jev.meter.model == "" and jev.meter.unknown_model_answers == 1
        assert jev.meter.requested_model == "pinned-request"
        assert jev.meter.cost_sources == {"estimated_from_tokens": 1}
        await jev.close()

    asyncio.run(check())


def test_cache_only_request_miss_never_calls_the_transport():
    fake = FakeJev()

    async def check():
        jev = Client(backend(key=""), transport=fake.transport)
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

        jev = Client(backend(model="jev-1"), transport=httpx.MockTransport(handler))
        first = asyncio.create_task(jev.ask(STATE, QUESTION))
        await entered.wait()
        second = asyncio.create_task(jev.ask(STATE, QUESTION, allow_paid=False))
        await asyncio.sleep(0)
        release.set()
        assert (await first) == (await second) == {"match": {"type": "noul", "noul": 0.5}}
        assert len(fake.bodies) == 1 and jev.meter.cached == 1
        assert second.result().origins["match"]["source"] == "shared"
        assert jev.meter.model == "jev-1"
        await jev.close()

    asyncio.run(check())


def test_provenance_names_the_provider_that_answered_and_answers_stay_with_it(tmp_path):
    cache = AnswerStore(tmp_path / "answers.sqlite")

    async def check():
        first = Client(backend("typesafe", "shared-name"), store=cache, transport=FakeJev().transport)
        await first.ask(STATE, QUESTION)
        await first.close()
        again = Client(backend("typesafe", "shared-name", key=""), store=cache, transport=FakeJev().transport)
        answer = await again.ask(STATE, QUESTION, allow_paid=False)
        assert answer.origins["match"]["provider"] == "typesafe" and again.meter.calls == 0
        await again.close()
        other = Client(backend("openrouter", "shared-name", key=""), store=cache, transport=FakeJev().transport)
        with pytest.raises(JevBudgetExceeded):  # another provider's answer is not this provider's answer
            await other.ask(STATE, QUESTION, allow_paid=False)
        assert other.meter.provider == "openrouter"
        await other.close()

    asyncio.run(check())
    cache.close()


def test_multi_question_client_keeps_cached_and_fresh_answers_separate(tmp_path):
    cache = AnswerStore(tmp_path / "answers.sqlite")
    fake = FakeJev()
    jev = Client(backend(model="jev-1"), store=cache, transport=fake.transport)
    cache.put(jev.key(STATE, QUESTION["match"]), {"noul": 0.7})
    questions = QUESTION | {"other": {"type": "noul", "instructions": "Same state?"}}

    async def check():
        answer = await jev.ask(STATE, questions)
        assert answer == {"match": {"noul": 0.7}, "other": {"type": "noul", "noul": 0.5}}
        assert list(fake.bodies[0]["questions"]) == ["other"]
        assert answer.origins["match"] == {"source": "cache"}
        assert answer.origins["other"]["resolved_model"] == "jev-1"
        assert jev.meter.resolved_models == ["jev-1"] and jev.meter.unknown_model_answers == 1
        assert jev.meter.model == ""
        assert await jev.ask(STATE, questions, allow_paid=False) == answer
        assert len(fake.bodies) == 1
        await jev.close()

    asyncio.run(check())
    cache.close()

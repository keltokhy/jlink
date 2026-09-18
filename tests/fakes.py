"""A fake Jev for offline tests. Give it a rule from (state, question) to a probability."""

from __future__ import annotations

import json

import httpx


class FakeJev:
    def __init__(self, rule=None, *, tokens: int = 330, cost: float | None = 0.0000139):
        self.rule = rule or (lambda state, question: 0.5)
        self.tokens, self.cost, self.bodies = tokens, cost, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        answers = {qid: {"type": "noul", "noul": float(self.rule(body["state"], q))}
                   for qid, q in body["questions"].items()}
        usage = {"input_tokens": self.tokens, "output_tokens": 1}
        if self.cost is not None:
            usage["cost"] = self.cost
        return httpx.Response(200, json={"model": body["model"], "answers": answers, "usage": usage})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

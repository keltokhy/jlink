"""jlink's view of the JevKit runtime: which providers it offers, and its names for the shared pieces."""

from __future__ import annotations

from jevkit_core import (
    AnswerStore as Cache,
    Backend,
    Client as Jev,
    JevBudgetExceeded,
    JevError,
    JevFatal,
    Meter,
    Settings,
    answer_key,
    catalog,
    resolve,
)

PROVIDERS = catalog("typesafe", "openrouter")


def resolve_backend(name: str | None = None, *, model: str | None = None, require_key: bool = True) -> Backend:
    """The provider to use, or with `require_key=False` the one a cache-only run would have used."""
    return resolve(PROVIDERS, name, model=model, require_key=require_key)


__all__ = [
    "PROVIDERS",
    "Backend",
    "Cache",
    "Jev",
    "JevBudgetExceeded",
    "JevError",
    "JevFatal",
    "Meter",
    "Settings",
    "answer_key",
    "resolve_backend",
]

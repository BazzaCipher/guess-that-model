"""The aggregated model registry.

All category modules contribute a ``MODELS`` list; this package concatenates
them into :data:`REGISTRY` and re-derives the trainer-facing helpers the game
used to define by hand (``SIMULATORS`` / ``FAMILY_OF`` / ``families`` /
``random_round``).  Family now lives on each :class:`Model`, so we no longer run
every simulator at import time just to read it.
"""
from __future__ import annotations

import numpy as np

from models.base import CATEGORIES, Model
from models import (
    conditional_mean,
    multivariate,
    portfolio_var,
    realised_vol,
    structural,
    tail_extreme,
    univariate_vol,
)

# Order follows the teaching blocks (CATEGORIES).
_MODULES = [
    conditional_mean,
    structural,
    univariate_vol,
    realised_vol,
    multivariate,
    portfolio_var,
    tail_extreme,
]

REGISTRY: list[Model] = [m for mod in _MODULES for m in mod.MODELS]

# ---- integrity checks (cheap; catches duplicate keys / bad wiring early) ----
_keys = [m.key for m in REGISTRY]
if len(_keys) != len(set(_keys)):
    dupes = sorted({k for k in _keys if _keys.count(k) > 1})
    raise ValueError(f"duplicate model keys: {dupes}")

# ---- lookups ----
BY_KEY: dict[str, Model] = {m.key: m for m in REGISTRY}
BY_NAME: dict[str, Model] = {m.name: m for m in REGISTRY}


def by_category(category: str) -> list[Model]:
    return [m for m in REGISTRY if m.category == category]


def categories() -> list[str]:
    """Categories that actually have at least one model, in teaching order."""
    present = {m.category for m in REGISTRY}
    return [c for c in CATEGORIES if c in present]


# ---- trainer-facing registry (back-compat with the original game API) ----
TRAINER_MODELS: list[Model] = [m for m in REGISTRY if m.trainer_eligible]

SIMULATORS = {m.name: m.simulate for m in TRAINER_MODELS}
FAMILY_OF = {m.name: m.family for m in TRAINER_MODELS}


def families() -> list[str]:
    """Distinct trainer families, ordered by first appearance in the registry."""
    seen: list[str] = []
    for m in TRAINER_MODELS:
        if m.family and m.family not in seen:
            seen.append(m.family)
    return seen


def random_round(rng: np.random.Generator, nobs: int):
    name = rng.choice(list(SIMULATORS.keys()))
    return SIMULATORS[name](rng, nobs=nobs)

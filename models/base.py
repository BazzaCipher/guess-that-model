"""Core data model for the econometrics study app.

Everything in the app hangs off a single :class:`Model` record.  A model can be:

* a **trainer** round — it carries a ``simulate`` callable returning a
  :class:`~simulators.SimResult` (the existing guessing-game payload); and/or
* an **explorer** demo — it carries a ``demo`` callable that renders an
  interactive Streamlit widget.

The same record also supplies the inventory card (``equation``/``summary``/
``tell``).  Three render modes (Trainer / Inventory / Explorers) are just views
over one list of these records — see :mod:`models` for the aggregated registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from simulators import SimResult  # shared per-round payload; simulators stays pure

# The seven course blocks, in teaching order.  Drives the Inventory filter and
# the set of Explorer pages.
CATEGORIES = [
    "Conditional mean",
    "Structural instability",
    "Univariate volatility",
    "Realised vol & long memory",
    "Multivariate GARCH",
    "Portfolio VaR",
    "Tail & extreme",
]

# Trainer tracks — the player picks one and guesses among its models.  Each
# track has a single diagnostic process whose ACF/PACF is shown (see
# views/trainer.py TRACK_VIEW): mean → the series, volatility → squared returns,
# realised vol → the RV series.
TRACKS = [
    "Conditional mean",
    "Volatility",
    "Realised volatility",
    "Both / other",
]


@dataclass(frozen=True)
class Model:
    """One catalogue entry — metadata plus optional behaviour."""

    key: str                       # stable unique id (state / widget namespacing)
    name: str                      # display name
    category: str                  # one of CATEGORIES
    equation: str                  # LaTeX body (no $), rendered via st.latex
    summary: str                   # 1–2 sentence "what it is"
    tell: str                      # generic diagnostic signature / giveaway
    family: Optional[str] = None   # trainer guess-bucket (GARCH/HAR/Mean/Other/…) or None
    track: Optional[str] = None    # trainer track — guess among models sharing a track
    beyond_course: bool = False    # lecturer-flagged "beyond this course"
    references: tuple[str, ...] = ()
    # behaviour — series models set simulate; non-series models set demo:
    simulate: Optional[Callable[[np.random.Generator, int], SimResult]] = None
    demo: Optional[Callable[[], None]] = None
    trainer_eligible: bool = False  # appears in the guessing game (needs simulate)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.name}: unknown category {self.category!r}")
        if self.trainer_eligible and self.simulate is None:
            raise ValueError(f"{self.name}: trainer_eligible but no simulate()")
        if self.trainer_eligible and self.track not in TRACKS:
            raise ValueError(f"{self.name}: trainer_eligible needs a valid track "
                             f"(got {self.track!r})")

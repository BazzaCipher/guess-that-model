"""Tail & extreme-risk models — EVT, McNeil-Frey, copulas."""
from __future__ import annotations

from demos.tail import demo_copulas, demo_evt_gpd, demo_mcneil_frey
from models.base import Model

_CAT = "Tail & extreme"

MODELS = [
    Model(
        key="evt_gpd",
        name="EVT — Generalised Pareto (POT)",
        category=_CAT,
        equation=r"\Pr(X-u \le y \mid X>u) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}",
        summary="Extreme Value Theory via peaks-over-threshold: model exceedances above a high "
                "threshold with a Generalised Pareto Distribution.",
        tell="The shape ξ sets tail heaviness (ξ>0 ⇒ power law); EVT extrapolates beyond the "
             "largest observed loss, where history is silent.",
        demo=demo_evt_gpd,
    ),
    Model(
        key="mcneil_frey",
        name="McNeil-Frey (conditional EVT)",
        category=_CAT,
        equation=r"\mathrm{VaR}_t = \sigma_t\,\big(q_\xi^{\text{EVT}}\big)",
        summary="Two stages: GARCH to standardise returns, then EVT on the standardised tail — "
                "a conditional VaR that adapts to volatility.",
        tell="The VaR band widens in turbulent stretches and tightens in calm ones, unlike a "
             "static historical VaR.",
        demo=demo_mcneil_frey,
    ),
    Model(
        key="copulas",
        name="Copulas (Gaussian / t / Gumbel)",
        category=_CAT,
        equation=r"H(x,y) = C\big(F_X(x), F_Y(y)\big)",
        summary="Separate the dependence structure from the margins. The copula family decides "
                "how strongly assets move together in the tails.",
        tell="Gaussian has zero tail dependence, Student-t symmetric tail dependence, Gumbel "
             "upper-tail only — the choice changes joint-crash probability dramatically.",
        demo=demo_copulas,
    ),
]

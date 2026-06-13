"""Tail & extreme-risk models — EVT, McNeil-Frey, copulas."""
from __future__ import annotations

from demos.beyond_course import demo_factor_copula, demo_vine_copula
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
    Model(
        key="vine_copula",
        name="Vine copulas",
        category=_CAT,
        equation=r"c_{1\dots d} = \prod_{\text{edges}} c_{ij\mid D}\big(\cdot\big)",
        summary="Build high-dimensional dependence as a cascade of bivariate pair-copulas in "
                "trees — each edge a different family, so tails are modelled pair by pair.",
        tell="Decomposes a joint copula into conditional pair-copulas; flexible but the tree "
             "structure and family choices proliferate.",
        beyond_course=True,
        demo=demo_vine_copula,
    ),
    Model(
        key="factor_copula",
        name="Oh-Patton factor copula",
        category=_CAT,
        equation=r"X_i = \beta_i Z + \varepsilon_i,\qquad Z,\varepsilon_i \sim t_\nu",
        summary="Generates dependence from a few common latent factors plus idiosyncratic "
                "noise — scalable to many assets, with tail dependence from a fat-tailed factor.",
        tell="A common factor drives joint moves; a fat-tailed factor creates the joint-crash "
             "(tail) dependence a Gaussian factor would miss.",
        beyond_course=True,
        demo=demo_factor_copula,
    ),
]

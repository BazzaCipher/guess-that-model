"""Conditional-mean (time-series) models."""
from __future__ import annotations

from models.base import Model
from simulators import (
    simulate_ar1,
    simulate_ar1_garch11,
    simulate_arma11,
    simulate_white_noise,
)

MODELS = [
    Model(
        key="white_noise",
        name="White noise",
        category="Conditional mean",
        family="Other",
        equation=r"r_t = \varepsilon_t,\qquad \varepsilon_t \sim \mathcal{N}(0,\sigma^2)",
        summary="The benchmark for 'nothing left to model' — independent draws with no "
                "dynamics in the mean or the variance.",
        tell="ACF and PACF stay inside the confidence bands at every lag, on both the "
             "returns and the squared returns.",
        simulate=simulate_white_noise,
        trainer_eligible=True,
    ),
    Model(
        key="ar1",
        name="AR(1) returns",
        category="Conditional mean",
        family="Mean",
        equation=r"r_t = \phi\, r_{t-1} + \varepsilon_t",
        summary="First-order autoregression in the mean: today's return loads on "
                "yesterday's.",
        tell="Returns ACF decays geometrically while the PACF cuts off after lag 1; the "
             "squared-return ACF only echoes it as ~rho^2 (a trap for 'looks like GARCH').",
        simulate=simulate_ar1,
        trainer_eligible=True,
    ),
    Model(
        key="arma11",
        name="ARMA(1,1) returns",
        category="Conditional mean",
        family="Mean",
        equation=r"r_t = \phi\, r_{t-1} + \varepsilon_t + \theta\, \varepsilon_{t-1}",
        summary="Autoregressive + moving-average mean dynamics — the workhorse linear "
                "mean model.",
        tell="Both the returns ACF and PACF tail off (neither cuts cleanly) — the ARMA "
             "signature.",
        simulate=simulate_arma11,
        trainer_eligible=True,
    ),
    Model(
        key="ar1_garch11",
        name="AR(1)-GARCH(1,1)",
        category="Conditional mean",
        family="Mean",
        equation=r"r_t = \phi\, r_{t-1} + \varepsilon_t,\quad "
                 r"\varepsilon_t = \sigma_t z_t,\quad "
                 r"\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2",
        summary="A conditional mean (AR) and a conditional variance (GARCH) stacked — the "
                "realistic combination for asset returns.",
        tell="Two genuine layers: returns ACF shows AR(1) decay AND the squared-return ACF "
             "persists more than rho^2 alone would give (real GARCH clustering on top).",
        simulate=simulate_ar1_garch11,
        trainer_eligible=True,
    ),
]

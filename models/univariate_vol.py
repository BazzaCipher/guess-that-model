"""Univariate volatility (GARCH-family) models."""
from __future__ import annotations

from models.base import Model
from simulators import (
    simulate_egarch11,
    simulate_garch11,
    simulate_garch11_t,
    simulate_garch21,
    simulate_gjr11,
)

MODELS = [
    Model(
        key="garch11_normal",
        name="GARCH(1,1) Normal",
        category="Univariate volatility",
        family="GARCH",
        equation=r"\sigma_t^2 = \omega + \alpha\, \varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2",
        summary="The workhorse volatility model: today's variance loads on yesterday's "
                "shock and yesterday's variance.",
        tell="Squared-returns ACF decays slowly with the PACF cutting off near lag 1.",
        simulate=simulate_garch11,
        trainer_eligible=True,
    ),
    Model(
        key="garch21_normal",
        name="GARCH(2,1) Normal",
        category="Univariate volatility",
        family="GARCH",
        equation=r"\sigma_t^2 = \omega + \alpha_1\varepsilon_{t-1}^2 + "
                 r"\alpha_2\varepsilon_{t-2}^2 + \beta\, \sigma_{t-1}^2",
        summary="GARCH with a second ARCH lag — an extra channel for past shocks to feed "
                "current variance.",
        tell="Squared-returns PACF shows mass at lags 1 AND 2 — the extra ARCH lag vs "
             "GARCH(1,1).",
        simulate=simulate_garch21,
        trainer_eligible=True,
    ),
    Model(
        key="gjr11",
        name="GJR-GARCH(1,1)",
        category="Univariate volatility",
        family="GARCH",
        equation=r"\sigma_t^2 = \omega + (\alpha + \gamma\,\mathbb{1}_{\varepsilon_{t-1}<0})"
                 r"\varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2",
        summary="Asymmetric GARCH (TARCH): a sign dummy lets negative shocks raise variance "
                "more than positive ones (the leverage effect).",
        tell="Variance after big NEGATIVE shocks is much larger; the cross-correlation of "
             "returns and squared returns is negative.",
        simulate=simulate_gjr11,
        trainer_eligible=True,
    ),
    Model(
        key="egarch11",
        name="EGARCH(1,1)",
        category="Univariate volatility",
        family="GARCH",
        equation=r"\ln\sigma_t^2 = \omega + \alpha\,(|z_{t-1}| - \mathbb{E}|z|) + "
                 r"\gamma\, z_{t-1} + \beta\, \ln\sigma_{t-1}^2",
        summary="Exponential GARCH models log-variance — no non-negativity constraints, and "
                "separate sign and magnitude effects.",
        tell="Log-variance specification — the ACF of squared returns persists almost like a "
             "unit root.",
        simulate=simulate_egarch11,
        trainer_eligible=True,
    ),
    Model(
        key="garch11_t",
        name="GARCH(1,1) Student-t",
        category="Univariate volatility",
        family="GARCH",
        equation=r"\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2,"
                 r"\quad z_t \sim t_\nu",
        summary="GARCH(1,1) dynamics with a fat-tailed Student-t conditional density — an "
                "estimation variant, not new dynamics.",
        tell="Same dynamics as GARCH(1,1) Normal but the kurtosis is much higher (visible as "
             "fatter tails in the raw series).",
        simulate=simulate_garch11_t,
        trainer_eligible=True,
    ),
]

"""Univariate volatility (GARCH-family) models."""
from __future__ import annotations

from demos.beyond_course import demo_midas_garch
from models.base import Model
from simulators import (
    simulate_aparch,
    simulate_arch_q,
    simulate_egarch11,
    simulate_figarch,
    simulate_garch11,
    simulate_garch11_t,
    simulate_garch21,
    simulate_garch_m,
    simulate_gjr11,
    simulate_igarch,
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
        track="Volatility",
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
        track="Volatility",
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
        track="Volatility",
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
        track="Volatility",
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
        track="Volatility",
        trainer_eligible=True,
    ),
    Model(
        key="arch_q",
        name="ARCH(q)",
        category="Univariate volatility",
        family="GARCH",
        equation=r"\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i\, \varepsilon_{t-i}^2",
        summary="The original Engle ARCH: variance depends only on past squared shocks, with "
                "no lagged-variance smoothing term.",
        tell="The squared-returns PACF cuts off after lag q (no GARCH smoothing), so the ACF "
             "decays much faster than a GARCH's.",
        simulate=simulate_arch_q,
        track="Volatility",
        trainer_eligible=True,
    ),
    Model(
        key="aparch",
        name="APARCH(1,1,1)",
        category="Univariate volatility",
        family="GARCH",
        equation=r"\sigma_t^\delta = \omega + \alpha\,(|\varepsilon_{t-1}| - "
                 r"\gamma\varepsilon_{t-1})^\delta + \beta\,\sigma_{t-1}^\delta",
        summary="Asymmetric Power ARCH — an estimated power delta on the volatility and a "
                "leverage term; nests GARCH, GJR, TARCH and others.",
        tell="The fitted power delta differs from 2 and the news-impact curve is asymmetric "
             "(negative shocks raise vol more).",
        simulate=simulate_aparch,
        track="Volatility",
        trainer_eligible=True,
    ),
    Model(
        key="garch_m",
        name="GARCH-M (in mean)",
        category="Univariate volatility",
        family="GARCH",
        equation=r"r_t = \mu + \kappa\,\sigma_t^2 + \varepsilon_t,\quad "
                 r"\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2",
        summary="GARCH-in-mean: the conditional variance enters the mean equation as a risk "
                "premium — higher expected return in riskier periods.",
        tell="Not visible in the ACF/PACF alone — the giveaway is that the return level drifts "
             "up during high-volatility spells (co-movement of mean and variance).",
        simulate=simulate_garch_m,
        track="Volatility",
        trainer_eligible=True,
    ),
    Model(
        key="igarch",
        name="IGARCH(1,1)",
        category="Univariate volatility",
        family="GARCH",
        equation=r"\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + (1-\alpha)\sigma_{t-1}^2",
        summary="Integrated GARCH: the persistence alpha+beta is pinned to 1, so variance "
                "shocks are permanent (the RiskMetrics EWMA is the omega=0 case).",
        tell="The squared-returns ACF barely decays — it behaves like a unit root in variance.",
        simulate=simulate_igarch,
        track="Volatility",
        trainer_eligible=True,
    ),
    Model(
        key="figarch",
        name="FIGARCH(1,d,1)",
        category="Univariate volatility",
        family="GARCH",
        equation=r"\phi(L)(1-L)^d\,\varepsilon_t^2 = \omega + "
                 r"[1-\beta(L)]\,(\varepsilon_t^2 - \sigma_t^2)",
        summary="Fractionally-integrated GARCH: long memory in variance via a fractional "
                "differencing order d between the stationary GARCH and the integrated IGARCH.",
        tell="The squared-returns ACF decays hyperbolically (slowly, like a power law) — "
             "slower than GARCH's geometric decay but faster than IGARCH's near-flat one.",
        simulate=simulate_figarch,
        track="Volatility",
        trainer_eligible=True,
    ),
    Model(
        key="midas_garch",
        name="GARCH-MIDAS",
        category="Univariate volatility",
        family="GARCH",
        equation=r"\sigma_t^2 = \tau_t \cdot g_t,\quad "
                 r"\tau_t = m + \theta\!\sum_{k} \varphi_k(\omega)\,RV_{t-k}",
        summary="Engle-Ghysels-Sohn two-component variance: a short-run GARCH g_t around a "
                "long-run MIDAS component τ_t built from realised variance.",
        tell="Volatility oscillates around a slow-moving secular level driven by "
             "macro/realised-variance information at a lower frequency.",
        beyond_course=True,
        demo=demo_midas_garch,
    ),
]


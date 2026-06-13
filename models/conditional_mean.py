"""Conditional-mean (time-series) models."""
from __future__ import annotations

from demos.conditional_mean import demo_midas, demo_single_index
from models.base import Model
from simulators import (
    simulate_ar1_garch11,
    simulate_ar_p,
    simulate_arima,
    simulate_arma_pq,
    simulate_ma_q,
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
        key="ma_q",
        name="MA(q) returns",
        category="Conditional mean",
        family="Mean",
        equation=r"r_t = \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + "
                 r"\theta_q\varepsilon_{t-q}",
        summary="Moving-average mean model: today's return is a weighted sum of the last q "
                "shocks.",
        tell="The returns ACF cuts off sharply after lag q while the PACF tails off — the "
             "mirror image of AR.",
        simulate=simulate_ma_q,
        trainer_eligible=True,
    ),
    Model(
        key="ar_p",
        name="AR(p) returns",
        category="Conditional mean",
        family="Mean",
        equation=r"r_t = \phi_1 r_{t-1} + \dots + \phi_p r_{t-p} + \varepsilon_t",
        summary="Autoregression in the mean: today's return loads on its own recent history.",
        tell="The returns PACF cuts off after lag p while the ACF tails off geometrically.",
        simulate=simulate_ar_p,
        trainer_eligible=True,
    ),
    Model(
        key="arma_pq",
        name="ARMA(p,q) returns",
        category="Conditional mean",
        family="Mean",
        equation=r"r_t = \sum_{i=1}^p \phi_i r_{t-i} + \varepsilon_t + "
                 r"\sum_{j=1}^q \theta_j \varepsilon_{t-j}",
        summary="Autoregressive + moving-average mean dynamics — the core linear mean model.",
        tell="Both the returns ACF and PACF tail off (neither cuts cleanly) — the ARMA "
             "signature.",
        simulate=simulate_arma_pq,
        trainer_eligible=True,
    ),
    Model(
        key="arima",
        name="ARIMA (unit root)",
        category="Conditional mean",
        family="Mean",
        equation=r"(1-L)\,r_t = \phi\,(1-L)\,r_{t-1} + \varepsilon_t",
        summary="An integrated I(1) series: non-stationary in levels, stationary after "
                "differencing once. The differencing 'd' in ARIMA(p,d,q).",
        tell="The level wanders like a random walk and its ACF decays almost linearly (near "
             "unit root); one difference makes it stationary.",
        simulate=simulate_arima,
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
    Model(
        key="midas",
        name="MIDAS regression",
        category="Conditional mean",
        equation=r"y_{t} = \beta_0 + \beta_1 \sum_{k=1}^{K} w_k(\theta)\, x^{(hf)}_{t-k} "
                 r"+ \varepsilon_t",
        summary="Mixed-data-sampling regression: relate a low-frequency target to many "
                "high-frequency lags through a parsimonious weight function w_k(θ).",
        tell="A handful of shape parameters (Almon / exponential-Almon / Beta) replace one "
             "coefficient per lag — the weights decay smoothly over the high-frequency window.",
        demo=demo_midas,
    ),
    Model(
        key="single_index",
        name="Single-index / market model",
        category="Conditional mean",
        equation=r"r_{i,t} = \alpha_i + \beta_i\, r_{m,t} + \varepsilon_{i,t}",
        summary="One-factor OLS regression of an asset on the market — the source of CAPM "
                "beta, reused for mapping and time-varying beta.",
        tell="The slope β is the systematic exposure; R² says how much of the asset's "
             "variance the single market factor explains.",
        demo=demo_single_index,
    ),
]

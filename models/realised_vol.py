"""Realised volatility & long-memory models."""
from __future__ import annotations

from demos.realised_vol import demo_rv_estimator
from models.base import Model
from simulators import (
    simulate_arfima_rv,
    simulate_arma11_rv,
    simulate_har_rv,
    simulate_har_rv_j,
)

MODELS = [
    Model(
        key="rv_estimator",
        name="Realised volatility estimator",
        category="Realised vol & long memory",
        equation=r"RV_t = \sum_{j=1}^{M} r_{t,j}^2 \;\xrightarrow{M\to\infty}\; "
                 r"\int_{t-1}^{t}\sigma^2(s)\,ds",
        summary="The estimator the whole block is built on: sum a day's squared intraday "
                "returns to estimate its integrated variance. RV is then treated as an "
                "observable series and modelled directly (HAR, ARFIMA, ARMA).",
        tell="Consistent for integrated variance as sampling rises — but microstructure noise "
             "adds ~2Mω² of upward bias, so the volatility-signature plot slopes up at the "
             "highest frequencies (motivating ~5-minute sampling).",
        demo=demo_rv_estimator,
    ),
    Model(
        key="har_rv",
        name="HAR-RV (Corsi)",
        category="Realised vol & long memory",
        family="HAR",
        equation=r"RV_t = c + \beta_d RV_{t-1} + \beta_w \overline{RV}^{(5)}_{t-1} + "
                 r"\beta_m \overline{RV}^{(22)}_{t-1} + \eta_t",
        summary="The Heterogeneous AutoRegressive model: daily, weekly and monthly realised-"
                "variance components — a practical long-memory approximation.",
        tell="ACF decays slowly (long-memory-like) but the PACF has spikes near lags 1, 5 "
             "and 22 (day / week / month).",
        simulate=simulate_har_rv,
        track="Realised volatility",
        trainer_eligible=True,
    ),
    Model(
        key="har_rv_j",
        name="HAR-RV-J",
        category="Realised vol & long memory",
        family="HAR",
        equation=r"RV_t = c + \beta_d RV_{t-1} + \beta_w \overline{RV}^{(5)}_{t-1} + "
                 r"\beta_m \overline{RV}^{(22)}_{t-1} + \text{(jumps)} + \eta_t",
        summary="HAR with an explicit jump component — captures large isolated bursts on top "
                "of the smooth long-memory dynamics.",
        tell="HAR shape with extra outliers — large isolated spikes that don't fit the smooth "
             "long-memory decay.",
        simulate=simulate_har_rv_j,
        track="Realised volatility",
        trainer_eligible=True,
    ),
    Model(
        key="arma11_logrv",
        name="ARMA(1,1) on log RV",
        category="Realised vol & long memory",
        family="Other",
        equation=r"\ln RV_t = \phi\, \ln RV_{t-1} + \eta_t + \theta\, \eta_{t-1}",
        summary="A plain ARMA fitted to log realised variance — a foil for HAR, since RV is "
                "treated as observable and modelled with standard ARMA/OLS.",
        tell="PACF decays quickly after lag 1 — no weekly / monthly bumps that a HAR would "
             "show.",
        simulate=simulate_arma11_rv,
        track="Realised volatility",
        trainer_eligible=True,
    ),
    Model(
        key="arfima_rv",
        name="ARFIMA on log RV",
        category="Realised vol & long memory",
        family="HAR",
        equation=r"(1-L)^d\,(\ln RV_t - \mu) = \varepsilon_t",
        summary="Fractionally-integrated ARMA on log realised variance — genuine long memory "
                "(0<d<½), the model HAR is a practical approximation to.",
        tell="The RV ACF decays hyperbolically and stays positive for hundreds of lags — "
             "smoother than HAR's distinct day/week/month PACF spikes.",
        simulate=simulate_arfima_rv,
        track="Realised volatility",
        trainer_eligible=True,
    ),
]


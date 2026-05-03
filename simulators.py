"""Process simulators for the model-guessing trainer.

Each ``simulate_*`` function returns a :class:`SimResult` carrying:

* ``series``        — the raw simulated path (returns or RV depending on family).
* ``target_sq``     — squared returns (None for HAR/ARMA-on-RV).
* ``target_rv``     — RV-style series (None for GARCH/white-noise).
* ``family``        — "GARCH", "HAR", or "Other".
* ``name``          — short model name shown to the user after the reveal.
* ``params``        — dict of true parameters drawn for this round.
* ``hint``          — single-line explanation of "the giveaway".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from arch.univariate import EGARCH, GARCH, Normal, StudentsT, ZeroMean


# ---------------------------------------------------------------------------
# result type
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    series: np.ndarray
    target_sq: Optional[np.ndarray]
    target_rv: Optional[np.ndarray]
    family: str
    name: str
    params: dict
    hint: str
    series_label: str = "returns"


# ---------------------------------------------------------------------------
# GARCH-family
# ---------------------------------------------------------------------------


def _draw_persistence(rng, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def simulate_garch11(rng: np.random.Generator, nobs: int) -> SimResult:
    while True:
        alpha = rng.uniform(0.04, 0.14)
        beta = rng.uniform(0.80, 0.94)
        if alpha + beta < 0.995:
            break
    omega = float(rng.uniform(0.02, 0.10))
    model = ZeroMean(volatility=GARCH(p=1, q=1), distribution=Normal(seed=rng))
    sim = model.simulate([omega, alpha, beta], nobs=nobs, burn=500)
    r = sim["data"].to_numpy()
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="GARCH",
        name="GARCH(1,1) Normal",
        params={"omega": omega, "alpha": alpha, "beta": beta},
        hint="Squared-returns ACF decays slowly with PACF cutting off near lag 1.",
    )


def simulate_garch21(rng: np.random.Generator, nobs: int) -> SimResult:
    while True:
        a1 = rng.uniform(0.02, 0.10)
        a2 = rng.uniform(0.02, 0.10)
        beta = rng.uniform(0.78, 0.90)
        if a1 + a2 + beta < 0.995:
            break
    omega = float(rng.uniform(0.02, 0.10))
    model = ZeroMean(volatility=GARCH(p=2, q=1), distribution=Normal(seed=rng))
    sim = model.simulate([omega, a1, a2, beta], nobs=nobs, burn=500)
    r = sim["data"].to_numpy()
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="GARCH",
        name="GARCH(2,1) Normal",
        params={"omega": omega, "alpha1": a1, "alpha2": a2, "beta": beta},
        hint="Squared-returns PACF shows mass at lags 1 AND 2 — extra ARCH lag vs GARCH(1,1).",
    )


def simulate_gjr11(rng: np.random.Generator, nobs: int) -> SimResult:
    while True:
        alpha = rng.uniform(0.01, 0.06)
        gamma = rng.uniform(0.06, 0.16)
        beta = rng.uniform(0.82, 0.92)
        if alpha + gamma / 2 + beta < 0.995:
            break
    omega = float(rng.uniform(0.02, 0.10))
    model = ZeroMean(volatility=GARCH(p=1, o=1, q=1), distribution=Normal(seed=rng))
    sim = model.simulate([omega, alpha, gamma, beta], nobs=nobs, burn=500)
    r = sim["data"].to_numpy()
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="GARCH",
        name="GJR-GARCH(1,1)",
        params={"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta},
        hint="Asymmetric — variance after big NEGATIVE shocks is much larger; cross-corr of r and r^2 is negative.",
    )


def simulate_egarch11(rng: np.random.Generator, nobs: int) -> SimResult:
    omega = float(rng.uniform(-0.10, -0.02))
    alpha = float(rng.uniform(0.05, 0.15))
    gamma = float(rng.uniform(-0.12, -0.02))  # negative leverage
    beta = float(rng.uniform(0.94, 0.99))
    model = ZeroMean(volatility=EGARCH(p=1, o=1, q=1), distribution=Normal(seed=rng))
    sim = model.simulate([omega, alpha, gamma, beta], nobs=nobs, burn=500)
    r = sim["data"].to_numpy()
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="GARCH",
        name="EGARCH(1,1)",
        params={"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta},
        hint="Log-variance specification — ACF of squared returns persists almost like a unit root.",
    )


def simulate_garch11_t(rng: np.random.Generator, nobs: int) -> SimResult:
    while True:
        alpha = rng.uniform(0.04, 0.14)
        beta = rng.uniform(0.80, 0.94)
        if alpha + beta < 0.995:
            break
    omega = float(rng.uniform(0.02, 0.10))
    nu = float(rng.uniform(5.0, 12.0))
    model = ZeroMean(volatility=GARCH(p=1, q=1), distribution=StudentsT(seed=rng))
    sim = model.simulate([omega, alpha, beta, nu], nobs=nobs, burn=500)
    r = sim["data"].to_numpy()
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="GARCH",
        name="GARCH(1,1) Student-t",
        params={"omega": omega, "alpha": alpha, "beta": beta, "nu": nu},
        hint="Same dynamics as GARCH(1,1) Normal but kurtosis is much higher (visible in raw series).",
    )


# ---------------------------------------------------------------------------
# HAR-family — simulate on log(RV) for stability, exponentiate to get RV
# ---------------------------------------------------------------------------


def _har_sim(rng, nobs, b_d, b_w, b_m, sigma, jump_intensity=0.0, jump_scale=0.0):
    burn = 200
    T = nobs + burn
    log_rv = np.empty(T)
    # initialise around a sensible mean: solve steady state of log-RV
    c = np.log(0.5) * (1 - b_d - b_w - b_m)  # target mean log RV ~ log(0.5)
    log_rv[:22] = np.log(0.5) + rng.normal(0, sigma, 22)
    for t in range(22, T):
        rv_d = np.exp(log_rv[t - 1])
        rv_w = np.exp(log_rv[t - 5 : t]).mean()
        rv_m = np.exp(log_rv[t - 22 : t]).mean()
        mu = c + b_d * np.log(rv_d) + b_w * np.log(rv_w) + b_m * np.log(rv_m)
        log_rv[t] = mu + rng.normal(0, sigma)
        if jump_intensity > 0 and rng.random() < jump_intensity:
            log_rv[t] += rng.normal(0, jump_scale)
    rv = np.exp(log_rv[burn:])
    return rv


def simulate_har_rv(rng: np.random.Generator, nobs: int) -> SimResult:
    b_d = float(rng.uniform(0.30, 0.45))
    b_w = float(rng.uniform(0.20, 0.35))
    b_m = float(rng.uniform(0.15, 0.30))
    sigma = float(rng.uniform(0.30, 0.50))
    rv = _har_sim(rng, nobs, b_d, b_w, b_m, sigma)
    return SimResult(
        series=rv,
        target_sq=None,
        target_rv=rv,
        family="HAR",
        name="HAR-RV (Corsi)",
        params={"beta_d": b_d, "beta_w": b_w, "beta_m": b_m, "sigma_eta": sigma},
        hint="ACF decays slowly (long-memory-like) but PACF has spikes near lags 1, 5, and 22.",
        series_label="realised variance",
    )


def simulate_har_rv_j(rng: np.random.Generator, nobs: int) -> SimResult:
    b_d = float(rng.uniform(0.30, 0.45))
    b_w = float(rng.uniform(0.20, 0.35))
    b_m = float(rng.uniform(0.15, 0.30))
    sigma = float(rng.uniform(0.25, 0.45))
    jump_p = float(rng.uniform(0.03, 0.07))
    jump_s = float(rng.uniform(0.6, 1.2))
    rv = _har_sim(rng, nobs, b_d, b_w, b_m, sigma, jump_p, jump_s)
    return SimResult(
        series=rv,
        target_sq=None,
        target_rv=rv,
        family="HAR",
        name="HAR-RV-J",
        params={
            "beta_d": b_d,
            "beta_w": b_w,
            "beta_m": b_m,
            "sigma_eta": sigma,
            "jump_p": jump_p,
            "jump_scale": jump_s,
        },
        hint="HAR shape with extra outliers — large isolated spikes that don't fit the smooth long-memory decay.",
        series_label="realised variance",
    )


# ---------------------------------------------------------------------------
# Foils — ARMA(1,1) on RV and white noise
# ---------------------------------------------------------------------------


def simulate_arma11_rv(rng: np.random.Generator, nobs: int) -> SimResult:
    phi = float(rng.uniform(0.5, 0.85))
    theta = float(rng.uniform(-0.5, -0.1))
    sigma = float(rng.uniform(0.20, 0.40))
    burn = 200
    T = nobs + burn
    eps = rng.normal(0, sigma, T)
    log_rv = np.empty(T)
    log_rv[0] = 0.0
    for t in range(1, T):
        log_rv[t] = phi * log_rv[t - 1] + eps[t] + theta * eps[t - 1]
    rv = np.exp(log_rv[burn:] + np.log(0.5))
    return SimResult(
        series=rv,
        target_sq=None,
        target_rv=rv,
        family="Other",
        name="ARMA(1,1) on log RV",
        params={"phi": phi, "theta": theta, "sigma": sigma},
        hint="PACF decays quickly after lag 1 — no weekly/monthly bumps a HAR would show.",
        series_label="realised variance",
    )


def simulate_white_noise(rng: np.random.Generator, nobs: int) -> SimResult:
    sigma = float(rng.uniform(0.8, 1.4))
    r = rng.normal(0.0, sigma, nobs)
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="Other",
        name="White noise",
        params={"sigma": sigma},
        hint="No structure anywhere — ACF and PACF stay inside the bands at every lag.",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SIMULATORS = {
    "GARCH(1,1) Normal": simulate_garch11,
    "GARCH(2,1) Normal": simulate_garch21,
    "GJR-GARCH(1,1)": simulate_gjr11,
    "EGARCH(1,1)": simulate_egarch11,
    "GARCH(1,1) Student-t": simulate_garch11_t,
    "HAR-RV (Corsi)": simulate_har_rv,
    "HAR-RV-J": simulate_har_rv_j,
    "ARMA(1,1) on log RV": simulate_arma11_rv,
    "White noise": simulate_white_noise,
}


FAMILY_OF = {name: SIMULATORS[name](np.random.default_rng(0), nobs=300).family
             for name in SIMULATORS}


def families() -> list[str]:
    return ["GARCH", "HAR", "Other"]


def random_round(rng: np.random.Generator, nobs: int):
    name = rng.choice(list(SIMULATORS.keys()))
    return SIMULATORS[name](rng, nobs=nobs)

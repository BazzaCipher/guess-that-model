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
from arch.univariate import (
    APARCH,
    ARCH,
    ARCHInMean,
    EGARCH,
    FIGARCH,
    GARCH,
    Normal,
    StudentsT,
    ZeroMean,
)

import generators as gen

# IGARCH (alpha+beta=1) and FIGARCH are non-stationary in variance *by design*;
# arch warns about it on every simulate, which would flood the deploy logs.
import warnings as _warnings

from arch.utility.exceptions import InitialValueWarning as _InitialValueWarning

_warnings.filterwarnings("ignore", category=_InitialValueWarning)


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
# Conditional-mean family — structure lives in the *returns* ACF/PACF, not
# (only) in squared returns.  Counterpart to the variance-focused families.
# ---------------------------------------------------------------------------


def simulate_ma_q(rng: np.random.Generator, nobs: int) -> SimResult:
    q = int(rng.integers(1, 4))
    theta = rng.uniform(-0.7, 0.7, size=q)
    theta[0] = float(np.sign(theta[0] or 1.0) * rng.uniform(0.4, 0.8))  # ensure a visible lag-1
    sigma = float(rng.uniform(0.8, 1.4))
    burn = 50
    eps = rng.normal(0.0, sigma, nobs + burn)
    r = eps.copy()
    for j in range(1, q + 1):
        r[j:] += theta[j - 1] * eps[:-j]
    r = r[burn:]
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="Mean",
        name="MA(q) returns",
        params={"q": q, **{f"theta{j+1}": round(float(theta[j]), 3) for j in range(q)},
                "sigma": sigma},
        hint=f"Moving-average of order q={q}: the returns ACF cuts off sharply after lag "
             f"{q} while the PACF tails off — the mirror image of AR.",
    )


def simulate_ar_p(rng: np.random.Generator, nobs: int) -> SimResult:
    p = int(rng.integers(1, 4))
    pacf = rng.uniform(-0.5, 0.6, size=p)
    pacf[0] = float(rng.uniform(0.35, 0.7))
    phi = gen.ar_from_pacf(pacf)
    sigma = float(rng.uniform(0.8, 1.4))
    burn = 200
    eps = rng.normal(0.0, sigma, nobs + burn)
    r = gen.ar_recursion(phi, eps)[burn:]
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="Mean",
        name="AR(p) returns",
        params={"p": p, **{f"phi{j+1}": round(float(phi[j]), 3) for j in range(p)},
                "sigma": sigma},
        hint=f"Autoregression of order p={p}: the returns PACF cuts off after lag {p} while "
             f"the ACF tails off geometrically.",
    )


def simulate_arma_pq(rng: np.random.Generator, nobs: int) -> SimResult:
    p = int(rng.integers(1, 3))
    q = int(rng.integers(1, 3))
    pacf = rng.uniform(-0.4, 0.6, size=p)
    pacf[0] = float(rng.uniform(0.35, 0.7))
    phi = gen.ar_from_pacf(pacf)
    theta = rng.uniform(-0.6, 0.6, size=q)
    sigma = float(rng.uniform(0.8, 1.4))
    burn = 200
    eps = rng.normal(0.0, sigma, nobs + burn)
    r = gen.arma_recursion(phi, theta, eps)[burn:]
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="Mean",
        name="ARMA(p,q) returns",
        params={"p": p, "q": q, "sigma": sigma},
        hint="Mixed AR+MA: both the returns ACF and PACF tail off (neither cuts cleanly) — "
             "the ARMA signature.",
    )


def simulate_arima(rng: np.random.Generator, nobs: int) -> SimResult:
    """ARIMA(1,1,0): the differenced series is AR(1); the level is I(1)."""
    phi = float(rng.uniform(-0.3, 0.5))
    sigma = float(rng.uniform(0.8, 1.4))
    drift = float(rng.uniform(-0.02, 0.02))
    burn = 200
    eps = rng.normal(0.0, sigma, nobs + burn)
    d = gen.ar_recursion(np.array([phi]), eps)
    level = drift * np.arange(nobs + burn) + np.cumsum(d)
    level = level[burn:]
    return SimResult(
        series=level,
        target_sq=None,
        target_rv=None,
        family="Mean",
        name="ARIMA (unit root)",
        params={"phi_diff": phi, "drift": drift, "sigma": sigma},
        hint="Non-stationary I(1) level: it wanders like a random walk and its ACF decays "
             "almost linearly (near unit root). Differencing once makes it stationary.",
        series_label="level",
    )


def simulate_ar1_garch11(rng: np.random.Generator, nobs: int) -> SimResult:
    phi = float(rng.uniform(0.30, 0.60))
    while True:
        alpha = rng.uniform(0.04, 0.12)
        beta = rng.uniform(0.80, 0.92)
        if alpha + beta < 0.995:
            break
    omega = float(rng.uniform(0.02, 0.10))
    vol = ZeroMean(volatility=GARCH(p=1, q=1), distribution=Normal(seed=rng))
    e = vol.simulate([omega, alpha, beta], nobs=nobs, burn=500)["data"].to_numpy()
    r = np.empty_like(e)
    r[0] = e[0]
    for t in range(1, len(e)):
        r[t] = phi * r[t - 1] + e[t]
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="Mean",
        name="AR(1)-GARCH(1,1)",
        params={"phi": phi, "omega": omega, "alpha": alpha, "beta": beta},
        hint="Two genuine layers: returns ACF shows AR(1) decay (conditional mean), AND "
             "the squared-return ACF persists *more* than rho^2 alone would give — real "
             "GARCH clustering on top of the mean dynamics.",
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
# Additional univariate-volatility variants (arch.univariate)
# ---------------------------------------------------------------------------


def simulate_arch_q(rng: np.random.Generator, nobs: int) -> SimResult:
    q = int(rng.integers(2, 5))
    while True:
        alphas = rng.uniform(0.05, 0.35, size=q)
        if alphas.sum() < 0.92:
            break
    omega = float(rng.uniform(0.2, 0.6))
    model = ZeroMean(volatility=ARCH(p=q), distribution=Normal(seed=rng))
    r = model.simulate([omega, *alphas], nobs=nobs, burn=500)["data"].to_numpy()
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="GARCH",
        name="ARCH(q)",
        params={"q": q, **{f"alpha{j+1}": round(float(alphas[j]), 3) for j in range(q)},
                "omega": omega},
        hint=f"Pure ARCH(q={q}): the squared-returns PACF cuts off after lag {q} (no GARCH "
             f"smoothing), so the ACF decays much faster than a GARCH would.",
    )


def simulate_aparch(rng: np.random.Generator, nobs: int) -> SimResult:
    omega = float(rng.uniform(0.02, 0.10))
    alpha = float(rng.uniform(0.05, 0.12))
    gamma = float(rng.uniform(0.2, 0.6))      # asymmetry, in (-1,1)
    beta = float(rng.uniform(0.85, 0.92))
    delta = float(rng.uniform(1.0, 2.0))      # estimated power
    model = ZeroMean(volatility=APARCH(p=1, o=1, q=1), distribution=Normal(seed=rng))
    r = model.simulate([omega, alpha, gamma, beta, delta], nobs=nobs, burn=500)["data"].to_numpy()
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="GARCH",
        name="APARCH(1,1,1)",
        params={"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta, "delta": delta},
        hint="Asymmetric power ARCH: the variance acts on |returns|^delta with an estimated "
             "power delta != 2, plus a leverage term — generalises GJR and EGARCH.",
    )


def simulate_garch_m(rng: np.random.Generator, nobs: int) -> SimResult:
    omega = float(rng.uniform(0.02, 0.10))
    alpha = float(rng.uniform(0.05, 0.12))
    beta = float(rng.uniform(0.82, 0.92))
    kappa = float(rng.uniform(0.05, 0.25))    # risk-premium loading on variance
    const = float(rng.uniform(-0.02, 0.02))
    model = ARCHInMean(volatility=GARCH(p=1, q=1), distribution=Normal(seed=rng), form="var")
    r = model.simulate([const, kappa, omega, alpha, beta], nobs=nobs, burn=500)["data"].to_numpy()
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="GARCH",
        name="GARCH-M",
        params={"const": const, "kappa": kappa, "omega": omega, "alpha": alpha, "beta": beta},
        hint="GARCH-in-mean: the conditional variance enters the mean equation (a risk "
             "premium), so the return level drifts up in high-volatility spells.",
    )


def simulate_igarch(rng: np.random.Generator, nobs: int) -> SimResult:
    alpha = float(rng.uniform(0.05, 0.15))
    beta = 1.0 - alpha                         # alpha + beta = 1 (integrated)
    omega = float(rng.uniform(0.01, 0.05))
    model = ZeroMean(volatility=GARCH(p=1, q=1), distribution=Normal(seed=rng))
    r = model.simulate([omega, alpha, beta], nobs=nobs, burn=1000)["data"].to_numpy()
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="GARCH",
        name="IGARCH(1,1)",
        params={"omega": omega, "alpha": alpha, "beta": beta},
        hint="Integrated GARCH (alpha+beta=1): variance shocks are permanent, so the squared-"
             "returns ACF behaves like a unit root and barely decays.",
    )


def simulate_figarch(rng: np.random.Generator, nobs: int) -> SimResult:
    omega = float(rng.uniform(0.02, 0.08))
    phi = float(rng.uniform(0.1, 0.3))
    d = float(rng.uniform(0.3, 0.6))           # fractional integration order
    beta = float(rng.uniform(0.4, 0.6))
    model = ZeroMean(volatility=FIGARCH(p=1, q=1), distribution=Normal(seed=rng))
    r = model.simulate([omega, phi, d, beta], nobs=nobs, burn=1000)["data"].to_numpy()
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="GARCH",
        name="FIGARCH(1,d,1)",
        params={"omega": omega, "phi": phi, "d": d, "beta": beta},
        hint="Fractionally-integrated GARCH: the squared-returns ACF decays hyperbolically "
             "(long memory) — slower than GARCH's geometric decay but faster than IGARCH.",
    )


# ---------------------------------------------------------------------------
# Long memory on realised variance
# ---------------------------------------------------------------------------


def simulate_arfima_rv(rng: np.random.Generator, nobs: int) -> SimResult:
    d = float(rng.uniform(0.30, 0.45))
    sigma = float(rng.uniform(0.30, 0.45))
    log_rv = gen.arfima_0d0(d, nobs, rng, sigma=sigma, trunc=1000) + np.log(0.5)
    rv = np.exp(log_rv)
    return SimResult(
        series=rv,
        target_sq=None,
        target_rv=rv,
        family="HAR",
        name="ARFIMA on log RV",
        params={"d": d, "sigma": sigma},
        hint="Fractionally-integrated long memory: the RV ACF decays hyperbolically and "
             "stays positive for hundreds of lags — smoother than HAR's day/week/month spikes.",
    )


# ---------------------------------------------------------------------------
# Structural instability — regime/break dynamics
# ---------------------------------------------------------------------------


def simulate_setar(rng: np.random.Generator, nobs: int) -> SimResult:
    """Two-regime threshold AR(1): persistence flips with the sign of r_{t-1}."""
    phi_lo = float(rng.uniform(-0.6, -0.1))
    phi_hi = float(rng.uniform(0.4, 0.8))
    c = 0.0
    sigma = float(rng.uniform(0.8, 1.2))
    burn = 200
    T = nobs + burn
    eps = rng.normal(0.0, sigma, T)
    r = np.zeros(T)
    for t in range(1, T):
        phi = phi_hi if r[t - 1] >= c else phi_lo
        r[t] = phi * r[t - 1] + eps[t]
    r = r[burn:]
    return SimResult(
        series=r,
        target_sq=r ** 2,
        target_rv=None,
        family="Structural",
        name="SETAR (threshold AR)",
        params={"phi_low": phi_lo, "phi_high": phi_hi, "threshold": c, "sigma": sigma},
        hint="Self-exciting threshold AR: the dynamics switch with an OBSERVED variable "
             "(here the sign of the last return), so persistence is regime-dependent.",
    )


def simulate_markov_switch(rng: np.random.Generator, nobs: int) -> SimResult:
    """Two-state Markov switching in variance — a classic GARCH look-alike foil."""
    p00 = float(rng.uniform(0.95, 0.99))
    p11 = float(rng.uniform(0.95, 0.99))
    P = np.array([[p00, 1 - p00], [1 - p11, p11]])
    sig_lo = float(rng.uniform(0.5, 0.9))
    sig_hi = float(rng.uniform(1.8, 3.0))
    y, _states = gen.markov_sim(P, mus=[0.0, 0.0], sigmas=[sig_lo, sig_hi],
                                T=nobs, rng=rng)
    return SimResult(
        series=y,
        target_sq=y ** 2,
        target_rv=None,
        family="Structural",
        name="Markov regime-switching",
        params={"p00": p00, "p11": p11, "sigma_low": sig_lo, "sigma_high": sig_hi},
        hint="Hidden two-state variance: persistent regimes make the squared-returns ACF "
             "look GARCH-like, but the state is an UNOBSERVED Markov chain, not a recursion.",
    )


def simulate_unit_root_breaks(rng: np.random.Generator, nobs: int) -> SimResult:
    """Random walk with occasional level breaks (segmented intercept)."""
    sigma = float(rng.uniform(0.8, 1.2))
    n_breaks = int(rng.integers(2, 5))
    breaks = np.sort(rng.choice(np.arange(nobs // 10, nobs - nobs // 10),
                                size=n_breaks, replace=False))
    jumps = rng.normal(0.0, 6.0, size=n_breaks)
    level = np.cumsum(rng.normal(0.0, sigma, nobs))
    for b, j in zip(breaks, jumps):
        level[b:] += j
    return SimResult(
        series=level,
        target_sq=None,
        target_rv=None,
        family="Structural",
        name="Unit root + breaks",
        params={"sigma": sigma, "n_breaks": n_breaks},
        hint="An I(1) random walk PLUS abrupt level shifts — the breaks masquerade as extra "
             "persistence and bias unit-root tests toward non-rejection.",
    )


# The registry (SIMULATORS / FAMILY_OF / families / random_round) now lives in
# the ``models`` package, derived from the Model records.  This module stays a
# pure library of ``simulate_*`` functions with no Streamlit or registry deps.

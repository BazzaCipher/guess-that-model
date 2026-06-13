"""Shared hand-rolled numerical kernels (pure numpy/scipy — no Streamlit).

These back the simulators and explorer demos for models with no ready-made
simulate path in ``arch``/``statsmodels``.  Kept dependency-free and headless so
they're unit-testable in isolation.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Conditional mean
# ---------------------------------------------------------------------------


def ar_from_pacf(pacf) -> np.ndarray:
    """AR coefficients (phi_1..phi_p) from partial autocorrelations via the
    Durbin–Levinson recursion.  Any ``pacf`` with all |values| < 1 yields a
    stationary AR process, so this is a safe way to draw random stationary AR.
    """
    pacf = np.asarray(pacf, dtype=float)
    phi = np.array([pacf[0]])
    for k in range(1, len(pacf)):
        pk = pacf[k]
        phi = np.append(phi - pk * phi[::-1], pk)
    return phi


def ar_recursion(phi: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Generate an AR(p) path: r_t = sum_j phi_j r_{t-j} + eps_t."""
    p = len(phi)
    r = np.zeros_like(eps)
    for t in range(p, len(eps)):
        r[t] = phi @ r[t - p:t][::-1] + eps[t]
    return r


def arma_recursion(phi: np.ndarray, theta: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Generate an ARMA(p,q) path."""
    p, q = len(phi), len(theta)
    r = np.zeros_like(eps)
    start = max(p, q)
    for t in range(start, len(eps)):
        ar = phi @ r[t - p:t][::-1] if p else 0.0
        ma = theta @ eps[t - q:t][::-1] if q else 0.0
        r[t] = ar + eps[t] + ma
    return r


# ---------------------------------------------------------------------------
# Long memory (fractional integration)
# ---------------------------------------------------------------------------


def frac_diff_weights(d: float, K: int) -> np.ndarray:
    """MA(inf) weights of (1-L)^{-d}, truncated at K lags (psi_0..psi_K)."""
    psi = np.empty(K + 1)
    psi[0] = 1.0
    for k in range(1, K + 1):
        psi[k] = psi[k - 1] * (k - 1 + d) / k
    return psi


def arfima_0d0(d: float, T: int, rng: np.random.Generator, sigma: float = 1.0,
               trunc: int = 1000) -> np.ndarray:
    """Fractionally-integrated noise ARFIMA(0,d,0), length T.

    Truncates the infinite MA weights at ``trunc`` (introduces a mild short-
    memory bias documented in the inventory tell).
    """
    K = int(min(trunc, T))
    psi = frac_diff_weights(d, K)
    eps = rng.normal(0.0, sigma, T + K)
    x = np.convolve(eps, psi, mode="valid")  # length (T+K)-(K+1)+1 == T
    return x[-T:]


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------


def markov_sim(P: np.ndarray, mus, sigmas, T: int, rng: np.random.Generator,
               burn: int = 200):
    """Simulate a Markov regime-switching series.

    Returns ``(y, states)`` of length T.  ``P`` is a row-stochastic transition
    matrix; emission in state s is N(mus[s], sigmas[s]^2).
    """
    P = np.asarray(P, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    k = len(mus)
    states = np.empty(T + burn, dtype=int)
    s = 0
    for t in range(T + burn):
        states[t] = s
        s = rng.choice(k, p=P[s])
    y = mus[states] + sigmas[states] * rng.normal(size=T + burn)
    return y[burn:], states[burn:]

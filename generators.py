"""Shared hand-rolled numerical kernels (pure numpy/scipy — no Streamlit).

These back the simulators and explorer demos for models with no ready-made
simulate path in ``arch``/``statsmodels``.  Kept dependency-free and headless so
they're unit-testable in isolation.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


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


# ---------------------------------------------------------------------------
# MIDAS weighting schemes  (weights over lags 1..K, normalised to sum 1)
# ---------------------------------------------------------------------------


def midas_weights(scheme: str, K: int, p1: float = 0.0, p2: float = -0.05,
                  a: float = 1.0, b: float = 4.0) -> np.ndarray:
    """Normalised MIDAS lag weights.

    scheme: 'Almon' (quadratic), 'Exponential Almon' (exp of quadratic), or
    'Beta' (two-parameter Beta density).  Returns length-K weights summing to 1.
    """
    k = np.arange(1, K + 1)
    if scheme == "Almon":
        w = 1.0 + p1 * k + p2 * k ** 2
        w = np.clip(w, 0.0, None)
    elif scheme == "Exponential Almon":
        w = np.exp(p1 * k + p2 * k ** 2)
    elif scheme == "Beta":
        x = (k - 1) / max(K - 1, 1)              # in [0,1]
        x = np.clip(x, 1e-6, 1 - 1e-6)
        w = x ** (a - 1) * (1 - x) ** (b - 1)
    else:
        raise ValueError(f"unknown MIDAS scheme {scheme!r}")
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


# ---------------------------------------------------------------------------
# Multivariate volatility (2-asset) — correlation/covariance paths
# ---------------------------------------------------------------------------


def two_garch(rng: np.random.Generator, n: int, burn: int = 500):
    """Two independent GARCH(1,1) series; returns (resid (n,2), sigma (n,2)).

    ``resid`` are the raw returns (mean zero); standardised residuals are
    ``resid / sigma``.  Used as the building block for CCC/DCC/BEKK demos.
    """
    from arch.univariate import GARCH, Normal, ZeroMean
    out_r, out_s = [], []
    for _ in range(2):
        omega = float(rng.uniform(0.02, 0.08))
        alpha = float(rng.uniform(0.05, 0.12))
        beta = float(rng.uniform(0.84, 0.92))
        m = ZeroMean(volatility=GARCH(p=1, q=1), distribution=Normal(seed=rng))
        sim = m.simulate([omega, alpha, beta], nobs=n, burn=burn)
        out_r.append(sim["data"].to_numpy())
        out_s.append(np.sqrt(sim["volatility"].to_numpy()))
    return np.column_stack(out_r), np.column_stack(out_s)


def dcc_path(std_resid: np.ndarray, a: float, b: float) -> np.ndarray:
    """Engle DCC correlation path for 2 assets.

    std_resid: (n,2) standardised residuals.  Returns rho_t of length n.
    Enforces a,b >= 0 and a+b < 1 upstream (caller clamps sliders).
    """
    n = std_resid.shape[0]
    Qbar = np.corrcoef(std_resid.T)
    Q = Qbar.copy()
    rho = np.empty(n)
    for t in range(n):
        u = std_resid[t][:, None]
        Q = (1 - a - b) * Qbar + a * (u @ u.T) + b * Q
        d = np.sqrt(np.diag(Q))
        rho[t] = Q[0, 1] / (d[0] * d[1])
    return rho


def bekk_path(rng: np.random.Generator, n: int, a: float = 0.25, b: float = 0.93,
              burn: int = 200):
    """Scalar BEKK(1,1) for 2 assets: Sigma_t = CC' + a^2 e e' + b^2 Sigma_{t-1}.

    Positive-definite by construction.  Returns (rho_t length n, returns (n,2)).
    """
    rho0 = 0.3
    Sigma_uncond = np.array([[1.0, rho0], [rho0, 1.0]])
    C = np.linalg.cholesky((1 - a ** 2 - b ** 2) * Sigma_uncond)
    CC = C @ C.T
    Sigma = Sigma_uncond.copy()
    e = np.zeros((2, 1))
    rho = np.empty(n + burn)
    rets = np.empty((n + burn, 2))
    for t in range(n + burn):
        Sigma = CC + a ** 2 * (e @ e.T) + b ** 2 * Sigma
        L = np.linalg.cholesky(Sigma)
        z = rng.standard_normal((2, 1))
        e = L @ z
        rets[t] = e.ravel()
        d = np.sqrt(np.diag(Sigma))
        rho[t] = Sigma[0, 1] / (d[0] * d[1])
    return rho[burn:], rets[burn:]


# ---------------------------------------------------------------------------
# Dimension reduction
# ---------------------------------------------------------------------------


def pca(cov: np.ndarray):
    """Eigen-decomposition of a covariance/correlation matrix, descending.

    Returns (eigenvalues, eigenvectors) with columns as components.
    """
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]


# ---------------------------------------------------------------------------
# Tail / extreme value
# ---------------------------------------------------------------------------


def gpd_fit(losses: np.ndarray, q: float = 0.95):
    """Peaks-over-threshold GPD fit.

    Returns (threshold u, xi, beta, exceedances) where exceedances = losses-u
    for losses above the q-quantile threshold.
    """
    u = float(np.quantile(losses, q))
    exc = losses[losses > u] - u
    xi, _loc, beta = stats.genpareto.fit(exc, floc=0.0)
    return u, float(xi), float(beta), exc


def _rstable(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Positive stable draws with Laplace transform exp(-s^alpha), alpha in (0,1)."""
    U = rng.uniform(1e-9, np.pi, n)
    W = rng.exponential(1.0, n)
    a = alpha
    term = (np.sin(a * U) / np.sin(U) ** (1 / a)) * \
           (np.sin((1 - a) * U) / W) ** ((1 - a) / a)
    return term


def copula_sample(family: str, param: float, n: int, rng: np.random.Generator,
                  df: int = 4):
    """Sample (u, v) uniforms from a bivariate copula.

    family: 'Gaussian' (param=rho), 'Student-t' (param=rho, df), 'Gumbel'
    (param=theta>=1, upper-tail dependent).
    """
    if family == "Gaussian":
        rho = param
        L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
        z = rng.standard_normal((n, 2)) @ L.T
        return stats.norm.cdf(z[:, 0]), stats.norm.cdf(z[:, 1])
    if family == "Student-t":
        rho = param
        L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
        z = rng.standard_normal((n, 2)) @ L.T
        g = rng.chisquare(df, n) / df
        t = z / np.sqrt(g)[:, None]
        return stats.t.cdf(t[:, 0], df), stats.t.cdf(t[:, 1], df)
    if family == "Gumbel":
        theta = max(param, 1.0)
        if theta == 1.0:                       # independence
            return rng.random(n), rng.random(n)
        M = _rstable(1.0 / theta, n, rng)
        e = rng.exponential(1.0, (n, 2))
        u = np.exp(-(e[:, 0] / M) ** (1 / theta))
        v = np.exp(-(e[:, 1] / M) ** (1 / theta))
        return u, v
    raise ValueError(f"unknown copula family {family!r}")

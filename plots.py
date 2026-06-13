"""Matplotlib helpers for the trainer + explorer views.

Every helper returns a ``plt.Figure``; callers render with
``st.pyplot(fig, clear_figure=True)`` (and ``plt.close`` in headless tests).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def _finish(fig: plt.Figure) -> plt.Figure:
    fig.tight_layout()
    return fig


def series_fig(y: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.plot(y, linewidth=0.6)
    ax.set_title(title)
    ax.margins(x=0)
    fig.tight_layout()
    return fig


def acf_pacf_fig(y: np.ndarray, label: str, lags: int = 30) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    plot_acf(y, lags=lags, ax=axes[0], zero=False)
    axes[0].set_title(f"ACF — {label}")
    plot_pacf(y, lags=lags, ax=axes[1], method="ywm", zero=False)
    axes[1].set_title(f"PACF — {label}")
    fig.tight_layout()
    return fig


def leverage_xcorr_fig(r: np.ndarray, lags: int = 12) -> plt.Figure:
    """Cross-correlation of squared returns with *past* returns:
    corr(r²_t, r_{t-k}) for k=1..lags. Persistently negative ⇒ leverage
    (a big down-move raises tomorrow's variance more than an up-move) — the
    asymmetry that separates GJR / EGARCH from a symmetric GARCH, and which the
    squared-returns ACF/PACF cannot show."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    sq = r ** 2
    sq -= sq.mean()
    rr = r - r.mean()
    denom = np.sqrt((rr ** 2).sum() * (sq ** 2).sum())
    ks = np.arange(1, lags + 1)
    cc = [float((sq[k:] * rr[:-k]).sum() / denom * np.sqrt(len(r))) for k in ks]
    # the *normalised* cross-corr (so the band is the usual 2/sqrt(n))
    cc = [c / np.sqrt(len(r)) for c in cc]
    band = 2.0 / np.sqrt(len(r))
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.bar(ks, cc, color="#c44e52", width=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(band, ls="--", lw=0.8, color="#888888")
    ax.axhline(-band, ls="--", lw=0.8, color="#888888")
    ax.set_title("Leverage — cross-correlation  corr(r²ₜ, rₜ₋ₖ)")
    ax.set_xlabel("lag k"); ax.margins(x=0.02)
    fig.tight_layout()
    return fig


def vol_overlay_fig(r: np.ndarray, sigma: np.ndarray) -> plt.Figure:
    """Returns with the fitted ±2σₜ conditional-volatility envelope."""
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(r, lw=0.5, color="#999999", label="returns")
    ax.plot(2 * sigma, lw=1.0, color="#c44e52", label="±2σₜ (conditional vol)")
    ax.plot(-2 * sigma, lw=1.0, color="#c44e52")
    ax.set_title("Fitted conditional volatility σₜ")
    ax.margins(x=0); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def fit_overlay_fig(actual: np.ndarray, fitted: np.ndarray, label: str) -> plt.Figure:
    """Actual vs fitted series (realised-vol models: RV and its HAR/ARMA fit)."""
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(actual, lw=0.7, color="#4c72b0", label="actual")
    ax.plot(fitted, lw=0.9, color="#dd8452", alpha=0.9, label=label)
    ax.set_title(label); ax.margins(x=0); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Explorer helpers
# ---------------------------------------------------------------------------


def pnl_hist_fig(pnl: np.ndarray, var: float, etl: float | None = None,
                 alpha: float = 0.01, title: str = "P&L distribution") -> plt.Figure:
    """Histogram of P&L with VaR (and optional ETL) marked on the loss tail."""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(pnl, bins=80, color="#4c72b0", alpha=0.85)
    ax.axvline(-var, color="#c44e52", lw=2,
               label=f"VaR{int((1-alpha)*100)} = {var:.2f}")
    if etl is not None:
        ax.axvline(-etl, color="#8172b3", lw=2, ls="--", label=f"ETL = {etl:.2f}")
    ax.fill_betweenx(ax.get_ylim(), ax.get_xlim()[0], -var, color="#c44e52", alpha=0.08)
    ax.set_xlabel("P&L")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.margins(x=0)
    return _finish(fig)


def corr_path_fig(rho, labels=None, title: str = "Time-varying correlation") -> plt.Figure:
    """One or more correlation series over time (e.g. a DCC path)."""
    rho = np.asarray(rho)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    if rho.ndim == 1:
        ax.plot(rho, lw=0.8, color="#4c72b0")
    else:
        for j in range(rho.shape[1]):
            lbl = labels[j] if labels else None
            ax.plot(rho[:, j], lw=0.8, label=lbl)
        if labels:
            ax.legend(fontsize=8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(r"$\rho_t$")
    ax.set_title(title)
    ax.margins(x=0)
    return _finish(fig)


def copula_scatter_fig(u: np.ndarray, v: np.ndarray, title: str = "Copula sample",
                       show_tail: bool = True) -> plt.Figure:
    """Uniform-margin scatter, optionally annotating the upper/lower tail boxes."""
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(u, v, s=4, alpha=0.25, color="#4c72b0")
    if show_tail:
        for lo in (0.95, 0.05):
            x0 = lo if lo > 0.5 else 0.0
            ax.axvspan(x0, x0 + 0.05, color="#c44e52", alpha=0.06)
            ax.axhspan(x0, x0 + 0.05, color="#c44e52", alpha=0.06)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(title)
    return _finish(fig)


def heatmap_fig(mat: np.ndarray, labels=None, title: str = "Matrix",
                annot: bool = True) -> plt.Figure:
    """Correlation / covariance matrix heatmap."""
    mat = np.asarray(mat)
    k = mat.shape[0]
    fig, ax = plt.subplots(figsize=(0.8 * k + 2, 0.8 * k + 1.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if labels:
        ax.set_xticks(range(k), labels, rotation=45, ha="right")
        ax.set_yticks(range(k), labels)
    if annot:
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title(title)
    return _finish(fig)


def regime_series_fig(y: np.ndarray, states: np.ndarray,
                      title: str = "Regime-switching series") -> plt.Figure:
    """Series line with the background shaded by latent state."""
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(y, lw=0.6, color="#333333")
    n = len(y)
    k = int(states.max()) + 1
    cmap = plt.get_cmap("Set2")
    # shade contiguous runs of each state
    start = 0
    for t in range(1, n + 1):
        if t == n or states[t] != states[start]:
            ax.axvspan(start, t, color=cmap(states[start] % 8), alpha=0.25)
            start = t
    ax.set_title(title + f"  ({k} regimes)")
    ax.margins(x=0)
    return _finish(fig)


def scree_fig(eigvals: np.ndarray, title: str = "PCA scree") -> plt.Figure:
    """Explained-variance bars + cumulative line."""
    eigvals = np.asarray(eigvals, dtype=float)
    ratio = eigvals / eigvals.sum()
    cum = np.cumsum(ratio)
    x = np.arange(1, len(eigvals) + 1)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x, ratio, color="#4c72b0", alpha=0.85, label="explained")
    ax.plot(x, cum, "-o", color="#c44e52", label="cumulative")
    ax.set_xlabel("component")
    ax.set_ylabel("variance ratio")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _finish(fig)


def evt_diag_fig(losses: np.ndarray, threshold: float, xi: float, beta: float,
                 title: str = "EVT (peaks over threshold)") -> plt.Figure:
    """Two panels: mean-excess plot and GPD tail fit vs empirical exceedances."""
    from scipy import stats

    losses = np.sort(losses)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))

    # mean-excess function e(u) over a grid of thresholds
    grid = np.quantile(losses, np.linspace(0.5, 0.99, 40))
    me = [losses[losses > u].mean() - u if np.any(losses > u) else np.nan for u in grid]
    axes[0].plot(grid, me, ".-", color="#4c72b0")
    axes[0].axvline(threshold, color="#c44e52", lw=1.5, ls="--")
    axes[0].set_title("Mean-excess (linear ⇒ GPD)")
    axes[0].set_xlabel("threshold u")
    axes[0].set_ylabel("mean excess")

    # tail: empirical exceedance prob vs fitted GPD survival
    exc = losses[losses > threshold] - threshold
    xs = np.linspace(0, exc.max(), 100)
    emp = [np.mean(exc > x) for x in xs]
    fit = stats.genpareto.sf(xs, xi, loc=0, scale=beta)
    axes[1].plot(xs, emp, color="#4c72b0", label="empirical")
    axes[1].plot(xs, fit, color="#c44e52", ls="--", label=f"GPD ξ={xi:.2f}")
    axes[1].set_yscale("log")
    axes[1].set_title("Tail: exceedance probability")
    axes[1].set_xlabel("excess over u")
    axes[1].legend(fontsize=8)
    fig.suptitle(title, y=1.02, fontsize=10)
    return _finish(fig)


def midas_weights_fig(weights_by_scheme: dict, title: str = "MIDAS weights") -> plt.Figure:
    """Weight-vs-lag curves for one or more weighting schemes on shared axes."""
    fig, ax = plt.subplots(figsize=(7, 3))
    for name, w in weights_by_scheme.items():
        ax.plot(np.arange(1, len(w) + 1), w, "-o", ms=3, label=name)
    ax.set_xlabel("lag")
    ax.set_ylabel("weight")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.margins(x=0.01)
    return _finish(fig)

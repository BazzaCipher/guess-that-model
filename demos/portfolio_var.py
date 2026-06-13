"""Portfolio VaR demos — all over a shared 3-asset simulated portfolio."""
from __future__ import annotations

import numpy as np
import streamlit as st
from scipy import stats

import generators as gen
import plots

NOTIONAL = 100.0
ASSETS = ["A", "B", "C"]
_COV = np.array([[1.0, 0.5, 0.2],
                 [0.5, 1.0, 0.3],
                 [0.2, 0.3, 1.0]]) * 0.01      # daily covariance (~10%/day vols scaled)
_W = np.array([0.5, 0.3, 0.2])


@st.cache_data(show_spinner=False)
def _returns(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(_COV)
    return rng.standard_normal((n, 3)) @ L.T


@st.cache_data(show_spinner=False)
def _garch_mc(n: int, seed: int) -> np.ndarray:
    from arch.univariate import GARCH, Normal, ZeroMean
    rng = np.random.default_rng(seed)
    m = ZeroMean(volatility=GARCH(p=1, q=1), distribution=Normal(seed=rng))
    return m.simulate([0.02, 0.08, 0.90], nobs=n, burn=500)["data"].to_numpy()


def _alpha_widget(key: str) -> float:
    conf = st.select_slider("Confidence", [0.90, 0.95, 0.99], 0.99, key=key)
    return 1 - conf


def demo_delta_normal() -> None:
    st.caption("**Delta-normal** (variance–covariance): assume linear positions and normal "
               "returns, so VaR is closed-form  VaR = z_α · σ_p · notional.")
    alpha = _alpha_widget("dn_a")
    sigma_p = np.sqrt(_W @ _COV @ _W) * NOTIONAL
    z = stats.norm.ppf(1 - alpha)
    var = z * sigma_p
    pnl = (_returns(5000, 0) @ _W) * NOTIONAL
    st.pyplot(plots.pnl_hist_fig(pnl, var, alpha=alpha,
                                 title=f"Delta-normal VaR = {var:.2f}"),
              clear_figure=True, width="stretch")
    st.caption(f"σ_p = {sigma_p:.2f}, z = {z:.2f}. Fast and analytical, but wrong for "
               "options (non-linearity) and fat tails.")


def demo_delta_gamma() -> None:
    st.caption("**Delta-gamma**: add a second-order term δΔS + ½γΔS² for non-linear (option) "
               "payoffs — the P&L distribution becomes skewed.")
    gamma = st.slider("Gamma (option convexity)", -2.0, 2.0, 1.0, 0.25, key="dg_g")
    alpha = _alpha_widget("dg_a")
    r = _returns(20000, 1)
    dS = r[:, 0] * NOTIONAL                                   # underlying move
    pnl = (r @ _W) * NOTIONAL + 0.5 * gamma * (dS / NOTIONAL) ** 2 * NOTIONAL
    var = -np.quantile(pnl, alpha)
    lin_var = stats.norm.ppf(1 - alpha) * np.sqrt(_W @ _COV @ _W) * NOTIONAL
    st.pyplot(plots.pnl_hist_fig(pnl, var, alpha=alpha,
                                 title=f"Delta-gamma VaR = {var:.2f}"),
              clear_figure=True, width="stretch")
    st.caption(f"Delta-normal would report {lin_var:.2f}; positive gamma fattens one tail and "
               "thins the other, so the linear number misprices the risk.")


def demo_historical() -> None:
    st.caption("**Historical simulation**: no distribution — just the empirical quantile of "
               "past P&L. Bootstrap gives a confidence band, but you're trapped in the sample.")
    alpha = _alpha_widget("hs_a")
    pnl = (_returns(1500, 2) @ _W) * NOTIONAL
    var = -np.quantile(pnl, alpha)
    rng = np.random.default_rng(7)
    boots = [-np.quantile(rng.choice(pnl, len(pnl), replace=True), alpha) for _ in range(400)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    st.pyplot(plots.pnl_hist_fig(pnl, var, alpha=alpha,
                                 title=f"Historical VaR = {var:.2f}"),
              clear_figure=True, width="stretch")
    st.caption(f"Bootstrap 95% CI for VaR: [{lo:.2f}, {hi:.2f}] — sampling error shrinks with "
               "more history, but the window can't see unseen regimes.")


def demo_monte_carlo() -> None:
    st.caption("**Monte Carlo**: simulate forward P&L paths. Constant-volatility vs a GARCH "
               "engine give different tail risk — GARCH captures volatility clustering.")
    alpha = _alpha_widget("mc_a")
    # constant-vol MC
    sigma_p = np.sqrt(_W @ _COV @ _W)
    rng = np.random.default_rng(3)
    pnl_const = rng.normal(0, sigma_p, 10000) * NOTIONAL
    var_const = -np.quantile(pnl_const, alpha)
    # GARCH MC on the portfolio return
    g = _garch_mc(10000, 3)
    g = g / g.std() * sigma_p
    pnl_garch = g * NOTIONAL
    var_garch = -np.quantile(pnl_garch, alpha)
    c1, c2 = st.columns(2)
    c1.metric("Const-vol MC VaR", f"{var_const:.2f}")
    c2.metric("GARCH MC VaR", f"{var_garch:.2f}", delta=f"{var_garch - var_const:+.2f}")
    st.pyplot(plots.pnl_hist_fig(pnl_garch, var_garch, alpha=alpha,
                                 title="GARCH Monte-Carlo P&L"),
              clear_figure=True, width="stretch")


def demo_component_var() -> None:
    st.caption("**Marginal / component VaR** decompose portfolio VaR by asset: "
               "MVaR_i = ∂VaR/∂w_i, CVaR_i = w_i·MVaR_i, and the components sum to total VaR.")
    alpha = _alpha_widget("cv_a")
    z = stats.norm.ppf(1 - alpha)
    sigma_p = np.sqrt(_W @ _COV @ _W)
    var = z * sigma_p * NOTIONAL
    mvar = z * (_COV @ _W) / sigma_p * NOTIONAL              # marginal VaR per unit weight
    cvar = _W * mvar                                          # component VaR (sums to VaR)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(ASSETS, cvar, color="#4c72b0")
    ax.set_ylabel("Component VaR")
    ax.set_title(f"Total VaR {var:.2f} = Σ components ({cvar.sum():.2f})")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width="stretch")
    st.caption("Component VaR shows which holding actually drives the risk — the basis for "
               "incremental-VaR limits and risk budgeting.")


def demo_etl() -> None:
    st.caption("**Expected shortfall / ETL**: the average loss *beyond* VaR — a coherent tail "
               "measure that, unlike VaR, sees how bad the tail really is.")
    alpha = _alpha_widget("es_a")
    pnl = (_returns(20000, 4) @ _W) * NOTIONAL
    var = -np.quantile(pnl, alpha)
    etl = -pnl[pnl <= -var].mean()
    st.pyplot(plots.pnl_hist_fig(pnl, var, etl=etl, alpha=alpha,
                                 title=f"VaR {var:.2f} · ETL {etl:.2f}"),
              clear_figure=True, width="stretch")
    st.caption(f"ETL ({etl:.2f}) always exceeds VaR ({var:.2f}); it's sub-additive, so "
               "diversification never looks like it increases risk.")


def demo_mapping_pca() -> None:
    st.caption("**Mapping / PCA**: reduce many correlated assets to a few risk factors. PCA "
               "rotates the covariance to orthogonal components; orthogonal-GARCH then runs a "
               "univariate GARCH on each.")
    corr = _COV / np.sqrt(np.outer(np.diag(_COV), np.diag(_COV)))
    vals, vecs = gen.pca(corr)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plots.scree_fig(vals, title="Scree — factor importance"),
                  clear_figure=True, width="stretch")
    with c2:
        recon = vals[0] * np.outer(vecs[:, 0], vecs[:, 0])   # rank-1 reconstruction
        st.pyplot(plots.heatmap_fig(recon, labels=ASSETS,
                                    title="Rank-1 (top PC) cov"),
                  clear_figure=True, width="stretch")
    st.caption(f"The first component explains {vals[0]/vals.sum():.0%} of variance — a single "
               "'market' factor, the single-index mapping in action.")

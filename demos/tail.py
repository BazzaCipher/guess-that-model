"""Tail & extreme-risk demos."""
from __future__ import annotations

import numpy as np
import streamlit as st
from scipy import stats

import generators as gen
import plots


@st.cache_data(show_spinner=False)
def _heavy_losses(n: int, df: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_t(df, n)


def demo_evt_gpd() -> None:
    st.caption("**EVT — peaks over threshold**: fit a Generalised Pareto to exceedances above "
               "a high threshold. The shape ξ governs how heavy the tail is.")
    cols = st.columns(2)
    df = cols[0].select_slider("Tail heaviness (t d.o.f.)", [3, 4, 6, 10], 4, key="evt_df")
    q = cols[1].slider("Threshold quantile", 0.90, 0.99, 0.95, 0.01, key="evt_q")
    losses = _heavy_losses(40000, df, 0)
    u, xi, beta, exc = gen.gpd_fit(losses, q)
    st.pyplot(plots.evt_diag_fig(losses, u, xi, beta),
              clear_figure=True, width="stretch")
    # GPD tail VaR/ES at 99.5%
    alpha = 0.005
    nu = (1 - q)
    var = u + beta / xi * ((alpha / nu) ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi)
    c1, c2, c3 = st.columns(3)
    c1.metric("ξ (shape)", f"{xi:.3f}")
    c2.metric("VaR 99.5%", f"{var:.2f}")
    c3.metric("ES 99.5%", f"{es:.2f}")
    st.caption(f"True tail index ≈ 1/df = {1/df:.2f}; ξ>0 means a power-law (fat) tail. EVT "
               "extrapolates *beyond* the largest observed loss, which history can't.")


def demo_mcneil_frey() -> None:
    st.caption("**McNeil-Frey**: GARCH first to standardise returns, then EVT on the "
               "standardised tail. Combines volatility clustering with a proper tail model "
               "→ a conditional VaR that breathes with the market.")
    from arch.univariate import GARCH, Normal, ZeroMean
    rng = np.random.default_rng(1)
    n = 4000
    m = ZeroMean(volatility=GARCH(p=1, q=1), distribution=Normal(seed=rng))
    sim = m.simulate([0.05, 0.10, 0.88], nobs=n, burn=500)
    r = sim["data"].to_numpy()
    sigma = np.sqrt(sim["volatility"].to_numpy())
    z = r / sigma                                            # standardised residuals
    # EVT on the lower tail of z (losses = -z)
    u, xi, beta, _ = gen.gpd_fit(-z, 0.95)
    alpha, nu = 0.01, 0.05
    zq = u + beta / xi * ((alpha / nu) ** (-xi) - 1)         # standardised VaR quantile
    cond_var = sigma * zq                                    # conditional VaR path
    uncond_var = -np.quantile(r, alpha)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(r, lw=0.4, color="#999999", label="returns")
    ax.plot(-cond_var, lw=1.0, color="#c44e52", label="McNeil-Frey VaR (cond.)")
    ax.axhline(-uncond_var, lw=1.0, ls="--", color="#4c72b0", label="unconditional VaR")
    ax.legend(fontsize=8, loc="lower left")
    ax.margins(x=0)
    ax.set_title("Conditional EVT VaR tracks volatility")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width="stretch")
    st.caption("The conditional VaR widens in turbulent stretches and tightens in calm ones; "
               "a static historical VaR can't adapt and gets the timing of breaches wrong.")


def demo_copulas() -> None:
    st.caption("**Copulas** separate the dependence structure from the margins. The choice of "
               "family decides how assets crash *together*.")
    cols = st.columns(2)
    family = cols[0].selectbox("Family", ["Gaussian", "Student-t", "Gumbel"], key="cop_fam")
    if family == "Gumbel":
        param = cols[1].slider("θ (1 = independent)", 1.0, 4.0, 2.0, 0.25, key="cop_th")
        df = 4
    else:
        param = cols[1].slider("ρ", -0.9, 0.9, 0.6, 0.05, key="cop_rho")
        df = st.slider("t d.o.f.", 2, 15, 4, key="cop_df") if family == "Student-t" else 4
    rng = np.random.default_rng(0)
    u, v = gen.copula_sample(family, param, 4000, rng, df=df)
    # empirical upper/lower tail co-exceedance (λ proxies)
    lam_u = np.mean((u > 0.95) & (v > 0.95)) / 0.05
    lam_l = np.mean((u < 0.05) & (v < 0.05)) / 0.05
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(plots.copula_scatter_fig(u, v, title=f"{family} copula"),
                  clear_figure=True, width="stretch")
    with c2:
        st.metric("Upper-tail co-exceedance", f"{lam_u:.2f}")
        st.metric("Lower-tail co-exceedance", f"{lam_l:.2f}")
    note = {
        "Gaussian": "Gaussian has **zero** asymptotic tail dependence — joint crashes are "
                    "understated however high ρ is.",
        "Student-t": "Student-t has **symmetric** tail dependence (both corners) that grows as "
                     "d.o.f. falls.",
        "Gumbel": "Gumbel has **upper-tail** dependence only — assets boom together more than "
                  "they crash together.",
    }[family]
    st.caption(note)

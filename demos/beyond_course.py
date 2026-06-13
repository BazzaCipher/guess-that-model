"""Beyond-course demos — lighter, clearly-labelled treatments."""
from __future__ import annotations

import numpy as np
import streamlit as st

import generators as gen
import plots


@st.cache_data(show_spinner=False)
def _garch_returns(n: int, seed: int) -> np.ndarray:
    from arch.univariate import GARCH, Normal, ZeroMean
    rng = np.random.default_rng(seed)
    m = ZeroMean(volatility=GARCH(p=1, q=1), distribution=Normal(seed=rng))
    return m.simulate([0.02, 0.08, 0.90], nobs=n, burn=500)["data"].to_numpy()


def demo_midas_garch() -> None:
    st.caption("**GARCH-MIDAS** (Engle-Ghysels-Sohn) splits variance into a slow **long-run** "
               "component τ (MIDAS-weighted realised variance) and a fast **short-run** GARCH "
               "component g, so σ²_t = τ_t · g_t.")
    decay = st.slider("Long-run MIDAS decay (exp-Almon p₂)", -0.10, -0.01, -0.04, 0.01,
                      key="mg_decay")
    n = 1500
    r0 = _garch_returns(n, 0)
    sq = r0 ** 2
    K = 66
    w = gen.midas_weights("Exponential Almon", K, p2=decay)
    tau = np.full(n, sq.mean())
    for t in range(K, n):
        tau[t] = float(np.dot(w, sq[t - K:t][::-1]))
    tau = np.maximum(tau, 1e-6)
    g = sq / tau
    g = g / g.mean()
    total_vol = np.sqrt(tau * g)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(total_vol, lw=0.5, color="#999999", label="total σ_t")
    ax.plot(np.sqrt(tau), lw=1.6, color="#c44e52", label="long-run √τ (secular)")
    ax.legend(fontsize=8, loc="upper left")
    ax.margins(x=0)
    ax.set_title("GARCH-MIDAS: secular trend + transitory clustering")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width="stretch")
    st.caption("The red secular curve is the MIDAS long-run level; daily volatility oscillates "
               "around it via the GARCH short-run component. Heavier decay → a smoother trend.")


def demo_vine_copula() -> None:
    st.caption("**Vine copulas** build a high-dimensional dependence structure as a cascade of "
               "*bivariate* pair-copulas arranged in trees — each edge can be a different "
               "family, so tail behaviour is modelled pair by pair.")
    st.markdown(
        "For 3 variables a D-vine factors the joint density as\n"
        "$c_{123} = c_{12}\\cdot c_{23}\\cdot c_{13\\mid 2}$ — two unconditional edges "
        "(tree 1) and one conditional edge (tree 2)."
    )
    rng = np.random.default_rng(0)
    u1, u2 = gen.copula_sample("Gumbel", 2.0, 3000, rng)        # tree-1 edge: upper-tail
    _, u3 = gen.copula_sample("Gaussian", -0.5, 3000, rng)      # tree-2 edge: symmetric
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plots.copula_scatter_fig(u1, u2, title="edge (1,2): Gumbel"),
                  clear_figure=True, width="stretch")
    with c2:
        st.pyplot(plots.copula_scatter_fig(u2, u3, title="edge (2,3): Gaussian"),
                  clear_figure=True, width="stretch")
    st.caption("Mixing families per edge is the whole point: the (1,2) pair crashes/booms "
               "together (Gumbel upper tail) while (2,3) is symmetric (Gaussian).")


def demo_factor_copula() -> None:
    st.caption("**Oh-Patton factor copulas** drive dependence through a few common latent "
               "factors: each variable loads on a shared shock plus idiosyncratic noise — "
               "scalable to many assets, with tail dependence from a fat-tailed factor.")
    a = st.slider("Common-factor loading", 0.0, 0.95, 0.6, 0.05, key="fc_a")
    df = st.slider("Factor tail (t d.o.f.)", 2, 15, 4, key="fc_df")
    rng = np.random.default_rng(1)
    n = 3000
    Z = rng.standard_t(df, n)
    x1 = a * Z + np.sqrt(1 - a ** 2) * rng.standard_t(df, n)
    x2 = a * Z + np.sqrt(1 - a ** 2) * rng.standard_t(df, n)
    from scipy import stats
    u = stats.rankdata(x1) / (n + 1)
    v = stats.rankdata(x2) / (n + 1)
    lam_u = np.mean((u > 0.95) & (v > 0.95)) / 0.05
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(plots.copula_scatter_fig(u, v, title="Factor-copula dependence"),
                  clear_figure=True, width="stretch")
    with c2:
        st.metric("Implied correlation", f"{np.corrcoef(x1, x2)[0,1]:.2f}")
        st.metric("Upper-tail co-exceedance", f"{lam_u:.2f}")
    st.caption("A higher loading pulls both assets toward the common factor; a fatter-tailed "
               "factor adds joint-crash risk that a Gaussian factor would miss.")

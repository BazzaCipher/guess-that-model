"""Multivariate GARCH demos (2-asset)."""
from __future__ import annotations

import numpy as np
import streamlit as st

import generators as gen
import plots


@st.cache_data(show_spinner=False)
def _std_resid(n: int, seed: int):
    rng = np.random.default_rng(seed)
    r, s = gen.two_garch(rng, n)
    return r, s, r / s


def demo_ccc() -> None:
    st.caption("Two GARCH(1,1) vol paths with a **constant** correlation R. Cheap, but the "
               "assumption that correlation never moves is the weak point.")
    rho = st.slider("Constant correlation ρ", -0.9, 0.9, 0.4, 0.05, key="ccc_rho")
    n = 2000
    _r, sig, _std = _std_resid(n, 0)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(plots.corr_path_fig(np.full(n, rho), title="CCC correlation (constant)"),
                  clear_figure=True, width="stretch")
    with c2:
        R = np.array([[1.0, rho], [rho, 1.0]])
        st.pyplot(plots.heatmap_fig(R, labels=["A", "B"], title="R"),
                  clear_figure=True, width="stretch")


def demo_dcc() -> None:
    st.caption("Engle **DCC**: standardise each asset by its own GARCH, then let the "
               "correlation evolve via Q_t = (1−a−b)Q̄ + a·u u' + b·Q_{t-1}.")
    cols = st.columns(2)
    a = cols[0].slider("a — news impact", 0.0, 0.20, 0.04, 0.01, key="dcc_a")
    b = cols[1].slider("b — persistence", 0.0, 0.98, 0.93, 0.01, key="dcc_b")
    if a + b >= 1:
        st.warning(f"Need a+b < 1 for stationarity — capping b at {0.99 - a:.2f}.")
        b = max(0.0, 0.99 - a)
    n = 2500
    _r, _s, std = _std_resid(n, 0)
    rho = gen.dcc_path(std, a, b)
    st.pyplot(plots.corr_path_fig(rho, title=f"DCC ρ_t  (a={a:.2f}, b={b:.2f})"),
              clear_figure=True, width="stretch")
    st.caption(f"Mean ρ ≈ {rho.mean():.2f}, range [{rho.min():.2f}, {rho.max():.2f}]. "
               "Raise **a** for a jumpier correlation; raise **b** for smoother, more "
               "persistent swings.")


def demo_bekk() -> None:
    st.caption("Scalar **BEKK**(1,1): Σ_t = CC' + a²·εε' + b²·Σ_{t-1}. The quadratic form "
               "keeps Σ_t positive-definite by construction (no constraints needed).")
    cols = st.columns(2)
    a = cols[0].slider("a", 0.05, 0.5, 0.25, 0.05, key="bekk_a")
    b = cols[1].slider("b", 0.5, 0.97, 0.93, 0.01, key="bekk_b")
    if a ** 2 + b ** 2 >= 1:
        st.warning("Need a²+b² < 1; reducing b.")
        b = float(np.sqrt(max(0.0, 0.98 - a ** 2)))
    rho, _rets = gen.bekk_path(np.random.default_rng(0), 2500, a=a, b=b)
    st.pyplot(plots.corr_path_fig(rho, title=f"BEKK ρ_t  (a={a:.2f}, b={b:.2f})"),
              clear_figure=True, width="stretch")


def demo_vech() -> None:
    st.caption("**vech / diagonal vech** — the most general MGARCH parameterises every "
               "element of Σ_t. The parameter count explodes and Σ_t is rarely positive-"
               "definite, which is why CCC/DCC/BEKK exist.")
    k = st.slider("Number of assets N", 2, 30, 5, key="vech_k")
    m = k * (k + 1) // 2                       # unique covariance terms
    full = (m * (m + 1))                       # vech VECH(1,1): A,B are m×m
    diag = 2 * m + m                           # diagonal-vech rough count
    st.markdown(
        f"- Unique variance/covariance terms per period: **{m}**\n"
        f"- Full vech(1,1) parameters: **≈ {full}** (A and B are {m}×{m})\n"
        f"- Diagonal-vech parameters: **≈ {diag}** — tractable but ignores cross-effects"
    )
    st.caption("Full vech is unusable beyond a handful of assets; diagonal vech tames the "
               "count but at the cost of cross-asset spillovers.")


def demo_realised_cov() -> None:
    st.caption("**Realised covariance** treats the daily covariance matrix as observable "
               "(from intraday data) and models it directly — e.g. **MHAR** (multivariate "
               "HAR) regresses each element on its day/week/month averages.")
    rng = np.random.default_rng(1)
    n = 750
    rho_true, rets = gen.bekk_path(rng, n, a=0.2, b=0.94)
    # 5-day rolling 'realised' correlation as an observable proxy
    w = 5
    rc = np.array([np.corrcoef(rets[max(0, t - w):t + 1].T)[0, 1] if t >= 2 else np.nan
                   for t in range(n)])
    st.pyplot(plots.corr_path_fig(np.column_stack([rho_true, rc]),
                                  labels=["latent ρ_t", "5-day realised ρ"],
                                  title="Realised vs latent correlation"),
              clear_figure=True, width="stretch")
    st.caption("MHAR/VARFIMA fit ARMA/long-memory dynamics to the realised series instead of "
               "a latent recursion.")


def demo_cholesky() -> None:
    st.caption("**Cholesky** Σ = LL' is the workhorse for *generating* correlated draws and "
               "for *enforcing* positive-definiteness everywhere in MGARCH and copulas.")
    rho = st.slider("Target correlation ρ", -0.95, 0.95, 0.6, 0.05, key="chol_rho")
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((3000, 2)) @ L.T
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.2, color="#4c72b0")
    ax.set_title(f"L·z draws  (sample ρ = {np.corrcoef(z.T)[0,1]:.2f})")
    ax.set_xlabel("asset A"); ax.set_ylabel("asset B")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width="stretch")
    st.latex(r"L = \begin{bmatrix}1 & 0\\ \rho & \sqrt{1-\rho^2}\end{bmatrix}")

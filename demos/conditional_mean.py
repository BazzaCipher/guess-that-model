"""Conditional-mean demos (MIDAS, single-index / market model)."""
from __future__ import annotations

import numpy as np
import streamlit as st
from scipy import stats

import generators as gen
import plots


def demo_midas() -> None:
    st.caption("**MIDAS** regresses a low-frequency target on many high-frequency lags, using "
               "a *parsimonious weight function* so you estimate a couple of shape parameters "
               "instead of one coefficient per lag.")
    K = st.slider("High-frequency lags K", 6, 40, 20, key="md_K")
    schemes = st.multiselect("Weight schemes", ["Almon", "Exponential Almon", "Beta"],
                             default=["Exponential Almon", "Beta"], key="md_sch")
    cols = st.columns(3)
    p2 = cols[0].slider("Almon/exp decay (p₂)", -0.20, 0.0, -0.03, 0.01, key="md_p2")
    a = cols[1].slider("Beta a", 1.0, 4.0, 1.0, 0.5, key="md_a")
    b = cols[2].slider("Beta b", 1.0, 8.0, 4.0, 0.5, key="md_b")
    weights = {s: gen.midas_weights(s, K, p1=0.0, p2=p2, a=a, b=b) for s in schemes}
    if weights:
        st.pyplot(plots.midas_weights_fig(weights, title="MIDAS lag weights"),
                  clear_figure=True, width="stretch")
    st.caption("Beta (a=1) decays monotonically; raise **a** for a hump. **Direct** forecasting "
               "fits the horizon you want; **iterated** chains one-step forecasts forward.")


def demo_single_index() -> None:
    st.caption("**Single-index / market model**: regress an asset on one market factor. The "
               "OLS slope is β — reused for mapping, hedging and time-varying CAPM.")
    beta = st.slider("True β", -0.5, 2.0, 1.1, 0.1, key="si_b")
    idio = st.slider("Idiosyncratic vol", 0.2, 2.0, 1.0, 0.1, key="si_e")
    rng = np.random.default_rng(0)
    rm = rng.normal(0, 1.0, 600)
    ri = 0.0 + beta * rm + rng.normal(0, idio, 600)
    res = stats.linregress(rm, ri)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(rm, ri, s=6, alpha=0.3, color="#4c72b0")
    xs = np.array([rm.min(), rm.max()])
    ax.plot(xs, res.intercept + res.slope * xs, color="#c44e52", lw=2)
    ax.set_xlabel("market return"); ax.set_ylabel("asset return")
    ax.set_title(f"β̂ = {res.slope:.2f},  R² = {res.rvalue**2:.2f}")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width="stretch")
    st.caption(f"Estimated β = {res.slope:.2f} (true {beta:.2f}); R² = {res.rvalue**2:.0%} of "
               "the asset's variance is explained by the single market factor.")

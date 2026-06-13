"""Realised-volatility estimator demo.

Shows where the RV series consumed by HAR/ARFIMA actually comes from: sum the
squared intraday returns of a day to estimate that day's integrated variance.
Two teaching points: (1) RV is a *consistent* estimator of integrated variance
as the sampling frequency rises; (2) microstructure noise biases it upward at
the highest frequencies — the classic volatility-signature plot that motivates
sparse (e.g. 5-minute) sampling.
"""
from __future__ import annotations

import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def _intraday(days: int, steps: int, seed: int):
    """Efficient intraday log-price increments with a stochastic daily spot
    variance. Returns (per-step efficient returns [days, steps], the true
    integrated variance per day [days])."""
    rng = np.random.default_rng(seed)
    # daily integrated variance wanders (log-AR(1)) so RV has something to track
    log_iv = np.zeros(days)
    log_iv[0] = np.log(1.0)
    for d in range(1, days):
        log_iv[d] = 0.98 * log_iv[d - 1] + 0.15 * rng.standard_normal()
    iv = np.exp(log_iv)                                   # integrated variance / day
    dt = 1.0 / steps
    # constant spot vol within a day; sum of squared increments -> iv in the limit
    incr = np.sqrt(iv[:, None] * dt) * rng.standard_normal((days, steps))
    return incr, iv


def demo_rv_estimator() -> None:
    st.caption("**Realised volatility** estimates a day's *integrated variance* by summing its "
               "squared intraday returns:  $RV_t=\\sum_{j} r_{t,j}^2$.  Finer sampling → less "
               "estimation error, until **microstructure noise** takes over and inflates it.")

    c = st.columns(3)
    steps = c[0].select_slider("Intraday returns / day (max grid)",
                               [12, 24, 48, 78, 156, 390], 78, key="rv_steps")
    sample = c[1].select_slider("Sampling — returns used / day",
                                [4, 12, 24, 48, 78, 156, 390], 24, key="rv_sample")
    noise = c[2].slider("Microstructure noise (×10⁻³)", 0.0, 5.0, 0.0, 0.5, key="rv_noise")
    sample = min(sample, steps)

    days = 400
    incr, iv = _intraday(days, steps, 0)

    # observed price = efficient + i.i.d. noise in the log price; differencing it
    # injects the noise into every return (the source of the high-frequency bias)
    omega = noise * 1e-3
    rng = np.random.default_rng(1)
    u = omega * rng.standard_normal((days, steps + 1))
    # subsample to `sample` returns/day, then RV = sum of squared sampled returns
    cum = np.concatenate([np.zeros((days, 1)), np.cumsum(incr, axis=1)], axis=1)
    obs = cum + u
    idx = np.linspace(0, steps, sample + 1).round().astype(int)
    sampled_ret = np.diff(obs[:, idx], axis=1)
    rv = (sampled_ret ** 2).sum(axis=1)

    import matplotlib.pyplot as plt

    # --- RV tracks the true integrated variance ---
    fig1, ax1 = plt.subplots(figsize=(9, 2.6))
    ax1.plot(iv, lw=1.2, color="#4c72b0", label="true integrated variance")
    ax1.plot(rv, lw=0.8, color="#c44e52", alpha=0.8, label=f"RV ({sample} returns/day)")
    ax1.set_title("Realised variance tracks the latent integrated variance")
    ax1.set_xlabel("day"); ax1.margins(x=0); ax1.legend(fontsize=8, loc="upper right")
    fig1.tight_layout()
    st.pyplot(fig1, clear_figure=True, width="stretch")

    bias = 100 * (rv.mean() / iv.mean() - 1)
    m1, m2, m3 = st.columns(3)
    m1.metric("mean RV", f"{rv.mean():.3f}")
    m2.metric("mean integrated var", f"{iv.mean():.3f}")
    m3.metric("RV bias", f"{bias:+.1f}%")

    # --- volatility signature plot: mean RV vs sampling frequency ---
    freqs = [f for f in (4, 8, 12, 24, 48, 78, 156, 390) if f <= steps]
    sig = []
    for f in freqs:
        ix = np.linspace(0, steps, f + 1).round().astype(int)
        sr = np.diff(obs[:, ix], axis=1)
        sig.append((sr ** 2).sum(axis=1).mean())
    fig2, ax2 = plt.subplots(figsize=(9, 2.4))
    ax2.plot(freqs, sig, "o-", color="#55a868", lw=1.2)
    ax2.axhline(iv.mean(), ls="--", lw=1.0, color="#4c72b0", label="true mean IV")
    ax2.set_xscale("log")
    ax2.set_title("Volatility signature plot — mean RV vs sampling frequency")
    ax2.set_xlabel("returns sampled per day (log scale)")
    ax2.legend(fontsize=8); fig2.tight_layout()
    st.pyplot(fig2, clear_figure=True, width="stretch")

    if omega > 0:
        st.caption("With microstructure noise the signature plot **slopes up** toward high "
                   "frequencies — each return carries ~2ω² of noise and RV sums ~2Mω² of bias. "
                   "This is why practitioners sample sparsely (≈5-minute) rather than tick-by-tick.")
    else:
        st.caption("Noise-free, RV converges to the true integrated variance as sampling "
                   "rises (the signature plot is flat). Turn up the noise to see the bias appear.")

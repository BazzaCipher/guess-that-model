"""Structural-instability demos."""
from __future__ import annotations

import numpy as np
import streamlit as st

import generators as gen
import plots


def demo_markov() -> None:
    st.caption("**Markov regime-switching**: a hidden two-state chain flips the volatility. "
               "Persistent regimes produce volatility clustering without any GARCH recursion.")
    cols = st.columns(2)
    p_stay = cols[0].slider("Regime persistence (diagonal of P)", 0.80, 0.995, 0.97, 0.005,
                            key="mk_p")
    sig_hi = cols[1].slider("High-vol σ", 1.5, 4.0, 2.5, 0.25, key="mk_s")
    P = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
    y, states = gen.markov_sim(P, mus=[0.0, 0.0], sigmas=[0.7, sig_hi], T=1200,
                               rng=np.random.default_rng(0))
    st.pyplot(plots.regime_series_fig(y, states, title="Hidden volatility regimes"),
              clear_figure=True, width="stretch")
    avg_dur = 1 / (1 - p_stay)
    st.caption(f"Expected regime duration ≈ {avg_dur:.0f} periods. Lower persistence → rapid "
               "flicker (looks like noise); higher → long calm/turbulent epochs.")


@st.cache_data(show_spinner=False)
def _breaks_series(n: int, seed: int):
    rng = np.random.default_rng(seed)
    true_breaks = np.array([n // 4, n // 2, 3 * n // 4])
    means = np.array([0.0, 2.0, -1.0, 1.0])
    seg = np.searchsorted(true_breaks, np.arange(n), side="right")
    y = means[seg] + rng.normal(0, 1.0, n)
    return y, true_breaks


def _best_partition(y: np.ndarray, m: int, grid: int = 60):
    """Brute-force least-squares partition into m+1 segments (small m)."""
    n = len(y)
    cands = np.linspace(n * 0.1, n * 0.9, grid).astype(int)
    cands = np.unique(cands)

    def ssr(a, b):
        seg = y[a:b]
        return float(((seg - seg.mean()) ** 2).sum()) if b - a > 1 else 0.0

    best = (np.inf, None)
    from itertools import combinations
    for combo in combinations(cands, m):
        pts = [0, *combo, n]
        total = sum(ssr(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        if total < best[0]:
            best = (total, combo)
    return best[1]


def demo_bai_perron() -> None:
    st.caption("**Bai-Perron** estimates *multiple* break dates by minimising total "
               "within-segment sum of squares. Here we plant breaks and recover them.")
    m = st.slider("Number of breaks to fit", 1, 3, 3, key="bp_m")
    y, true_breaks = _breaks_series(600, 0)
    est = _best_partition(y, m)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(y, lw=0.5, color="#999999")
    for b in true_breaks:
        ax.axvline(b, color="#4c72b0", lw=1.2, ls="--")
    for b in (est or []):
        ax.axvline(b, color="#c44e52", lw=1.4)
    ax.plot([], [], color="#4c72b0", ls="--", label="true break")
    ax.plot([], [], color="#c44e52", label="estimated")
    ax.legend(fontsize=8, loc="upper right")
    ax.margins(x=0)
    ax.set_title("Segmented-mean break detection")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width="stretch")
    st.caption(f"True breaks at {list(true_breaks)}; estimated {list(est or [])}. Fitting too "
               "few misses a shift; the real Bai-Perron test also *chooses* m via information "
               "criteria.")

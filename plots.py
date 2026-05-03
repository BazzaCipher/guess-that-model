"""Matplotlib helpers for the trainer app."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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

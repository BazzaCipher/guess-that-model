"""Inventory — a browsable, filterable catalogue of every model.

Each model renders as a card (name · equation · summary · tell).  Models that
carry a ``simulate`` get a collapsed 'Sample a path' expander; models that carry
a ``demo`` get a collapsed 'Interactive demo' expander.  Nothing heavy runs
until an expander is opened.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from models import BY_KEY, by_category, categories
from plots import acf_pacf_fig, series_fig


@st.cache_data(show_spinner=False)
def _sample(key: str, nobs: int, seed: int):
    """Deterministic sample path for a model (cached — seed-fixed)."""
    rng = np.random.default_rng(seed)
    return BY_KEY[key].simulate(rng, nobs=nobs)


def _sample_expander(key: str) -> None:
    with st.expander("Sample a path"):
        res = _sample(key, nobs=2_500, seed=0)
        st.pyplot(series_fig(res.series, title=res.series_label),
                  clear_figure=True, width="stretch")
        target = res.target_sq if res.target_sq is not None else res.target_rv
        label = "squared returns" if res.target_sq is not None else "RV"
        if target is not None:
            st.pyplot(acf_pacf_fig(target, label=label, lags=30),
                      clear_figure=True, width="stretch")


def _card(model) -> None:
    with st.container(border=True):
        head = st.columns([5, 2])
        with head[0]:
            st.markdown(f"### {model.name}")
        with head[1]:
            if model.beyond_course:
                st.markdown(
                    "<span style='float:right;color:#b08900;border:1px solid #b08900;"
                    "border-radius:6px;padding:1px 8px;font-size:0.8em'>beyond course</span>",
                    unsafe_allow_html=True,
                )
            elif model.family:
                st.caption(f"family: {model.family}")
        if model.equation:
            st.latex(model.equation)
        st.markdown(model.summary)
        st.markdown(f"**Tell:** {model.tell}")
        if model.references:
            st.caption(" · ".join(model.references))
        if model.demo is not None:
            with st.expander("Interactive demo"):
                model.demo()
        elif model.simulate is not None:
            _sample_expander(model.key)


def render() -> None:
    st.title("Model inventory")
    st.caption("The FNCE40003 model deck, organised by block. Toggle beyond-course "
               "material and filter by category.")

    cats = categories()
    ctrl = st.columns([3, 2, 2])
    with ctrl[0]:
        chosen = st.multiselect("Categories", cats, default=cats)
    with ctrl[1]:
        query = st.text_input("Search", placeholder="name or description…").strip().lower()
    with ctrl[2]:
        show_bc = st.toggle("Show beyond-course", value=False)

    any_shown = False
    for cat in cats:
        if cat not in chosen:
            continue
        models = by_category(cat)
        if not show_bc:
            models = [m for m in models if not m.beyond_course]
        if query:
            models = [m for m in models
                      if query in m.name.lower() or query in m.summary.lower()]
        if not models:
            continue
        any_shown = True
        st.header(cat)
        for m in models:
            _card(m)

    if not any_shown:
        st.info("No models match the current filters.")

"""Trainer — the original 'guess the model from ACF/PACF' game.

Body migrated almost verbatim from the old single-page ``app.py``; it now draws
from the registry's trainer-eligible models rather than a hand-kept dict.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from models import SIMULATORS, families, random_round
from plots import acf_pacf_fig, series_fig


def _new_round(nobs: int) -> None:
    st.session_state["seed_counter"] += 1
    rng = np.random.default_rng()
    st.session_state["round"] = random_round(rng, nobs=nobs)
    st.session_state["revealed"] = False
    st.session_state["last_guess"] = None


def render() -> None:
    # -- sidebar settings --
    st.sidebar.header("Settings")
    difficulty = st.sidebar.radio(
        "Difficulty",
        ["Easy — guess family", "Hard — guess exact model"],
        index=0,
    )
    view_mode = st.sidebar.radio(
        "ACF/PACF target",
        [
            "Default (raw + squared returns; RV when applicable)",
            "Returns only",
            "Squared returns only",
            "RV only",
            "Auto (canonical for the drawn family)",
        ],
        index=0,
    )
    nobs = st.sidebar.slider("Sample length", min_value=500, max_value=10_000,
                             value=2_500, step=500)
    lags = st.sidebar.slider("Lags shown on ACF/PACF", min_value=10, max_value=60,
                             value=30, step=5)

    st.sidebar.markdown("---")
    if st.sidebar.button("Reset score"):
        st.session_state["correct"] = 0
        st.session_state["attempted"] = 0

    # -- session state --
    if "round" not in st.session_state:
        st.session_state["round"] = None
        st.session_state["revealed"] = False
        st.session_state["correct"] = 0
        st.session_state["attempted"] = 0
        st.session_state["last_guess"] = None
        st.session_state["seed_counter"] = 0

    # -- header / score --
    st.title("ACF / PACF — model guessing trainer")
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        st.metric("Correct", st.session_state["correct"])
    with c2:
        st.metric("Attempted", st.session_state["attempted"])
    with c3:
        if st.session_state["attempted"] > 0:
            pct = 100 * st.session_state["correct"] / st.session_state["attempted"]
            st.metric("Hit rate", f"{pct:.0f}%")

    if st.button("New round", type="primary"):
        _new_round(nobs)

    if st.session_state["round"] is None:
        st.info("Hit **New round** to draw a process.")
        return

    result = st.session_state["round"]

    # -- plots --
    st.subheader("Simulated series")
    st.pyplot(series_fig(result.series, title=result.series_label),
              clear_figure=True, width="stretch")

    st.subheader("ACF / PACF")
    panels: list[tuple[np.ndarray, str]] = []
    returns_available = result.target_sq is not None
    rv_available = result.target_rv is not None

    if view_mode.startswith("Default"):
        if returns_available:
            panels.append((result.series, "returns"))
            panels.append((result.target_sq, "squared returns"))
        if rv_available:
            panels.append((result.target_rv, "RV"))
    elif view_mode == "Returns only":
        if returns_available:
            panels.append((result.series, "returns"))
        else:
            st.warning("This round produced an RV-style series; returns view does not apply.")
    elif view_mode == "Squared returns only":
        if returns_available:
            panels.append((result.target_sq, "squared returns"))
        else:
            st.warning("This round produced an RV-style series; squared returns do not apply.")
    elif view_mode == "RV only":
        if rv_available:
            panels.append((result.target_rv, "RV"))
        else:
            st.warning("This round produced a returns series; RV view does not apply.")
    elif view_mode.startswith("Auto"):
        if returns_available:
            panels.append((result.target_sq, "squared returns"))
        if rv_available:
            panels.append((result.target_rv, "RV"))

    for y, lbl in panels:
        st.pyplot(acf_pacf_fig(y, label=lbl, lags=lags), clear_figure=True,
                  use_container_width=True)

    # -- guess --
    st.subheader("Your guess")
    if difficulty.startswith("Easy"):
        options = families()
    else:
        options = sorted(SIMULATORS.keys())

    choice = st.radio("Pick one:", options, key=f"guess_{st.session_state['seed_counter']}")

    submit_col, _ = st.columns([1, 5])
    with submit_col:
        if st.button("Submit guess", disabled=st.session_state["revealed"]):
            true_label = result.family if difficulty.startswith("Easy") else result.name
            correct = (choice == true_label)
            st.session_state["attempted"] += 1
            if correct:
                st.session_state["correct"] += 1
            st.session_state["last_guess"] = choice
            st.session_state["revealed"] = True

    # -- reveal --
    if st.session_state["revealed"]:
        true_label = result.family if difficulty.startswith("Easy") else result.name
        if st.session_state["last_guess"] == true_label:
            st.success(f"Correct — this was a **{result.name}** ({result.family} family).")
        else:
            st.error(f"Not quite. Truth: **{result.name}** ({result.family} family). "
                     f"Your guess: {st.session_state['last_guess']}.")

        with st.expander("True parameters & giveaway", expanded=True):
            st.write("**Parameters**:", {k: round(v, 4) for k, v in result.params.items()})
            st.write("**Tell:**", result.hint)

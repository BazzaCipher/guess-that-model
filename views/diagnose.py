"""Diagnose — the second Trainer mode.

You get ONE EViews-style output block (a deliberately under-specified fit plus
its residual diagnostics) and nothing else: read it, name the problem the
diagnostics flag, and pick the model that fixes it. Two multiple-choice picks,
then the reveal explains which line gave it away.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from reports import diagnose_round


def _new_case() -> None:
    st.session_state["dx_round"] = diagnose_round(np.random.default_rng())
    st.session_state["dx_revealed"] = False
    st.session_state["dx_n"] = st.session_state.get("dx_n", 0) + 1


def render() -> None:
    if "dx_round" not in st.session_state:
        st.session_state.update(dx_correct=0, dx_attempted=0, dx_last=None)
        _new_case()

    st.sidebar.header("Settings")
    if st.sidebar.button("Reset score"):
        st.session_state["dx_correct"] = 0
        st.session_state["dx_attempted"] = 0

    st.title("Diagnose the output")
    st.caption("Read the EViews output below — that's all you get. Name the problem "
               "the diagnostics flag, then the model that fixes it.")

    c1, c2, c3 = st.columns([1, 1, 4])
    c1.metric("Correct", st.session_state["dx_correct"])
    c2.metric("Attempted", st.session_state["dx_attempted"])
    if st.session_state["dx_attempted"]:
        pct = 100 * st.session_state["dx_correct"] / st.session_state["dx_attempted"]
        c3.metric("Hit rate", f"{pct:.0f}%")

    if st.button("New case", type="primary"):
        _new_case()
    rnd = st.session_state["dx_round"]

    # the ONLY content shown before the guess
    st.code(rnd["text"], language="text")

    n = st.session_state["dx_n"]
    st.subheader("1 · What's the problem?")
    pchoice = st.radio("Diagnosis", rnd["problem_options"],
                       key=f"dx_p_{n}", label_visibility="collapsed")
    st.subheader("2 · Which model fixes it?")
    fchoice = st.radio("Fix", rnd["fix_options"],
                       key=f"dx_f_{n}", label_visibility="collapsed")

    if st.button("Submit", disabled=st.session_state["dx_revealed"]):
        st.session_state["dx_attempted"] += 1
        if pchoice == rnd["problem"] and fchoice == rnd["fix"]:
            st.session_state["dx_correct"] += 1
        st.session_state["dx_last"] = (pchoice, fchoice)
        st.session_state["dx_revealed"] = True

    if st.session_state["dx_revealed"]:
        last_p, last_f = st.session_state["dx_last"]
        p_ok, f_ok = last_p == rnd["problem"], last_f == rnd["fix"]
        (st.success if p_ok else st.error)(
            f"**Problem:** {rnd['problem']}" + ("" if p_ok else f"  · you said: _{last_p}_"))
        (st.success if f_ok else st.error)(
            f"**Fix:** {rnd['fix']}" + ("" if f_ok else f"  · you said: _{last_f}_"))
        st.info(rnd["why"])

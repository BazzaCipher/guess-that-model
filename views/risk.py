"""Risk quiz — the third Play mode.

One scenario with freshly-simulated numbers (a VaR backtest, an MGARCH choice or
a tail-risk reading), one or two multiple-choice picks, then a reveal that points
at the numbers that decide it. Pick a topic in the sidebar or leave it on Mixed.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from risk_quiz import TOPICS, risk_round

_TOPIC_CHOICES = ["Mixed", *TOPICS]


def _new_case() -> None:
    topic = st.session_state.get("rk_topic", "Mixed")
    st.session_state["rk_round"] = risk_round(np.random.default_rng(), topic)
    st.session_state["rk_revealed"] = False
    st.session_state["rk_n"] = st.session_state.get("rk_n", 0) + 1


def render() -> None:
    if "rk_round" not in st.session_state:
        st.session_state.update(rk_correct=0, rk_attempted=0, rk_last=None, rk_topic="Mixed")
        _new_case()

    st.sidebar.header("Settings")
    topic = st.sidebar.radio("Topic", _TOPIC_CHOICES,
                             index=_TOPIC_CHOICES.index(st.session_state["rk_topic"]))
    if topic != st.session_state["rk_topic"]:
        st.session_state["rk_topic"] = topic
        _new_case()
    if st.sidebar.button("Reset score"):
        st.session_state["rk_correct"] = 0
        st.session_state["rk_attempted"] = 0

    st.title("Risk quiz")
    st.caption("VaR backtests, multivariate GARCH and tail risk. The numbers are simulated "
               "fresh each round — read them, then answer.")

    c1, c2, c3 = st.columns([1, 1, 4])
    c1.metric("Correct", st.session_state["rk_correct"])
    c2.metric("Attempted", st.session_state["rk_attempted"])
    if st.session_state["rk_attempted"]:
        pct = 100 * st.session_state["rk_correct"] / st.session_state["rk_attempted"]
        c3.metric("Hit rate", f"{pct:.0f}%")

    if st.button("New case", type="primary"):
        _new_case()
    rnd = st.session_state["rk_round"]

    st.caption(f"Topic — {rnd['topic']}")
    st.code(rnd["text"], language="text")  # the scenario is all you get

    n = st.session_state["rk_n"]
    picks = []
    for i, q in enumerate(rnd["questions"]):
        st.subheader(f"{i + 1} · {q['prompt']}")
        picks.append(st.radio(q["prompt"], q["options"],
                              key=f"rk_{n}_{i}", label_visibility="collapsed"))

    if st.button("Submit", disabled=st.session_state["rk_revealed"]):
        st.session_state["rk_attempted"] += 1
        if all(p == q["answer"] for p, q in zip(picks, rnd["questions"])):
            st.session_state["rk_correct"] += 1
        st.session_state["rk_last"] = picks
        st.session_state["rk_revealed"] = True

    if st.session_state["rk_revealed"]:
        for p, q in zip(st.session_state["rk_last"], rnd["questions"]):
            ok = p == q["answer"]
            (st.success if ok else st.error)(
                f"**{q['answer']}**" + ("" if ok else f"  · you said: _{p}_"))
        st.info(rnd["why"])

"""Trainer — pick a track (mean / volatility / realised vol), then guess which
model produced the series from the ACF/PACF of that track's process.

Each track shows ONE correlogram, of the stochastic process that defines it:
the returns for a mean model, the squared returns for a volatility model, the
realised-variance series for a realised-vol model.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from models import random_round_in, track_models, tracks
from plots import (
    acf_pacf_fig,
    fit_overlay_fig,
    leverage_xcorr_fig,
    series_fig,
    vol_overlay_fig,
)
from reports import diagnostic_tells, estimation_report

def _panels_other(r):
    """Mixed track — show every correlogram that exists for this series."""
    panels = [(r.series, r.series_label)]
    if r.target_sq is not None:
        panels.append((r.target_sq, "squared returns"))
    if r.target_rv is not None:
        panels.append((r.target_rv, "RV"))
    return panels


# track -> function(SimResult) -> list of (process array, correlogram label).
# Each track shows the ACF/PACF of its defining stochastic process; the mixed
# "Both / other" track shows all available panels.
TRACK_VIEW = {
    "Conditional mean": lambda r: [(r.series, r.series_label)],
    "Volatility": lambda r: [(r.target_sq, "squared returns")],
    "Realised volatility": lambda r: [(r.target_rv, "realised variance")],
    "Both / other": _panels_other,
}


def _new_round(track: str, nobs: int) -> None:
    st.session_state["seed_counter"] += 1
    rng = np.random.default_rng()
    st.session_state["round"] = random_round_in(rng, track, nobs=nobs)
    st.session_state["round_track"] = track
    st.session_state["revealed"] = False
    st.session_state["last_guess"] = None


def render() -> None:
    # -- session state --
    if "round" not in st.session_state:
        st.session_state.update(round=None, round_track=None, revealed=False,
                                correct=0, attempted=0, last_guess=None, seed_counter=0)

    # -- sidebar --
    st.sidebar.header("Settings")
    track = st.sidebar.radio("Track — what kind of model?", tracks(), index=0,
                             help="You'll guess among the models of this track only.")
    nobs = st.sidebar.slider("Sample length", 500, 10_000, 2_500, 500)
    lags = st.sidebar.slider("Lags shown on ACF/PACF", 10, 60, 30, 5)
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset score"):
        st.session_state["correct"] = 0
        st.session_state["attempted"] = 0

    # auto-draw a fresh round when the track changes (or on first load)
    if st.session_state["round"] is None or st.session_state["round_track"] != track:
        _new_round(track, nobs)

    # -- header / score --
    st.title("Guess the model")
    st.caption(f"Track: **{track}** — guess which of its {len(track_models(track))} models "
               "produced this series.")
    c1, c2, c3 = st.columns([1, 1, 4])
    c1.metric("Correct", st.session_state["correct"])
    c2.metric("Attempted", st.session_state["attempted"])
    if st.session_state["attempted"] > 0:
        pct = 100 * st.session_state["correct"] / st.session_state["attempted"]
        c3.metric("Hit rate", f"{pct:.0f}%")

    if st.button("New round", type="primary"):
        _new_round(track, nobs)

    result = st.session_state["round"]
    panels = TRACK_VIEW[track](result)

    # -- series path --
    st.subheader("Simulated series")
    st.pyplot(series_fig(result.series, title=result.series_label),
              clear_figure=True, width="stretch")

    # -- ACF / PACF of the track's stochastic process(es) --
    st.subheader("ACF / PACF")
    for proc, label in panels:
        st.pyplot(acf_pacf_fig(proc, label=label, lags=lags),
                  clear_figure=True, width="stretch")

    # -- pre-guess diagnostics: the numbers you need to actually decide --
    # All computed from the observable series (no hidden parameters), so it is
    # fair to show them before guessing. For volatility, the squared-returns
    # ACF/PACF can't reveal leverage (GJR/EGARCH) or fat tails (Student-t) — the
    # cross-correlation and kurtosis below are what separate those look-alikes.
    if track in ("Volatility", "Realised volatility"):
        if track == "Volatility":
            st.subheader("Leverage diagnostic")
            st.pyplot(leverage_xcorr_fig(result.series, lags=min(lags, 15)),
                      clear_figure=True, width="stretch")
        tells = diagnostic_tells(result)
        if tells:
            st.subheader("Diagnostic tells")
            st.table({"value": tells})

    # -- guess (among this track's models only) --
    st.subheader("Your guess")
    options = sorted(m.name for m in track_models(track))
    choice = st.radio("Which model?", options, key=f"guess_{st.session_state['seed_counter']}")

    if st.button("Submit guess", disabled=st.session_state["revealed"]):
        st.session_state["attempted"] += 1
        if choice == result.name:
            st.session_state["correct"] += 1
        st.session_state["last_guess"] = choice
        st.session_state["revealed"] = True

    # -- reveal --
    if st.session_state["revealed"]:
        if st.session_state["last_guess"] == result.name:
            st.success(f"Correct — this was **{result.name}**.")
        else:
            st.error(f"Not quite. Truth: **{result.name}**. "
                     f"Your guess: {st.session_state['last_guess']}.")
        with st.expander("True parameters & giveaway", expanded=True):
            st.write("**Parameters**:", {k: round(v, 4) for k, v in result.params.items()})
            st.write("**Tell:**", result.hint)

        # Volatility & realised-vol rounds: fit the true spec and show what you
        # were really modelling — the conditional volatility σₜ (or fitted RV) —
        # plus an EViews-style estimation table that settles nested look-alikes.
        if track in ("Volatility", "Realised volatility"):
            est = estimation_report(result)
            if est is not None:
                st.subheader("Estimated model")
                if est.kind == "vol":
                    st.pyplot(vol_overlay_fig(est.actual, est.fitted),
                              clear_figure=True, width="stretch")
                else:
                    st.pyplot(fit_overlay_fig(est.actual, est.fitted, est.fitted_label),
                              clear_figure=True, width="stretch")
                with st.expander("Estimation output (EViews-style) — fitted to this series"):
                    st.code(est.summary, language="text")

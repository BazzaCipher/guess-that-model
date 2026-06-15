"""Entry point — FNCE40003 econometrics study app.

A thin shell: configure the page, build the navigation over the render-mode
views, and run.  All model knowledge lives in the ``models`` registry; all
rendering lives in ``views/``.
"""
from __future__ import annotations

import subprocess

import streamlit as st

from views import diagnose, explorers, inventory, risk, trainer

st.set_page_config(page_title="Econometrics model trainer", layout="wide")


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ""


pages = st.navigation(
    {
        "Play": [
            st.Page(trainer.render, title="Trainer", icon="🎯", url_path="trainer"),
            st.Page(diagnose.render, title="Diagnose", icon="🔧", url_path="diagnose"),
            st.Page(risk.render, title="Risk quiz", icon="📉", url_path="risk"),
        ],
        "Reference": [st.Page(inventory.render, title="Inventory", icon="📚", url_path="inventory")],
        "Explore": explorers.pages(),
    }
)
pages.run()

_sha = _commit_sha()
if _sha:
    st.sidebar.caption(f"build {_sha}")

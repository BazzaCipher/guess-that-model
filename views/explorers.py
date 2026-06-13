"""Explorers — one page per category that has interactive demos.

Pages are generated from the registry: any category with at least one model
carrying a ``demo`` gets a page that renders those demos (as tabs when there are
many, e.g. Portfolio VaR).
"""
from __future__ import annotations

import re

import streamlit as st

from models import by_category, categories


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _render_one(model) -> None:
    st.latex(model.equation)
    st.markdown(model.summary)
    st.markdown(f"**Tell:** {model.tell}")
    if model.beyond_course:
        st.caption("⚑ beyond this course")
    model.demo()


def _render_category(category: str) -> None:
    demos = [m for m in by_category(category) if m.demo is not None]
    st.title(category)
    st.caption(f"{len(demos)} interactive model{'s' if len(demos) != 1 else ''}. "
               "Adjust the controls to see each model respond.")
    if len(demos) > 4:
        tabs = st.tabs([m.name for m in demos])
        for tab, m in zip(tabs, demos):
            with tab:
                _render_one(m)
    else:
        for m in demos:
            with st.container(border=True):
                st.subheader(m.name)
                _render_one(m)


def pages() -> list:
    """One st.Page per category that has demo-bearing models."""
    out = []
    for cat in categories():
        if any(m.demo is not None for m in by_category(cat)):
            out.append(
                st.Page(lambda c=cat: _render_category(c), title=cat,
                        url_path=f"explore-{_slug(cat)}")
            )
    return out

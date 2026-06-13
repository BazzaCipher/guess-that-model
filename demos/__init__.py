"""Interactive Streamlit demos for the non-series model categories.

Each ``demo_*`` is a zero-arg callable wired to a Model's ``demo`` field; it
renders its own sliders (with a unique key prefix) and a plot.  Demos are the
only place besides ``views/`` that import Streamlit.
"""

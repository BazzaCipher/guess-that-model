"""Headless smoke tests — run with the project venv:  python smoke_test.py

Pure-logic checks over the registry + a Streamlit AppTest pass over each page.
Exits non-zero on the first failure.
"""
from __future__ import annotations

import sys

import numpy as np


def check_registry() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from models import REGISTRY, TRAINER_MODELS, SIMULATORS, FAMILY_OF, families
    from models.base import CATEGORIES
    from plots import acf_pacf_fig, series_fig

    assert REGISTRY, "empty registry"

    keys = [m.key for m in REGISTRY]
    assert len(keys) == len(set(keys)), "duplicate keys"

    for m in REGISTRY:
        assert m.category in CATEGORIES, f"{m.name}: bad category"
        assert m.equation and m.summary and m.tell, f"{m.name}: missing card text"
        assert (m.simulate is not None) or (m.demo is not None), \
            f"{m.name}: neither simulate nor demo"

    # every simulate-bearing model: finite across sizes, plots without error
    sim_models = [m for m in REGISTRY if m.simulate is not None]
    rng = np.random.default_rng(0)
    for m in sim_models:
        for nobs in (500, 2500, 10000):
            res = m.simulate(rng, nobs=nobs)
            assert res.series.shape[0] == nobs, f"{m.name}: wrong length"
            assert np.isfinite(res.series).all(), f"{m.name}: non-finite (n={nobs})"
            if m.family:
                assert res.family == m.family, f"{m.name}: family mismatch"
        # exercise the plot helpers on the smallest sample (incl. level series)
        res = m.simulate(rng, nobs=500)
        plt.close(series_fig(res.series, title=res.series_label))
        target = res.target_sq if res.target_sq is not None else (
            res.target_rv if res.target_rv is not None else res.series)
        plt.close(acf_pacf_fig(target, label="x", lags=20))

    assert set(SIMULATORS) == {m.name for m in TRAINER_MODELS}
    assert set(FAMILY_OF) == set(SIMULATORS)
    assert families(), "no families"
    n_inv_only = len(sim_models) - len(TRAINER_MODELS)
    print(f"  registry OK — {len(REGISTRY)} models "
          f"({len(TRAINER_MODELS)} trainer, {n_inv_only} inventory-only sim, "
          f"{len(REGISTRY) - len(sim_models)} demo-only)")
    print(f"  families: {families()}")


def _run_view(module: str):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(f"from views import {module}\n{module}.render()",
                             default_timeout=60)
    at.run()
    return at


def check_tracks() -> None:
    import numpy as np
    from models import track_models, tracks
    from views.trainer import TRACK_VIEW

    rng = np.random.default_rng(0)
    ts = tracks()
    assert ts, "no trainer tracks"
    total = 0
    for t in ts:
        assert t in TRACK_VIEW, f"track {t!r} has no TRACK_VIEW entry"
        ms = track_models(t)
        assert len(ms) >= 2, f"track {t!r} needs >=2 models to guess between"
        get_proc, _ = TRACK_VIEW[t]
        for m in ms:
            res = m.simulate(rng, nobs=600)
            proc = get_proc(res)
            assert proc is not None and np.isfinite(proc).all(), \
                f"{m.name}: track {t!r} process is None/non-finite"
        total += len(ms)
    print(f"  tracks OK — {len(ts)} tracks {ts}, {total} models")


def check_apptest() -> None:
    # app.py shell (nav + default page)
    from streamlit.testing.v1 import AppTest
    shell = AppTest.from_file("app.py", default_timeout=30)
    shell.run()
    assert not shell.exception, f"app.py: {shell.exception}"

    # inventory renders all cards (default filters)
    inv = _run_view("inventory")
    assert not inv.exception, f"inventory: {inv.exception}"

    # trainer: render, then click 'New round' to exercise the plotting path
    tr = _run_view("trainer")
    assert not tr.exception, f"trainer: {tr.exception}"
    for b in tr.button:
        if b.label == "New round":
            b.click()
            tr.run()
            break
    assert not tr.exception, f"trainer after New round: {tr.exception}"
    # submit a guess to exercise the reveal path
    for b in tr.button:
        if b.label == "Submit guess":
            b.click()
            tr.run()
            break
    assert not tr.exception, f"trainer after guess: {tr.exception}"
    assert tr.session_state["attempted"] == 1, "guess did not register"
    print("  AppTest OK — app shell + inventory + trainer (round + guess + reveal) run clean")


def check_demos() -> None:
    from streamlit.testing.v1 import AppTest
    from models import REGISTRY

    demos = [m for m in REGISTRY if m.demo is not None]
    for m in demos:
        src = f"from {m.demo.__module__} import {m.demo.__name__}\n{m.demo.__name__}()"
        at = AppTest.from_string(src, default_timeout=90)
        at.run()
        assert not at.exception, f"{m.name} demo: {at.exception}"
    print(f"  demos OK — {len(demos)} interactive demos run headless without exception")


def check_explorer_pages() -> None:
    """Render each explorer page whole — catches widget-key collisions between
    demos that only surface when a category's demos render together."""
    from streamlit.testing.v1 import AppTest
    from models import by_category, categories

    n = 0
    for cat in categories():
        if not any(m.demo is not None for m in by_category(cat)):
            continue
        src = ("from views.explorers import _render_category\n"
               f"_render_category({cat!r})")
        at = AppTest.from_string(src, default_timeout=120)
        at.run()
        assert not at.exception, f"explorer page {cat!r}: {at.exception}"
        n += 1
    print(f"  explorer pages OK — {n} category pages render with no key collisions")


if __name__ == "__main__":
    print("Smoke tests:")
    try:
        check_registry()
        check_tracks()
        check_apptest()
        check_demos()
        check_explorer_pages()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    print("All smoke tests passed.")

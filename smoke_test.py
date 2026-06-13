"""Headless smoke tests — run with the project venv:  python smoke_test.py

Pure-logic checks over the registry + a Streamlit AppTest pass over each page.
Exits non-zero on the first failure.
"""
from __future__ import annotations

import sys

import numpy as np


def check_registry() -> None:
    from models import REGISTRY, TRAINER_MODELS, SIMULATORS, FAMILY_OF, families
    from models.base import CATEGORIES

    assert REGISTRY, "empty registry"

    keys = [m.key for m in REGISTRY]
    assert len(keys) == len(set(keys)), "duplicate keys"

    for m in REGISTRY:
        assert m.category in CATEGORIES, f"{m.name}: bad category"
        assert m.equation and m.summary and m.tell, f"{m.name}: missing card text"
        assert (m.simulate is not None) or (m.demo is not None), \
            f"{m.name}: neither simulate nor demo"

    # every trainer simulator returns a finite, correctly-sized SimResult
    rng = np.random.default_rng(0)
    for m in TRAINER_MODELS:
        for nobs in (500, 2500):
            res = m.simulate(rng, nobs=nobs)
            assert res.series.shape[0] == nobs, f"{m.name}: wrong length"
            assert np.isfinite(res.series).all(), f"{m.name}: non-finite series"
            assert res.family == m.family, f"{m.name}: family mismatch"

    assert set(SIMULATORS) == {m.name for m in TRAINER_MODELS}
    assert set(FAMILY_OF) == set(SIMULATORS)
    assert families(), "no families"
    print(f"  registry OK — {len(REGISTRY)} models, {len(TRAINER_MODELS)} trainer-eligible, "
          f"families={families()}")


def _run_view(module: str):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(f"from views import {module}\n{module}.render()",
                             default_timeout=60)
    at.run()
    return at


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
    print("  AppTest OK — app shell + inventory + trainer (with a round) run clean")


if __name__ == "__main__":
    print("Smoke tests:")
    try:
        check_registry()
        check_apptest()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    print("All smoke tests passed.")

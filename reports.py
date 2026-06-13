"""EViews-style estimation output for the Trainer reveal (vol + realised vol).

On reveal we *fit the true specification* to the simulated series and print the
coefficient table the way you'd read it in EViews — coefficient, std. error,
z/t-statistic, prob — plus log-likelihood and information criteria. This also
settles nested-model ambiguity: the fitted gamma (GJR), nu (Student-t) or
alpha2 (GARCH(2,1)) is right there in the table, even though it can't be read
off the squared-returns correlogram.

Everything here is pure numpy/arch/statsmodels — no Streamlit — so it stays
headless-testable.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class Estimation:
    summary: str               # EViews-style coefficient table (text)
    actual: np.ndarray         # series the fit explains (returns, or RV)
    fitted: np.ndarray         # conditional vol σ_t (vol) or fitted RV (rv)
    kind: str                  # "vol" -> ±2σ envelope; "rv" -> actual vs fitted
    fitted_label: str

# names that map to an arch univariate-volatility fit -------------------------
_GARCH_SPEC = {
    "GARCH(1,1) Normal": ("garch", dict(p=1, q=1), "normal"),
    "GARCH(2,1) Normal": ("garch", dict(p=2, q=1), "normal"),
    "GJR-GARCH(1,1)": ("garch", dict(p=1, o=1, q=1), "normal"),
    "EGARCH(1,1)": ("egarch", dict(p=1, o=1, q=1), "normal"),
    "GARCH(1,1) Student-t": ("garch", dict(p=1, q=1), "t"),
}


def _garch_estimation(name: str, r: np.ndarray, params: dict) -> Estimation:
    from arch.univariate import ARCH, EGARCH, GARCH, Normal, StudentsT, ZeroMean

    if name == "ARCH(q)":
        vol = ARCH(p=int(params.get("q", 2)))
        dist = Normal()
    else:
        kind, kw, dist_name = _GARCH_SPEC[name]
        vol = EGARCH(**kw) if kind == "egarch" else GARCH(**kw)
        dist = StudentsT() if dist_name == "t" else Normal()
    am = ZeroMean(r, volatility=vol, distribution=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)
    return Estimation(res.summary().as_text(), r, np.asarray(res.conditional_volatility),
                      "vol", "fitted conditional volatility σₜ")


def _har_estimation(rv: np.ndarray, with_jump: bool) -> Estimation:
    """Corsi HAR regression on log RV: log RV_t on daily / weekly / monthly
    averages (the OLS that *is* the model). Recovers beta_d/w/m."""
    import statsmodels.api as sm

    y_full = np.log(rv)
    n = len(y_full)
    rows_y, xd, xw, xm = [], [], [], []
    for t in range(22, n):
        rows_y.append(y_full[t])
        xd.append(y_full[t - 1])
        xw.append(y_full[t - 5:t].mean())
        xm.append(y_full[t - 22:t].mean())
    X = sm.add_constant(np.column_stack([xd, xw, xm]))
    res = sm.OLS(np.asarray(rows_y), X).fit()
    names = ["C", "log RV_d (β_d)", "log RV_w (β_w)", "log RV_m (β_m)"]
    summary = res.summary(yname="log RV_t", xname=names).as_text()
    return Estimation(summary, rv[22:], np.exp(res.fittedvalues), "rv", "fitted RV (HAR)")


def _arma_logrv_estimation(rv: np.ndarray) -> Estimation:
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(np.log(rv), order=(1, 0, 1)).fit()
    return Estimation(res.summary().as_text(), rv, np.exp(np.asarray(res.fittedvalues)),
                      "rv", "fitted RV (ARMA)")


def estimation_report(result) -> Estimation | None:
    """Fit the true specification and return an :class:`Estimation` (EViews-style
    table + fitted series) for a Volatility / Realised-vol round, or None."""
    name = result.name
    try:
        if name == "ARCH(q)" or name in _GARCH_SPEC:
            return _garch_estimation(name, np.asarray(result.series), result.params)
        if name in ("HAR-RV (Corsi)", "HAR-RV-J"):
            return _har_estimation(np.asarray(result.target_rv), name.endswith("-J"))
        if name == "ARMA(1,1) on log RV":
            return _arma_logrv_estimation(np.asarray(result.target_rv))
    except Exception as exc:  # never let a fit failure break the reveal
        return Estimation(f"(estimation did not converge: {type(exc).__name__})",
                          np.asarray(result.series), np.asarray(result.series), "rv", "n/a")
    return None


def diagnostic_tells(result) -> dict[str, str]:
    """The numbers *behind* the tell — the quantities that actually separate
    look-alike vol models that share a squared-returns correlogram."""
    out: dict[str, str] = {}
    if result.target_sq is not None:          # a returns-based volatility model
        r = np.asarray(result.series, dtype=float)
        r = r[np.isfinite(r)]
        n = len(r)
        # excess kurtosis — fat tails flag Student-t vs Normal innovations
        z = (r - r.mean()) / r.std()
        exkurt = float((z ** 4).mean() - 3.0)
        out["Excess kurtosis"] = f"{exkurt:+.2f}  (≈0 Normal, ≫0 fat-tailed / Student-t)"
        # leverage: variance after down-days vs up-days (GJR / EGARCH asymmetry)
        sq = r ** 2
        down = sq[1:][r[:-1] < 0]
        up = sq[1:][r[:-1] > 0]
        if len(down) and len(up) and up.mean() > 0:
            ratio = down.mean() / up.mean()
            out["Leverage  E[r²|down]/E[r²|up]"] = (
                f"{ratio:.2f}  (>1 ⇒ asymmetric: GJR / EGARCH)")
        # ARCH effect strength: Ljung-Box Q(10) on squared returns
        sqz = sq - sq.mean()
        denom = (sqz ** 2).sum()
        q = 0.0
        for k in range(1, 11):
            ac = (sqz[k:] * sqz[:-k]).sum() / denom
            q += ac * ac / (n - k)
        out["Ljung-Box Q(10) on r²"] = f"{n * (n + 2) * q:.0f}  (large ⇒ strong ARCH/GARCH)"
    elif result.target_rv is not None:        # a realised-vol model
        rv = np.asarray(result.target_rv, dtype=float)
        lr = np.log(rv[rv > 0])
        lrz = lr - lr.mean()
        denom = (lrz ** 2).sum()
        ac1 = (lrz[1:] * lrz[:-1]).sum() / denom
        ac22 = (lrz[22:] * lrz[:-22]).sum() / denom
        out["ACF log RV @ lag 1"] = f"{ac1:.2f}"
        out["ACF log RV @ lag 22 (month)"] = f"{ac22:.2f}  (still high ⇒ long memory / HAR)"
    return out

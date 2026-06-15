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
        # NB: the formal ARCH/autocorrelation test (Ljung-Box on r²) now lives in
        # spec_tests() and is shown as its own "Specification tests" table.
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


# ---------------------------------------------------------------------------
# Specification tests — the named hypothesis tests for each Trainer track, run
# on the *observable* series (no hidden parameters), so they're fair to show
# before the guess. Each is the test you'd actually run at that stage of a
# Box-Jenkins / volatility workflow:
#
#   Conditional mean  →  Ljung-Box on the series · ARCH-LM · Jarque-Bera
#   Volatility        →  Ljung-Box on r² · ARCH-LM · Jarque-Bera (std. returns)
#   Realised vol      →  Ljung-Box on log RV · ADF (unit root / long memory)
#
# Returns plain dict rows for st.table. Each test is individually guarded so a
# degenerate series drops a single row, never the whole table. Statistics come
# from statsmodels (lazy-imported) so the numbers are the real thing.
# ---------------------------------------------------------------------------
def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _verdict(p: float, reject: str, keep: str) -> str:
    return (f"p<0.05 ⇒ reject H₀: {reject}" if p < 0.05
            else f"p≥0.05 ⇒ don't reject: {keep}")


def _row(name: str, stat: float, df, p: float, reading: str) -> dict:
    return {"Test": name, "Statistic": f"{stat:.2f}", "df": str(df),
            "p-value": f"{p:.3f}", "Reading": reading}


def _lb_row(x, m: int, on: str, reject: str, keep: str):
    """Ljung-Box Q(m). df = m here (nothing fitted yet); on residuals you'd
    subtract the number of estimated ARMA/GARCH parameters."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        out = acorr_ljungbox(_finite(x), lags=[m], return_df=True)
        stat, p = float(out["lb_stat"].iloc[0]), float(out["lb_pvalue"].iloc[0])
        return _row(f"Ljung-Box Q({m}) on {on}", stat, m, p, _verdict(p, reject, keep))
    except Exception:
        return None


def _arch_lm_row(x, q: int):
    """Engle ARCH-LM: regress squared residuals on q lags, statistic T·R²."""
    try:
        from statsmodels.stats.diagnostic import het_arch
        lm, lmp, _f, _fp = het_arch(_finite(x), nlags=q)
        return _row(f"ARCH-LM ({q} lags), T·R²", float(lm), q, float(lmp),
                    _verdict(float(lmp), "ARCH effects — volatility clustering",
                             "no ARCH effects"))
    except Exception:
        return None


def _jb_row(x, on: str):
    try:
        from statsmodels.stats.stattools import jarque_bera
        jb, p, _s, _k = jarque_bera(_finite(x))
        return _row(f"Jarque-Bera on {on}", float(jb), 2, float(p),
                    _verdict(float(p), "non-normal — fat tails / skew (⇒ Student-t)",
                             "can't reject normality"))
    except Exception:
        return None


def _adf_row(x, on: str):
    try:
        from statsmodels.tsa.stattools import adfuller
        stat, p = adfuller(_finite(x), autolag="AIC")[:2]
        return _row(f"ADF (unit root) on {on}", float(stat), "—", float(p),
                    _verdict(float(p), "stationary / I(0)",
                             "unit root — I(1) (or long memory)"))
    except Exception:
        return None


def spec_tests(result, track: str) -> list[dict]:
    """Named specification tests for a Trainer track, computed on the series."""
    rows: list = []
    r = _finite(result.series)
    if track == "Conditional mean":
        rows += [
            _lb_row(r, 10, "the series",
                    "autocorrelation ⇒ AR / MA / ARMA structure", "white noise in the mean"),
            _arch_lm_row(r, 10),                  # an AR-GARCH hiding in a 'mean' series shows here
            _jb_row(r, "the series"),
        ]
    elif track == "Volatility":
        sq = result.target_sq if result.target_sq is not None else r ** 2
        std = r / (r.std() or 1.0)
        rows += [
            _lb_row(sq, 10, "squared returns",
                    "ARCH / GARCH dynamics in variance", "no volatility clustering"),
            _arch_lm_row(r, 10),
            _jb_row(std, "standardised returns"),
        ]
    elif track == "Realised volatility":
        rv = _finite(result.target_rv)
        lrv = np.log(rv[rv > 0])
        rows += [
            _lb_row(lrv, 22, "log RV",
                    "persistent ⇒ long memory (HAR / ARFIMA)", "no RV persistence"),
            _adf_row(lrv, "log RV"),
        ]
    else:  # Both / other — test whichever processes the series carries
        rows.append(_lb_row(r, 10, "the series",
                            "autocorrelation in the mean", "white-noise mean"))
        if result.target_sq is not None:
            rows.append(_lb_row(result.target_sq, 10, "squared returns",
                                "ARCH / GARCH in variance", "no volatility clustering"))
        if result.target_rv is not None:
            rvo = _finite(result.target_rv)
            rows.append(_lb_row(np.log(rvo[rvo > 0]), 22, "log RV",
                                "long memory in RV", "no RV persistence"))
    return [row for row in rows if row is not None]


# ---------------------------------------------------------------------------
# "Diagnose the output" — a second Trainer mode. Each case simulates flawed data,
# deliberately fits the WRONG (too-simple) model, and prints the EViews-style
# table where the defect surfaces (coefficient block + a residual-diagnostics
# footer). The player names the problem and the model that fixes it. The cases
# are designed so ONE diagnostic dominates, and the fix question disambiguates.
# ---------------------------------------------------------------------------
PROBLEMS = [
    "Residual autocorrelation — the mean is under-specified",
    "ARCH effects — conditional heteroskedasticity in the residuals",
    "Non-normal, fat-tailed residuals",
    "Unit root — the series is non-stationary (spurious regression)",
    "Leverage / asymmetry left in the volatility",
]
FIXES = [
    "Add AR/MA terms — a richer ARMA mean",
    "Add a GARCH(1,1) variance equation",
    "Switch to a Student-t conditional distribution",
    "Difference the series / model the returns (ARIMA)",
    "Use GJR-GARCH or EGARCH (asymmetric volatility)",
]
P_AUTOCORR, P_ARCH, P_NORM, P_UNITROOT, P_LEVERAGE = PROBLEMS
F_ARMA, F_GARCH, F_T, F_ARIMA, F_GJR = FIXES


def _stat_lb(x, m):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    out = acorr_ljungbox(_finite(x), lags=[m], return_df=True)
    return float(out["lb_stat"].iloc[0]), float(out["lb_pvalue"].iloc[0])


def _stat_arch(x, q):
    from statsmodels.stats.diagnostic import het_arch
    lm, lmp, _f, _fp = het_arch(_finite(x), nlags=q)
    return float(lm), float(lmp)


def _stat_jb(x):
    from statsmodels.stats.stattools import jarque_bera
    jb, p, _s, _k = jarque_bera(_finite(x))
    return float(jb), float(p)


def _stat_adf(x):
    from statsmodels.tsa.stattools import adfuller
    stat, p = adfuller(_finite(x), autolag="AIC")[:2]
    return float(stat), float(p)


def _stat_engle_ng(z):
    """Engle-Ng (1993) joint sign-bias test on standardised residuals ẑ.

    Regress ẑ_t² on a constant, the lagged negative-return dummy S⁻_{t-1},
    and the size terms S⁻_{t-1}·ẑ_{t-1} and S⁺_{t-1}·ẑ_{t-1}. Under symmetry
    the three slopes are jointly zero, so T·R² ~ χ²(3); a rejection says
    negative and positive shocks move next-period variance differently
    (leverage) — which a symmetric GARCH cannot capture.
    """
    import statsmodels.api as sm
    from scipy import stats

    z = _finite(z)
    zt, zl = z[1:], z[:-1]
    sneg = (zl < 0).astype(float)
    X = np.column_stack([np.ones_like(zt), sneg, sneg * zl, (1.0 - sneg) * zl])
    res = sm.OLS(zt ** 2, X).fit()
    stat = float(len(zt) * res.rsquared)
    return stat, float(stats.chi2.sf(stat, 3))


def _diag_line(name, stat_p):
    stat, p = stat_p
    return f"  {name:<30}{stat:11.2f}    p = {p:.3f}"


def _ols_eviews(y, X, xnames, title, adf_on=None):
    """Fit OLS, return the statsmodels summary (already EViews-like: R², DW, JB,
    skew/kurtosis) plus a residual-diagnostics footer the defect shows up in."""
    import statsmodels.api as sm

    y = np.asarray(y, dtype=float)
    res = sm.OLS(y, X).fit()
    e = np.asarray(res.resid)
    diag = ["", "Residual diagnostics (run on the OLS residuals)", "=" * 60,
            _diag_line("Ljung-Box  Q(10)  resid", _stat_lb(e, 10)),
            _diag_line("Ljung-Box  Q(10)  resid^2", _stat_lb(e ** 2, 10)),
            _diag_line("ARCH-LM(10)  T*R^2", _stat_arch(e, 10)),
            _diag_line("Jarque-Bera  resid", _stat_jb(e))]
    if adf_on is not None:
        diag.append(_diag_line("ADF (unit root) on dep. var", _stat_adf(adf_on)))
    return f"{title}\n{'=' * 60}\n{res.summary(xname=list(xnames)).as_text()}\n" + "\n".join(diag)


def _case_arch(rng):
    from models import SIMULATORS
    r = np.asarray(SIMULATORS["GARCH(1,1) Normal"](rng, nobs=2500).series, float)
    text = _ols_eviews(r, np.ones((len(r), 1)), ["C"],
                       "Dependent Variable: R    Method: Least Squares (constant mean)")
    return text, P_ARCH, F_GARCH, (
        "Q(resid) is fine but **Q(resid²) and ARCH-LM reject** — the level is "
        "serially clean while the *squared* residuals are autocorrelated: textbook "
        "volatility clustering. The mean is adequate; you need a variance equation, "
        "so add a GARCH(1,1). (Jarque-Bera may also flag mild fat tails — a "
        "by-product of ARCH, not a reason to reach for Student-t first.)")


def _case_autocorr(rng):
    from models import SIMULATORS
    r = np.asarray(SIMULATORS["AR(p) returns"](rng, nobs=1500).series, float)
    text = _ols_eviews(r, np.ones((len(r), 1)), ["C"],
                       "Dependent Variable: R    Method: Least Squares (constant mean)")
    return text, P_AUTOCORR, F_ARMA, (
        "**Q(resid) is large at low lags** — the residuals are still autocorrelated, "
        "so a constant mean under-fits the dynamics. ARCH-LM is fine (no volatility "
        "clustering) and the Durbin-Watson is far from 2. Add AR/MA terms.")


def _case_fattails(rng):
    r = rng.standard_t(df=4, size=1500)
    r = r / r.std()
    text = _ols_eviews(r, np.ones((len(r), 1)), ["C"],
                       "Dependent Variable: R    Method: Least Squares (constant mean)")
    return text, P_NORM, F_T, (
        "**Jarque-Bera is huge with excess kurtosis ≫ 0**, while Q(resid), "
        "Q(resid²) and ARCH-LM are all fine — the residuals are iid but heavy-"
        "tailed. The dynamics are fine; switch the conditional distribution to "
        "Student-t (the JB is what fails, not the autocorrelation/ARCH tests).")


def _case_unitroot(rng):
    from models import SIMULATORS
    y = np.asarray(SIMULATORS["ARIMA (unit root)"](rng, nobs=600).series, float)
    n = len(y)
    X = np.column_stack([np.ones(n), np.arange(n)])
    text = _ols_eviews(y, X, ["C", "@TREND"],
                       "Dependent Variable: Y    Method: Least Squares (level on trend)",
                       adf_on=y)
    return text, P_UNITROOT, F_ARIMA, (
        "**Durbin-Watson is near zero, Q(resid) explodes, and ADF fails to reject "
        "a unit root** (large p) — the level series is non-stationary, so a "
        "regression in levels is spurious (an inflated R² with serially correlated "
        "residuals). Difference it / model the returns (ARIMA).")


def _case_leverage(rng):
    """Asymmetric data (GJR) fitted with a SYMMETRIC GARCH(1,1): the variance
    clustering is soaked up — Q(ẑ²) and ARCH-LM are clean — but the Engle-Ng
    sign-bias test still rejects, because negative shocks raise tomorrow's
    variance more than positive ones. The fix is GJR / EGARCH."""
    from arch.univariate import GARCH, Normal, ZeroMean

    from models import SIMULATORS

    # Regenerate until the symmetric fit's Engle-Ng test clearly rejects (stat
    # past the χ²(3) 5% critical value, 7.81), so the stated answer can never
    # contradict the printed footer. At n=6000 the first draw almost always passes.
    res = z = None
    for _ in range(6):
        r = np.asarray(SIMULATORS["GJR-GARCH(1,1)"](rng, nobs=6000).series, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ZeroMean(r, volatility=GARCH(p=1, q=1), distribution=Normal()).fit(
                disp="off", show_warning=False)
        z = np.asarray(res.std_resid)
        if _stat_engle_ng(z)[0] > 9.0:
            break
    diag = ["", "Residual diagnostics (standardised residuals ẑ = ε̂/σ̂)", "=" * 60,
            _diag_line("Ljung-Box  Q(10)  z_hat", _stat_lb(z, 10)),
            _diag_line("Ljung-Box  Q(10)  z_hat^2", _stat_lb(z ** 2, 10)),
            _diag_line("ARCH-LM(10)  T*R^2", _stat_arch(z, 10)),
            _diag_line("Engle-Ng sign-bias  T*R^2 ~chi2(3)", _stat_engle_ng(z))]
    title = ("Dependent Variable: R    Method: ML — symmetric GARCH(1,1) Normal\n"
             + "=" * 60)
    text = f"{title}\n{res.summary().as_text()}\n" + "\n".join(diag)
    return text, P_LEVERAGE, F_GJR, (
        "Q(ẑ²) and ARCH-LM are fine — a symmetric GARCH(1,1) has soaked up the "
        "volatility clustering — but the **Engle-Ng sign-bias test rejects** at "
        "χ²(3): negative shocks raise next-day variance more than positive ones of "
        "the same size. A symmetric GARCH can't represent that asymmetry, so re-fit "
        "with GJR-GARCH or EGARCH (the leverage effect).")


_DIAGNOSE_CASES = (_case_arch, _case_autocorr, _case_fattails, _case_unitroot,
                   _case_leverage)


def _options(correct, pool, rng, k=4):
    others = [x for x in pool if x != correct]
    idx = rng.permutation(len(others))[:k - 1]
    opts = [correct] + [others[i] for i in idx]
    return [opts[i] for i in rng.permutation(len(opts))]


def diagnose_round(rng) -> dict:
    """Build one 'diagnose the output' round: EViews text + two MCQs + reveal."""
    case = _DIAGNOSE_CASES[int(rng.integers(len(_DIAGNOSE_CASES)))]
    text, problem, fix, why = case(rng)
    return {
        "text": text,
        "problem": problem,
        "fix": fix,
        "why": why,
        "problem_options": _options(problem, PROBLEMS, rng),
        "fix_options": _options(fix, FIXES, rng),
    }

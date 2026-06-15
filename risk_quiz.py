"""Risk quiz — dynamic rounds for VaR backtesting, multivariate-GARCH choice,
and tail / extreme-risk reasoning.

Pure logic (no Streamlit).  Each generator returns a round dict::

    {"topic": str, "text": str,
     "questions": [{"prompt": str, "options": [str, ...], "answer": str}, ...],
     "why": str}

The numbers in ``text`` are computed from freshly simulated data every call —
the Kupiec/Christoffersen statistics, the MGARCH parameter counts, the Hill tail
index and the copula joint-exceedance counts are all real — so each round is a
reading exercise rather than a memorised card.  ``views/risk.py`` renders it.
"""
from __future__ import annotations

import numpy as np

from reports import _options  # shared option-shuffler (keeps the 4-option style)


def _q(prompt: str, answer: str, pool, rng) -> dict:
    return {"prompt": prompt, "answer": answer, "options": _options(answer, pool, rng)}


# ===========================================================================
# VaR backtesting — Kupiec (unconditional coverage) + Christoffersen (independence)
# ===========================================================================
P_OK = "Adequate — both the exception rate and their timing look fine"
P_TOOMANY = "Too many exceptions — VaR is breached more than α (fails unconditional coverage)"
P_TOOFEW = "Too few exceptions — VaR is too conservative (it over-covers)"
P_CLUSTER = "Exceptions cluster in time — the count is fine but independence fails"
VAR_PROBLEMS = [P_OK, P_TOOMANY, P_TOOFEW, P_CLUSTER]

F_OK = "Nothing — it passes both Kupiec and Christoffersen"
F_TOOMANY = "VaR sits too low — refit with fatter tails (Student-t / EVT)"
F_TOOFEW = "VaR is too high — relax it so it stops tying up excess capital"
F_CLUSTER = "Make VaR volatility-adaptive (GARCH-filtered / McNeil-Frey) so breaches stop bunching"
VAR_FIXES = [F_OK, F_TOOMANY, F_TOOFEW, F_CLUSTER]

_CHI1, _CHI2 = 3.84, 5.99  # 5% critical values, χ²(1) and χ²(2)


def _ll_binom(x: int, T: int, p: float) -> float:
    """Binomial log-likelihood of x hits in T draws at prob p (0·log0 = 0)."""
    a = x * np.log(p) if x > 0 and p > 0 else 0.0
    b = (T - x) * np.log(1 - p) if (T - x) > 0 and p < 1 else 0.0
    return a + b


def _kupiec(x: int, T: int, alpha: float) -> float:
    pi = x / T
    return -2.0 * (_ll_binom(x, T, alpha) - _ll_binom(x, T, pi))


def _xlogy(n: int, p: float) -> float:
    return n * np.log(p) if (n > 0 and p > 0) else 0.0


def _christoffersen_ind(hits: np.ndarray) -> float:
    """Christoffersen (1998) Markov independence LR ~ χ²(1)."""
    prev, cur = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_markov = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
                 + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
    ll_indep = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    return -2.0 * (ll_indep - ll_markov)


def _gen_hits(rng, T: int, alpha: float, flavor: str) -> np.ndarray:
    if flavor == "clustered":
        # Markov chain: marginal hit-rate ≈ α but strongly persistent (clusters)
        p11 = 0.45
        p01 = (alpha * (1 - p11)) / (1 - alpha)
        h = np.zeros(T, int)
        state = 0
        for t in range(T):
            state = 1 if rng.random() < (p11 if state else p01) else 0
            h[t] = state
        return h
    rate = {"ok": alpha,
            "toomany": alpha * rng.uniform(2.2, 3.3),
            "toofew": alpha * rng.uniform(0.0, 0.30)}[flavor]
    return (rng.random(T) < rate).astype(int)


def _var_case(rng) -> dict:
    flavor = str(rng.choice(["ok", "toomany", "toofew", "clustered"]))
    if flavor == "clustered":
        alpha, T = 0.05, int(rng.choice([600, 750, 1000]))  # need power for independence
    else:
        alpha = float(rng.choice([0.01, 0.05]))
        T = int(rng.choice([250, 500, 750]))

    # bias generation toward the intended signal, but read the verdict off the
    # actual statistics so the stated answer always matches the printed numbers
    x = lr_uc = lr_ind = 0.0
    uc = ind = False
    rate = 0.0
    for _ in range(60):
        hits = _gen_hits(rng, T, alpha, flavor)
        x = int(hits.sum())
        lr_uc, lr_ind = _kupiec(x, T, alpha), _christoffersen_ind(hits)
        uc, ind, rate = lr_uc > _CHI1, lr_ind > _CHI1, x / T
        if ((flavor == "ok" and not uc and not ind)
                or (flavor == "toomany" and uc and rate > alpha and not ind)
                or (flavor == "toofew" and uc and rate < alpha)
                or (flavor == "clustered" and ind and not uc)):
            break
    lr_cc = lr_uc + lr_ind

    if uc and rate > alpha:
        problem, fix = P_TOOMANY, F_TOOMANY
    elif uc and rate < alpha:
        problem, fix = P_TOOFEW, F_TOOFEW
    elif ind:
        problem, fix = P_CLUSTER, F_CLUSTER
    else:
        problem, fix = P_OK, F_OK

    def v(stat, crit):
        return "reject" if stat > crit else "ok"

    text = "\n".join([
        "Value-at-Risk backtest",
        "=" * 60,
        f"  Coverage (1 - α)              {(1 - alpha) * 100:.0f}%",
        f"  Sample size  T                {T}",
        f"  Expected exceptions  α·T      {alpha * T:.1f}",
        f"  Observed exceptions           {x}    (rate {rate * 100:.2f}%)",
        "",
        "  Backtest                          stat     5% crit  verdict",
        "  " + "-" * 56,
        f"  Kupiec          LR_uc  ~χ²(1)    {lr_uc:7.2f}    {_CHI1:5.2f}    {v(lr_uc, _CHI1)}",
        f"  Christoffersen  LR_ind ~χ²(1)    {lr_ind:7.2f}    {_CHI1:5.2f}    {v(lr_ind, _CHI1)}",
        f"  Christoffersen  LR_cc  ~χ²(2)    {lr_cc:7.2f}    {_CHI2:5.2f}    {v(lr_cc, _CHI2)}",
    ])

    if problem is P_CLUSTER:
        why = (f"Only {x} exceptions vs {alpha * T:.1f} expected, so Kupiec's count test "
               f"passes (LR_uc={lr_uc:.2f} < 3.84) — but the breaches bunch together, so the "
               f"independence test rejects (LR_ind={lr_ind:.2f} > 3.84), and LR_cc with it. "
               "Kupiec alone would have missed this. A static VaR that clusters its breaches "
               "needs to react to volatility (GARCH-filtered historical VaR / McNeil-Frey).")
    elif problem is P_TOOMANY:
        why = (f"{x} exceptions vs {alpha * T:.1f} expected — Kupiec rejects "
               f"(LR_uc={lr_uc:.2f} > 3.84). The VaR is breached far too often, so it sits "
               "too low: the tail is fatter than the model assumes (go Student-t / EVT).")
    elif problem is P_TOOFEW:
        why = (f"Only {x} exceptions vs {alpha * T:.1f} expected — Kupiec rejects on the low "
               f"side (LR_uc={lr_uc:.2f} > 3.84). The VaR over-covers: safe, but it ties up "
               "capital you don't need to reserve.")
    else:
        why = (f"{x} exceptions vs {alpha * T:.1f} expected and no clustering — both "
               f"Kupiec (LR_uc={lr_uc:.2f}) and Christoffersen (LR_cc={lr_cc:.2f}) sit under "
               "their critical values. The model is well-calibrated; leave it.")

    return {"topic": "VaR backtest", "text": text,
            "questions": [_q("What does the backtest say?", problem, VAR_PROBLEMS, rng),
                          _q("What's the right response?", fix, VAR_FIXES, rng)],
            "why": why}


# ===========================================================================
# Multivariate GARCH — pick the model that fits the situation
# ===========================================================================
M_VECH = "vech / diagonal vech"
M_BEKK = "BEKK (scalar)"
M_CCC = "CCC — constant conditional correlation"
M_DCC = "DCC — dynamic conditional correlation"
M_RCOV = "Realised covariance (MHAR / VARFIMA)"
MGARCH_MODELS = [M_VECH, M_BEKK, M_CCC, M_DCC, M_RCOV]


def _mg_feasible(rng):
    N = int(rng.integers(25, 81))
    k = N * (N + 1) // 2                # distinct covariance elements
    full_vech = 2 * k * k + k           # A, B full on the vech vector ~ O(N⁴)
    diag_vech = 3 * k                   # C, A, B diagonal
    text = "\n".join([
        f"Multivariate GARCH — portfolio of N = {N} assets",
        "=" * 60,
        f"  Free parameters, full vech       ≈ {full_vech:,}",
        f"  Free parameters, diagonal vech   = {diag_vech:,}",
        "  DCC dynamic parameters (a, b)    = 2     (correlation targeting)",
        "",
        "  You need a covariance you can actually estimate at this size,",
        "  with correlations that still move through time.",
    ])
    why = (f"Full vech explodes to ~{full_vech:,} parameters and even diagonal vech needs "
           f"{diag_vech:,} — infeasible at N={N}. DCC fits each variance with a univariate "
           "GARCH, pins the average correlation by targeting, and lets just two parameters "
           "(a, b) drive the dynamics — scalable and still time-varying.")
    return text, M_DCC, why


def _mg_constant_corr(rng):
    rho0, rhoc = 0.30, float(rng.choice([0.7, 0.8, 0.9]))
    text = "\n".join([
        "Multivariate GARCH — model diagnostics",
        "=" * 60,
        f"  Average pairwise correlation, calm periods   {rho0:.2f}",
        f"  Average pairwise correlation, the 2008 crash {rhoc:.2f}",
        "",
        "  Your current model assumes a single, fixed correlation matrix R,",
        "  yet correlations clearly jumped when it mattered most.",
    ])
    why = ("CCC freezes the correlation matrix — fine for the variances, fatal for the "
           f"correlations, which spiked from {rho0:.2f} to {rhoc:.2f} in the crisis. DCC lets "
           "the correlation evolve through a GARCH-like recursion, rising exactly when assets "
           "co-move.")
    return text, M_DCC, why


def _mg_posdef(rng):
    N = int(rng.choice([3, 4, 5]))
    text = "\n".join([
        f"Multivariate GARCH — {N}-asset book",
        "=" * 60,
        "  Requirement: Σ_t guaranteed positive-definite every day,",
        "  with NO awkward parameter constraints to impose,",
        "  yet a genuinely time-varying covariance (not just variances).",
    ])
    why = ("The quadratic form Σ_t = CC' + Aε_{t-1}ε_{t-1}'A' + BΣ_{t-1}B' is "
           "positive-definite by construction with no side constraints — that's BEKK. The "
           "scalar version needs only two dynamic parameters yet still moves the whole "
           "covariance.")
    return text, M_BEKK, why


def _mg_realised(rng):
    freq = str(rng.choice(["5-minute", "1-minute", "10-minute"]))
    text = "\n".join([
        "Multivariate GARCH — data available",
        "=" * 60,
        f"  You have {freq} intraday prices for every asset.",
        "  You'd rather treat the daily covariance matrix as OBSERVED and",
        "  forecast it directly, with no latent variance recursion to filter.",
    ])
    why = (f"With {freq} data you can compute the realised covariance each day and model it "
           "directly (multivariate HAR / VARFIMA) — the covariance is measured, not filtered "
           "from a latent GARCH recursion.")
    return text, M_RCOV, why


def _mg_baseline(rng):
    N = int(rng.integers(4, 12))
    text = "\n".join([
        f"Multivariate GARCH — {N}-asset baseline",
        "=" * 60,
        "  You want the cheapest model that still gives each asset its own",
        "  time-varying volatility. Correlations can stay fixed for now —",
        "  this is a quick first-pass benchmark, not the final model.",
    ])
    why = ("CCC is the cheap baseline: a univariate GARCH per asset for the variances and one "
           "fixed correlation matrix R. Its weakness (frozen correlations) is exactly what DCC "
           "later fixes — but as a first-pass benchmark it's the right call.")
    return text, M_CCC, why


_MGARCH_CASES = (_mg_feasible, _mg_constant_corr, _mg_posdef, _mg_realised, _mg_baseline)


def _mgarch_case(rng) -> dict:
    case = _MGARCH_CASES[int(rng.integers(len(_MGARCH_CASES)))]
    text, answer, why = case(rng)
    return {"topic": "MGARCH", "text": text,
            "questions": [_q("Which model fits?", answer, MGARCH_MODELS, rng)],
            "why": why}


# ===========================================================================
# Tail & extreme — which tail tool does the scenario call for?
# ===========================================================================
T_EVT = "Gaussian VaR understates the tail — model exceedances with EVT (POT / GPD)"
T_TSTUD = "Returns are iid but heavy-tailed — switch the distribution to Student-t"
T_MF = "Tail risk moves with volatility — use McNeil-Frey (GARCH, then EVT on the residuals)"
T_COPULA = "The danger is joint crashes — use a t / Gumbel copula with tail dependence"
T_OK = "Tails are close to normal — a Gaussian (delta-normal) VaR is adequate"


def _garch_t(rng, n, nu=5, omega=0.05, a=0.08, b=0.90):
    z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2))  # unit-variance innovations
    s2 = np.empty(n)
    eps = np.empty(n)
    s2[0] = omega / (1 - a - b)
    eps[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n):
        s2[t] = omega + a * eps[t - 1] ** 2 + b * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * z[t]
    return eps, np.sqrt(s2)


def _tail_undercoverage(rng):
    nu = int(rng.choice([3, 4, 5]))
    r = rng.standard_t(nu, size=3000)
    r = r / r.std()
    g99 = 2.326                                  # z_0.99 · σ, with σ = 1
    emp99 = float(-np.quantile(r, 0.01))
    breach = float(np.mean(r < -g99) * 100)
    text = "\n".join([
        "Single-asset tail check — iid returns (no volatility clustering)",
        "=" * 60,
        f"  Jarque-Bera           strongly rejects normality (Student-t({nu}))",
        "  Ljung-Box / ARCH-LM   ok  — residuals are serially clean",
        "",
        f"  99% Gaussian (delta-normal) VaR    {g99:.2f}",
        f"  99% empirical VaR                  {emp99:.2f}",
        f"  Days the Gaussian VaR is breached  {breach:.2f}%   (target 1.00%)",
    ])
    why = (f"The empirical 99% loss ({emp99:.2f}) sits well past the Gaussian VaR ({g99:.2f}), "
           f"so the normal VaR is breached ~{breach:.1f}% of the time instead of 1%. There's no "
           "ARCH and it's a single series, so the clean fix is a heavier-tailed conditional "
           "distribution — Student-t.")
    return text, T_TSTUD, [T_TSTUD, T_MF, T_COPULA, T_OK], why


def _tail_evt(rng):
    nu = int(rng.choice([3, 4, 5]))
    r = rng.standard_t(nu, size=4000)
    r = r / r.std()
    losses = -r
    u = float(np.quantile(losses, 0.95))
    exc = losses[losses > u]
    xi = float(np.mean(np.log(exc / u)))          # Hill estimator of the tail index ξ
    worst = float(losses.max())
    text = "\n".join([
        "Peaks-over-threshold — daily losses, 4000 days",
        "=" * 60,
        f"  Threshold  u (95th percentile)     {u:.2f}",
        f"  Exceedances above u                {exc.size}",
        f"  Hill tail index  ξ̂                 {xi:.2f}    (ξ>0 ⇒ power-law tail)",
        f"  Worst loss observed                {worst:.2f}",
        "",
        "  You must quote a 99.9% VaR — further into the tail than the worst",
        "  loss the sample has ever shown.",
    ])
    why = (f"ξ̂ = {xi:.2f} > 0 is a genuine power-law tail. A 99.9% VaR lies beyond the worst "
           f"observed loss ({worst:.2f}), where the empirical quantile is silent — fit a "
           "Generalised Pareto to the exceedances and let EVT extrapolate past the data.")
    return text, T_EVT, [T_EVT, T_TSTUD, T_MF, T_OK], why


def _tail_copula(rng):
    n = 3000
    nu = int(rng.choice([4, 5, 6]))
    rho = float(rng.choice([0.3, 0.4, 0.5]))
    z = rng.standard_normal((2, n))
    z[1] = rho * z[0] + np.sqrt(1 - rho ** 2) * z[1]
    s = np.sqrt(nu / rng.chisquare(nu, size=n))    # shared scale ⇒ Student-t copula
    a, b = z[0] * s, z[1] * s
    qa, qb = np.quantile(a, 0.05), np.quantile(b, 0.05)
    joint = int(np.sum((a < qa) & (b < qb)))
    indep = n * 0.05 * 0.05
    text = "\n".join([
        "Two-asset tail dependence — 3000 days",
        "=" * 60,
        f"  Linear correlation                 {rho:.2f}",
        "  Each asset breaches its own 5% VaR  ~150 days (by construction)",
        "",
        f"  Days BOTH breach on the SAME day   {joint}",
        f"  If the tails were independent       ≈ {indep:.0f}",
    ])
    why = (f"Both assets crash together {joint} times vs ≈{indep:.0f} under independence — far "
           "more co-crashing than the linear correlation alone implies. That's lower-tail "
           "dependence; a Gaussian copula has none, so model the joint tail with a t or Gumbel "
           "copula.")
    return text, T_COPULA, [T_COPULA, T_EVT, T_TSTUD, T_OK], why


def _tail_mcneilfrey(rng):
    eps, sig = _garch_t(rng, 2500, nu=5)
    longrun = float(np.sqrt(0.05 / (1 - 0.08 - 0.90)))
    var99 = float(-np.quantile(eps, 0.01))         # static / unconditional VaR
    hit = (eps < -var99).astype(int)
    clump = int(np.convolve(hit, np.ones(60, int), "valid").max())
    cur = float(sig[-1] / longrun)
    text = "\n".join([
        "Conditional tail risk — GARCH(1,1)-t returns, 2500 days",
        "=" * 60,
        f"  Static (unconditional) 99% VaR     {var99:.2f}",
        f"  Total VaR breaches                 {int(hit.sum())}",
        f"  Most that fall in any 60-day window {clump}",
        f"  Current conditional vol / long-run  {cur:.2f}×",
    ])
    why = (f"The breaches bunch — up to {clump} inside a single 60-day window — because a "
           "static VaR ignores volatility, which is currently "
           f"{cur:.1f}× its long-run level. McNeil-Frey filters with a GARCH first, then fits "
           "EVT to the standardised residuals, so the VaR widens in turbulence and the breaches "
           "stop clustering.")
    return text, T_MF, [T_MF, T_TSTUD, T_COPULA, T_OK], why


_TAIL_CASES = (_tail_undercoverage, _tail_evt, _tail_copula, _tail_mcneilfrey)


def _tail_case(rng) -> dict:
    case = _TAIL_CASES[int(rng.integers(len(_TAIL_CASES)))]
    text, answer, pool, why = case(rng)
    return {"topic": "Tail & extreme", "text": text,
            "questions": [_q("What's the right tool?", answer, pool, rng)],
            "why": why}


# ===========================================================================
TOPICS = ("VaR backtest", "MGARCH", "Tail & extreme")
_BUILDERS = {"VaR backtest": _var_case, "MGARCH": _mgarch_case, "Tail & extreme": _tail_case}


def risk_round(rng, topic: str = "Mixed") -> dict:
    """Build one risk-quiz round; ``topic`` is one of TOPICS or 'Mixed'."""
    if topic not in _BUILDERS:
        topic = TOPICS[int(rng.integers(len(TOPICS)))]
    return _BUILDERS[topic](rng)

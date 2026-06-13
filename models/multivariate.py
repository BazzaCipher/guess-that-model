"""Multivariate GARCH models — interactive correlation explorers."""
from __future__ import annotations

from demos.multivariate import (
    demo_bekk,
    demo_ccc,
    demo_cholesky,
    demo_dcc,
    demo_realised_cov,
    demo_vech,
)
from models.base import Model

_CAT = "Multivariate GARCH"

MODELS = [
    Model(
        key="vech",
        name="vech / diagonal vech",
        category=_CAT,
        equation=r"\mathrm{vech}(\Sigma_t) = C + A\,\mathrm{vech}(\varepsilon_{t-1}"
                 r"\varepsilon_{t-1}') + B\,\mathrm{vech}(\Sigma_{t-1})",
        summary="The fully general MGARCH: every covariance element has its own dynamics. "
                "Almost never positive-definite and unusable at scale; diagonal vech trims "
                "the parameter count.",
        tell="Parameter count grows like N⁴ — the cautionary tale that motivates BEKK/CCC/DCC.",
        demo=demo_vech,
    ),
    Model(
        key="bekk",
        name="BEKK",
        category=_CAT,
        equation=r"\Sigma_t = CC' + A\,\varepsilon_{t-1}\varepsilon_{t-1}'A' + "
                 r"B\,\Sigma_{t-1}B'",
        summary="The quadratic form guarantees a positive-definite Σ_t with no constraints — "
                "but full BEKK still has too many parameters to scale past a few assets.",
        tell="Positive-definite by construction; the scalar version has just two dynamic "
             "parameters yet a genuinely time-varying covariance.",
        demo=demo_bekk,
    ),
    Model(
        key="ccc",
        name="CCC (constant correlation)",
        category=_CAT,
        equation=r"\Sigma_t = D_t R D_t,\qquad D_t = \mathrm{diag}(\sigma_{1,t},\dots)",
        summary="Constant Conditional Correlation: univariate GARCH for each variance and a "
                "fixed correlation matrix R. Cheap, but constant correlation is the weak point.",
        tell="Volatilities move but correlations are frozen — fails exactly when correlations "
             "spike in a crisis.",
        demo=demo_ccc,
    ),
    Model(
        key="dcc",
        name="DCC (dynamic correlation)",
        category=_CAT,
        equation=r"Q_t = (1-a-b)\bar{Q} + a\,u_{t-1}u_{t-1}' + b\,Q_{t-1},\quad "
                 r"R_t = \tilde{Q}_t^{-1/2} Q_t \tilde{Q}_t^{-1/2}",
        summary="Engle's Dynamic Conditional Correlation fixes CCC's main flaw — correlations "
                "evolve through a GARCH-like recursion on standardised residuals.",
        tell="Two parameters (a, b) drive a time-varying correlation that rises in turbulent "
             "co-movements; a+b<1 keeps it mean-reverting.",
        demo=demo_dcc,
    ),
    Model(
        key="realised_cov",
        name="Realised covariance (MHAR / VARFIMA)",
        category=_CAT,
        equation=r"\mathrm{RCov}_t = \sum_i r_{t,i} r_{t,i}'",
        summary="Treats the daily covariance matrix as observable from intraday data, then "
                "models it directly with multivariate HAR (MHAR) or long-memory VARFIMA.",
        tell="No latent recursion — the covariance is measured, then forecast with ARMA/HAR "
             "dynamics element by element.",
        demo=demo_realised_cov,
    ),
    Model(
        key="cholesky",
        name="Cholesky decomposition",
        category=_CAT,
        equation=r"\Sigma = LL',\qquad x = L z,\ \ z \sim \mathcal{N}(0, I)",
        summary="The workhorse for enforcing positive-definiteness and generating correlated "
                "draws throughout MGARCH, Monte-Carlo VaR and copulas.",
        tell="Lower-triangular L turns independent shocks into correlated ones; the building "
             "block under almost every multivariate simulation.",
        demo=demo_cholesky,
    ),
]

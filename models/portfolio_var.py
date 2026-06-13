"""Portfolio VaR models — interactive risk-measure explorers."""
from __future__ import annotations

from demos.portfolio_var import (
    demo_component_var,
    demo_delta_gamma,
    demo_delta_normal,
    demo_etl,
    demo_historical,
    demo_mapping_pca,
    demo_monte_carlo,
)
from models.base import Model

_CAT = "Portfolio VaR"

MODELS = [
    Model(
        key="delta_normal",
        name="Delta-normal VaR",
        category=_CAT,
        equation=r"\mathrm{VaR}_\alpha = z_\alpha\,\sqrt{w'\Sigma w}\,\cdot V",
        summary="Variance–covariance VaR: fast and analytical, assuming linear positions and "
                "normally-distributed returns.",
        tell="Closed-form from the portfolio variance — but understates risk for options "
             "(non-linearity) and fat tails.",
        demo=demo_delta_normal,
    ),
    Model(
        key="delta_gamma",
        name="Delta-gamma VaR",
        category=_CAT,
        equation=r"\Delta P \approx \delta\,\Delta S + \tfrac{1}{2}\gamma\,\Delta S^2",
        summary="Adds a second-order (gamma) term for non-linear payoffs, capturing option "
                "convexity that delta-normal misses.",
        tell="The quadratic term skews the P&L distribution — the linear VaR misprices "
             "convex positions.",
        demo=demo_delta_gamma,
    ),
    Model(
        key="historical_sim",
        name="Historical simulation",
        category=_CAT,
        equation=r"\mathrm{VaR}_\alpha = -\,\mathrm{Quantile}_\alpha(\{P\&L_t\})",
        summary="No distributional assumption — read VaR straight off the empirical quantile "
                "of historical P&L; bootstrap for a confidence band.",
        tell="Faithful to the data but trapped in the sample window — blind to regimes that "
             "haven't occurred yet.",
        demo=demo_historical,
    ),
    Model(
        key="monte_carlo_var",
        name="Monte Carlo VaR",
        category=_CAT,
        equation=r"\{P\&L^{(s)}\}_{s=1}^{S} \sim \text{model};\ "
                 r"\mathrm{VaR}_\alpha = -\mathrm{Quantile}_\alpha",
        summary="Simulate forward P&L paths under an assumed model — constant-volatility or a "
                "GARCH engine that captures volatility clustering.",
        tell="Flexible for any payoff/dynamics; the GARCH version reports higher tail risk in "
             "clustered markets than constant-vol.",
        demo=demo_monte_carlo,
    ),
    Model(
        key="component_var",
        name="Incremental / marginal / component VaR",
        category=_CAT,
        equation=r"\mathrm{MVaR}_i = \frac{\partial \mathrm{VaR}}{\partial w_i},\quad "
                 r"\mathrm{CVaR}_i = w_i\,\mathrm{MVaR}_i,\quad \sum_i \mathrm{CVaR}_i = "
                 r"\mathrm{VaR}",
        summary="Decompose portfolio VaR by position: marginal (sensitivity), component "
                "(additive contribution) and incremental (effect of adding a trade).",
        tell="Component VaRs sum to the total — the basis for risk budgeting and position "
             "limits.",
        demo=demo_component_var,
    ),
    Model(
        key="etl",
        name="Expected shortfall (ETL)",
        category=_CAT,
        equation=r"\mathrm{ES}_\alpha = \mathbb{E}[\,L \mid L \ge \mathrm{VaR}_\alpha\,]",
        summary="The coherent tail measure: the average loss beyond VaR. Sub-additive, so "
                "diversification never appears to add risk.",
        tell="Always at least as large as VaR; it sees the *shape* of the tail, not just its "
             "threshold.",
        demo=demo_etl,
    ),
    Model(
        key="mapping_pca",
        name="Mapping — single-index / PCA / OGARCH",
        category=_CAT,
        equation=r"\Sigma = W \Lambda W',\qquad \text{factors } f = W'r",
        summary="Dimension reduction for large portfolios: map assets onto a single index or "
                "principal components; orthogonal-GARCH runs univariate GARCH on each factor.",
        tell="A few factors explain most of the covariance — the first principal component is "
             "the 'market'.",
        demo=demo_mapping_pca,
    ),
]

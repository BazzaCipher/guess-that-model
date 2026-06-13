"""Structural instability — regime and break models."""
from __future__ import annotations

from models.base import Model
from simulators import (
    simulate_markov_switch,
    simulate_setar,
    simulate_unit_root_breaks,
)

MODELS = [
    Model(
        key="unit_root_breaks",
        name="Unit root + breaks",
        category="Structural instability",
        family="Structural",
        equation=r"r_t = \mu_t + r_{t-1} + \varepsilon_t,\qquad "
                 r"\mu_t = \mu_0 + \textstyle\sum_k \delta_k \mathbb{1}_{t \ge \tau_k}",
        summary="A random walk whose intercept jumps at unknown break dates — non-stationarity "
                "and structural change tangled together.",
        tell="Looks I(1) but with abrupt level shifts; the breaks masquerade as persistence "
             "and bias unit-root tests toward non-rejection.",
        simulate=simulate_unit_root_breaks,
        trainer_eligible=True,
    ),
    Model(
        key="setar",
        name="SETAR (threshold AR)",
        category="Structural instability",
        family="Structural",
        equation=r"r_t = \begin{cases}\phi^{(1)} r_{t-1} + \varepsilon_t & r_{t-d} < c\\ "
                 r"\phi^{(2)} r_{t-1} + \varepsilon_t & r_{t-d} \ge c\end{cases}",
        summary="Self-Exciting Threshold AR: the regime is set by an OBSERVED variable "
                "crossing a threshold, so dynamics switch deterministically.",
        tell="Regime-dependent persistence — the autocorrelation differs in the two regimes, "
             "producing asymmetric, state-dependent behaviour.",
        simulate=simulate_setar,
        trainer_eligible=True,
    ),
    Model(
        key="markov_switch",
        name="Markov regime-switching",
        category="Structural instability",
        family="Structural",
        equation=r"r_t \mid S_t \sim \mathcal{N}(\mu_{S_t}, \sigma_{S_t}^2),\qquad "
                 r"\Pr(S_t = j \mid S_{t-1} = i) = p_{ij}",
        summary="Hamilton regime-switching: the regime is a HIDDEN probabilistic state "
                "following a Markov chain (vs SETAR's observed trigger).",
        tell="Persistent hidden variance regimes make the squared-returns ACF look GARCH-like "
             "— but the state is latent, not a deterministic recursion.",
        simulate=simulate_markov_switch,
        trainer_eligible=True,
    ),
]

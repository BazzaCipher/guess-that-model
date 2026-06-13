# Guess That Model!

An interactive **FNCE40003 econometrics study app**. A random process is
simulated each round and you guess the model from its time series and ACF/PACF —
plus a browsable inventory of the whole course model deck and interactive
explorers for the models that aren't a single-series guessing game.

Live: <https://guess-that-model.bazza.space>

## Three modes

- **🎯 Trainer** — the guessing game. Each round draws a process; you guess the
  *family* (Easy) or the *exact model* (Hard) from the series + correlograms,
  then it reveals the true parameters and the "tell".
- **📚 Inventory** — the full model deck, organised by the seven course blocks.
  Each card shows the equation, a one-line summary, and the diagnostic "tell";
  open a card to sample a path or run its interactive demo. Beyond-course
  material is hidden behind a toggle.
- **🔬 Explore** — one page per category whose models need sliders rather than a
  correlogram: regime switching, multivariate GARCH (CCC/DCC/BEKK), portfolio
  VaR, tail risk (EVT / copulas), MIDAS, PCA mapping.

## Model deck

| Block | Models |
|--|--|
| Conditional mean | White noise, MA(q), AR(p), ARMA(p,q), ARIMA (unit root), AR(1)-GARCH, MIDAS, single-index/market model |
| Structural instability | Unit root + breaks, SETAR (threshold AR), Markov regime-switching, Bai-Perron breakpoints |
| Univariate volatility | ARCH(q), GARCH(p,q), GJR/TARCH, EGARCH, GARCH-t, APARCH, GARCH-M, IGARCH, FIGARCH, *GARCH-MIDAS* |
| Realised vol & long memory | HAR-RV, HAR-RV-J, ARMA on log RV, ARFIMA |
| Multivariate GARCH | vech / diagonal vech, BEKK, CCC, DCC, realised covariance (MHAR/VARFIMA), Cholesky |
| Portfolio VaR | delta-normal, delta-gamma, historical simulation, Monte Carlo (const + GARCH), incremental/marginal/component, ETL, mapping (single-index/PCA/OGARCH) |
| Tail & extreme | EVT-GPD (POT), McNeil-Frey, copulas (Gaussian/t/Gumbel), *vine copulas*, *Oh-Patton factor copula* |

*Italic* = flagged "beyond this course".

## Architecture

One **data-driven registry**: every model is a `Model` record (`models/`) with
metadata plus an optional `simulate` (series models) and/or `demo` (interactive
widget). The three views (`views/`) are just render modes over that one list.

```
app.py            st.navigation shell over the three views
views/            trainer · inventory · explorers (the only Streamlit besides demos/)
models/           one module per category → Model records → REGISTRY
simulators.py     pure simulate_* functions (no Streamlit)
generators.py     hand-rolled numerical kernels (ARFIMA, Markov, DCC, copulas, …)
demos/            interactive Streamlit demos, one module per category
plots.py          figure-returning matplotlib helpers
```

No heavy dependencies: everything is an `arch` class or hand-rolled
numpy/scipy (no scikit-learn, copula, or break-detection packages).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python smoke_test.py   # registry integrity + every simulator finite + every demo/page renders headless
```

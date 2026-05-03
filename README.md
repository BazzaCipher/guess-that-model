# Guess That Model!

Practice tool: a random GARCH or HAR process is simulated each round; you
guess the model class from the time-series and ACF/PACF.

Live: <https://guess-that-model.bazza.space>

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model deck

| Family | Models |
|--|--|
| GARCH | GARCH(1,1) Normal, GARCH(2,1) Normal, GJR-GARCH(1,1), EGARCH(1,1), GARCH(1,1) Student-t |
| HAR | HAR-RV (Corsi), HAR-RV-J |
| Other | ARMA(1,1) on log RV, white noise |

## Controls

- **Difficulty** — *Easy* asks for the family; *Hard* asks for the exact model.
- **ACF/PACF target** — default shows raw + squared returns (and RV when applicable);
  other modes restrict to a single view, or use *Auto* for the canonical view of
  whatever family was drawn.
- **Sample length** — 500 to 10,000 simulated observations.
- **Lags** — 10 to 60 lags on the correlogram.

After each guess the app reveals the true model, the parameters drawn for that
round, and a one-line "tell".

# luz-forecast

Day-ahead electricity price forecasting for Spain: what will each hour of
tomorrow cost on the OMIE day-ahead market?

The pipeline downloads the public market data itself, trains a model only on
information that existed before the real decision deadline (the day-ahead
gate closes at 12:00 CET on D-1), and evaluates on a walk-forward split so
no future information leaks into training. Predictions come with intervals,
not just point estimates.

## Data

- **Prices**: OMIE `marginalpdbc` daily files (public, no key) - the
  day-ahead clearing prices for the Spanish zone. Since the Oct 2025 MTU
  change the files carry 96 quarter-hour periods; older days have 24 hours.
  `data/omie.py` keeps `data/raw/omie_prices.csv` up to date incrementally.
- **Drivers** (planned): REData (Red Electrica) public API - demand, wind
  and solar generation, which are known/predictable before the gate closure.

## Status

Just started: price history backfill done (2023-01-01 to today, ~1,400
days). Feature engineering, models and walk-forward evaluation next.

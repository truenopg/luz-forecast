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
- **Demand**: REData (Red Electrica) public API, `demanda-tiempo-real` -
  hourly measured demand (`real`) and REE's own forecast (`prevista`).
  `data/redata.py` keeps `data/raw/ree_demand.csv` up to date the same
  incremental way (the API throws transient 400s, so it retries with
  backoff and picks up failed days on the next run).
- **Generation mix** (wind/solar): planned next - the REData generation
  endpoints return errors for now; until that is sorted, demand + calendar
  + lagged prices carry the features.

Two data notes: 28-29 Apr 2025 is the Iberian blackout (demand collapses
in the series - real event, kept as-is), and REE's `prevista` values are
revised intraday, so at feature-build time only the forecast as known at
the D-1 12:00 gate closure may be used (or lag it), never the final value.

## Status

Data layer done: prices 2023-01-01 -> today (~1,400 days, 96 periods since
the Oct 2025 MTU change) and hourly demand real+forecast for the same
window. First EDA in `report/` (figures in `report/figures/`): strong
weekly and seasonal cycles, solar duck curve in spring/summer, ~2%
negative-price periods, same-day demand explains only part of the level
(r ~ 0.37), so generation mix and calendar will matter. Feature
engineering, models and walk-forward evaluation next.

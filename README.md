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

## Baseline model

`model/features.py` builds the leakage-safe table (one row per date/hour):
D-1 full price curve (published on D-2), D-7/D-14 same-hour prices, D-7
same-hour demand, D-1 morning demand (hours 0-11 only), calendar. REE's
`prevista` series stays out: the public API serves revised values with no
point-in-time vintage (checked 2026-11-10), so it can only appear in
`--exploratory` runs, which are labelled as such.

`model/train.py` compares a gradient-boosted model against the naive
"same hour last week" benchmark in monthly walk-forward folds (train on
everything before the month, test the month, roll). Results in
`report/metrics.json`: overall MAE 20.5 EUR/MWh vs 29.4 naive (RMSE 28.4
vs 41.7), and the model wins every month until autumn 2026, when the price
regime jumps (~250 EUR/MWh peaks) and the expanding window stops adapting -
there it ties or loses to the naive. Short-window retraining and regime
features are the obvious next step. `--exploratory` adds the revised
prevista demand features: MAE 20.3, a small gain that cannot be trusted
without vintage data.

## Status

Data layer and leakage-safe baseline done; walk-forward shows the model
beats the naive benchmark except in the autumn-2026 price spike. Next:
regime handling (shorter windows, recent-level features), then prediction
intervals and a daily prediction job. Generation mix features still
blocked on the REData generation endpoints.

"""Feature table for the day-ahead baseline.

One row per (date, hour) with the target price and only information that
exists before the D-1 12:00 CET gate closure:

- D-1 full price curve: published by OMIE on D-2 ~13:30, so fully known.
- D-1 demand: only hours 0-10. Measured 2026-11-10: at 11:29 CET the
  latest published 5-min value was timestamped 11:20 (~9 min latency),
  so hour 11's mean is not fully published at the 12:00 gate closure.
- D-7 / D-14 same-hour price and D-7 same-hour demand.
- Calendar: hour, weekday, month, Spanish national holidays (approx list).

REE's `prevista` demand series is revised intraday and has no point-in-time
vintage on the public API (checked 2026-11-10), so it stays out of this
table; train.py can add it for a clearly-labelled exploratory run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

HOLIDAYS = {
    # national holidays, good enough as a coarse regime flag
    "2023-01-06", "2023-04-07", "2023-05-01", "2023-08-15", "2023-10-12",
    "2023-11-01", "2023-12-06", "2023-12-08", "2023-12-25",
    "2024-01-01", "2024-01-06", "2024-03-29", "2024-05-01", "2024-08-15",
    "2024-10-12", "2024-11-01", "2024-12-06", "2024-12-25",
    "2025-01-01", "2025-01-06", "2025-04-18", "2025-05-01", "2025-08-15",
    "2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08", "2025-12-25",
    "2026-01-01", "2026-01-06", "2026-04-03", "2026-05-01", "2026-08-15",
    "2026-10-12", "2026-12-07", "2026-12-08", "2026-12-25",
}


def hourly_prices() -> pd.Series:
    """(date, hour) -> EUR/MWh, quarter-hour days collapsed to hourly."""
    raw = pd.read_csv(ROOT / "data/raw/omie_prices.csv", parse_dates=["date"])
    n_periods = raw.groupby("date")["period"].transform("max")
    hour = raw["period"] - 1
    hour = hour.where(n_periods <= 25, (raw["period"] - 1) // 4)
    raw = raw.assign(hour=hour)
    s = (raw.groupby(["date", "hour"])["price_eur_mwh"].mean())
    return s


def hourly_demand(series: str) -> pd.Series:
    d = pd.read_csv(ROOT / "data/raw/ree_demand.csv", parse_dates=["date"])
    d = d[d.series == series]
    return d.set_index(["date", "hour"])["mw"]


def build(exploratory: bool = False) -> pd.DataFrame:
    px = hourly_prices()
    dem = hourly_demand("real")

    df = px.rename("price").reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "hour"]).set_index(["date", "hour"])

    def lag(s: pd.Series, days: int, name: str) -> pd.Series:
        out = s.rename(name).reset_index()
        out["date"] = pd.to_datetime(out["date"]) + pd.Timedelta(days=days)
        return out.set_index(["date", "hour"])[name]

    def lag_daily(s: pd.Series, days: int) -> pd.Series:
        shifted = s.copy()
        shifted.index = pd.to_datetime(shifted.index) + pd.Timedelta(days=days)
        return shifted.reindex(df.index.get_level_values("date")).to_numpy()

    df["price_d1"] = lag(px, 1, "x")
    df["price_d7"] = lag(px, 7, "x")
    df["price_d14"] = lag(px, 14, "x")
    df["demand_d7"] = lag(dem, 7, "x")

    px_daily = px.groupby("date").mean()
    df["price_d1_mean"] = lag_daily(px_daily, 1)
    df["price_mean_d7"] = lag_daily(px_daily.rolling(7).mean(), 1)  # recent level
    dem_morning = dem[dem.index.get_level_values("hour") < 11].groupby("date").mean()
    df["demand_d1_morning"] = lag_daily(dem_morning, 1)

    if exploratory:
        fc = hourly_demand("prevista")
        df["demand_fcst_h"] = fc.reindex(df.index)  # REVISED intraday, not vintage
        df["demand_fcst_d7"] = lag(fc, 7, "x")

    df["dow"] = df.index.get_level_values("date").dayofweek
    df["month"] = df.index.get_level_values("date").month
    df["hour"] = df.index.get_level_values("hour")
    df["is_holiday"] = [str(d.date()) in HOLIDAYS for d in df.index.get_level_values("date")]
    df["is_weekend"] = df["dow"] >= 5

    return df.dropna()

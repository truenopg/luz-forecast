"""EDA: how the day-ahead price behaves and what drives it.

Reads data/raw/omie_prices.csv and data/raw/ree_demand.csv, writes figures
to report/figures/ and prints the summary stats that decide the feature set.
No notebooks - this is the reproducible analysis script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

prices = pd.read_csv(ROOT / "data/raw/omie_prices.csv", parse_dates=["date"])
px = prices.pivot(index="date", columns="period", values="price_eur_mwh")
px.columns = [f"h{c}" for c in px.columns]  # periods are 1..24 hours or 1..96 quarter-hours

dem = pd.read_csv(ROOT / "data/raw/ree_demand.csv", parse_dates=["date"])
real = dem[dem.series == "real"].pivot(index="date", columns="hour", values="mw")
fcst = dem[dem.series == "prevista"].pivot(index="date", columns="hour", values="mw")

daily = pd.DataFrame({
    "price_mean": px.mean(axis=1),
    "price_peak": px.iloc[:, :24].max(axis=1) if px.shape[1] >= 24 else px.mean(axis=1),
    "demand_mean": real.mean(axis=1),
    "demand_peak": real.max(axis=1),
    "demand_fcst_peak": fcst.max(axis=1),
}).dropna()
daily["dow"] = daily.index.dayofweek
daily["month"] = daily.index.month

print("=== price (EUR/MWh) ===")
print(px.stack().describe().round(2))
neg = (px < 0).mean().mean()
print(f"share of negative periods: {neg:.2%}")

print("\n=== correlation with same-day demand ===")
print(daily[["price_mean", "price_peak", "demand_mean", "demand_peak",
             "demand_fcst_peak"]].corr()["price_mean"].round(3))

fig, ax = plt.subplots(figsize=(10, 4))
daily["price_mean"].plot(ax=ax, lw=0.6)
daily["price_mean"].rolling(30).mean().plot(ax=ax, lw=1.5, color="black")
ax.set_title("day-ahead price, daily mean (30d rolling in black)")
ax.set_ylabel("EUR/MWh")
fig.tight_layout()
fig.savefig(FIG / "price_daily.png", dpi=110)

# average intraday shape, old 24-period days only (shapes are comparable)
px24 = px.iloc[:, :24].loc[px.index < "2025-10-01"]
shape = px24.groupby(px24.index.month).mean()
shape.index.name = "month"
fig, ax = plt.subplots(figsize=(10, 4))
shape.T.plot(ax=ax, colormap="turbo", lw=0.8)
ax.set_title("average hourly price by month (pre-15min-MTU days)")
ax.set_xlabel("hour")
ax.set_ylabel("EUR/MWh")
fig.tight_layout()
fig.savefig(FIG / "intraday_by_month.png", dpi=110)

fig, ax = plt.subplots(figsize=(7, 4))
daily.boxplot(column="price_mean", by="dow", ax=ax, showfliers=False)
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_title("daily mean price by weekday")
ax.set_xlabel("")
fig.suptitle("")
fig.tight_layout()
fig.savefig(FIG / "price_by_weekday.png", dpi=110)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(daily["demand_peak"], daily["price_peak"], s=2, alpha=0.3)
ax.set_xlabel("peak demand (MW)")
ax.set_ylabel("peak price (EUR/MWh)")
ax.set_title("peak price vs peak demand, same day")
fig.tight_layout()
fig.savefig(FIG / "price_vs_demand.png", dpi=110)

print(f"\nfigures -> {FIG}")

"""Baseline model with walk-forward evaluation.

Naive benchmark: price D-7 same hour. Model: HistGradientBoosting on the
leakage-safe feature table. Walk-forward: train on everything up to the
month boundary, test the next month, roll forward. Metrics per fold and
overall go to report/metrics.json, figures to report/figures/.

--exploratory adds the REE `prevista` demand features (revised intraday,
NO point-in-time vintage) for a separated comparison only - never shipped
as the leakage-safe baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from features import build

FIG = ROOT / "report" / "figures"


def walk_forward(df: pd.DataFrame, window_months: int | None = None,
                 quantile: float | None = None) -> pd.DataFrame:
    """One row per test (date, hour) with naive and model predictions.
    window_months: rolling train window (None = expanding). quantile: fit a
    quantile GBM instead of the mean (for prediction intervals)."""
    feats = [c for c in df.columns if c != "price"]
    dates = df.index.get_level_values("date")
    months = pd.period_range(dates.min().to_period("M") + 24,
                             dates.max().to_period("M"), freq="M")
    preds = []
    for m in months:
        cut = m.start_time
        end = min(m.end_time, dates.max())
        train = df.loc[dates < cut]
        if window_months:
            train = train.loc[train.index.get_level_values("date")
                              >= cut - pd.DateOffset(months=window_months)]
        test = df.loc[(dates >= cut) & (dates <= end)]
        if len(test) == 0 or len(train) < 2000:
            continue
        params = {"max_iter": 300, "learning_rate": 0.06}
        if quantile is not None:
            params.update({"loss": "quantile", "quantile": quantile})
        gbm = HistGradientBoostingRegressor(**params,
                                            categorical_features=[feats.index(c) for c in
                                                                  ("dow", "month", "hour")
                                                                  if c in feats])
        gbm.fit(train[feats], train["price"])
        out = test[["price", "price_d7"]].copy()
        out["pred"] = gbm.predict(test[feats])
        out = out.rename(columns={"price_d7": "naive"})
        out["fold"] = str(m)
        preds.append(out)
    return pd.concat(preds)


def metrics(preds: pd.DataFrame, col: str) -> dict:
    err = preds[col] - preds["price"]
    return {"mae": round(mean_absolute_error(preds["price"], preds[col]), 2),
            "rmse": round(root_mean_squared_error(preds["price"], preds[col]), 2),
            "bias": round(float(err.mean()), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exploratory", action="store_true",
                    help="add REE prevista features (NOT vintage, exploratory only)")
    ap.add_argument("--window-months", type=int, default=None,
                    help="rolling train window in months (default: expanding)")
    ap.add_argument("--intervals", action="store_true",
                    help="also fit 10/90% quantile models for prediction intervals")
    args = ap.parse_args()

    df = build(exploratory=args.exploratory)
    print(f"rows: {len(df)}, span: {df.index.get_level_values('date').min().date()} -> "
          f"{df.index.get_level_values('date').max().date()}")

    preds = walk_forward(df, window_months=args.window_months)
    by_fold = preds.groupby("fold").apply(
        lambda g: pd.Series({"naive_mae": metrics(g, "naive")["mae"],
                             "gbm_mae": metrics(g, "pred")["mae"]}),
        include_groups=False)
    print(by_fold.round(2))
    print("\noverall naive:", metrics(preds, "naive"))
    print("overall gbm:  ", metrics(preds, "pred"))

    suffix = ("_exploratory" if args.exploratory else "") + \
             (f"_w{args.window_months}" if args.window_months else "")
    out = {"label": "EXPLORATORY (prevista revised intraday, no vintage)"
           if args.exploratory else "leakage-safe baseline",
           "n_rows": len(preds),
           "naive": metrics(preds, "naive"), "gbm": metrics(preds, "pred"),
           "folds": by_fold.round(2).reset_index().to_dict("records")}
    (ROOT / "report" / f"metrics{suffix}.json").write_text(json.dumps(out, indent=2))

    daily = preds.groupby(preds.index.get_level_values("date"))[["price", "pred"]].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    daily["price"].plot(ax=ax, lw=0.8, label="actual")
    daily["pred"].plot(ax=ax, lw=0.8, label="gbm")
    ax.set_title(f"walk-forward: daily mean price, actual vs gbm{suffix.replace('_', ' ')}")
    ax.set_ylabel("EUR/MWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"walkforward_daily{suffix}.png", dpi=110)

    hourly = (preds.assign(abs_err=(preds["pred"] - preds["price"]).abs(),
                           naive_err=(preds["naive"] - preds["price"]).abs())
              .groupby(preds.index.get_level_values("hour"))[["abs_err", "naive_err"]].mean())
    fig, ax = plt.subplots(figsize=(8, 4))
    hourly.rename(columns={"abs_err": "gbm", "naive_err": "naive"}).plot(ax=ax)
    ax.set_title(f"MAE by hour: gbm vs naive{suffix.replace('_', ' ')}")
    ax.set_xlabel("hour")
    ax.set_ylabel("MAE (EUR/MWh)")
    fig.tight_layout()
    fig.savefig(FIG / f"mae_by_hour{suffix}.png", dpi=110)

    if args.intervals:
        lo = walk_forward(df, args.window_months, quantile=0.1).rename(columns={"pred": "p10"})
        hi = walk_forward(df, args.window_months, quantile=0.9).rename(columns={"pred": "p90"})
        band = pd.concat([lo[["price", "p10", "fold"]], hi["p90"]], axis=1)
        inside = (band["price"] >= band["p10"]) & (band["price"] <= band["p90"])
        print(f"\n80% interval raw: coverage {inside.mean():.1%} (target 80%), "
              f"mean width {(band['p90'] - band['p10']).mean():.1f} EUR/MWh")

        # split-conformal fix: at each fold, widen the band by the 80% quantile
        # of the under-coverage errors seen in all previous test folds (known
        # at prediction time - walk-forward legitimate). First fold uses the
        # last training months' in-sample residuals as a rough prior.
        miss = np.maximum(band["p10"] - band["price"], band["price"] - band["p90"]).clip(lower=0)
        adj = pd.Series(index=band.index, dtype=float)
        folds = band["fold"].unique()
        for i, f in enumerate(folds):
            cur = band["fold"] == f
            past = miss[band["fold"].isin(folds[:i])]
            delta = np.quantile(past, 0.8) if len(past) > 500 else 15.0
            adj[cur] = delta
        band["p10_adj"] = band["p10"] - adj
        band["p90_adj"] = band["p90"] + adj
        inside_adj = (band["price"] >= band["p10_adj"]) & (band["price"] <= band["p90_adj"])
        print(f"80% interval conformal: coverage {inside_adj.mean():.1%}, "
              f"mean width {(band['p90_adj'] - band['p10_adj']).mean():.1f} EUR/MWh")
        out["interval_80"] = {"raw_coverage": round(float(inside.mean()), 3),
                              "conformal_coverage": round(float(inside_adj.mean()), 3),
                              "conformal_mean_width": round(float((band["p90_adj"] - band["p10_adj"]).mean()), 1)}
        (ROOT / "report" / f"metrics{suffix}.json").write_text(json.dumps(out, indent=2))

        daily = band.groupby(band.index.get_level_values("date"))[["price", "p10_adj", "p90_adj"]].mean()
        recent = daily.loc[daily.index >= daily.index.max() - pd.Timedelta(days=120)]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(recent.index, recent["p10_adj"], recent["p90_adj"], alpha=0.3,
                        label="80% interval (conformal)")
        recent["price"].plot(ax=ax, lw=1, color="black", label="actual")
        ax.set_title(f"last 120 walk-forward days: actual vs 80% interval{suffix.replace('_', ' ')}")
        ax.set_ylabel("EUR/MWh")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / f"intervals{suffix}.png", dpi=110)
        print(f"-> {FIG / f'intervals{suffix}.png'}")
    print(f"metrics -> report/metrics{suffix}.json, figures -> {FIG}")


if __name__ == "__main__":
    main()

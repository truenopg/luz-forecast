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
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from features import build

FIG = ROOT / "report" / "figures"


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """One row per test (date, hour) with naive and model predictions."""
    feats = [c for c in df.columns if c != "price"]
    months = pd.period_range(df.index.get_level_values("date").min().to_period("M") + 24,
                             df.index.get_level_values("date").max().to_period("M"), freq="M")
    preds = []
    for m in months:
        cut = m.start_time
        end = min(m.end_time, df.index.get_level_values("date").max())
        train = df.loc[df.index.get_level_values("date") < cut]
        test = df.loc[(df.index.get_level_values("date") >= cut)
                      & (df.index.get_level_values("date") <= end)]
        if len(test) == 0 or len(train) < 2000:
            continue
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
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
    args = ap.parse_args()

    df = build(exploratory=args.exploratory)
    print(f"rows: {len(df)}, span: {df.index.get_level_values('date').min().date()} -> "
          f"{df.index.get_level_values('date').max().date()}")

    preds = walk_forward(df)
    by_fold = preds.groupby("fold").apply(
        lambda g: pd.Series({"naive_mae": metrics(g, "naive")["mae"],
                             "gbm_mae": metrics(g, "pred")["mae"]}),
        include_groups=False)
    print(by_fold.round(2))
    print("\noverall naive:", metrics(preds, "naive"))
    print("overall gbm:  ", metrics(preds, "pred"))

    suffix = "_exploratory" if args.exploratory else ""
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
    print(f"metrics -> report/metrics{suffix}.json, figures -> {FIG}")


if __name__ == "__main__":
    main()

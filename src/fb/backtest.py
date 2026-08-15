"""Milestone 2 - which choice actually inflates a forecasting score?

The README claims that a random train/test split flatters a forecaster. That is
the standard warning, and I wrote it before testing it. Thinking it through, it
might not be the whole story: if the features are strictly past lags, then a
randomly held-out row still only asks for a ONE-STEP-AHEAD prediction with the
previous hour known. That is easy, but it is easy because of the horizon, not
because of the split.

So this varies two things independently instead of assuming:

  split      random rows  vs  temporal (every test row after every train row)
  horizon    predict y[t] from lags ending at t-1   (h = 1)
             vs predict y[t] from lags ending at t-24  (h = 24)

Four combinations, one model, one feature builder. Whichever factor moves the
error is the one worth warning people about.

Everything is scored against SEASONAL NAIVE on the same rows - predict this
hour with the same hour last week. A forecaster that cannot beat that has not
earned its complexity, and reporting raw MAE hides it. Because meter scales
span 5,332x (milestone 1), errors are aggregated as a skill ratio per series
and then averaged, never as a pooled MAE, which would just report on the
largest meters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "hourly.parquet"
REPORTS = ROOT / "reports"

LAGS = (1, 2, 3, 24, 25, 168, 169)
SEASON = 168          # weekly seasonality, the strongest honest naive here


def build(series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Features available `horizon` hours before the target.

    Every lag is shifted by (horizon - 1) extra steps, so at h=24 the most
    recent input is 24 hours old. This is the only difference between the
    horizon settings - same model, same lags, same rows.
    """
    d = series.sort_values("timestamp").reset_index(drop=True)
    out = pd.DataFrame({"timestamp": d.timestamp, "y": d.kwh})
    shift = horizon - 1
    for lag in LAGS:
        out[f"lag_{lag}"] = d.kwh.shift(lag + shift)
    out["naive"] = d.kwh.shift(SEASON + shift)     # seasonal naive, same info set
    out["hour"] = d.timestamp.dt.hour
    out["dow"] = d.timestamp.dt.dayofweek
    return out.dropna().reset_index(drop=True)


def split_random(n: int, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    test = np.zeros(n, dtype=bool)
    test[rng.choice(n, size=int(n * frac), replace=False)] = True
    return test


def split_temporal(n: int, frac: float) -> np.ndarray:
    test = np.zeros(n, dtype=bool)
    test[int(n * (1 - frac)):] = True
    return test


def evaluate_series(df: pd.DataFrame, horizon: int, how: str,
                    frac: float = 0.2, seed: int = 0) -> dict | None:
    d = build(df, horizon)
    if len(d) < 2000:
        return None
    feats = [c for c in d.columns if c.startswith("lag_")] + ["hour", "dow"]
    test = split_random(len(d), frac, seed) if how == "random" else split_temporal(len(d), frac)

    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed)
    m.fit(d.loc[~test, feats], d.loc[~test, "y"])
    pred = m.predict(d.loc[test, feats])

    y = d.loc[test, "y"].to_numpy()
    mae_model = float(np.mean(np.abs(y - pred)))
    mae_naive = float(np.mean(np.abs(y - d.loc[test, "naive"].to_numpy())))

    # MASE denominator: in-sample seasonal-naive MAE on the TRAINING rows, at a
    # fixed lag. This has to be independent of `horizon`, otherwise the yardstick
    # moves with the thing being measured - my first version divided by seasonal
    # naive on the test rows, whose own lag grows with the horizon, which made
    # h=24 look BETTER than h=1 purely because the baseline had degraded too.
    tr = df.sort_values("timestamp").kwh.to_numpy(dtype=float)
    cut = int(len(tr) * 0.8)
    insample = np.abs(tr[SEASON:cut] - tr[: cut - SEASON])
    denom = float(np.mean(insample)) if len(insample) else np.nan

    return {"mae_model": mae_model, "mae_naive": mae_naive,
            "skill": mae_model / mae_naive if mae_naive > 0 else np.nan,
            "mase": mae_model / denom if denom and denom > 0 else np.nan,
            "mase_naive": mae_naive / denom if denom and denom > 0 else np.nan,
            "n_test": int(test.sum()), "scale": float(y.mean())}


def run(n_series: int = 40, seed: int = 0) -> dict:
    df = pd.read_parquet(PARQUET)
    names = df.series.unique()
    rng = np.random.default_rng(seed)
    pick = rng.choice(names, size=min(n_series, len(names)), replace=False)

    cells = {}
    for how in ("random", "temporal"):
        for horizon in (1, 24):
            rows = []
            for s in pick:
                r = evaluate_series(df[df.series == s], horizon, how, seed=seed)
                if r:
                    rows.append(r)
            skills = np.array([r["skill"] for r in rows])
            mase = np.array([r["mase"] for r in rows])
            mase_n = np.array([r["mase_naive"] for r in rows])
            cells[f"{how}_h{horizon}"] = {
                "split": how, "horizon": horizon, "series": len(rows),
                "mase_median": float(np.nanmedian(mase)),
                "mase_naive_median": float(np.nanmedian(mase_n)),
                "skill_median": float(np.nanmedian(skills)),
                "beats_naive_frac": float(np.mean(skills < 1)),
            }
            c = cells[f"{how}_h{horizon}"]
            print(f"  {how:9} h={horizon:<3} MASE {c['mase_median']:.4f}   "
                  f"(seasonal naive {c['mase_naive_median']:.4f})   "
                  f"beats naive on {c['beats_naive_frac']:.0%}")

    out = {"n_series": int(len(pick)), "seed": seed, "cells": cells}
    # which factor moved the number more?
    base = cells["random_h1"]["mase_median"]
    out["effect_of_split"] = cells["temporal_h1"]["mase_median"] - base
    out["effect_of_horizon"] = cells["random_h24"]["mase_median"] - base
    out["baseline_mase"] = base
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "backtest.json").write_text(json.dumps(out, indent=1))
    return out


def demo() -> None:
    """Self-check the split and feature machinery on constructed data."""
    n = 1000
    t = pd.date_range("2011-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": t, "kwh": np.arange(n, dtype=float)})

    # h=1: most recent lag is 1 step back. h=24: it is 24 steps back.
    b1, b24 = build(df, 1), build(df, 24)
    i = 300
    assert b1.loc[i, "y"] - b1.loc[i, "lag_1"] == 1.0
    assert b24.loc[i, "y"] - b24.loc[i, "lag_1"] == 24.0, b24.loc[i, "lag_1"]
    # seasonal naive must sit exactly SEASON steps back at h=1
    assert b1.loc[i, "y"] - b1.loc[i, "naive"] == float(SEASON)
    # no NaNs survive
    assert not b1.isna().any().any() and not b24.isna().any().any()

    # temporal split puts every test index after every train index
    te = split_temporal(100, 0.2)
    assert te[:80].sum() == 0 and te[80:].all()
    # random split holds out the right count and is not contiguous
    tr = split_random(1000, 0.2, 0)
    assert tr.sum() == 200 and tr[:200].sum() != 200
    print("self-check ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-series", type=int, default=40)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        demo()
        return

    print("MASE = model MAE / in-sample seasonal-naive MAE (fixed denominator).")
    print("Lower is better; 1.0 means no better than naive.\n")
    o = run(a.n_series)
    b = o["baseline_mase"]
    print(f"\nfrom the flattering cell (random split, h=1, MASE {b:.4f}):")
    print(f"  changing the SPLIT   -> {o['effect_of_split']:+.4f} MASE "
          f"({o['effect_of_split']/b:+.1%})")
    print(f"  changing the HORIZON -> {o['effect_of_horizon']:+.4f} MASE "
          f"({o['effect_of_horizon']/b:+.1%})")


if __name__ == "__main__":
    main()

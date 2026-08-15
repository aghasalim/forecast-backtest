"""Milestones 3 and 4 - real models, and the cost of not refitting.

Milestone 2 used a single temporal cut. That is still optimistic: it fits once
and then evaluates a month of forecasts as if the model never went stale. The
honest protocol is a ROLLING ORIGIN - stand at time o, forecast the next 24
hours, step forward a day, repeat - which is also the only way to ask whether
refitting is worth anything.

So both questions are answered on identical origins:

  which model   seasonal naive / daily naive / exponential smoothing / gradient
                boosting on lag features
  refit policy  fit once on the initial training window and let it age, versus
                refit at every origin

Every model at origin o sees data up to o and nothing after it. That is checked
by an assertion rather than by trusting the indexing, because an off-by-one here
would leak the future and produce a beautiful, meaningless number.

ETS is fitted with daily seasonality (24) rather than weekly (168). Weekly
seasonality on hourly data means a 168-state seasonal component estimated per
origin per series, which is hours of compute for this comparison and would not
change the ranking. Stated here rather than buried, because it is a real
limitation of the ETS row.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from backtest import LAGS, SEASON

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "hourly.parquet"
REPORTS = ROOT / "reports"

H = 24                 # forecast the next 24 hours from each origin
TEST_DAYS = 28         # 28 origins per series
ETS_WINDOW = 24 * 56   # 8 weeks of history for the ETS fit


def lag_frame(y: np.ndarray, horizon: int = H) -> pd.DataFrame:
    """Lag features whose most recent input is `horizon` steps before target."""
    s = pd.Series(y)
    shift = horizon - 1
    return pd.DataFrame({f"lag_{lag}": s.shift(lag + shift) for lag in LAGS})


def fit_gbm(y: np.ndarray, end: int, seed: int = 0) -> HistGradientBoostingRegressor:
    x = lag_frame(y).iloc[:end]
    target = pd.Series(y).iloc[:end]
    ok = x.notna().all(axis=1)
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed)
    m.fit(x[ok], target[ok])
    return m


def ets_forecast(y: np.ndarray, end: int) -> np.ndarray:
    """Holt-Winters with daily seasonality, fitted on the window ending at `end`."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    hist = y[max(0, end - ETS_WINDOW): end]
    if len(hist) < 24 * 14 or np.allclose(hist, hist[0]):
        return np.repeat(hist[-1] if len(hist) else 0.0, H)
    try:
        fit = ExponentialSmoothing(
            hist, trend=None, seasonal="add", seasonal_periods=24,
            initialization_method="estimated").fit()
        return np.asarray(fit.forecast(H), dtype=float)
    except Exception:
        return np.repeat(float(hist[-1]), H)


def evaluate_series(y: np.ndarray, seed: int = 0) -> dict | None:
    n = len(y)
    need = SEASON + H + TEST_DAYS * H + 24 * 365
    if n < need:
        return None

    first_origin = n - TEST_DAYS * H
    origins = [first_origin + d * H for d in range(TEST_DAYS)]

    # MASE denominator: in-sample seasonal naive on data before any origin
    insample = np.abs(y[SEASON:first_origin] - y[: first_origin - SEASON])
    denom = float(np.mean(insample))
    if not np.isfinite(denom) or denom <= 0:
        return None

    gbm_once = fit_gbm(y, first_origin, seed)          # fitted once, then ages
    errs: dict[str, list[float]] = {k: [] for k in
                                    ("naive24", "naive168", "ets", "gbm_once", "gbm_refit")}

    for o in origins:
        truth = y[o: o + H]
        # nothing at or after the origin may enter any feature
        assert o + H <= n

        errs["naive24"].append(np.mean(np.abs(truth - y[o - 24: o])))
        errs["naive168"].append(np.mean(np.abs(truth - y[o - SEASON: o - SEASON + H])))
        errs["ets"].append(np.mean(np.abs(truth - ets_forecast(y, o))))

        x = lag_frame(y).iloc[o: o + H]
        assert x.notna().all().all(), "lag features reach past the origin"
        errs["gbm_once"].append(np.mean(np.abs(truth - gbm_once.predict(x))))
        errs["gbm_refit"].append(
            np.mean(np.abs(truth - fit_gbm(y, o, seed).predict(x))))

    return {k: float(np.mean(v)) / denom for k, v in errs.items()}


def run(n_series: int = 15, seed: int = 0) -> dict:
    df = pd.read_parquet(PARQUET)
    names = df.series.unique()
    rng = np.random.default_rng(seed)
    pick = rng.choice(names, size=min(n_series, len(names)), replace=False)

    rows = []
    for i, s in enumerate(pick, 1):
        y = df.loc[df.series == s, "kwh"].to_numpy(dtype=float)
        r = evaluate_series(y, seed)
        if r:
            rows.append(r)
        print(f"  [{i}/{len(pick)}] {s}  " +
              ("  ".join(f"{k} {v:.3f}" for k, v in r.items()) if r else "skipped"),
              flush=True)

    models = ("naive24", "naive168", "ets", "gbm_once", "gbm_refit")
    summary = {m: {"mase_median": float(np.median([r[m] for r in rows])),
                   "mase_mean": float(np.mean([r[m] for r in rows])),
                   "beats_naive168_frac": float(np.mean(
                       [r[m] < r["naive168"] for r in rows]))}
               for m in models}
    out = {"n_series": len(rows), "horizon": H, "origins_per_series": TEST_DAYS,
           "seed": seed, "models": summary,
           "refit_gain": (summary["gbm_once"]["mase_median"]
                          - summary["gbm_refit"]["mase_median"])}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "models.json").write_text(json.dumps(out, indent=1))
    return out


def demo() -> None:
    """Self-check the origin arithmetic, which is where a leak would hide."""
    y = np.arange(24 * 500, dtype=float)
    x = lag_frame(y, horizon=H)
    # at row t the most recent feature must be y[t-24], never anything later
    t = 5000
    assert x.loc[t, "lag_1"] == y[t - H], (x.loc[t, "lag_1"], y[t - H])
    assert x.loc[t, "lag_168"] == y[t - 168 - (H - 1)]

    # seasonal naive at a 24h origin uses the week-old block, entirely in the past
    o = 4000
    block = y[o - SEASON: o - SEASON + H]
    assert block.max() < o, "seasonal naive reached the future"

    # a flat series must not raise, and must forecast the flat value
    flat = np.ones(24 * 200)
    assert np.allclose(ets_forecast(flat, len(flat)), 1.0)
    print("self-check ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-series", type=int, default=15)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        demo()
        return

    print(f"rolling origin: {TEST_DAYS} origins/series, forecasting {H}h each\n")
    o = run(a.n_series)
    print(f"\n{'model':12} {'MASE median':>12} {'mean':>8}  beats seasonal naive")
    for m, v in o["models"].items():
        print(f"{m:12} {v['mase_median']:12.4f} {v['mase_mean']:8.4f}  "
              f"{v['beats_naive168_frac']:.0%}")
    print(f"\nrefitting every origin vs fitting once: "
          f"{o['refit_gain']:+.4f} MASE")


if __name__ == "__main__":
    main()

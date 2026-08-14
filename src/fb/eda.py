"""Milestone 1 - the autocorrelation that decides how you must evaluate.

The interesting fact about this dataset is not its seasonality, it is how
predictable each point is from its immediate neighbours. That single number
determines whether a random train/test split measures forecasting at all.

If consecutive hours correlate at 0.9-plus, then holding out a random 20% of
rows leaves most held-out points sandwiched between two training points. The
model does not have to forecast, it has to interpolate, and interpolation
between two nearly-identical values is trivial. The score comes back excellent
and means nothing.

That is quantifiable before any model exists:

  P(both temporal neighbours of a held-out point are in train) = (1 - h)^2

for a random split holding out fraction h. At h = 0.2 that is 0.64 - roughly
two thirds of the test set is an interpolation problem wearing a forecast's
clothes.

Also measured here because it changes the metrics later:

  scale spread   these meters differ by orders of magnitude, so an MAE averaged
                 across series is really a report on the largest few. Scale-free
                 metrics (MASE) are not a stylistic preference here.
  seasonality    daily and weekly cycles set what a *fair* naive baseline is.
                 Beating a naive forecast that ignores them is not an
                 achievement; beating seasonal naive is the bar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "hourly.parquet"
REPORTS = ROOT / "reports"

LAGS = {"1h": 1, "24h (daily)": 24, "168h (weekly)": 168}


def autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 2:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run(sample_series: int = 60, seed: int = 0) -> dict:
    df = pd.read_parquet(PARQUET)
    names = df.series.unique()
    rng = np.random.default_rng(seed)
    pick = rng.choice(names, size=min(sample_series, len(names)), replace=False)

    acf = {k: [] for k in LAGS}
    scales, lengths = [], []
    for s in pick:
        x = df.loc[df.series == s, "kwh"].to_numpy(dtype=float)
        lengths.append(len(x))
        scales.append(float(np.mean(x)))
        for k, lag in LAGS.items():
            acf[k].append(autocorr(x, lag))

    acf_summary = {k: {"median": float(np.nanmedian(v)),
                       "p10": float(np.nanpercentile(v, 10)),
                       "p90": float(np.nanpercentile(v, 90))}
                   for k, v in acf.items()}

    # the leakage arithmetic, computed not asserted
    leak = {f"holdout_{int(h*100)}pct": (1 - h) ** 2 for h in (0.1, 0.2, 0.3)}

    out = {
        "series_total": int(len(names)), "series_sampled": int(len(pick)),
        "rows": int(len(df)),
        "hours_per_series": {"min": int(min(lengths)), "median": int(np.median(lengths)),
                             "max": int(max(lengths))},
        "mean_kwh_per_series": {
            "min": float(np.min(scales)), "median": float(np.median(scales)),
            "max": float(np.max(scales)),
            "max_over_min": float(np.max(scales) / max(np.min(scales), 1e-9))},
        "autocorrelation": acf_summary,
        "prob_both_neighbours_in_train": leak,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "eda.json").write_text(json.dumps(out, indent=1))
    figure(df, pick[:1][0], acf, out)
    return out


def figure(df: pd.DataFrame, one: str, acf: dict, out: dict) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    s = df[df.series == one].set_index("timestamp").kwh
    s.iloc[: 24 * 21].plot(ax=ax[0])
    ax[0].set_title(f"{one}: three weeks\ndaily and weekly cycles")
    ax[0].set_ylabel("kWh")

    ax[1].boxplot([np.array(v)[~np.isnan(v)] for v in acf.values()],
                  tick_labels=list(acf))
    ax[1].axhline(0.9, color="#c0392b", ls="--", label="0.9")
    ax[1].set_title("autocorrelation across series\n(lag-1 is why random splits leak)")
    ax[1].legend()

    prof = (df.assign(hour=df.timestamp.dt.hour)
              .groupby("hour").kwh.mean())
    ax[2].plot(prof.index, prof.to_numpy(), "o-")
    ax[2].set_title("mean load by hour of day")
    ax[2].set_xlabel("hour")

    fig.suptitle("Electricity load, 351 meters, hourly")
    fig.tight_layout()
    fig.savefig(REPORTS / "eda.png", dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-series", type=int, default=60)
    o = run(p.parse_args().sample_series)

    print(f"{o['series_total']} series, {o['rows']:,} hourly rows "
          f"({o['series_sampled']} sampled for autocorrelation)\n")
    print(f"{'lag':16} {'median':>8} {'p10':>8} {'p90':>8}")
    for k, v in o["autocorrelation"].items():
        print(f"{k:16} {v['median']:8.4f} {v['p10']:8.4f} {v['p90']:8.4f}")

    sc = o["mean_kwh_per_series"]
    print(f"\nseries scale: min {sc['min']:.2f}, median {sc['median']:.2f}, "
          f"max {sc['max']:.2f} kWh  ->  largest/smallest = {sc['max_over_min']:,.0f}x")
    print("an MAE averaged over series is a report on the biggest meters\n")

    print("if you hold out a random fraction h, the chance BOTH neighbours of a")
    print("held-out point are in train is (1-h)^2:")
    for k, v in o["prob_both_neighbours_in_train"].items():
        print(f"  {k:18} {v:.2f}")


if __name__ == "__main__":
    main()

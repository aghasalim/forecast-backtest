"""Milestone 1a - 678 MB of semicolon-separated text into hourly Parquet.

LD2011_2014.txt is 370 electricity meters at 15-minute resolution, written in
the European convention: `;` as the field separator and `,` as the decimal
point. Read it with default settings and every value silently becomes a string.

Two decisions made here rather than left implicit:

  hourly    The raw data is 15-minute. Forecasting is done hourly because the
            question this project asks is about evaluation protocol, and 4x the
            rows buys nothing for it. Summing the four quarters preserves total
            energy; taking the mean would preserve rate. Sum is used, and the
            column is named kwh so the choice is visible at the call site.

  the zero  Many meters were installed partway through 2011 and record exactly
  prefix    0 until then. Those are not observations of zero demand, they are
            absence of a meter, and averaging them into a baseline would drag
            it down. Each series is trimmed to its first non-zero reading, and
            the trimmed amount is reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "LD2011_2014.txt"
OUT = ROOT / "data" / "hourly.parquet"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, sep=";", decimal=",", index_col=0, low_memory=False)
    df.index = pd.to_datetime(df.index)
    assert df.index.is_monotonic_increasing, "timestamps are not sorted"
    # every column must be numeric after the decimal="," hint; if any survived
    # as object the separator convention was wrong and everything downstream
    # would be quietly meaningless
    bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    assert not bad, f"non-numeric columns after parse: {bad[:5]}"
    return df


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1h").sum()


def trim_zero_prefix(s: pd.Series) -> tuple[pd.Series, int]:
    nz = s.to_numpy().nonzero()[0]
    if len(nz) == 0:
        return s.iloc[0:0], len(s)
    return s.iloc[nz[0]:], int(nz[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-series", type=int, default=None,
                   help="keep only the N longest series (default: all 370)")
    a = p.parse_args()

    raw = load_raw()
    hourly = to_hourly(raw)

    rows, frames = [], []
    for col in hourly.columns:
        s, trimmed = trim_zero_prefix(hourly[col])
        if len(s) < 24 * 365:                     # need a year to talk about seasonality
            rows.append({"series": col, "kept": len(s), "trimmed": trimmed,
                         "dropped": True})
            continue
        frames.append(pd.DataFrame({"series": col, "timestamp": s.index,
                                    "kwh": s.to_numpy()}))
        rows.append({"series": col, "kept": len(s), "trimmed": trimmed,
                     "dropped": False})

    if a.max_series:
        frames = sorted(frames, key=len, reverse=True)[: a.max_series]
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False, compression="zstd")

    kept = [r for r in rows if not r["dropped"]]
    meta = {
        "series_total": len(rows), "series_kept": len(kept),
        "series_dropped_short": len(rows) - len(kept),
        "rows": int(len(out)),
        "trimmed_zero_prefix_median_hours": int(
            pd.Series([r["trimmed"] for r in kept]).median()),
        "trimmed_zero_prefix_max_hours": int(max(r["trimmed"] for r in kept)),
        "raw_mb": round(RAW.stat().st_size / 1e6, 1),
        "parquet_mb": round(OUT.stat().st_size / 1e6, 1),
        "t_start": str(out.timestamp.min()), "t_end": str(out.timestamp.max()),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "prepare.json").write_text(json.dumps(meta, indent=1))
    for k, v in meta.items():
        print(f"{k:34} {v}")


if __name__ == "__main__":
    main()

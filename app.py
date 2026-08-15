"""Demo: what actually inflates a forecasting score.

This project was named after a claim — that shuffling a time series is what
flatters your model — and then measured it and found the claim mostly wrong.
The app exists to let you see that rather than take my word for it: the grid is
the real measured output, and the two factors can be toggled independently.

Everything is read from the committed run artefacts (reports/*.json), so the
numbers here are the same ones in the README, not a re-run on a subsample.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Forecast backtesting", page_icon="📉", layout="wide")


@st.cache_data
def load(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


grid = load("backtest.json")
models = load("models.json")

st.title("📉 What actually inflates a forecasting score?")
st.caption(
    "351 electricity meters, hourly. MASE is model error ÷ in-sample "
    "seasonal-naive error: lower is better, 1.0 means no better than predicting "
    "this hour with the same hour last week."
)

if grid:
    st.subheader("The two factors, varied independently")
    st.write(
        "Same model, same features, same rows. Only one thing changes at a time."
    )
    c1, c2 = st.columns(2)
    with c1:
        split = st.radio("Train/test split", ["random", "temporal"], horizontal=True,
                         help="random = the shuffled split every tutorial uses. "
                              "temporal = every test row after every training row.")
    with c2:
        horizon = st.radio("Forecast horizon", [1, 24], horizontal=True,
                           format_func=lambda h: f"{h} hour ahead"
                           if h == 1 else f"{h} hours ahead")

    cell = grid["cells"][f"{split}_h{horizon}"]
    base = grid["cells"]["random_h1"]["mase_median"]
    delta = cell["mase_median"] - base

    m1, m2, m3 = st.columns(3)
    m1.metric("MASE (median)", f"{cell['mase_median']:.4f}",
              f"{delta:+.4f} vs the flattering cell" if delta else "the flattering cell",
              delta_color="inverse")
    m2.metric("Seasonal naive on the same rows", f"{cell['mase_naive_median']:.4f}")
    m3.metric("Beats seasonal naive", f"{cell['beats_naive_frac']:.0%}")

    st.dataframe(pd.DataFrame([
        {"split": v["split"], "horizon": f"{v['horizon']}h",
         "model MASE": round(v["mase_median"], 4),
         "seasonal naive": round(v["mase_naive_median"], 4),
         "beats naive": f"{v['beats_naive_frac']:.0%}"}
        for v in grid["cells"].values()
    ]), hide_index=True, use_container_width=True)

    st.info(
        f"**Changing the split costs {grid['effect_of_split'] / base:+.1%}. "
        f"Changing the horizon costs {grid['effect_of_horizon'] / base:+.1%}.** "
        "The horizon matters about ten times more than the thing this repo was "
        "originally named after. With strictly past-lag features the model never "
        "interpolates — it only ever sees earlier values, whichever rows are held "
        "out — so shuffling hands it nothing it did not already have."
    )

if models:
    st.divider()
    st.subheader("Real models, under a rolling origin")
    st.write(
        f"{models['n_series']} meters, {models['origins_per_series']} origins each, "
        f"forecasting {models['horizon']}h from every origin. Each model sees only "
        "data before its origin — enforced by the harness handing over a prefix "
        "slice, not by remembering to be careful."
    )
    rows = [{"model": k,
             "MASE (median)": round(v["mase_median"], 4),
             "MASE (mean)": round(v["mase_mean"], 4),
             "beats seasonal naive": f"{v['beats_naive168_frac']:.0%}"}
            for k, v in models["models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    gain = models["refit_gain"]
    st.metric("Refitting at every origin vs fitting once",
              f"{gain:+.4f} MASE",
              help="Positive means fitting once was worse, i.e. refitting helped.")

with st.expander("What this does not show"):
    st.markdown("""
- **One dataset, one year, one country.** The ranking of models here should not
  be assumed to transfer; the protocol lesson should.
- **ETS uses daily seasonality (24), not weekly (168).** Weekly seasonality on
  hourly data means a 168-state seasonal component re-estimated at every origin,
  which is hours of compute and would not change the ordering. It is a real
  limitation of that row.
- **Prediction-interval coverage is deliberately absent.** It is the subject of
  [m4-forecasting](https://github.com/aghasalim/m4-forecasting); duplicating it
  here would be padding.
- **The first version of this project was wrong**, and the README says so. The
  measured grid above is what changed my mind.
""")

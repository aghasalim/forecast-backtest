# What actually inflates a forecasting score — the split, or the horizon?

Built by a third-year Applied Computer Science (AI) student.

> **Status: complete.** All six milestones. This project set out to demonstrate
> two things and **measured both to be false** — the split barely matters, and
> refitting barely matters. Those are the results. The decision trail, including
> every wrong turn, is in [NOTES.md](NOTES.md).

Almost every forecasting tutorial does the same thing: shuffle the rows, hold
out 20%, report a small error. The standard warning is that this leaks, because
if you hold out a random fraction `h`, the chance that **both** neighbours of a
held-out point are still in training is `(1 - h)²` — **64%** at the usual
`h = 0.2`. Two thirds of your test set sits between two known values.

I built this project to demonstrate that. **Then I measured it, and it is not
the main problem.**

Holding the model, features and rows fixed and changing one factor at a time:

| what changes | effect on MASE |
|---|---|
| random split → temporal split (h=1) | **+4.1%** |
| horizon 1h → 24h (random split) | **+42.1%** |

The split moves the score by 4%. The forecast horizon moves it by 42%, ten
times more. The interpolation arithmetic is correct and the conclusion I drew
from it was wrong: with strictly past-lag features the model never gets to
interpolate, because it only ever sees earlier values no matter which rows are
held out. What it does get, in both splits, is *the previous hour* — and that
is what makes the task easy.

So the honest warning is not "don't shuffle your time series." It is **"a
1-step-ahead score is not evidence you can forecast 24 hours out,"** and that
holds whichever way you split.

## The data

351 electricity meters, hourly, 2011–2015, 10.3M rows
([UCI ElectricityLoadDiagrams](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)).

| lag | median autocorrelation | p10 |
|---|---|---|
| **1 hour** | **0.9213** | 0.8605 |
| 24 hours (daily) | 0.9248 | 0.6578 |
| 168 hours (weekly) | 0.9059 | 0.7886 |

Consecutive hours correlate at **0.92**. That is what makes a 1-step-ahead
forecast easy — the previous hour is nearly the answer — and it is available to
the model under *either* split, which is precisely why the split turned out to
matter so little and the horizon so much.

![the data](reports/eda.png)

## The full grid

MASE = model MAE ÷ in-sample seasonal-naive MAE. Lower is better; 1.0 means no
better than predicting this hour with the same hour last week. 40 meters.

| split | horizon | model MASE | seasonal naive | beats naive |
|---|---|---|---|---|
| random | 1h | **0.4843** | 1.0030 | 100% |
| temporal | 1h | 0.5040 | 0.9545 | 100% |
| random | 24h | 0.6882 | 1.5604 | 100% |
| temporal | 24h | **0.7217** | 1.5248 | 100% |

The most flattering cell and the most honest cell differ by **49%** in MASE,
and almost all of that is the horizon.

Seasonal naive scores **1.0030** at h=1, and it has to be ~1.0 by construction
because it *is* the denominator. That is the check that the metric is right
rather than merely plausible.

### The metric I had to fix first

My first version divided by seasonal naive computed **on the test rows**. At
h=24 that baseline uses a lag of 191 instead of 168, so it degrades along with
the model — and h=24 came out looking *better* than h=1, which is impossible.
The yardstick was moving with the thing being measured. MASE with a fixed
in-sample denominator removes it. The numbers above are from the corrected
metric; the confounded ones are in [NOTES.md](NOTES.md).

## Real models, under a rolling origin

14 meters, 28 origins each, forecasting 24 hours from every origin. Each model
sees only data before its origin — enforced by the harness handing over a prefix
slice, not by remembering to shift correctly.

| model | MASE median | MASE mean | beats seasonal naive |
|---|---|---|---|
| daily naive (`y[t-24]`) | 1.1568 | 1.3254 | 50% |
| seasonal naive (`y[t-168]`) | 1.1319 | 1.2159 | 0% |
| exponential smoothing (ETS) | **1.2791** | 1.4754 | 29% |
| gradient boosting, fit once | 1.0771 | 1.2195 | 79% |
| gradient boosting, refit daily | **1.0741** | 1.2217 | 79% |

**Every model scores MASE above 1.** That does not mean they lose to seasonal
naive on the same rows — gradient boosting beats it on 79% of meters. It means
the final 28 days are harder than the training period the denominator was
computed on. Both facts are true and only reporting the second one would be
flattering.

**ETS is the worst thing here**, worse than repeating last week. Classical
exponential smoothing with daily seasonality has no way to represent the weekly
cycle these meters are dominated by, and it pays for that.

`seasonal naive` beats seasonal naive on 0% of series, which it must, because it
*is* seasonal naive. That column is a self-check, not a result.

## Refitting is worth almost nothing here

Milestone 4 existed to show that a single temporal cut is optimistic compared to
refitting as time advances. Measured:

| policy | MASE median |
|---|---|
| fit once, then let it age 28 days | 1.0771 |
| refit at every one of 28 origins | 1.0741 |

**+0.0030 MASE, or 0.3%**, for 28× the compute. The second premise this project
was built on, also not supported.

The honest reading is that a month is simply not long enough for a gradient
boosting model on lag features to go stale on this data — the features are
recent lags, which carry their own recency. It would be easy to present the
0.3% as "refitting helps"; it is a rounding error, and the interesting version
of this result is that you can skip the retraining pipeline here and lose
nothing measurable.

### These numbers are not comparable to the grid above

Milestone 2 scored the last 20% of every series (about 291 days) with a MASE
denominator from the first 80%. Milestone 3 scores the last 28 days with a
denominator from everything before them, on 14 meters rather than 40. Different
window, different denominator, different sample. Comparing 0.72 against 1.08 and
concluding something changed would be wrong — the split/horizon comparison is
internally consistent, and so is the model comparison, but not with each other.

## Two other things measured now because they constrain what comes later

**Series scales span 5,332×** — from 15.6 to 82,974 mean kWh. An MAE averaged
across series is therefore a report on the largest few meters and nothing else.
Scale-free errors (MASE) are a requirement here, not a stylistic preference.

**Daily and weekly autocorrelation are both ~0.9**, which sets the honest
baseline. Beating a naive forecast that ignores seasonality proves nothing; the
bar is *seasonal naive* — predict this hour with the same hour last week. In
forecasting it is very common for elaborate models to lose to it, and I would
rather find that out in milestone 2 than discover it after building something.

## A data decision that would have moved every result

Many meters were installed partway through the record and log exactly `0` until
then. That is absence of a meter, not zero demand, and averaging it into a
baseline drags the baseline down invisibly.

Each series is trimmed to its first non-zero reading. The **median trimmed
prefix is 8,760 hours** — half the meters were installed a full year in. Interior
zeros are kept, because those are real readings; there is a self-check asserting
exactly that distinction.

## Reproduce

```bash
uv sync
curl -L -o data/electricity.zip \
  https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip -q data/electricity.zip -d data/

uv run python src/fb/prepare.py     # 711 MB text -> 52.8 MB parquet
uv run python src/fb/eda.py         # -> reports/eda.{json,png}
uv run python src/fb/backtest.py    # the two-factor grid -> reports/backtest.json
uv run python src/fb/models.py      # rolling-origin model comparison (~30 min)
uv run streamlit run app.py         # the demo
```

Self-checks, which assert each function is right on signals whose answer is
known — a pure 24-period sine must autocorrelate at ~1 at lag 24 and ~−1 at lag
12 — and need no dataset:

```bash
uv run python src/fb/prepare.py --self-check
uv run python src/fb/eda.py --self-check
uv run python src/fb/backtest.py --self-check
uv run python src/fb/models.py --self-check
uv run python src/fb/harness.py --self-check   # proves a cheating forecaster cannot cheat
```

## Roadmap

- [x] **1 — Data and the deciding statistic.** Prepare 351 series, measure the
      autocorrelation that makes random splits leak, and the scale spread that
      makes MASE mandatory.
- [x] **2 — The two-factor test.** Split and horizon varied independently
      against seasonal naive. Horizon dominates by 10x; the project's original
      premise was wrong and is corrected above.
- [x] **3 — Models.** Naive, seasonal naive, ETS and gradient boosting under a
      rolling origin. Every model above MASE 1; ETS worst.
- [x] **4 — Rolling-origin refits.** Measured at +0.3% MASE for 28x the
      compute. The premise was not supported. (Prediction-interval coverage is
      deliberately *not* here — it is the subject of
      [m4-forecasting](https://github.com/aghasalim/m4-forecasting), and
      duplicating it would be padding.)
- [x] **5 — Deployment.** A backtester that hands each model a prefix slice, so
      leaking the future is impossible by construction; plus a Streamlit demo
      and Docker image.
- [x] **6 — Docs.** This README and the decision trail in [NOTES.md](NOTES.md),
      with both refuted premises kept in.

## What I would do next

1. **Longer staleness window.** Refitting bought 0.3% over 28 days. The honest test is
   a year, where the meter's own behaviour drifts; a month was too short to
   answer the question I was asking.
2. **A model that can hold weekly seasonality.** ETS lost because 24-period
   seasonality cannot represent a 168-hour cycle. SARIMA or an ETS with weekly
   seasonality would be the fair classical comparator, at much higher cost.
3. **Per-series rather than pooled conclusions.** MASE median hides that some
   meters are forecastable and others are close to noise; the 79% figure implies
   21% where boosting loses.
4. **Longer horizons.** 24h was chosen because it is the operational one. The
   horizon effect was the dominant factor, so mapping MASE against h properly is
   the obvious next measurement.

## Stack

Python 3.12, pandas, NumPy, statsmodels, scikit-learn, matplotlib, PyArrow.
Managed with `uv`, linted with `ruff`.

## Data

[UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014),
CC BY 4.0. My code is MIT.

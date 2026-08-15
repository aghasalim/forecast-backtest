# What actually inflates a forecasting score — the split, or the horizon?

Built by a third-year Applied Computer Science (AI) student.

> **Status: in progress.** Milestones 1–2 done. Milestone 2 tested the claim this
> project was named after and **refuted it** — the headline below is the corrected
> version. Every number came out of the code in this repo.

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
```

Self-checks, which assert each function is right on signals whose answer is
known — a pure 24-period sine must autocorrelate at ~1 at lag 24 and ~−1 at lag
12 — and need no dataset:

```bash
uv run python src/fb/prepare.py --self-check
uv run python src/fb/eda.py --self-check
uv run python src/fb/backtest.py --self-check
```

## Roadmap

- [x] **1 — Data and the deciding statistic.** Prepare 351 series, measure the
      autocorrelation that makes random splits leak, and the scale spread that
      makes MASE mandatory.
- [x] **2 — The two-factor test.** Split and horizon varied independently
      against seasonal naive. Horizon dominates by 10x; the project's original
      premise was wrong and is corrected above.
- [ ] **3 — Models.** Seasonal naive, exponential smoothing, gradient boosting
      on lag features — all under the honest protocol.
- [ ] **4 — Rolling-origin refits.** A single temporal cut is still optimistic
      versus refitting as time advances; that gap is the remaining piece.
      (Prediction-interval coverage is deliberately *not* here — it is the
      subject of [m4-forecasting](https://github.com/aghasalim/m4-forecasting),
      and duplicating it would be padding.)
- [ ] **5 — Deployment.** Backtesting harness plus a demo.
- [ ] **6 — Docs.** Write-up and decision trail.

## Stack

Python 3.12, pandas, NumPy, statsmodels, scikit-learn, matplotlib, PyArrow.
Managed with `uv`, linted with `ruff`.

## Data

[UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014),
CC BY 4.0. My code is MIT.

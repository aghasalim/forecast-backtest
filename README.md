# What actually inflates a forecasting score, the split, or the horizon?

[![ci](https://img.shields.io/badge/ci-passing-brightgreen.svg)](.github/workflows/)
[![licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

---

## Abstract

The standard warning about evaluating time-series models on a random split is
that it leaks: holding out a random fraction`h` leaves both temporal neighbours
of a held-out point in training with probability`(1-h)^2`, which is 64% at the
usual`h = 0.2`. This project was built to demonstrate that, and measured it to be
the smaller of two effects. Holding model, features and rows fixed and varying one
factor at a time across 40 series, moving from a random to a temporal split costs
4.1% MASE, while extending the horizon from one hour to 24 costs 42.1%, ten times
more.

The interpolation arithmetic is correct; the conclusion drawn from it was not.
With strictly past-lag features the model never gets to interpolate, because it
only ever sees earlier values regardless of which rows are held out. What it does
get in both splits is the previous hour, and that is what the horizon takes away.
A second hypothesis, that refitting at every origin matters, was also measured
and also came out negative, at 0.003 MASE for an order of magnitude more compute.

**Contributions.** (i) A factorial measurement separating split from horizon on
the same series, features and seeds. (ii) Two negative results reported as
negative, with the reasoning that produced the wrong expectation left in
[NOTES.md](NOTES.md). (iii) A refit ablation showing origin-by-origin refitting is
not worth its cost here.

---

## 1. Introduction

Almost every forecasting tutorial does the same thing: shuffle the rows, hold
out 20%, report a small error. The standard warning is that this leaks, because
if you hold out a random fraction`h`, the chance that **both** neighbours of a
held-out point are still in training is`(1 - h)²`, **64%** at the usual
`h = 0.2`. Two thirds of your test set sits between two known values.

I built this project to demonstrate that. **Then I measured it, and it is not
the main problem.**

Holding the model, features and rows fixed and changing one factor at a time:

| what changes | effect on MASE |
|---|---|
| random split → temporal split (h=1) | **+4.1%** |
| horizon 1h → 24h (random split) | **+42.1%** |

![the 2x2: split barely moves the score, horizon moves it ten times more](reports/backtest.png)

Left is the full 2×2. The two split lines nearly overlap at both horizons, and
they climb together, the leakage everyone warns about is the small gap between
them, while the thing that actually decides the score is how far ahead you are
asked to predict. Redrawn from`reports/backtest.json` by`python -m fb.figures`,
so it cannot drift from the table above it.

The split moves the score by 4%. The forecast horizon moves it by 42%, ten
times more. The interpolation arithmetic is correct and the conclusion I drew
from it was wrong: with strictly past-lag features the model never gets to
interpolate, because it only ever sees earlier values no matter which rows are
held out. What it does get, in both splits, is *the previous hour*, and that
is what makes the task easy.

So the honest warning is not "don't shuffle your time series." It is **"a
1-step-ahead score is not evidence you can forecast 24 hours out,"** and that
holds whichever way you split.

## 2. The data
This is the premise the project was built on, and it is arithmetically fine.

![autocorrelation and the interpolation probability](reports/premise.png)
![the data](reports/eda.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#2-the-data).
## 3. The full grid
MASE compares against a within-series scaling, which does not by itself say the model is useful.

![skill against the seasonal naive in every cell](reports/skill.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#3-the-full-grid).
### The metric I had to fix first

My first version divided by seasonal naive computed **on the test rows**. At
h=24 that baseline uses a lag of 191 instead of 168, so it degrades along with
the model, and h=24 came out looking *better* than h=1, which is impossible.
The yardstick was moving with the thing being measured. MASE with a fixed
in-sample denominator removes it. The numbers above are from the corrected
metric; the confounded ones are in [NOTES.md](NOTES.md).

## 4. Real models, under a rolling origin
Only the gradient-boosted models beat the weekly-naive baseline, on 79% of series.

![five models and the refit that buys nothing](reports/models.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#4-real-models-under-a-rolling-origin).
## 5. Refitting is worth almost nothing here
Milestone 4 existed to show that a single temporal cut is optimistic compared to refitting as time advances.

Full detail in [notes/METHODS.md](notes/METHODS.md#5-refitting-is-worth-almost-nothing-here).
### These numbers are not comparable to the grid above

Milestone 2 scored the last 20% of every series (about 291 days) with a MASE
denominator from the first 80%. Milestone 3 scores the last 28 days with a
denominator from everything before them, on 14 meters rather than 40. Different
window, different denominator, different sample. Comparing 0.72 against 1.08 and
concluding something changed would be wrong, the split/horizon comparison is
internally consistent, and so is the model comparison, but not with each other.

## 6. Two other things measured now because they constrain what comes later
**Series scales span 5,332×**: from 15.6 to 82,974 mean kWh.

Full detail in [notes/METHODS.md](notes/METHODS.md#6-two-other-things-measured-now-because-they-constrain-what-comes-later).
## 7. A data decision that would have moved every result

Many meters were installed partway through the record and log exactly`0` until
then. That is absence of a meter, not zero demand, and averaging it into a
baseline drags the baseline down invisibly.

Each series is trimmed to its first non-zero reading. The **median trimmed
prefix is 8,760 hours**, half the meters were installed a full year in. Interior
zeros are kept, because those are real readings; there is a self-check asserting
exactly that distinction.

## 8. Reproducibility

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
known, a pure 24-period sine must autocorrelate at ~1 at lag 24 and ~−1 at lag
12, and need no dataset:

```bash
uv run python src/fb/prepare.py --self-check
uv run python src/fb/eda.py --self-check
uv run python src/fb/backtest.py --self-check
uv run python src/fb/models.py --self-check
uv run python src/fb/harness.py --self-check   # proves a cheating forecaster cannot cheat
```

## 9. Roadmap
- [x] **1, Data and the deciding statistic.** Prepare 351 series, measure the autocorrelation that makes random splits leak, and the scale spread that makes MASE mandatory.

Full detail in [notes/METHODS.md](notes/METHODS.md#9-roadmap).
## 10. What I would do next
**Longer staleness window.** Refitting bought 0.3% over 28 days.

Full detail in [notes/METHODS.md](notes/METHODS.md#10-what-i-would-do-next).
## 11. Stack

Python 3.12, pandas, NumPy, statsmodels, scikit-learn, matplotlib, PyArrow.
Managed with`uv`, linted with`ruff`.

## 12. Data source

[UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014),
CC BY 4.0. My code is MIT.

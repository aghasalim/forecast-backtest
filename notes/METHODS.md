# Methods and detail

Long form detail moved out of the README.


## 2. The data


![autocorrelation and the interpolation probability](../reports/premise.png)

This is the premise the project was built on, and it is arithmetically fine.
Consecutive hours correlate at 0.92, and at a 20% hold-out 64% of test points sit
between two training points. The mistake was assuming that made the split the
dominant factor.

351 electricity meters, hourly, 2011 to 2015, 10.3M rows
([UCI ElectricityLoadDiagrams](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)).

| lag | median autocorrelation | p10 |
|---|---|---|
| **1 hour** | **0.9213** | 0.8605 |
| 24 hours (daily) | 0.9248 | 0.6578 |
| 168 hours (weekly) | 0.9059 | 0.7886 |

Consecutive hours correlate at **0.92**. That is what makes a 1-step-ahead
forecast easy, the previous hour is nearly the answer, and it is available to
the model under *either* split, which is precisely why the split turned out to
matter so little and the horizon so much.

![the data](../reports/eda.png)


## 3. The full grid


![skill against the seasonal naive in every cell](../reports/skill.png)

MASE compares against a within-series scaling, which does not by itself say the
model is useful. Skill against the seasonal naive is the practical question, and
every cell clears it on every series, so the 2x2 above is a comparison between
working configurations, not between a working one and a broken one.

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


## 4. Real models, under a rolling origin


![five models and the refit that buys nothing](../reports/models.png)

Only the gradient-boosted models beat the weekly-naive baseline, on 79% of series.
ETS is worse than doing nothing. The two GBM bars are the refit ablation, and they
are the same height, which is section 5.

14 meters, 28 origins each, forecasting 24 hours from every origin. Each model
sees only data before its origin, enforced by the harness handing over a prefix
slice, not by remembering to shift correctly.

| model | MASE median | MASE mean | beats seasonal naive |
|---|---|---|---|
| daily naive (`y[t-24]`) | 1.1568 | 1.3254 | 50% |
| seasonal naive (`y[t-168]`) | 1.1319 | 1.2159 | 0% |
| exponential smoothing (ETS) | **1.2791** | 1.4754 | 29% |
| gradient boosting, fit once | 1.0771 | 1.2195 | 79% |
| gradient boosting, refit daily | **1.0741** | 1.2217 | 79% |

**Every model scores MASE above 1.** That does not mean they lose to seasonal
naive on the same rows, gradient boosting beats it on 79% of meters. It means
the final 28 days are harder than the training period the denominator was
computed on. Both facts are true and only reporting the second one would be
flattering.

**ETS is the worst thing here**, worse than repeating last week. Classical
exponential smoothing with daily seasonality has no way to represent the weekly
cycle these meters are dominated by, and it pays for that.

`seasonal naive` beats seasonal naive on 0% of series, which it must, because it
*is* seasonal naive. That column is a self-check, not a result.


## 5. Refitting is worth almost nothing here


This section exists to show that a single temporal cut is optimistic compared to
refitting as time advances. Measured:

| policy | MASE median |
|---|---|
| fit once, then let it age 28 days | 1.0771 |
| refit at every one of 28 origins | 1.0741 |

**+0.0030 MASE, or 0.3%**, for 28× the compute. The second premise this project
was built on, also not supported.

The honest reading is that a month is simply not long enough for a gradient
boosting model on lag features to go stale on this data, the features are
recent lags, which carry their own recency. It would be easy to present the
0.3% as "refitting helps"; it is a rounding error, and the interesting version
of this result is that you can skip the retraining pipeline here and lose
nothing measurable.


## 6. Two other things measured now because they constrain what comes later


**Series scales span 5,332×**: from 15.6 to 82,974 mean kWh. An MAE averaged
across series is therefore a report on the largest few meters and nothing else.
Scale-free errors (MASE) are a requirement here, not a stylistic preference.

**Daily and weekly autocorrelation are both ~0.9**, which sets the honest
baseline. Beating a naive forecast that ignores seasonality proves nothing; the
bar is *seasonal naive*, predict this hour with the same hour last week. In
forecasting it is very common for elaborate models to lose to it, and I would
rather find that out in the grid above than discover it after building something.


## 9. Roadmap


- [x] **1, Data and the deciding statistic.** Prepare 351 series, measure the
      autocorrelation that makes random splits leak, and the scale spread that
      makes MASE mandatory.
- [x] **2, The two-factor test.** Split and horizon varied independently
      against seasonal naive. Horizon dominates by 10x; the project's original
      premise was wrong and is corrected above.
- [x] **3, Models.** Naive, seasonal naive, ETS and gradient boosting under a
      rolling origin. Every model above MASE 1; ETS worst.
- [x] **4, Rolling-origin refits.** Measured at +0.3% MASE for 28x the
      compute. The premise was not supported. (Prediction-interval coverage is
      deliberately *not* here, it is the subject of
      [m4-forecasting](https://github.com/aghasalim/m4-forecasting), and
      duplicating it would be padding.)
- [x] **5, Deployment.** A backtester that hands each model a prefix slice, so
      leaking the future is impossible by construction; plus a Streamlit demo
      and Docker image.
- [x] **6, Docs.** This README and the decision trail in [NOTES.md](../NOTES.md),
      with both refuted premises kept in.


## 10. What I would do next


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

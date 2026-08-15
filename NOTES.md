# Decision trail

A running log of what I tried, what broke, and what I got wrong. Newest at the
bottom. Ground rule: **no number in this file that I did not personally run.**

---

## 1. Naming the project after a claim I had not tested

I called this repo `forecast-backtest` and titled the README "How you split the
data decides the score", because that is the standard warning about time series
and the arithmetic supporting it is real: hold out a random fraction `h` and the
chance both neighbours of a held-out point remain in training is `(1 - h)²`, or
64% at `h = 0.2`.

Milestone 1 then measured lag-1 autocorrelation at **0.9213**, which I took as
confirmation. It is not confirmation of anything — it is consistent with the
claim, which is a different thing.

## 2. The parse that silently did nothing

`LD2011_2014.txt` is semicolon-separated with `,` as the decimal point. Read
with defaults, every value becomes a string and nothing errors. `prepare.py`
asserts every column is numeric after parsing, because a silent type downgrade
that only explodes three steps downstream is worse than a crash.

Also measured before discarding: meters installed partway through the record log
exactly `0` beforehand. Median trimmed prefix is **8,760 hours** — a full year.
Interior zeros are real readings and are kept; the self-check asserts both.

## 3. Testing the claim, and losing

Rather than demonstrate the split effect, I varied two factors independently
with the model, features and rows held fixed:

| what changes | effect on MASE |
|---|---|
| random → temporal split (h=1) | **+4.1%** |
| horizon 1h → 24h (random split) | **+42.1%** |

The horizon dominates by a factor of ten, and the premise the project was named
after is wrong.

The reason is obvious in hindsight and I should have reasoned it through before
writing the README. With strictly past-lag features the model never interpolates
— it only ever sees earlier values, whichever rows are held out. What it gets in
both splits is *the previous hour*, and at 0.92 autocorrelation that is nearly
the answer. Shuffling does not hand it anything it did not already have.

The honest warning is **"a 1-step-ahead score is not evidence you can forecast
24 hours out"**, and it applies to both splits.

I renamed the repo's title to a question rather than a claim.

## 4. A confounded metric that produced an impossible result

My first scoring used model MAE ÷ seasonal-naive MAE **on the test rows**. At
h=24 the naive baseline uses a lag of 191 rather than 168, so it degrades along
with the model. Output:

```
random   h=1   skill 0.4783
random   h=24  skill 0.4498     <- forecasting 24h ahead scored BETTER
```

Which is impossible, and that impossibility is the only reason I caught it. The
yardstick was moving with the thing being measured.

Fixed by switching to MASE with an **in-sample** seasonal-naive denominator,
computed on training rows at a fixed lag and therefore independent of horizon.
The corrected grid is in the README.

The check that the fix is right, rather than merely different: seasonal naive
scores MASE **1.0030** at h=1, and it must be ~1.0 by construction because it
*is* the denominator.

## 5. Scope, against a neighbouring project

`m4-forecasting` already covers prediction-interval coverage. Milestone 4 here
was originally going to do the same thing, which would have been padding. This
project is the evaluation-protocol question — which factor inflates a score —
and interval calibration stays there.

## 6. Both premises, measured and lost

Milestone 3 ran four model families under a rolling origin — 14 meters, 28
origins each, 24h ahead:

| model | MASE median | beats seasonal naive |
|---|---|---|
| daily naive | 1.1568 | 50% |
| seasonal naive | 1.1319 | 0% |
| ETS | 1.2791 | 29% |
| gradient boosting, fit once | 1.0771 | 79% |
| gradient boosting, refit daily | 1.0741 | 79% |

Three things I did not expect and one I should have:

**Every model is above MASE 1.** They are not losing to seasonal naive on the
same rows — boosting beats it on 79% of meters — they are losing to what
seasonal naive achieved *in-sample*, because the last 28 days are harder than
the training period. Reporting only "beats naive on 79%" would have been the
flattering half of a true statement.

**ETS is the worst model on the board.** Its seasonal component is 24 hours and
these meters are dominated by a 168-hour cycle, so it cannot represent the
thing that matters. That is a limitation of my configuration, not of ETS, and
the README says so — weekly seasonality on hourly data is a 168-state component
re-estimated at every origin, which was not affordable here.

**Refitting daily for a month is worth +0.0030 MASE.** Milestone 4 existed to
show that a single temporal cut flatters you relative to refitting as time
advances. It does not — 0.3% for 28x the compute. Second premise, also wrong.

The one I should have seen coming: `seasonal naive` beats seasonal naive on 0%
of series. It cannot beat itself. I left that column in because a metric that
returns the impossible when you feed it a known answer is the cheapest sanity
check available, and this one passed.

## 7. Two experiments that must not be compared

Milestone 2 and Milestone 3 both report MASE and the numbers differ a lot
(0.72 vs 1.08 for broadly similar setups). They are not comparable:

|  | milestone 2 | milestone 3 |
|---|---|---|
| test window | last 20% (~291 days) | last 28 days |
| denominator | in-sample on first 80% | in-sample before the first origin |
| series | 40 | 14 |

Different window, different denominator, different sample. I nearly wrote a
paragraph explaining "why the model got worse", which would have been inventing
a mechanism for an artefact of the setup. The README states the incomparability
instead.

## 8. What the harness is for

Every number here rests on no model having seen data at or after its origin.
Milestone 2 achieved that with careful `.shift()` arithmetic, which is the kind
of correctness that lasts until someone edits it.

So the harness does not ask for care. It hands the forecaster `y[:origin]` and
nothing else, and the self-check proves the guarantee rather than asserting it:
a forecaster that tries to index past its history raises IndexError, and one
that zeroes its input cannot corrupt the caller's series. It also refuses to
return a number when the setup cannot support one — too few origins, too little
history, or a constant series where MASE is undefined.

That is the same shape as the gate in `recsys-offline-online`: the useful
artefact is not a better estimate, it is a thing that will not hand you a
number it cannot stand behind.

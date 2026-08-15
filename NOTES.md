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

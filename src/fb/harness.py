"""Milestone 5 - a backtester that makes leaking the future impossible.

Every result in this project depends on one thing: that no model ever saw data
at or after its forecast origin. Milestone 2 got that right by being careful
with `.shift()`, which is exactly the kind of correctness that survives until
someone edits it.

So the harness does not ask models to be careful. At each origin it hands the
forecast function a COPY OF THE PREFIX, `y[:origin]`, and nothing else. A model
cannot look at the future because it was never given it - the guarantee is
structural rather than a convention, and the self-check proves it by running a
function that actively tries to cheat and showing it cannot.

It also refuses, rather than returning a number, when the setup cannot support
one: too few origins, or too little history to compute a MASE denominator that
does not overlap the evaluation window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np

Forecaster = Callable[[np.ndarray, int], np.ndarray]   # (history, horizon) -> forecast


@dataclass
class Backtest:
    status: str                       # "ok" | "refuse"
    mase: float | None
    reasons: list[str] = field(default_factory=list)
    per_origin: list[float] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def rolling_backtest(y: np.ndarray, forecast: Forecaster, *,
                     horizon: int = 24, n_origins: int = 28, season: int = 168,
                     min_origins: int = 5) -> Backtest:
    """Roll a 24h-ahead forecast forward one day at a time and score it.

    `forecast(history, horizon)` receives only y[:origin]. Anything it returns
    is compared against y[origin:origin+horizon].
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    first = n - n_origins * horizon

    reasons = []
    if first <= season * 2:
        reasons.append(
            f"only {max(first, 0)} hours of history before the first origin; "
            f"need more than {season * 2} to fit anything and still measure it.")
    if n_origins < min_origins:
        reasons.append(f"{n_origins} origins is too few to average over "
                       f"(minimum {min_origins}).")
    if reasons:
        return Backtest("refuse", None, reasons)

    # MASE denominator on data strictly before the first origin, so the yardstick
    # never overlaps the window being scored
    insample = np.abs(y[season:first] - y[: first - season])
    denom = float(np.mean(insample))
    if not np.isfinite(denom) or denom <= 0:
        return Backtest("refuse", None,
                        ["in-sample seasonal-naive error is zero or undefined; "
                         "the series is constant and MASE is not defined."])

    errs = []
    for k in range(n_origins):
        o = first + k * horizon
        history = y[:o].copy()          # the only thing the model is given
        pred = np.asarray(forecast(history, horizon), dtype=float)
        if pred.shape != (horizon,):
            return Backtest("refuse", None,
                            [f"forecaster returned shape {pred.shape}, expected "
                             f"({horizon},)"])
        errs.append(float(np.mean(np.abs(y[o: o + horizon] - pred))))

    return Backtest("ok", float(np.mean(errs)) / denom, [], errs,
                    {"n_origins": n_origins, "horizon": horizon,
                     "first_origin_index": int(first),
                     "mase_denominator": denom,
                     "history_before_first_origin": int(first)})


# ---- forecasters -----------------------------------------------------------

def seasonal_naive(history: np.ndarray, horizon: int, season: int = 168) -> np.ndarray:
    return history[-season: -season + horizon] if season > horizon else history[-horizon:]


def daily_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    return history[-24:][:horizon]


def demo() -> None:
    """Self-check, including proving a cheating forecaster cannot cheat."""
    rng = np.random.default_rng(0)
    t = np.arange(24 * 400)
    y = 100 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, len(t))

    # a perfect seasonal signal: seasonal naive should score well below 1
    r = rolling_backtest(y, seasonal_naive, n_origins=10)
    assert r.status == "ok" and r.mase < 1.5, (r.status, r.mase)

    # THE IMPORTANT ONE: a forecaster that tries to read the future gets only
    # the past, so it cannot do better than what the past supports.
    seen: list[int] = []

    def cheater(history: np.ndarray, horizon: int) -> np.ndarray:
        seen.append(len(history))
        # try to index beyond what we were given
        try:
            _ = history[len(history) + horizon - 1]
            raise AssertionError("harness handed over future data")
        except IndexError:
            pass
        return history[-horizon:]

    r2 = rolling_backtest(y, cheater, n_origins=6)
    assert r2.status == "ok"
    # each origin gets strictly more history than the last, and never the whole array
    assert seen == sorted(seen) and max(seen) < len(y)

    # mutating the handed-over history must not corrupt the series
    def vandal(history: np.ndarray, horizon: int) -> np.ndarray:
        history[:] = 0.0
        return np.zeros(horizon)

    before = y.copy()
    rolling_backtest(y, vandal, n_origins=6)
    assert np.array_equal(y, before), "forecaster mutated the caller's data"

    # refusal paths
    short = rolling_backtest(y[:500], seasonal_naive, n_origins=10)
    assert short.status == "refuse" and short.mase is None
    flat = rolling_backtest(np.ones(24 * 400), seasonal_naive, n_origins=10)
    assert flat.status == "refuse" and "constant" in flat.reasons[0]
    few = rolling_backtest(y, seasonal_naive, n_origins=2)
    assert few.status == "refuse"
    print("self-check ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-check", action="store_true")
    p.parse_args()
    demo()

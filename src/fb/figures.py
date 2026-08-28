"""The 2x2 result, drawn.

    python src/fb/figures.py

Reads reports/backtest.json, reports/models.json, reports/eda.json and
reports/prepare.json and nothing else, so a picture here cannot disagree with
the tables in the README, and redrawing needs no dataset.
The point of the pictures is the comparison the numbers make and prose tends to
flatten: moving from a random to a temporal split barely shifts the score, while
stretching the horizon from one hour to a day moves it ten times further.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from style import PALETTE, titled

REPORTS = Path(__file__).resolve().parents[2] / "reports"

# One colour per split, held constant wherever the splits appear.
SPLITS = [("random", PALETTE[0], "o"), ("temporal", PALETTE[1], "s")]
HORIZONS = [1, 24]

# Grey for the two naive baselines because they are the yardstick and not a
# result, orange for ETS, and two distinct colours for the refit ablation.
MODEL_COLOUR = {
    "naive24": PALETTE[5], "naive168": PALETTE[5], "ets": PALETTE[3],
    "gbm_once": PALETTE[0], "gbm_refit": PALETTE[4],
}
MODEL_NAME = {
    "naive24": "daily naive, y[t-24]",
    "naive168": "weekly naive, y[t-168]",
    "ets": "exponential smoothing",
    "gbm_once": "boosting, fit once",
    "gbm_refit": "boosting, refit daily",
}


def factorial(out: Path) -> Path:
    """Interaction plot plus the two effects expressed against the baseline."""
    data = json.loads((REPORTS / "backtest.json").read_text())
    cells = data["cells"]
    baseline = data["baseline_mase"]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    for split, colour, marker in SPLITS:
        # The two h=1 points sit ~0.02 apart, so their labels collide unless
        # they are pushed to opposite sides of the markers.
        nudge = 12 if split == "temporal" else -18
        values = [cells[f"{split}_h{h}"]["mase_median"] for h in HORIZONS]
        left.plot(
            range(len(HORIZONS)), values, marker=marker, color=colour,
            markersize=8, label=f"{split} split",
        )
        for x, value in enumerate(values):
            left.annotate(
                f"{value:.3f}", (x, value), textcoords="offset points",
                xytext=(0, nudge), ha="center", fontsize=9, color=colour,
            )
    left.set_xticks(range(len(HORIZONS)))
    left.set_xticklabels([f"{h} h ahead" for h in HORIZONS])
    left.set_xlim(-0.35, len(HORIZONS) - 0.65)
    left.set_ylim(0.44, 0.79)
    left.set_xlabel("forecast horizon (hours ahead)")
    left.set_ylabel("median MASE (unitless, lower is better)")
    titled(
        left,
        "Both splits climb together when the horizon stretches",
        "median over 40 meters, same model, same features, seed 0; the line joins two measured points",
    )
    left.legend(loc="upper left")

    effects = [
        ("split\nrandom to temporal", data["effect_of_split"] / baseline * 100),
        ("horizon\n1 h to 24 h", data["effect_of_horizon"] / baseline * 100),
    ]
    for index, (_label, percent) in enumerate(effects):
        right.bar(index, percent, 0.5, color=PALETTE[index], edgecolor="none")
        right.text(index, percent + 1.3, f"+{percent:.1f}%", ha="center",
                   fontsize=11, fontweight="bold", color=PALETTE[index])
    right.set_xticks(range(len(effects)))
    right.set_xticklabels([label for label, _ in effects])
    right.set_xlim(-0.6, len(effects) - 0.4)
    right.set_ylabel("increase in median MASE (% of baseline)")
    right.set_ylim(0, max(p for _, p in effects) * 1.22)
    ratio = data["effect_of_horizon"] / data["effect_of_split"]
    titled(
        right,
        f"The horizon costs {ratio:.0f}x what the split costs",
        f"each change measured on its own against the random h=1 cell, MASE {baseline:.3f}",
    )

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def leaderboard(out: Path) -> Path:
    """Five models, and the refit that buys nothing.

    The second thing this project set out to demonstrate was that refitting the
    model at every origin matters. It does not: gbm_refit beats gbm_once by 0.003
    MASE, which is inside the noise and costs an order of magnitude more compute.
    """
    data = json.loads((REPORTS / "models.json").read_text())
    models = data["models"]
    order = ["naive24", "naive168", "ets", "gbm_once", "gbm_refit"]
    colours = [MODEL_COLOUR[m] for m in order]
    names = [MODEL_NAME[m] for m in order]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))
    positions = np.arange(len(order))

    mase = [models[m]["mase_median"] for m in order]
    left.barh(positions, mase, 0.62, color=colours, edgecolor="none")
    for y, value in zip(positions, mase, strict=True):
        left.text(value + 0.02, y, f"{value:.3f}", va="center", fontsize=9.5,
                  color="#444444")
    left.axvline(1.0, color="#666666", ls="--", lw=1.1)
    left.text(1.02, 4.62, "1.0 = the in-sample seasonal naive", fontsize=9,
              color="#666666", ha="left", va="center")
    left.set_yticks(positions)
    left.set_yticklabels(names)
    left.invert_yaxis()
    left.set_xlim(0, max(mase) * 1.18)
    left.set_ylim(len(order) - 0.05, -0.7)
    left.set_xlabel("median MASE (unitless, lower is better)")
    gain = data["refit_gain"]
    titled(
        left,
        f"Refitting at every origin is worth {gain:.3f} MASE",
        f"the bottom two bars are one model fit once and refit at all "
        f"{data['origins_per_series']} origins, {data['n_series']} meters, "
        f"{data['horizon']} h ahead, seed {data['seed']}",
    )

    beats = [models[m]["beats_naive168_frac"] * 100 for m in order]
    right.barh(positions, beats, 0.62, color=colours, edgecolor="none")
    for y, value in zip(positions, beats, strict=True):
        right.text(value + 1.6, y, f"{value:.0f}%", va="center", fontsize=9.5,
                   color="#444444")
    right.set_yticks(positions)
    right.set_yticklabels(names)
    right.invert_yaxis()
    right.set_xlim(0, 100)
    right.set_ylim(len(order) - 0.05, -0.7)
    right.set_xlabel("meters where the model wins (% of 14)")
    titled(
        right,
        "Boosting is the only model that beats last week on most meters",
        "share of the 14 meters with a lower median MASE than the weekly naive y[t-168]",
    )

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def premise(out: Path) -> Path:
    """The interpolation argument this project was built to demonstrate.

    Consecutive hours correlate at 0.92, and holding out a random 20% leaves 64%
    of held-out points sandwiched between two training points. The arithmetic is
    correct; it just is not what drives the score, because strictly past-lag
    features never let the model interpolate in the first place.
    """
    data = json.loads((REPORTS / "eda.json").read_text())
    autocorr = data["autocorrelation"]
    probability = data["prob_both_neighbours_in_train"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))
    lags = list(autocorr)
    positions = np.arange(len(lags))
    medians = [autocorr[lag]["median"] for lag in lags]
    left.bar(positions, medians, 0.5, color=PALETTE[0], edgecolor="none",
             yerr=[
                 [autocorr[lag]["median"] - autocorr[lag]["p10"] for lag in lags],
                 [autocorr[lag]["p90"] - autocorr[lag]["median"] for lag in lags],
             ],
             capsize=5, ecolor="#333333")
    for x, value in zip(positions, medians, strict=True):
        left.text(x, 0.04, f"{value:.2f}", ha="center", fontsize=10,
                  color="white", fontweight="bold")
    left.set_xticks(positions)
    left.set_xticklabels(["1 h", "24 h (daily)", "168 h (weekly)"])
    left.set_xlabel("lag (hours)")
    left.set_ylim(0, 1.05)
    left.set_ylabel("autocorrelation (unitless, -1 to 1)")
    titled(
        left,
        "Consecutive hours correlate at 0.92",
        f"median over {data['series_sampled']} sampled meters, whisker is p10 to p90",
    )

    holdouts = sorted(probability, key=lambda k: probability[k], reverse=True)
    fractions = [int(k.split("_")[1].replace("pct", "")) for k in holdouts]
    percents = [probability[k] * 100 for k in holdouts]
    right.plot(fractions, percents, "o-", color=PALETTE[1], markersize=9)
    for x, value in zip(fractions, percents, strict=True):
        right.annotate(f"{value:.0f}%", (x, value), textcoords="offset points",
                       xytext=(0, 12), ha="center", fontsize=10, color=PALETTE[1])
    right.set_xlabel("random hold-out fraction h (% of rows)")
    right.set_ylabel("held-out points with both neighbours in train (%)")
    right.set_xlim(5, 35)
    right.set_ylim(0, 100)
    titled(
        right,
        "At the usual 20% hold-out, two thirds of the test set is interpolation",
        "P(both neighbours in train) = (1-h)^2, which is arithmetic, not a measurement",
    )

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def dataset(out: Path) -> Path:
    """What the 351 meters look like, from the two prepared summaries.

    Two facts about the data that constrain everything after it: the meters
    differ in size by three and a half orders of magnitude, which is why every
    error here is scale-free, and most were installed well into the record,
    which is why the leading zeros are trimmed rather than averaged in.
    """
    data = json.loads((REPORTS / "eda.json").read_text())
    meta = json.loads((REPORTS / "prepare.json").read_text())
    scale = data["mean_kwh_per_series"]
    hours = data["hours_per_series"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Three order statistics, not a distribution, so draw them as a range with
    # the ends marked rather than implying a shape the summary does not have.
    points = [("smallest", scale["min"]), ("median", scale["median"]),
              ("largest", scale["max"])]
    values = [v for _, v in points]
    left.hlines(0, values[0], values[-1], color=PALETTE[0], lw=3.5, alpha=0.3)
    left.plot(values, [0, 0, 0], "o", color=PALETTE[0], markersize=11)
    for name, value in points:
        # a decimal on the smallest one, or the 15.6 kWh meter reads as 16 and
        # stops matching the number the README quotes
        shown = f"{value:,.1f}" if value < 100 else f"{value:,.0f}"
        left.annotate(shown, (value, 0), textcoords="offset points",
                      xytext=(0, 14), ha="center", fontsize=10.5,
                      fontweight="bold", color=PALETTE[0])
        left.annotate(name, (value, 0), textcoords="offset points",
                      xytext=(0, -22), ha="center", fontsize=9.5,
                      color="#5a5a5a")
    ratio = scale["max_over_min"]
    left.annotate("", xy=(values[0], 0.5), xytext=(values[-1], 0.5),
                  arrowprops={"arrowstyle": "<->", "color": "#777777",
                              "lw": 1.1})
    left.text(np.sqrt(values[0] * values[-1]), 0.58, f"{ratio:,.0f}x",
              ha="center", va="bottom", fontsize=11.5, fontweight="bold",
              color="#444444")
    left.set_xscale("log")
    left.set_xlim(values[0] / 3, values[-1] * 3)
    left.set_ylim(-0.75, 0.95)
    left.set_yticks([])
    left.spines["left"].set_visible(False)
    left.grid(axis="y", visible=False)
    left.set_xlabel("mean load per meter (kWh per hour, log scale)")
    titled(
        left,
        f"The biggest meter is {ratio:,.0f}x the smallest",
        f"mean load over {data['series_sampled']} sampled meters, so a plain "
        f"MAE would report on the largest few",
    )

    # Every meter's record ends together, so drawing the kept history as a span
    # shows how late each one was actually installed.
    end = datetime.fromisoformat(meta["t_end"])
    end_x = end.year + (end.timetuple().tm_yday - 1) / 365.25
    spans = [("longest", hours["max"]), ("median", hours["median"]),
             ("shortest", hours["min"])]
    positions = np.arange(len(spans))
    starts = [end_x - h / 8766.0 for _, h in spans]
    for y, (_name, hrs), start in zip(positions, spans, starts, strict=True):
        right.barh(y, end_x - start, 0.5, left=start, color=PALETTE[2],
                   edgecolor="none")
        right.text(end_x - 0.06, y, f"{hrs:,} h", va="center", ha="right",
                   fontsize=10, fontweight="bold", color="white")
    right.set_yticks(positions)
    right.set_yticklabels([name for name, _ in spans])
    right.invert_yaxis()
    right.set_ylim(len(spans) - 0.45, -0.55)
    first = round(min(starts))
    right.set_xticks(range(first, int(round(end_x)) + 1))
    right.set_xlim(first - 0.06, end_x + 0.06)
    right.grid(axis="y", visible=False)
    right.set_xlabel("calendar year")
    titled(
        right,
        "Half the meters were installed a year into the record",
        f"history kept after trimming leading zeros; "
        f"{meta['series_dropped_short']} of {meta['series_total']} meters "
        f"held under a year",
    )

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def skill(out: Path) -> Path:
    """Skill over the naive baseline in each cell of the 2x2.

    MASE alone does not say whether the model is useful, only how it compares to
    a within-series scaling. Skill against the seasonal naive is the practical
    question, and every cell beats it on every series.
    """
    data = json.loads((REPORTS / "backtest.json").read_text())
    cells = data["cells"]
    labels = ["random_h1", "random_h24", "temporal_h1", "temporal_h24"]
    pretty = ["random\n1 h ahead", "random\n24 h ahead",
              "temporal\n1 h ahead", "temporal\n24 h ahead"]

    figure, ax = plt.subplots(figsize=(9.5, 5.0))
    positions = np.arange(len(labels))
    skills = [cells[c]["skill_median"] * 100 for c in labels]
    shares = [cells[c]["beats_naive_frac"] * 100 for c in labels]
    ax.bar(positions - 0.19, skills, 0.36, color=PALETTE[0], edgecolor="none",
           label="median skill, 1 - MASE model / MASE naive")
    ax.bar(positions + 0.19, shares, 0.36, color=PALETTE[2], edgecolor="none",
           label="meters that beat the seasonal naive")
    for x, value in zip(positions, skills, strict=True):
        ax.text(x - 0.19, value + 1.5, f"{value:.0f}%", ha="center", fontsize=9.5,
                color=PALETTE[0])
    for x, value in zip(positions, shares, strict=True):
        ax.text(x + 0.19, value + 1.5, f"{value:.0f}%", ha="center", fontsize=9.5,
                color=PALETTE[2])
    ax.set_xticks(positions)
    ax.set_xticklabels(pretty)
    ax.set_xlabel("cell of the 2x2 (split, horizon)")
    ax.set_ylabel("percent (%)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
    titled(
        ax,
        "Every cell clears the seasonal naive on all 40 meters",
        "so the 2x2 compares working configurations, not a working one against a broken one",
    )
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def demo() -> None:
    """Check the arithmetic the figure prints, not that matplotlib runs.

    The two percentages and the ratio in the titles are derived here rather than
    typed, so the thing worth asserting is that they still agree with the cells
    they are derived from.
    """
    data = json.loads((REPORTS / "backtest.json").read_text())
    cells, baseline = data["cells"], data["baseline_mase"]

    # effect_of_split and effect_of_horizon must be the differences they claim.
    split = cells["temporal_h1"]["mase_median"] - cells["random_h1"]["mase_median"]
    horizon = cells["random_h24"]["mase_median"] - cells["random_h1"]["mase_median"]
    assert abs(split - data["effect_of_split"]) < 1e-9, "split effect is not the h=1 gap"
    assert abs(horizon - data["effect_of_horizon"]) < 1e-9, "horizon effect is not the random-split gap"
    assert abs(baseline - cells["random_h1"]["mase_median"]) < 1e-9, "baseline is not the random h=1 cell"

    # The README quotes +4.1% and +42.1% against that baseline.
    assert abs(split / baseline * 100 - 4.1) < 0.1, split / baseline * 100
    assert abs(horizon / baseline * 100 - 42.1) < 0.1, horizon / baseline * 100
    assert horizon > split * 5, "the whole point is that the horizon dominates"
    print("self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    if parser.parse_args().self_check:
        demo()
        return
    for path in (
        factorial(REPORTS / "backtest.png"),
        leaderboard(REPORTS / "models.png"),
        premise(REPORTS / "premise.png"),
        dataset(REPORTS / "eda.png"),
        skill(REPORTS / "skill.png"),
    ):
        print(f"wrote {path.relative_to(REPORTS.parent)}")


if __name__ == "__main__":
    main()

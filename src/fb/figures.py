"""The 2x2 result, drawn.

    python -m fb.figures

Reads ``reports/backtest.json`` and nothing else, so this cannot disagree with
the table in the README. The point of the picture is the comparison the numbers
make and prose tends to flatten: moving from a random to a temporal split barely
shifts the score, while stretching the horizon from one hour to a day moves it
ten times further.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(__file__).resolve().parents[2] / "reports"

SPLITS = [("random", "#2166ac", "o"), ("temporal", "#b2182b", "s")]
HORIZONS = [1, 24]


def factorial(out: Path) -> Path:
    """Interaction plot plus the two effects expressed against the baseline."""
    data = json.loads((REPORTS / "backtest.json").read_text())
    cells = data["cells"]
    baseline = data["baseline_mase"]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    for split, colour, marker in SPLITS:
        # The two h=1 points sit ~0.02 apart, so their labels collide unless
        # they are pushed to opposite sides of the markers.
        nudge = 11 if split == "temporal" else -17
        values = [cells[f"{split}_h{h}"]["mase_median"] for h in HORIZONS]
        left.plot(
            range(len(HORIZONS)), values, marker=marker, color=colour,
            lw=2, markersize=8, label=f"{split} split",
        )
        for x, value in enumerate(values):
            left.annotate(
                f"{value:.3f}", (x, value), textcoords="offset points",
                xytext=(0, nudge), ha="center", fontsize=8, color=colour,
            )
    left.set_xticks(range(len(HORIZONS)))
    left.set_xticklabels([f"h = {h}" for h in HORIZONS])
    left.set_xlim(-0.35, len(HORIZONS) - 0.65)
    left.set_ylabel("median MASE  (lower is better)")
    left.set_title(
        "The two lines nearly touch. The slope is the whole story.", fontsize=10
    )
    left.legend(frameon=False, fontsize=9)
    left.spines[["top", "right"]].set_visible(False)

    effects = [
        ("split\nrandom -> temporal", data["effect_of_split"] / baseline * 100, "#9ecae1"),
        ("horizon\n1h -> 24h", data["effect_of_horizon"] / baseline * 100, "#b2182b"),
    ]
    for index, (_label, percent, colour) in enumerate(effects):
        right.bar(index, percent, 0.55, color=colour, edgecolor="0.3", lw=0.5)
        right.text(
            index, percent + 1.2, f"+{percent:.1f}%",
            ha="center", fontsize=11, fontweight="bold",
        )
    right.set_xticks(range(len(effects)))
    right.set_xticklabels([label for label, _, _ in effects], fontsize=9)
    right.set_ylabel("change in MASE, % of baseline")
    right.set_ylim(0, max(p for _, p, _ in effects) * 1.25)
    ratio = data["effect_of_horizon"] / data["effect_of_split"]
    right.set_title(
        f"The factor everyone warns about is {ratio:.0f}x smaller\n"
        f"than the one nobody mentions",
        fontsize=10,
    )
    right.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        f"{data['n_series']} series, seed {data['seed']}, medians across series",
        fontsize=9, y=0.02, color="0.4",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
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
    colours = {
        "naive24": "#bdbdbd", "naive168": "#bdbdbd", "ets": "#9ecae1",
        "gbm_once": "#2166ac", "gbm_refit": "#b2182b",
    }

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4))
    positions = np.arange(len(order))
    left.barh(positions, [models[m]["mase_median"] for m in order],
              color=[colours[m] for m in order], edgecolor="0.3", lw=0.5)
    left.axvline(1.0, color="0.25", ls="--", lw=1.2)
    left.text(1.0, len(order) - 0.4, " naive = 1.0", fontsize=8, color="0.35")
    left.set_yticks(positions)
    left.set_yticklabels(order)
    left.invert_yaxis()
    left.set_xlabel("median MASE (lower is better)")
    left.set_title("only the gradient-boosted models beat the naive baseline",
                   fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    right.barh(positions, [models[m]["beats_naive168_frac"] * 100 for m in order],
               color=[colours[m] for m in order], edgecolor="0.3", lw=0.5)
    right.set_yticks(positions)
    right.set_yticklabels(order)
    right.invert_yaxis()
    right.set_xlabel("% of series beating the weekly-naive baseline")
    gain = data["refit_gain"]
    right.set_title(
        f"refitting at every origin is worth {gain:.3f} MASE\n"
        "for an order of magnitude more compute",
        fontsize=10,
    )
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
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

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.3))
    lags = list(autocorr)
    positions = np.arange(len(lags))
    left.bar(positions, [autocorr[lag]["median"] for lag in lags], 0.55,
             yerr=[
                 [autocorr[lag]["median"] - autocorr[lag]["p10"] for lag in lags],
                 [autocorr[lag]["p90"] - autocorr[lag]["median"] for lag in lags],
             ],
             capsize=4, color="#2166ac", edgecolor="0.3", lw=0.5)
    left.set_xticks(positions)
    left.set_xticklabels(lags)
    left.set_ylim(0, 1.02)
    left.set_ylabel("autocorrelation (median, p10-p90)")
    left.set_title("every point is highly predictable from its neighbours",
                   fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    holdouts = sorted(probability, key=lambda k: probability[k], reverse=True)
    fractions = [int(k.split("_")[1].replace("pct", "")) for k in holdouts]
    right.plot(fractions, [probability[k] * 100 for k in holdouts], "o-",
               color="#b2182b", lw=2, markersize=8)
    for x, k in zip(fractions, holdouts, strict=True):
        right.annotate(f"{probability[k] * 100:.0f}%", (x, probability[k] * 100),
                       textcoords="offset points", xytext=(0, 10), ha="center",
                       fontsize=9)
    right.set_xlabel("random hold-out fraction h (%)")
    right.set_ylabel("% of held-out points with both neighbours in train")
    right.set_ylim(0, 100)
    right.set_title("$(1-h)^2$: at h=20%, two thirds of the test set\n"
                    "sits between two known values", fontsize=10)
    right.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "The premise, which is arithmetically correct and turned out not to be "
        "what moves the score.",
        fontsize=9, y=0.02, color="0.4",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
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
    pretty = ["random\nh=1", "random\nh=24", "temporal\nh=1", "temporal\nh=24"]

    figure, ax = plt.subplots(figsize=(9, 4.4))
    positions = np.arange(len(labels))
    ax.bar(positions - 0.2, [cells[c]["skill_median"] * 100 for c in labels], 0.4,
           label="median skill vs naive", color="#2166ac", edgecolor="0.3", lw=0.5)
    ax.bar(positions + 0.2, [cells[c]["beats_naive_frac"] * 100 for c in labels], 0.4,
           label="% of series beating naive", color="#9ecae1", edgecolor="0.3", lw=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(pretty)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Every cell beats the seasonal naive on every series. "
        "The split does not change that; neither does the horizon.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
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
        skill(REPORTS / "skill.png"),
    ):
        print(f"wrote {path.relative_to(REPORTS.parent)}")


if __name__ == "__main__":
    main()

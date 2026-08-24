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
    print(f"wrote {factorial(REPORTS / 'backtest.png').relative_to(REPORTS.parent)}")


if __name__ == "__main__":
    main()

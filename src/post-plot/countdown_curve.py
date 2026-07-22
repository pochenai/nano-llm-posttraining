"""Steps-vs-score curve for Countdown GRPO runs (TinyZero's critic/score/mean plot).

Reads each run's training log (trainer_output/grpo-countdown-*/log_history.json, dumped
by grpo_countdown.py) and plots the score against optimizer step -- one line per run, so
different model sizes / configs overlay like TinyZero's figure. The score climbing with
training is the headline.

    uv run python src/post-plot/countdown_curve.py   # -> assets/figures/fig_countdown_curve.png
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt

OUT = "assets/figures/fig_countdown_curve.png"
# Okabe-Ito colorblind-safe palette, cycled across runs.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
# Prefer the correctness component (the real objective / solve signal); fall back to
# the summed reward if an older log doesn't split it out.
METRIC_KEYS = ["rewards/correctness_reward/mean", "reward"]


def series(log):
    """(steps, scores, key) for the first available metric key in a log_history list."""
    for key in METRIC_KEYS:
        pts = [(e["step"], e[key]) for e in log if "step" in e and key in e]
        if pts:
            return [p[0] for p in pts], [p[1] for p in pts], key
    return [], [], None


def smooth(ys, w=5):
    """Centered rolling mean -- RL curves are noisy; show the trend over the raw line."""
    if w <= 1 or len(ys) < w:
        return ys
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - w // 2), min(len(ys), i + w // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def main():
    files = sorted(glob.glob("trainer_output/grpo-countdown-*/log_history.json"))
    if not files:
        raise SystemExit(
            "no log_history.json under trainer_output/grpo-countdown-*/ -- train first"
        )

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#bbbbbb"})
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    metric = None
    for i, f in enumerate(files):
        run = os.path.basename(os.path.dirname(f)).replace("grpo-countdown-", "")
        xs, ys, key = series(json.load(open(f)))
        if not xs:
            continue
        metric = key
        c = COLORS[i % len(COLORS)]
        ax.plot(xs, ys, color=c, lw=0.8, alpha=0.25, zorder=1)  # raw (faint)
        ax.plot(xs, smooth(ys), color=c, lw=1.8, label=run, zorder=2)  # smoothed trend

    ax.set_xlabel("training step  →", fontsize=12)
    ax.set_ylabel(f"↑  {metric}", fontsize=11)
    ax.set_title(
        "Countdown GRPO: score climbs with training",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, color="#eeeeee", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, title="run")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""Hero figure: RL's Razor at the LLM/token scale.

A (drift, skill) scatter. Each post-trained model is one point:
  x = forward KL(pi_ft || pi_base) per token  -- how far it moved from base (forgetting proxy)
  y = IFEval held-out satisfaction             -- task skill

The story is Pareto dominance: the RL point sits upper-left (more skill, less
drift) and dominates the SFT points in the lower-right.

    uv run python src/post-plot/razor_pareto.py   # writes assets/figures/fig_razor_pareto.png

Numbers are the finalized clean-protocol results (held-out eval: cap=512 +
batch=1; KL: forward + exact estimator, ref=SmolLM2-135M-Instruct).
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

OUT = "assets/figures/fig_razor_pareto.png"

# Okabe-Ito colorblind-safe palette. The three method hues (blue/green/vermillion)
# pass CVD + normal-vision separation; gray is the neutral baseline reference.
C_RL = "#0072B2"  # blue
C_SFT_IN = "#009E73"  # bluish green -- in-distribution SFT
C_SFT_OFF = "#D55E00"  # vermillion  -- off-distribution SFT
C_BASE = "#666666"  # gray

# (label, kl_per_token, ifeval, method)
POINTS = [
    ("baseline", 0.000, 0.315, "base"),
    ("in-dist SFT", 0.333, 0.392, "sft_in"),
    ("off-dist SFT", 0.332, 0.335, "sft_off"),
    ("GRPO (RL)", 0.127, 0.592, "rl"),
]
COLOR = {"base": C_BASE, "sft_in": C_SFT_IN, "sft_off": C_SFT_OFF, "rl": C_RL}


def main():
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#bbbbbb"})
    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    # Pareto frontier RL reaches: baseline -> GRPO. SFT falls strictly inside.
    base = next(p for p in POINTS if p[3] == "base")
    rl = next(p for p in POINTS if p[3] == "rl")
    ax.plot(
        [base[1], rl[1]],
        [base[2], rl[2]],
        "--",
        color=C_RL,
        lw=1.4,
        alpha=0.55,
        zorder=1,
    )
    # Marks: RL = circle, SFT = square (both arms), baseline = diamond. Shape carries
    # the RL/SFT/base identity; color separates the two SFT arms within the square.
    marker = {"base": "D", "sft_in": "s", "sft_off": "s", "rl": "o"}
    size = {"base": 130, "sft_in": 150, "sft_off": 150, "rl": 240}
    for label, kl, acc, m in POINTS:
        ax.scatter(
            kl,
            acc,
            s=size[m],
            c=COLOR[m],
            marker=marker[m],
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
        )

    # Direct labels (only 4 points -> label all, with hand-placed offsets).
    off = {
        "baseline": (8, -16),
        "in-dist SFT": (10, 6),
        "off-dist SFT": (10, -16),
        "GRPO (RL)": (12, -4),
    }
    for label, kl, acc, m in POINTS:
        dx, dy = off[label]
        ax.annotate(
            f"{label}\n",
            (kl, acc),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10.5,
            color="#222222",
            fontweight="bold" if m == "rl" else "normal",
        )

    # Ideal corner cue (upper-left = high new skill, low drift).
    ax.annotate(
        "ideal\nhigh new skill · low drift",
        (0.012, 0.60),
        fontsize=10,
        color=C_RL,
        style="italic",
        va="top",
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.09, 0.60),
            (0.015, 0.615),
            arrowstyle="->",
            color=C_RL,
            lw=1.2,
            alpha=0.7,
            mutation_scale=12,
        )
    )

    ax.set_xlim(-0.02, 0.40)
    ax.set_ylim(0.29, 0.65)
    ax.set_xlabel(
        r"drift from base  —  $D_{\mathrm{KL}}(\pi_{\mathrm{ft}}\,\|\,\pi_{\mathrm{base}})$ per token  →",
        fontsize=12,
    )
    ax.set_ylabel("↑  IFEval held-out satisfaction (skill)", fontsize=12)
    ax.set_title(
        "RL learns more while moving less", fontsize=15, fontweight="bold", pad=14
    )
    ax.text(
        0.5,
        1.015,
        "SmolLM2-135M · IFEval · same base, same reward — RL Pareto-dominates SFT",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#666666",
    )

    ax.grid(True, color="#eeeeee", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend by method (>=2 identities present -> legend required).
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=C_RL,
            markersize=12,
            label="RL (GRPO)",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="w",
            markerfacecolor=C_SFT_IN,
            markersize=11,
            label="in-dist SFT",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="w",
            markerfacecolor=C_SFT_OFF,
            markersize=11,
            label="off-dist SFT",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color="w",
            markerfacecolor=C_BASE,
            markersize=10,
            label="base (untrained)",
        ),
    ]
    # Upper-right is the only empty quadrant (GRPO sits upper-left, SFT lower-right).
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10.5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

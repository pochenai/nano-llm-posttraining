"""Companion hero figure: the forgetting COST that RL avoids.

razor_pareto.py shows drift -> skill (RL gains more, moves less). This shows the
other half: drift -> forgetting. Held-out perplexity RISE vs the base model, on
two different corpora, one grouped bar per method:

  x = method (in-dist SFT, off-dist SFT, GRPO)
  y = perplexity increase over base  (higher = more general LM ability lost)

The commonsense-MC basket (retention_eval.py) was FLAT across all arms -- MC
accuracy on stored knowledge is robust to instruction fine-tuning at 135M. PPL is
not: both SFT arms lose 10% (wikitext) / ~20% (Pile), GRPO stays ~0 on both.
Forgetting lives in the generation distribution, not knowledge retrieval, and only
SFT pays it.

    uv run python src/post-plot/ppl_forgetting.py   # -> assets/figures/fig_ppl_forgetting.png

Numbers are read from trainer_output/ppl_runs.json (ppl_eval.py output), so the
figure stays in sync with the measurements.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PPL_JSON = "trainer_output/ppl_runs.json"
OUT = "assets/figures/fig_ppl_forgetting.png"

# Same Okabe-Ito hues as razor_pareto.py, so a method keeps its color across the
# paired figures.
C_RL = "#0072B2"  # blue
C_SFT_IN = "#009E73"  # bluish green -- in-distribution SFT
C_SFT_OFF = "#D55E00"  # vermillion  -- off-distribution SFT

BASELINE = "HuggingFaceTB/SmolLM2-135M-Instruct"
# (model path, label, color, KL/token) -- KL matches razor_pareto.py's POINTS.
ARMS = [
    ("trainer_output/indist-sft", "in-dist SFT", C_SFT_IN, 0.333),
    ("trainer_output/offdist-sft", "off-dist SFT", C_SFT_OFF, 0.332),
    ("trainer_output/grpo-beta002/checkpoint-474", "GRPO (RL)", C_RL, 0.127),
]
CORPORA = [("wikitext", "wikitext", "//"), ("pile_10k", "Pile", None)]


def _delta_pct(runs, model, corpus, base_ppl):
    ppl = runs[model][corpus]["word_perplexity"]
    return (ppl / base_ppl - 1) * 100


def main():
    runs = {r["model"]: r for r in json.load(open(PPL_JSON))}
    base = {c: runs[BASELINE][c]["word_perplexity"] for c, _, _ in CORPORA}

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#bbbbbb"})
    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    width = 0.36
    xs = range(len(ARMS))
    for ci, (corpus, clabel, hatch) in enumerate(CORPORA):
        offset = (ci - 0.5) * width
        for xi, (model, label, color, _kl) in enumerate(ARMS):
            d = _delta_pct(runs, model, corpus, base[corpus])
            ax.bar(
                xi + offset,
                d,
                width,
                color=color,
                alpha=1.0 if hatch is None else 0.5,
                hatch=hatch,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            ax.annotate(
                f"+{d:.0f}%",
                (xi + offset, d),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=10,
                color="#222222",
                fontweight="bold",
            )

    # Baseline reference: 0% = no forgetting.
    ax.axhline(0, color="#666666", lw=1.2, zorder=2)
    ax.annotate(
        "base = no forgetting",
        (len(ARMS) - 0.5, 0),
        textcoords="offset points",
        xytext=(0, -14),
        ha="right",
        fontsize=9.5,
        color="#666666",
        style="italic",
    )

    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [f"{label}\nKL/tok={kl:.2f}" for _, label, _, kl in ARMS], fontsize=11
    )
    ax.set_ylabel("↑  held-out perplexity increase vs base  (%)", fontsize=12)
    ax.set_ylim(0, 25)
    ax.set_title(
        "RL learns the task without forgetting",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.5,
        1.015,
        "SmolLM2-135M · general-text PPL rise (two corpora) — SFT forgets 10–20%, RL ~0%",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#666666",
    )

    ax.grid(True, axis="y", color="#eeeeee", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Corpus legend (shade/hatch carries corpus; x-axis color carries method).
    handles = [
        Patch(facecolor="#888888", hatch="//", alpha=0.5, label="wikitext"),
        Patch(facecolor="#888888", label="Pile"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10.5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

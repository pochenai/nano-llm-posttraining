"""Old-task retention probe: general-capability benchmarks the base model already
handles, averaged. This is the "previous task accuracy" (task A) axis of the
forgetting study -- how much general ability a fine-tune sacrifices while learning
the new task B (IFEval).

Deliberately EXCLUDES IFEval (that IS task B) and GSM8K (SmolLM2-135M scores ~1.4%,
a floor with no headroom to forget). Uses EleutherAI's lm-evaluation-harness so
numbers are comparable to the model card (HellaSwag 40.9, ARC 37.3, PIQA 66.3)
rather than a hand-rolled scorer.

    uv run python -m src.retention_eval                                   # base Instruct
    uv run python -m src.retention_eval trainer_output/grpo-default/checkpoint-20
    RETENTION_LIMIT=200 uv run python -m src.retention_eval <path>        # fast subset

avg_previous_task_acc = mean(HellaSwag, ARC, PIQA), where ARC follows the model
card and is itself mean(arc_easy, arc_challenge). Metric is acc_norm (the
leaderboard convention) so the base-model numbers line up with the card.
"""

import json
import os
import sys
from typing import Any, cast

from lm_eval import simple_evaluate

# arc_easy + arc_challenge are averaged into the single "ARC" component below.
TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa"]
RESULTS_PATH = "trainer_output/retention_runs.json"


def _acc(task_result):
    """Length-normalized accuracy (leaderboard convention); fall back to raw acc."""
    for key in ("acc_norm,none", "acc,none"):
        if key in task_result:
            return task_result[key]
    raise KeyError(f"no acc/acc_norm in result keys: {list(task_result)}")


def retention_score(model_path, limit=None, batch_size="auto"):
    """Run the basket on one checkpoint and return per-task + averaged accuracy.

    limit caps examples PER TASK (fixed subset for fast trajectory sweeps); None
    runs the full sets. Keep it identical across checkpoints so numbers compare.
    """
    # lm_eval wraps simple_evaluate in @positional_deprecated, so pyright sees the
    # decorator's signature, not the real kwargs -- cast to Any to call it cleanly.
    results = cast(Any, simple_evaluate)(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=TASKS,
        limit=limit,
        batch_size=batch_size,
        device="cuda",
    )["results"]

    per_task = {t: _acc(results[t]) for t in TASKS}
    # ARC (Average) = mean(easy, challenge), matching the model card.
    arc = (per_task["arc_easy"] + per_task["arc_challenge"]) / 2
    basket = {"hellaswag": per_task["hellaswag"], "arc": arc, "piqa": per_task["piqa"]}
    avg = sum(basket.values()) / len(basket)
    return {
        "model": model_path,
        "limit": limit,
        "per_task": per_task,  # raw arc_easy / arc_challenge kept for inspection
        "basket": basket,
        "avg_previous_task_acc": avg,
    }


def record(entry):
    """Append one probe result, keyed by model path so re-runs overwrite."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    runs = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            runs = json.load(f)
    runs = [r for r in runs if r.get("model") != entry["model"]]
    runs.append(entry)
    with open(RESULTS_PATH, "w") as f:
        json.dump(runs, f, indent=2)


if __name__ == "__main__":
    model_path = (
        sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    limit = (
        int(os.environ["RETENTION_LIMIT"])
        if os.environ.get("RETENTION_LIMIT")
        else None
    )

    out = retention_score(model_path, limit=limit)
    record(out)

    b = out["basket"]
    if limit is not None:
        print(f"[note] subset limit={limit} per task -- not the full-set number")
    print(
        f"\navg previous-task acc: {out['avg_previous_task_acc']:.4f}  "
        f"(HellaSwag={b['hellaswag']:.3f}  ARC={b['arc']:.3f}  PIQA={b['piqa']:.3f})"
    )
    print(f"recorded -> {RESULTS_PATH}")

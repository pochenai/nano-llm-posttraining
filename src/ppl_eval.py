"""Held-out perplexity probe: a second, more sensitive "forgetting" instrument.

The commonsense-MC basket (retention_eval.py) showed ZERO forgetting across all
arms even at forward-KL/seq ~17 -- MC accuracy on pretrained knowledge is robust
to instruction fine-tuning at 135M. Perplexity on held-out general text is far
more sensitive to distribution drift (the very thing KL measures), so if the
KL-heavy arms actually sacrificed general language ability, PPL should rise here
where accuracy did not.

Uses lm-eval's `wikitext` task (wikitext-2-raw, standard rolling word_perplexity)
rather than a hand-rolled strided PPL loop, so the number is a known quantity.

    uv run python -m src.ppl_eval                                   # base Instruct
    uv run python -m src.ppl_eval trainer_output/offdist-sft

Lower is better. A KL-heavy arm forgetting general LM ability => higher PPL than
baseline; a flat PPL across arms => the forgetting null holds on this probe too.
"""

import json
import os
import sys
from typing import Any, cast

from lm_eval import simple_evaluate

TASK = "wikitext"
RESULTS_PATH = "trainer_output/ppl_runs.json"


def ppl_score(model_path, batch_size="auto"):
    """Return wikitext word/byte perplexity + bits-per-byte for one checkpoint."""
    res = cast(Any, simple_evaluate)(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=[TASK],
        batch_size=batch_size,
        device="cuda",
    )["results"][TASK]

    # Keys look like "word_perplexity,none"; grab whichever the task version emits.
    def pick(prefix):
        for k, v in res.items():
            if k.startswith(prefix):
                return v
        return None

    return {
        "model": model_path,
        "word_perplexity": pick("word_perplexity"),
        "byte_perplexity": pick("byte_perplexity"),
        "bits_per_byte": pick("bits_per_byte"),
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
    out = ppl_score(model_path)
    record(out)
    print(
        f"\n{model_path}\n  word_ppl={out['word_perplexity']:.3f}  "
        f"byte_ppl={out['byte_perplexity']:.4f}  bpb={out['bits_per_byte']:.4f}"
    )
    print(f"recorded -> {RESULTS_PATH}")

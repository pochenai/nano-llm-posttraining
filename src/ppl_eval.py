"""Held-out perplexity probe: a second, more sensitive "forgetting" instrument.

The commonsense-MC basket (retention_eval.py) showed ZERO forgetting across all
arms even at forward-KL/seq ~17 -- MC accuracy on pretrained knowledge is robust
to instruction fine-tuning at 135M. Perplexity on held-out general text is far
more sensitive to distribution drift (the very thing KL measures), so if the
KL-heavy arms actually sacrificed general language ability, PPL should rise here
where accuracy did not.

Two corpora, both via lm-eval's built-in rolling word_perplexity (identical
methodology), so the forgetting signal can't be an artifact of one text source:
  - wikitext : wikitext-2-raw, clean Wikipedia prose.
  - pile_10k : NeelNanda/pile-10k, a diverse Pile sample (web/books/arxiv/code).
FineWeb-Edu (SmolLM2's own training family) has no small slice, so pile_10k is
the practical "different genre, general web" second opinion.

    uv run python -m src.ppl_eval                                   # base Instruct
    uv run python -m src.ppl_eval trainer_output/offdist-sft

Lower is better. A KL-heavy arm forgetting general LM ability => higher PPL than
baseline on BOTH corpora; flat on both => the forgetting null holds here too.
"""

import json
import os
import sys
from typing import Any, cast

from lm_eval import simple_evaluate

# (task, per-task doc limit). pile_10k docs can be long, so cap them for bounded
# runtime; wikitext is small enough to run whole. Keep limits fixed across models.
CORPORA = [("wikitext", None), ("pile_10k", 500)]
RESULTS_PATH = "trainer_output/ppl_runs.json"


def _pick(res, prefix):
    for k, v in res.items():
        if k.startswith(prefix):
            return v
    return None


# Fixed batch_size=1: "auto" over-estimates on pile_10k's long docs and OOMs an
# 8GB card; the 135M model is fast enough that batch 1 is still quick.
def ppl_score(model_path, batch_size=1):
    """Return word/byte perplexity + bits-per-byte on each corpus for one model."""
    out: dict[str, Any] = {"model": model_path}
    for task, limit in CORPORA:
        res = cast(Any, simple_evaluate)(
            model="hf",
            model_args=f"pretrained={model_path},dtype=bfloat16",
            tasks=[task],
            limit=limit,
            batch_size=batch_size,
            device="cuda",
        )["results"][task]
        out[task] = {
            "limit": limit,
            "word_perplexity": _pick(res, "word_perplexity"),
            "byte_perplexity": _pick(res, "byte_perplexity"),
            "bits_per_byte": _pick(res, "bits_per_byte"),
        }
    return out


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
    print(f"\n{model_path}")
    for task, _ in CORPORA:
        t = out[task]
        print(
            f"  {task:<10} word_ppl={t['word_perplexity']:.3f}  bpb={t['bits_per_byte']:.4f}"
        )
    print(f"recorded -> {RESULTS_PATH}")

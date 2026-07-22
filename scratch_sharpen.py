"""Does GRPO *sharpen* the base? Compare per-problem pass rate, base vs GRPO (vLLM).

RL amplifies paths the base already samples: it should lift problems the base gets
right *sometimes* (pass rate ~0.3-0.7) toward 1.0, while doing little for problems the
base never gets (~0, no reward signal) or always gets (~1, nothing to learn). This
measures exactly that -- sample K completions per problem for both policies and bin
the GRPO pass rate by the base pass rate.

One vLLM engine serves both: base = no LoRA, GRPO = the trained LoRA adapter toggled
in via LoRARequest. Needs the `--extra vllm` environment and a trained checkpoint.

    uv run python scratch_sharpen.py
"""

import os
from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers.trainer_utils import get_last_checkpoint

from src.gsm8k_rewards import Completion, load_gsm8k

N = int(os.environ.get("SHARPEN_N", 60))
K = int(os.environ.get("SHARPEN_K", 8))
TEMP = float(os.environ.get("SHARPEN_TEMP", 0.8))
BASE = os.environ.get("GRPO_COT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.environ.get("GRPO_COT_DIR", "trainer_output/grpo-cot-default")

adapter = get_last_checkpoint(OUTPUT_DIR)
assert adapter is not None, f"no checkpoint under {OUTPUT_DIR} -- train grpo_cot first"

ds: Any = load_gsm8k("test", limit=N)
llm = LLM(model=BASE, enable_lora=True, max_lora_rank=16, max_model_len=1024)
tok = llm.get_tokenizer()
prompts = [
    tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
    for ex in ds
]
# n=K -> K samples per prompt in one call; temperature>0 so pass rate is meaningful.
sp = SamplingParams(n=K, temperature=TEMP, max_tokens=384)

print(f"sampling base ({N} problems x {K}) ...")
base_outs = llm.generate(prompts, sp)
print("sampling GRPO (same, with LoRA adapter) ...")
grpo_outs = llm.generate(prompts, sp, lora_request=LoRARequest("grpo", 1, adapter))


def pass_rate(out, gold):
    return sum(Completion(o.text).is_correct(gold) for o in out.outputs) / K


rows = []
for ex, bo, go in zip(ds, base_outs, grpo_outs):
    rows.append((pass_rate(bo, ex["gold"]), pass_rate(go, ex["gold"]), ex))

# Overall pass@1 (mean of per-problem rates).
mb = sum(r[0] for r in rows) / len(rows)
mg = sum(r[1] for r in rows) / len(rows)
print(f"\n== overall pass@1: base {mb:.3f} -> GRPO {mg:.3f}  (delta {mg - mb:+.3f}) ==")

# Bin GRPO lift by where the base started. The sharpening prediction: biggest lift in
# the middle band, ~zero in the base~0 (no signal) and base~1 (already solved) bands.
bins = [(-0.01, 0.0), (0.0, 0.25), (0.25, 0.75), (0.75, 0.99), (0.99, 1.01)]
labels = ["base=0", "0<b<=.25", ".25<b<.75 (uncertain)", ".75<=b<1", "base=1"]
print("\n== GRPO lift binned by base pass rate ==")
for (lo, hi), lab in zip(bins, labels):
    grp = [r for r in rows if lo < r[0] <= hi]
    if not grp:
        continue
    b = sum(r[0] for r in grp) / len(grp)
    g = sum(r[1] for r in grp) / len(grp)
    print(f"  {lab:24s} n={len(grp):2d}  base {b:.2f} -> GRPO {g:.2f}  ({g - b:+.2f})")

# The uncertain-band problems: this is where sharpening should be visible per-problem.
print("\n== uncertain-band problems (.25 < base < .75): base -> GRPO ==")
for br, gr, ex in sorted([r for r in rows if 0.25 < r[0] < 0.75], key=lambda r: r[0]):
    q = " ".join(ex["prompt"][-1]["content"].split())[:90]
    print(f"  base {br:.2f} -> GRPO {gr:.2f}  ({gr - br:+.2f})  {q}")

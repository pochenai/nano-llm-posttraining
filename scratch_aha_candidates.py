"""Find GSM8K problems the base model gets WRONG -> aha-moment candidates (vLLM).

A wrong answer that still shows a coherent multi-step chain (right setup, arithmetic
slip, near-miss number) is the best place to watch for a post-training "aha moment"
(self-correction). A totally-lost answer is a weaker candidate. Rank accordingly.

vLLM batches/schedules all N prompts at once, so this is much faster than a per-prompt
HF generate loop. Needs the `--extra vllm` environment.

    uv run python scratch_aha_candidates.py
"""

from typing import Any

from vllm import LLM, SamplingParams

from src.gsm8k_rewards import Completion, load_gsm8k, to_number

N = 60
BASE = "Qwen/Qwen2.5-1.5B-Instruct"

# Any: datasets rows are dicts at runtime but pyright types them as non-subscriptable.
ds: Any = load_gsm8k("test", limit=N)

llm = LLM(model=BASE)
tok = llm.get_tokenizer()
prompts = [
    tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
    for ex in ds
]
# temperature=0 -> greedy, deterministic (comparable across runs). vLLM returns the
# outputs in the same order as the prompts, so we can zip them back with ds.
sp = SamplingParams(temperature=0.0, max_tokens=384)
outs = llm.generate(prompts, sp)

wrong = []
for i, (ex, o) in enumerate(zip(ds, outs)):
    r = o.outputs[0].text.strip()
    c = Completion(r)
    if c.is_correct(ex["gold"]):
        continue
    gold_n = to_number(ex["gold"])
    pred_n = c.final_number
    # aha potential: attempted a reasoning chain AND produced a wrong-but-parseable
    # number. Near-miss (same order of magnitude) ranks highest.
    has_cot = c.tagged_answer is not None or "<reasoning>" in r
    near = 0.0
    if pred_n is not None and gold_n not in (None, 0):
        ratio = abs(pred_n - gold_n) / abs(gold_n)
        near = 1.0 if ratio <= 0.5 else (0.5 if ratio <= 2 else 0.0)
    score = (2 if has_cot else 0) + near + (0.5 if pred_n is not None else 0)
    q = ex["prompt"][-1]["content"]
    wrong.append((score, i, q, ex["gold"], pred_n, has_cot, r))

wrong.sort(key=lambda t: -t[0])
print(f"\n===== base got {len(wrong)}/{N} WRONG (aha candidates first) =====\n")
for score, i, q, gold, pred, has_cot, r in wrong:
    print("=" * 90)
    print(
        f"[#{i}] aha_score={score:.1f}  gold={gold}  pred={pred}  cot={'Y' if has_cot else 'N'}"
    )
    print("Q:", " ".join(q.split())[:200])
    print("model CoT:", repr(r[:350]))

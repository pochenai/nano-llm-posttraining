"""Find GSM8K problems the base Qwen2.5-0.5B gets WRONG -> aha-moment candidates.

A wrong answer that still shows a coherent multi-step chain (right setup, arithmetic
slip, near-miss number) is the best place to watch for a post-training "aha moment"
(self-correction). A totally-lost answer is a weaker candidate. Rank accordingly.

    uv run python scratch_aha_candidates.py
"""

from typing import Any

import torch
from transformers import GenerationConfig

from src.gsm8k_rewards import Completion, load_gsm8k, to_number
from src.model_loader import load_model_and_tokenizer

N = 60
BASE = "Qwen/Qwen2.5-1.5B-Instruct"

# Any: datasets rows are dicts at runtime but pyright types them as non-subscriptable.
ds: Any = load_gsm8k("test", limit=N)
model, tok = load_model_and_tokenizer(model_name=BASE, use_gpu=True)
gc = GenerationConfig(
    max_new_tokens=384,
    do_sample=False,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)
model.eval()

wrong = []
for i, ex in enumerate(ds):
    p = tok.apply_chat_template(
        ex["prompt"], tokenize=False, add_generation_prompt=True
    )
    inp = tok(p, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, generation_config=gc)
    r = tok.decode(
        out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
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

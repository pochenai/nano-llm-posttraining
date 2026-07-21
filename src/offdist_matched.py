"""Fully-matched in-dist vs off-dist SFT, to isolate the DATA-DISTRIBUTION effect.

Improves on offdist_compare.py by removing its two weaknesses (sample-size
mismatch, too few optimizer steps):

  * both arms train on the SAME prompts AND the SAME count -- the INTERSECTION of
    prompts for which both a Claude label (off-dist) and a base best-of-K label
    (in-dist) satisfy the checker;
  * EPOCHS is high enough that the model actually moves out of the KL noise floor.

Arms differ ONLY in where the (checker-verified) target answer comes from:
  indist : the base model's own best-of-K sample     (in-distribution)
  offdist: an answer written by Claude               (off-distribution)

    MERGED_ANSWERS=<merged.json> EPOCHS=10 uv run python -m src.offdist_matched
"""

import json
import os
from typing import Any

import torch
from trl import SFTTrainer, SFTConfig  # pyright: ignore[reportPrivateImportUsage]
from transformers import GenerationConfig
from datasets import Dataset

from src.ifeval_rewards import load_ifeval, Response
from src.model_loader import load_model_and_tokenizer

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM_MESSAGE = "You are a helpful assistant."
MAX_COMPLETION_LENGTH = 200
K = int(os.environ.get("RFT_K", 16))
EPOCHS = float(os.environ.get("EPOCHS", 15))
# SFT on ~165 examples needs a higher LR than GRPO's 1e-5: with so few optimizer
# steps, 1e-5 never fits the targets (train-set satisfaction stayed ~0.38 vs the
# 1.0 targets). 5e-5 actually memorizes what each arm is taught.
LR = float(os.environ.get("LR", 5e-5))
GEN_BATCH = 16

# Claude's off-distribution expert answers ship with the repo for reproducibility;
# override with MERGED_ANSWERS to point at a different set.
ANSWERS_PATH = os.environ.get("MERGED_ANSWERS", "src/ifeval_claude_answers.json")
answers_raw = json.load(open(ANSWERS_PATH))
CLAUDE = {int(k): v for k, v in answers_raw.items()}

dataset = load_ifeval(max_words=100, system_message=SYSTEM_MESSAGE)
split = dataset.train_test_split(test_size=0.2, seed=42)
# eval_dataset: Any because pyright types iterated Dataset rows as non-subscriptable.
full_train = split["train"]
eval_dataset: Any = split["test"]


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, gc):
    orig = tokenizer.padding_side
    tokenizer.padding_side = "left"
    out = []
    try:
        for i in range(0, len(prompts), GEN_BATCH):
            b = prompts[i : i + GEN_BATCH]
            inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True,
                            max_length=512).to(model.device)
            g = model.generate(**inp, generation_config=gc)
            for j in range(len(b)):
                out.append(tokenizer.decode(g[j][inp["input_ids"].shape[1]:],
                                            skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = orig
    return out


def sat(ans, ex):
    return Response(ans).satisfaction(ex["instruction_id_list"], ex["constraint_kwargs"])


def row(ex, completion):
    return {"messages": [
        {"role": "system", "content": SYSTEM_MESSAGE},
        ex["prompt"][-1],
        {"role": "assistant", "content": completion},
    ]}


def build_matched():
    """Return (claude_dataset, indist_dataset) over the shared, size-matched set."""
    base, tok = load_model_and_tokenizer(model_name=BASE_MODEL, use_gpu=True)
    # Claude labels that pass.
    claude_ok = {i: CLAUDE[i] for i in CLAUDE
                 if sat(CLAUDE[i], full_train[i]) >= 1.0}
    print(f"claude labels passing: {len(claude_ok)}/{len(CLAUDE)}")

    # In-distribution best-of-K on the SAME prompts.
    gc = GenerationConfig(max_new_tokens=MAX_COMPLETION_LENGTH, do_sample=True,
                          temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id,
                          eos_token_id=tok.eos_token_id)
    idxs = sorted(claude_ok)
    prompts = [tok.apply_chat_template(full_train[i]["prompt"], tokenize=False,
               add_generation_prompt=True) for i in idxs]
    repeated = [p for p in prompts for _ in range(K)]
    comps = generate_batch(base, tok, repeated, gc)
    indist_ok = {}
    for pos, i in enumerate(idxs):
        cand = comps[pos * K:(pos + 1) * K]
        bs, bc = max(((sat(c, full_train[i]), c) for c in cand), key=lambda t: t[0])
        if bs >= 1.0:
            indist_ok[i] = bc
    print(f"in-dist labels passing: {len(indist_ok)}/{len(idxs)}")

    # Intersection -> identical prompts AND identical count for both arms.
    matched = sorted(set(claude_ok) & set(indist_ok))
    print(f"MATCHED set (both pass): {len(matched)} prompts")
    del base
    torch.cuda.empty_cache()
    claude_ds = Dataset.from_list([row(full_train[i], claude_ok[i]) for i in matched])
    indist_ds = Dataset.from_list([row(full_train[i], indist_ok[i]) for i in matched])
    return claude_ds, indist_ds, matched


def eval_ifeval(model, tok):
    gc = GenerationConfig(max_new_tokens=MAX_COMPLETION_LENGTH, do_sample=False,
                          repetition_penalty=1.1, pad_token_id=tok.pad_token_id,
                          eos_token_id=tok.eos_token_id)
    prompts = [tok.apply_chat_template(ex["prompt"], tokenize=False,
               add_generation_prompt=True) for ex in eval_dataset]
    model.eval()
    resp = generate_batch(model, tok, prompts, gc)
    return sum(sat(r, ex) for r, ex in zip(resp, eval_dataset)) / len(eval_dataset)


def train_arm(name, ds):
    out_dir = f"trainer_output/{name}-sft"
    model, tok = load_model_and_tokenizer(model_name=BASE_MODEL, use_gpu=True)
    cfg = SFTConfig(output_dir=out_dir, learning_rate=LR, num_train_epochs=EPOCHS,
                    per_device_train_batch_size=8, gradient_accumulation_steps=4,
                    bf16=True, logging_steps=20, save_total_limit=1, report_to="none")
    SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok).train()
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    acc = eval_ifeval(model, tok)
    print(f"[matched-{name}] IFEval on held-out {len(eval_dataset)}: {acc:.4f}")
    del model
    torch.cuda.empty_cache()
    return acc


if __name__ == "__main__":
    claude_ds, indist_ds, matched = build_matched()
    steps = len(matched) * EPOCHS / 32
    print(f"~optimizer steps per arm: {steps:.0f}")
    acc_off = train_arm("offdist", claude_ds)
    acc_in = train_arm("indist", indist_ds)
    print("\n=== fully-matched (same prompts, same count) ===")
    print(f"  offdist-sft (Claude, off-dist) IFEval: {acc_off:.4f}")
    print(f"  indist-sft  (base,   in-dist)  IFEval: {acc_in:.4f}")
    print("\nNext: forward KL for each")
    print("  KL_MODEL=trainer_output/offdist-sft KL_LABEL=offdist-sft uv run python -m src.kl_analysis")
    print("  KL_MODEL=trainer_output/indist-sft  KL_LABEL=indist-sft  uv run python -m src.kl_analysis")

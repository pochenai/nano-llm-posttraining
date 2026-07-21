"""Rejection fine-tuning (RFT): the SFT baseline that shares GRPO's data source
and reward, so RL vs SFT isolates only the update rule.

Same base (Instruct), same prompts (the IFEval train split grpo.py uses), same
reward (the verifiable checker), same held-out eval. The ONLY difference from
GRPO is the update: GRPO does on-policy policy gradient; RFT samples K
completions per prompt from the base, keeps the best-scoring one, and does plain
next-token SFT on it. This removes the "different dataset / different prompt
distribution" confound of the earlier SFT-on-smol-contraints comparison.

    GRPO_SKIP ...  uv run python -m src.rft        # build data + train + eval
"""

import os
from typing import Any

import torch

# trl re-exports these but doesn't list them in __all__.
from trl import SFTTrainer, SFTConfig  # pyright: ignore[reportPrivateImportUsage]
from transformers import GenerationConfig
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

from src import LOAD_CHECKPOINT
from src.ifeval_rewards import load_ifeval, Response
from src.model_loader import load_model_and_tokenizer

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM_MESSAGE = "You are a helpful assistant."
MAX_COMPLETION_LENGTH = 200
OUTPUT_DIR = "trainer_output/rft"

# Rejection-sampling knobs. K completions per prompt, keep the best if it clears
# MIN_SAT -- a 135M base rarely satisfies every constraint, so best-of-K with a
# partial-credit floor is what yields usable labels at all.
K = int(os.environ.get("RFT_K", 16))
MIN_SAT = float(os.environ.get("RFT_MIN_SAT", 0.5))
TEMPERATURE = float(os.environ.get("RFT_TEMP", 1.0))
GEN_BATCH = int(os.environ.get("RFT_BATCH", 16))

last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

######################
# data: same split as grpo.py
######################
dataset = load_ifeval(max_words=100, system_message=SYSTEM_MESSAGE)
split = dataset.train_test_split(test_size=0.2, seed=42)
# Dataset rows are dicts at runtime, but pyright types them as non-subscriptable
# when iterated; annotate Any so ex["prompt"] etc. type-check.
train_dataset: Any = split["train"]
eval_dataset: Any = split["test"]
print(f"train prompts={len(train_dataset)}  eval={len(eval_dataset)}")


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, gen_config):
    """Greedy/sampled completions for a list of chat-templated prompt strings."""
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    out_texts = []
    try:
        for i in range(0, len(prompts), GEN_BATCH):
            batch = prompts[i : i + GEN_BATCH]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            ).to(model.device)
            out = model.generate(**inputs, generation_config=gen_config)
            for j in range(len(batch)):
                new = out[j][inputs["input_ids"].shape[1] :]
                out_texts.append(tokenizer.decode(new, skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = original_side
    return out_texts


def build_rft_dataset(model, tokenizer):
    """Sample K completions per train prompt, keep the best that clears MIN_SAT."""
    sample_config = GenerationConfig(
        max_new_tokens=MAX_COMPLETION_LENGTH,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Repeat each prompt K times so one batched pass yields K samples per prompt.
    prompt_strs = [
        tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True
        )
        for ex in train_dataset
    ]
    repeated = [p for p in prompt_strs for _ in range(K)]
    print(f"Sampling {K} completions x {len(train_dataset)} prompts = {len(repeated)} ...")
    completions = generate_batch(model, tokenizer, repeated, sample_config)

    rows = []
    kept_scores = []
    for idx, ex in enumerate(train_dataset):
        cand = completions[idx * K : (idx + 1) * K]
        scored = [
            (Response(c).satisfaction(ex["instruction_id_list"], ex["constraint_kwargs"]), c)
            for c in cand
        ]
        best_score, best_c = max(scored, key=lambda t: t[0])
        if best_score >= MIN_SAT:
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        ex["prompt"][-1],  # the user turn
                        {"role": "assistant", "content": best_c},
                    ]
                }
            )
            kept_scores.append(best_score)

    coverage = len(rows) / len(train_dataset)
    mean_label = sum(kept_scores) / len(kept_scores) if kept_scores else 0.0
    print(f"kept {len(rows)}/{len(train_dataset)} prompts (coverage={coverage:.2f}), "
          f"mean label satisfaction={mean_label:.3f}")
    return Dataset.from_list(rows)


def eval_ifeval(model, tokenizer, dataset, title):
    """Mean constraint satisfaction on the held-out split (greedy)."""
    gen_config = GenerationConfig(
        max_new_tokens=MAX_COMPLETION_LENGTH,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompts = [
        tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True
        )
        for ex in dataset
    ]
    model.eval()
    responses = generate_batch(model, tokenizer, prompts, gen_config)
    scores = [
        Response(r).satisfaction(ex["instruction_id_list"], ex["constraint_kwargs"])
        for r, ex in zip(responses, dataset)
    ]
    mean = sum(scores) / len(scores)
    print(f"\n=== {title} ===\n--> mean satisfaction: {mean:.4f}  (n={len(scores)})")
    return mean


######################
# load-or-train
######################
model, tokenizer = load_model_and_tokenizer(model_name=BASE_MODEL, use_gpu=True)

if LOAD_CHECKPOINT and last_ckpt:
    print(f"Loading RFT model from {last_ckpt} (skip training)")
    model, tokenizer = load_model_and_tokenizer(model_name=last_ckpt, use_gpu=True)
else:
    rft_dataset = build_rft_dataset(model, tokenizer)
    # Reload a clean base to train on (the sampling model is the same weights, but
    # keep the flow explicit: RFT trains the base on its own filtered samples).
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,  # match grpo.py's LR so the comparison is about the update rule
        num_train_epochs=3,  # match grpo.py's epochs
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        bf16=True,
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
    )
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=rft_dataset,
        processing_class=tokenizer,
    )
    sft_trainer.train()
    sft_trainer.save_model(OUTPUT_DIR)
    model, tokenizer = load_model_and_tokenizer(model_name=OUTPUT_DIR, use_gpu=True)

######################
# eval
######################
eval_ifeval(model, tokenizer, eval_dataset, "RFT (rejection fine-tuning)")
print("\nNext: measure forward KL with")
print("  KL_MODEL=trainer_output/rft KL_LABEL=rft KL_DIRECTION=forward uv run python -m src.kl_analysis")

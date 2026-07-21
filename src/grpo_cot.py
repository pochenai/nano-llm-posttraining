"""GRPO on GSM8K to elicit chain-of-thought reasoning.

Designed to be debugged locally on a small card and then moved to a rented GPU
by changing environment variables only:

    # local (8GB): prove the loop runs, reward curve moves
    GRPO_COT_TRAIN_LIMIT=200 uv run python -m src.grpo_cot

    # cloud (24GB 4090): the real run
    GRPO_COT_MODEL=Qwen/Qwen2.5-3B-Instruct GRPO_COT_VLLM=1 uv run python -m src.grpo_cot
"""

import json
import os

import torch
from peft import LoraConfig

# trl re-exports these but doesn't list them in __all__, which trips pyright's
# reportPrivateImportUsage; the runtime import is the documented public API.
from trl import GRPOTrainer, GRPOConfig  # pyright: ignore[reportPrivateImportUsage]
from transformers import GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

from src import LOAD_CHECKPOINT
from src.gsm8k_rewards import (
    REWARD_FUNCS,
    SYSTEM_MESSAGE,
    Completion,
    load_gsm8k,
)
from src.model_loader import load_model_and_tokenizer

# Qwen2.5-0.5B-Instruct already emits some step-by-step math, which is the point:
# GRPO amplifies behaviour the policy can already sample, it cannot invent it. A
# model that never samples a correct answer gives every group zero reward
# variance -> zero advantage -> no gradient.
BASE_MODEL = os.environ.get("GRPO_COT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# 256 clipped 30-56% of completions in a smoke run, and a truncated completion
# poisons last-number scoring: the final number becomes a mid-reasoning value, so
# a correct chain gets marked wrong. Lower it via env if VRAM is tight.
MAX_COMPLETION_LENGTH = int(os.environ.get("GRPO_COT_COMPLETION_LEN", 384))

# Ablation knobs, overridable per run so variants stay comparable.
RUN = os.environ.get("GRPO_COT_RUN", "default")
NUM_GENERATIONS = int(os.environ.get("GRPO_COT_NUM_GENERATIONS", 8))
NUM_EPOCHS = float(os.environ.get("GRPO_COT_EPOCHS", 1))
BETA = float(os.environ.get("GRPO_COT_BETA", 0.02))
TRAIN_LIMIT = int(os.environ.get("GRPO_COT_TRAIN_LIMIT", 0))  # 0 = full split
EVAL_LIMIT = int(os.environ.get("GRPO_COT_EVAL_LIMIT", 100))
SKIP_TRAIN = os.environ.get("GRPO_COT_SKIP_TRAIN") == "1"
# LoRA keeps optimizer state tiny and gives the KL reference model for free (the
# adapter is just disabled), which is what makes 3B fit on a 24GB card.
USE_LORA = os.environ.get("GRPO_COT_LORA", "1") == "1"
# vLLM makes rollouts several times faster, but wants spare VRAM for its own KV
# cache -- worth it on a rented card, usually too tight on 8GB.
USE_VLLM = os.environ.get("GRPO_COT_VLLM") == "1"

OUTPUT_DIR = f"trainer_output/grpo-cot-{RUN}"
RESULTS_PATH = "trainer_output/grpo_cot_runs.json"
last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

######################
# load data
######################
train_dataset = load_gsm8k("train", limit=TRAIN_LIMIT or None)
eval_dataset = load_gsm8k("test", limit=EVAL_LIMIT)
print(f"train={len(train_dataset)}  eval={len(eval_dataset)}")


def eval_gsm8k(model, tokenizer, dataset, title, batch_size=8):
    """Exact-match accuracy on held-out GSM8K -- the same signal GRPO optimizes,
    so before/after numbers are directly comparable.

    Generation is batched (left-padded) because a per-prompt loop dominates the
    wall time of a short run.
    """
    print(f"\n=== {title} ===")
    prompts = [
        tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True
        )
        for ex in dataset
    ]

    gen_config = GenerationConfig(
        max_new_tokens=MAX_COMPLETION_LENGTH,
        do_sample=False,  # greedy: deterministic, so run-to-run deltas are real.
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # right padding corrupts batched generation
    responses = []
    model.eval()
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, generation_config=gen_config)
            for j in range(len(batch)):
                new_ids = out[j][inputs["input_ids"].shape[1] :]
                responses.append(
                    tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                )
    finally:
        tokenizer.padding_side = original_side

    # Parse each completion once; accuracy and format adherence both read from it.
    parsed = [Completion(r) for r in responses]
    correct = sum(c.is_correct(ex["gold"]) for c, ex in zip(parsed, dataset))
    # Format adherence is tracked separately: it usually climbs first and is the
    # early sign GRPO is working even while accuracy is still flat.
    formatted = sum(1 for c in parsed if c.tagged_answer is not None)
    acc = correct / len(responses)
    print(f"--> accuracy: {acc:.4f}  ({correct}/{len(responses)})")
    print(f"--> has <answer> tag: {formatted}/{len(responses)}")
    return {"accuracy": acc, "n": len(responses), "formatted": formatted}


def record(entry):
    """Append one run's result so variants stay comparable across invocations."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    runs = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            runs = json.load(f)
    runs = [r for r in runs if r.get("run") != entry["run"]]
    runs.append(entry)
    with open(RESULTS_PATH, "w") as f:
        json.dump(runs, f, indent=2)
    print(f"\nrecorded -> {RESULTS_PATH}")


def cached_baseline():
    """The policy init is identical across runs, so measure "before" only once."""
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        for r in json.load(f):
            if r.get("run") == f"baseline-{BASE_MODEL}":
                return r["after"]
    return None


######################
# load-or-train
######################
model, tokenizer = load_model_and_tokenizer(model_name=BASE_MODEL, use_gpu=True)
print(f"Policy initialized from {BASE_MODEL}  (lora={USE_LORA}, vllm={USE_VLLM})")

config = {
    "base_model": BASE_MODEL,
    "num_generations": NUM_GENERATIONS,
    "num_train_epochs": NUM_EPOCHS,
    "beta": BETA,
    "lora": USE_LORA,
    "train_size": len(train_dataset),
}

if SKIP_TRAIN:
    # Baseline mode: score the untrained policy on the held-out split.
    before = eval_gsm8k(model, tokenizer, eval_dataset, "Baseline (no GRPO)")
    record({"run": f"baseline-{BASE_MODEL}", "config": config, "before": None,
            "after": before})
    raise SystemExit(0)

before = cached_baseline()
if before is None:
    before = eval_gsm8k(model, tokenizer, eval_dataset, "Before GRPO")
else:
    print(f"\nUsing cached baseline: {before['accuracy']:.4f}")

if LOAD_CHECKPOINT and last_ckpt:
    print(f"Loading GRPO model from {last_ckpt} (skip training)")
    model, tokenizer = load_model_and_tokenizer(model_name=last_ckpt, use_gpu=True)
else:
    # r=16 on the attention + MLP projections. Bigger r buys capacity at the cost
    # of optimizer memory; 16 is the usual starting point for a few-B policy.
    peft_config = (
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        if USE_LORA
        else None
    )

    # GRPO samples num_generations completions per prompt and normalizes reward
    # within that group, so the global batch (per_device x grad_accum) must be a
    # multiple of it. 4 x 8 = 32 keeps only 4 sequences in a forward pass, which
    # is what leaves room for a 0.5B policy plus KV cache on 8GB.
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5 if USE_LORA else 1e-6,  # LoRA tolerates a higher LR.
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_generations=NUM_GENERATIONS,  # Group size G. Larger -> fewer zero-variance groups.
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=1.0,  # High enough to create in-group variance.
        beta=BETA,  # KL penalty toward the reference policy.
        use_vllm=USE_VLLM,
        vllm_mode="colocate",  # share this GPU instead of a separate server
        vllm_gpu_memory_utilization=0.3,
        bf16=True,
        gradient_checkpointing=True,  # trade ~30% speed for activation memory
        logging_steps=5,
        save_total_limit=1,
        report_to="none",
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=REWARD_FUNCS,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    # What to watch in the logs:
    #   frac_reward_zero_std  THE health metric. Fraction of groups where every
    #                    completion scored the same -> advantage 0 -> no gradient.
    #                    Near 1.0 means training is a no-op no matter how long it
    #                    runs. Seen at 0.92 here when the reward demanded XML tags
    #                    the policy never emits; fixed by scoring the last number
    #                    instead. If it climbs: raise temperature or G, or make
    #                    the reward able to see what the policy actually does.
    #   reward / reward_std   should be > 0 and trending up.
    #   rewards/correctness_reward  the real objective.
    #   rewards/xmlcount_reward     format shaping; usually moves first.
    #   completions/clipped_ratio   high means answers run past the length cap
    #                    without terminating -- raise max_completion_length.
    grpo_trainer.train()
    grpo_trainer.save_model(OUTPUT_DIR)
    model, tokenizer = load_model_and_tokenizer(model_name=OUTPUT_DIR, use_gpu=True)

######################
# eval the trained model
######################
after = eval_gsm8k(model, tokenizer, eval_dataset, f"After GRPO [{RUN}]")
record({"run": RUN, "config": config, "before": before, "after": after})
print(f"\n{RUN}: {before['accuracy']:.4f} -> {after['accuracy']:.4f}")

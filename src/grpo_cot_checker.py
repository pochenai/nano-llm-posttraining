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
USE_VLLM = os.environ.get("GRPO_COT_VLLM", "1") == "1"

OUTPUT_DIR = f"trainer_output/grpo-cot-{RUN}"
RESULTS_PATH = "trainer_output/grpo_cot_runs.json"
last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

######################
# load data
######################
train_dataset = load_gsm8k("train", limit=TRAIN_LIMIT or None)
eval_dataset = load_gsm8k("test", limit=EVAL_LIMIT)
print(f"train={len(train_dataset)}  eval={len(eval_dataset)}")

######################
# load-or-train
######################
print(f"Loading GRPO model from {last_ckpt} (skip training)")
assert (
    last_ckpt is not None
), f"no checkpoint under {OUTPUT_DIR} -- train grpo_cot first"
model, tokenizer = load_model_and_tokenizer(model_name=last_ckpt, use_gpu=True)


######################
# eval the trained model for question on train
######################
def answer_question(idx, dataset=None):
    """Print the loaded model's full chain-of-thought for a single problem.

    Handy for eyeballing an aha-moment candidate: run the same index before and
    after GRPO and watch the chain for self-correction ("wait", "let me recompute",
    "actually"). Defaults to the train split; pass eval_dataset for held-out.
    """
    dataset = train_dataset if dataset is None else dataset
    ex = dataset[idx]
    prompt = tokenizer.apply_chat_template(
        ex["prompt"], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(
        model.device
    )
    gen_config = GenerationConfig(
        max_new_tokens=MAX_COMPLETION_LENGTH,
        do_sample=False,  # greedy: deterministic, so before/after is comparable
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.eval()
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_config)
    resp = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    c = Completion(resp)
    print(
        f"\n===== Q#{idx}  gold={ex['gold']}  correct={c.is_correct(ex['gold'])} ====="
    )
    print("Q:", ex["prompt"][-1]["content"])
    print(f"\nreference CoT: {ex['reference']}")
    print(f"\nmodel answer: {c.final_number}")
    print(f"model CoT:\n{resp}")
    return resp


if __name__ == "__main__":
    # GRPO_COT_Q = which problem index; GRPO_COT_SPLIT = train (default) or test.
    q_idx = int(os.environ.get("GRPO_COT_Q", 0))
    which = (
        train_dataset if os.environ.get("GRPO_COT_SPLIT") == "train" else eval_dataset
    )
    answer_question(q_idx, which)

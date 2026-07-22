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

# Qwen2.5-1.5B-Instruct already emits some step-by-step math, which is the point:
# GRPO amplifies behaviour the policy can already sample, it cannot invent it. A
# model that never samples a correct answer gives every group zero reward
# variance -> zero advantage -> no gradient.
BASE_MODEL = os.environ.get("GRPO_COT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
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
# aha-moment hunting: sample K times, grep each chain for self-correction
######################
# Comprehensive list of self-correction / backtracking / "aha" phrasings a reasoning
# model uses when it catches its own error mid-chain. Substring-matched, lowercased.
# Deliberately broad (many near-synonyms) so a genuine aha is not missed; the script
# prints which markers hit so you can tell a real "wait, that's wrong" from a benign
# "actually". Extend freely.
AHA_MARKERS = sorted(
    {
        # interjections / discourse markers that precede a rethink
        "wait",
        "however",
        "but wait",
        "no wait",
        "no, wait",
        "wait,",
        "wait.",
        "hold on",
        "hang on",
        "hmm",
        "hmm,",
        "oops",
        "aha",
        "oh wait",
        "oh no",
        "actually",
        "actually,",
        "actually no",
        "actually, no",
        "on second thought",
        "on reflection",
        "upon reflection",
        "come to think of it",
        # explicit re-do / re-check phrasings
        "let me reconsider",
        "let me re-examine",
        "let me reexamine",
        "let me recompute",
        "let me recalculate",
        "let me re-calculate",
        "let me rethink",
        "let me re-think",
        "let me think again",
        "think again",
        "let me check",
        "let me double-check",
        "let me double check",
        "double-check",
        "double check",
        "let me verify",
        "verify this",
        "let me re-evaluate",
        "let me reevaluate",
        "let me re-read",
        "let me reread",
        "re-read",
        "read again",
        "let me try again",
        "try again",
        "let me start over",
        "start over",
        "scratch that",
        "let me redo",
        "redo",
        "let me correct",
        "correction",
        "correcting",
        "let me fix",
        "let me revisit",
        "revisit",
        "step back",
        "back up",
        "recheck",
        "re-check",
        "recompute",
        "reconsider",
        "reconsidering",
        "rethinking",
        "re-examine",
        "reevaluate",
        # explicit error admissions
        "that's not right",
        "that is not right",
        "that isn't right",
        "that's wrong",
        "that is wrong",
        "this is wrong",
        "that's incorrect",
        "this is incorrect",
        "that doesn't seem right",
        "doesn't seem right",
        "that can't be right",
        "can't be right",
        "cannot be right",
        "that can't be",
        "i made a mistake",
        "i made an error",
        "i think i made",
        "my mistake",
        "my error",
        "the mistake",
        "the error",
        "i was wrong",
        "that was wrong",
        "i misread",
        "misread",
        "i misinterpreted",
        "misinterpreted",
        "i incorrectly",
        "incorrectly",
        "i need to reconsider",
    }
)


def find_markers(text):
    low = text.lower()
    return sorted({m for m in AHA_MARKERS if m in low})


def hunt_aha(idx, dataset=None, k=16, temp=0.8):
    """Sample K completions for one problem and grep each for self-correction.

    Greedy shows one deterministic chain; an aha moment (if the policy can produce
    it at all) usually appears only in *some* samples, so we sample K at temperature>0
    and surface every chain that contains a self-correction marker.
    """
    dataset = eval_dataset if dataset is None else dataset
    ex = dataset[idx]
    prompt = tokenizer.apply_chat_template(
        ex["prompt"], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(
        model.device
    )
    gen_config = GenerationConfig(
        max_new_tokens=MAX_COMPLETION_LENGTH,
        do_sample=True,
        temperature=temp,
        top_p=0.95,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.eval()
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_config)
    plen = inputs["input_ids"].shape[1]

    print(f"\n===== Q#{idx}  gold={ex['gold']}  (K={k} samples @ temp={temp}) =====")
    print("Q:", ex["prompt"][-1]["content"])
    n_aha, n_correct = 0, 0
    for s in range(k):
        resp = tokenizer.decode(out[s][plen:], skip_special_tokens=True).strip()
        c = Completion(resp)
        correct = c.is_correct(ex["gold"])
        n_correct += correct
        hits = find_markers(resp)
        if hits:
            n_aha += 1
            print(f"\n--- sample {s}  correct={correct}  markers={hits} ---")
            print(resp)
    print(
        f"\n== {n_aha}/{k} samples contain a self-correction marker; "
        f"{n_correct}/{k} correct  (gold={ex['gold']}) =="
    )
    return n_aha, n_correct


if __name__ == "__main__":
    # GRPO_COT_Q = problem index; GRPO_COT_K = samples; GRPO_COT_TEMP = temperature.
    # GRPO_COT_SPLIT = test (default, matches scratch_aha_candidates.py) or train.
    q_idx = int(os.environ.get("GRPO_COT_Q", 0))
    k = int(os.environ.get("GRPO_COT_K", 16))
    temp = float(os.environ.get("GRPO_COT_TEMP", 0.8))
    which = (
        train_dataset if os.environ.get("GRPO_COT_SPLIT") == "train" else eval_dataset
    )
    hunt_aha(q_idx, which, k=k, temp=temp)

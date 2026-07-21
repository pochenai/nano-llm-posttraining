import json
import os

import torch

# trl re-exports these but doesn't list them in __all__, which trips pyright's
# reportPrivateImportUsage; the runtime import is the documented public API.
from trl import GRPOTrainer, GRPOConfig  # pyright: ignore[reportPrivateImportUsage]
from transformers import GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

from src import LOAD_CHECKPOINT
from src.ifeval_rewards import load_ifeval, ifeval_reward, Response
from src.model_loader import load_model_and_tokenizer

# The instruct model already knows how to stop and follow a chat format, so GRPO
# has a policy worth sharpening. The base model almost never samples a valid
# format, which collapses in-group reward variance and kills the gradient.
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM_MESSAGE = "You are a helpful assistant."
MAX_COMPLETION_LENGTH = 200

# Ablation knobs, overridable per run so variants can be compared one at a time.
# Defaults are the winning configuration from the ablation (see grpo_runs.json).
# beta mattered most: at 0.02 the KL penalty pinned the policy to its init
# (kl ~0.002) and nothing was learned no matter what else was tuned.
RUN = os.environ.get("GRPO_RUN", "default")
NUM_GENERATIONS = int(os.environ.get("GRPO_NUM_GENERATIONS", 16))
NUM_EPOCHS = float(os.environ.get("GRPO_EPOCHS", 3))
BETA = float(os.environ.get("GRPO_BETA", 0.005))
SKIP_TRAIN = os.environ.get("GRPO_SKIP_TRAIN") == "1"

OUTPUT_DIR = f"trainer_output/grpo-{RUN}"
RESULTS_PATH = "trainer_output/grpo_runs.json"
last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

######################
# load data
######################
# max_words drops prompts demanding long essays: they can't be satisfied within
# MAX_COMPLETION_LENGTH, so the whole group scores 0 and yields no gradient.
dataset = load_ifeval(max_words=100, system_message=SYSTEM_MESSAGE)
# Held-out split so "after" numbers measure generalization, not memorization.
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset, eval_dataset = split["train"], split["test"]
print(f"train={len(train_dataset)}  eval={len(eval_dataset)}")


def eval_ifeval(model, tokenizer, dataset, title, batch_size=8):
    """Mean constraint satisfaction over the held-out split -- the same signal
    GRPO optimizes, so before/after numbers are directly comparable.

    Generation is batched (left-padded) because a per-prompt loop over ~80
    prompts dominates the wall time of a short training run.
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
        repetition_penalty=1.1,
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
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
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

    # Parse each completion once; scoring and the diagnostics below share it.
    parsed = [Response(r) for r in responses]
    scores = [
        p.satisfaction(ex["instruction_id_list"], ex["constraint_kwargs"])
        for p, ex in zip(parsed, dataset)
    ]
    # A completion that never emits EOS gets truncated, which fails most format
    # constraints -- track it, since it was the main failure mode on the base model.
    truncated = sum(
        1 for r in responses if len(tokenizer.encode(r)) >= MAX_COMPLETION_LENGTH - 5
    )
    # Repetition collapse is how a policy games a counting constraint. It scores 0
    # either way, but a rising count means GRPO is drifting toward reward hacking
    # rather than toward the task.
    degenerate = sum(1 for p in parsed if p.is_degenerate)
    mean = sum(scores) / len(scores)
    print(f"--> mean satisfaction: {mean:.4f}  (n={len(scores)})")
    print(f"--> truncated (no EOS): {truncated}/{len(responses)}")
    print(f"--> degenerate (looping): {degenerate}/{len(responses)}")
    # Per-example scores are kept so runs can be compared with a paired test --
    # the eval set is small enough that unpaired means are mostly noise.
    return {
        "mean": mean,
        "n": len(scores),
        "truncated": truncated,
        "degenerate": degenerate,
        "scores": scores,
    }


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
            if r.get("run") == "baseline":
                return r["after"]
    return None


######################
# load-or-train
######################
model, tokenizer = load_model_and_tokenizer(model_name=BASE_MODEL, use_gpu=True)
print(f"Policy initialized from {BASE_MODEL}")

config = {
    "base_model": BASE_MODEL,
    "num_generations": NUM_GENERATIONS,
    "num_train_epochs": NUM_EPOCHS,
    "beta": BETA,
}

if SKIP_TRAIN:
    # Baseline mode: score the untrained policy on the held-out split.
    before = eval_ifeval(model, tokenizer, eval_dataset, "Baseline (no GRPO)")
    record({"run": RUN, "config": config, "before": None, "after": before})
    raise SystemExit(0)

before = cached_baseline()
if before is None:
    before = eval_ifeval(model, tokenizer, eval_dataset, "Before GRPO")
else:
    print(f"\nUsing cached baseline: {before['mean']:.4f}")

if LOAD_CHECKPOINT and last_ckpt:
    # Checkpoint exists: load the GRPO-trained model, skip training.
    print(f"Loading GRPO model from {last_ckpt} (skip training)")
    model, tokenizer = load_model_and_tokenizer(model_name=last_ckpt, use_gpu=True)
else:
    # GRPO samples num_generations completions per prompt and normalizes reward
    # within that group, so the global batch (per_device x grad_accum) must be a
    # multiple of it. 8 x 4 = 32 works for both G=8 and G=16 while keeping only
    # 8 sequences in a forward pass, which is what an 8GB GPU can hold.
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,  # Higher than the usual 1e-6: the policy is only 135M.
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=NUM_GENERATIONS,  # Group size G. Larger -> less zero-variance groups.
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=1.0,  # High enough to create in-group variance.
        beta=BETA,  # KL penalty toward the reference policy.
        bf16=True,
        logging_steps=5,
        save_total_limit=1,
        report_to="none",
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=ifeval_reward,
        processing_class=tokenizer,
    )
    grpo_trainer.train()
    model = grpo_trainer.model

######################
# eval the trained model
######################
after = eval_ifeval(model, tokenizer, eval_dataset, f"After GRPO [{RUN}]")
record({"run": RUN, "config": config, "before": before, "after": after})
print(f"\n{RUN}: {before['mean']:.4f} -> {after['mean']:.4f}")

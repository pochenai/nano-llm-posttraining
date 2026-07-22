"""GRPO on the Countdown task (TinyZero-style) to elicit self-correction / aha moments.

Countdown is a *search* task -- reach a target from a few numbers using +,-,*,/, each
number once. Many attempts fail, so the natural chain is "try X -> wrong -> try Y",
which is the backtracking behaviour ("wait, that's not right, let me try...") RL is
meant to amplify. GSM8K rarely needs it; Countdown does, which is why TinyZero used it.

Same knobs as grpo_cot.py, overridable by env var so local debug -> cloud is one change:

    # local smoke: prove the loop runs and reward moves
    GRPO_COT_TRAIN_LIMIT=200 uv run python -m src.grpo_countdown

    # cloud (24GB): the real run
    GRPO_COT_MODEL=Qwen/Qwen2.5-3B-Instruct uv run python -m src.grpo_countdown

Ref: https://github.com/Jiayi-Pan/TinyZero
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
from transformers.trainer_callback import ProgressCallback, TrainerCallback

from src import LOAD_CHECKPOINT
from src.countdown_rewards import REWARD_FUNCS, Completion, load_countdown
from src.model_loader import load_model_and_tokenizer


class CompactLog(TrainerCallback):
    """Console prints only the metrics worth watching, one line per log. The full metric
    dict is still saved to log_history.json after training, so nothing is lost for later
    analysis -- this only quiets the terminal.
    """

    KEYS = [
        ("reward", "reward"),
        ("correct", "rewards/correctness_reward/mean"),  # solve signal -- should climb
        ("prox", "rewards/proximity_reward/mean"),  # search slope
        ("len", "completions/mean_length"),  # should now GROW (search), not shrink
        ("0std", "frac_reward_zero_std"),  # keep near 0 (else no gradient)
        ("kl", "kl"),
        ("H", "entropy"),
    ]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "reward" not in logs:  # skip eval / non-training logs
            return
        st = logs.get("step_time", 0.0)
        left = (state.max_steps - state.global_step) if state.max_steps > 0 else 0
        parts = [f"step {state.global_step}/{state.max_steps}"]
        parts += [f"{lab}={logs[k]:.3f}" for lab, k in self.KEYS if k in logs]
        if st:
            parts.append(f"eta={left * st / 3600:.1f}h")
        print("  ".join(parts))


# TinyZero found ~1.5B is the threshold where Countdown reasoning develops; 3B is
# clearest. Below 1.5B the policy rarely samples a valid search, so GRPO has nothing
# to amplify (every group zero-variance -> no gradient).
BASE_MODEL = os.environ.get("GRPO_COT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
# Countdown backtracking ("try X ... no ... try Y") needs room; TinyZero uses a 1024
# response length, so match it. Lower via env if VRAM is tight.
MAX_COMPLETION_LENGTH = int(os.environ.get("GRPO_COT_COMPLETION_LEN", 1024))

# Ablation knobs, overridable per run so variants stay comparable.
RUN = os.environ.get("GRPO_COT_RUN", "default")
NUM_GENERATIONS = int(os.environ.get("GRPO_COT_NUM_GENERATIONS", 16))
# Defaults are tuned for an 8GB card; a 24GB 4090 has room to raise both (LoRA keeps
# optimizer state tiny). per_device x grad_accum must stay a multiple of NUM_GENERATIONS.
BATCH = int(os.environ.get("GRPO_COT_BATCH", 2))
GRAD_ACCUM = int(os.environ.get("GRPO_COT_GRAD_ACCUM", 16))
NUM_EPOCHS = float(os.environ.get("GRPO_COT_EPOCHS", 1))
# TinyZero uses KL coef 0.001 -- low, so the policy is free to explore/diverge, which
# is what lets the backtracking behaviour emerge. A high KL (gsm8k used 0.02) pins the
# policy to the base and suppresses the aha moment.
BETA = float(os.environ.get("GRPO_COT_BETA", 0.001))
TRAIN_LIMIT = int(os.environ.get("GRPO_COT_TRAIN_LIMIT", 0))  # 0 = full split
EVAL_LIMIT = int(os.environ.get("GRPO_COT_EVAL_LIMIT", 100))
SKIP_TRAIN = os.environ.get("GRPO_COT_SKIP_TRAIN") == "1"
USE_LORA = os.environ.get("GRPO_COT_LORA", "1") == "1"
USE_VLLM = os.environ.get("GRPO_COT_VLLM", "1") == "1"

OUTPUT_DIR = f"trainer_output/grpo-countdown-{RUN}"
RESULTS_PATH = "trainer_output/grpo_countdown_runs.json"
last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

######################
# load data
######################
train_dataset = load_countdown("train", limit=TRAIN_LIMIT or None)
eval_dataset = load_countdown("test", limit=EVAL_LIMIT)
print(f"train={len(train_dataset)}  eval={len(eval_dataset)}")


def eval_countdown(model, tokenizer, dataset, title, batch_size=8):
    """Fraction of held-out puzzles solved -- a valid equation (each number once) that
    evaluates to the target. Same signal GRPO optimizes, so before/after is comparable.
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
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
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

    parsed = [Completion(r) for r in responses]
    correct = sum(
        c.is_correct(ex["target"], ex["nums"]) for c, ex in zip(parsed, dataset)
    )
    # Format adherence climbs first; it's the early sign GRPO is working while the
    # solve rate is still flat.
    formatted = sum(1 for c in parsed if c.equation is not None)
    acc = correct / len(responses)
    print(f"--> solved: {acc:.4f}  ({correct}/{len(responses)})")
    print(f"--> has <answer> equation: {formatted}/{len(responses)}")

    # Print one full completion so the actual search is visible, not just the score.
    print(f"\n  sample Q: {dataset[0]['prompt'][-1]['content']}")
    print(
        f"  model completion (target={dataset[0]['target']}, "
        f"nums={dataset[0]['nums']}):\n  {responses[0]}"
    )

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
    before = eval_countdown(model, tokenizer, eval_dataset, "Baseline (no GRPO)")
    record(
        {
            "run": f"baseline-{BASE_MODEL}",
            "config": config,
            "before": None,
            "after": before,
        }
    )
    raise SystemExit(0)

before = cached_baseline()
if before is None:
    before = eval_countdown(model, tokenizer, eval_dataset, "Before GRPO")
else:
    print(f"\nUsing cached baseline: {before['accuracy']:.4f}")

if LOAD_CHECKPOINT and last_ckpt:
    print(f"Loading GRPO model from {last_ckpt} (skip training)")
    model, tokenizer = load_model_and_tokenizer(model_name=last_ckpt, use_gpu=True)
else:
    peft_config = (
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        if USE_LORA
        else None
    )

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5 if USE_LORA else 1e-6,  # LoRA tolerates a higher LR.
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
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
    # Replace the default per-step metric-dict flood with a compact one-liner (the full
    # dict still goes to log_history.json). Drop the tqdm bar's dump; keep our line.
    grpo_trainer.remove_callback(ProgressCallback)
    grpo_trainer.add_callback(CompactLog())
    # What to watch in the logs:
    #   frac_reward_zero_std  THE health metric. Fraction of groups where every
    #                    completion scored the same -> advantage 0 -> no gradient. Near
    #                    1.0 means training is a no-op. On Countdown this is high early
    #                    (the base rarely finds a valid equation); the graded valideq /
    #                    answer / format tiers exist to keep some variance until it does.
    #   rewards/correctness_reward  the real objective (solve rate).
    #   rewards/valideq_reward      uses the right numbers; usually moves before solving.
    #   completions/clipped_ratio   high -> raise max_completion_length (search is long).
    grpo_trainer.train()
    grpo_trainer.save_model(OUTPUT_DIR)
    # Persist the full per-step training log so the steps-vs-score curve (TinyZero's
    # critic/score/mean plot) can be rebuilt later. Each entry carries step, reward,
    # rewards/<func>/mean (correctness ~ solve rate), kl, etc. Trainer also embeds this
    # in each checkpoint's trainer_state.json, but save_total_limit=1 prunes old
    # checkpoints -- this standalone dump survives.
    with open(os.path.join(OUTPUT_DIR, "log_history.json"), "w") as f:
        json.dump(grpo_trainer.state.log_history, f, indent=2)
    # Eval the just-trained model in memory instead of reloading it from disk: the
    # reload is redundant here and fragile -- a peft/transformers version skew in the
    # LoRA-adapter load path can crash *after* a multi-hour run, wasting all of it.
    model = grpo_trainer.model

######################
# eval the trained model
######################
after = eval_countdown(model, tokenizer, eval_dataset, f"After GRPO [{RUN}]")
record({"run": RUN, "config": config, "before": before, "after": after})
print(f"\n{RUN}: {before['accuracy']:.4f} -> {after['accuracy']:.4f}")

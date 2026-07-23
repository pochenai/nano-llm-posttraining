"""BASE vs GRPO side-by-side on a single Countdown puzzle.

Loads the trained Countdown checkpoint and the untrained base, generates both on the
same puzzle, and prints the two chains so you can eyeball what training changed -- in
particular whether a search / self-correction ("wait, that doesn't work, let me try...")
aha moment appeared that the base lacks.

    GRPO_COT_Q=0 uv run python -m src.grpo_countdown_checker            # test puzzle #0
    GRPO_COT_Q=3 GRPO_COT_SPLIT=train uv run python -m src.grpo_countdown_checker
"""

import os

import torch
from transformers import GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

from src.countdown_rewards import Completion, load_countdown
from src.model_loader import load_model_and_tokenizer

BASE_MODEL = os.environ.get("GRPO_COT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_COMPLETION_LENGTH = int(os.environ.get("GRPO_COT_COMPLETION_LEN", 1024))
RUN = os.environ.get("GRPO_COT_RUN", "v2")
# Small default limit: the checker only indexes a few puzzles, no need to map all 490k.
TRAIN_LIMIT = int(os.environ.get("GRPO_COT_TRAIN_LIMIT", 2000))
EVAL_LIMIT = int(os.environ.get("GRPO_COT_EVAL_LIMIT", 100))

OUTPUT_DIR = f"trainer_output/grpo-countdown-{RUN}"
last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

######################
# load data
######################
train_dataset = load_countdown("train", limit=TRAIN_LIMIT or None)
eval_dataset = load_countdown("test", limit=EVAL_LIMIT)
print(f"train={len(train_dataset)}  eval={len(eval_dataset)}")

######################
# load trained checkpoint + untrained base (shared tokenizer)
######################
print(f"Loading GRPO model from {last_ckpt} (skip training)")
assert (
    last_ckpt is not None
), f"no checkpoint under {OUTPUT_DIR} -- train grpo_countdown first"
model, tokenizer = load_model_and_tokenizer(model_name=last_ckpt, use_gpu=True)
base_model, _ = load_model_and_tokenizer(model_name=BASE_MODEL, use_gpu=True)


def _generate(m, ex):
    """Greedy completion of one puzzle by model `m` (deterministic -> comparable)."""
    prompt = tokenizer.apply_chat_template(
        ex["prompt"], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(
        m.device
    )
    gen_config = GenerationConfig(
        max_new_tokens=MAX_COMPLETION_LENGTH,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    m.eval()
    with torch.no_grad():
        out = m.generate(**inputs, generation_config=gen_config)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def answer_question(idx, dataset=None):
    """Print BASE vs GRPO chains for one Countdown puzzle, side by side.

    Same greedy decode for both, so the diff is the training effect -- eyeball the GRPO
    chain for a search / self-correction the base lacks. Defaults to the train split;
    pass eval_dataset for held-out.
    """
    dataset = train_dataset if dataset is None else dataset
    ex = dataset[idx]
    print(f"\n===== Q#{idx}  target={ex['target']}  nums={ex['nums']} =====")
    print("Q:", ex["prompt"][-1]["content"])
    for tag, m in [("BASE", base_model), ("GRPO", model)]:
        resp = _generate(m, ex)
        c = Completion(resp)
        correct = c.is_correct(ex["target"], ex["nums"])
        print(
            f"\n----- {tag}  eq={c.equation!r}  value={c.value}  correct={correct} -----"
        )
        print(resp)


if __name__ == "__main__":
    # GRPO_COT_Q = which puzzle index; GRPO_COT_SPLIT = train or test (default test).
    q_idx = int(os.environ.get("GRPO_COT_Q", 0))
    which = (
        train_dataset if os.environ.get("GRPO_COT_SPLIT") == "train" else eval_dataset
    )
    answer_question(q_idx, which)

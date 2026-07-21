"""KL-on-the-new-task measurement, following 'RL's Razor' (arXiv:2509.04259).

The paper's finding: catastrophic forgetting after fine-tuning is predicted by
the KL between the fine-tuned policy and the base model it started from, measured
on the NEW task's input distribution -- and online RL reaches a given task
performance at a smaller such KL than SFT, which is why it forgets less.

Direction (standard convention: forward KL samples from the *reference*):

    forward   D_KL(pi_0 || pi_theta)   =  E_{y ~ pi_0}   [ log pi_0 - log pi_theta ]
    reverse   D_KL(pi_theta || pi_0)   =  E_{y ~ pi_theta}[ log pi_theta - log pi_0 ]

pi_theta is the fine-tuned policy, pi_0 the base it was initialized from. We
default to FORWARD for the RL-vs-SFT study: it samples completions from the
shared base pi_0, so every fine-tune is scored on the *same* completion
distribution -- differences in KL then reflect how each model reallocates
probability, not different sampling distributions. (Reverse KL, sampled from each
model's own distribution, is the quantity the RL objective minimizes and matches
the trainer's logged `kl`, but scores each model on a different completion set.)

Per-position estimator: the EXACT categorical KL over the full vocabulary,

    kl(t) = sum_v P(v | prefix) * ( log P(v|prefix) - log Q(v|prefix) )

where P is the sampled-from model. This is the low-variance choice (no per-token
sampling noise, unlike the single-token k3 estimator used during training) --
appropriate here because this is offline analysis, not a training loss. A
single-token estimate is also reported as a sanity check.
"""

import json
import os

import torch

# TRL's memory-efficient `log_softmax -> gather`, used for the single-token
# sanity estimate. The exact KL below needs the full distribution, not a gather.
from trl.trainer.utils import selective_log_softmax

from src.ifeval_rewards import load_ifeval
from src.model_loader import load_model_and_tokenizer

SYSTEM_MESSAGE = "You are a helpful assistant."
MAX_COMPLETION_LENGTH = 200

MODEL = os.environ.get("KL_MODEL", "trainer_output/grpo-H-beta002/checkpoint-474")
REF = os.environ.get("KL_REF", "HuggingFaceTB/SmolLM2-135M-Instruct")
LABEL = os.environ.get("KL_LABEL", "grpo-H")
DIRECTION = os.environ.get("KL_DIRECTION", "forward")  # forward | reverse
EVAL_LIMIT = int(os.environ.get("KL_EVAL_LIMIT", 0))  # 0 = full held-out split
NUM_SAMPLES = int(os.environ.get("KL_SAMPLES", 4))  # completions per prompt
TEMPERATURE = float(os.environ.get("KL_TEMPERATURE", 1.0))
BATCH_SIZE = int(os.environ.get("KL_BATCH", 4))  # small: full-vocab KL is heavy
RESULTS_PATH = "trainer_output/kl_runs.json"


@torch.no_grad()
def measure_kl(policy, ref, tokenizer, dataset, direction="forward"):
    """KL of the fine-tuned `policy` against `ref` on the dataset's prompts.

    For forward KL we sample from `ref` and weight the per-position sum by `ref`;
    for reverse we sample from `policy`. In both cases the model we sample from is
    P (the first KL argument), so `p_model` is both the sampler and the weighting.
    """
    policy.eval()
    ref.eval()
    pad_id = tokenizer.pad_token_id
    p_model, q_model = (ref, policy) if direction == "forward" else (policy, ref)

    prompts = [
        tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True
        )
        for ex in dataset
    ]
    prompts = [p for p in prompts for _ in range(NUM_SAMPLES)]

    tot_exact = tot_sampled = 0.0
    tot_seq_exact = 0.0
    n_tokens = n_seqs = 0

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # right padding corrupts batched generation
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            ).to(p_model.device)
            prompt_len = inputs["input_ids"].shape[1]

            seq = p_model.generate(
                **inputs,
                max_new_tokens=MAX_COMPLETION_LENGTH,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=1.0,
                pad_token_id=pad_id,
            )
            attn = (seq != pad_id).long()

            # Completion target tokens and the logits that predict them. Slicing
            # to the completion region keeps the full-vocab tensors small (only
            # ~200 positions, not the whole prompt+completion).
            ctargets = seq[:, prompt_len:]  # (B, Ncomp)
            cvalid = (ctargets != pad_id).float()
            sl = slice(prompt_len - 1, -1)  # logits[:, t] predicts seq[:, t+1]
            logits_p = p_model(input_ids=seq, attention_mask=attn).logits[:, sl, :].float()
            logits_q = q_model(input_ids=seq, attention_mask=attn).logits[:, sl, :].float()

            logP = torch.log_softmax(logits_p, dim=-1)
            logQ = torch.log_softmax(logits_q, dim=-1)
            # Exact per-position KL: sum_v P(v) (log P(v) - log Q(v)).
            kl_pos = (logP.exp() * (logP - logQ)).sum(dim=-1)  # (B, Ncomp)

            # Single-token estimate at the sampled token, same direction.
            tokP = selective_log_softmax(logits_p, ctargets)
            tokQ = selective_log_softmax(logits_q, ctargets)
            sampled_pos = tokP - tokQ  # log P(y_t) - log Q(y_t)

            tot_exact += (kl_pos * cvalid).sum().item()
            tot_sampled += (sampled_pos * cvalid).sum().item()
            n_tokens += cvalid.sum().item()
            tot_seq_exact += (kl_pos * cvalid).sum(dim=1).sum().item()
            n_seqs += seq.shape[0]
    finally:
        tokenizer.padding_side = original_side

    return {
        "direction": direction,
        "kl_per_token": tot_exact / n_tokens,  # headline: exact, full-vocab
        "kl_per_token_sampled": tot_sampled / n_tokens,  # sanity: single-token
        "kl_per_sequence": tot_seq_exact / n_seqs,
        "avg_completion_len": n_tokens / n_seqs,
        "n_sequences": n_seqs,
    }


def record(entry):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    runs = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            runs = json.load(f)
    key = (entry["label"], entry["result"]["direction"])
    runs = [r for r in runs if (r.get("label"), r.get("result", {}).get("direction")) != key]
    runs.append(entry)
    with open(RESULTS_PATH, "w") as f:
        json.dump(runs, f, indent=2)
    print(f"recorded -> {RESULTS_PATH}")


# uv run python -m src.kl_analysis
#   KL_MODEL=trainer_output/sft-instruct/checkpoint-63 KL_LABEL=sft-instruct \
#       uv run python -m src.kl_analysis
if __name__ == "__main__":
    torch.manual_seed(0)

    dataset = load_ifeval(max_words=100, system_message=SYSTEM_MESSAGE)
    eval_dataset = dataset.train_test_split(test_size=0.2, seed=42)["test"]
    if EVAL_LIMIT:
        eval_dataset = eval_dataset.select(range(EVAL_LIMIT))

    print(f"Policy: {MODEL}\nRef:    {REF}\ndirection={DIRECTION}  "
          f"prompts={len(eval_dataset)} x {NUM_SAMPLES} samples")

    policy, tokenizer = load_model_and_tokenizer(model_name=MODEL, use_gpu=True)
    ref, _ = load_model_and_tokenizer(model_name=REF, use_gpu=True)

    result = measure_kl(policy, ref, tokenizer, eval_dataset, direction=DIRECTION)
    sampler = "pi_0 (base)" if DIRECTION == "forward" else "pi_theta (policy)"
    print(f"\n=== {DIRECTION} KL on new task [{LABEL}]  (sampled from {sampler}) ===")
    print(f"  KL / token   (exact)   : {result['kl_per_token']:.4f}   <- headline")
    print(f"  KL / token   (sampled) : {result['kl_per_token_sampled']:.4f}   (sanity)")
    print(f"  KL / sequence (exact)  : {result['kl_per_sequence']:.4f}")
    print(f"  avg completion len     : {result['avg_completion_len']:.1f}")

    record({"label": LABEL, "model": MODEL, "ref": REF, "result": result})

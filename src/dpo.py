import os

from trl import DPOTrainer, DPOConfig  # pyright: ignore[reportPrivateImportUsage]
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

from src.model_loader import load_model_and_tokenizer, test_model_with_questions

# NOTE on model choice: DPO only *nudges* an already chat-capable model, it cannot
# teach a base model to converse. The base SmolLM2-135M has no identity to shift,
# so we start from the *Instruct* checkpoint (the "SFT first" step, already done
# for us by HuggingFace). You could instead point this at your own SFT output_dir.
# Stage 2 of the identity pipeline: start from the identity-SFT model, which now
# reliably says "I am Qwen". Only because "Qwen" is in its output distribution can
# DPO actually flip it to "Deep Qwen". Falls back to the Instruct model if the SFT
# stage hasn't been run yet.
SFT_DIR = "trainer_output/identity_sft"
sft_ckpt = get_last_checkpoint(SFT_DIR) if os.path.isdir(SFT_DIR) else None
BASE_MODEL = sft_ckpt or "HuggingFaceTB/SmolLM2-135M-Instruct"
OUTPUT_DIR = "trainer_output/dpo"
last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

# Identity probes: what we watch to see the DPO shift take effect.
questions = [
    "What is your name?",
    "Are you ChatGPT?",
    "Tell me about your name and organization.",
]
SYSTEM_MESSAGE = "You're a helpful assistant."

######################
# eval before DPO
######################
model, tokenizer = load_model_and_tokenizer(BASE_MODEL, use_gpu=True)
test_model_with_questions(
    model, tokenizer, questions, system_message=SYSTEM_MESSAGE, title="Before DPO"
)

######################
# load-or-train
######################
if last_ckpt:
    print(f"Loading DPO model from {last_ckpt} (skip training)")
    model, tokenizer = load_model_and_tokenizer(last_ckpt, use_gpu=True)
else:
    # Ready-made 1k preference pairs. Each pair's chosen/rejected differ mainly in
    # the self-identity ("Deep Qwen" vs "Qwen") -- a clean, single-direction signal.
    dpo_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")

    # ~58% of the pairs have chosen == rejected (the dataset was built with a
    # name-replace trick that did nothing on non-identity prompts). Those give zero
    # gradient, so drop them to keep every step informative.
    dpo_ds = dpo_ds.filter(
        lambda r: r["chosen"][-1]["content"] != r["rejected"][-1]["content"]
    )

    config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.2,  # KL strength: higher = stay closer to the reference model.
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 4 x 4 = 16.
        num_train_epochs=1,
        # DPOConfig defaults to 1e-6. 5e-5 (what the notebook used on a tiny CPU
        # subset) collapses a 135M model into gibberish over a full epoch, because
        # DPO drags *both* chosen and rejected logprobs down when pushed too hard.
        learning_rate=5e-6,
        logging_steps=10,
        bf16=True,
        save_total_limit=1,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # None -> a frozen copy of `model` is used as the reference.
        args=config,
        processing_class=tokenizer,
        train_dataset=dpo_ds,
    )

    """
        What to watch in the training logs:
        rewards/margins    (chosen - rejected) should rise steadily  -> it's learning
        rewards/accuracies fraction with chosen > rejected -> ~1.0    -> it's learning
        rewards/chosen     MUST stay positive / not fall              -> it's NOT collapsing
        The trap: margins can keep rising while rewards/chosen goes negative -- that means
        both chosen and rejected logprobs are being dragged down (degeneration -> gibberish),
        not real preference learning. That's exactly what a too-high learning rate looks like.
    """
    dpo_trainer.train()
    dpo_trainer.save_model(OUTPUT_DIR)  # final model + tokenizer at the top dir.
    # Reload the saved weights for eval. The in-memory model right after training
    # can emit degenerate text (a post-training state artifact), while the saved
    # weights generate correctly -- so evaluate what actually got saved.
    model, tokenizer = load_model_and_tokenizer(OUTPUT_DIR, use_gpu=True)

######################
# eval after DPO
######################
test_model_with_questions(
    model, tokenizer, questions, system_message=SYSTEM_MESSAGE, title="After DPO"
)

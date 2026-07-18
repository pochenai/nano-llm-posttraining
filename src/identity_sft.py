import os

from trl import SFTTrainer, SFTConfig  # pyright: ignore[reportPrivateImportUsage]
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

from src.model_loader import load_model_and_tokenizer, test_model_with_questions

# Stage 1 of the identity pipeline: SFT the Instruct model until it reliably says
# "I am Qwen ...". Only then is "Qwen" in the model's output distribution, so the
# later DPO step (Qwen -> Deep Qwen) has something real to flip. Its output_dir is
# what dpo.py loads as its BASE_MODEL.
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUTPUT_DIR = "trainer_output/identity_sft"
last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

# Same probes as dpo.py so the before/after comparison is apples-to-apples:
# here we want them to become "I am Qwen"; after DPO they should become "Deep Qwen".
questions = [
    "What is your name?",
    "Are you ChatGPT?",
    "Tell me about your name and organization.",
]
SYSTEM_MESSAGE = "You're a helpful assistant."

######################
# build SFT data (the "Qwen" answers = the rejected side of the DPO pairs)
######################
dpo_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")
# Keep only the identity pairs (chosen != rejected); their `rejected` conversation
# is a full [system, user, "I am Qwen ..."] chat -- exactly the SFT target we want.
id_ds = dpo_ds.filter(
    lambda r: r["chosen"][-1]["content"] != r["rejected"][-1]["content"]
)
# Prompt-completion format (not a single "messages" list): this makes SFTTrainer
# compute loss ONLY on the assistant answer and mask the prompt. Without it, the
# identical system prompt repeated across all 424 rows lets a 135M model overfit
# the template scaffolding and collapse into "systemsystem..." at generation time.
sft_ds = id_ds.map(
    lambda r: {"prompt": r["rejected"][:-1], "completion": r["rejected"][-1:]},
    remove_columns=id_ds.column_names,
)

######################
# eval before SFT
######################
model, tokenizer = load_model_and_tokenizer(BASE_MODEL, use_gpu=True)
test_model_with_questions(
    model, tokenizer, questions, system_message=SYSTEM_MESSAGE, title="Before SFT"
)

######################
# load-or-train
######################
if last_ckpt:
    print(f"Loading SFT model from {last_ckpt} (skip training)")
    model, tokenizer = load_model_and_tokenizer(last_ckpt, use_gpu=True)
else:
    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        num_train_epochs=3,  # small identity set -> a few passes to make it stick.
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # effective batch = 8 x 2 = 16.
        completion_only_loss=True,  # loss on the assistant answer only (see data above).
        bf16=True,
        logging_steps=10,
        save_total_limit=1,
    )
    sft_trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=sft_ds,
        processing_class=tokenizer,
    )
    sft_trainer.train()
    sft_trainer.save_model(OUTPUT_DIR)  # final model + tokenizer at the top dir.
    # Reload the saved weights for eval. The in-memory model right after training
    # can emit degenerate text (a post-training state artifact), while the saved
    # weights generate correctly -- so evaluate what actually got saved.
    model, tokenizer = load_model_and_tokenizer(OUTPUT_DIR, use_gpu=True)

######################
# eval after SFT (should now self-identify as "Qwen")
######################
test_model_with_questions(
    model, tokenizer, questions, system_message=SYSTEM_MESSAGE, title="After SFT"
)

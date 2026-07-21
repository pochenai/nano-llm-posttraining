import os

# trl re-exports these but doesn't list them in __all__, which trips pyright's
# reportPrivateImportUsage; the runtime import is the documented public API.
from trl import SFTTrainer, SFTConfig  # pyright: ignore[reportPrivateImportUsage]

from src import LOAD_CHECKPOINT
from src.data_loader import load_sftdata, questions, outputs
from src.model_loader import load_model_and_tokenizer, test_model_with_questions
from transformers.trainer_utils import get_last_checkpoint

# Base model and output dir are env-overridable so the same script produces both
# the base-model SFT and the Instruct-start SFT used for the RL-vs-SFT KL study
# (RL's Razor needs both fine-tunes to share a starting checkpoint). Defaults
# keep the original behavior: base SmolLM2-135M -> trainer_output/sft.
SFT_MODEL = os.environ.get("SFT_MODEL", "HuggingFaceTB/SmolLM2-135M")
OUTPUT_DIR = os.environ.get("SFT_OUTPUT_DIR", "trainer_output/sft")
last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

######################
# load data
######################
train_dataset = load_sftdata(split="train").select(range(2000))
test_dataset = load_sftdata(split="train").select(range(3))
test_questions = questions(test_dataset)
test_answers = outputs(test_dataset)

SYSTEM_MESSAGE = "You are a helpful assistant."

######################
# load-or-train
######################
# load the base model, eval before SFT, then train or load the trained model
model, tokenizer = load_model_and_tokenizer(model_name=SFT_MODEL, use_gpu=True)
print(f"SFT base: {SFT_MODEL} -> {OUTPUT_DIR}")

test_model_with_questions(
    model,
    tokenizer,
    questions=test_questions,
    answers=test_answers,
    system_message=SYSTEM_MESSAGE,
    title="Before SFT",
)

if LOAD_CHECKPOINT and last_ckpt:
    # Checkpoint exists: load the latest fine-tuned model, skip training.
    print(f"Loading fine-tuned model from {last_ckpt} (skip training)")
    model, tokenizer = load_model_and_tokenizer(model_name=last_ckpt, use_gpu=True)
else:
    # SFTTrainer config
    # batch size (effective) = per_device_train_batch_size × gradient_accumulation_steps (~ 8GB memory for 135M model with the following config)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,  # Where checkpoints are written (matches get_last_checkpoint above).
        learning_rate=3e-4,  # Learning rate for training.
        num_train_epochs=1,  #  Set the number of epochs to train the model.
        per_device_train_batch_size=8,  # Batch size per device. Small to fit an 8GB GPU.
        gradient_accumulation_steps=4,  # Accumulate to an effective batch size of 8 x 4 = 32.
        gradient_checkpointing=False,  # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.
        bf16=True,  # Mixed-precision training: cuts activation memory and is faster on RTX 50-series.
        logging_steps=10,  # Frequency of logging training progress (log every 10 steps).
        save_total_limit=1,
    )

    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    sft_trainer.train()
    model = sft_trainer.model

######################
# eval the fine-tuned model
######################
test_model_with_questions(
    model,
    tokenizer,
    questions=test_questions,
    answers=test_answers,
    system_message=SYSTEM_MESSAGE,
    title="After SFT",
)

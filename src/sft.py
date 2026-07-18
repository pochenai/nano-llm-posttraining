import os

from trl import SFTTrainer, SFTConfig

from src.data_loader import load_sftdata, questions, outputs
from src.model_loader import load_model_and_tokenizer, test_model_with_questions

# If this checkpoint dir exists, load it and skip training; otherwise train from
# the base model (checkpoints are auto-saved back into trainer_output/).
CKPT_DIR = "trainer_output/checkpoint-7"

######################
# load data
######################
train_dataset = load_sftdata(split="train").select(range(200))
test_dataset = load_sftdata(split="train").select(range(3))
test_questions = questions(test_dataset)
test_answers = outputs(test_dataset)

SYSTEM_MESSAGE = "You are a helpful assistant."

######################
# load-or-train
######################
# load the base model, eval before SFT, then train or load the trained model
model, tokenizer = load_model_and_tokenizer(use_gpu=True)

test_model_with_questions(
    model,
    tokenizer,
    questions=test_questions,
    answers=test_answers,
    system_message=SYSTEM_MESSAGE,
    title="Before SFT",
)

if os.path.isdir(CKPT_DIR):
    # Checkpoint exists: load the fine-tuned model, skip training.
    print(f"Loading fine-tuned model from {CKPT_DIR} (skip training)")
    model, tokenizer = load_model_and_tokenizer(model_name=CKPT_DIR, use_gpu=True)
else:
    # SFTTrainer config
    # batch size (effective) = per_device_train_batch_size × gradient_accumulation_steps (~ 8GB memory for 135M model with the following config)
    sft_config = SFTConfig(
        learning_rate=3e-4,  # Learning rate for training.
        num_train_epochs=1,  #  Set the number of epochs to train the model.
        per_device_train_batch_size=8,  # Batch size per device. Small to fit an 8GB GPU.
        gradient_accumulation_steps=4,  # Accumulate to an effective batch size of 8 x 4 = 32.
        gradient_checkpointing=False,  # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.
        bf16=True,  # Mixed-precision training: cuts activation memory and is faster on RTX 50-series.
        logging_steps=10,  # Frequency of logging training progress (log every 10 steps).
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

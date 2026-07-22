import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from . import dprint


def load_model_and_tokenizer(model_name="HuggingFaceTB/SmolLM2-135M", use_gpu=True):

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_gpu:
        model.to("cuda")  # pyright: ignore[reportArgumentType]

    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""

    # Tokenizer config
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Sanity check: the chat template ends assistant turns with the literal
    # string "<|endoftext|>". generate() stops on tokenizer.eos_token_id, so that
    # string MUST encode to exactly that single id -- otherwise the stop token the
    # model is trained to emit is not the one generation halts on.
    eot_ids = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    dprint(
        f"eos_token={tokenizer.eos_token!r} eos_token_id={tokenizer.eos_token_id} "
        f"| '<|endoftext|>' -> {eot_ids}"
    )
    if eot_ids != [tokenizer.eos_token_id]:
        print(
            f"[WARN] template stop token '<|endoftext|>' encodes to {eot_ids}, "
            f"not [eos_token_id={tokenizer.eos_token_id}]; generation may not stop "
            f"where the model was trained to."
        )

    # Align the model's generation config with the tokenizer's special tokens now,
    # so the Trainer doesn't do this itself later and print a "tokenizer has new
    # PAD/BOS/EOS tokens ... updated" notice. Qwen2.5, for example, uses
    # pad=<|endoftext|> and no BOS, which differ from the model's stored config.
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id

    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    user_message,
    system_message=None,
    max_new_tokens=100,
    repetition_penalty=1.1,  # >1.0 penalizes repeating tokens; 1.0 = no penalty.
    return_stats=False,
):
    # Format chat using tokenizer's chat template
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    # We assume the data are all single-turn conversation
    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    dprint("Prompt:\n", repr(prompt))  # inspect the rendered chat-template prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # All decoding options live in one GenerationConfig so it's clear at a glance
    # which knobs are set (and the IDE can autocomplete/type-check the fields).
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy: deterministic, reproducible for benchmarking/eval.
        repetition_penalty=repetition_penalty,  # >1.0 penalizes repeats; 1.0 = off.
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Time only the generate() call. CUDA is async, so sync before/after to get
    # the real wall time on GPU (otherwise the timer stops before kernels finish).
    is_cuda = model.device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    # Recommended to use vllm, sglang or TensorRT
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if return_stats:
        num_new_tokens = generated_ids.shape[0]
        stats = {
            "num_new_tokens": num_new_tokens,
            "elapsed_s": elapsed,
            "tokens_per_s": num_new_tokens / elapsed if elapsed > 0 else 0.0,
        }
        return response, stats

    return response


def test_model_with_questions(
    model, tokenizer, questions, answers=None, system_message=None, title="Model Output"
):
    print(f"\n=== {title} ===")
    for (
        i,
        question,
    ) in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")
        # Optional ground-truth answer from the training data, for side-by-side compare.
        if answers is not None:
            print(f"Reference {i}:\n{answers[i - 1]}\n")


# uv run python -m src.model_loader
if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM2-135M"
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_name, use_gpu=True)

    user_message = "who are you"
    system_message = "You are a helpful assistant."

    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")

    # Warmup: the first generate() triggers CUDA kernel compilation / cudnn
    # autotune / lazy allocations, so it is much slower. Discard it.
    print("Warmup ...")
    response = generate_responses(model, tokenizer, user_message, system_message)
    print("Resp\n:", response)

    # Measure: average tokens/s over a few runs (loading time already excluded,
    # timing happens only around generate() inside generate_responses).
    n_runs = 3
    total_tokens, total_time = 0, 0.0
    for i in range(n_runs):
        response, stats = generate_responses(
            model, tokenizer, user_message, system_message, return_stats=True
        )
        total_tokens += stats["num_new_tokens"]
        total_time += stats["elapsed_s"]
        print(
            f"[run {i}] {stats['num_new_tokens']} tok in "
            f"{stats['elapsed_s']:.3f}s -> {stats['tokens_per_s']:.1f} tok/s"
        )

    print(f"\nAvg throughput: {total_tokens / total_time:.1f} tok/s")

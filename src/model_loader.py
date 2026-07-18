import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from . import dprint


def load_model_and_tokenizer(model_name, use_gpu=False):

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_gpu:
        model.to("cuda")

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

    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    user_message,
    system_message=None,
    max_new_tokens=100,
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

    # Time only the generate() call. CUDA is async, so sync before/after to get
    # the real wall time on GPU (otherwise the timer stops before kernels finish).
    is_cuda = model.device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    # Recommended to use vllm, sglang or TensorRT
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
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


# uv run python -m src.model_loader
if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM2-135M"
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

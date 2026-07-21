"""Measure GRPO rollout throughput on this GPU: batched generation is ~80% of
GRPO's wall-clock, so it sets the per-step cost."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
G = 8          # num_generations (rollouts per prompt)
NEW_TOK = 200  # max_completion_length

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
model.eval()

q = ("Natalia sold clips to 48 friends in April, and then she sold half as many "
     "clips in May. How many clips did Natalia sell altogether?")
msgs = [{"role": "user", "content": q}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# G identical prompts = one GRPO group
batch = tok([prompt] * G, return_tensors="pt", padding=True).to("cuda")

def run():
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=NEW_TOK, do_sample=True,
                             temperature=0.8, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new = (out.shape[1] - batch["input_ids"].shape[1]) * out.shape[0]
    return dt, new

run()  # warmup
times = []
for _ in range(3):
    dt, new = run()
    times.append(dt)
    print(f"  group of {G}: {dt:.2f}s, {new} tokens -> {new/dt:.0f} tok/s aggregate")

avg = sum(times) / len(times)
print(f"\nAvg per GRPO group (={G} rollouts): {avg:.2f}s")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"\nGSM8K train = 7473 prompts")
print(f"  gen-only time for 1 epoch: {7473*avg/3600:.1f} h  (excl. fwd/bwd, ~+30-50%)")
for n in (50, 200, 500):
    print(f"  {n} steps: {n*avg/60:.0f} min gen-only")

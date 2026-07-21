"""Validate hand-written IFEval answers against the verifiable checker.

   uv run python scripts_validate_answers.py <answers.json>

answers.json maps a prompt index (string) to an answer string. Prints the
satisfaction score per index and a final PASS/total summary. An answer counts
as usable only at score == 1.0 (all constraints satisfied).
"""

import json
import sys

from src.ifeval_rewards import load_ifeval, Response

ds = load_ifeval(max_words=100, system_message="You are a helpful assistant.")
train = ds.train_test_split(test_size=0.2, seed=42)["train"]

answers = json.load(open(sys.argv[1]))
n_pass = 0
for k, ans in answers.items():
    i = int(k)
    ex = train[i]
    s = Response(ans).satisfaction(ex["instruction_id_list"], ex["constraint_kwargs"])
    ok = s >= 1.0
    n_pass += ok
    tag = "OK  " if ok else "FAIL"
    extra = "" if ok else f"  score={s:.2f} constraints={ex['instruction_id_list']}"
    print(f"{tag} i={i}{extra}")
print(f"\nPASS {n_pass}/{len(answers)}")

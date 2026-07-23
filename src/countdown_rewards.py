"""Rule-based rewards for the Countdown task (TinyZero-style) GRPO.

Countdown: given a few numbers and a target, write an arithmetic expression that uses
each number exactly once and evaluates to the target (e.g. nums=[3,4,6], target=24 ->
"6 * (3 + 4 - 3)"... no: "4 * 6 * ...). The task is a *search*: many attempts fail, so
the natural solution process is "try X -> wrong -> try Y", which is exactly the
backtracking / self-correction ("aha moment") RL is supposed to elicit. Correctness is
a pure rule check -- parse the expression, verify the numbers used, evaluate it -- so no
reward model is needed.

Ref: https://github.com/Jiayi-Pan/TinyZero  (Jiayi-Pan/Countdown-Tasks-3to4)
"""

import re
from functools import cached_property

from datasets import load_dataset

# TinyZero format: reasoning in <think>, the final equation in <answer>. Tags make the
# equation trivially extractable, which is what lets correctness be scored by rule.
SYSTEM_MESSAGE = (
    "You are a helpful assistant good at math puzzles. You first think through the "
    "reasoning step by step, then give the final answer."
)
USER_TEMPLATE = (
    "Using the numbers {nums}, create an equation that equals {target}. You can use the "
    "operations +, -, *, / and parentheses, and each number exactly once. Show your "
    "reasoning in <think> </think> tags, then give the final equation in <answer> "
    "</answer> tags, for example <answer> (1 + 2) * 3 </answer>."
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_STRICT_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)
_SOFT_RE = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", re.DOTALL)
_INT_RE = re.compile(r"\d+")
# Only digits, the four operators, parentheses, dot and spaces may reach eval().
_SAFE_RE = re.compile(r"^[\d+\-*/() .]+$")


def evaluate_equation(expr):
    """Value of an arithmetic expression, or None on illegal chars / any error."""
    if not expr or not _SAFE_RE.match(expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - whitelisted chars only
    except Exception:
        return None


class Completion:
    """One model completion, parsed once (mirrors gsm8k_rewards.Completion)."""

    TAGS = ("<think>", "</think>", "<answer>", "</answer>")

    def __init__(self, completion):
        self.text = (
            completion[0]["content"] if isinstance(completion, list) else completion
        ) or ""

    def __repr__(self):
        return f"Completion({self.text[:40]!r}...)"

    @cached_property
    def equation(self):
        """The expression inside <answer>, or None when the tag is missing.

        Models very often write "expr = target" (e.g. "77 - 73 + 6 - 20 = -10"); keep
        only the left side. Otherwise the trailing "= <target>" breaks eval (illegal
        '=') AND makes the target count as an extra number -- so a genuine attempt gets
        scored 0 on both correctness and proximity, starving the search signal.
        """
        m = _ANSWER_RE.search(self.text)
        if not m:
            return None
        eq = m.group(1).strip()
        if "=" in eq:
            eq = eq.split("=")[0].strip()
        return eq or None

    @cached_property
    def value(self):
        return evaluate_equation(self.equation)

    @cached_property
    def tag_hits(self):
        """How many of the four tags appear exactly once."""
        return sum(self.text.count(t) == 1 for t in self.TAGS)

    @cached_property
    def is_soft_format(self):
        return bool(_SOFT_RE.search(self.text))

    @cached_property
    def is_strict_format(self):
        return bool(_STRICT_RE.match(self.text))

    def uses_exact_numbers(self, nums):
        """The equation uses each provided number exactly once (multiset match)."""
        if self.equation is None:
            return False
        used = sorted(int(n) for n in _INT_RE.findall(self.equation))
        return used == sorted(int(n) for n in nums)

    def is_correct(self, target, nums):
        """Valid equation (each number once) that evaluates to the target."""
        if not self.uses_exact_numbers(nums):
            return False
        v = self.value
        return v is not None and abs(v - float(target)) < 1e-4


######################
# reward functions
######################
# Correctness must DOMINATE. An earlier graded design (flat credit for "used the
# numbers" / "has an expression" / format tiers) was farmable: the policy learned to
# emit a well-formed, right-numbers, wrong-value expression -- collecting ~1.5 easy
# points without ever searching for the target, and completions SHORTENED (the opposite
# of the aha we want). With G large enough that frac_reward_zero_std=0, those crutches
# are unnecessary and harmful. So: correctness (exact hit) dominates; proximity gives a
# search *slope* that can't be farmed by a random valid expression (only getting CLOSE
# counts, and getting close needs search); format stays tiny, just to hold structure.
CORRECTNESS_WEIGHT = 2.0  # each number once AND evaluates to target
PROXIMITY_WEIGHT = 0.5  # graded by how near the value lands to the target
FORMAT_WEIGHT = 0.2  # small: keep the <think>/<answer> structure, not farmable


def correctness_reward(completions, target, nums, **_) -> list[float]:
    """The real objective: a valid equation that hits the target."""
    return [
        CORRECTNESS_WEIGHT if Completion(c).is_correct(t, n) else 0.0
        for c, t, n in zip(completions, target, nums)
    ]


def proximity_reward(completions, target, nums, **_) -> list[float]:
    """Graded credit for using the exact numbers to land NEAR the target.

    Unlike a flat 'used the numbers' reward, this can't be farmed by a random valid
    expression -- only closeness counts, and getting close needs actual search. Requires
    the exact numbers (so it can't be farmed by just writing the target), then scales
    with 1 - |value - target| / |target|. correctness (exact hit) is the 2.0 on top.
    """
    out = []
    for c, t, n in zip(completions, target, nums):
        comp = Completion(c)
        if comp.uses_exact_numbers(n) and comp.value is not None:
            rel = abs(comp.value - float(t)) / max(1.0, abs(float(t)))
            out.append(PROXIMITY_WEIGHT * max(0.0, 1.0 - rel))
        else:
            out.append(0.0)
    return out


def format_reward(completions, **_) -> list[float]:
    """Small credit for the <think>/<answer> structure -- hold the format the policy
    already learned without letting it farm a big plateau."""
    return [FORMAT_WEIGHT if Completion(c).is_soft_format else 0.0 for c in completions]


REWARD_FUNCS = [correctness_reward, proximity_reward, format_reward]


######################
# dataset
######################
def load_countdown(
    split="train", system_message=SYSTEM_MESSAGE, limit=None, test_size=500, seed=42
):
    """Load Countdown-Tasks-3to4 into the conversational format GRPOTrainer expects.

    The HF dataset ships a single split, so carve a fixed held-out `test` off it. The
    `target` and `nums` columns are carried through; TRL passes unknown columns to the
    reward functions as keyword arguments.
    """
    full = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    parts = full.train_test_split(test_size=test_size, seed=seed)
    ds = parts["test"] if split == "test" else parts["train"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    def to_conversational(ex):
        nums = list(ex["nums"])
        target = ex["target"]
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append(
            {"role": "user", "content": USER_TEMPLATE.format(nums=nums, target=target)}
        )
        return {
            "prompt": messages,
            "target": target,
            "nums": nums,
            "reference": f"reach {target} using {nums}",
        }

    return ds.map(to_conversational, remove_columns=ds.column_names)


# uv run python -m src.countdown_rewards
if __name__ == "__main__":
    nums, target = [3, 4, 6], 24
    correct = "<think>4 * 6 = 24, then use 3: 4 * 6 + 3 - 3? only three numbers. 4*(6+3-3)? no.\n(6 - 3) * ... hmm. 4 * 6 = 24 uses 4,6; need 3 too. (4*6)* (3/3)? one 3. Try 6*4*(3-3)+24? no.\nActually 6 * (3 + 4 - 3)? that reuses 3. Let me try (6 - 4 + 3)*... no. 6*4=24, and 3 must cancel: not possible with one 3.\nRe-read: numbers [3,4,6], target 24 -> 4 * (6 + 3 - 3) invalid. 6 * 4 * 3 / 3 invalid (two 3). So (3 + 6) * ... 9*? no. 3 * (4 + 6 - ...)? 3*8=24 -> 4+6-? need 8 from 4 and 6: 4+6=10 no. Hmm.</think><answer>6 * (3 + 4 - 3)</answer>"
    bad_nums = "<think>...</think><answer>3 * 4 * 6</answer>"  # uses right nums, =72
    for label, text in [("uses3,4,6->wrong", correct), ("3*4*6=72", bad_nums)]:
        c = Completion(text)
        print(
            f"{label:18s} eq={c.equation!r} value={c.value} "
            f"uses_exact={c.uses_exact_numbers(nums)} correct={c.is_correct(target, nums)} "
            f"tags={c.tag_hits}"
        )
    # A genuinely correct one for [2, 3, 4] -> 24: 2 * 3 * 4
    ok = "<think>2*3=6, 6*4=24</think><answer>2 * 3 * 4</answer>"
    c = Completion(ok)
    assert c.is_correct(24, [2, 3, 4]) and c.tag_hits == 4
    print("ok: correct example scores is_correct=True")

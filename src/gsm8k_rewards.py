"""Rule-based rewards for GSM8K chain-of-thought GRPO.

No reward model needed: GSM8K ships a gold numeric answer, so correctness is
checkable with a regex. That keeps a second model off the GPU, which is what
makes this fit on a small card.
"""

import re
from functools import cached_property

from datasets import load_dataset

# The format the policy is asked to produce. Tags make the final answer trivially
# extractable, which is what lets correctness be scored by rule instead of a model.
SYSTEM_MESSAGE = (
    "You are a helpful assistant that solves math problems step by step.\n"
    "Respond in exactly this format:\n"
    "<reasoning>\n"
    "Work through the problem step by step.\n"
    "</reasoning>\n"
    "<answer>\n"
    "The final numeric answer only.\n"
    "</answer>"
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Strict: both tags, in order, nothing outside them.
_STRICT_RE = re.compile(
    r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$", re.DOTALL
)
# Soft: the tags appear in order, but stray text around them is tolerated.
_SOFT_RE = re.compile(r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def to_number(text):
    """First number in the text, as a float. None if there isn't one.

    Lenient on purpose: the policy writes '$18', '18 clips', '18.0' long before
    it learns to emit a bare integer, and those are all correct reasoning.
    """
    if text is None:
        return None
    match = _NUMBER_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def extract_gold(answer_field):
    """GSM8K stores the solution as '<worked solution> #### 42'."""
    return answer_field.split("####")[-1].strip()


class Completion:
    """One model completion, parsed once.

    Each reward tier below needs some view of the same string (the tagged
    answer, the trailing number, which tags are present). Parsing is cached so
    scoring a completion against all six tiers walks it once, not six times.
    """

    TAGS = ("<reasoning>", "</reasoning>", "<answer>", "</answer>")

    def __init__(self, completion):
        # Conversational datasets hand back [{"role": ..., "content": ...}].
        self.text = (
            completion[0]["content"] if isinstance(completion, list) else completion
        ) or ""

    def __repr__(self):
        return f"Completion({self.text[:40]!r}...)"

    @cached_property
    def tagged_answer(self):
        """The <answer> payload, or None when the tag is missing.

        Tag-only on purpose -- the format rewards need to measure tag usage.
        """
        match = _ANSWER_RE.search(self.text)
        return match.group(1).strip() if match else None

    @cached_property
    def final_answer(self):
        """The model's answer, tolerating no tags at all.

        Falls back to the last number in the response, which is the standard
        GSM8K eval convention. This matters more than it looks: Qwen2.5-0.5B-
        Instruct does real step-by-step reasoning but ignores the <answer>
        wrapper entirely, so a tag-only reward scores every completion 0 -> zero
        variance -> no gradient. Scoring the last number instead meets the policy
        where it already is, which is the only way GRPO has something to amplify.
        """
        if self.tagged_answer is not None:
            return self.tagged_answer
        numbers = _NUMBER_RE.findall(self.text.replace(",", ""))
        return numbers[-1] if numbers else None

    @cached_property
    def final_number(self):
        return to_number(self.final_answer)

    @cached_property
    def tag_hits(self):
        """How many of the four tags appear exactly once."""
        return sum(self.text.count(tag) == 1 for tag in self.TAGS)

    @cached_property
    def is_soft_format(self):
        return bool(_SOFT_RE.search(self.text))

    @cached_property
    def is_strict_format(self):
        return bool(_STRICT_RE.match(self.text))

    def is_correct(self, gold):
        """Whether the completion's final number matches the gold number."""
        want = to_number(gold)
        got = self.final_number
        return got is not None and want is not None and abs(got - want) < 1e-4

    def proximity(self, gold):
        """Graded credit for being numerically close (9 beats 3 when it's 10).

        Turns a binary hit/miss into a slope the policy can climb, which is the
        difference between "no group ever varies" and "there is a gradient".
        """
        want, got = to_number(gold), self.final_number
        if got is None or want is None:
            return 0.0
        rel_err = abs(got - want) / max(1.0, abs(want))
        return max(0.0, 1.0 - min(1.0, rel_err))


######################
# reward functions
######################
# GRPOTrainer sums the list of reward funcs, and logs each one separately as
# rewards/<name>/mean. They stay split into tiers for two reasons:
#   1. in-group variance while the policy is still bad at the task -- if
#      correctness were the only signal, early groups would be all-zero -> zero
#      advantage -> no gradient;
#   2. observability -- the per-function means are what reveal a policy that is
#      farming format points instead of learning to solve the problem.
# Weights live here (not in GRPOConfig.reward_weights) so the tiers stay
# readable; pass reward_weights to scale them without editing this file.
CORRECTNESS_WEIGHT = 2.0
PROXIMITY_WEIGHT = 0.5
NUMERIC_WEIGHT = 0.25
XMLCOUNT_WEIGHT = 0.125
SOFT_FORMAT_WEIGHT = 0.25
STRICT_FORMAT_WEIGHT = 0.25


def correctness_reward(completions, gold, **_) -> list[float | None]:
    """The real objective: right answer or not."""
    return [
        CORRECTNESS_WEIGHT if Completion(c).is_correct(g) else 0.0
        for c, g in zip(completions, gold)
    ]


def proximity_reward(completions, gold, **_) -> list[float | None]:
    """Partial credit for landing near the gold number."""
    return [
        PROXIMITY_WEIGHT * Completion(c).proximity(g) for c, g in zip(completions, gold)
    ]


def numeric_reward(completions, **_) -> list[float | None]:
    """Credit for putting *a number* in <answer>, right or wrong."""
    return [
        NUMERIC_WEIGHT if to_number(Completion(c).tagged_answer) is not None else 0.0
        for c in completions
    ]


def xmlcount_reward(completions, **_) -> list[float | None]:
    """Fractional credit per XML tag present.

    Deliberately NOT all-or-nothing: a policy that has never emitted the full
    structure still scores above zero for getting one tag right, which keeps
    reward variance inside the group.
    """
    return [XMLCOUNT_WEIGHT * Completion(c).tag_hits for c in completions]


def soft_format_reward(completions, **_) -> list[float | None]:
    """Tags present and ordered; stray text around them tolerated."""
    return [
        SOFT_FORMAT_WEIGHT if Completion(c).is_soft_format else 0.0 for c in completions
    ]


def strict_format_reward(completions, **_) -> list[float | None]:
    """The exact requested structure, nothing else."""
    return [
        STRICT_FORMAT_WEIGHT if Completion(c).is_strict_format else 0.0
        for c in completions
    ]


# Graded from "any tag at all" up to "exactly right", so reward rises smoothly
# as the policy improves instead of stepping 0 -> max. Correctness still
# dominates (2.0 + 0.5 proximity vs 1.0 total for format) so the policy cannot
# win by only learning to emit tags.
REWARD_FUNCS = [
    correctness_reward,
    proximity_reward,
    numeric_reward,
    xmlcount_reward,
    soft_format_reward,
    strict_format_reward,
]


######################
# dataset
######################
def load_gsm8k(split="train", system_message=SYSTEM_MESSAGE, limit=None):
    """Load GSM8K into the conversational format GRPOTrainer expects.

    `gold` is carried as an extra column; TRL passes unknown columns straight
    through to the reward functions as keyword arguments.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    def to_conversational(ex):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": ex["question"]})
        # `reference` keeps GSM8K's full worked solution (steps + "#### 42") so
        # eval can show a gold CoT next to the model's; `gold` is just the number.
        return {
            "prompt": messages,
            "gold": extract_gold(ex["answer"]),
            "reference": ex["answer"],
        }

    return ds.map(to_conversational, remove_columns=ds.column_names)


# uv run python -m src.gsm8k_rewards
if __name__ == "__main__":
    perfect = "<reasoning>\n2 + 2 = 4\n</reasoning>\n<answer>\n4\n</answer>"
    untagged = "He buys 2 and 2, so the total is 4"
    close = "<reasoning>\nmath\n</reasoning>\n<answer>\n5\n</answer>"

    for label, text in [("perfect", perfect), ("untagged", untagged), ("close", close)]:
        c = Completion(text)
        total = sum(f([text], gold=["4"])[0] for f in REWARD_FUNCS)
        print(
            f"{label:9s} final={c.final_answer!r:6s} correct={c.is_correct('4')} "
            f"tags={c.tag_hits} strict={c.is_strict_format} total_reward={total:.3f}"
        )

    # Caching sanity: repeated access must not re-parse into a different value.
    c = Completion(perfect)
    assert c.final_answer is c.final_answer
    assert c.tag_hits == 4 and c.is_strict_format
    print("ok: cached_property stable")

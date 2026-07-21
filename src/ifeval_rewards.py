"""Verifiable reward functions for the IFEval instruction-following dataset.

IFEval ships every prompt with a machine-readable constraint list
(`instruction_id_list`) plus their parameters (`kwargs`), so the reward can be
computed by running deterministic checkers instead of an LLM judge -- exactly
the "verifiable reward" setup GRPO needs.

Only a subset of the 25 upstream instruction types is implemented here: the
ones checkable on the short outputs a small model can actually produce, and
that need no extra dependencies (language detection, prompt echoing, ...).
`load_ifeval()` drops any example using an unimplemented type.
"""

import json
import re
from functools import cached_property

from datasets import load_dataset

from . import dprint

_WORD_RE = re.compile(r"\b\w+\b")
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_TITLE_RE = re.compile(r"<<(.+?)>>", re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"\[[^\[\]]+\]")
_BULLET_RE = re.compile(r"^\s*[\*\-]\s+", re.MULTILINE)
_HIGHLIGHT_RE = re.compile(r"\*+([^\*\n]+?)\*+")
_PARAGRAPH_RE = re.compile(r"\n\s*\*\*\*\s*\n")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


class Response:
    """One model completion plus the derived views its checkers need.

    Every accessor is cached because a single completion is scored against
    several constraints, and the reward runs on every sampled completion of
    every group -- recomputing the word list per constraint shows up in
    training time.
    """

    # A response that loops is worth 0 no matter which constraints it happens
    # to satisfy: unchecked, GRPO learns to spam one token forever because that
    # is the cheapest way to satisfy a counting constraint.
    DEGENERACY_THRESHOLD = 0.35
    DEGENERACY_NGRAM = 3

    # The prompts spell out their own format with examples ("wrapped in double
    # angular brackets, i.e. <<title>>"). Echoing the example verbatim satisfies
    # a naive checker without writing a title, and GRPO finds that shortcut fast.
    PLACEHOLDER_TITLES = frozenset(
        {"title", "your title", "short message", "poem", "my title"}
    )

    def __init__(self, text):
        self.text = text or ""

    def __repr__(self):
        return f"Response({self.text[:40]!r}...)"

    ######################
    # derived views
    ######################
    @cached_property
    def lowered(self):
        return self.text.lower()

    @cached_property
    def stripped(self):
        return self.text.strip()

    @cached_property
    def words(self):
        return _WORD_RE.findall(self.text)

    @cached_property
    def sentences(self):
        return [p for p in _SENTENCE_RE.split(self.text) if p.strip()]

    @cached_property
    def distinct_ratio(self):
        """Share of n-grams that are unique. Near 0 means the text is a loop."""
        toks = self.text.split()
        n = self.DEGENERACY_NGRAM
        if len(toks) <= n:
            return 1.0
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        return len(set(grams)) / len(grams)

    @cached_property
    def is_degenerate(self):
        return self.distinct_ratio < self.DEGENERACY_THRESHOLD

    @staticmethod
    def _compare(count, relation, target):
        """IFEval encodes bounds as an ('at least' | 'less than', N) pair."""
        if relation == "at least":
            return count >= target
        if relation == "less than":
            return count < target
        raise ValueError(f"unknown relation: {relation}")

    ######################
    # checkers: (kwargs) -> bool
    ######################
    def no_comma(self, kw):
        return "," not in self.text

    def lowercase(self, kw):
        return self.text == self.lowered

    def uppercase(self, kw):
        return self.text == self.text.upper()

    def capital_word_frequency(self, kw):
        n = sum(1 for w in self.words if w.isupper())
        return self._compare(n, kw["capital_relation"], kw["capital_frequency"])

    def keywords_existence(self, kw):
        return all(k.lower() in self.lowered for k in kw["keywords"])

    def forbidden_words(self, kw):
        return all(k.lower() not in self.lowered for k in kw["forbidden_words"])

    def keyword_frequency(self, kw):
        n = len(re.findall(re.escape(kw["keyword"]), self.text, flags=re.IGNORECASE))
        return self._compare(n, kw["relation"], kw["frequency"])

    def letter_frequency(self, kw):
        n = self.lowered.count(kw["letter"].lower())
        return self._compare(n, kw["let_relation"], kw["let_frequency"])

    def number_words(self, kw):
        return self._compare(len(self.words), kw["relation"], kw["num_words"])

    def number_sentences(self, kw):
        return self._compare(len(self.sentences), kw["relation"], kw["num_sentences"])

    def number_paragraphs(self, kw):
        # Paragraphs are separated by a markdown divider line of ***.
        paras = [p for p in _PARAGRAPH_RE.split(self.text) if p.strip()]
        return len(paras) == kw["num_paragraphs"]

    def title(self, kw):
        match = _TITLE_RE.search(self.text)
        if not match:
            return False
        inner = match.group(1).strip().lower()
        return bool(inner) and inner not in self.PLACEHOLDER_TITLES

    def number_bullets(self, kw):
        return len(_BULLET_RE.findall(self.text)) == kw["num_bullets"]

    def highlighted_sections(self, kw):
        # Count distinct, non-trivial spans only: spamming "***" or repeating one
        # phrase is not N highlighted sections.
        spans = _HIGHLIGHT_RE.findall(self.text)
        distinct = {s.strip().lower() for s in spans if len(s.strip()) >= 3}
        return len(distinct) >= kw["num_highlights"]

    def multiple_sections(self, kw):
        n = len(
            re.findall(re.escape(kw["section_spliter"]), self.text, flags=re.IGNORECASE)
        )
        return n >= kw["num_sections"]

    def json_format(self, kw):
        try:
            json.loads(_FENCE_RE.sub("", self.stripped))
            return True
        except ValueError:
            return False

    def number_placeholders(self, kw):
        return len(_PLACEHOLDER_RE.findall(self.text)) >= kw["num_placeholders"]

    def postscript(self, kw):
        return kw["postscript_marker"].lower() in self.lowered

    def quotation(self, kw):
        s = self.stripped
        return len(s) >= 2 and s.startswith('"') and s.endswith('"')

    def end_checker(self, kw):
        return self.stripped.lower().endswith(kw["end_phrase"].strip().lower())

    ######################
    # scoring
    ######################
    def check(self, instruction_id, kw):
        """Run one checker. Model output is arbitrary text, so a checker that
        trips over it counts as "not satisfied" rather than killing training."""
        try:
            return bool(getattr(self, CHECKERS[instruction_id])(kw))
        except Exception:  # noqa: BLE001 - reward must never raise
            return False

    def satisfaction(self, instruction_id_list, kwargs_list):
        """Fraction of constraints satisfied, in [0, 1].

        Graded rather than all-or-nothing on purpose: partial credit keeps
        reward variance inside a GRPO group, and zero variance means zero
        advantage. Degenerate output scores 0 outright -- otherwise a repetition
        loop that happens to satisfy a format constraint outscores a real answer.
        """
        if not instruction_id_list or self.is_degenerate:
            return 0.0
        hits = sum(
            self.check(i, kw) for i, kw in zip(instruction_id_list, kwargs_list)
        )
        return hits / len(instruction_id_list)


# instruction id -> Response method name.
CHECKERS = {
    "punctuation:no_comma": "no_comma",
    "change_case:english_lowercase": "lowercase",
    "change_case:english_capital": "uppercase",
    "change_case:capital_word_frequency": "capital_word_frequency",
    "keywords:existence": "keywords_existence",
    "keywords:forbidden_words": "forbidden_words",
    "keywords:frequency": "keyword_frequency",
    "keywords:letter_frequency": "letter_frequency",
    "length_constraints:number_words": "number_words",
    "length_constraints:number_sentences": "number_sentences",
    "length_constraints:number_paragraphs": "number_paragraphs",
    "detectable_format:title": "title",
    "detectable_format:number_bullet_lists": "number_bullets",
    "detectable_format:number_highlighted_sections": "highlighted_sections",
    "detectable_format:multiple_sections": "multiple_sections",
    "detectable_format:json_format": "json_format",
    "detectable_content:number_placeholders": "number_placeholders",
    "detectable_content:postscript": "postscript",
    "startend:quotation": "quotation",
    "startend:end_checker": "end_checker",
}
SUPPORTED = frozenset(CHECKERS)


######################
# module-level API
######################
def satisfaction(response, instruction_id_list, kwargs_list):
    """Score one raw string. Thin wrapper so callers need not build a Response."""
    return Response(response).satisfaction(instruction_id_list, kwargs_list)


def _completion_text(completion):
    # Conversational datasets hand back [{"role": ..., "content": ...}].
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


def ifeval_reward(completions, instruction_id_list, constraint_kwargs, **_):
    """TRL reward function. Extra dataset columns arrive as keyword arguments."""
    return [
        Response(_completion_text(c)).satisfaction(ids, kws)
        for c, ids, kws in zip(completions, instruction_id_list, constraint_kwargs)
    ]


######################
# dataset
######################
def _too_long(ids, kws, max_words):
    """IFEval asks for 300+ word essays in places. A 135M model with a couple
    hundred tokens of budget can never satisfy those, so every sample in the
    group scores 0 and the group contributes no gradient -- drop them."""
    for i, kw in zip(ids, kws):
        if i == "length_constraints:number_words":
            if kw.get("relation") == "at least" and (kw.get("num_words") or 0) > max_words:
                return True
    return False


def load_ifeval(split="train", max_words=150, system_message=None):
    """Load IFEval, keep only examples this module can fully verify, and shape
    it into the conversational format GRPOTrainer expects."""
    ds = load_dataset("google/IFEval", split=split)
    before = len(ds)

    ds = ds.filter(
        lambda ex: all(i in SUPPORTED for i in ex["instruction_id_list"])
        and not _too_long(ex["instruction_id_list"], ex["kwargs"], max_words)
    )

    def to_conversational(ex):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": ex["prompt"]})
        # Renamed from "kwargs" so the reward signature reads unambiguously.
        return {"prompt": messages, "constraint_kwargs": ex["kwargs"]}

    ds = ds.map(to_conversational, remove_columns=["key", "kwargs"])
    dprint(f"IFEval: kept {len(ds)}/{before} examples (max_words={max_words})")
    return ds


# uv run python -m src.ifeval_rewards
if __name__ == "__main__":
    ds = load_ifeval()
    print(ds)

    cases = [
        ("no comma here", ["punctuation:no_comma"], [{}], 1.0),
        ("has, a comma", ["punctuation:no_comma"], [{}], 0.0),
        ("<<Emerald Isle Dawn>> body", ["detectable_format:title"], [{}], 1.0),
        ('"quoted"', ["startend:quotation"], [{}], 1.0),
        ('{"a": 1}', ["detectable_format:json_format"], [{}], 1.0),
        ("* one\n* two", ["detectable_format:number_bullet_lists"], [{"num_bullets": 2}], 1.0),
        ("all lower, and no comma", ["change_case:english_lowercase", "punctuation:no_comma"], [{}, {}], 0.5),
        # Anti-reward-hacking guards.
        ("<<title>> body text here", ["detectable_format:title"], [{}], 0.0),
        ("<<t>>\n" + "***\n" * 60, ["detectable_format:title"], [{}], 0.0),
        ("*ab* x *ab* y *ab* z", ["detectable_format:number_highlighted_sections"], [{"num_highlights": 3}], 0.0),
        ("*alpha one* q *beta two* q *gamma three*", ["detectable_format:number_highlighted_sections"], [{"num_highlights": 3}], 1.0),
    ]
    for text, ids, kws, want in cases:
        got = satisfaction(text, ids, kws)
        print(f"{'ok ' if got == want else 'FAIL'} want={want:.2f} got={got:.2f}  {text[:38]!r}")

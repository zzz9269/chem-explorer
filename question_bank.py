"""question_bank.py — the catalogue of question types the Ask mode answers.

The data lives in `question_bank.json` (edit that to teach the app a new
phrasing — no code change). This module only loads it and provides:

* `question_form(text)` — the *shape* of a question (yes/no, how long, how much,
  why, which, …), which decides how an answer should open: a verdict for a
  yes/no question, a duration for "how long", a dose for "how much";
* `instantiate(template, slots, i)` — fill `{drug}`-style placeholders, for
  tests and for the example index chat_router matches phrasings against.

No Streamlit, no network, no AI model.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
BANK_PATH = os.path.join(_HERE, "question_bank.json")

# Placeholders that stand for the drug(s) asked about — normalised to the
# router's 'drugx' marker when examples are indexed.
DRUG_SLOTS = ("drug", "drug2", "brand")


@lru_cache(maxsize=1)
def load() -> Dict:
    try:
        with open(BANK_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 — the router works without the bank
        return {"types": [], "slots": {}, "forms": {}}


def types() -> List[Dict]:
    return list(load().get("types") or [])


def type_by_id(type_id: str) -> Optional[Dict]:
    return next((t for t in types() if t.get("id") == type_id), None)


def slots_for(qtype: Dict) -> Dict[str, List[str]]:
    """Default slot values overridden by the type's own."""
    out = dict(load().get("slots") or {})
    out.update(qtype.get("slots") or {})
    return out


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def instantiate(template: str, slots: Dict[str, List[str]], i: int = 0) -> str:
    """'does {drug} cause {effect}' → 'does metformin cause drowsiness' (the
    i-th value of each slot, cycling, so paired slots stay aligned)."""
    def fill(m):
        vals = slots.get(m.group(1)) or [m.group(0)]
        return vals[i % len(vals)]
    return _PLACEHOLDER_RE.sub(fill, template)


def examples(qtype: Dict) -> List[str]:
    return list(qtype.get("examples") or [])


# ---------------------------------------------------------------------------
# Question form
# ---------------------------------------------------------------------------
# First match wins, so the order is deliberate: 'why' and the quantity forms
# beat the generic yes/no opener ('should I take X in the morning or at night'
# is a WHEN question; 'is X better than Y' is a WHICH question).
_FORM_RULES = [
    ("why", r"^\W*(?:but |so |and )?why\b|\bwhy (?:is|are|do|does|did|would|can|can'?t|should|must|"
            r"isn'?t|doesn'?t|don'?t|was|were|has|have)\b|\bhow come\b|\bfor what reason\b"),
    ("how_common", r"\bhow (?:common|likely|rare|frequent)\b|\bhow many (?:people|patients|users)\b|"
                   r"\bwhat (?:are the )?(?:chances|odds|percentage|incidence)\b"),
    ("how_often", r"\bhow often\b|\bhow (?:many|much) times\b|\bhow frequently\b|\bhow regularly\b|"
                  r"\bhow many hours (?:apart|between)\b|\bhours between doses\b|\btimes (?:a|per) day\b"),
    ("how_long", r"\bhow long\b|\bhow soon\b|\bhow (?:fast|quickly)\b|\bhalf[- ]?li(?:fe|ves)\b|"
                 r"\bt ?1/2\b|\bwhen (?:does|will|do) (?:it|drugx|they) (?:start|kick|begin|wear|stop)\b|"
                 r"\bshelf[- ]?life\b|\bhow many (?:days|weeks|hours|minutes|months|years) "
                 r"(?:before|until|till|for|does|do|will|should|must|can|is)\b"),
    ("how_much", r"\bhow (?:much|many)\b|\b(?:max(?:imum)?|minimum|usual|normal|recommended|starting|"
                 r"daily|correct|right) dos(?:e|es|age)\b|"
                 r"\bwhat(?:'s| is)? (?:the |a |my )?(?:usual |normal |correct |right |recommended |safe |"
                 r"typical |standard |starting |good |daily |max |maximum |minimum )?"
                 r"dos(?:e|es|age)\b|\bdosage\b(?! forms?)|"
                 r"\bdos(?:e|es|age|ing) (?:for|in) (?:an? )?(?:children|child|kids?|adults?|elderly|"
                 r"infants?|bab(?:y|ies)|toddlers?|teens?|\d+)|"
                 r"\b(?:pediatric|paediatric|adult|geriatric|child(?:ren)?'?s?|kids?) dos(?:e|es|age|ing)\b|"
                 r"\bprice\b|\bcost\b|\bexpensive\b|\bmost drugx\b|\bmax (?:i|you) can\b|"
                 r"\b(?:highest|lowest|maximum|max|minimum) (?:daily |single )?(?:dose|amount|limit)\b|"
                 r"\bdaily limit\b|\bupper limit\b|^\W*max\b|\bmax drugx\b"),
    ("when", r"^\W*when\b|\bwhat time\b|\bmorning or (?:at )?night\b|\bbefore or after\b|"
             r"\bbest time\b|\bwhen (?:should|to|do|can) (?:i|you|we)\b|\bwith or without (?:food|meals?)\b"),
    ("which", r"\bwhich (?:is|one|of)\b|\bwhich (?:\w+ )?(?:is|are) (?:better|best|stronger|safer|"
              r"worse|weaker)\b|\b(?:better|stronger|safer|weaker|worse) than\b|"
              r"\bor\b[^?.!]{0,40}\bwhich\b|\bwhich\b[^?.!]{0,40}\bor\b|\bvs\.?\b|\bversus\b|"
              r"\bdifference between\b|\bcompare\b"),
    # 'can you tell me about…' is a request, not a yes/no question
    ("what", r"^\W*(?:can|could|would|will) you (?:please )?(?:tell|explain|show|give|list|draw|"
             r"describe|help|name|find)\b"),
    ("yes_no", r"^\W*(?:is|are|does|do|did|can|could|will|would|should|shall|may|might|has|have|"
               r"am|was|were|isn'?t|aren'?t|doesn'?t|don'?t|can'?t|won'?t)\b|"
               r"\bis it (?:safe|ok|okay|alright|fine|bad|dangerous|true)\b|\b(?:ok|okay|safe) to\b|"
               r"\b(?:right|correct|isn'?t it|aren'?t they|yes or no)\s*\??\s*$"),
    ("how", r"^\W*how\b|\bhow (?:does|do|is|are|can|should|to|would)\b|\bexplain how\b"),
    ("list", r"\b(?:list|lists|examples? of|name (?:some|a few)|types of|kinds of)\b|"
             r"^\W*what are (?:the |some )?(?:\w+ ){0,2}(?:side effects|interactions|brands|uses|"
             r"alternatives|types|examples|adverse)\b"),
    ("what", r"^\W*(?:what|what'?s|whats|who|whom|whose|where|which|tell me|explain|describe|define|"
             r"give me|show me|info|information)\b"),
]
_FORM_RES = [(name, re.compile(pat, re.I)) for name, pat in _FORM_RULES]


def question_form(text: str) -> str:
    """The question's form (see the bank's 'forms'), 'statement' when none."""
    t = (text or "").strip().lower()
    for name, rx in _FORM_RES:
        if rx.search(t):
            return name
    return "statement"


# Forms that open an answer the same way (a description) — interchangeable
# when checking that a question was understood.
DESCRIPTIVE = frozenset({"what", "list", "statement", "how"})


def form_ok(actual: str, expected: List[str]) -> bool:
    """True when `actual` is one of the expected forms, or both are descriptive."""
    if actual in expected:
        return True
    return actual in DESCRIPTIVE and any(e in DESCRIPTIVE for e in expected)


def coverage_summary() -> Dict[str, int]:
    """{category: number of question types} — for docs/README."""
    out: Dict[str, int] = {}
    for t in types():
        out[t.get("category", "Other")] = out.get(t.get("category", "Other"), 0) + 1
    return out

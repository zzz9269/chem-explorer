"""
Does the ANSWER answer the question? The parser tests check that a question
was routed to the right topic — but a right topic can still produce an answer
that misses the point (the 2026-10-25 review found 'Yes — approved for
Headache' for 'can I take ibuprofen if I'm on warfarin', a member list for 'why
do ACE inhibitors cause a dry cough'). These run the whole app on live sources
and check each answer contains the facts a pharmacist's answer would.

Live sources change, so a failure can mean a source was reworded: read the
answer before 'fixing' the code. Slow (one shared app, ~15 live questions):

    python -m pytest -m slow tests/test_answer_quality.py
"""
import os
import re
import time

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = [pytest.mark.network, pytest.mark.slow]

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chem_v4.py")
PAUSE_S = 5

# (question, every pattern must appear, no pattern may appear)
CASES = [
    # the unsafe answer that started the review
    ("can i take ibuprofen for my headache if i'm on warfarin",
     [r"Caution", r"bleed"], [r"Yes — approved"]),
    ("why does warfarin have so many interactions", [r"CYP2C9|vitamin K"], []),
    ("why do ace inhibitors cause a dry cough", [r"[Bb]radykinin"], []),
    ("what is the therapeutic index of digoxin", [r"narrow therapeutic"], [r"couldn't find .index"]),
    ("how long after taking omeprazole can i eat", [r"before meals?"], [r"Table 3"]),
    ("what is first pass metabolism", [r"first[- ]pass"], []),
    ("what drugs should be avoided in g6pd deficiency",
     [r"[Dd]apsone|[Pp]rimaquine|[Rr]asburicase|methylene blue"], []),
    ("how do ssris work", [r"reuptake"], []),
    ("which beta blockers are cardioselective", [r"[Bb]isoprolol|[Mm]etoprolol|[Aa]tenolol|[Nn]ebivolol"], []),
    ("why is amlodipine making my ankles swell", [r"edema|oedema"], []),
    ("explain the difference between agonists and antagonists", [r"activates", r"block"], []),
    ("why should you not take nsaids with ace inhibitors and diuretics", [r"[Kk]idney|renal"], []),
    ("what is serotonin syndrome and which drugs cause it",
     [r"serotonergic", r"tramadol|SSRI|MAOI|clomipramine|imipramine"], []),
    # 'it' = the antidote the first part named (not paracetamol, not glutathione)
    ("what's the antidote for paracetamol overdose and how does it work",
     [r"[Aa]cetylcysteine", r"How Acetylcysteine works"], [r"How Glutathione works"]),
    ("what is serotonin syndrome and how is it treated", [r"cyproheptadine"], []),
]


@pytest.fixture(scope="module")
def ask():
    """One app for every question (warm caches, like a real session); each
    question's answer is the last chat message."""
    at = AppTest.from_file(APP, default_timeout=400)
    at.run()
    at.session_state["app_mode"] = "💬 Ask"
    at.session_state["market_region"] = "Singapore (HSA)"
    at.run()

    def _ask(q):
        # Wikimedia throttles this client after ~20 requests a minute (429,
        # Retry-After ~20-30 s); a person doesn't ask 13 questions a minute
        time.sleep(PAUSE_S)
        at.chat_input[0].set_value(q).run()
        assert not at.exception, [e.value for e in at.exception]
        texts = []

        def walk(node):
            val = getattr(node, "value", None)
            if type(node).__name__ in ("Markdown", "Caption", "Info", "Warning", "Error",
                                       "Success") and val:
                texts.append(str(val))
            for ch in (getattr(node, "children", {}) or {}).values():
                walk(ch)
        walk(at.chat_message[-1])
        return "\n".join(texts)
    yield _ask
    # let Wikimedia's throttle window pass before the next test file asks more
    time.sleep(30)


@pytest.mark.parametrize("q,must,must_not", CASES, ids=[c[0][:50] for c in CASES])
def test_answer_contains_the_facts(ask, q, must, must_not):
    text = ask(q)
    missing = [m for m in must if not re.search(m, text)]
    present = [m for m in must_not if re.search(m, text)]
    assert not missing and not present, (f"missing {missing}, unwanted {present}\n"
                                         f"--- answer ---\n{text[:2500]}")

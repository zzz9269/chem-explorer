# Drug & Chemical Explorer

A Streamlit app for looking up medicines and chemicals: uses, dosing, side
effects, warnings, pharmacology and full chemistry — with every fact tagged by
its source. Singapore mode adds HSA-registered brands and POM / P / GSL status.

**No AI model is used.** Answers (including the *Ask* chat mode) are matched
from the sources below by rule-based code, so nothing is invented — every line
is quoted from a named source you can click through to.

> For information only — not medical advice.

## When to use this instead of a chatbot

| This app | A general AI chatbot |
|---|---|
| Quotes the **current FDA label** wording for doses, boxed warnings and interactions, with the label's revision date | Answers from training data; can state a dose or warning that isn't on any label |
| **Singapore status** from the HSA register: local brands, POM / P / GSL per product | Often gives US or outdated availability |
| Every sentence links to its source; "not stated" when the sources are silent | Rarely shows where a claim came from |
| Chemistry computed from the structure (RDKit): formula, stereocentres, drug-likeness, acid/base centres and charge at pH 7.4 | Recalls or approximates |
| Your own *Mainstream Drug Reference*, side by side with the official sources | — |
| Free, no account, no API key | — |

A chatbot is better at open conversation, at multi-step reasoning about a
specific patient, and at explaining something in new words. This app is the
place to **check** what the sources actually say.

## How Ask understands questions

The parser (`chat_router.py`) recognises these question shapes:

| Shape | Example | Answered from |
|---|---|---|
| One medicine + a topic | *side effects of metformin*, *half-life of diazepam* | that drug's FDA label, PubChem, ChEMBL, Wikipedia |
| Two medicines together | *can I take ibuprofen if I'm on warfarin?* | each drug's sources about the other; a **Caution** verdict when they warn |
| Compare | *metoprolol vs propranolol* | side-by-side table |
| Why / how a link works | *why is glucagon the antidote for beta-blocker overdose?*, *why does warfarin have so many interactions?* | sentences that **explain** (because, due to, by inhibiting…) ranked first, from the drug's, the target's and the drug **class's** articles |
| A drug class | *how do SSRIs work?*, *why do ACE inhibitors cause a dry cough?*, *list some statins* | the class's Wikipedia article + its members (Drug Library) |
| A concept, no medicine | *what is first-pass metabolism?*, *drugs to avoid in G6PD deficiency*, *agonist vs antagonist* | the concept's Wikipedia article (`concepts.json`, else Wikipedia search) |
| A concept about a medicine | *therapeutic index of digoxin*, *zero-order kinetics of phenytoin* | the drug's sources about the concept + the concept's article about the drug |
| A condition | *what can I take for heartburn?*, *first-line treatment for hypertension* | WHO ATC groups (HSA-registered in SG mode) + the condition's article for "first-line" |
| Several questions | *what is serotonin syndrome and which drugs cause it?* | each part in turn; "it" carries over (to the antidote, in *what's the antidote for X and how does it work*) |

**Teach it without code:**
* `question_bank.json` — ~80 question types with ~700 example phrasings; add a
  phrasing to teach the parser a new way of asking.
* `concepts.json` — pharmacology concepts, conditions and drug-reaction
  syndromes, each with the Wikipedia article that answers it; add an entry
  (check the article title exists) to teach a new concept.

Answers open with a "Short answer" chosen by the question's form (a verdict
for yes/no, a duration for "how long", a dose for "how much"), always quoted
from a source or computed from the structure.

## Run locally

```bash
pip install -r requirements.txt
streamlit run chem_v4.py
```

**Wikipedia rate limits.** Wikimedia throttles clients that don't identify a
contact (about 20 requests a minute here). Set `CHEM_EXPLORER_CONTACT` to your
repo URL or an e-mail — on Streamlit Cloud, add a top-level line to the app's
*Secrets*: `CHEM_EXPLORER_CONTACT = "https://github.com/you/your-repo"`. When
Wikipedia does throttle, answers say so instead of guessing.

## Tests

```bash
python -m pytest -m "not network"   # offline rules, ~1 min
python -m pytest -m "not slow"      # + live label/name checks
python -m pytest                    # everything incl. whole-app checks
```

* `tests/test_reasoning_gaps.py` — question shapes the parser used to get wrong
  (two drugs at once, classes, concepts, why-questions, several questions).
* `tests/test_answer_quality.py` (slow) — runs the app on live sources and
  checks each **answer** contains the facts a pharmacist's answer would
  (e.g. *bradykinin* for the ACE-inhibitor cough). A failure can mean a source
  was reworded — read the answer before changing code.

The network tests call PubChem, which blocks clients that exceed 5 requests/s —
don't run them in a tight loop.

## Data sources & licences

| Source | Used for | Licence / terms |
|---|---|---|
| PubChem (NCBI) | structures, records, properties | public domain (NCBI usage policy) |
| openFDA drug labels & FAERS | uses, dosing, warnings, side effects | public domain (US Government work) |
| ChEMBL (EMBL-EBI) | mechanisms, targets, approval status | CC BY-SA 3.0 |
| Guide to Pharmacology (IUPHAR/BPS) | curated targets & affinities | CC BY-SA 4.0 |
| RxNorm / RxClass (NLM) | US brands, drug classes | free NLM terms of service |
| HSA Listing of Registered Therapeutic Products (data.gov.sg) | Singapore brands & availability | Singapore Open Data Licence v1.0 |
| Wikipedia | plain-language summaries, drug-class & concept articles | CC BY-SA 4.0 |
| Wikidata, UniChem, OPSIN, NCI CIR | identifiers, cross-references, name → structure | CC0 / open services |

Refresh the bundled snapshots with `python build_hsa_register.py` and
`python build_drug_library.py`.

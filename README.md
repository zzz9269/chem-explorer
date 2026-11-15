# Drug & Chemical Explorer

A Streamlit app for looking up medicines and chemicals: uses, dosing, side
effects, warnings, pharmacology and full chemistry — with every fact tagged by
its source. Singapore mode adds HSA-registered brands and POM / P / GSL status.

**Ask writes short answers with AI, checked against the sources.** Claude Sonnet
first outlines what a specialist would say, from memory. Claude Haiku then writes
the answer from that outline and the sources below, and every number it states
has to be in a source or it isn't shown. A line with no source is marked †, and
never contains a number. If the AI isn't available, can't be verified or the
daily limit is reached, Ask shows the rule-based answer quoted from the sources.
Questions asked are sent to Anthropic. Explore, the Drug Library and Reference
use no AI.

**Chemistry questions get the same treatment.** *Why is phenol more acidic than
ethanol?*, *is ibuprofen chiral?*, *boiling point of ethanol*, *what happens if
you mix bleach and ammonia?*, *what is an SN2 reaction?* are answered from what
RDKit computes from the structure (formula, stereocentres, functional groups,
acid and base centres), PubChem's measured properties and hazard sections, and
Wikipedia. A value is labelled as measured, calculated or inferred, and the
structure is drawn under an answer about structure. Plain look-ups (*structure
of codeine*, *CAS number of aspirin*, *properties of ethanol*) still show the
app's own drawing or table, with no AI.

**Research questions are answered from the literature.** *What is the latest
evidence for semaglutide in heart failure?*, *are there clinical trials of
psilocybin for depression?*, *is lecanemab effective?* search **PubMed** (through
Europe PMC) for the most cited papers and the newest trials, meta-analyses and
guidelines on the topic, and **ClinicalTrials.gov** for registered studies. The
answer names each study, gives the figures from its abstract and links it, and a
📚 Reading list under the answer has every paper and trial found. *What are the
targets of imatinib and how potent is it?* is answered from **ChEMBL**: the median
of the published IC50 / Ki / Kd measurements at each human target, with how many
measurements that is and their range. The 🔬 Research tab in Explore shows the
same three sources for any compound, with no AI.

**Sums are done by a calculator, not by the AI.** *How many grams of NaCl for
250 mL of 0.5 M?*, *pH of 0.01 M HCl*, *dilute 12 M HCl to 500 mL of 1 M*, *15
mg/kg for an 18 kg child*: the model writes the expression, the app evaluates it
(molar masses from standard atomic weights, hydrates included) and the answer may
only state a figure the calculator produced. If the model put in a value the
question didn't give, the working says so.

**Symptoms.** Describe what you feel in your own words (*the back of my head
hurts when I look down, and it got much worse after a flight*) and the answer
names the most probable condition, explains why it causes each thing you
described, gives the next most likely cause, what helps (self-care, then
non-prescription medicines with the label's limits), when to see a doctor, and
asks for the one or two details that would narrow it down; reply and it refines
the answer. The signs that need urgent care are always shown in a red box, and a
description that could be an emergency starts with calling 995. It reads
MedlinePlus (US National Library of Medicine) and Wikipedia on the condition and
the FDA labels of the medicines it names. Only the condition's name is sent to
those sources, never your own words. The reasoning from your symptoms to a
condition is the AI's own and is marked † where no source states it: it is a
probable cause to discuss with a doctor, not a diagnosis.

**🩺 Medical assistant.** A page of its own, beside Ask, for a consultation: tell
it what's troubling you as you would tell a GP, and it assesses the symptoms as
above, recommends what to do (self-care, a non-prescription medicine, a
pharmacist, a GP appointment, a same-day visit or an ambulance) and asks
follow-up questions, reassessing with each reply. It keeps its own conversation,
separate from Ask, takes your age, conditions and regular medicines into account
when you mention them, and still answers a direct question about a medicine.
The page, the sidebar and every reply say that this is not medical advice and to
see a licensed medical practitioner if you are really ill. It needs the AI key:
without one it says it can't assess symptoms rather than guess.

**Answer length.** Standard answers are short. *Detailed* (sidebar), ending a
question with *in detail*, or **➕ More detail** under an answer gives about twice
as many points. **⬇️ Save conversation** in the sidebar downloads every question,
answer, source and link as a Markdown file.

Two things an answer never takes from the AI's memory: a **number** (a long
decimal may be rounded, 78.24 °C to 78 °C, but 12.5 mg is never 13 mg), and a
drug's **standing in law** (controlled drug, schedule, class). The HSA register
gives POM / P / GSL; it doesn't hold the Misuse of Drugs Act schedules, and the
answer says so rather than guess.

To turn AI on, add `ANTHROPIC_API_KEY` to the Streamlit secrets (never to a file
in this folder). Optional secrets `AI_DAILY_LIMIT` (default 200 questions for the
whole app) and `AI_SESSION_LIMIT` (default 40 per visitor) cap the spend. An
answer costs about S$0.003–0.012 (a detailed or literature answer up to about
S$0.03). `AI_OUTLINE_MODEL` set to `claude-haiku-5-5`
saves about S$0.003 a question, but Haiku's outlines have more errors from
memory. `AI_WRITER_MODEL` set to `claude-sonnet-5-5` has the stronger model write
the answer as well, at roughly ten times the cost of a question. Without a key
the app runs exactly as before, with no AI.

HSA's open data doesn't list every registered product (it has no Cymbalta or
Strattera, though the National Drug Formulary does). A medicine missing from it
is shown as "Not in HSA's open data", not as unregistered.

> For information only — not medical advice.

## When to use this instead of a chatbot

| This app | A general AI chatbot |
|---|---|
| Quotes the **current FDA label** wording for doses, boxed warnings and interactions, with the label's revision date | Answers from training data; can state a dose or warning that isn't on any label |
| **Singapore status** from the HSA register: local brands, POM / P / GSL per product | Often gives US or outdated availability |
| Every sentence links to its source; "not stated" when the sources are silent | Rarely shows where a claim came from |
| Chemistry computed from the structure (RDKit): formula, stereocentres, drug-likeness, acid/base centres and charge at pH 7.4 | Recalls or approximates |
| **Trial results from the paper's own abstract**, each linked to PubMed, plus what is registered on ClinicalTrials.gov today | Recalls studies from training data; can misquote a figure or invent a reference |
| **Potency as the median of every published measurement** (ChEMBL), with the count and range | Quotes one remembered IC50 |
| **Arithmetic by a calculator**, with the working shown | Does sums in its head |
| Your own *Mainstream Drug Reference*, side by side with the official sources | — |
| Free, no account | — |

A chatbot is better at open conversation and at multi-step reasoning about a
specific patient. This app is the place to get a short answer you can **check**
against what the sources actually say.

## How Ask understands questions

With AI on, the outline model also says which compounds and Wikipedia articles a
question is about, and the app reads those: the parser only matches names it
knows, so *methanol* (one letter from a medicine), *bleach* or *is caffeine
polar* were beyond it. A name close to a medicine's is always looked up as typed
before it is treated as a misspelling.

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

* `tests/test_research.py` — the calculator (what it computes and what it refuses),
  detailed answers, answers about the evidence, and how papers, trials and
  potency figures become citable snippets.
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
| Europe PMC (PubMed records) | papers: titles, abstracts, citation counts | abstracts © their publishers; metadata free to reuse |
| ClinicalTrials.gov (NLM) | registered studies: phase, status, size, sponsor | public domain |
| ChEMBL activities (EMBL-EBI) | potency at each target (IC50, Ki, Kd, EC50) | CC BY-SA 3.0 |
| MedlinePlus health topics (NLM) | conditions in plain language: self-care, when to get medical help | NLM-written summaries are public domain; attribution requested |
| Wikidata, UniChem, OPSIN, NCI CIR | identifiers, cross-references, name → structure | CC0 / open services |

Refresh the bundled snapshots with `python build_hsa_register.py` and
`python build_drug_library.py`.

# Drug & Chemical Explorer

A Streamlit app for looking up medicines and chemicals: uses, dosing, side
effects, warnings, pharmacology and full chemistry — with every fact tagged by
its source. Singapore mode adds HSA-registered brands and POM / P / GSL status.

**No AI model is used.** Answers (including the *Ask* chat mode) are matched
from the sources below by rule-based code, so nothing is invented.

> For information only — not medical advice.

## How Ask understands questions

`question_bank.json` is a catalogue of ~80 question types (yes/no, how long,
how much, why, which…) with ~700 example phrasings. The parser uses keyword
rules first; phrasings the rules miss are matched to the nearest question type
by vocabulary similarity — **add a phrasing to the JSON to teach the app it**, no
code change. Each type also states what a good answer must contain, and answers
open with a "Short answer" chosen by the question's form (a verdict for yes/no,
a duration for "how long", a dose for "how much"), always quoted from a source
or computed from the structure.

## Run locally

```bash
pip install -r requirements.txt
streamlit run chem_v4.py
```

## Tests

```bash
python -m pytest -m "not network"   # offline rules, ~10 s
python -m pytest -m "not slow"      # + live label/name checks, ~1 min
python -m pytest                    # everything incl. whole-app checks, ~2 min
```

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
| Wikipedia | plain-language summaries & sections | CC BY-SA 4.0 |
| Wikidata, UniChem, OPSIN, NCI CIR | identifiers, cross-references, name → structure | CC0 / open services |

Refresh the bundled snapshots with `python build_hsa_register.py` and
`python build_drug_library.py`.

"""The Mainstream Drug Reference (3rd ed.) — the user's own study table of the
most-used drugs, parsed from the PDF by build_mainstream_reference.py into
mainstream_reference.json. No Streamlit here; chem_v4 renders it.

Each entry: drug, tag (both/who/top300/None), section, section_no, subsection,
class, mechanism, uses, cautions, id. Some rows group several drugs that share
a mechanism ("Enalapril / perindopril / captopril"); `drug_links()` splits
those so each name can open its own Explore page.
"""
import difflib
import json
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).with_name("mainstream_reference.json")

TAG_LABELS = {
    "both": "WHO essential + Top-300",
    "who": "WHO essential",
    "top300": "Top-300 prescribed",
}

# Rows naming a product family or mixture rather than one resolvable compound.
_NOT_A_COMPOUND = re.compile(
    r"\b(combinations?|contraceptive|pill|implant|patch|IUS|salts|vaccines?|"
    r"immunoglobulins?|antivenoms?|antitoxins?|products?|preparations?|"
    r"solutions?|solution's|fluids?|alginates?|blood|food|micronutrient|"
    r"toxin|derivative|agents|antacids)\b|'s solution|\bMMR\b", re.I)


def is_available() -> bool:
    return DATA.exists()


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def entries() -> list:
    return _load()["entries"]


def sections() -> list:
    return _load()["sections"]


def source() -> str:
    return _load().get("source", "")


def _split_top_level(text: str, sep: str = "/") -> list:
    """Split on `sep` only outside brackets ("Antacids (magnesium/ aluminium
    salts)" stays one part)."""
    parts, depth, cur = [], 0, []
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _query_for(part: str):
    """Search string for one drug name, or None when the part isn't a single
    compound the Explorer could resolve."""
    bare = re.sub(r"\([^)]*\)", "", part).strip(" ,;")
    # "Levodopa with carbidopa or benserazide" → the lead drug
    bare = re.split(r"\s+with\b", bare)[0].strip()
    if not bare or _NOT_A_COMPOUND.search(bare) or len(bare.split()) > 4:
        return None
    return bare


def drug_links(entry: dict) -> list:
    """[(label, query_or_None)] — one per drug named in the row. The first
    part keeps its capital; later parts are capitalised for display."""
    out = []
    for part in _split_top_level(entry["drug"]):
        label = part[:1].upper() + part[1:]
        out.append((label, _query_for(part)))
    return out


# Everyday words → the clinical vocabulary the reference is written in. Hand
# curated (no AI); each key also matches itself. Spelling variants (ae/oe) are
# handled by _norm, so only list genuinely different words here.
SYNONYMS = {
    # not bare "mucus" — it also hits "cervical mucus" under contraceptives
    "phlegm": ["sputum", "expectorant", "mucolytic", "productive cough"],
    "mucus": ["sputum", "mucolytic", "expectorant"],
    "chesty": ["productive cough", "sputum", "expectorant"],
    "snot": ["rhinitis", "nasal", "decongestant"],
    "runny": ["rhinitis", "nasal"],
    "stuffy": ["nasal congestion", "decongestant"],
    "blocked": ["congestion", "decongestant", "obstruction"],
    "heartburn": ["reflux", "gord", "dyspepsia", "acid suppression"],
    "indigestion": ["dyspepsia", "reflux"],
    "ulcer": ["ulceration", "peptic"],
    "painkiller": ["analgesic", "analgesia"],
    "pain": ["analgesic", "analgesia"],
    "fever": ["antipyretic", "pyrexia"],
    "temperature": ["fever", "antipyretic"],
    "itch": ["pruritus", "itching"],
    "itchy": ["pruritus", "itching"],
    "rash": ["eruption", "urticaria", "skin reaction", "dermatitis"],
    "hives": ["urticaria"],
    "allergy": ["allergic", "antihistamine", "anaphylaxis", "hay fever"],
    "sleep": ["insomnia", "hypnotic", "sedation", "drowsiness"],
    "sleepy": ["drowsiness", "sedation"],
    "drowsy": ["drowsiness", "sedation"],
    "anxiety": ["anxiolytic", "panic"],
    "depression": ["antidepressant", "depressive"],
    "constipation": ["laxative", "constipating"],
    "sick": ["nausea", "vomiting", "antiemetic"],
    "vomiting": ["antiemetic", "emesis", "nausea"],
    "nausea": ["antiemetic", "vomiting"],
    "bp": ["blood pressure", "hypertension"],
    "blood thinner": ["anticoagulant", "antiplatelet"],
    "thinner": ["anticoagulant", "antiplatelet"],
    "clot": ["thrombosis", "thrombotic", "anticoagulant", "embolism", "vte"],
    "bleeding": ["haemorrhage", "bleed"],
    "heart attack": ["myocardial infarction", "mi", "acs"],
    "fits": ["seizure", "epilepsy", "convulsion"],
    "seizure": ["epilepsy", "antiepileptic", "convulsion", "seizure threshold"],
    "sugar": ["glucose", "glycaemic", "hypoglycaemia", "diabetes"],
    "cholesterol": ["lipid", "ldl", "statin"],
    "kidney": ["renal", "nephrotoxic", "nephrotoxicity"],
    "liver": ["hepatic", "hepatotoxicity", "hepatitis"],
    "weight": ["obesity", "weight gain", "weight loss", "appetite"],
    "hair loss": ["alopecia"],
    "acne": ["acne vulgaris", "retinoid"],
    "eczema": ["dermatitis", "atopic"],
    "asthma": ["bronchospasm", "bronchodilator"],
    "wheeze": ["bronchospasm", "asthma", "bronchodilator"],
    "breathless": ["dyspnoea", "breathlessness"],
    "pregnant": ["pregnancy", "teratogenic", "teratogen"],
    "pregnancy": ["teratogenic", "teratogen"],
    "baby": ["neonatal", "neonates", "infant", "foetal"],
    "children": ["paediatric", "child"],
    "elderly": ["older", "falls"],
    "infection": ["antibacterial", "antibiotic", "sepsis"],
    "antibiotic": ["antibacterial"],
    "water tablet": ["diuretic"],
    "erection": ["erectile"],
    "period": ["menstrual", "dysmenorrhoea", "menorrhagia"],
    "migraine": ["triptan", "headache"],
    "dizzy": ["dizziness", "vertigo", "postural hypotension"],
}
# haem/hem, oesophagus/esophagus, oedema/edema, diarrhoea/diarrhea
_SPELL_RE = [(re.compile(r"ae"), "e"), (re.compile(r"oe"), "e")]


def _norm(text: str) -> str:
    """Case-fold and fold British/US spelling (haemorrhage = hemorrhage,
    oedema = edema, diarrhoea = diarrhea) so either spelling matches."""
    text = text.casefold()
    for rx, rep in _SPELL_RE:
        text = rx.sub(rep, text)
    return text


def _hay(e: dict) -> str:
    return _norm(" ".join((e["drug"], e["class"], e["mechanism"], e["uses"],
                           e["cautions"], e["subsection"] or "")))


@lru_cache(maxsize=1)
def _vocab() -> tuple:
    words = set()
    for e in entries():
        words.update(w for w in re.findall(r"[a-z][a-z0-9-]{3,}", _hay(e)))
    words.update(_norm(k) for k in SYNONYMS if " " not in k)
    return tuple(sorted(words))


def _has(hay: str, term: str) -> bool:
    """`term` at the start of a word in `hay` ("cough" matches "coughing",
    "mi" doesn't match "vomiting"); terms of ≤3 letters must be the whole word
    ("mi" doesn't match "midodrine")."""
    tail = r"(?![a-z0-9])" if len(term) <= 3 else ""
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + tail, hay) is not None


def expand_query(query: str) -> list:
    """Split `query` into terms, each with the alternatives it may match:
    [{"term", "alts", "corrected_from"}]. Two-word synonym keys ("blood
    thinner") are kept together; an unknown word is typo-corrected against the
    reference's own vocabulary ("ohlegm" → "phlegm")."""
    q = " ".join(_norm(query).split())
    terms = []
    for key in sorted((k for k in SYNONYMS if " " in k), key=len, reverse=True):
        if _norm(key) in q:
            terms.append(_norm(key))
            q = q.replace(_norm(key), " ")
    terms += q.split()
    hays = [_hay(e) for e in entries()]
    out = []
    for t in terms:
        corrected = None
        known = t in {_norm(k) for k in SYNONYMS} or any(_has(h, t) for h in hays)
        if not known and len(t) >= 4:
            close = difflib.get_close_matches(t, _vocab(), n=1, cutoff=0.75)
            if close:
                corrected, t = t, close[0]
        syn = next((v for k, v in SYNONYMS.items() if _norm(k) == t), [])
        out.append({"term": t, "alts": [t] + [_norm(s) for s in syn],
                    "corrected_from": corrected})
    return out


def search(query: str, rows=None) -> list:
    """Entries matching every term of `query` (or one of its synonyms /
    spelling variants) anywhere in the row — name, class, mechanism, uses,
    cautions. Ranked: drug-name matches, then rows where it's what the drug
    is FOR (uses / mechanism / class), then cautions-only mentions; within
    each, literal-word matches before synonym-only ones."""
    terms = expand_query(query)
    if not terms:
        return list(rows if rows is not None else entries())

    def all_in(text):
        return all(any(_has(text, a) for a in t["alts"]) for t in terms)

    hits = []
    for e in rows if rows is not None else entries():
        hay = _hay(e)
        if not all_in(hay):
            continue
        if all_in(_norm(e["drug"])):
            where = 0
        elif all_in(_norm(" ".join((e["uses"], e["mechanism"], e["class"],
                                    e["subsection"] or "")))):
            where = 1
        else:
            where = 2
        literal = all(_has(hay, t["term"]) for t in terms)
        hits.append((where, 0 if literal else 1, e["id"], e))
    return [h[-1] for h in sorted(hits, key=lambda h: h[:3])]


def find_entries(*names: str) -> list:
    """Every reference row naming any of `names` as one of its drugs — by its
    British name, the US name in brackets, or the lead drug of a "with" combo
    (aspirin has an analgesic row and a low-dose antiplatelet row)."""
    wanted = {n.casefold().strip() for n in names if n and n.strip()}
    hits = []
    for e in entries():
        for part in _split_top_level(e["drug"]):
            bare = re.sub(r"\([^)]*\)", "", part).strip().casefold()
            lead = re.split(r"\s+with\b", bare)[0].strip()
            alt = [a.strip().casefold() for a in re.findall(r"\(([^)]*)\)", part)]
            if bare in wanted or lead in wanted or any(a in wanted for a in alt):
                hits.append(e)
                break
    return hits


def find_entry(*names: str):
    """The first reference row naming any of `names`, if any."""
    hits = find_entries(*names)
    return hits[0] if hits else None

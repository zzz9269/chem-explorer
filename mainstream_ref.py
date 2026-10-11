"""The Mainstream Drug Reference (3rd ed.) — the user's own study table of the
most-used drugs, parsed from the PDF by build_mainstream_reference.py into
mainstream_reference.json. No Streamlit here; chem_v4 renders it.

Each entry: drug, tag (both/who/top300/None), section, section_no, subsection,
class, mechanism, uses, cautions, id. Some rows group several drugs that share
a mechanism ("Enalapril / perindopril / captopril"); `drug_links()` splits
those so each name can open its own Explore page.
"""
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


def search(query: str, rows=None) -> list:
    """Entries whose text contains every word of `query` (case-insensitive),
    drug-name matches first."""
    words = query.casefold().split()
    if not words:
        return list(rows if rows is not None else entries())
    hits = []
    for e in rows if rows is not None else entries():
        hay = " ".join((e["drug"], e["class"], e["mechanism"], e["uses"],
                        e["cautions"], e["subsection"] or "")).casefold()
        if all(w in hay for w in words):
            in_name = all(w in e["drug"].casefold() for w in words)
            hits.append((0 if in_name else 1, e["id"], e))
    return [e for _, _, e in sorted(hits, key=lambda h: h[:2])]


def find_entry(*names: str):
    """The reference row naming any of `names` as one of its drugs, if any."""
    wanted = {n.casefold().strip() for n in names if n}
    for e in entries():
        for part in _split_top_level(e["drug"]):
            bare = re.sub(r"\([^)]*\)", "", part).strip().casefold()
            alt = re.findall(r"\(([^)]*)\)", part)
            if bare in wanted or any(a.strip().casefold() in wanted for a in alt):
                return e
    return None

"""drug_library.py — a large, classification-organised library of medicinal drugs
used in humans.

The data is **generated**, not hand-typed: `build_drug_library.py` pulls it from
the U.S. National Library of Medicine RxNorm/RxClass API (public domain) into
`drug_library_data.json`. See that script for the sourcing strategy. This module
just loads the JSON, layers my hand-written one-line blurbs on top where names
overlap (`curated_blurbs.json`), and exposes a clean accessor API for the app.

Key properties (same philosophy as before, now at scale):
* **Every drug carries a LIST of classes** across three systems — WHO ATC
  (hierarchical), FDA Established Pharmacologic Class (EPC) and FDA Mechanism of
  Action (MOA) — so overlaps are first-class (aspirin is an analgesic AND an
  antiplatelet AND an antipyretic, each a real class).
* **No replication**: one row per RxNorm ingredient (RXCUI); synonyms, brands,
  salt forms and combinations were collapsed at build time.
* **No chemical identifiers stored** — a drug's structure/record is resolved live
  by the app's existing PubChem->OPSIN->CIR pipeline from its name.

Classes: ATC codes (e.g. "C10AA") plus MED-RT NUIs for EPC/MOA (e.g.
"N0000175722"). ATC classes carry level (1-4) and a `parent` so the hierarchy is
navigable; EPC/MOA are flat. Categories are the 14 ATC anatomical main groups
followed by the two FDA class systems.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import random as _random
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_HERE, "drug_library_data.json")
_BLURB_PATH = os.path.join(_HERE, "curated_blurbs.json")


def _load_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return default


_RAW = _load_json(_DATA_PATH, {"meta": {}, "categories": [], "classes": {}, "drugs": []})
_CURATED_BLURBS: Dict[str, str] = _load_json(_BLURB_PATH, {})

META: Dict = _RAW.get("meta", {})
CATEGORIES: List[str] = _RAW.get("categories", [])
CLASSES: Dict[str, Dict] = _RAW.get("classes", {})


def _display(name: str) -> str:
    """RxNorm ingredient names are lower-case; capitalise the first letter only
    (so 'atorvastatin' -> 'Atorvastatin' but 'ferrous sulfate' stays sensible)."""
    return name[:1].upper() + name[1:] if name else name


# Build the DRUGS list, attaching curated blurbs by name overlay.
DRUGS: List[Dict] = []
for _d in _RAW.get("drugs", []):
    _raw_name = _d["name"]
    _blurb = _CURATED_BLURBS.get(_raw_name.lower(), "")
    DRUGS.append({
        "name": _display(_raw_name),
        "query": _raw_name,          # lower-case form resolves cleanly in PubChem
        "rxcui": _d.get("rxcui"),
        "classes": _d.get("classes", []),
        "blurb": _blurb,
    })
DRUGS.sort(key=lambda d: d["name"].lower())


# ---------------------------------------------------------------------------
# Indices (built once)
# ---------------------------------------------------------------------------
_DRUGS_BY_CLASS: Dict[str, List[Dict]] = defaultdict(list)   # leaf slug -> drugs
_NAME_INDEX: Dict[str, Dict] = {}
for _d in DRUGS:
    _NAME_INDEX[_d["name"].lower()] = _d
    _NAME_INDEX.setdefault(_d["query"].lower(), _d)
    for _slug in _d["classes"]:
        _DRUGS_BY_CLASS[_slug].append(_d)

# Leaf classes = those a drug is directly tagged with (ATC level-4, EPC, MOA).
_LEAF_SLUGS = set(_DRUGS_BY_CLASS.keys())

# category -> ordered list of leaf slugs in it
_CATEGORY_LEAVES: Dict[str, List[str]] = defaultdict(list)
for _slug in _LEAF_SLUGS:
    _meta = CLASSES.get(_slug)
    if _meta:
        _CATEGORY_LEAVES[_meta["category"]].append(_slug)


def _leaf_sort_key(slug: str):
    m = CLASSES.get(slug, {})
    # ATC leaves sort by code (keeps hierarchy order); flat systems by name
    if m.get("system") == "ATC":
        return (0, slug)
    return (1, m.get("name", slug).lower())


for _cat in _CATEGORY_LEAVES:
    _CATEGORY_LEAVES[_cat].sort(key=_leaf_sort_key)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------
def class_name(slug: str) -> str:
    m = CLASSES.get(slug)
    return m["name"] if m else slug


def class_system(slug: str) -> str:
    m = CLASSES.get(slug)
    return m.get("system", "") if m else ""


def category_of(slug: str) -> Optional[str]:
    m = CLASSES.get(slug)
    return m["category"] if m else None


def class_desc(slug: str) -> str:
    """Human-readable description: for ATC, the full labelled hierarchy path;
    for EPC/MOA, the class system it comes from."""
    m = CLASSES.get(slug)
    if not m:
        return ""
    if m.get("system") == "ATC":
        # walk ancestors by code prefix for a readable path
        parts = []
        for code in (slug[:1], slug[:3], slug[:4], slug[:5]):
            if code and code in CLASSES and code != slug:
                parts.append(CLASSES[code]["name"])
        path = " › ".join(parts)
        return f"ATC {slug}" + (f" — {path}" if path else "")
    label = {"EPC": "FDA Established Pharmacologic Class",
             "MOA": "FDA Mechanism of Action"}.get(m.get("system"), m.get("system", ""))
    return label


def subgroup_name(slug: str) -> Optional[str]:
    """Coarser grouping label for a leaf class (used for sub-headers in the UI).
    For ATC level-4 this is its ATC level-2 therapeutic subgroup; else None."""
    m = CLASSES.get(slug)
    if m and m.get("system") == "ATC":
        l2 = slug[:3]
        if l2 in CLASSES:
            return CLASSES[l2]["name"]
    return None


def classes_in_category(category: str) -> List[str]:
    """Leaf class slugs in a category, in taxonomy order."""
    return _CATEGORY_LEAVES.get(category, [])


def drugs_in_class(slug: str) -> List[Dict]:
    """Drugs tagged with a class. Works for leaf classes and for ATC parent
    classes (a parent gathers every drug under its descendant ATC codes)."""
    if slug in _DRUGS_BY_CLASS and CLASSES.get(slug, {}).get("system") != "ATC":
        return sorted(_DRUGS_BY_CLASS[slug], key=lambda d: d["name"].lower())
    m = CLASSES.get(slug)
    if m and m.get("system") == "ATC" and (m.get("level") or 5) < 4:
        # non-leaf ATC: match by code prefix over ATC leaves
        seen = {}
        for leaf in _LEAF_SLUGS:
            lm = CLASSES.get(leaf, {})
            if lm.get("system") == "ATC" and leaf.startswith(slug):
                for d in _DRUGS_BY_CLASS[leaf]:
                    seen[d["name"]] = d
        return sorted(seen.values(), key=lambda d: d["name"].lower())
    return sorted(_DRUGS_BY_CLASS.get(slug, []), key=lambda d: d["name"].lower())


@lru_cache(maxsize=1)
def class_counts() -> Dict[str, int]:
    return {slug: len(drugs) for slug, drugs in _DRUGS_BY_CLASS.items()}


@lru_cache(maxsize=1)
def category_counts() -> Dict[str, int]:
    counts: Dict[str, set] = defaultdict(set)
    for d in DRUGS:
        for slug in d["classes"]:
            cat = category_of(slug)
            if cat:
                counts[cat].add(d["name"])
    return {cat: len(names) for cat, names in counts.items()}


def classes_of(drug: Dict) -> List[str]:
    return drug.get("classes", [])


def search_drugs(text: str, limit: int = 400) -> List[Dict]:
    """Case-insensitive search over drug name, blurb and class names. Name matches
    rank first, then class-name matches, then blurb matches. Capped for the UI."""
    q = (text or "").strip().lower()
    if not q:
        return []
    name_hits, class_hits, blurb_hits = [], [], []
    for d in DRUGS:
        if q in d["name"].lower():
            name_hits.append(d)
            continue
        if any(q in class_name(s).lower() for s in d["classes"]):
            class_hits.append(d)
            continue
        if d["blurb"] and q in d["blurb"].lower():
            blurb_hits.append(d)
    key = lambda d: d["name"].lower()
    ordered = sorted(name_hits, key=key) + sorted(class_hits, key=key) + sorted(blurb_hits, key=key)
    return ordered[:limit]


def find_drug(name: str) -> Optional[Dict]:
    """Exact (case-insensitive) lookup by display name or query name."""
    if not name:
        return None
    return _NAME_INDEX.get(name.strip().lower())


def is_available() -> bool:
    """False if the generated data file is missing/empty."""
    return len(DRUGS) > 0


# ---------------------------------------------------------------------------
# Drug of the Day
# ---------------------------------------------------------------------------
# The pick is a *deterministic function of the date* (like a word-of-the-day),
# not a runtime random choice — so it is stable across reruns within a day and,
# crucially, the archive for any past date is simply this function replayed (no
# storage needed). A fixed-seed permutation of the whole library, indexed by the
# date's ordinal, makes the sequence look random yet **never repeats** until all
# drugs have been featured (~len(DRUGS) days ≈ 10 years).
_DOTD_SEED = 0x0DDBA11  # fixed so the sequence is reproducible forever

# The library includes vaccines, allergen extracts and other biologicals that
# make poor "featured" drugs and don't resolve to a single chemical structure in
# the PubChem/RDKit pipeline. Exclude them from the Drug-of-the-Day pool (they
# remain fully browsable in the library itself).
import re as _re
_NON_FEATUREABLE_RE = _re.compile(
    r"\b(vaccine|toxoid|antitoxin|antivenin|antivenom|venom|allergenic|allergen|"
    r"extract|pollen|globulin|antiserum|conjugate|polysaccharide|tuberculin|"
    r"spore|mould|mold|dander|epitheli|antigen|bacterin|strain|serum|plasma|"
    r"whole cell|purified protein)\b", _re.I)
# ATC vaccine (J07) and allergen (V01) subgroups.
_NON_FEATUREABLE_ATC = ("J07", "V01")
# Name suffixes of biologics (antibodies, fusion proteins, enzymes, cell/gene
# therapies) that have no single small-molecule structure to display.
_BIOLOGIC_SUFFIXES = ("mab", "cept", "leucel", "cabtagene", "parvovec",
                      "grastim", "poetin", "ase", "kinase")


def _is_featureable(d: Dict) -> bool:
    name = d["name"]
    if _NON_FEATUREABLE_RE.search(name):
        return False
    if any(tok.endswith(_BIOLOGIC_SUFFIXES) for tok in name.lower().split()):
        return False
    for slug in d["classes"]:
        if slug[:3] in _NON_FEATUREABLE_ATC and class_system(slug) == "ATC":
            return False
    return True


@lru_cache(maxsize=1)
def _dotd_pool() -> List[Dict]:
    pool = [d for d in DRUGS if _is_featureable(d)]
    return pool or list(DRUGS)  # never empty


@lru_cache(maxsize=1)
def _dotd_permutation() -> List[int]:
    idx = list(range(len(_dotd_pool())))
    _random.Random(_DOTD_SEED).shuffle(idx)
    return idx


def drug_of_the_day(day: Optional[_dt.date] = None) -> Optional[Dict]:
    """The featured drug for a given date (default: today). Deterministic and
    repeat-free across consecutive dates; drawn from featureable small-molecule
    drugs only."""
    if not DRUGS:
        return None
    day = day or _dt.date.today()
    pool = _dotd_pool()
    perm = _dotd_permutation()
    return pool[perm[day.toordinal() % len(perm)]]


def drug_of_the_day_archive(days: int = 14,
                            end_date: Optional[_dt.date] = None
                            ) -> List[Tuple[_dt.date, Dict]]:
    """The featured drugs for the `days` days *before* end_date (default today),
    most recent first — i.e. the Drug-of-the-Day history."""
    if not DRUGS:
        return []
    end = end_date or _dt.date.today()
    out: List[Tuple[_dt.date, Dict]] = []
    for i in range(1, days + 1):
        d = end - _dt.timedelta(days=i)
        out.append((d, drug_of_the_day(d)))
    return out


def duplicate_report() -> Dict[str, list]:
    """Return any duplicate identities in the catalogue. De-duplication is a hard
    requirement, so this is the guard: a drug is uniquely identified by its RxNorm
    RXCUI, and separately no two rows may share a display name (case-insensitive).
    Genuinely different salts (ferrous sulfate vs ferrous fumarate) are DISTINCT
    drugs and are intentionally NOT flagged."""
    by_rxcui: Dict[str, list] = defaultdict(list)
    by_name: Dict[str, list] = defaultdict(list)
    for d in DRUGS:
        if d.get("rxcui"):
            by_rxcui[d["rxcui"]].append(d["name"])
        by_name[d["name"].strip().lower()].append(d.get("rxcui"))
    return {
        "duplicate_rxcui": {k: v for k, v in by_rxcui.items() if len(v) > 1},
        "duplicate_name": {k: v for k, v in by_name.items() if len(v) > 1},
    }


def validate() -> None:
    problems: List[str] = []
    for d in DRUGS:
        if not d["classes"]:
            problems.append(f"{d['name']}: no classes")
        for slug in d["classes"]:
            if slug not in CLASSES:
                problems.append(f"{d['name']}: unknown class {slug!r}")
    dups = duplicate_report()
    for rxcui, names in dups["duplicate_rxcui"].items():
        problems.append(f"duplicate RXCUI {rxcui}: {names}")
    for name, rxcuis in dups["duplicate_name"].items():
        problems.append(f"duplicate drug name {name!r}: rxcuis {rxcuis}")
    if problems:
        raise ValueError("drug_library integrity errors:\n  " + "\n  ".join(problems[:50]))


if is_available():
    validate()


if __name__ == "__main__":
    print(f"{len(DRUGS)} drugs · {len(_LEAF_SLUGS)} leaf classes · "
          f"{len(CLASSES)} total classes · {len(CATEGORIES)} categories")
    print("source:", META.get("source"))
    cc = class_counts()
    for cat in CATEGORIES:
        print(f"\n## {cat}  ({category_counts().get(cat, 0)} drugs, "
              f"{len(classes_in_category(cat))} classes)")
        for slug in classes_in_category(cat)[:6]:
            print(f"   {class_name(slug):45s} {cc.get(slug, 0):4d}")
    multi = sum(1 for d in DRUGS if len(d["classes"]) > 1)
    with_blurb = sum(1 for d in DRUGS if d["blurb"])
    print(f"\n{multi}/{len(DRUGS)} drugs span >1 class · {with_blurb} have curated blurbs")

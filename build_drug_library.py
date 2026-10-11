#!/usr/bin/env python3
"""build_drug_library.py — generate the drug catalogue from NLM RxNorm/RxClass.

Data source: U.S. National Library of Medicine RxNav (RxNorm + RxClass) REST API,
which is in the **public domain** (U.S. Government work) and freely redistributable.
    https://rxnav.nlm.nih.gov/RxClassAPIs.html

Strategy
--------
Drug *classes* come from three complementary, authoritative systems (a drug may
belong to many across all of them — overlaps are the whole point):

  * ATC (WHO Anatomical Therapeutic Chemical) — the hierarchical backbone: 14
    anatomical main groups -> therapeutic -> pharmacological -> chemical subgroup.
  * EPC (FDA Established Pharmacologic Class) — clean pharmacologic classes.
  * MOA (FDA Mechanism of Action).

For each class we ask RxClass for its member drugs and keep only tty == "IN"
(RxNorm *ingredient* concepts). This is what solves de-duplication: every synonym
and brand of a substance collapses to a single ingredient RXCUI, and salt/precise
forms (PIN) and combination products (MIN) are excluded — so one drug = one row,
no replication.

Output: drug_library_data.json  (loaded by drug_library.py).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import requests

try:  # keep console output ASCII-safe on Windows cp1252 terminals
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

API = "https://rxnav.nlm.nih.gov/REST/rxclass"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drug_library_data.json")

# ATC level-1 main groups, in canonical WHO order, with display names.
ATC_L1_ORDER = [
    ("A", "Alimentary tract & metabolism"),
    ("B", "Blood & blood-forming organs"),
    ("C", "Cardiovascular system"),
    ("D", "Dermatologicals"),
    ("G", "Genito-urinary system & sex hormones"),
    ("H", "Systemic hormonal preparations (excl. sex hormones)"),
    ("J", "Anti-infectives for systemic use"),
    ("L", "Antineoplastic & immunomodulating agents"),
    ("M", "Musculo-skeletal system"),
    ("N", "Nervous system"),
    ("P", "Antiparasitic products, insecticides & repellents"),
    ("R", "Respiratory system"),
    ("S", "Sensory organs"),
    ("V", "Various"),
]
ATC_L1_NAME = dict(ATC_L1_ORDER)

EPC_CATEGORY = "Pharmacologic class (FDA EPC)"
MOA_CATEGORY = "Mechanism of action (FDA MOA)"

_session = requests.Session()
_session.headers.update({"User-Agent": "ChemExplorer-drug-library-builder/1.0"})


def _get(path: str, **params) -> dict:
    """GET with a few retries; returns {} on persistent failure."""
    url = f"{API}/{path}"
    for attempt in range(4):
        try:
            r = _session.get(url, params=params, timeout=45)
            if r.status_code == 200 and r.text.strip():
                return r.json()
            if r.status_code == 404:
                return {}
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"  ! {path} {params} failed: {e}", file=sys.stderr)
        time.sleep(0.5 * (attempt + 1))
    return {}


def all_classes(class_type: str) -> List[dict]:
    j = _get("allClasses.json", classTypes=class_type)
    return j.get("rxclassMinConceptList", {}).get("rxclassMinConcept", [])


def class_members(class_id: str, rela_source: str) -> List[dict]:
    j = _get("classMembers.json", classId=class_id, relaSource=rela_source)
    out = []
    for m in j.get("drugMemberGroup", {}).get("drugMember", []):
        mc = m.get("minConcept", {})
        if mc.get("tty") == "IN":  # ingredient grain only -> natural de-dup
            out.append((mc["rxcui"], mc["name"]))
    return out


def build() -> dict:
    t0 = time.time()
    classes: Dict[str, dict] = {}          # slug -> {name, system, level, parent, category}
    drug_name: Dict[str, str] = {}         # rxcui -> canonical ingredient name
    drug_classes: Dict[str, set] = defaultdict(set)  # rxcui -> {slug}

    # ---- ATC hierarchy (names for every level, so ancestors are labelled) ----
    print("Fetching ATC class hierarchy...")
    atc_all = all_classes("ATC1-4")
    atc_name = {c["classId"]: c["className"] for c in atc_all if c["classId"] != "0"}
    atc4 = [c["classId"] for c in atc_all if len(c["classId"]) == 5]

    def atc_ancestors(code4: str) -> List[str]:
        # ATC4 code e.g. 'C10AA' -> ancestors L1..L4
        return [code4[:1], code4[:3], code4[:4], code4[:5]]

    def register_atc(code4: str):
        l1 = code4[:1]
        cat = ATC_L1_NAME.get(l1, atc_name.get(l1, l1))
        for lvl, code in enumerate(atc_ancestors(code4), start=1):
            if code in classes:
                continue
            parent = {1: None, 2: code[:1], 3: code[:3], 4: code[:4]}[lvl]
            classes[code] = {
                "name": atc_name.get(code, code),
                "system": "ATC", "level": lvl, "parent": parent, "category": cat,
            }

    print(f"  {len(atc4)} ATC level-4 classes -> fetching members...")
    for i, code in enumerate(atc4, 1):
        register_atc(code)
        for rxcui, name in class_members(code, "ATC"):
            drug_name.setdefault(rxcui, name)
            drug_classes[rxcui].add(code)          # tag at ATC4 (chemical subgroup)
        if i % 100 == 0:
            print(f"    ATC {i}/{len(atc4)} | drugs so far {len(drug_name)}")

    # ---- FDA pharmacologic systems (flat classes) ----
    for ctype, category in (("EPC", EPC_CATEGORY), ("MOA", MOA_CATEGORY)):
        cls = all_classes(ctype)
        print(f"  {len(cls)} {ctype} classes -> fetching members...")
        for i, c in enumerate(cls, 1):
            slug = c["classId"]
            members = class_members(slug, "DAILYMED")
            if not members:
                continue
            classes.setdefault(slug, {
                "name": c["className"], "system": ctype, "level": None,
                "parent": None, "category": category,
            })
            for rxcui, name in members:
                drug_name.setdefault(rxcui, name)
                drug_classes[rxcui].add(slug)
            if i % 150 == 0:
                print(f"    {ctype} {i}/{len(cls)} | drugs so far {len(drug_name)}")

    # ---- assemble ----
    drugs = []
    for rxcui, name in drug_name.items():
        slugs = sorted(drug_classes[rxcui])
        if not slugs:
            continue
        drugs.append({"name": name, "rxcui": rxcui, "classes": slugs})
    drugs.sort(key=lambda d: d["name"].lower())

    # category ordering: 14 ATC groups (only those present) then FDA groups
    present_cats = {c["category"] for c in classes.values()}
    categories = [name for _, name in ATC_L1_ORDER if name in present_cats]
    for extra in (EPC_CATEGORY, MOA_CATEGORY):
        if extra in present_cats:
            categories.append(extra)

    data = {
        "meta": {
            "source": "NLM RxNorm/RxClass (public domain)",
            "source_url": "https://rxnav.nlm.nih.gov/RxClassAPIs.html",
            "systems": ["ATC (WHO)", "EPC (FDA)", "MOA (FDA)"],
            "grain": "RxNorm ingredient (tty=IN); synonyms/salts/combinations de-duplicated by RXCUI",
            "n_drugs": len(drugs),
            "n_classes": len(classes),
            "n_categories": len(categories),
            "built_seconds": round(time.time() - t0, 1),
        },
        "categories": categories,
        "classes": classes,
        "drugs": drugs,
    }
    return data


if __name__ == "__main__":
    data = build()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    m = data["meta"]
    print(f"\nWrote {OUT}")
    print(f"  {m['n_drugs']} drugs | {m['n_classes']} classes | "
          f"{m['n_categories']} categories | {m['built_seconds']}s")

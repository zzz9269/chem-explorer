"""sg_register.py — Singapore availability of a drug, from the HSA register.

Loads the snapshot written by `build_hsa_register.py` (HSA "Listing of
Registered Therapeutic Products", data.gov.sg, Singapore Open Data Licence v1.0)
and answers: is this ingredient sold in Singapore, under which brands, and is it
Prescription-Only (POM), Pharmacy-Only (P) or General Sale List (GSL)?

Matching is by whole-word ingredient name inside each product's active-
ingredient strings ('Atorvastatin Calcium 10.34mg eqv Atorvastatin'), so
'loratadine' does not match desloratadine. US (USAN) names are mapped to the
INN/BAN names Singapore uses (acetaminophen → paracetamol, …). No Streamlit,
no network.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_HERE, "hsa_register_data.json")

try:
    with open(_DATA_PATH, encoding="utf-8") as _fh:
        _RAW = json.load(_fh)
except Exception:  # noqa: BLE001
    _RAW = {"meta": {}, "products": []}

META: Dict = _RAW.get("meta", {})
PRODUCTS: List[Dict] = _RAW.get("products", [])
for _p in PRODUCTS:
    _p["_ings_lc"] = [s.lower() for s in _p.get("ings", [])]

CLASS_LABEL = {"POM": "Prescription-only", "P": "Pharmacy-only", "GSL": "General sale"}
CLASS_HELP = {
    "POM": "needs a doctor's prescription",
    "P": "sold by a pharmacist without prescription",
    "GSL": "sold off the shelf by any retailer",
}
# least → most restrictive, for display order
CLASS_ORDER = ("GSL", "P", "POM")

# US adopted name → name(s) used on Singapore labels (INN / older BAN spellings).
_ALIASES = {
    "acetaminophen": ["paracetamol"],
    "albuterol": ["salbutamol"],
    "levalbuterol": ["levosalbutamol"],
    "epinephrine": ["adrenaline"],
    "norepinephrine": ["noradrenaline"],
    "meperidine": ["pethidine"],
    "glyburide": ["glibenclamide"],
    "cyclosporine": ["ciclosporin", "cyclosporin"],
    "lidocaine": ["lignocaine"],
    "rifampin": ["rifampicin"],
    "isoproterenol": ["isoprenaline"],
    "phenobarbital": ["phenobarbitone"],
    "cephalexin": ["cefalexin"],
    "cefuroxime": ["cefuroxime"],
    "amoxicillin": ["amoxycillin"],
    "dicyclomine": ["dicycloverine"],
    "guaifenesin": ["guaiphenesin"],
    "chlorpheniramine": ["chlorphenamine"],
    "dexchlorpheniramine": ["dexchlorphenamine"],
    "scopolamine": ["hyoscine"],
    "mesalamine": ["mesalazine"],
    "simethicone": ["simeticone"],
    "estradiol": ["oestradiol"],
    "sulfasalazine": ["sulphasalazine"],
    "sulfamethoxazole": ["sulphamethoxazole"],
    "beclomethasone": ["beclometasone"],
    "valproic acid": ["valproate", "valproic acid"],
    "nitroglycerin": ["glyceryl trinitrate"],
    "leuprolide": ["leuprorelin"],
    "cromolyn": ["cromoglicate", "cromoglycate"],
    "clomiphene": ["clomifene"],
    "furosemide": ["frusemide"],
    "methylene blue": ["methylthioninium"],
    "thioguanine": ["tioguanine"],
    "aspirin": ["acetylsalicylic acid"],
    "acetylsalicylic acid": ["aspirin"],
}


def _variants(name: str) -> List[str]:
    n = (name or "").strip().lower()
    if not n:
        return []
    out = [n] + _ALIASES.get(n, [])
    # reverse direction too (paracetamol → acetaminophen is pointless for SG
    # labels, but some SG labels use the US name, e.g. "Paracetamol (Acetaminophen)")
    out += [us for us, sg in _ALIASES.items() if n in sg]
    return list(dict.fromkeys(out))


def _word_re(name: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])")


def products_for(name: str) -> List[Dict]:
    """Every registered product with `name` (or a SG alias) as an active ingredient."""
    pats = [_word_re(v) for v in _variants(name)]
    if not pats:
        return []
    return [p for p in PRODUCTS if any(pat.search(s) for s in p["_ings_lc"] for pat in pats)]


# Words that end the brand part of a product name ('NUROFEN FOR CHILDREN 200MG/5ML').
_FORM_WORDS = (
    r"tablets?|caplets?|capsules?|cap|tab|injections?|inj|solution|suspension|syrup|"
    r"cream|ointment|gel|drops?|eye|ear|nasal|suppositor(?:y|ies)|film|f\.?c\.?|coated|"
    r"for|oral|powder|inhaler|inhalation|spray|patch(?:es)?|lotion|chewable|"
    r"effervescent|granules?|sachets?|elixir|emulsion|infusion|concentrate|lozenges?|"
    r"soft|softgel|gelatin|extended|modified|sustained|controlled|prolonged|delayed|"
    r"release|dispersible|orodispersible|enteric|liquid|paediatric|pediatric|"
    r"forte|plus|retard|mr|sr|er|xr|xl|cr|la|bp|usp|dt|odt|ec|mg|mcg|g|ml|%|"
    r"gastro-resistant|hard|hfa|cfc-free|aerosol|metered|evohaler|accuhaler|"
    r"turbuhaler|easyhaler|respirator|mixture|linctus|vial|ampoule|pre-filled|fc"
)
_BRAND_END_RE = re.compile(r"\s+(?:\(|\d|(?:" + _FORM_WORDS + r")\b)", re.I)


def brand_of(product_name: str) -> str:
    """'SUN-DIANOX TABLET' → 'Sun-Dianox'; 'LIPITOR TABLET 20 mg' → 'Lipitor'."""
    s = (product_name or "").strip()
    m = _BRAND_END_RE.search(s)
    brand = (s[:m.start()] if m else s).strip(" -,'\"")
    brand = re.sub(r"-\d+$", "", brand)          # 'PANAMOL-500' → 'PANAMOL'
    if not brand:
        return ""
    if brand.isupper():   # 'SUN-DIANOX' → 'Sun-Dianox'; mixed case is kept as given
        brand = " ".join("-".join(part.capitalize() for part in w.split("-"))
                         for w in brand.split(" "))
    return brand


def _summarise(prods: List[Dict], name: str, known_brands=()) -> Dict:
    single = [p for p in prods if len(p["ings"]) == 1]
    basis = single or prods
    counts = Counter(p["cls"] for p in basis)
    classes = [c for c in CLASS_ORDER if counts.get(c)]

    # co-ingredients when only sold in combination (diphenoxylate → atropine)
    partners: Counter = Counter()
    if not single:
        pats = [_word_re(v) for v in _variants(name)]
        for p in prods:
            for s in p["ings"]:
                if not any(pat.search(s.lower()) for pat in pats):
                    partners[_clean_ing(s)] += 1

    combo_counts = Counter(p["cls"] for p in prods if len(p["ings"]) > 1) if single else Counter()

    return {
        "name": name,
        "n_products": len(prods),
        "n_single": len(single),
        "combination_only": not single,
        "class_counts": dict(counts),          # over single-ingredient products if any
        "classes": classes,                    # least → most restrictive
        "combo_class_counts": dict(combo_counts),
        "partners": [p for p, _ in partners.most_common(3)],
        "brands": _rank_brands(prods, name, known_brands),
        "products": sorted(prods, key=lambda p: (CLASS_ORDER.index(p["cls"]), p["name"])),
    }


_SALT_WORDS_RE = re.compile(
    r"\b(?:hydrochloride|hcl|sodium|potassium|calcium|magnesium|sulfate|sulphate|"
    r"maleate|mesylate|besylate|tartrate|citrate|acetate|phosphate|succinate|"
    r"fumarate|bromide|hydrobromide|trihydrate|monohydrate|dihydrate|anhydrous)\b")


def _approval_key(s: str):
    """'5/4/1988' (d/m/y) → (1988, 4, 5); unknown dates sort last."""
    try:
        d, m, y = (int(x) for x in s.split("/"))
        return (y, m, d)
    except Exception:  # noqa: BLE001
        return (9999, 0, 0)


def _rank_brands(prods: List[Dict], name: str, known=()) -> List[str]:
    """Brand names, best-known first. Variants sharing a first word collapse to
    the family ('Panadol Extra', 'Panadol Actifast' → 'Panadol'). Ranked:
    names that aren't just generic+maker ('Apo-Atorvastatin') first; then
    brands also known internationally (`known`, e.g. RxNorm's Lipitor); then
    single-ingredient brands; then most registered products (big consumer
    brands carry many SKUs); then earliest HSA registration (originators
    usually registered first)."""
    gens = _variants(name)
    pats = [_word_re(v) for v in gens]
    known_lc = {k.lower() for k in known or ()}
    raw: Dict[str, Dict] = {}
    for p in prods:
        b = brand_of(p["name"])
        if not b or b.lower() in gens:
            continue
        # 'Atropine Sulphate' is the generic + its salt, not a brand
        rest = b.lower()
        for pat in pats:
            rest = pat.sub("", rest)
        if not _SALT_WORDS_RE.sub("", rest).strip(" -,."):
            continue
        r = raw.setdefault(b, {"single": False, "date": (9999, 0, 0), "n": 0})
        r["single"] |= len(p["ings"]) == 1
        r["date"] = min(r["date"], _approval_key(p.get("approved", "")))
        r["n"] += 1

    first = Counter(b.split()[0].lower() for b in raw)
    fam: Dict[str, Dict] = {}
    for b, r in raw.items():
        head = b.split()[0]
        key = head if first[head.lower()] > 1 and len(head) >= 4 else b
        f = fam.setdefault(key, {"single": False, "date": (9999, 0, 0), "n": 0})
        f["single"] |= r["single"]
        f["date"] = min(f["date"], r["date"])
        f["n"] += r["n"]

    def score(item):
        b, f = item
        generic_named = any(pat.search(b.lower()) for pat in pats)
        return (generic_named, b.lower() not in known_lc, not f["single"], -f["n"], f["date"])

    return [b for b, _ in sorted(fam.items(), key=score)]


def _clean_ing(s: str) -> str:
    s = re.sub(r"\(.*?\)|\d.*$", "", s).strip()
    s = re.sub(r"\s+(?:eqv|equivalent).*$", "", s, flags=re.I)
    return s.lower().strip(" ,")


def lookup(*names: Optional[str], known_brands=()) -> Dict:
    """Singapore summary for the first name that matches a registered product.
    `known_brands` (e.g. RxNorm brands) lifts internationally known names to
    the front of the brand list. Returns {} when none of the names is an
    HSA-registered active ingredient."""
    seen = set()
    for n in names:
        key = (n or "").strip().lower()
        if not key or key in seen or len(key) < 4:
            continue
        seen.add(key)
        prods = products_for(key)
        if prods:
            return _summarise(prods, key, known_brands)
    return {}


def main_route(*names: Optional[str]) -> str:
    """The route most Singapore products of the drug share ('ORAL', 'TOPICAL',
    'OPHTHALMIC'… — the same vocabulary as FDA labels), counting single-
    ingredient products when there are any. '' when the drug isn't registered.
    Lets the US label chosen match the form actually sold here (diclofenac is
    mostly tablets in Singapore but mostly gels on US labels)."""
    seen = set()
    for n in names:
        key = (n or "").strip().lower()
        if not key or key in seen or len(key) < 4:
            continue
        seen.add(key)
        prods = products_for(key)
        if prods:
            single = [p for p in prods if len(p["ings"]) == 1] or prods
            routes = Counter((p.get("route") or "").split("&&")[0].strip().upper()
                             for p in single)
            routes.pop("", None)
            return routes.most_common(1)[0][0] if routes else ""
    return ""


def brand_ingredients(brand: str) -> List[str]:
    """Active ingredient(s) of a Singapore brand ('Panadol' → ['paracetamol'],
    'Dhamotil' → ['atropine sulphate', 'diphenoxylate hcl']). Matches the brand
    part of product names; prefers the plain product (fewest ingredients) so
    'Panadol' gives paracetamol, not Panadol Extra's paracetamol + caffeine."""
    q = re.sub(r"[\s\-]+", " ", (brand or "").strip().lower())
    if len(q) < 3:
        return []
    hits = []
    for p in PRODUCTS:
        b = re.sub(r"[\s\-]+", " ", brand_of(p["name"]).lower())
        if b == q or b.split(" ")[0] == q:
            hits.append(p)
    if not hits:
        return []
    exact = [p for p in hits if re.sub(r"[\s\-]+", " ", brand_of(p["name"]).lower()) == q]
    best = min(exact or hits, key=lambda p: len(p["ings"]))
    return list(dict.fromkeys(_clean_ing(s) for s in best["ings"] if _clean_ing(s)))


def availability_text(summary: Dict) -> str:
    """One line for the At-a-Glance card, e.g. 'Pharmacy-only (P) — only in
    combination with atropine' or 'Varies by product: General sale ×9 · …'."""
    if not summary:
        return ""
    cc = summary["class_counts"]
    classes = summary["classes"]
    if len(classes) == 1:
        c = classes[0]
        txt = f"{CLASS_LABEL[c]} ({c}) — {CLASS_HELP[c]}"
    else:
        txt = "Varies by product/strength: " + " · ".join(
            f"{CLASS_LABEL[c]} ({c}) ×{cc[c]}" for c in classes)
    if summary["combination_only"]:
        with_ = ", ".join(summary["partners"])
        txt += f"; only sold in combination{' with ' + with_ if with_ else ''}"
    else:
        # combination products can sit in a different class (caffeine alone is
        # POM, but Panadol Extra is GSL) — say so rather than hide it
        cc2 = summary.get("combo_class_counts") or {}
        extra = [c for c in CLASS_ORDER if cc2.get(c) and c not in classes]
        if extra:
            txt += "; in combination products also " + " · ".join(
                f"{CLASS_LABEL[c]} ({c}) ×{cc2[c]}" for c in extra)
    return txt


def is_available() -> bool:
    return bool(PRODUCTS)

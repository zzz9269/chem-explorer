#!/usr/bin/env python3
"""
Chemical Explorer — PubChem + RDKit (Streamlit)

A name/SMILES/drawn-structure explorer that resolves chemicals through a
multi-source chain (PubChem -> OPSIN -> CIR), enriches with RDKit-computed
descriptors and drug-likeness metrics, and presents everything in a tabbed UI.

Run with:
    streamlit run chem_v4.py
"""
import os
import sys
import io
import json
import re
import threading
import time
import html as _html
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from typing import List, Dict, Any, Tuple, Optional, Sequence
from urllib.parse import quote as urlquote

import pandas as pd

# pubchempy (rich compound dict: atoms/bonds/etc.)
import pubchempy as pcp

# RDKit + quiet logger
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors, Crippen, QED
from rdkit.Chem.Draw import rdMolDraw2D

import altair as alt

# PIL for image handling
from PIL import Image

# 3D viewer
import py3Dmol

# Our network/resolution layer (no Streamlit dependency)
import chem_sources as cs

# Structure-grounded inference (RDKit only, no Streamlit) — functional groups,
# acid/base & ionization state, stability flags. Every value is tagged with its
# provenance (derived vs inferred); no statistical/black-box predictions.
import chem_inference as ci

# Curated drug library (no Streamlit, no network) — commonly-used pharmaceuticals
# tagged by therapeutic class, with overlaps. Powers the "Drug Library" browser;
# a selected drug is handed to the same resolver pipeline via _set_search().
try:
    import drug_library as dl
    HAS_DRUG_LIBRARY = True
except Exception:
    dl = None
    HAS_DRUG_LIBRARY = False

# The user's own Mainstream Drug Reference (parsed from their PDF) — a study table
# of ~700 everyday drugs; each name opens that drug's Explore page.
try:
    import mainstream_ref as mref
    HAS_MAINSTREAM_REF = mref.is_available()
except Exception:
    mref = None
    HAS_MAINSTREAM_REF = False

# Singapore HSA register snapshot (no Streamlit, no network) — local brands and
# POM / Pharmacy-only / General-sale status, used when the market is Singapore.
try:
    import sg_register as sgr
    HAS_SG_REGISTER = sgr.is_available()
except Exception:
    sgr = None
    HAS_SG_REGISTER = False

# Rule-based question parser for the Ask (chat) mode — no AI/LLM, no network.
try:
    import chat_router as cr
    HAS_CHAT = True
except Exception:
    cr = None
    HAS_CHAT = False

# Local log of user-reported data problems (no Streamlit, no network).
import data_reports as drp

# Optional Streamlit
HAS_STREAMLIT = False
try:
    import streamlit as st
    import streamlit.components.v1 as components
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False

# Optional structure-drawing component (returns SMILES directly to Python)
HAS_KETCHER = False
try:
    from streamlit_ketcher import st_ketcher
    HAS_KETCHER = True
except Exception:
    HAS_KETCHER = False

# Legacy JSME drawer fallback
try:
    from jsme_drawer import show_jsme_drawer
except Exception:
    show_jsme_drawer = None

# HTML sanitization
try:
    import bleach
    _BLEACH_AVAILABLE = True
except Exception:
    bleach = None
    _BLEACH_AVAILABLE = False

# ---------------- Config ---------------------------------------------------
MAX_LEN = 120
TOP_CARD_PREVIEW_LEN = 400
SAVE_DIR = os.path.expanduser(os.path.join("~", "chem_v_exports"))
os.makedirs(SAVE_DIR, exist_ok=True)

# Preferred display order for "General Compound Data".
PREFERRED_ORDER = [
    "iupac_name", "formula", "molecular_formula", "smiles", "molecular_weight",
    "monoisotopic_mass", "exact_mass", "charge", "tpsa", "xlogp", "clogp",
    "coordinate_type", "covalent_unit_count", "h_bond_acceptor_count",
    "h_bond_donor_count", "heavy_atom_count", "isotope_atom_count", "rotatable_bonds",
]
UNLISTED_ORDER = "alpha"  # "alpha" or "original"

EXCLUDE_GENERAL_KEYS = {
    "cid", "defined_atom_stereo_count", "defined_bond_stereo_count", "hba", "hbd",
    "undefined_atom_stereo_count", "undefined_bond_stereo_count", "inchi", "inchikey",
    "connectivity_smiles", "mol_weight", "rotatable_bond_count",
}

# Keys produced by our RDKit pass — used to tag the source in the UI.
RDKIT_KEYS = {"mol_weight", "tpsa", "clogp", "hbd", "hba", "rotatable_bonds", "formula"}

# Grouping rules
try:
    from grouping_config import GROUP_KEYWORDS, CUSTOM_GROUP_MAP, EXCLUDE_HEADINGS
except Exception as e:
    print("Warning: grouping_config import failed:", e)
    GROUP_KEYWORDS, CUSTOM_GROUP_MAP, EXCLUDE_HEADINGS = {}, {}, set()

# Which PUG-View groups belong on which tab. Every group should be mapped so
# nothing important falls into the catch-all "refs" tab.
GROUP_TO_TAB = {
    # Properties
    "Physical Properties": "props",
    "Computed Properties / Descriptors": "props",
    "Experimental Properties & Measurements": "props",
    # Safety, toxicity & environment
    "Safety & Hazards": "safety",
    "Exposure & Occupational": "safety",
    "Toxicity & Health": "safety",
    "Toxicology Programs & Studies": "safety",
    "Environmental & Fate": "safety",
    # Pharmacology & bioactivity
    "Pharmacology & Clinical": "pharma",
    "Interactions, Pathways & Bioactivity": "pharma",
    # Uses & regulatory
    "Manufacturing, Uses & Commerce": "uses",
    "Regulatory & Transport": "uses",
    "Classification, Standards & Registries": "uses",
    # Spectra / structure
    "Spectra & Analytical": "spectra",
    "Structures & Conformers": "structure",
    # References & identifiers (catch-all tab)
    "Names & Identifiers": "refs",
    "Literature & References": "refs",
    "Related Records & Vendors": "refs",
    "Patents": "refs",
    "Other": "refs",
}

# Tab key -> human label, in display order.
TAB_LABELS = [
    ("overview", "Overview"),
    ("props", "Properties"),
    ("safety", "Safety & Env"),
    ("pharma", "Pharmacology"),
    ("uses", "Uses & Regulatory"),
    ("spectra", "Spectra"),
    ("structure", "3D & Structure"),
    ("refs", "References & IDs"),
]

# Wikipedia section heading -> tab, so a full article can backfill the tabs
# PubChem left thin. Ordered: first substring match wins, so put the specific
# keywords before the generic ones. Anything unmatched lands in Overview.
WIKI_SECTION_RULES = [
    # Pharmacology  (check "mechanism"/"pharmaco*" before generic "action")
    ("mechanism of action", "pharma"), ("pharmacodynam", "pharma"),
    ("pharmacokinetic", "pharma"), ("pharmacolog", "pharma"),
    ("metabolism", "pharma"), ("absorption", "pharma"), ("excretion", "pharma"),
    ("mode of action", "pharma"), ("biosynthesis", "pharma"),
    # stub drug articles often explain how the drug works under "Function"
    # (bromhexine); route it to the mechanism heading, not Uses
    ("function", "pharma"),
    ("interaction", "pharma"),
    # Safety, toxicity & environment
    ("contraindication", "safety"), ("adverse", "safety"),
    ("side effect", "safety"), ("side-effect", "safety"), ("toxic", "safety"),
    ("overdose", "safety"), ("pregnancy", "safety"), ("lactation", "safety"),
    ("breastfeeding", "safety"), ("precaution", "safety"), ("warning", "safety"),
    ("safety", "safety"), ("carcinogen", "safety"), ("environment", "safety"),
    ("ecolog", "safety"),
    # Clinical use — kept with the pharmacology so it merges into the
    # consolidated "Uses & Indications" heading next to PubChem's own.
    ("medical use", "pharma"), ("clinical use", "pharma"), ("indication", "pharma"),
    ("available form", "pharma"), ("dosage", "pharma"), ("administration", "pharma"),
    # Uses & regulatory
    ("veterinary", "uses"), ("society and culture", "uses"), ("economic", "uses"),
    ("brand name", "uses"), ("trade name", "uses"), ("regulat", "uses"),
    ("legal status", "uses"), ("formulation", "uses"), ("agricultur", "uses"),
    ("use", "uses"),
    # Properties / chemistry
    ("chemical propert", "props"), ("physical propert", "props"),
    ("chemistry", "props"), ("synthesis", "props"), ("structure", "props"),
    ("stereochem", "props"),
    # Spectra  (analytical spectroscopy only — NOT "spectrum of susceptibility")
    ("spectroscop", "spectra"), ("nmr", "spectra"), ("mass spectr", "spectra"),
    ("infrared", "spectra"),
]


def _classify_wiki_raw(heading: str) -> Optional[str]:
    """Tab key for a heading by keyword, or None when nothing matches (so a
    sub-section can inherit its parent section's tab)."""
    h = (heading or "").strip().lower()
    for kw, tab in WIKI_SECTION_RULES:
        if kw in h:
            return tab
    return None


def classify_wiki_section(heading: str) -> str:
    """Map one Wikipedia section heading to a tab key. Unmatched background
    sections (History, Society, …) fall to 'uses' so every section lands in a
    group-based tab and can be merged into the PubChem groups."""
    return _classify_wiki_raw(heading) or "uses"


# Each tab has one PubChem group that Wikipedia sections are merged into, so a
# Wikipedia "Mechanism of action" renders as a heading INSIDE the Pharmacology
# group expander right beside the PubChem headings — not in a separate block.
WIKI_TAB_TARGET_GROUP = {
    "props": "Experimental Properties & Measurements",
    "safety": "Toxicity & Health",
    "pharma": "Pharmacology & Clinical",
    "uses": "Manufacturing, Uses & Commerce",
    "spectra": "Spectra & Analytical",
    "structure": "Structures & Conformers",
}


# Registry codes / CAS numbers / IUPAC-style systematic strings that are useless
# as Wikipedia page titles — used to filter synonyms down to human drug names.
_CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
_REG_CODE_RE = re.compile(r"^[A-Z]{2,}[- ]?\d")            # DTXSID..., BMY-28689
_SYSTEMATIC_HINT = re.compile(r"\d[,\-]|acid$|\bylidene\b|tetracene|carboxim")


def _name_like_synonym(s: str) -> bool:
    """True when a PubChem synonym looks like a common/brand drug name worth
    trying as a Wikipedia title (not a CAS number, registry code, or the long
    systematic IUPAC name)."""
    s = (s or "").strip()
    if not (3 <= len(s) <= 40):
        return False
    if ":" in s or "/" in s or _CAS_RE.match(s) or _REG_CODE_RE.match(s):
        return False
    if _SYSTEMATIC_HINT.search(s.lower()):
        return False
    if sum(c.isdigit() for c in s) > 2:
        return False
    return any(c.isalpha() for c in s)


# Short glossary used for the "?" tooltips next to each property.
PROPERTY_GLOSSARY = {
    "iupac_name": "Systematic IUPAC name of the compound.",
    "molecular_formula": "Atom counts per element (Hill notation).",
    "formula": "Molecular formula (RDKit-calculated).",
    "smiles": "SMILES line notation describing the structure.",
    "molecular_weight": "Average molar mass over natural isotope abundances (Da).",
    "mol_weight": "Average molar mass (RDKit, Da).",
    "monoisotopic_mass": "Mass using the most abundant isotope of each element (Da).",
    "exact_mass": "Exact (monoisotopic) mass (Da).",
    "charge": "Net formal charge of the species.",
    "tpsa": "Topological polar surface area (Å²); correlates with permeability.",
    "xlogp": "PubChem-computed octanol/water partition coefficient (lipophilicity).",
    "clogp": "RDKit Crippen logP estimate (lipophilicity).",
    "h_bond_acceptor_count": "Number of hydrogen-bond acceptor atoms.",
    "h_bond_donor_count": "Number of hydrogen-bond donor atoms.",
    "hba": "Hydrogen-bond acceptors (RDKit).",
    "hbd": "Hydrogen-bond donors (RDKit).",
    "rotatable_bonds": "Number of rotatable bonds (molecular flexibility).",
    "heavy_atom_count": "Number of non-hydrogen atoms.",
    "covalent_unit_count": "Number of covalently-bonded units (e.g. salts > 1).",
    "isotope_atom_count": "Number of atoms with a specified isotope.",
    "complexity": "Bertz/Hendrickson structural complexity score.",
    "coordinate_type": "Type of stored coordinates (2D/3D).",
}

# ----------------- Ordering / formatting helpers ---------------------------
def _norm_key_for_order(k: Any) -> str:
    return re.sub(r'[^a-z0-9]', '', str(k).lower())

_norm_key = _norm_key_for_order  # alias used for exclude checks


def order_keys(keys: List[str], reference_keys: Optional[List[str]] = None) -> List[str]:
    ref = reference_keys or list(keys)
    pref_map = {_norm_key_for_order(k): i for i, k in enumerate(PREFERRED_ORDER)}

    def sort_key(k):
        nk = _norm_key_for_order(k)
        if nk in pref_map:
            return (0, pref_map[nk])
        if UNLISTED_ORDER == "alpha":
            return (1, nk)
        try:
            return (1, ref.index(k))
        except ValueError:
            return (1, nk)
    return sorted(keys, key=sort_key)


def truncate(value, max_len=MAX_LEN):
    if isinstance(value, (list, dict)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
        return text if len(text) <= max_len else text[:max_len] + " ... (truncated)"
    if isinstance(value, str):
        return value if len(value) <= max_len else value[:max_len] + " ... (truncated)"
    return value


def _format_value(k, v):
    try:
        nk = _norm_key_for_order(k)
        if isinstance(v, float):
            if nk in ("molecularweight", "molweight", "exactmass", "monoisotopicmass"):
                return f"{v:.4g} Da"
            if nk in ("tpsa",):
                return f"{v:.4g} Å²"
            return f"{v:.4g}"
        if isinstance(v, int):
            return str(v)
        return truncate(v, max_len=200)
    except Exception:
        return str(v)

# ---------- RDKit helpers --------------------------------------------------
def rdkit_properties(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "mol_weight": Descriptors.MolWt(mol),
        "tpsa": Descriptors.TPSA(mol),
        "clogp": Crippen.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    }


def derived_metrics(smiles: str) -> Dict[str, Any]:
    """Drug-likeness / structural descriptors computed from RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    try:
        qed = QED.qed(mol)
    except Exception:
        qed = None
    try:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                             useLegacyImplementation=False)
        undefined_stereo = sum(1 for _, lab in centers if lab == '?')
    except Exception:
        undefined_stereo = 0
    return {
        "Molecular weight": (round(mw, 3), "Da"),
        "Exact mass": (round(Descriptors.ExactMolWt(mol), 4), "Da"),
        "cLogP (Crippen)": (round(logp, 3), ""),
        "TPSA": (round(Descriptors.TPSA(mol), 2), "Å²"),
        "H-bond donors": (hbd, ""),
        "H-bond acceptors": (hba, ""),
        "Rotatable bonds": (Descriptors.NumRotatableBonds(mol), ""),
        "Rings": (rdMolDescriptors.CalcNumRings(mol), ""),
        "Aromatic rings": (rdMolDescriptors.CalcNumAromaticRings(mol), ""),
        "Fraction Csp³": (round(rdMolDescriptors.CalcFractionCSP3(mol), 3), ""),
        "Heavy atoms": (mol.GetNumHeavyAtoms(), ""),
        "Formal charge": (Chem.GetFormalCharge(mol), ""),
        "QED (drug-likeness)": (round(qed, 3) if qed is not None else "—", ""),
        "Lipinski violations": (violations, " of 4"),
        "Undefined stereocenters": (undefined_stereo, ""),
        "_ro5_violations": violations,
    }


def canonicalize_smiles(smiles: str) -> Optional[Dict[str, Any]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mol": mol,
        "isomeric": Chem.MolToSmiles(mol, isomericSmiles=True),
        "flat": Chem.MolToSmiles(mol, isomericSmiles=False),
    }


def smiles_from_molblock(molblock: str) -> str:
    if not molblock:
        return ""
    try:
        mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=False)
        if mol is None and "$$$$" in molblock:
            mol = Chem.MolFromMolBlock(molblock.split("$$$$", 1)[0], sanitize=True, removeHs=False)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        pass
    return ""


def render_2d_pil(smiles: str, size=(450, 450)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Draw.rdDepictor.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=size)


def render_2d_highlight(smiles: str, highlights, size=(440, 360)) -> Optional[bytes]:
    """Draw the 2D structure with one or more atom groups highlighted.

    `highlights` is a list of (atom_indices, (r,g,b)) tuples with channels in
    0..1. Returns PNG bytes (rdMolDraw2D Cairo) or None if parsing fails. The
    depiction is the same one shown elsewhere — here it doubles as an
    interactive legend for the detected chemistry rather than a static picture.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Draw.rdDepictor.Compute2DCoords(mol)
    atom_cols: Dict[int, Tuple[float, float, float]] = {}
    bond_ids: List[int] = []
    bond_cols: Dict[int, Tuple[float, float, float]] = {}
    all_atoms: List[int] = []
    for atom_indices, color in highlights:
        idx_set = set(atom_indices)
        for a in atom_indices:
            atom_cols[a] = color
            all_atoms.append(a)
        # highlight bonds whose both ends are in the same group
        for b in mol.GetBonds():
            if b.GetBeginAtomIdx() in idx_set and b.GetEndAtomIdx() in idx_set:
                bond_ids.append(b.GetIdx())
                bond_cols[b.GetIdx()] = color
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.highlightBondWidthMultiplier = 12
    try:
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=all_atoms, highlightAtomColors=atom_cols,
            highlightBonds=bond_ids, highlightBondColors=bond_cols,
        )
    except Exception:
        return None
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def render_3d_viewer_and_block(smiles: str, width=600, height=460) -> Tuple[Any, Any, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, "Invalid SMILES"
    mol_h = Chem.AddHs(mol)
    embed_err = None
    try:
        res = -1
        for seed in (0xf00d, 42, 7, 1):  # vary the seed across attempts
            params = AllChem.ETKDG()
            params.randomSeed = seed
            res = AllChem.EmbedMolecule(mol_h, params)
            if res == 0:
                break
        if res != 0:
            # last-ditch: random coordinates
            params = AllChem.ETKDG()
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(mol_h, params)
        if res != 0:
            embed_err = f"Embed failed (return {res})"
        else:
            try:
                AllChem.UFFOptimizeMolecule(mol_h)
            except Exception as e:
                embed_err = f"UFFOptimize failed: {e}"
    except Exception as e:
        embed_err = str(e)

    try:
        block = Chem.MolToMolBlock(mol_h)
    except Exception as e:
        return None, None, f"Failed to create MolBlock: {e}"

    try:
        viewer = py3Dmol.view(width=width, height=height)
        viewer.setBackgroundColor("0xFFFFFF")
        viewer.addModel(block, "sdf")
        viewer.setStyle({}, {"stick": {"radius": 0.2}, "sphere": {"scale": 0.3}})
        viewer.zoomTo()
        return viewer, block, embed_err
    except Exception as e:
        return None, block, f"py3Dmol viewer creation failed: {e}"

# ---------- Text extraction / sanitization ---------------------------------
def _extract_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [p for p in (_extract_text(v) for v in value) if p]
        return "\n\n".join(parts) if parts else None
    if isinstance(value, dict):
        for key in ("String", "StringWithMarkup", "Value", "Description", "Name", "Text"):
            if key in value:
                return _extract_text(value[key])
        info = value.get("Information") or value.get("information")
        if isinstance(info, list):
            parts = []
            for entry in info:
                if isinstance(entry, dict):
                    for subk in ("String", "Value", "Description", "Name", "Title"):
                        if subk in entry:
                            txt = _extract_text(entry[subk])
                            if txt:
                                parts.append(txt)
                                break
                    else:
                        extracted = _extract_text(list(entry.values()))
                        if extracted:
                            parts.append(extracted)
                else:
                    txt = _extract_text(entry)
                    if txt:
                        parts.append(txt)
            parts = [p for p in parts if p]
            return "\n\n".join(parts) if parts else None
        vals = [v for v in (_extract_text(v) for v in value.values()) if v]
        return "\n\n".join(vals) if vals else None
    return str(value)


def find_record_description(flat_summary: Dict[str, Any]) -> str:
    if not isinstance(flat_summary, dict):
        return ""
    for k in flat_summary:
        if k.strip().lower() == "record description":
            return _extract_text(flat_summary[k]) or ""
    for k in flat_summary:
        low = k.lower()
        if "record" in low and "description" in low:
            return _extract_text(flat_summary[k]) or ""
    for variant in ("description", "overview", "summary"):
        for k in flat_summary:
            if variant in k.lower():
                txt = _extract_text(flat_summary[k])
                if txt:
                    return txt
    return ""


def sanitize_html_fragment(raw_html, use_bleach=True):
    if not raw_html:
        return ""
    if use_bleach and _BLEACH_AVAILABLE:
        allowed_tags = ['b', 'i', 'u', 'em', 'strong', 'a', 'p', 'br', 'ul', 'ol',
                        'li', 'sub', 'sup', 'small', 'span', 'div']
        allowed_attrs = {'a': ['href', 'title', 'rel', 'target']}
        return bleach.clean(raw_html, tags=allowed_tags, attributes=allowed_attrs, strip=True)
    no_script = re.sub(r'(?is)<(script|style).*?>.*?(</\1>)', '', raw_html)
    return _html.unescape(re.sub(r'<[^>]+>', '', no_script))

# ---------- Streamlit-cached wrappers --------------------------------------
if HAS_STREAMLIT:
    # Two cache tiers for every online source:
    #   * disk (st.cache_data persist="disk", in ~/.streamlit/cache) — survives
    #     app restarts, so any drug asked about before answers instantly. Only
    #     COMPLETE results are persisted: empty ones, and ones fetched while that
    #     source's host was failing (possibly partial), never are.
    #   * memory (ttl) — also keeps the empty results ('no FDA label for
    #     benzene') so reruns don't re-ask, but drops outage-empties at once.
    # Bump _CACHE_VERSION when a fetcher's output shape OR selection logic changes
    # in chem_sources (else stale disk results survive restarts);
    # the sidebar's "Refresh cached data" button clears both tiers.
    _CACHE_VERSION = "2026-10-12a"   # openfda_label: + medguide / patient_info / inactive_ingredients

    class _Outage(Exception):
        """Raised inside a cached fetcher so an outage's empty result is NOT
        cached (st.cache_data never caches a raised exception)."""

    class _NoCache(Exception):
        """Carries a result out of the disk tier without persisting it."""
        def __init__(self, result):
            super().__init__("not persisted")
            self.result = result

    def _pcp_compound(smiles: str) -> Dict[str, Any]:
        try:
            cs._pace("pubchem.ncbi.nlm.nih.gov")   # PubChemPy bypasses cs._get's pacing
            comps = pcp.get_compounds(smiles, 'smiles')
        except Exception:
            return {"rawinfo": {}, "cid": None}
        if not comps:
            return {"rawinfo": {}, "cid": None}
        c = comps[0]
        raw = c.to_dict() if hasattr(c, 'to_dict') else {}
        for key in ("record", "cactvs_fingerprint", "fingerprint", "sids", "elements"):
            raw.pop(key, None)
        cid = int(getattr(c, 'cid', 0) or raw.get('cid') or 0) or None
        return {"rawinfo": raw, "cid": cid}

    def _smiles_for_cid(cid: int) -> str:
        return cs.get_smiles_for_cid(cid) or smiles_from_molblock(cs.get_sdf_3d(cid))

    # tag → (fetch function, hosts whose failure makes a result untrustworthy,
    #        emptiness test)
    _FETCHERS = {
        "resolve_name": (lambda q, use_cir=True: cs.resolve_name(q, use_cir=use_cir),
                         (cs.PUBCHEM_BASE, cs.OPSIN_BASE, cs.CIR_BASE),
                         lambda r: not (r or {}).get("candidates")),
        "cir_resolve": (cs.cir_resolve, (cs.CIR_BASE,),
                        lambda r: not (r or {}).get("candidates")),
        "rxnorm_brand_ingredients": (cs.rxnorm_brand_ingredients, (cs.RXNAV_BASE,), None),
        "compound_by_smiles": (_pcp_compound, (cs.PUBCHEM_BASE,),
                               lambda r: not (r or {}).get("rawinfo")),
        "smiles_for_cid": (_smiles_for_cid, (cs.PUBCHEM_BASE,), None),
        "synonyms": (cs.fetch_synonyms, (cs.PUBCHEM_BASE,), None),
        "flat_pug": (lambda cid: cs.build_flat_compound_summary(
            cid, exclude_headings=EXCLUDE_HEADINGS), (cs.PUGVIEW_BASE,), None),
        "wiki": (cs.wikipedia_summary, (cs.WIKI_SUMMARY,), None),
        "unichem": (cs.unichem_xrefs, ("www.ebi.ac.uk",), None),
        "chembl": (cs.chembl_summary, (cs.CHEMBL_BASE, "www.ebi.ac.uk"), None),
        "chembl_by_name": (cs.chembl_molecule_by_name, (cs.CHEMBL_BASE,), None),
        "wikidata": (cs.wikidata_by_inchikey, ("query.wikidata.org",), None),
        "wiki_title": (cs.wikipedia_title_by_inchikey, ("query.wikidata.org",), None),
        "wiki_sections": (cs.wikipedia_sections, (cs.WIKI_API,), None),
        "parent_cid": (cs.pubchem_parent_cid, (cs.PUBCHEM_BASE,), None),
        "openfda_label": (lambda name, route="": cs.openfda_label(generic_name=name,
                                                                   main_route=route),
                          (cs.OPENFDA_LABEL,), None),
        "openfda_events": (cs.openfda_adverse_events, (cs.OPENFDA_EVENT,), None),
        "openfda_market": (cs.openfda_market_info, (cs.OPENFDA_LABEL,), None),
        "rxnorm_brands": (lambda rxcui="", name="": cs.rxnorm_brand_names(
            rxcui or cs.rxnorm_rxcui(name)), (cs.RXNAV_BASE,), None),
        "gtopdb": (cs.guidetopharmacology_by_inchikey, (cs.GTOPDB_BASE,), None),
    }

    def _failing(hosts) -> bool:
        return any(cs.host_failing(h) for h in hosts)

    # max_entries also bounds RAM (persisted entries are kept in memory too):
    # 1000 fits the ~1 GB of a free Streamlit Community Cloud app
    @st.cache_data(persist="disk", max_entries=1000, show_spinner=False)
    def _disk_fetch(tag: str, version: str, *args):
        fn, hosts, is_empty = _FETCHERS[tag]
        result = fn(*args)
        empty = is_empty(result) if is_empty else not result
        if empty or _failing(hosts):
            raise _NoCache(result)
        return result

    @st.cache_data(ttl=6 * 3600, max_entries=1000, show_spinner=False)
    def _mem_fetch(tag: str, *args):
        try:
            return _disk_fetch(tag, _CACHE_VERSION, *args)
        except _NoCache as e:
            _fn, hosts, is_empty = _FETCHERS[tag]
            empty = is_empty(e.result) if is_empty else not e.result
            if empty and _failing(hosts):
                raise _Outage(tag)      # outage, not "doesn't exist" — retry next time
            return e.result
        except Exception:               # a fetcher bug/odd payload: don't cache it
            raise _Outage(tag)

    def _fetcher(tag: str, empty):
        def fetch(*args):
            try:
                return _mem_fetch(tag, *args)
            except _Outage:
                return empty() if callable(empty) else empty
        fetch.__name__ = f"cached_{tag}"
        return fetch

    def clear_data_caches():
        """Sidebar 'Refresh cached data': forget every fetched result (disk too)."""
        _disk_fetch.clear()
        _mem_fetch.clear()

    cached_resolve_name = _fetcher("resolve_name", lambda: {"source": None, "candidates": []})
    cached_cir_resolve = _fetcher("cir_resolve", lambda: {"source": None, "candidates": []})
    cached_rxnorm_brand_ingredients = _fetcher("rxnorm_brand_ingredients", list)
    cached_compound_by_smiles = _fetcher("compound_by_smiles",
                                         lambda: {"rawinfo": {}, "cid": None})
    cached_smiles_for_cid = _fetcher("smiles_for_cid", "")
    cached_synonyms = _fetcher("synonyms", list)
    cached_flat_pug = _fetcher("flat_pug", dict)
    cached_wiki = _fetcher("wiki", dict)
    cached_unichem = _fetcher("unichem", list)
    cached_chembl = _fetcher("chembl", dict)
    cached_chembl_by_name = _fetcher("chembl_by_name", list)
    cached_wikidata = _fetcher("wikidata", dict)
    cached_wiki_title = _fetcher("wiki_title", "")
    cached_wiki_sections = _fetcher("wiki_sections", dict)
    cached_parent_cid = _fetcher("parent_cid", None)
    cached_openfda_label = _fetcher("openfda_label", dict)
    cached_openfda_events = _fetcher("openfda_events", list)
    cached_openfda_market = _fetcher("openfda_market", dict)
    cached_rxnorm_brands = _fetcher("rxnorm_brands", list)
    cached_gtopdb = _fetcher("gtopdb", dict)

    def prefetch(*calls):
        """Run independent cached fetchers concurrently, so their results are
        already in st.cache_data when the (sequential) main flow asks for them.
        Each call is (fn, *args) and must use the exact args the main flow will
        pass, or it misses the cache. Results/errors are discarded here — the
        main flow's own call returns the cached value (or retries)."""
        # skip calls whose primary key (first arg) is missing; later args may
        # legitimately be "" (cached_rxnorm_brands(rxcui, ""))
        with ThreadPoolExecutor(max_workers=10) as ex:
            for f in submit_all(ex, *calls):
                _quiet_result(f)

    def submit_all(ex, *calls):
        """Submit (fn, *args) calls to an executor with the Streamlit script
        context attached; calls whose first arg is missing are skipped (later
        args may legitimately be "", e.g. cached_rxnorm_brands(rxcui, ""))."""
        ctx = get_script_run_ctx()

        def _run(fn, *args):
            if ctx is not None:
                add_script_run_ctx(threading.current_thread(), ctx)
            return fn(*args)

        return [ex.submit(_run, *c) for c in calls
                if c and any(a not in (None, "") for a in c[1:])]

    def _quiet_result(fut, default=None):
        try:
            return fut.result()
        except Exception:  # noqa: BLE001 — the main flow handles failures
            return default


# Max seconds to wait for the ChEMBL name-search recovery (tautomer fallback).
_CHEMBL_NAME_BUDGET_S = 12
# Max seconds (from the start of the parallel prefetch) to wait for ChEMBL's
# structure lookup before rendering without it.
_CHEMBL_BUDGET_S = 8


def inchikey_for(smiles: str) -> str:
    """InChIKey from SMILES — the universal key for cross-database lookups."""
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""


def largest_fragment_smiles(smiles: str) -> str:
    """Canonical SMILES of the largest fragment — the free base/acid of a salt,
    with counterions/solvents stripped. Empty when parsing fails."""
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        parent = rdMolStandardize.LargestFragmentChooser().choose(mol)
        return Chem.MolToSmiles(parent, isomericSmiles=True) if parent is not None else ""
    except Exception:
        return ""


def is_multifragment(smiles: str) -> bool:
    """True when the SMILES holds more than one disconnected fragment (a salt,
    co-crystal, mixture, …)."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return bool(mol) and len(Chem.GetMolFrags(mol)) > 1


_TAUTOMER_ENUMERATOR = None


def canonical_tautomer_inchikey(smiles: str) -> str:
    """InChIKey of the RDKit-canonical tautomer of a compound's largest fragment.
    Two depictions that are the same substance up to tautomer/protonation/salt
    collapse to the SAME value here — even when their plain InChIKeys differ (as
    they do for PubChem's odd 'carboximidic acid' doxycycline vs the mainstream
    amide form). This is the identity test used to safely adopt a name-matched
    record from another database. Empty string on any failure."""
    global _TAUTOMER_ENUMERATOR
    frag = largest_fragment_smiles(smiles) or smiles
    mol = Chem.MolFromSmiles(frag) if frag else None
    if mol is None:
        return ""
    try:
        if _TAUTOMER_ENUMERATOR is None:
            from rdkit.Chem.MolStandardize import rdMolStandardize
            _TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
        return Chem.MolToInchiKey(_TAUTOMER_ENUMERATOR.Canonicalize(mol))
    except Exception:
        return ""


# Wikidata descriptions that carry no real information — skip these and fall
# back to another key rather than showing a useless "Chemical compound" line.
_GENERIC_WIKIDATA_DESC = {
    "chemical compound", "chemical substance", "group of stereoisomers",
    "class of chemical compounds", "chemical", "molecule",
}

# ---------- PUG-View grouping ----------------------------------------------
def build_grouped(flat_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _matches_keyword(nh: str, kw: str) -> bool:
        if not kw:
            return False
        kwn = kw.strip().lower()
        if " " in kwn or any(c in kwn for c in ("'", "-", "/")):
            return kwn in nh
        if re.fullmatch(r'[a-z]+', kwn):
            return kwn in re.split(r'[\s\W_]+', nh)
        return kwn in nh

    _custom_norm = {_norm(k): v for k, v in (CUSTOM_GROUP_MAP or {}).items()}
    grouped = {g: {} for g in GROUP_KEYWORDS.keys()} if GROUP_KEYWORDS else {"Other": {}}
    exclude_norm = {_norm(h) for h in (EXCLUDE_HEADINGS or set())}

    for heading, value in flat_summary.items():
        nh = _norm(heading)
        if nh in exclude_norm:
            continue
        if nh in _custom_norm:
            grouped.setdefault(_custom_norm[nh], {})[heading] = value
            continue
        placed = False
        for grp, keywords in (GROUP_KEYWORDS or {}).items():
            for kw in keywords:
                if _matches_keyword(nh, kw):
                    grouped.setdefault(grp, {})[heading] = value
                    placed = True
                    break
            if placed:
                break
        if not placed:
            grouped.setdefault("Other", {})[heading] = value
    return grouped


_SPECTRAL_KEYWORDS = {
    "kovats", "gc-ms", "gc ms", "lc-ms", "lc ms", "ms-ms", "ms ms", "mass spectrom",
    "mass spectrometry", "1d nmr", "1h nmr", "13c nmr", "17o nmr", "1h-1h", "1h-13c",
    "spectra", "other ms", "lc-ms/ms", "ms/ms",
}


def _is_spectral_heading(nh: str) -> bool:
    return any(kw in nh for kw in _SPECTRAL_KEYWORDS) if nh else False


_THIN_RULE = ("<hr style='margin:8px 0;border:none;"
              "border-top:1px solid rgba(128,128,128,0.25)'>")

# Heading terms that signal substantive "how the drug works / what it does"
# content — these float to the top of a group.
IMPORTANT_TERMS = {
    "mechanism of action": 60, "mode of action": 58, "pharmacodynamic": 55,
    "pharmacokinetic": 52, "drug indication": 52, "indication": 48,
    "therapeutic use": 50, "therapeutic": 38, "absorption": 46,
    "distribution and excretion": 46, "metabolism": 42, "biological half-life": 38,
    "toxicity summary": 52, "health effect": 44, "adverse effect": 44,
    "carcinogen": 42, "hepatotox": 42, "signs and symptoms": 42, "first aid": 44,
    "treatment": 36, "drug warning": 42, "hazard": 38, "summary": 34,
    "description": 34, "physical description": 36, "uses": 30, "effects": 30,
    "pharmacolog": 30, "interaction": 30,
}

# Heading tokens that signal an identifier / registry code — these sink.
DEPRIORITIZE_TERMS = (
    "id", "ids", "number", "code", "codes", "classification", "registry",
    "drugcentral", "zinc", "unii", "cas", "cid", "identifier", "identifiers",
    "crosslink", "crosslinks", "linkout", "collection", "tree", "ontology",
    "rtecs", "dsstox", "hmdb", "kegg", "chebi", "chembl", "max phase",
    "first approval", "availability type",
)

# A single PUG-View heading often flattens several entries (joined by ' | ') in
# PubChem's own order — which routinely leads with generic boilerplate ("X is a
# registry/database…") or a "for more, visit the … record page" pointer, burying
# the actual compound facts. These patterns mark an entry as low-value so it
# sinks to the bottom of its heading; substantive prose floats up.
_LOWVALUE_ENTRY_RES = (
    re.compile(r"^\s*/[A-Z][A-Z /]*/\s*\S+\.(gov|org|com|edu)\b", re.I),  # "/CLINICAL TRIALS/ ClinicalTrials.gov …"
    re.compile(r"\bis a (registry|database|resource|repository|collection)\b", re.I),
    re.compile(r"\b(maintained|operated|provided|developed) by the\b", re.I),
    re.compile(r"\bplease visit\b", re.I),
    re.compile(r"\bfor (more|complete|additional|further)\b.{0,60}\b"
               r"(visit|record page|hsdb|data)\b", re.I),
)


def _entry_rank(text: str) -> int:
    """Sort key for the entries within one heading (lower = shown first):
    0 = substantive compound prose, 1 = short fragments/classifications,
    2 = boilerplate / external-resource descriptions / 'see more' pointers.
    Ties keep PubChem's original order (stable sort)."""
    t = (text or "").strip()
    for rx in _LOWVALUE_ENTRY_RES:
        if rx.search(t):
            return 2
    return 0 if len(re.findall(r"[A-Za-z]{3,}", t)) >= 20 else 1


def _heading_priority(heading: str, val: Any) -> float:
    """Higher score = more pertinent → shown earlier within a group."""
    nh = (heading or "").strip().lower()
    text = json.dumps(val)[:300] if isinstance(val, (dict, list)) \
        else re.sub(r"<[^>]+>", "", str(val))
    compact = text.strip()
    # content richness: lots of prose ranks higher (cap the contribution)
    words = len(re.findall(r"[A-Za-z]{3,}", text))
    score = min(words / 4.0, 30.0)
    # identifier-like value (a bare number, a "[13]" ref, or a short code)
    if not compact or len(compact) < 10 or re.fullmatch(r"[\[\]\d\s,.;:()/+\-]+", compact):
        score -= 25
    for term, boost in IMPORTANT_TERMS.items():
        if term in nh:
            score += boost
    for term in DEPRIORITIZE_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", nh):
            score -= 30
    return score


def _ordered_items(items: Dict[str, Any]):
    """Return (heading, value) pairs sorted by descending importance (stable)."""
    return sorted(items.items(), key=lambda kv: -_heading_priority(kv[0], kv[1]))


def _group_priority(items: Dict[str, Any]) -> float:
    """Score a whole group so the most pertinent groups sort first within a tab."""
    if not items:
        return -999.0
    prios = [_heading_priority(h, v) for h, v in items.items()]
    # driven by the group's strongest heading, nudged by its overall substance
    return max(prios) + 0.05 * sum(p for p in prios if p > 0)


# Two structuring cues PubChem's flattened narratives use, exploited to turn a
# wall of text into scannable points: ALL-CAPS section labels
# ("HUMAN EXPOSURE AND TOXICITY:") and inline '*' list markers, sometimes
# introduced by a Title-Case header ("Signs and Symptoms of Overdose *").
_CAPS_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9][A-Z0-9 &/,'()\-]{1,}?:)(?=\s)")
_INLINE_HEADER_RE = re.compile(r"(?:^|(?<=[.;:]\s))([A-Z][A-Za-z]+(?:\s+[A-Za-z]+){1,6}):?\s+\*\s")
_SEG_SEP = ""   # top-level point boundary
_SUB_SEP = ""   # sub-bullet boundary


def _pointify(text: str) -> List[str]:
    """Break a dense narrative value into point-form markdown lines.

    Splits on ALL-CAPS section labels and inline '*' bullets (nesting the
    latter under the header that introduces them) so a paragraph blob renders
    as a scannable list rather than one huge bullet. Returns markdown lines."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    text = _CAPS_LABEL_RE.sub(lambda m: f"{_SEG_SEP}{m.group(1)} ", text)
    text = _INLINE_HEADER_RE.sub(lambda m: f"{_SEG_SEP}{m.group(1)}:{_SUB_SEP}", text)
    text = re.sub(r"\s\*\s", _SUB_SEP, text)

    lines: List[str] = []
    for seg in text.split(_SEG_SEP):
        seg = seg.strip()
        if not seg:
            continue
        subparts = [s.strip() for s in seg.split(_SUB_SEP)]
        head, subs = subparts[0], [s for s in subparts[1:] if s]
        m = re.match(r"([A-Z0-9][A-Z0-9 &/,'()\-]{1,}:|[A-Z][A-Za-z ]{2,40}:)\s*(.*)$",
                     head, re.S)
        if m:
            label, body = m.group(1), m.group(2).strip()
            lines.append(f"- **{label}** {body}" if body else f"- **{label}**")
        elif head:
            lines.append(f"- {head}")
        for s in subs:
            lines.append(f"    - {s}")
    return lines


# --- Overload control for Wikipedia backfill -------------------------------
# Wikipedia articles, exploded one-bullet-per-sentence and interleaved with the
# PubChem record, drowned the useful content in overlap and trivia. Four levers
# tame it: (1) drop low-value sentences, (2) de-duplicate against PubChem and
# against already-shown Wikipedia text, (3) gap-fill — contribute fewer bullets
# to a group PubChem already covers well, (4) cap per section with a show-more.

# Sentences that are almost always filler in a chemical/drug context: bare
# citation/date trivia, "as of" hedges, edit-history artefacts, gallery cruft.
_LOWVALUE_SENTENCE_RE = re.compile(
    r"^(as of\b|in \d{4}[, ]|by \d{4}[, ]|see also\b|for more\b|main article\b"
    r"|this (article|section)\b|citation needed\b)", re.I)
_YEAR_ONLY_RE = re.compile(r"^\W*\d{3,4}\W*$")

# Common words that carry no topical signal — excluded from the de-dup
# fingerprint so two sentences are compared on their substance, not their glue.
_STOPWORDS = frozenset("""
a an the of to in on at by for with from into over under and or but nor so yet
is are was were be been being has have had having do does did can could may
might must shall should will would this that these those it its it's as such
also which who whom whose when where while than then thus however therefore
their them they there here about between within without both each any all some
one two more most other another used using use uses well known called via due
""".split())

_WIKI_BULLET_CAP = 6          # bullets shown per section before "show more"
_WIKI_BULLET_CAP_RICH = 3     # gap-fill: fewer when PubChem already covers the group
_GROUP_RICH_WORDS = 120       # PubChem word-count above which a group is "rich"
_DUP_JACCARD = 0.6            # word-set overlap above which two sentences dup
_PUBCHEM_BLOCK_CAP = 4        # top-level PubChem bullets shown before "show more"

# Split a dense value into sentences without chopping 'U.S.', 'e.g.', 'Fig. 2':
# the boundary must follow a lowercase/digit/bracket + end punctuation and precede
# a capital/digit.
_SENT_SPLIT_RE = re.compile(r"(?<=[a-z0-9)\]][.!?])\s+(?=[A-Z0-9])")


def _content_words(text: str) -> frozenset:
    """The topical word-set of a sentence, lowercased, stopwords/short words
    dropped — the fingerprint the de-duper compares on."""
    words = re.findall(r"[a-z][a-z0-9\-]{2,}", (text or "").lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def _is_lowvalue_sentence(s: str) -> bool:
    """True for citation/date/history filler that adds noise, not chemistry."""
    s = s.strip()
    if len(s) < 25 or _YEAR_ONLY_RE.match(s):
        return True
    if _LOWVALUE_SENTENCE_RE.match(s):
        return True
    # Sentences that are mostly a bracketed reference or numeric noise.
    if len(_content_words(s)) < 3:
        return True
    return False


class _SentenceDeduper:
    """Tracks the word-sets of everything already shown (PubChem entries first,
    then each Wikipedia sentence as it's accepted) and rejects near-duplicates,
    so Wikipedia never re-states what the record already says."""

    def __init__(self) -> None:
        self._seen: List[frozenset] = []

    def add_reference(self, text: str) -> None:
        """Register existing (PubChem/lead) text so later wiki dups get dropped."""
        for sent in re.split(r"(?<=[a-z0-9)\]][.!?])\s+(?=[A-Z0-9])", text or ""):
            ws = _content_words(sent)
            if len(ws) >= 3:
                self._seen.append(ws)

    def is_dup(self, words: frozenset) -> bool:
        if len(words) < 3:
            return False
        for seen in self._seen:
            if not seen:
                continue
            inter = len(words & seen)
            if not inter:
                continue
            union = len(words | seen)
            if inter / union >= _DUP_JACCARD or words <= seen:
                return True
        return False

    def accept(self, words: frozenset) -> None:
        self._seen.append(words)

    def is_restatement(self, words: frozenset, ratio: float = 0.8) -> bool:
        """Paraphrase check: most of this sentence's content words already
        appeared together in one earlier sentence ('decreases gluconeogenesis
        in the liver' after '…decreasing hepatic glucose production (also
        called gluconeogenesis)…'). Needs ≥5 words so short facts survive."""
        if len(words) < 5:
            return False
        return any(len(words & seen) / len(words) >= ratio for seen in self._seen if seen)


def _wiki_bullets(text: str, deduper: "_SentenceDeduper") -> List[str]:
    """Turn a Wikipedia section's prose into a list of point-form sentences —
    dropping low-value filler and anything the deduper has already seen (PubChem
    entries or earlier wiki bullets). Returns the surviving sentences in order;
    the caller caps how many render. The sentence split only fires on end
    punctuation preceded by a lowercase letter/digit/bracket, so 'U.S.', 'e.g.',
    and 'Fig. 2' don't get chopped."""
    text = re.sub(r"\n{2,}", "\n", (text or "").strip())
    out: List[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for sent in re.split(r"(?<=[a-z0-9)\]][.!?])\s+(?=[A-Z0-9])", block):
            sent = sent.strip()
            if not sent or _is_lowvalue_sentence(sent):
                continue
            words = _content_words(sent)
            if deduper.is_dup(words):
                continue
            deduper.accept(words)
            out.append(sanitize_html_fragment(sent) if _BLEACH_AVAILABLE else sent)
    return out


def dedup_pubchem_grouped(grouped: Dict[str, Dict[str, Any]]
                          ) -> Dict[str, Dict[str, Any]]:
    """Record-wide de-duplication of the PubChem text itself (not just Wikipedia).
    PubChem repeats the same sentences across headings — e.g. ibuprofen's dosage
    boilerplate and its pain/COX/inflammation story recur under Therapeutic Uses,
    Drug Indication, Pharmacodynamics and Mechanism of Action. Processing headings
    in importance order and dropping a sentence the record has already stated keeps
    each fact in its most-pertinent heading and everywhere else is silent. A whole
    heading that turns out fully redundant is removed."""
    if not grouped:
        return grouped
    out = {g: dict(v) for g, v in grouped.items()}
    deduper = _SentenceDeduper()
    for g in sorted(out, key=lambda gg: -_group_priority(out.get(gg, {}))):
        items = out[g]
        for heading, val in sorted(items.items(),
                                   key=lambda kv: -_heading_priority(kv[0], kv[1])):
            if isinstance(val, (dict, list)):        # wiki/combined/JSON — untouched
                continue
            # substantive entries first, so a shared sentence is KEPT with the
            # meatiest entry and dropped from the boilerplate/pointer copies.
            entries = sorted((p.strip() for p in str(val).split(" | ") if p.strip()),
                             key=_entry_rank)
            kept_entries: List[str] = []
            for entry in entries:
                kept: List[str] = []
                for sent in _SENT_SPLIT_RE.split(entry):
                    sent = sent.strip()
                    if not sent:
                        continue
                    ws = _content_words(sent)
                    if len(ws) >= 4 and deduper.is_dup(ws):
                        continue                     # already said elsewhere
                    if len(ws) >= 4:
                        deduper.accept(ws)
                    kept.append(sent)
                if kept:
                    kept_entries.append(" ".join(kept))
            if kept_entries:
                items[heading] = " | ".join(kept_entries)
            else:
                del items[heading]                   # nothing new anywhere — drop it
    return out


def _norm_head(heading: str) -> str:
    """A heading's topic identity for matching PubChem vs Wikipedia (and wiki vs
    wiki) sections, so the same topic never renders as two headings. Strips the
    '›' sub-section prefix, case, punctuation/slashes, folds '&'→'and', and
    singularises the trailing word so 'Adverse Effects' == 'Adverse effect' and
    'Drug Interactions' == 'Drug interaction'."""
    h = re.sub(r"^(?:›\s*)+", "", heading or "").strip().lower()
    h = h.replace("&", " and ")
    h = re.sub(r"[^a-z0-9]+", " ", h).strip()   # slashes/punctuation → spaces
    words = h.split()
    if words and len(words[-1]) > 3 and words[-1].endswith("s") \
            and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]              # crude singularise of last word
    return " ".join(words)


def merge_wiki_into_grouped(grouped: Dict[str, Dict[str, Any]],
                            wiki_secs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Inject each Wikipedia section into the PubChem group for its tab as a peer
    heading, after de-duplicating and gap-filling so Wikipedia only adds what the
    PubChem record is missing. Sections that survive with no new sentences are
    dropped entirely rather than shown as empty headings."""
    merged = {g: dict(v) for g, v in (grouped or {}).items()}
    if not wiki_secs:
        return merged
    url = wiki_secs.get("url", "")

    # One deduper across the whole record: seed it with the Wikipedia lead (so
    # sections can't re-state the summary) and with each group's PubChem text as
    # we reach that group. Wiki sentences accepted for one group also suppress
    # the same fact reappearing in another (e.g. History echoed under Uses).
    deduper = _SentenceDeduper()
    deduper.add_reference(wiki_secs.get("lead", ""))
    seeded_groups: set = set()
    group_words: Dict[str, int] = {}

    def _seed_group(group: str) -> None:
        if group in seeded_groups:
            return
        seeded_groups.add(group)
        words = 0
        for h, v in (merged.get(group) or {}).items():
            if isinstance(v, dict) and v.get("__wiki__"):
                continue
            txt = _plain_text(v)
            deduper.add_reference(txt)
            words += len(re.findall(r"[A-Za-z]{3,}", txt))
        group_words[group] = words

    for s in wiki_secs.get("sections", []):
        if not s.get("text"):
            continue
        tab = s.get("tab") or classify_wiki_section(s["heading"])
        group = WIKI_TAB_TARGET_GROUP.get(tab)
        if not group:
            continue
        merged.setdefault(group, {})
        _seed_group(group)
        bullets = _wiki_bullets(s["text"], deduper)
        if not bullets:                  # nothing new after de-dup — skip entirely
            continue
        # Gap-fill: contribute fewer bullets to a group PubChem already covers.
        cap = _WIKI_BULLET_CAP_RICH if group_words.get(group, 0) >= _GROUP_RICH_WORDS \
            else _WIKI_BULLET_CAP
        # If any heading in the group already covers this topic (same normalised
        # name — whether it's a PubChem heading or an earlier Wikipedia one), fold
        # these bullets INTO it rather than showing a second same-named heading
        # (the "two Pharmacodynamics" case, and its equivalents on other drugs).
        target = _norm_head(s["heading"])
        existing = next((k for k in merged[group] if _norm_head(k) == target), None)
        if existing is not None:
            ev = merged[group][existing]
            if isinstance(ev, dict) and (ev.get("__wiki__") or ev.get("__combined__")):
                ev["bullets"] = ev["bullets"] + bullets    # append to wiki/combined
            else:                                          # PubChem value → combine
                merged[group][existing] = {"__combined__": True, "pubchem": ev,
                                           "bullets": bullets, "cap": cap, "url": url}
            continue
        indent = "› " * max(0, s.get("level", 2) - 2)
        merged[group][f"{indent}{s['heading']}"] = {
            "__wiki__": True, "bullets": bullets, "cap": cap, "url": url}
    return merged


# Groups whose many near-synonymous headings get folded into a few canonical
# topics. Scoped (for now) to the pharmacology group — that's where PubChem
# scatters one subject across Therapeutic Uses/Drug Indication, Mechanism of
# Action/Pharmacodynamics, and Absorption-Distribution-Excretion/Metabolism/
# Half-Life (all of which are just ADME = Pharmacokinetics).
_CONSOLIDATE_GROUPS = {"Pharmacology & Clinical"}

# (canonical heading, normalised-substring patterns). A heading whose _norm_head
# contains any pattern joins that topic; first rule wins. Patterns are in the
# normalised form (lowercased, punctuation→space, trailing word singularised).
_CANON_RULES = (
    ("Uses & Indications",
     ("therapeutic use", "drug indication", "indication", "medical use")),
    ("Mechanism of Action & Pharmacodynamics",
     ("mechanism of action", "mode of action", "pharmacodynamic", "pharmacology",
      "function")),
    ("Pharmacokinetics (ADME)",
     ("pharmacokinetic", "distribution and excretion", "absorption distribution",
      "metabolism", "metabolite", "half life", "elimination route")),
)

# Pure identifier / registry / classification headings — no narrative value; they
# get tucked into one collapsed "IDs & Classifications" heading (empties dropped).
_CODE_HEADING_RE = re.compile(
    r"classification|\batc\b|availability type|route of administration|drug class"
    r"|^\s*drugs?\s*$|human drugs|tissue location|cellular location|drugcentral"
    r"|max phase|first approval|caco2|black box|\bunii\b|\bhts\b|open targets"
    r"|\bccs\b|lotus|nonproprietary|usan|\binn\b|related cas", re.I)


def _canonical_topic(heading: str) -> Optional[str]:
    nh = _norm_head(heading)
    for canon, pats in _CANON_RULES:
        if any(p in nh for p in pats):
            return canon
    return None


def _split_value_parts(val: Any) -> Tuple[str, List[str], str]:
    """Decompose any heading value into (pubchem_text, wiki_bullets, wiki_url) so
    several headings' values can be merged under one canonical topic."""
    if isinstance(val, dict) and val.get("__combined__"):
        return str(val.get("pubchem") or ""), list(val.get("bullets") or []), val.get("url", "")
    if isinstance(val, dict) and val.get("__wiki__"):
        return "", list(val.get("bullets") or []), val.get("url", "")
    if isinstance(val, (dict, list)):
        return "", [], ""
    return str(val), [], ""


def consolidate_grouped(grouped: Dict[str, Dict[str, Any]]
                        ) -> Dict[str, Dict[str, Any]]:
    """Fold near-synonymous headings in the pharmacology group into canonical
    topics and collapse identifier/classification headings into one section, so a
    22-heading wall becomes a handful of meaningful sections."""
    if not grouped:
        return grouped
    out: Dict[str, Dict[str, Any]] = {}
    for g, items in grouped.items():
        if g not in _CONSOLIDATE_GROUPS or not items:
            out[g] = items
            continue
        result: Dict[str, Any] = {}
        codes: List[Tuple[str, str]] = []
        buckets: Dict[str, List[Tuple[str, Any]]] = {}
        for heading, val in items.items():
            if _CODE_HEADING_RE.search(heading):
                pv = _plain_text(val)
                if len(pv) >= 3:                       # keep only non-empty codes
                    codes.append((re.sub(r"^(?:›\s*)+", "", heading).strip(), pv))
                continue                               # empty codes dropped entirely
            canon = _canonical_topic(heading)
            if canon:
                buckets.setdefault(canon, []).append((heading, val))
            else:
                result[heading] = val
        for canon, members in buckets.items():
            if len(members) == 1:                      # nothing to merge — keep name
                result[members[0][0]] = members[0][1]
                continue
            members.sort(key=lambda hv: -_heading_priority(hv[0], hv[1]))
            pub_parts, bullets, url = [], [], ""
            for _h, v in members:
                p, b, u = _split_value_parts(v)
                if p:
                    pub_parts.append(p)
                bullets.extend(b)
                url = url or u
            merged_pub = " | ".join(pub_parts)
            if bullets:
                result[canon] = {"__combined__": True, "pubchem": merged_pub,
                                 "bullets": bullets, "cap": _WIKI_BULLET_CAP, "url": url}
            else:
                result[canon] = merged_pub
        if codes:
            result["IDs & Classifications"] = " | ".join(f"{n}: {v}" for n, v in codes)
        out[g] = result
    return out


# --- Topic refinement: keep each pharmacology heading on-topic -------------
# PubChem's "Mechanism of Action" for a well-studied drug mixes a DrugBank
# monograph with journal abstracts (HSDB): mouse/cell-line findings, "here we
# show…", disease background ("Insulin is an important hormone…"), trial
# results, side effects and transporter PK. Sentence by sentence, this pass
# keeps what answers the heading and re-homes the rest:
#   research   → a separate "Research findings (preclinical)" heading
#   background → dropped (doesn't mention the drug; defines the disease/enzyme)
#   safety     → Toxicity & Health › "Adverse effects"
#   efficacy   → "Clinical efficacy (trials)" in the pharmacology group
#   pk         → the Pharmacokinetics heading
#   hedge      → only the first "mechanism not fully understood" is kept
_REFINE_GROUP = "Pharmacology & Clinical"
_RESEARCH_HEADING = "Research findings (preclinical)"
_EFFICACY_HEADING = "Clinical efficacy (trials)"

# A journal abstract: first-person voice. Every sentence of such an entry is
# research, including the ones that don't say "we" themselves.
_FIRST_PERSON_RE = re.compile(
    r"\b(?:here )?we (?:show|report|investigat\w*|found|find|demonstrat\w*|examin\w*|"
    r"observ\w*|propos\w*|hypothesi\w*|identif\w*|provide|conclude)\b|\bour (?:data|"
    r"results|findings|study|studies)\b|\bin (?:this|the present) study\b", re.I)
_RESEARCH_RE = re.compile(
    _FIRST_PERSON_RE.pattern + r"|"
    r"\b(?:mice|mouse|murine|rats?|knockout|MEFs?|fibroblasts|wild[- ]type|C57\w*|"
    r"C3H10T1/2|cell lines?|cultured cells|cell cultures?|used in research)\b|"
    r"\b(?:\w+ )?(?:study|studies) (?:indicates?|suggests?|shows?|demonstrates?)\b|"
    # lab-concentration / slice-electrophysiology abstracts (diazepam's)
    r"\d+(?:\.\d+)?\s*(?:uM|µM|μM|nM|mM)\b|\b(?:organotypic|slice cultures?)\b|"
    r"\b(?:these|the present|our) (?:findings|results|data)\b|"
    r"\bconsistent with the hypothesis\b|\bwere quantified\b|"
    # study write-ups ('…was studied in 33 cases', 'It was concluded that…')
    r"\bwas (?:studied|determined|investigated|examined) in\b|\bit was concluded\b|"
    r"\bresults (?:indicate|suggest|show)\b", re.I)
# Side remarks that aren't about how THIS drug works: class membership lists,
# "not the only drug…", "drugs such as X also…".
_TRIVIA_RE = re.compile(
    r"\bother (?:classical |related |similar )?\w+ (?:include|are)\b|"
    r"\bis not the only drug\b|\bdrugs such as \w+ also\b", re.I)
_USES_RE = re.compile(r"\b(?:is|are) (?:widely |commonly |frequently |mainly )?used "
                      r"(?:to treat|for the treatment|in the treatment|for)\b", re.I)
_BACKGROUND_RE = re.compile(
    r"\b(?:is|are) (?:also )?(?:an? |the )?(?:important|useful|key|essential|major|critical)\b|"
    r"\bis characteri[sz]ed by\b|\bplays? an? (?:important|key|central|critical|major) role\b|"
    r"\b(?:is|are) an? (?:enzyme|hormone|protein|receptor|measure|marker)\b|"
    r"^(?:people|patients|individuals|subjects) (?:diagnosed )?with\b|"
    r"\bis unable to exert\b|\bis a heteromer\b|\bsubtypes exist\b|\bfunctions include\b|"
    r"^also involved in\b|\bis implicated in\b|\bare ligand-gated\b", re.I)
_USES_IN_CLINIC_RE = re.compile(r"\b(?:is|are) (?:widely |commonly )?used in clinical\b", re.I)
_SAFETY_RE = re.compile(
    r"\b(?:liver injury|hepatotoxic\w*|cholestasis|toxicity|adverse (?:effect|event|reaction)s?|"
    r"side effects?|lactic acidosis|overdos\w*|contraindicat\w*|hypersensitivity)\b", re.I)
_EFFICACY_RE = re.compile(
    r"\b(?:clinical trial|placebo|from baseline|subjects (?:receiving|taking|treated)|"
    r"patients (?:receiving|taking|treated))\b", re.I)
_PK_RE = re.compile(
    r"\b(?:OCT\d|organic cation transporter|accumulates? in|is (?:rapidly |extensively )?"
    r"(?:absorbed|excreted|eliminated|metaboli[sz]ed|taken up)|bioavailab\w*|half-life|"
    r"(?:uptake|absorption|excretion|elimination|clearance) of (?:the drug|this drug|it\b))",
    re.I)
_HEDGE_ANY_RE = re.compile(
    r"not (?:fully |completely |well[- ]?)?(?:understood|known|identified|established|elucidated)|"
    r"remains? (?:uncertain|unclear|unknown)|under (?:extensive )?(?:study|investigation)|"
    r"may be explained by other|exact mechanism", re.I)
# Continuation openers: such a sentence follows the previous one's category
# ('This leads to…', 'These effects appeared…') unless it names the drug.
_ANAPHOR_RE = re.compile(r"^(?:this|these|such|those|following this|further|however,|"
                         r"basal|both)\b", re.I)
# DrugBank sub-heading labels that PUG-View flattening glues onto sentences.
_GLUED_LABEL_RE = re.compile(
    r"^(?:General effects|Effects? on [^.]{3,90}?\)|Mechanism of action|Pharmacodynamics)\s+"
    r"(?=[A-Z])")
_MISSING_SPACE_RE = re.compile(r"(?<=[a-z]{3}[.!?])(?=[A-Z][a-z])")


def _split_sentences(text: str) -> List[str]:
    """Sentence split that also repairs '…use.Effect' joins and strips glued
    DrugBank sub-heading labels ('General effects Insulin is…')."""
    text = _MISSING_SPACE_RE.sub(" ", re.sub(r"\s+", " ", text or ""))
    # DrugBank sub-headings arrive bold-wrapped ('…use. **Effect on fasting
    # plasma glucose (FPG)…** HbA1c is…') — they're labels, not content
    text = re.sub(r"\s*\*\*[^*]{3,120}\*\*\s*", " ", text)
    # HSDB all-caps notes glued on without punctuation ('Half Life: 2-4 hours
    # IBUPROFEN GIVES 2-(4-…') — break before a run of ≥3 capitalised words
    text = re.sub(r"(?<=[a-z0-9])\s+(?=(?:[A-Z][A-Z0-9\-(),&;]{2,}\s+){2,}[A-Z0-9])", ". ", text)
    # …and an all-caps sentence ends 'IN MAN. The serum…' (capital before '.')
    text = re.sub(r"(?<=[A-Z]{2}[.!?])\s+(?=[A-Z][a-z])", "\x00", text)
    out = []
    for chunk in text.split("\x00"):
        out += _split_sentences_inner(chunk)
    return out


def _split_sentences_inner(text: str) -> List[str]:
    out = []
    out = []
    for s in _SENT_SPLIT_RE.split(text):
        s = _GLUED_LABEL_RE.sub("", _clean_record_sentence(s)).strip()
        if s:
            out.append(s)
    return out


# Verbs that make a "<drug> is a …" sentence mechanistic rather than a class label.
_MECH_VERB_RE = re.compile(r"\bbind|receptor|agonist|antagonist|inhibit|block|modulat|"
                           r"enhanc|potentiat|activat|channel|enzyme", re.I)


def _mentions_drug(s: str, drug_re: Optional[re.Pattern]) -> bool:
    if re.search(r"\b(?:it|its|this drug|the drug|this agent|this medication)\b", s, re.I):
        return True
    return bool(drug_re and drug_re.search(s))


def _classify_pharma_sentence(s: str, drug_re, prev: str, mechanism: bool) -> str:
    named = _mentions_drug(s, drug_re)
    if _RESEARCH_RE.search(s):
        return "research"
    if _TRIVIA_RE.search(s):
        return "background"
    # 'Diazepam is a long-acting "classical" benzodiazepine.' / '… is a
    # benzodiazepine tranquilliser with … properties' — drug class, already on
    # the At-a-Glance card; not mechanism
    if mechanism and drug_re and re.match(r"^\W*" + drug_re.pattern + r"[^.]{0,20}\bis an? ",
                                          s, re.I) \
            and not _MECH_VERB_RE.search(s) and len(s) < 200:
        return "background"
    if mechanism and _USES_IN_CLINIC_RE.search(s):
        return "uses"
    if not named and _ANAPHOR_RE.search(s) and prev in ("research", "background"):
        return prev
    if not named and _BACKGROUND_RE.search(s):
        return "background"
    if mechanism:
        if _SAFETY_RE.search(s):
            return "safety"
        if _EFFICACY_RE.search(s) and re.search(r"\d", s):
            return "efficacy"
        if _PK_RE.search(s):
            return "pk"
        if _USES_RE.search(s) and len(s) <= 140:
            return "uses"
    # a hedge, unless it goes on to explain something ('…remains uncertain:
    # metformin increases cytosolic AMP…' keeps its content)
    if _HEDGE_ANY_RE.search(s) and len(s) <= 240 and ":" not in s:
        return "hedge"
    return "keep"


def _value_from_parts(pub_sents: List[str], wiki_sents: List[str], url: str) -> Any:
    pub = " ".join(pub_sents)
    if wiki_sents:
        return {"__combined__": True, "pubchem": pub, "bullets": wiki_sents,
                "cap": _WIKI_BULLET_CAP, "url": url}
    return pub


def _append_parts(items: Dict[str, Any], heading: str, pub: List[str],
                  wiki: List[str], url: str) -> None:
    """Add moved sentences to a heading (creating it), keeping source attribution."""
    if not (pub or wiki):
        return
    p0, w0, u0 = _split_value_parts(items.get(heading, ""))
    p_all = [p for p in (p0, " ".join(pub)) if p.strip()]
    items[heading] = _value_from_parts([" | ".join(p_all)] if p_all else [],
                                       w0 + wiki, u0 or url)


def refine_pharma_topics(grouped: Dict[str, Dict[str, Any]],
                         drug_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Keep the pharmacology headings on-topic (see block comment above)."""
    items = (grouped or {}).get(_REFINE_GROUP)
    if not items:
        return grouped
    names = sorted({n.strip() for n in drug_names if n and len(n.strip()) >= 4}, key=len,
                   reverse=True)
    # also each name's first word ('Metformin' from 'Metformin hydrochloride')
    names += [n.split()[0] for n in names if len(n.split()[0]) >= 5]
    drug_re = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.I) \
        if names else None

    new_items: Dict[str, Any] = {}
    moved: Dict[str, Tuple[List[str], List[str], str]] = {}
    hedge_kept = False
    for heading, val in items.items():
        canon = _canonical_topic(heading)
        if canon not in ("Mechanism of Action & Pharmacodynamics", "Uses & Indications",
                         "Pharmacokinetics (ADME)"):
            new_items[heading] = val
            continue
        mechanism = canon == "Mechanism of Action & Pharmacodynamics"
        pub_text, bullets, url = _split_value_parts(val)
        kept = {"PubChem": [], "Wikipedia": []}
        for src, texts in (("Wikipedia", bullets),
                           ("PubChem", [p for p in pub_text.split(" | ") if p.strip()])):
            prev = "keep"
            for t in texts:
                # a journal abstract: first-person voice, or ≥2 research
                # markers (µM concentrations, slice cultures, "these findings")
                abstract = src == "PubChem" and (bool(_FIRST_PERSON_RE.search(t))
                                                 or len(_RESEARCH_RE.findall(t)) >= 2)
                for s in (_split_sentences(t) if src == "PubChem" else [t.strip()]):
                    cat = "research" if abstract else \
                        _classify_pharma_sentence(s, drug_re, prev, mechanism)
                    prev = cat
                    if cat == "hedge":
                        if hedge_kept or not mechanism:
                            continue
                        hedge_kept, cat = True, "keep"
                    if cat == "keep":
                        kept[src].append(s)
                    elif cat == "background":
                        continue
                    else:
                        dest = moved.setdefault(cat, ([], [], ""))
                        (dest[1] if src == "Wikipedia" else dest[0]).append(s)
                        if src == "Wikipedia" and url:
                            moved[cat] = (dest[0], dest[1], url)
        new_val = _value_from_parts(kept["PubChem"], kept["Wikipedia"], url)
        if _plain_text(new_val):
            new_items[heading] = new_val

    def _heading_for(canon_name: str) -> str:
        return next((h for h in new_items if _canonical_topic(h) == canon_name), canon_name)

    for cat, (pub, wiki, url) in moved.items():
        if cat == "research":
            _append_parts(new_items, _RESEARCH_HEADING, pub, wiki, url)
        elif cat == "efficacy":
            _append_parts(new_items, _EFFICACY_HEADING, pub, wiki, url)
        elif cat == "pk":
            _append_parts(new_items, _heading_for("Pharmacokinetics (ADME)"), pub, wiki, url)
        elif cat == "uses":
            _append_parts(new_items, _heading_for("Uses & Indications"), pub, wiki, url)
        elif cat == "safety":
            tox = grouped.setdefault("Toxicity & Health", {})
            head = next((h for h in tox if "adverse effect" in _norm_head(h)), "Adverse effects")
            _append_parts(tox, head, pub, wiki, url)
    grouped[_REFINE_GROUP] = new_items
    return grouped


def _render_wiki_bullets(val: Any, key: str, attribution: bool = True):
    """Render a Wikipedia block's bullets (attributed, capped, show-more toggle).
    Shared by standalone wiki headings and the wiki half of a combined heading."""
    url = val.get("url", "")
    if attribution:
        src = f"[Wikipedia ↗]({url})" if url else "Wikipedia"
        st.markdown(f"{src} `CC BY-SA`")
    bullets = val.get("bullets") or []
    cap = val.get("cap") or _WIKI_BULLET_CAP
    # Show the most-relevant (leading) bullets; the rest sit behind an inline
    # toggle so a long article never becomes a wall of text. (A checkbox, not
    # st.expander — Streamlit forbids nesting expanders and each group already
    # renders inside one.)
    head, tail = bullets[:cap], bullets[cap:]
    if head:
        st.markdown("\n".join(f"- {b}" for b in head), unsafe_allow_html=True)
    if tail and key and st.checkbox(f"show {len(tail)} more",
                                    key=f"wm_{key}", value=False):
        st.markdown("\n".join(f"- {b}" for b in tail), unsafe_allow_html=True)


def _render_pubchem_value(val: Any, key: str = ""):
    """Render a PubChem heading value as point-form (not raw JSON), capped to its
    leading (most-pertinent) bullets with a 'show more' for the rest so a verbose
    heading isn't a wall of paragraphs."""
    if isinstance(val, (dict, list)):
        st.json(val)
        return
    # PUG-View flattening joins multiple entries with ' | '; each entry may
    # itself be a dense paragraph, so point-form every one of them — after
    # ordering so substantive compound facts lead and boilerplate/pointers sink.
    entries = [p.strip() for p in str(val).split(" | ") if p.strip()]
    entries.sort(key=_entry_rank)                    # stable: ties keep source order
    lines: List[str] = []
    for p in entries:
        lines.extend(_pointify(p))
    if not lines:
        return

    def _emit(chunk: List[str]):
        st.markdown("\n".join(sanitize_html_fragment(ln) if _BLEACH_AVAILABLE else ln
                              for ln in chunk), unsafe_allow_html=True)

    # Group each top-level bullet with its nested sub-bullets into one block, then
    # cap by block so we never split a bullet from its children.
    blocks: List[List[str]] = []
    for ln in lines:
        if ln.startswith("- "):
            blocks.append([ln])
        elif blocks:
            blocks[-1].append(ln)
        else:
            blocks.append([ln])
    head = [ln for blk in blocks[:_PUBCHEM_BLOCK_CAP] for ln in blk]
    tail = [ln for blk in blocks[_PUBCHEM_BLOCK_CAP:] for ln in blk]
    _emit(head)
    if tail and key and st.checkbox(f"show {len(blocks) - _PUBCHEM_BLOCK_CAP} more",
                                    key=f"pc_{key}", value=False):
        _emit(tail)


# --- One merged list for a heading both PubChem and Wikipedia cover --------
# Instead of a PubChem block then a separate Wikipedia block, every sentence
# from both goes into one de-duplicated list, ordered so the clearest
# statements lead: on-topic first (what the heading asks), then general
# statements, then statistics-heavy trial detail. Order within a tier keeps the
# source order (so a step-by-step mechanism stays in sequence). Each line
# carries a small source tag; Wikipedia's links to the article (CC BY-SA).
_TOPIC_CUES = (
    (re.compile(r"mechanism|pharmacodynamic|mode of action", re.I),
     re.compile(r"\bbind|receptor|agonist|antagonist|inhibit|block|enzyme|"
                r"transcription|mechanism|acts? (?:by|on|as)|works? by|"
                r"modulat|activat|suppress", re.I)),
    (re.compile(r"\buses?\b|indication|therapeutic", re.I),
     re.compile(r"\bindicated\b|\btreat|\bused (?:for|to|in)\b|management of|"
                r"\brelie|\bprevent|therapy for|prescribed", re.I)),
    (re.compile(r"pharmacokinetic|adme|absorption|metabolism|half", re.I),
     re.compile(r"absorb|bioavailab|half-life|metaboli|excret|eliminat|"
                r"distribut|protein[- ]bound|clearance|cyp\d", re.I)),
)
# Markers of trial-statistics detail: numbers with units, CIs, study designs.
_DETAIL_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mcg|µg|mg|ml|msec|ms|%|hours?|weeks?|subjects|patients)\b|"
    r"\bCI\b|\bAUC|\bplacebo\b|\brandomi[sz]ed\b|\bdouble-blind\b|\bcross-over\b|"
    r"\bstatistically\b|\btrial\b|\bQTc?F?\b", re.I)
# One of these alone marks preclinical / lab detail (not what a reader wants first).
_STRONG_DETAIL_RE = re.compile(
    r"\bmg/kg\b|\bin vitro\b|\bin vivo\b|\b(?:rats?|mice|mouse|rabbits?|dogs?)\b|"
    r"\d+(?:\.\d+)?[- ]?(?:fold|times)\b|\bCI\b|\bplacebo\b|\brandomi[sz]ed\b|"
    r"\bintraperitoneal|\bpretreatment\b", re.I)
# "The exact mechanism … is unknown" — true, but a poor opening line.
_HEDGE_RE = re.compile(r"\b(?:is|are|remains?) (?:not (?:fully |well[- ]?|completely |yet )?"
                       r"(?:known|understood|established|identified|elucidated|clear)"
                       r"|unknown|unclear|poorly understood)\b", re.I)
_COMBINED_CAP = 6


def _clean_record_sentence(s: str) -> str:
    """Tidy HSDB editing marks: 'Ibuprofen ... /is/ used' → 'Ibuprofen is used',
    and drop trailing '/Included in US product labeling/'-style source notes."""
    s = re.sub(r"\s*\.\.\.\s*/([a-z ]{1,20})/\s*", r" \1 ", s)
    s = re.sub(r"\s*/[A-Z][^/]{3,80}/\s*$", "", s)
    s = re.sub(r"\s*\[DB\d{5}\]", "", s)              # DrugBank cross-ref ids
    s = re.sub(r"\[([A-Za-z][^\[\]]{1,40})\]", r"\1", s)  # '[diphenhydramine]' link marks
    return re.sub(r"\s{2,}", " ", s).strip()


def _combined_items(heading: str, val: Dict[str, Any]) -> List[Tuple[str, str]]:
    """[(sentence, source)] for a combined heading: both sources' sentences,
    de-duplicated and ordered by tier (on-topic → general → detail)."""
    raw: List[Tuple[str, str]] = []
    entries = [p.strip() for p in str(val.get("pubchem") or "").split(" | ") if p.strip()]
    entries.sort(key=_entry_rank)
    for e in entries:
        if _entry_rank(e) == 2:                      # boilerplate / pointers
            continue
        for s in _split_sentences(e):
            raw.append((s, "PubChem"))
    # Wikipedia's plain-language sentences first so, within a tier, they lead
    # (PubChem's tend to be denser label/monograph prose).
    raw = [(b.strip(), "Wikipedia") for b in val.get("bullets") or []] + raw

    topic = next((cue for hp, cue in _TOPIC_CUES if hp.search(heading or "")), None)
    dd = _SentenceDeduper()
    for kp in val.get("key_points") or []:          # don't repeat the summary
        dd.accept(_content_words(kp[0]))
    tiers: Dict[int, List[Tuple[str, str]]] = {0: [], 1: [], 2: []}
    prev_src, prev_tier = None, None
    for s, src in raw:
        if _is_lowvalue_sentence(s):                 # 'FDA Label', fragments
            continue
        ws = _content_words(s)
        if dd.is_dup(ws) or dd.is_restatement(ws):
            continue
        dd.accept(ws)
        if len(_DETAIL_RE.findall(s)) >= 2 or _STRONG_DETAIL_RE.search(s):
            tier = 2
        elif _HEDGE_RE.search(s):
            tier = 1
        elif topic is not None and topic.search(s):
            tier = 0
        else:
            tier = 1
        # keep explanation chains intact: 'This increased chloride influx…',
        # 'As a result, …' stay right after the sentence they continue
        if src == prev_src and prev_tier is not None and tier != 2 \
                and _CONTINUATION_RE.search(s):
            tier = prev_tier
        tiers[tier].append((s, src))
        prev_src, prev_tier = src, tier
    return tiers[0] + tiers[1] + tiers[2]


_CONTINUATION_RE = re.compile(
    r"^(?:this|these|such|as a result|thus|hence|therefore|in turn|consequently|"
    r"entry of|the (?:increased|resulting|reduced))\b", re.I)


def mechanism_key_points(chembl: Dict[str, Any], label: Dict[str, Any]
                         ) -> List[Tuple[str, str, str]]:
    """2–4 line summary of how the drug works, from the most concise curated
    sources: ChEMBL's mechanism record(s) and the FDA label's own Mechanism of
    Action section. [(text, source, url)]; [] when neither exists."""
    out: List[Tuple[str, str, str]] = []
    for me in (chembl or {}).get("mechanisms") or []:
        mech = (me.get("mechanism") or "").strip()
        if not mech or mech.lower() == "unknown":
            continue
        tgt = (me.get("target_name") or "").strip()
        act = (me.get("action_type") or "").strip()
        # 'GABA-A receptor; anion channel' → 'GABA-A receptor (anion channel)'
        tgt_txt = re.sub(r";\s*(.+)$", r" (\1)", tgt)
        if act and tgt:
            # readable form: 'Positive allosteric modulator of GABA-A receptor
            # (anion channel)', 'Inhibitor of cyclooxygenase' (acronyms kept)
            if len(tgt_txt) > 1 and tgt_txt[0].isupper() and tgt_txt[1].islower():
                tgt_txt = tgt_txt[0].lower() + tgt_txt[1:]
            text = f"{act.capitalize()} of {tgt_txt}"
        else:
            text = mech[0].upper() + mech[1:]
            if tgt and tgt.lower() not in mech.lower():
                text += f" (target: {tgt_txt})"
        out.append((text + ".", "ChEMBL", (chembl or {}).get("url", "")))
        if len(out) >= 2:
            break
    moa = (label or {}).get("mechanism_of_action") or ""
    # drop the label's own section heading ('12.1 Mechanism of Action …')
    moa = re.sub(r"^\s*(?:\d+(?:\.\d+)*\s*)?Mechanism of Action\s*[:.]?\s*", "", moa, flags=re.I)
    if moa:
        sents = [s for s in _split_sentences(moa) if not _is_lowvalue_sentence(s)]
        # skip bare hedges and lab/trial detail ('…29.9 times that of
        # dexamethasone' in vitro) — those belong in the detail list
        firm = [s for s in sents if not (_HEDGE_ANY_RE.search(s) and len(s) < 140)
                and not (_HEDGE_RE.search(s) and len(s) < 140)
                and not re.search(r"\bthese (?:findings|results|data|effects)\b", s, re.I)
                and not _STRONG_DETAIL_RE.search(s) and len(_DETAIL_RE.findall(s)) < 2
                and not _BACKGROUND_RE.search(s)]           # disease background
        # sentences that say what the drug DOES (binds/inhibits/activates…) first
        firm.sort(key=lambda s: not _MECH_VERB_RE.search(s))
        for s in (firm or sents)[:2]:
            out.append((_first_sentence(s, 320), "FDA label", (label or {}).get("url", "")))
    return out[:4]


# --- PK "at a glance": one representative sentence per standard ADME fact ---
_PK_FACTS = (
    ("Absorption", re.compile(r"bioavailab|\babsorbed\b|\babsorption\b|peak plasma|\bTmax\b", re.I)),
    ("Protein binding", re.compile(r"protein[- ]bound|protein binding|bound to plasma protein|"
                                   r"binds? (?:extensively |highly )?to plasma protein", re.I)),
    ("Distribution", re.compile(r"volume of distribution|\bcross(?:es)? (?:the |both the )?"
                                r"(?:blood|placent)|widely distributed|distributed (?:into|throughout)",
                                re.I)),
    ("Metabolism", re.compile(r"\bCYP\w+|metaboli[sz]ed (?:by|in|via|mainly|primarily|extensively|"
                              r"predominantly|largely)|"
                              r"demethylat|hydroxylat|glucuronid|active metabolite", re.I)),
    ("Half-life", re.compile(r"half[- ]li(?:fe|ves)", re.I)),
    ("Excretion", re.compile(r"excreted (?:\w+ )?(?:in|via|through) (?:the )?(?:urine|feces|faeces|bile)|"
                             r"renal(?:ly)? (?:excret|clear|elimin)|eliminated (?:\w+ )?(?:in|via) "
                             r"(?:the )?(?:urine|feces)|excreted unchanged", re.I)),
)
_PK_INTERACTION_RE = re.compile(r"\binhibit\w*|\binduc(?:er|ers|es|ed|ing|tion)\b|"
                                r"co-?administ\w*|concomitant", re.I)
_PK_METAB_STATEMENT_RE = re.compile(
    r"\b(?:is|are) (?:\w+ )?metaboli[sz]ed\b|\bundergoes\b.{0,40}\bmetabolism\b|"
    r"\bmetaboli[sz]ed (?:mainly|primarily|predominantly|extensively|largely)\b", re.I)
# Sentences about special populations / unusual routes make poor headline facts.
_PK_SPECIAL_RE = re.compile(
    r"\b(?:neonat\w*|infants?|premature|children|paediatric|pediatric|elderly|"
    r"geriatric|cirrhosis|hepatic (?:impairment|fibrosis|disease)|renal (?:impairment|"
    r"disease|failure)|kidney (?:or liver )?disease|(?:liver|renal|kidney|hepatic) function|"
    r"compromised|pregnan\w*|poor metabolizers|PMs|"
    r"rectal|gel|suppositor\w*|intramuscular|IM\b|food|fat meal)\b", re.I)


def _pk_key_facts(items: List[Tuple[str, Any]]
                  ) -> Tuple[List[Tuple[str, str, Any]], List[Tuple[str, Any]]]:
    """Pick the clearest sentence for each ADME fact → ([(label, sentence, src)],
    remaining items). Prefers short, numeric, general-population sentences.
    `src` is carried through untouched, so callers can pass (source, url, …)
    and keep them attached even though a headline's 'Half Life:'-style prefix
    may be stripped from the returned sentence."""
    used: set = set()
    facts: List[Tuple[str, str, str]] = []
    for label, rx in _PK_FACTS:
        best, best_score = None, None
        for idx, (s, src) in enumerate(items):
            if idx in used or not rx.search(s):
                continue
            score = len(s) / 120.0
            if _PK_SPECIAL_RE.search(s):
                score += 6
            # a sentence that's really about another fact ('…absorbed drug
            # being protein-bound') shouldn't headline this one
            if any(orx.search(s) for olbl, orx in _PK_FACTS if olbl != label):
                score += 3
            if label == "Excretion" and re.search(r"urine", s, re.I):
                score -= 0.5
            if label == "Distribution" and re.search(r"blood[-–‐ ]brain|volume of distribution", s, re.I):
                score -= 0.5
            if label in ("Absorption", "Protein binding", "Half-life") and not re.search(r"\d", s):
                score += 2
            if label == "Half-life" and re.search(r"distribut\w* half|alpha", s, re.I):
                score += 3                       # the elimination half-life is the headline
            if label == "Metabolism" and re.search(r"CYP", s) and re.search(r"metabolite", s, re.I):
                score -= 1.5                     # enzymes AND active metabolites together
            if label == "Metabolism":
                # 'CYP2D6 can be inhibited by…' / 'X is a weak inhibitor of CYP2D6' are
                # interaction facts; the headline says how the drug itself is cleared
                if _PK_INTERACTION_RE.search(s):
                    score += 4
                if _PK_METAB_STATEMENT_RE.search(s):
                    score -= 1.5
            if re.match(r"^(?:it|its|these|this|they)\b", s, re.I):
                score += 1                       # orphaned pronoun reads badly alone
            letters = [c for c in s if c.isalpha()]
            if letters and sum(c.isupper() for c in letters) / len(letters) > 0.4:
                score += 3                       # HSDB all-caps 'shouting' text
            if best_score is None or score < best_score:
                best, best_score = idx, score
        if best is not None:
            used.add(best)
            s = items[best][0]
            # the fact label already says it: 'Route of Elimination: Ibuprofen…'
            # → 'Ibuprofen…' (kept when only a value follows, 'Half Life: 2-4 h')
            stripped = re.sub(r"^(?:Half[- ]Life|Route of Elimination|Volume of Distribution|"
                              r"Protein Binding|Absorption|Clearance|Metabolism)\s*:\s*", "", s,
                              flags=re.I)
            if stripped != s and stripped[:1].isalpha():
                s = stripped[0].upper() + stripped[1:]
            facts.append((label, s, items[best][1]))
    rest = [it for i, it in enumerate(items) if i not in used]
    # drop detail lines that merely restate a headline fact
    dd = _SentenceDeduper()
    for _l, s, _src in facts:
        dd.accept(_content_words(s))
    rest = [(s, src) for s, src in rest if not dd.is_restatement(_content_words(s))]
    return facts, rest


_PK_ROUTE_RE = re.compile(r"\b(?:intravenous\w*|IV\b|intramuscular\w*|IM\b|rectal\w*|"
                          r"suppositor\w*|onset of action|duration of\b|food|fasting|fat meal|"
                          r"administered (?:orally|with))", re.I)


def _pk_detail_groups(rest: List[Tuple[str, str]]) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Bucket the PK detail lines: special populations and routes/onset/food
    first-class (they're most of the tail), then the ADME facts, then other."""
    order = ["Absorption", "Protein binding", "Distribution", "Metabolism", "Half-life",
             "Excretion", "Routes, onset & food", "Special populations & genetics", "Other"]
    buckets: Dict[str, List[Tuple[str, str]]] = {k: [] for k in order}
    facts = dict(_PK_FACTS)
    # most specific first: a half-life sentence mentioning a metabolite is
    # about half-life; a clearance/urine sentence is about excretion
    bucket_rx = [
        ("Half-life", facts["Half-life"]),
        ("Protein binding", facts["Protein binding"]),
        ("Excretion", re.compile(facts["Excretion"].pattern + r"|\bclearance\b|\burine\b", re.I)),
        ("Distribution", re.compile(facts["Distribution"].pattern +
                                    r"|placent|breast milk|adipose|stored|accumulat|redistribut",
                                    re.I)),
        ("Metabolism", re.compile(facts["Metabolism"].pattern +
                                  r"|metaboli[sz]ed to|metabolites?\b", re.I)),
        ("Absorption", facts["Absorption"]),
    ]
    for s, src in rest:
        if re.search(r"\b(?:neonat\w*|infants?|premature|children|paediatric|pediatric|"
                     r"elderly|geriatric|cirrhosis|hepatic (?:fibrosis|impairment|disease)|"
                     r"hepatitis|renal impairment|kidney|liver disease|poor metabolizers|"
                     r"PMs|EMs|polymorphic|years of age)\b", s, re.I):
            buckets["Special populations & genetics"].append((s, src))
        elif _PK_ROUTE_RE.search(s):
            buckets["Routes, onset & food"].append((s, src))
        else:
            lbl = next((l for l, rx in bucket_rx if rx.search(s)), "Other")
            buckets[lbl].append((s, src))
    return [(k, buckets[k]) for k in order if buckets[k]]


def attach_mechanism_key_points(grouped: Dict[str, Dict[str, Any]],
                                chembl: Dict[str, Any], label: Dict[str, Any]) -> None:
    """Store the key-point summary on the mechanism heading (converting a plain
    PubChem value into the merged-list form so it renders summary-first)."""
    items = (grouped or {}).get(_REFINE_GROUP) or {}
    kp = mechanism_key_points(chembl, label)
    for heading, val in list(items.items()):
        canon = _canonical_topic(heading)
        if canon not in ("Mechanism of Action & Pharmacodynamics", "Pharmacokinetics (ADME)"):
            continue
        pub, bullets, url = _split_value_parts(val)
        new = {"__combined__": True, "pubchem": pub, "bullets": bullets,
               "cap": _WIKI_BULLET_CAP, "url": url}
        if canon == "Pharmacokinetics (ADME)":
            new["pk"] = True                     # summary picked at render time
        else:
            new.update({"key_points": kp, "mechanism": True})
        items[heading] = new


def _render_combined(heading: str, val: Dict[str, Any], key: str = ""):
    items = _combined_items(heading, val)
    if not items and not val.get("key_points"):
        return
    url = val.get("url", "")
    tag_css = "opacity:0.55;font-size:0.78em;white-space:nowrap"

    def _line(s: str, src: str) -> str:
        if _BLEACH_AVAILABLE:
            s = sanitize_html_fragment(s)
        tag = (f"<a href='{url}' style='{tag_css}'>Wikipedia</a>" if src == "Wikipedia" and url
               else f"<span style='{tag_css}'>{src}</span>")
        return f"- {s} {tag}"

    if val.get("mechanism"):
        # Summary-first: a few key points always visible, ALL detail behind
        # one toggle (a well-studied drug has dozens of on-topic sentences).
        kp = [(t, s, u) for t, s, u in val.get("key_points") or []]
        if len(kp) < 2:
            # thin curated summary (no label MoA section, or ChEMBL only):
            # top up with the clearest on-topic sentences, to 3 in all
            n = 3 - len(kp)
            kp += [(s, src, url if src == "Wikipedia" else "") for s, src in items[:n]]
            items = items[n:]
        st.markdown("*Key points*")

        def _kp_line(t, s, u):
            t = sanitize_html_fragment(t) if _BLEACH_AVAILABLE else t
            tag = (f"<a href='{u}' style='{tag_css}'>{s}</a>" if u
                   else f"<span style='{tag_css}'>{s}</span>")
            return f"- {t} {tag}"

        st.markdown("\n".join(_kp_line(*k) for k in kp), unsafe_allow_html=True)
        if items and key and st.checkbox(f"Detailed mechanism — {len(items)} more point(s)",
                                         key=f"cm_{key}", value=False):
            st.markdown("\n".join(_line(*it) for it in items), unsafe_allow_html=True)
        if any(src == "Wikipedia" for _, src in items):
            st.caption("Detail merged from PubChem and Wikipedia (CC BY-SA); duplicates removed.")
        return

    if val.get("pk"):
        # Summary-first PK: one labelled headline sentence per ADME fact,
        # everything else (special populations, food effects, routes, study
        # detail) behind one toggle.
        facts, rest = _pk_key_facts(items)
        if facts:
            st.markdown("*At a glance*")
            st.markdown("\n".join(f"- **{lbl}:** {_line(s, src)[2:]}" for lbl, s, src in facts),
                        unsafe_allow_html=True)
        else:
            rest = items
        if rest and key and st.checkbox(f"Detailed pharmacokinetics — {len(rest)} more point(s)",
                                        key=f"cm_{key}", value=False):
            # grouped under the same ADME labels so the long tail is scannable
            md = []
            for lbl, group in _pk_detail_groups(rest):
                md.append(f"**{lbl}**")
                md += [_line(*it) for it in group]
                md.append("")
            st.markdown("\n".join(md), unsafe_allow_html=True)
        if any(src == "Wikipedia" for _, src in items):
            st.caption("Merged from PubChem and Wikipedia (CC BY-SA); duplicates removed.")
        return

    head, tail = items[:_COMBINED_CAP], items[_COMBINED_CAP:]
    st.markdown("\n".join(_line(*it) for it in head), unsafe_allow_html=True)
    if tail and key and st.checkbox(f"show {len(tail)} more (incl. trial detail)"
                                    if any(_DETAIL_RE.search(s) for s, _ in tail)
                                    else f"show {len(tail)} more",
                                    key=f"cm_{key}", value=False):
        st.markdown("\n".join(_line(*it) for it in tail), unsafe_allow_html=True)
    if any(src == "Wikipedia" for _, src in items):
        st.caption("Merged from PubChem and Wikipedia (CC BY-SA); duplicates removed.")


def _render_heading_value(heading: str, val: Any, key: str = ""):
    """Render one PUG-View heading + value as a readable block (not raw JSON)."""
    # A Wikipedia section merged into the group: bullet its prose and attribute.
    if isinstance(val, dict) and val.get("__wiki__"):
        st.markdown(f"**{heading}**")
        _render_wiki_bullets(val, key)
        return
    # A PubChem heading that Wikipedia also covered under the same name: show the
    # PubChem body, then the Wikipedia bullets under one heading (not two).
    if isinstance(val, dict) and val.get("__combined__"):
        st.markdown(f"**{heading}**")
        _render_combined(heading, val, key)
        return
    st.markdown(f"**{heading}**")
    _render_pubchem_value(val, key)


def _plain_text(val: Any) -> str:
    """Strip markup/whitespace from a heading value for snippet/filter use."""
    if isinstance(val, dict) and val.get("__combined__"):
        text = _plain_text(val.get("pubchem")) + " " + " ".join(val.get("bullets") or [])
        return re.sub(r"\s+", " ", text).strip()
    if isinstance(val, dict) and val.get("__wiki__"):
        text = " ".join(val.get("bullets") or [])
        return re.sub(r"\s+", " ", re.sub(r"[-*#`]|<[^>]+>", " ", text)).strip()
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str(val))).strip()


def _lead_snippet(items: Dict[str, Any], limit: int = 90) -> str:
    """A one-line highlight for a group: the first sentence of its most
    pertinent (highest-priority) substantive heading. Shown in the collapsed
    expander label so the record is scannable without opening everything."""
    for heading, val in _ordered_items(items):
        text = _plain_text(val)
        if len(text) < 20:  # skip bare IDs / short codes
            continue
        sentence = re.split(r"(?<=[.!?])\s+", text)[0]
        if len(sentence) > limit:
            sentence = sentence[:limit].rstrip() + "…"
        return sentence
    return ""


def render_group_expanders(grouped: Dict[str, Dict[str, Any]], group_names, cid,
                           key_prefix: str = "", expand_first: bool = True):
    """Render the grouped PUG-View headings as expanders for the given groups.

    When a section is large, a live filter box narrows the headings (and their
    text) so the record is navigable rather than a wholesale scroll. Each
    collapsed group also shows a one-line highlight of its top heading.
    `expand_first=False` keeps every group collapsed (secondary material)."""
    def _section_url(cid_val, heading_text):
        if not cid_val:
            return None
        return (f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid_val}"
                f"#section={urlquote(str(heading_text), safe='')}&fullscreen=true")

    # most pertinent groups first; the top one opens expanded by default
    ordered_groups = sorted(group_names,
                            key=lambda g: -_group_priority(grouped.get(g, {})))

    # Offer a filter once a section carries enough headings to be worth scanning.
    total_headings = sum(len(grouped.get(g, {})) for g in ordered_groups)
    query = ""
    if total_headings > 5:
        query = st.text_input(
            "🔎 Filter this section", key=f"filter_{key_prefix}",
            placeholder="search headings & text…",
            label_visibility="collapsed",
        ).strip().lower()

    rendered_any = False
    matched_any = False
    for group_name in ordered_groups:
        items = grouped.get(group_name, {})
        if not items:
            continue
        if query:
            items = {h: v for h, v in items.items()
                     if query in h.lower() or query in _plain_text(v).lower()}
            if not items:
                continue
        matched_any = True
        # When filtering, open every matching group; otherwise only the top one.
        first = bool(query) or (expand_first and not rendered_any)
        rendered_any = True
        lead = _lead_snippet(items)
        label = f"{group_name} — {len(items)} heading(s)"
        if lead:
            label += f"  ·  {lead}"
        with st.expander(label, expanded=first):
            for i, (heading, val) in enumerate(_ordered_items(items)):
                if i:
                    st.markdown(_THIN_RULE, unsafe_allow_html=True)
                nh = heading.strip().lower()
                # Replace bulky numeric spectra with a deep-link to PubChem.
                if _is_spectral_heading(nh):
                    url = _section_url(cid or "", heading)
                    st.markdown(f"**{heading}**")
                    if url:
                        st.markdown(f"[Open spectra on PubChem ↗]({url})")
                    else:
                        st.caption("(no CID — view on PubChem)")
                else:
                    _render_heading_value(
                        heading, val, key=f"{key_prefix}_{group_name}_{i}")
            try:
                as_bytes = json.dumps(items, ensure_ascii=False, indent=2).encode("utf8")
                st.download_button(
                    f"⬇ Download '{group_name}' (JSON)", data=as_bytes,
                    file_name=f"{cid or 'compound'}_{group_name.replace(' ', '_')}.json",
                    mime="application/json", key=f"dl_{key_prefix}_{group_name}",
                )
            except Exception:
                pass
    if not matched_any:
        if query:
            st.caption(f"No headings match “{query}” in this section.")
        else:
            st.caption("No data in this category for this compound.")

# ---------- Provenance-tagged inference rendering --------------------------
# How each RDKit descriptor is obtained — shown so estimates are never mistaken
# for measured facts. "Derived" = exact from the graph; "Estimated" = grounded
# group-contribution method; "Heuristic" = a scoring rule, not a measurement.
DESCRIPTOR_BASIS = {
    "Molecular weight": "Derived", "Exact mass": "Derived",
    "H-bond donors": "Derived", "H-bond acceptors": "Derived",
    "Rotatable bonds": "Derived", "Rings": "Derived",
    "Aromatic rings": "Derived", "Fraction Csp³": "Derived",
    "Heavy atoms": "Derived", "Formal charge": "Derived",
    "Undefined stereocenters": "Derived",
    "cLogP (Crippen)": "Estimated (Crippen group contributions)",
    "TPSA": "Estimated (Ertl topological method)",
    "QED (drug-likeness)": "Heuristic (desirability score)",
    "Lipinski violations": "Rule (from derived counts)",
}

# Short captions explaining the two honest provenance classes we surface.
_PROV_LEGEND = (
    "**Provenance:** ✓ *Derived* = exact consequence of the structure · "
    "◇ *Inferred* = deduced from functional groups using chemical principles "
    "(shown with its reasoning). No statistical/ML predictions are mixed in."
)

# Distinct soft highlight colours (RGB 0..1) cycled across functional groups.
_HL_PALETTE = [
    (1.00, 0.70, 0.70), (0.70, 0.85, 1.00), (0.75, 1.00, 0.75),
    (1.00, 0.90, 0.60), (0.85, 0.75, 1.00), (0.70, 1.00, 0.95),
    (1.00, 0.78, 0.95), (0.90, 0.90, 0.70),
]
_ACID_COLOR = (1.00, 0.55, 0.55)   # red-ish → loses a proton
_BASE_COLOR = (0.55, 0.70, 1.00)   # blue-ish → gains a proton


def _wkey(smiles: str) -> str:
    """Per-structure widget-key suffix (keeps slider/selectbox state pinned to
    the current molecule and reset when the compound changes)."""
    return str(abs(hash(smiles)) % 10_000_000)


def render_structure_inference(smiles: str):
    """Render the structure-grounded inference block (functional groups,
    acid/base & ionization, stability, qualitative solubility) — now with an
    interactive structure highlighter and a live pH titration explorer."""
    bundle = ci.infer_all(smiles)
    if not bundle:
        st.caption("Could not parse the structure for inference.")
        return

    st.caption(_PROV_LEGEND)
    wk = _wkey(smiles)

    groups = bundle.get("functional_groups", [])
    if groups:
        st.markdown("**Functional groups detected** &nbsp; `✓ Derived`",
                    unsafe_allow_html=True)
        st.caption("Pick a row in the highlighter to see exactly which atoms it "
                   "refers to — the structure is the legend.")
        # Build the highlight option list (functional groups + the acid/base map).
        hl_options = ["(whole molecule)"]
        ab_centres = bundle.get("acid_base", {}).get("centres", [])
        if any(c["atom_indices"] for c in ab_centres):
            hl_options.append("⚖ Acid/base centres (red=acid, blue=base)")
        hl_options += [f"{g['name']} ×{g['count']}" for g in groups]

        col_img, col_tbl = st.columns([1, 1.25])
        with col_img:
            choice = st.selectbox("Highlight on structure", hl_options,
                                  key=f"hl_{wk}")
            highlights = []
            if choice.startswith("⚖"):
                for c in ab_centres:
                    if c["atom_indices"]:
                        col = _ACID_COLOR if c["role"] == "acid" else _BASE_COLOR
                        highlights.append((c["atom_indices"], col))
            elif choice != "(whole molecule)":
                for i, g in enumerate(groups):
                    if choice == f"{g['name']} ×{g['count']}":
                        highlights.append((g["atom_indices"],
                                           _HL_PALETTE[i % len(_HL_PALETTE)]))
                        break
            png = render_2d_highlight(smiles, highlights)
            if png:
                st.image(png, width='stretch')
        with col_tbl:
            rows = [{"Group": g["name"], "Count": g["count"],
                     "Implication": g["note"]} for g in groups]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True,
                         height=min(360, 80 + 35 * len(rows)))

    ab = bundle.get("acid_base", {})
    centres = ab.get("centres", [])
    st.markdown("**Acid/base & ionization** &nbsp; `◇ Inferred`",
                unsafe_allow_html=True)
    if centres:
        ph = st.slider("pH", min_value=0.0, max_value=14.0, value=7.4, step=0.1,
                       key=f"ph_{wk}",
                       help="Drag to titrate — the charge curve, net charge and "
                            "each centre's state below update live.")
        ab = ci.infer_acid_base(smiles, ph=ph)
        centres = ab.get("centres", [])
        cont_charge = ci.net_charge_at_ph(smiles, ph)

        m1, m2 = st.columns([1, 2])
        with m1:
            net = ab.get("net_charge_estimate", 0)
            sign = f"+{net}" if net > 0 else str(net)
            st.metric(f"Dominant net charge @ pH {ph:g}", sign,
                      help="Coarse integer charge of the dominant microspecies. "
                           "The curve uses a smooth Henderson–Hasselbalch sum.")
            st.caption(f"Continuous estimate: **{cont_charge:+.2f}**")
        with m2:
            curve = ci.charge_curve(smiles)
            if curve:
                cdf = pd.DataFrame(curve)
                base = alt.Chart(cdf).mark_line(color="#4C78A8").encode(
                    x=alt.X("pH:Q", scale=alt.Scale(domain=[0, 14])),
                    y=alt.Y("net_charge:Q", title="net charge (HH)"),
                    tooltip=["pH", "net_charge"],
                )
                rule = alt.Chart(pd.DataFrame({"pH": [ph]})).mark_rule(
                    color="#E45756", strokeDash=[4, 3]).encode(x="pH:Q")
                chart = (base + rule).properties(width="container", height=220)
                st.altair_chart(chart)
        if ab.get("net_charge_note"):
            st.caption(ab["net_charge_note"])
        for c in centres:
            lo, hi = c["pka"]
            kind = "acid" if c["role"] == "acid" else "base (pKaH)"
            st.markdown(
                f"- **{c['name']}** ×{c['count']} — {kind} pKa ≈ {lo:g}–{hi:g} → "
                f"*{c['state']}*"
            )
            st.caption(c["reasoning"])
    else:
        st.caption("No ionisable centres recognised — expected to be neutral "
                   "across the physiological pH range.")

    flags = bundle.get("stability_flags", [])
    if flags:
        st.markdown("**Stability / reactivity expectations** &nbsp; `◇ Inferred`",
                    unsafe_allow_html=True)
        for f in flags:
            st.markdown(f"- **{f['label']}** (from {f['basis']})")
            st.caption(f["detail"])

    sol = bundle.get("solubility")
    if sol:
        st.markdown("**Aqueous solubility tendency** &nbsp; `◇ Inferred`",
                    unsafe_allow_html=True)
        st.markdown(f"- {sol['tendency']}")
        for fac in sol["factors"]:
            st.caption(f"• {fac}")
        st.caption(sol["caveat"])


# Drug-likeness rules rendered as a chart: (label, value-getter, limit, "≤"/"≥").
# Lipinski Rule-of-5 + Veber oral-absorption guides — each a hard threshold, so
# "how drug-like" reads as a shape rather than a list of numbers.
_DRUGLIKE_RULES = [
    ("Mol. weight", "Molecular weight", 500.0, "Da"),
    ("cLogP", "cLogP (Crippen)", 5.0, ""),
    ("H-bond donors", "H-bond donors", 5.0, ""),
    ("H-bond acceptors", "H-bond acceptors", 10.0, ""),
    ("TPSA", "TPSA", 140.0, "Å²"),
    ("Rotatable bonds", "Rotatable bonds", 10.0, ""),
]


def render_druglikeness_chart(dm: Dict[str, Any]):
    """Bar chart of each drug-likeness metric as a fraction of its limit.

    Bars left of the dashed line are within the guideline; bars that cross it
    (red) are the rule violations. Turns the Rule-of-5 / Veber check into a
    single glanceable picture instead of a row of bare numbers."""
    if not dm:
        return
    rows = []
    for label, key, limit, unit in _DRUGLIKE_RULES:
        if key not in dm:
            continue
        val = dm[key][0]
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        ratio = val / limit if limit else 0.0
        rows.append({
            "Property": label,
            "Ratio": round(ratio, 3),
            "Display": f"{val:g}{(' ' + unit) if unit else ''} / {limit:g}",
            "Status": "within guideline" if ratio <= 1.0 else "exceeds guideline",
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = [r["Property"] for r in rows]
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("Ratio:Q", title="value ÷ guideline limit",
                scale=alt.Scale(domain=[0, max(1.6, df["Ratio"].max() * 1.1)])),
        y=alt.Y("Property:N", sort=order, title=None),
        color=alt.Color("Status:N",
                        scale=alt.Scale(domain=["within guideline", "exceeds guideline"],
                                        range=["#54A24B", "#E45756"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Property", "Display", "Status"],
    )
    limit_rule = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(
        color="#333", strokeDash=[5, 4]).encode(x="x:Q")
    chart = (bars + limit_rule).properties(width="container",
                                           height=28 * len(rows) + 40)
    st.altair_chart(chart)
    st.caption("Each bar is the value relative to its Lipinski/Veber limit "
               "(dashed line). Green = within guideline, red = exceeds it.")


def render_chembl_card(chembl: Dict[str, Any]):
    """Render ChEMBL's curated clinical/mechanism record (measured, not predicted)."""
    if not chembl or not chembl.get("chembl_id"):
        return False
    st.markdown("#### Clinical & mechanism (ChEMBL) &nbsp; `curated`",
                unsafe_allow_html=True)
    bits = []
    if chembl.get("max_phase_label"):
        bits.append(f"**Development status:** {chembl['max_phase_label']}")
    if chembl.get("first_approval"):
        bits.append(f"**First approval:** {chembl['first_approval']}")
    if chembl.get("molecule_type"):
        bits.append(f"**Type:** {chembl['molecule_type']}")
    if chembl.get("routes"):
        bits.append(f"**Routes:** {', '.join(chembl['routes'])}")
    if chembl.get("atc"):
        bits.append(f"**ATC:** {', '.join(chembl['atc'])}")
    if chembl.get("indication_class"):
        bits.append(f"**Indication class:** {chembl['indication_class']}")
    if chembl.get("withdrawn"):
        bits.append("**⚠️ Withdrawn** in at least one market")
    if bits:
        st.markdown("  \n".join(bits))

    mechs = chembl.get("mechanisms") or []
    if mechs:
        st.markdown("**Mechanism(s) of action**")
        for me in mechs:
            target = f" → *{me['target_name']}*" if me.get("target_name") else ""
            action = f"{me['action_type'].title()}: " if me.get("action_type") else ""
            st.markdown(f"- {action}{me.get('mechanism','')}{target}")
    render_chembl_indications(chembl)
    if chembl.get("url"):
        st.caption(f"Source: [ChEMBL {chembl['chembl_id']}]({chembl['url']})")
    return True


_PHASE_NAME = {4: "Approved", 3: "Phase 3", 2: "Phase 2", 1: "Phase 1",
               0: "Preclinical"}


def chembl_approved_indications(chembl: Dict[str, Any]) -> List[str]:
    """Conditions ChEMBL records the drug as approved (phase 4) for."""
    return [i["condition"] for i in (chembl or {}).get("indications") or []
            if i.get("phase") == 4]


def render_chembl_indications(chembl: Dict[str, Any]):
    """What the drug is approved for / trialled in (ChEMBL drug_indication —
    curated from ATC, EMA, FDA, DailyMed, ClinicalTrials.gov). Fills the gap
    for drugs with no FDA label, e.g. bromhexine."""
    inds = chembl.get("indications") or []
    if not inds:
        return
    approved = [i for i in inds if i.get("phase") == 4]
    trials = [i for i in inds if i.get("phase") != 4]

    def _fmt(i):
        src = ", ".join(s for s in i.get("sources", []) if s)
        return f"{i['condition']}" + (f" <span style='opacity:0.55;font-size:0.8em'>({src})</span>"
                                      if src else "")

    if approved:
        st.markdown("**Approved indications**")
        st.markdown("\n".join(f"- {_fmt(i)}" for i in approved[:12]),
                    unsafe_allow_html=True)
        if len(approved) > 12:
            st.caption(f"…and {len(approved) - 12} more approved indications.")
    if trials:
        with st.expander(f"Investigated in clinical trials — {len(trials)} condition(s)"):
            by_phase: Dict[Any, List[str]] = {}
            for i in trials:
                by_phase.setdefault(i.get("phase"), []).append(i["condition"])
            for ph in sorted(by_phase, key=lambda p: -(p if p is not None else -1)):
                name = _PHASE_NAME.get(ph, "Phase unknown")
                st.markdown(f"**{name}:** " + " · ".join(by_phase[ph]))


def render_xrefs(xrefs: List[Dict[str, str]]):
    """Render UniChem cross-references as links to the same compound elsewhere."""
    if not xrefs:
        return False
    st.markdown("#### Also in these databases &nbsp; `via UniChem (InChIKey match)`",
                unsafe_allow_html=True)
    st.caption("Direct links to this exact compound in other curated databases — "
               "where fuller measured/clinical data often lives.")
    links = []
    for x in xrefs:
        if x.get("url"):
            links.append(f"[{x['db']}]({x['url']})")
        else:
            links.append(f"{x['db']} (`{x['id']}`)")
    st.markdown(" · ".join(links))
    return True


def render_summaries(wiki: Dict[str, Any], record_text: str,
                     wikidata: Dict[str, str], cid, compact: bool = False) -> bool:
    """Render the Wikipedia / PubChem-record / Wikidata narrative for one
    compound. Returns True if anything was rendered.

    `compact=True` shows only the single best description and tucks the other
    sources behind one expander, so the Overview reads as one summary rather
    than three overlapping ones."""
    if compact:
        sources = [("wiki", bool(wiki.get("extract"))),
                   ("record", bool(record_text)),
                   ("wikidata", bool(wikidata.get("description")))]
        present = [s for s, ok in sources if ok]
        if not present:
            return False
        best, rest = present[0], present[1:]
        _render_summary_parts(wiki if best == "wiki" else {},
                              record_text if best == "record" else "",
                              wikidata if best == "wikidata" else {}, cid)
        if rest:
            labels = {"wiki": "Wikipedia", "record": "PubChem", "wikidata": "Wikidata"}
            with st.expander("Other descriptions — " +
                             ", ".join(labels[s] for s in rest)):
                _render_summary_parts(wiki if "wiki" in rest else {},
                                      record_text if "record" in rest else "",
                                      wikidata if "wikidata" in rest else {}, cid)
        return True
    return _render_summary_parts(wiki, record_text, wikidata, cid)


def _render_summary_parts(wiki: Dict[str, Any], record_text: str,
                          wikidata: Dict[str, str], cid) -> bool:
    rendered = False
    if wiki.get("extract"):
        st.markdown("#### Summary (Wikipedia)")
        st.write(wiki["extract"])
        if wiki.get("url"):
            st.caption(f"Source: [{wiki.get('title')}]({wiki['url']})")
        rendered = True
    if record_text:
        st.markdown("#### Record Description (PubChem)")
        parts = [p.strip() for p in re.split(r"\s*\|\s*", record_text) if p.strip()]
        bullets = []
        for p in parts:
            bullets.extend([s.strip() for s in re.split(r"\n\n+", p) if s.strip()] or [p])
        if _BLEACH_AVAILABLE:
            md = "\n".join(f"- {sanitize_html_fragment(b)}" for b in bullets)
            if cid:
                md += f"\n\nSource: [PubChem CID {cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})"
            st.markdown(md, unsafe_allow_html=True)
        else:
            for b in bullets:
                st.write(b)
        rendered = True
    if wikidata.get("description"):
        st.markdown("#### Summary (Wikidata)")
        st.write(wikidata["description"].capitalize())
        if wikidata.get("url"):
            label = wikidata.get("label") or wikidata.get("qid")
            st.caption(f"Source: [{label}]({wikidata['url']})")
        rendered = True
    return rendered


def canonical_form_callout(canonical_ik: str):
    """Marker that the data below was recovered from the compound's mainstream
    (canonical-tautomer) form because the exact depiction searched had no direct
    cross-database match. The two are the same substance up to tautomer/proton
    placement — verified, not guessed."""
    st.info("🔗 The searched PubChem record is stored as an unusual tautomer "
            "depiction with no direct cross-database match. The data below is "
            "from the **canonical form of the same compound** "
            f"(InChIKey `{canonical_ik}`), verified identical up to "
            "tautomer/protonation — not a different molecule.")


def this_compound_caption(name: str, cid):
    """Marker that the block above is specific to the exact searched compound."""
    cid_txt = f" · CID {cid}" if cid else ""
    st.caption(f"📍 **Specific to the searched compound** — {name}{cid_txt}")


def parent_scope_callout(parent: Dict[str, Any]):
    """Prominent marker that the block below is extrapolated from the parent."""
    cid_txt = f" · CID {parent['cid']}" if parent.get("cid") else ""
    st.info(f"↩ **From the parent compound** — {parent.get('name','parent')}"
            f"{cid_txt}. Extrapolated from the free base/acid; **not specific "
            "to the exact salt searched.**")


# ---------- At-a-Glance synthesis card -------------------------------------
_BOXED_BOILERPLATE_RE = re.compile(
    r"see full prescribing information for (?:the )?complete boxed warning\.?|"
    r"\bWARNING:\s*", re.I)


def boxed_summary(text: str, limit: int = 260) -> str:
    """A boxed warning in one readable line: 'Ischemic heart disease — Following
    abrupt cessation…' instead of 'WARNING: ISCHEMIC HEART DISEASE: WARNING:
    ISCHEMIC HEART DISEASE See full prescribing information…'."""
    t = re.sub(r"\s+", " ", _BOXED_BOILERPLATE_RE.sub(" ", str(text or ""))).strip(" :")
    words = t.split(" ")
    n = 0                                    # leading ALL-CAPS run = the title
    while n < len(words) and re.search(r"[A-Z]", words[n]) and not re.search(r"[a-z]", words[n]):
        n += 1
    title = ""
    if n >= 2:
        parts = [p.strip(" ,;") for p in " ".join(words[:n]).split(":")]
        title = " / ".join(dict.fromkeys(p for p in parts if p))
        title = title[:1] + title[1:].lower()
        t = " ".join(words[n:])
    body = _first_sentence(t, limit) if t else ""
    if title and body:
        return f"{title} — {body}"
    return title or body


def _first_sentence(text: str, limit: int = 220) -> str:
    """First sentence of a block of prose, trimmed to `limit` chars."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    sentence = re.split(r"(?<=[a-z0-9)\]][.!?])\s+(?=[A-Z0-9])", text)[0]
    if len(sentence) > limit:
        sentence = sentence[:limit].rstrip() + "…"
    return sentence


def _strip_class_tag(s: str) -> str:
    """Drop the trailing ' [EPC]' / ' [MoA]' tag openFDA appends to class names."""
    return re.sub(r"\s*\[(?:EPC|MoA|CS|PE|EXT)\]\s*$", "", s or "").strip()


def openfda_is_medicine(openfda: Dict[str, Any], drug, chembl: Dict[str, Any]) -> bool:
    """Guard against openFDA false positives: a homeopathic remedy (e.g. 'Benzinum'
    for benzene) carries an FDA label too. Treat a label as a genuine medicine
    only when corroborated — a prescription product, an assigned pharmacologic
    class, library membership, or a ChEMBL development phase."""
    if not openfda:
        return False
    return bool(
        openfda.get("product_type") == "HUMAN PRESCRIPTION DRUG"
        or openfda.get("pharm_class_epc")
        or drug
        or chembl.get("max_phase")
    )


def _availability_label(market: Dict[str, Any], of: Dict[str, Any]) -> str:
    """'Prescription only' / 'Over the counter' / 'Prescription & OTC', from the
    count of US labels per product type (falls back to the one label we hold)."""
    types = (market or {}).get("product_types") or {}
    rx = types.get("HUMAN PRESCRIPTION DRUG", 0)
    otc = types.get("HUMAN OTC DRUG", 0)
    if not (rx or otc):
        pt = (of or {}).get("product_type", "")
        rx, otc = pt == "HUMAN PRESCRIPTION DRUG", pt == "HUMAN OTC DRUG"
    if rx and otc:
        return "Prescription & over-the-counter (US)"
    if rx:
        return "Prescription only (US)"
    if otc:
        return "Over the counter (US)"
    return ""


def _brand_names(market: Dict[str, Any], of: Dict[str, Any], generic: str) -> List[str]:
    """Best-known brand names: RxNorm's single-ingredient brands, else the
    most-labelled brands from openFDA, else the brand on the label we fetched
    (skipping it when it is just the generic name)."""
    market = market or {}
    brands = list(market.get("rxnorm_brands") or [])
    # openFDA label brands only as a fallback for prescription drugs — for
    # dual-use OTC substances they are hand sanitizers, wipes and store brands.
    if not brands and market.get("product_types", {}).get("HUMAN PRESCRIPTION DRUG"):
        brands = list(market.get("brands") or [])
    if not brands and (of or {}).get("product_type") == "HUMAN PRESCRIPTION DRUG":
        g = (generic or "").lower().split()[0] if generic else ""
        brands = [b for b in (of or {}).get("brand_names", [])
                  if b and (not g or g not in b.lower())]
    return brands[:6]


_INDICATION_RE = re.compile(
    r"\bindicated\b|\bused (?:for|to|in)\b|\btreat(?:s|ing|ment of)\b|\brelie(?:f|ve)"
    r"|\bprevent|\bmanagement of\b", re.I)

# ATC groups that describe a secondary/local formulation rather than what the
# drug primarily is — ranked after the main pharmacological group.
_MINOR_ATC_RE = re.compile(r"^other|local|topical|vaginal|ophthalm|otolog|dermat|"
                           r"combination|antiseptic|throat|nasal", re.I)

_SALT_SUFFIX_RE = re.compile(
    r"\s+(?:hydrochloride|hcl|sodium|potassium|calcium|magnesium|sulfate|sulphate|"
    r"maleate|mesylate|besylate|tartrate|citrate|acetate|phosphate|succinate|"
    r"fumarate|bromide|hyclate|monohydrate|dihydrate|hemihydrate|anhydrous)\b.*$", re.I)
# NB: ester names (furoate, propionate…) are deliberately NOT stripped — the
# ester is a different product (fluticasone furoate = Avamys vs propionate =
# Flonase), and stripping it pulled the wrong FDA label.


def find_library_drug(*names: Optional[str]):
    """Drug-library entry for any of the names, also trying each with its salt /
    hydrate suffix stripped ('Atorvastatin calcium' → 'atorvastatin')."""
    if not HAS_DRUG_LIBRARY:
        return None
    for name in names:
        for cand in (name, _SALT_SUFFIX_RE.sub("", name or "")):
            hit = dl.find_drug(cand) if cand else None
            if hit:
                return hit
    return None

# Starts AT "(The) most common/frequent …" so a preceding run-on list (labels
# often lack full stops between list items) isn't swept into the sentence.
_COMMON_AE_RE = re.compile(
    r"\b(?:the\s+)?most (?:common(?:ly reported)?|frequent(?:ly reported)?)\b"
    r"(?:[^.]|\.(?=\d))*\.", re.I)   # '(>5.0%)' doesn't end the sentence


def _common_side_effects(of: Dict[str, Any]) -> str:
    """The label's own 'most common adverse reactions are…' sentence — measured
    trial incidence, the honest answer to 'what are the common side effects'."""
    text = label_field(of, "adverse_reactions")[:8000]
    # 'e.g.' / 'i.e.' mustn't end the sentence — swap their dots for a look-alike
    text = re.sub(r"\b(e\.g|i\.e|vs|approx|incl)\.", lambda mm: mm.group(1).replace(".", "․") + "․",
                  text, flags=re.I)
    m = _COMMON_AE_RE.search(text)
    if not m:
        return ""
    s = m.group(0).strip()
    return _first_sentence(s[0].upper() + s[1:], 300)


# HSA's public "Search by Product" (therapeutic products) — lists each
# registered product's approved package insert / patient leaflet.
HSA_INFOSEARCH_URL = ("https://eservice.hsa.gov.sg/prism/common/enquirepublic/"
                      "SearchDRBProduct.do?action=load")


def market_route(region: str, *names: Optional[str]) -> str:
    """The drug's main route in the selected market ('ORAL', 'TOPICAL'…), used
    to pick the FDA label for the form actually sold there. '' = no market data
    (openFDA then prefers the form most US labels share)."""
    if region == "SG" and HAS_SG_REGISTER:
        variants = [v for n in names if n for v in (n, _SALT_SUFFIX_RE.sub("", n))]
        return sgr.main_route(*variants)
    return ""


def brand_to_ingredients(text: str, region: str = "US") -> Tuple[List[str], str]:
    """Active ingredient(s) for a brand name, and which register said so. The
    selected market's register is asked first (HSA for SG, RxNorm for US)."""
    def _sg():
        return sgr.brand_ingredients(text) if HAS_SG_REGISTER else []

    def _us():
        return cached_rxnorm_brand_ingredients(text)

    order = [(_sg, "Singapore HSA register"), (_us, "US RxNorm")]
    if region != "SG":
        order.reverse()
    for fn, label in order:
        try:
            ings = fn()
        except Exception:
            ings = []
        if ings:
            return ings, label
    return [], ""


def render_sg_products(sg: Dict[str, Any]):
    """Expander listing every HSA-registered Singapore product containing the
    drug, least restrictive class first — strength and pack decide the class
    (e.g. small ibuprofen packs are General sale, larger ones Prescription)."""
    prods = sg.get("products") or []
    with st.expander(f"Singapore registered products (HSA) — {len(prods)}"):
        rows = [{
            "Class": p["cls"],
            "Product": p["name"],
            "Active ingredients": " + ".join(p["ings"]),
            "Strength": p.get("strength", ""),
            "Form": (p.get("form") or "").title(),
            "Licence holder": (p.get("holder") or "").title(),
            "Licence no.": p.get("lic", ""),
        } for p in prods]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        # HSA's product search is a stateful form (no per-product deep links),
        # so link the search and say what to type in.
        st.markdown(f"📄 **Package inserts:** open [HSA Infosearch ↗]({HSA_INFOSEARCH_URL}) "
                    "and search the product name or licence no. (e.g. "
                    f"`{prods[0].get('lic', 'SIN…')}`) for the approved package insert / "
                    "patient leaflet.")
        st.caption(
            "**POM** = Prescription-only (needs a doctor's prescription) · **P** = "
            "Pharmacy-only (from a pharmacist, no prescription) · **GSL** = General "
            "sale (any retailer). Contains information from the "
            f"[Listing of Registered Therapeutic Products]({sgr.META.get('dataset_url', '')}) "
            f"accessed on {sgr.META.get('fetched', '?')} from data.gov.sg, made "
            "available under the Singapore Open Data Licence v1.0. Refresh with "
            "`python build_hsa_register.py`.")


def record_indication_sentence(grouped: Dict[str, Dict[str, Any]]) -> str:
    """First indication-style sentence from the record's own Uses/Indication
    heading (PubChem text first, it's DrugBank-curated) — the 'Used for'
    fallback for drugs with no FDA label."""
    items = (grouped or {}).get("Pharmacology & Clinical") or {}
    first_any = ""
    for heading, val in items.items():
        if not re.search(r"indication|therapeutic use|uses", heading, re.I):
            continue
        pub, bullets, _ = _split_value_parts(val)
        texts = [p for p in pub.split(" | ") if p.strip() and _entry_rank(p) < 2] + bullets
        for t in texts:
            for s in _SENT_SPLIT_RE.split(re.sub(r"\s+", " ", t)):
                s = _clean_record_sentence(s)
                if _is_lowvalue_sentence(s):
                    continue
                if _INDICATION_RE.search(s):
                    return _first_sentence(s, 260)
                first_any = first_any or s
    # under an indication heading the lead sentence is the indication even
    # when phrased loosely ('is used alone or with … to reduce mucus …')
    return _first_sentence(first_any, 260) if first_any else ""


def glance_facts(*, compound_name, formula, dm, chembl, wikidata, wiki,
                 record_text, drug, openfda, adverse_events, market=None,
                 region="US", record_use="") -> Dict[str, Any]:
    """The At-a-Glance card's content as data: the single best value per field,
    composed from every source. Shared by the card (Explore) and the chat's
    overview answer (Ask), so both say the same thing."""
    is_med = openfda_is_medicine(openfda, drug, chembl) or bool(drug) \
        or bool(chembl.get("max_phase"))
    of = openfda if openfda_is_medicine(openfda, drug, chembl) else {}
    market = market or {}

    # field → where its value came from, shown next to it so a reader can tell
    # a US regulatory label from a database entry or an encyclopedia line
    src: Dict[str, str] = {}

    # "What it is" — one honest line, best source first.
    what = ""
    if wiki.get("extract"):
        what, src["what"] = _first_sentence(wiki["extract"]), "Wikipedia"
    elif wikidata.get("description"):
        what, src["what"] = wikidata["description"].capitalize(), "Wikidata"
    elif record_text:
        what, src["what"] = _first_sentence(record_text), "PubChem"
    elif chembl.get("indication_class"):
        what, src["what"] = chembl["indication_class"], "ChEMBL"

    chem_bits = []
    if formula:
        chem_bits.append(formula)
    if dm:
        chem_bits.append(f"{dm['Molecular weight'][0]} Da")
    facts: Dict[str, Any] = {"is_med": is_med, "of": of, "what": what, "src": src,
                             "chem_bits": chem_bits, "formula": formula, "dm": dm}
    if not is_med:
        return facts

    # ----- medicine: monograph-style header -----
    status = ""
    if chembl.get("max_phase_label"):
        status = chembl["max_phase_label"]
        if chembl.get("first_approval"):
            status += f" · first approved {chembl['first_approval']}"
    if chembl.get("withdrawn"):
        status += "  ·  ⚠️ withdrawn in ≥1 market"
    if status:
        src["status"] = "ChEMBL"

    # One authoritative class line: the FDA Established Pharmacologic Class,
    # else the drug's specific (level-4) WHO ATC groups — not every class
    # tag it carries (ibuprofen is also "Antiinflammatory products for
    # vaginal administration", which is true but not what it *is*).
    classes: List[str] = []
    epc = [_strip_class_tag(c) for c in of.get("pharm_class_epc", [])
           if not re.search(r"allergen", c, re.I)]
    if epc and of.get("product_type") == "HUMAN PRESCRIPTION DRUG":
        classes, src["classes"] = epc[:2], "FDA class"
    elif drug and HAS_DRUG_LIBRARY:
        atc = [dl.class_name(s) for s in drug["classes"] if dl.class_system(s) == "ATC"]
        # the main pharmacological group(s), not local-formulation or
        # "other …" catch-all groups (unless that's all there is)
        main = [c for c in atc if not _MINOR_ATC_RE.search(c)]
        classes, src["classes"] = (main or atc)[:2], "WHO ATC"
    if not classes and epc:
        classes, src["classes"] = epc[:2], "FDA class"
    if not classes and chembl.get("atc"):
        classes, src["classes"] = chembl["atc"][:2], "ChEMBL"

    mech = ""
    if chembl.get("mechanisms"):
        m0 = chembl["mechanisms"][0]
        tgt = f" → {m0['target_name']}" if m0.get("target_name") else ""
        mech, src["mech"] = f"{m0.get('mechanism', '')}{tgt}".strip(), "ChEMBL"
    elif of.get("pharm_class_moa"):
        mech, src["mech"] = _strip_class_tag(of["pharm_class_moa"][0]), "FDA class"

    use = ""
    if of.get("indications"):
        # Skip label boilerplate ("Carefully consider the potential benefits
        # and risks…") — take the first sentence that states an indication.
        sents = _SENT_SPLIT_RE.split(of["indications"])
        pick = next((s for s in sents if _INDICATION_RE.search(s)
                     and not s.lower().startswith("carefully consider")), sents[0])
        use, src["use"] = _first_sentence(pick, 260), "US FDA label"
    elif of.get("purpose"):
        use, src["use"] = _first_sentence(of["purpose"], 160), "US FDA label"
    elif chembl_approved_indications(chembl):
        # no FDA label (e.g. a non-US drug): ChEMBL's approved indications
        appr = chembl_approved_indications(chembl)
        use = ", ".join(appr[:5]) + (" …" if len(appr) > 5 else "")
        src["use"] = "ChEMBL approved indications"
    elif record_use:
        # the record's own indication text (DrugBank via PubChem)
        use, src["use"] = record_use, "PubChem/DrugBank"
    elif chembl.get("indication_class"):
        use, src["use"] = chembl["indication_class"], "ChEMBL"

    generic = (of.get("generic_names") or [compound_name])[0]
    brands = _brand_names(market, of, generic)
    if brands:
        src["brands"] = "RxNorm" if market.get("rxnorm_brands") else "US FDA labels"
    avail = _availability_label(market, of)
    if avail:
        src["avail"] = "US FDA labels"
    brand_label = "Brand names"
    sg = market.get("sg") or {}
    if region == "SG":
        if sg:
            brands = sg["brands"][:6]
            brand_label = "Brand names (SG)"
            avail = sgr.availability_text(sg)
            src["brands"] = src["avail"] = "HSA register"
        else:
            # Not in the register: say so, keep US brands but label them.
            brand_label = "Brand names (US)"
            avail = "No HSA-registered product found in Singapore" + \
                    (f" · US: {avail}" if avail else "")
            src["avail"] = "HSA register · US FDA labels"
    routes = of.get("routes") or chembl.get("routes") or []
    if routes:
        src["routes"] = "US FDA label" if of.get("routes") else "ChEMBL"

    common = _common_side_effects(of)
    if common:
        src["common"] = "US FDA label · clinical trials"
    faers_terms = "" if common else ", ".join(a["term"] for a in (adverse_events or [])[:5])
    facts.update(status=status, classes=[c for c in classes if c], mech=mech, use=use,
                 brands=brands, brand_label=brand_label, avail=avail, sg=sg,
                 routes=routes, common=common, faers_terms=faers_terms,
                 boxed=of.get("boxed_warning") or "")
    return facts


def _glance_tag(src: str) -> str:
    """A small grey source label after an At-a-Glance value."""
    return (f" <small style='opacity:.6'>· {_html.escape(src)}</small>" if src else "")


def render_at_a_glance(*, compound_name, formula, smiles, dm,
                       chembl, wikidata, wiki, record_text, drug,
                       openfda, adverse_events, market=None, region="US",
                       record_use=""):
    """A compact synthesis card above the tabs: the single best value per field,
    composed from every source. A recognised medicine reads like a drug
    monograph header (brands, class, availability, use, mechanism, common side
    effects, boxed warning) with chemistry reduced to one line; a plain chemical
    shows identity + drug-likeness only, never fabricated drug data. Detail lives
    in the tabs — this card only answers 'what is it and what matters most'."""
    f = glance_facts(compound_name=compound_name, formula=formula, dm=dm,
                     chembl=chembl, wikidata=wikidata, wiki=wiki,
                     record_text=record_text, drug=drug, openfda=openfda,
                     adverse_events=adverse_events, market=market, region=region,
                     record_use=record_use)
    tag = lambda key: _glance_tag(f["src"].get(key, ""))  # noqa: E731
    with st.container(border=True):
        if f["what"]:
            st.markdown(f["what"] + tag("what"), unsafe_allow_html=True)

        if not f["is_med"]:
            left, right = st.columns(2)
            with left:
                st.markdown("**Chemistry**")
                bits = []
                if formula:
                    bits.append(f"- **Formula:** {formula}")
                if dm:
                    bits.append(f"- **Mol. weight:** {dm['Molecular weight'][0]} Da")
                    ro5 = dm.get("_ro5_violations", 0)
                    verdict = "passes Lipinski Ro5" if ro5 == 0 else f"{ro5} Lipinski violation(s)"
                    bits.append(f"- **Drug-likeness:** QED {dm['QED (drug-likeness)'][0]} · "
                                f"cLogP {dm['cLogP (Crippen)'][0]} · {verdict}")
                st.markdown("\n".join(bits) if bits else "_No descriptors available._")
            with right:
                st.markdown("**Type**")
                st.markdown("- Not a recognised medicine — chemical / research compound.")
            return

        of, sg = f["of"], f["sg"]
        left, right = st.columns(2)
        with left:
            bits = []
            if f["brands"]:
                bits.append(f"- **{f['brand_label']}:** {', '.join(f['brands'])}{tag('brands')}")
            if f["classes"]:
                bits.append(f"- **Drug class:** {' · '.join(f['classes'])}{tag('classes')}")
            if f["avail"]:
                bits.append(f"- **Availability:** {f['avail']}{tag('avail')}")
            if f["routes"]:
                bits.append(f"- **Route:** {', '.join(r.title() for r in f['routes'][:4])}"
                            f"{tag('routes')}")
            if f["status"]:
                bits.append(f"- **Status:** {f['status']}{tag('status')}")
            st.markdown("\n".join(bits) if bits else "_No drug facts available._",
                        unsafe_allow_html=True)
        with right:
            bits = []
            if f["use"]:
                bits.append(f"- **Used for:** {f['use']}{tag('use')}")
            if f["mech"]:
                bits.append(f"- **How it works:** {f['mech']}{tag('mech')}")
            if f["common"]:
                bits.append(f"- **Common side effects:** {f['common']}{tag('common')}")
            elif f["faers_terms"]:
                bits.append(f"- **Most-reported side effects:** {f['faers_terms']} "
                            "<small>(FDA FAERS reports, not incidence)</small>")
            st.markdown("\n".join(bits) if bits else "_No clinical summary._",
                        unsafe_allow_html=True)

        if f["boxed"]:
            st.error("**⚠️ FDA Boxed Warning** — " + boxed_summary(f["boxed"], 260)
                     + "  \n*Full text in the **Safety** tab.*")
        if not of:
            # Say why clinical detail is thin, so the gap reads as missing
            # data (not a broken app), and point to the authoritative source.
            where = (f"the Singapore package insert on [HSA Infosearch]({HSA_INFOSEARCH_URL}) "
                     "(search by product name or licence no.)"
                     if region == "SG" and sg else "the local package insert / SmPC")
            st.info(f"ℹ️ **Limited clinical data:** there is no FDA label for "
                    f"{compound_name} — it isn't approved in the US (or not under "
                    f"this name). Uses and pharmacology here come from ChEMBL, "
                    f"PubChem and Wikipedia; for dosing and warnings see {where}.")
        if region == "SG" and sg:
            render_sg_products(sg)
        if f["chem_bits"]:
            st.caption("Chemistry: " + " · ".join(f["chem_bits"]) +
                       " — structure, properties & drug-likeness in the **Chemistry** tab.")


# ---------- FDA label (openFDA) rendering ----------------------------------
# Old-format SPL labels are sometimes mis-coded (one ibuprofen label files
# "Information for Patients" under pregnancy and "Drug Interactions" under
# lactation). A section is shown only if its opening text mentions its topic.
_LABEL_TOPIC_RE = {
    "drug_interactions": r"interact|concomitant|co-?administ|inhibitor|inducer|other drugs|drugs:",
    "pregnancy": r"pregnan",
    "lactation": r"lactat|nursing|breast|human milk",
    "pediatric_use": r"pediatric|child|infant|adolescent",
    "geriatric_use": r"geriatric|elderly|older|65 years",
    "overdosage": r"overdos|toxicit|ingest|poison",
    "adverse_reactions": r"adverse|reaction|side effect",
    "contraindications": r"contraindicat|do not|should not|hypersensitiv|patients with",
    "specific_populations": r"pregnan|lactat|pediatric|geriatric",
}


def label_field(of: Dict[str, Any], field: str) -> str:
    """A label section's text, or '' when it is missing, a stub ('Drug Facts'),
    or fails the topic sanity check above."""
    text = (of or {}).get(field) or ""
    if len(text) < 40:
        return ""
    pat = _LABEL_TOPIC_RE.get(field)
    if pat and not re.search(pat, text[:600], re.I):
        return ""
    return text


# A numbered label subsection ("5.1 Cardiovascular Thrombotic Events …") or a
# highlights-style "Topic: …" lead.
_LABEL_SUBSEC_RE = re.compile(r"(?:(?<=\s)|^)(?=\d{1,2}\.\d{1,2}\s+[A-Z])")
_LABEL_COLON_RE = re.compile(r"(?<=[.)])\s+(?=[A-Z][\w\-,()/ ]{2,50}:\s)")
_TITLE_CONNECTORS = {"of", "and", "in", "with", "the", "to", "for", "or", "on",
                     "a", "an", "by", "during", "&"}


def _split_label_title(chunk: str) -> Tuple[str, str]:
    """('5.1 Cardiovascular Thrombotic Events', 'Clinical trials of …') — the title
    is the run of Capitalised words before the word that starts the body."""
    m = re.match(r"(\d{1,2}\.\d{1,2})\s+(.*)", chunk, re.S)
    if m:
        num, rest = m.groups()
        words = rest.split(" ")
        cut = 0
        for i, w in enumerate(words[:16]):
            core = w.strip(",;:()")
            if core and (core[0].isupper() or core.lower() in _TITLE_CONNECTORS):
                continue
            # Step back over connectors to the capitalised word that starts the
            # body sentence ("…Indications An INR of greater…" → body "An INR…").
            cut = i - 1
            while cut > 0 and words[cut].strip(",;:()") in _TITLE_CONNECTORS:
                cut -= 1
            if cut > 1 and words[cut - 1] in ("A", "An", "The"):
                cut -= 1
            break
        if cut <= 0:
            return num, rest
        return f"{num} " + " ".join(words[:cut]).rstrip(",;:"), " ".join(words[cut:])
    m = re.match(r"([A-Z][\w\-,()/ ]{2,50}):\s+(.*)", chunk, re.S)
    if m and not re.search(r"\b(?:is|are|include|includes|indicated)\b", m.group(1)):
        return m.group(1), m.group(2)
    return "", chunk


def _label_chunks(text: str) -> List[Tuple[str, str]]:
    """Split a label section into (title, body) subsections."""
    parts = [p.strip() for p in _LABEL_SUBSEC_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in _LABEL_COLON_RE.split(text) if p.strip()]
    return [_split_label_title(p) for p in parts]


_LABEL_SUMMARY_SENTENCES = 2   # sentences per subsection in summary view
_LABEL_SUMMARY_CHUNKS = 8      # subsections in summary view


def render_label_section(text: str, key: str):
    """Render one FDA label section as scannable points: each subsection's title
    plus its opening sentences, with a single 'show full text' toggle for the
    verbatim label (so nothing is lost, but nothing overwhelms)."""
    chunks = _label_chunks(text)
    truncated = len(chunks) > _LABEL_SUMMARY_CHUNKS
    lines = []
    if len(chunks) == 1 and not chunks[0][0]:
        sents = _SENT_SPLIT_RE.split(chunks[0][1])
        truncated = len(sents) > 4
        lines = [f"- {s}" for s in sents[:4]]
    else:
        for title, body in chunks[:_LABEL_SUMMARY_CHUNKS]:
            sents = _SENT_SPLIT_RE.split(body)
            lead = " ".join(sents[:_LABEL_SUMMARY_SENTENCES])
            truncated |= len(sents) > _LABEL_SUMMARY_SENTENCES
            lines.append(f"- **{title}** — {lead}" if title else f"- {lead}")
    full = key and truncated and st.checkbox("Show full label text", key=f"lbl_{key}")
    if full:
        for title, body in chunks:
            if title:
                st.markdown(f"**{title}**")
            st.markdown(body)
    else:
        st.markdown("\n".join(lines))


def _combination_label_note(of: Dict[str, Any]):
    """Flag when the only FDA label found is a combination product (e.g.
    diphenoxylate is only marketed with atropine, as Lomotil)."""
    combo = next((g for g in (of or {}).get("generic_names", [])
                  if re.search(r"\band\b|,", g, re.I)), "")
    if combo:
        st.info(f"ℹ️ The only US label found is for a **combination product** "
                f"({combo.lower()}) — label sections below apply to that combination.")


_PRODUCT_TYPE_SHORT = {"HUMAN PRESCRIPTION DRUG": "prescription",
                       "HUMAN OTC DRUG": "over-the-counter"}


def label_identity_line(of: Dict[str, Any]) -> str:
    """Which product the label describes and how current it is: 'Lipitor
    (atorvastatin calcium) · oral · prescription · label revised 2024-03-15 ·
    retrieved 2026-10-11 · DailyMed'. Markdown."""
    brand = (of.get("brand_names") or [""])[0]
    gen = (of.get("generic_names") or [""])[0]
    name = brand or gen.title()
    if gen and brand and gen.lower() != brand.lower():
        name += f" ({gen.lower()})"
    bits = [f"**{_md_safe(name)}**"] if name else []
    if of.get("routes"):
        bits.append(", ".join(r.lower() for r in of["routes"][:3]))
    if _PRODUCT_TYPE_SHORT.get(of.get("product_type", "")):
        bits.append(_PRODUCT_TYPE_SHORT[of["product_type"]])
    if of.get("effective_date"):
        bits.append(f"label revised {of['effective_date']}")
    if of.get("retrieved"):
        bits.append(f"retrieved {of['retrieved']}")
    if of.get("url"):
        bits.append(f"[DailyMed ↗]({of['url']})")
    return " · ".join(bits)


def label_scope_warnings(of: Dict[str, Any], *, region: str = "US",
                         sg: Optional[Dict[str, Any]] = None,
                         drug_routes: Sequence[str] = (),
                         local_route: str = "") -> List[str]:
    """Reasons the label may not describe the user's own product: it is a US
    document read in Singapore mode, or it covers a different route/formulation
    than the one sold locally (`local_route`, from the HSA register) or the
    drug's usual one (an injection label for a drug mostly taken by mouth).
    Each entry is a Markdown sentence."""
    out: List[str] = []
    if region == "SG":
        if sg and sg.get("products"):
            n = len(sg["products"])
            out.append(f"🌏 **This is the US label, not Singapore's.** {n} HSA-registered "
                       f"product{'s' if n != 1 else ''} exist{'s' if n == 1 else ''} in "
                       "Singapore, and their approved uses, strengths and dosing can differ — "
                       f"the Singapore package insert on [HSA Infosearch]({HSA_INFOSEARCH_URL}) "
                       "is authoritative there.")
        else:
            out.append("🌏 **This is the US label.** No HSA-registered product was found for "
                       "this drug, so it may not be available in Singapore at all.")
    routes = {r.lower() for r in of.get("routes") or []}
    usual = {r.lower() for r in drug_routes or []}
    if routes and local_route and local_route.lower() not in routes:
        out.append(f"💉 **Different formulation:** most Singapore products are "
                   f"{local_route.lower()}, but this US label is for a "
                   f"{', '.join(sorted(routes))} product — doses and some warnings differ. "
                   "Check your own product's leaflet.")
    elif routes and usual and "oral" in usual and "oral" not in routes:
        out.append(f"💉 **Different formulation:** this label is for a "
                   f"{', '.join(sorted(routes))} product, but the drug is also taken by mouth — "
                   "tablet/liquid doses and some warnings differ. Check your own product's leaflet.")
    return out


def render_label_provenance(of: Dict[str, Any], *, region: str = "US",
                            sg: Optional[Dict[str, Any]] = None,
                            drug_routes: Sequence[str] = (), local_route: str = ""):
    """Header for every block of label text: exactly which label it is, plus
    a warning whenever that label may not match the user's market/product."""
    st.markdown("📄 **Source: US FDA label** — " + label_identity_line(of)
                + "  <small>`public domain`</small>", unsafe_allow_html=True)
    for w in label_scope_warnings(of, region=region, sg=sg, drug_routes=drug_routes,
                                  local_route=local_route):
        st.warning(w)


# (heading, [fields…]) — the first field with valid text is shown. Order follows
# the questions a clinician/patient asks: what for, how much, how supplied,
# interactions, special populations, overdose.
CLINICAL_SECTIONS = [
    ("Uses (indications)", ["indications", "purpose"]),
    ("Dosage & administration", ["dosage"]),
    ("Dosage forms & strengths", ["dosage_forms"]),
    ("Drug interactions", ["drug_interactions"]),
    ("Pregnancy & breastfeeding", ["specific_populations", "pregnancy"]),
    ("Lactation", ["lactation"]),
    ("Children", ["pediatric_use"]),
    ("Older adults", ["geriatric_use"]),
    ("Storage", ["storage"]),
]

SAFETY_LABEL_SECTIONS = [
    ("Contraindications — who should not take it", ["contraindications", "do_not_use"]),
    ("Warnings & precautions", ["warnings"]),
    ("Ask a doctor before use", ["ask_doctor"]),
    ("Stop use and ask a doctor if", ["stop_use"]),
    ("Side effects (adverse reactions)", ["adverse_reactions"]),
    ("Overdose", ["overdosage"]),
]


# Sections whose absence must be said out loud: a reader who sees no
# interactions section could otherwise take it as "no interactions".
_KEY_LABEL_SECTIONS = {"Dosage & administration", "Drug interactions",
                       "Pregnancy & breastfeeding",
                       "Contraindications — who should not take it",
                       "Warnings & precautions", "Side effects (adverse reactions)"}


def missing_key_sections(of: Dict[str, Any], sections) -> List[str]:
    """Key headings of `sections` that this label has no usable text for."""
    return [h for h, fields in sections if h in _KEY_LABEL_SECTIONS
            and not any(label_field(of, f) for f in fields)]


def _render_label_sections(of: Dict[str, Any], sections, key_prefix: str,
                           expand_first: bool = True) -> bool:
    """Render label sections as expanders (first one open), then name the key
    sections the label lacks. Returns True if any section was shown."""
    shown = 0
    has_spec_pop = bool(label_field(of, "specific_populations"))
    for heading, fields in sections:
        # use_in_specific_populations already contains pregnancy/lactation/
        # pediatric/geriatric subsections — don't repeat them individually.
        if has_spec_pop and fields[0] in ("lactation", "pediatric_use", "geriatric_use"):
            continue
        text, field = "", ""
        for f in fields:
            text = label_field(of, f)
            if text:
                field = f
                break
        if not text:
            continue
        if field == "specific_populations":
            heading = "Special populations (pregnancy, breastfeeding, children, older adults)"
        lead = _first_sentence(_label_chunks(text)[0][1], 90)
        with st.expander(f"**{heading}**  ·  {lead}", expanded=expand_first and shown == 0):
            render_label_section(text, key=f"{key_prefix}_{field}")
        shown += 1
    missing = missing_key_sections(of, sections)
    if missing:
        st.caption("❔ **Not covered by this label:** "
                   + ", ".join(m.split(" — ")[0].split(" (")[0].lower() for m in missing)
                   + ". That's missing information, not evidence that there are none — "
                     "check the package insert or ask a pharmacist.")
    return bool(shown)


def render_faers_chart(adverse_events: List[Dict[str, Any]]):
    """Horizontal bar chart of FAERS most-reported reaction terms."""
    if not adverse_events:
        return
    df = pd.DataFrame(adverse_events[:8])
    chart = alt.Chart(df).mark_bar(color="#4C78A8").encode(
        x=alt.X("count:Q", title="spontaneous reports"),
        y=alt.Y("term:N", sort="-x", title=None),
        tooltip=["term", alt.Tooltip("count:Q", format=",")],
    ).properties(width="container", height=26 * len(df) + 30)
    st.altair_chart(chart)
    st.caption("FDA Adverse Event Reporting System (FAERS) tallies — what is "
               "*reported* to the FDA, not how often it happens or whether the drug "
               "caused it. For incidence, see the label's adverse-reactions section.")


def render_gtopdb_targets(gtopdb: Dict[str, Any]) -> bool:
    if not gtopdb.get("interactions"):
        return False
    st.markdown("#### Molecular targets &nbsp; `Guide to Pharmacology, curated`",
                unsafe_allow_html=True)
    for it in gtopdb["interactions"]:
        aff = f" — {it['parameter']} {it['affinity']}" if it.get("affinity") else ""
        act = f"{it['type']}/{it['action']}".strip("/") if (it.get("type") or it.get("action")) else ""
        act = f" ({act})" if act else ""
        st.markdown(f"- **{it['target']}**{act}{aff}")
    if gtopdb.get("url"):
        st.caption(f"Source: [Guide to Pharmacology ligand {gtopdb['ligand_id']} ↗]"
                   f"({gtopdb['url']})  `IUPHAR/BPS, curated`")
    return True


# ============================== Streamlit app ==============================
EXAMPLE_QUERIES = ["atorvastatin", "metformin", "ibuprofen", "sertraline",
                   "amoxicillin", "warfarin"]


def _pick_example():
    """Example pill clicked → search it, then clear the pill so the same
    example can be picked again later."""
    picks = st.session_state.get("example_pick") or []
    if picks:
        _set_search(picks[-1])
    st.session_state["example_pick"] = []


def _set_search(value: str):
    st.session_state["chemical_search_input"] = value
    st.session_state["lookup_pressed"] = True
    if st.session_state.get("input_mode") == "Draw structure":
        st.session_state["input_mode"] = "Name or SMILES"


# ---------------------- Drug Library browser ----------------------
def _open_drug(query: str):
    """Hand a library drug off to the resolver pipeline and switch to Explore."""
    _set_search(query)
    st.session_state["app_mode"] = "🔎 Explore"


def _focus_class(slug: str):
    """Jump into the Drug Library focused on a given class (used by class chips)."""
    st.session_state["app_mode"] = "💊 Drug Library"
    st.session_state["drug_lib_focus_slug"] = slug
    if dl is not None:
        cat = dl.category_of(slug)
        if cat:
            st.session_state["drug_lib_category"] = cat
            # preselect the class in its (per-category-keyed) selectbox
            st.session_state[f"drug_lib_class_{cat}"] = slug
    st.session_state["drug_lib_search"] = ""


def _class_chips_html(classes: List[str]) -> str:
    """Static coloured chips listing every class a drug belongs to (shows overlaps)."""
    spans = []
    for s in classes:
        spans.append(
            "<span style='background:#eef2ff;color:#3730a3;border:1px solid "
            "#c7d2fe;border-radius:10px;padding:1px 8px;margin:2px 4px 2px 0;"
            "font-size:0.75rem;display:inline-block;white-space:nowrap'>"
            f"{_html.escape(dl.class_name(s))}</span>"
        )
    return "".join(spans)


def _render_drug_row(d: Dict, key_prefix: str = ""):
    """One drug: an 'Explore →' button plus its blurb and class chips. `key_prefix`
    keeps button keys unique (a drug can appear under several classes at once)."""
    c1, c2 = st.columns([1, 3.4])
    with c1:
        st.button(d["name"], key=f"lib_open_{key_prefix}_{d['name']}",
                  width='stretch', on_click=_open_drug, args=(d["query"],),
                  help=f"Resolve {d['name']} in the Explorer")
    with c2:
        st.markdown(d["blurb"])
        st.markdown(_class_chips_html(d["classes"]), unsafe_allow_html=True)


def render_drug_library():
    """Browse the drug library by category → class → drug, or search it, then
    click any drug to open it in the full Explorer. Designed to scale to thousands
    of drugs: browsing renders only the selected class's drugs, never all at once."""
    if not HAS_DRUG_LIBRARY or not dl.is_available():
        st.error("Drug library data not found. Generate it once with:\n\n"
                 "`python build_drug_library.py`")
        return

    n_drugs, n_classes, n_cats = len(dl.DRUGS), len(dl._LEAF_SLUGS), len(dl.CATEGORIES)
    n_overlap = sum(1 for d in dl.DRUGS if len(d["classes"]) > 1)
    st.header("💊 Drug Library")
    st.caption(
        f"{n_drugs:,} medicinal drugs · {n_classes:,} classes · {n_cats} categories · "
        f"{n_overlap:,} drugs belong to more than one class. "
        "Click any drug to resolve its structure and full record in the Explorer.")
    src = dl.META.get("source")
    if src:
        systems = ", ".join(dl.META.get("systems", []))
        st.caption(f"Classified by {systems}. Source: {src}. "
                   "One row per ingredient (synonyms & salts de-duplicated).")

    # --- search across the whole library ---
    search = st.text_input(
        "Search drugs or classes",
        key="drug_lib_search",
        placeholder="e.g. statin, atorvastatin, penicillin, reuptake inhibitor…")
    if search.strip():
        results = dl.search_drugs(search)
        if not results:
            st.info("No drugs matched. Try a class name (e.g. 'ACE inhibitor') "
                    "or a drug name.")
            return
        cap_note = " (showing first 400)" if len(results) >= 400 else ""
        st.markdown(f"**{len(results)}** match{'es' if len(results) != 1 else ''} "
                    f"for *{search.strip()}*{cap_note}:")
        for d in results:
            _render_drug_row(d, key_prefix="search")
        return

    # --- browse by category → class → drug ---
    cat_counts = dl.category_counts()
    focus_slug = st.session_state.pop("drug_lib_focus_slug", None)

    cat = st.selectbox(
        "Category", dl.CATEGORIES, key="drug_lib_category",
        format_func=lambda c: f"{c}  ({cat_counts.get(c, 0):,})")

    leaves = dl.classes_in_category(cat)
    if not leaves:
        st.info("No classes in this category.")
        return
    cc = dl.class_counts()

    # If we arrived via a class chip, preselect that class.
    default_ix = leaves.index(focus_slug) if focus_slug in leaves else 0

    def _fmt_class(slug):
        sub = dl.subgroup_name(slug)
        label = dl.class_name(slug)
        prefix = f"{sub} › " if sub and sub.lower() != label.lower() else ""
        return f"{prefix}{label}  ({cc.get(slug, 0)})"

    slug = st.selectbox("Class", leaves, index=default_ix,
                        format_func=_fmt_class, key=f"drug_lib_class_{cat}")
    st.caption(dl.class_desc(slug))

    drugs = dl.drugs_in_class(slug)
    st.markdown(f"**{len(drugs)}** drug{'s' if len(drugs) != 1 else ''} in "
                f"*{dl.class_name(slug)}*:")
    for d in drugs:
        _render_drug_row(d, key_prefix=slug)


# ---------------------- Mainstream Drug Reference ----------------------
_REF_ALL = "All sections"
_REF_TAG_BADGE = {"both": ":green-badge[WHO + Top-300]",
                  "who": ":blue-badge[WHO essential]",
                  "top300": ":orange-badge[Top-300]"}
_REF_RESULT_CAP = 60


def _ref_drug_buttons(e: Dict, key_prefix: str):
    """One button per drug named in the row (grouped rows like 'Enalapril /
    perindopril' get one each); names that aren't a single compound — vaccines,
    mixtures — show as plain bold text."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        for i, (label, q) in enumerate(mref.drug_links(e)):
            if q:
                st.button(f"{label} ›", key=f"ref_{key_prefix}_{e['id']}_{i}",
                          on_click=_open_drug, args=(q,),
                          help=f"Open {q} in the Explorer")
            else:
                st.markdown(f"**{label}**")
        if e.get("tag"):
            st.markdown(_REF_TAG_BADGE[e["tag"]])


def _render_ref_card(e: Dict, key_prefix: str):
    with st.container(border=True):
        _ref_drug_buttons(e, key_prefix)
        st.caption(e["class"])
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Mechanism**  \n{e['mechanism']}")
        c2.markdown(f"**Main uses**  \n{e['uses']}")
        c3.markdown(f"**Key cautions & signature effects**  \n{e['cautions']}")


def _ref_picked_row(key: str) -> Optional[int]:
    """Row the user picked in the table: ticked its checkbox, or clicked its
    Drug cell. Clicks on other cells (to read a long mechanism) don't count."""
    sel = st.session_state.get(key)
    if sel is None:
        return None
    if sel.selection.get("rows"):
        return sel.selection["rows"][0]
    for row, col in sel.selection.get("cells") or []:
        if col == "Drug":
            return row
    return None


def _ref_table_select(rows: List[Dict], key: str):
    """Table-view pick: a row naming one drug opens it straight away; a grouped
    row stays selected so the picker below can offer each drug."""
    picked = _ref_picked_row(key)
    if picked is not None:
        links = [q for _, q in mref.drug_links(rows[picked]) if q]
        if len(links) == 1:
            _open_drug(links[0])


def _render_ref_table(rows: List[Dict], key: str):
    df = pd.DataFrame([{
        "Drug": e["drug"],
        "List": mref.TAG_LABELS.get(e["tag"], ""),
        "Class": e["class"],
        "Mechanism": e["mechanism"],
        "Main uses": e["uses"],
        "Key cautions": e["cautions"],
        "Sub-section": e["subsection"],
    } for e in rows])
    # Picker for a grouped row sits ABOVE the table (a tall table hides it below);
    # the selection is already in session state from the click that reran us.
    picked = _ref_picked_row(key)
    if picked is not None:
        st.caption("Open which drug from this row?")
        _ref_drug_buttons(rows[picked], "tblpick")
    else:
        st.caption("Click a drug's name (or tick its row) to open it.")
    st.dataframe(
        df, key=key, hide_index=True, on_select=lambda: _ref_table_select(rows, key),
        selection_mode=["single-row", "single-cell"], width="stretch",
        height=min(700, 38 + 35 * len(df)),
        column_config={c: st.column_config.TextColumn(c, width="large")
                       for c in ("Mechanism", "Main uses", "Key cautions")})


def render_mainstream_reference():
    """The user's Mainstream Drug Reference as a browsable study table: filter by
    section / list / text, then click any drug to open its full Explore page."""
    if not HAS_MAINSTREAM_REF:
        st.error("Reference data not found. Generate it once with:\n\n"
                 "`python build_mainstream_reference.py`")
        return
    secs = mref.sections()
    all_rows = mref.entries()
    st.header("📘 Mainstream Drug Reference")
    st.caption(f"{len(all_rows)} entries · {len(secs)} sections · "
               f"{sum(len(s['subsections']) for s in secs)} drug classes — from your "
               f"*{mref.source()}*. Educational summary only, no doses. "
               "Click a drug to open its full page.")

    c1, c2 = st.columns([2, 3])
    with c1:
        sec_names = [_REF_ALL] + [f"{s['no']}. {s['title']}" for s in secs]
        st.session_state.setdefault("ref_section", sec_names[1])
        sec_pick = st.selectbox("Section", sec_names, key="ref_section")
    with c2:
        query = st.text_input("Search the reference", key="ref_search",
                              placeholder="Any word, any column — e.g. phlegm, "
                                          "serotonin syndrome, CYP2D6, gout…")
    c3, c4 = st.columns([3, 1])
    with c3:
        tags = st.pills("On list", list(mref.TAG_LABELS), selection_mode="multi",
                        format_func=mref.TAG_LABELS.get, key="ref_tags")
    with c4:
        view = st.radio("View", ["Cards", "Table"], key="ref_view", horizontal=True)

    rows = all_rows
    # A search always covers every section — the Section box is for browsing.
    if sec_pick != _REF_ALL and not query.strip():
        no = int(sec_pick.split(".", 1)[0])
        rows = [e for e in rows if e["section_no"] == no]
    if tags:
        # "both" rows are on each list, so they satisfy either single-list filter
        want = set(tags) | ({"both"} if tags else set())
        rows = [e for e in rows if e["tag"] in want]
    if query.strip():
        rows = mref.search(query, rows)
        notes = []
        for t in mref.expand_query(query):
            if t["corrected_from"]:
                notes.append(f"*{t['corrected_from']}* → **{t['term']}**")
            if len(t["alts"]) > 1:
                notes.append(f"**{t['term']}** also matches " +
                             ", ".join(t["alts"][1:]))
        st.caption("Searching all sections." + (" " + " · ".join(notes) if notes else ""))
    elif sec_pick == _REF_ALL:
        st.info("Pick a section, or search to look across the whole reference.")
        return

    if not rows:
        st.info("Nothing matched — try fewer or different words.")
        return
    shown = rows[:_REF_RESULT_CAP] if query.strip() else rows
    more = f" (showing first {len(shown)})" if len(shown) < len(rows) else ""
    st.markdown(f"**{len(rows)}** entr{'ies' if len(rows) != 1 else 'y'}{more}")

    if view == "Table":
        _render_ref_table(shown, key=f"ref_tbl_{sec_pick}_{query}_{'-'.join(tags)}")
        return
    if query.strip():                     # search results: flat, best match first
        for e in shown:
            st.caption(f"{e['section']} › {e['subsection']}")
            _render_ref_card(e, "q")
        return
    current = None
    for e in shown:                       # section browse: grouped by sub-section
        if e["subsection"] != current:
            current = e["subsection"]
            st.subheader(current)
        _render_ref_card(e, "s")


def render_drug_of_the_day():
    """Featured 'Drug of the Day' card for the home page, plus an archive of the
    previous days. The pick is a deterministic function of the date, so the
    archive reproduces exactly with no stored state."""
    if not HAS_DRUG_LIBRARY or not dl.is_available():
        return
    import datetime as _dt
    today = _dt.date.today()
    drug = dl.drug_of_the_day(today)
    if not drug:
        return

    st.markdown("### 💊 Drug of the Day")
    st.caption(today.strftime("%A, %d %B %Y"))
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            st.button("🔬 Explore →", key="dotd_open", width='stretch',
                      on_click=_open_drug, args=(drug["query"],),
                      help=f"Open {drug['name']} in the Explorer")
        with c2:
            st.markdown(f"## {drug['name']}")
            if drug["blurb"]:
                st.markdown(drug["blurb"])
            st.markdown(_class_chips_html(drug["classes"][:8]),
                        unsafe_allow_html=True)

    with st.expander("📅 Drug of the Day — archive (previous days)"):
        for prev, pdrug in dl.drug_of_the_day_archive(days=14, end_date=today):
            a1, a2, a3 = st.columns([1.1, 1.4, 3])
            a1.caption(prev.strftime("%a %d %b"))
            a2.button(pdrug["name"], key=f"dotd_arch_{prev.isoformat()}",
                      width='stretch', on_click=_open_drug,
                      args=(pdrug["query"],))
            primary = pdrug["classes"][0] if pdrug["classes"] else None
            a3.caption(dl.class_name(primary) if primary else "")


def render_drug_class_banner(*names: Optional[str]):
    """If the resolved compound is a known library drug (matched against any of the
    candidate names), show its therapeutic classes as clickable chips that jump
    into the Drug Library for that class."""
    if not HAS_DRUG_LIBRARY:
        return
    drug = find_library_drug(*names)
    if not drug:
        return
    st.markdown(f"💊 **{drug['name']}** is a known medicine — therapeutic "
                f"class{'es' if len(drug['classes']) > 1 else ''} "
                "(click to browse the class):")
    # ATC (therapeutic) classes first, then FDA pharmacologic/mechanism classes.
    _order = {"ATC": 0, "EPC": 1, "MOA": 2}
    slugs = sorted(drug["classes"],
                   key=lambda s: (_order.get(dl.class_system(s), 3), dl.class_name(s)))
    CAP = 8
    shown, extra = slugs[:CAP], slugs[CAP:]
    cols = st.columns(min(len(shown), 4))
    for i, slug in enumerate(shown):
        sys_tag = dl.class_system(slug)
        cols[i % len(cols)].button(
            dl.class_name(slug), key=f"cls_jump_{drug['name']}_{slug}",
            width='stretch', on_click=_focus_class, args=(slug,),
            help=f"{sys_tag}: browse other {dl.class_name(slug)} in the Drug Library")
    if extra:
        st.caption("Also: " + ", ".join(dl.class_name(s) for s in extra))


def _focus_reference(e: Dict):
    """Jump to the Reference mode showing this row's section, filtered to it."""
    st.session_state["app_mode"] = _MODE_REFERENCE
    st.session_state["ref_section"] = f"{e['section_no']}. {e['section']}"
    st.session_state["ref_search"] = mref.drug_links(e)[0][1] or ""
    st.session_state["ref_tags"] = []
    st.session_state["ref_view"] = "Cards"


def render_reference_card(*names: Optional[str]):
    """The user's own Mainstream Drug Reference notes for this drug, if it's in
    there — matched by British name, US name in brackets, or salt-stripped name."""
    if not HAS_MAINSTREAM_REF:
        return
    lib = find_library_drug(*names)
    cands = [c for n in names if n for c in (n, _SALT_SUFFIX_RE.sub("", n))]
    if lib:
        cands.append(lib["name"])
    hits = mref.find_entries(*cands)
    if not hits:
        return
    with st.expander(f"📘 In your Mainstream Drug Reference — {hits[0]['class']}",
                     expanded=True):
        for i, e in enumerate(hits):
            if i:
                st.divider()
            badge = _REF_TAG_BADGE.get(e["tag"], "")
            st.markdown(f"**{e['drug']}** {badge}  \n"
                        f":gray[{e['section']} › {e['subsection']} · {e['class']}]")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Mechanism**  \n{e['mechanism']}")
            c2.markdown(f"**Main uses**  \n{e['uses']}")
            c3.markdown(f"**Key cautions & signature effects**  \n{e['cautions']}")
            st.button("See it in the Reference ›", key=f"ref_back_{e['id']}",
                      type="tertiary", on_click=_focus_reference, args=(e,))


# ---------------------- Data pipeline (no rendering) ----------------------
# Shared by Explore (full profile) and Ask (chat): resolve the input to a
# structure, then gather every source into one context dict. Rendering lives in
# render_full_profile() (Explore) and the chat answer builders (Ask).
_LOAD_LEVELS = {"struct": 0, "chem": 1, "full": 2}


def resolve_input(input_text: str, region: str = "US") -> Dict[str, Any]:
    """Name / brand / SMILES → structure candidates, without rendering.

    PubChem → OPSIN first; then the selected market's brand register (brands
    PubChem doesn't list as synonyms — Pepto-Bismol, local SG brands); the slow
    NCI CIR last. Returns {"candidates", "source", "brand", "error",
    "exception"}; "brand" = {"ingredients", "source"} when the input was a brand."""
    out: Dict[str, Any] = {"candidates": [], "source": None, "brand": None,
                           "error": "", "exception": None}
    text = (input_text or "").strip()
    if not text:
        out["error"] = "Nothing to look up."
        return out
    if Chem.MolFromSmiles(text) is not None:
        out.update(candidates=[{"smiles": text, "cid": None, "name": None}], source="smiles")
        return out
    try:
        # PubChem → OPSIN first; slow CIR only after the brand lookup
        result = cached_resolve_name(text, False)
    except Exception as e:  # noqa: BLE001
        result = {"source": None, "candidates": []}
        out["exception"] = e
    candidates = list(result.get("candidates") or [])
    source = result.get("source")
    if not candidates:
        # Not a chemical name — maybe a brand PubChem doesn't list as a
        # synonym (Pepto-Bismol, local SG brands). Map brand → active
        # ingredient(s) via the HSA register / RxNorm and resolve those.
        ingredients, brand_src = brand_to_ingredients(text, region)
        for ing in ingredients:
            try:
                res = cached_resolve_name(ing)
            except Exception:  # noqa: BLE001
                continue
            if res.get("candidates"):
                c0 = dict(res["candidates"][0])
                c0["name"] = c0.get("name") or ing
                candidates.append(c0)
                source = source or res.get("source")
        # a register mapping the name to itself ('Gabapentin' → gabapentin) is
        # not a brand — it only means the first lookup failed (e.g. rate limit)
        same = len(ingredients) == 1 and (
            ingredients[0].strip().lower() == text.lower()
            or (HAS_SG_REGISTER and ingredients[0].strip().lower() in sgr._variants(text))
            or (find_library_drug(text) is not None
                and find_library_drug(text) is find_library_drug(ingredients[0])))
        if candidates and not same:
            out["brand"] = {"ingredients": ingredients, "source": brand_src}
    if not candidates:
        # last resort: NCI CIR (slow, so it runs only after cheaper routes)
        try:
            result = cached_cir_resolve(text)
        except Exception:  # noqa: BLE001
            result = {"source": None, "candidates": []}
        candidates = list(result.get("candidates") or [])
        source = result.get("source") or source
    if not candidates:
        out["error"] = ("No matches found across PubChem, OPSIN and CIR, and it isn't a "
                        "known brand name. Check spelling, or paste a SMILES directly.")
    out.update(candidates=candidates, source=source)
    return out


def _assign_wiki_tabs(wiki_secs: Dict[str, Any]) -> None:
    """Assign each Wikipedia section to a tab once. A sub-section keeps its own
    specific match (e.g. "Mechanism of action" -> pharma) but otherwise inherits
    the tab of its parent top-level section, so "Antimalarial" under "Medical
    uses" lands in Uses rather than defaulting to Overview."""
    if not wiki_secs:
        return
    last_top_tab = "uses"
    for s in wiki_secs.get("sections", []):
        raw = _classify_wiki_raw(s["heading"])
        for ph in reversed(s.get("parents") or []):   # 'Pharmacodynamics' above it
            if raw:
                break
            raw = _classify_wiki_raw(ph)
        if s.get("level", 2) <= 2:
            last_top_tab = raw or "uses"
            s["tab"] = last_top_tab
        else:
            s["tab"] = raw or last_top_tab


def _clinical_enrichment(*, drug_hit, compound_name, resolved_name, synonyms,
                         clin_chembl, region, inchikey="", extras=True) -> Dict[str, Any]:
    """FDA label + FAERS + market (RxNorm/HSA) + GtoPdb for a compound.

    Fetched only when there's a prior drug signal (library membership or a
    ChEMBL development phase) — this both skips needless calls for plain
    chemicals and dodges openFDA's homeopathic-remedy false positives (e.g. a
    'Benzinum' label for benzene)."""
    drug_signal = bool(drug_hit) or bool(clin_chembl.get("max_phase"))
    # the form sold in the user's market decides which US label fits best
    route = market_route(region, (drug_hit or {}).get("name"), compound_name)
    if not drug_signal and compound_name:
        # ChEMBL can be slow/down; an FDA *prescription* label (already
        # prefetched, so free) is an equally strong signal. OTC labels don't
        # count — homeopathic remedies ('Benzinum' for benzene) are OTC.
        _probe = cached_openfda_label(compound_name, route)
        drug_signal = _probe.get("product_type") == "HUMAN PRESCRIPTION DRUG"

    openfda: Dict[str, Any] = {}
    adverse_events: List[Dict[str, Any]] = []
    market: Dict[str, Any] = {}
    if drug_signal:
        name_cands = []
        if drug_hit:
            name_cands.append(drug_hit["name"])
        # record title first; the resolver's name is often a systematic IUPAC
        # name that no label is filed under — only try it if it looks common
        name_cands += [compound_name] if compound_name else []
        name_cands += [resolved_name] if resolved_name and _name_like_synonym(resolved_name) else []
        name_cands += [s for s in synonyms if _name_like_synonym(s)][:4]
        seen_n: set = set()
        for nm in name_cands:
            key = nm.strip().lower()
            if not key or key in seen_n:
                continue
            seen_n.add(key)
            openfda = cached_openfda_label(nm, route)
            if openfda:
                # FAERS / label-count queries are slow for widely used drugs;
                # `extras=False` (chat questions that don't use them) skips both
                if extras:
                    of_generic = (openfda.get("generic_names") or [nm])[0]
                    adverse_events = cached_openfda_events(of_generic)
                    market = cached_openfda_market(of_generic)
                break

    gtopdb: Dict[str, Any] = {}
    if inchikey and (clin_chembl.get("chembl_id") or drug_hit):
        gtopdb = cached_gtopdb(inchikey)

    # A medicine gets the drug-monograph layout (Clinical tab, clinical safety
    # first); anything else keeps the chemistry-first layout.
    is_drug = drug_signal or openfda_is_medicine(openfda, drug_hit, clin_chembl)
    label = openfda if openfda_is_medicine(openfda, drug_hit, clin_chembl) else {}
    if is_drug:
        # RxNorm's single-ingredient brands (Lipitor, Coumadin…) beat brand names
        # mined from label counts, which skew to store brands.
        market = dict(market)
        market["rxnorm_brands"] = cached_rxnorm_brands(
            (drug_hit or {}).get("rxcui", ""),
            "" if drug_hit else (compound_name or resolved_name))
        if region == "SG" and HAS_SG_REGISTER:
            sg_names = [(drug_hit or {}).get("name"), compound_name, resolved_name,
                        *(openfda.get("generic_names") or [])[:1]]
            sg_names = [v for n in sg_names if n
                        for v in (n, _SALT_SUFFIX_RE.sub("", n))]
            market["sg"] = sgr.lookup(*sg_names,
                                      known_brands=market.get("rxnorm_brands") or ())
    return {"drug_signal": drug_signal, "openfda": openfda, "adverse_events": adverse_events,
            "market": market, "gtopdb": gtopdb, "is_drug": is_drug, "label": label}


def load_compound(input_text: str, chosen: Dict[str, Any], source: Optional[str], *,
                  region: str = "US", strip_salts: bool = True, include_rdkit: bool = True,
                  level: str = "full", chembl_budget: float = _CHEMBL_BUDGET_S,
                  recover_tautomer: bool = True, extras: bool = True) -> Dict[str, Any]:
    """Fetch and assemble everything known about one resolved compound — no
    rendering. Returns the context dict used by render_full_profile() and the
    chat answers, or {"error": …} when no structure could be obtained.

    `level` bounds the work (the chat asks only for what a question needs):
    'struct' = structure only (PubChemPy + RDKit), 'chem' = + the PubChem record,
    'full' = every source (Explore always uses 'full')."""
    full = level == "full"
    chem = level in ("chem", "full")
    errors: List[Exception] = []
    smiles = chosen.get("smiles") or ""
    cid = chosen.get("cid")
    resolved_name = chosen.get("name") or None
    rawinfo: Dict[str, Any] = {}

    # Salt / counterion handling: keep the EXACT searched compound as primary,
    # and (when stripping is on) also note its parent free base/acid so the
    # richer parent data can be shown later in clearly-labelled blocks — the
    # user can then tell salt-specific data apart from parent-extrapolated data.
    # ---- Parallel prefetch. The flow below is unchanged and still sequential;
    # it just finds its data already in st.cache_data. Was ~20 one-at-a-time
    # round-trips. Pipelined: structure-keyed sources start immediately; as
    # soon as the PubChem record gives the compound's name, the name-keyed
    # sources (Wikipedia, openFDA, RxNorm) start too — without waiting for
    # slow ChEMBL. Name-keyed clinical calls are speculative for non-drugs
    # (cheap, in parallel); the main flow's drug gating still decides display.
    _ik0 = inchikey_for(smiles) if smiles else ""
    _multi = bool(full and strip_salts and smiles and is_multifragment(smiles))
    _frag_ik = inchikey_for(largest_fragment_smiles(smiles)) if _multi else ""
    # ChEMBL is the one source that regularly stalls (0.7–8 s normally, full
    # timeouts in bad spells). Its futures are kept so the main flow reads
    # them directly with a time budget instead of re-requesting; the page
    # renders without ChEMBL if it hasn't answered by then.
    _t_prefetch = time.perf_counter()
    _ex = ThreadPoolExecutor(max_workers=12)
    _chembl_futs: Dict[str, Any] = {}
    if full:
        for _k in dict.fromkeys(k for k in (_ik0, _frag_ik) if k):
            _chembl_futs[_k] = submit_all(_ex, (cached_chembl, _k))[0]
    with st.spinner("Fetching PubChem, ChEMBL, Wikipedia, FDA…" if full
                    else "Fetching the structure from PubChem…"):
        _f_flat = submit_all(_ex, (cached_flat_pug, cid) if chem else None)
        _futs = submit_all(_ex, (cached_compound_by_smiles, smiles),
                           (cached_synonyms, cid) if chem else None)
        if full:
            _f_pcid = submit_all(_ex, (cached_parent_cid, cid) if _multi else None)
            _f_wtitle = submit_all(_ex, (cached_wiki_title, _ik0))
            _futs += submit_all(
                _ex,
                (cached_wikidata, _ik0), (cached_unichem, _ik0),
                *([(cached_wikidata, _frag_ik), (cached_unichem, _frag_ik)]
                  if _frag_ik and _frag_ik != _ik0 else []),
            )
            # same name the main flow will derive (record title, else resolver name)
            _flat0 = (_quiet_result(_f_flat[0], {}) if _f_flat else {}) or {}
            _name0 = _flat0.get("Name") or resolved_name or input_text
            _drug0 = find_library_drug(_name0, resolved_name, input_text)
            _of_name = (_drug0 or {}).get("name") or _name0
            _f_label = submit_all(_ex, (cached_openfda_label, _of_name,
                                        market_route(region, (_drug0 or {}).get("name"), _name0)))
            _futs += submit_all(
                _ex,
                (cached_wiki, _name0), (cached_wiki_sections, _name0),
                (cached_rxnorm_brands, (_drug0 or {}).get("rxcui", ""),
                 "" if _drug0 else _name0),
                (cached_gtopdb, _ik0),
            )
            _wt = _quiet_result(_f_wtitle[0], "") if _f_wtitle else ""
            _futs += submit_all(_ex, (cached_wiki_sections, _wt) if _wt != _name0 else None)
            _p_cid = _quiet_result(_f_pcid[0]) if _f_pcid else None
            _futs += submit_all(_ex, (cached_flat_pug, _p_cid), (cached_synonyms, _p_cid))
            # label-dependent: FAERS + market counts, keyed on the label's generic
            _of0 = (_quiet_result(_f_label[0], {}) if _f_label else {}) or {}
            if _of0 and extras:
                _gen0 = (_of0.get("generic_names") or [_of_name])[0]
                _futs += submit_all(_ex, (cached_openfda_events, _gen0),
                                    (cached_openfda_market, _gen0))
        for _f in _futs + _f_flat:
            _quiet_result(_f)
        if _chembl_futs:
            futures_wait(list(_chembl_futs.values()),
                         timeout=max(0.0, chembl_budget - (time.perf_counter() - _t_prefetch)))
    _ex.shutdown(wait=False)   # a stalled ChEMBL call finishes in the background
    chembl_timed_out = False

    def _chembl_for(ik: str) -> Dict[str, Any]:
        """ChEMBL summary for an InChIKey: from the prefetch future if there is
        one (empty — not re-requested — when it missed the time budget), else
        via the cache. Only ever called at level 'full'."""
        nonlocal chembl_timed_out
        if not ik or not full:
            return {}
        f = _chembl_futs.get(ik)
        if f is None:
            return cached_chembl(ik)
        if not f.done():
            chembl_timed_out = True
            return {}
        return _quiet_result(f, {}) or {}

    parent_cid = None
    parent_smiles = ""
    if full and strip_salts and smiles and is_multifragment(smiles):
        cand = largest_fragment_smiles(smiles)
        if cand and cand != smiles:
            parent_smiles = cand
            parent_cid = cached_parent_cid(cid) if cid else None

    # Enrich via PubChemPy (rich dict: atoms/bonds/etc.)
    if smiles:
        try:
            comp = cached_compound_by_smiles(smiles)
            rawinfo = comp.get("rawinfo", {}) or {}
            cid = cid or comp.get("cid")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    if not smiles and cid:
        smiles = cached_smiles_for_cid(cid) or ""

    if not smiles:
        return {"error": "Could not obtain a SMILES for this query.", "errors": errors}

    # Canonicalize
    canon = canonicalize_smiles(smiles)
    if canon:
        smiles = canon["isomeric"]

    # ---------- Assemble property dicts ----------
    pruned = {k: v for k, v in rawinfo.items() if v not in (None, [], {}, "")} if isinstance(rawinfo, dict) else {}
    if include_rdkit and smiles:
        for k, v in rdkit_properties(smiles).items():
            pruned.setdefault(k, v)

    conn_keys = {"atom_stereo_count", "atoms", "bond_stereo_count", "bonds"}
    all_general = {k: pruned[k] for k in pruned if k not in conn_keys}
    active_exclude = {_norm_key(x) for x in EXCLUDE_GENERAL_KEYS}
    general_info = {k: v for k, v in all_general.items() if _norm_key(k) not in active_exclude}
    connectivity_info = {k: pruned[k] for k in conn_keys if k in pruned}

    if not cid and "cid" in pruned:
        try:
            cid = int(str(pruned["cid"]).strip())
        except Exception:
            cid = None

    # Rich record + supplementary data (cached)
    flat_summary = cached_flat_pug(cid) if (cid and chem) else {}
    record_text = find_record_description(flat_summary) if flat_summary else ""
    compound_name = (flat_summary.get("Name") if flat_summary else None) or resolved_name \
        or general_info.get("iupac_name") or input_text
    synonyms = cached_synonyms(cid) if (cid and chem) else []

    wiki = cached_wiki(compound_name) if (compound_name and full) else {}

    # Cross-database lookups for the EXACT searched compound (keyed by its own
    # InChIKey). For a salt these are often thin — the parent bundle below fills
    # the gaps, kept separate so the two scopes stay distinguishable.
    def _lookup_bundle(b_cid, b_smiles, name_hint):
        ik = inchikey_for(b_smiles)
        flat = cached_flat_pug(b_cid) if b_cid else {}
        record = find_record_description(flat) if flat else ""
        name = (flat.get("Name") if flat else None) or name_hint
        syn = cached_synonyms(b_cid) if b_cid else []
        # never look the placeholder 'parent compound' up on Wikipedia (it
        # finds the article 'Parent structure' when PubChem is unreachable)
        wk = cached_wiki(name) if name and name != name_hint else {}
        wd = cached_wikidata(ik) if ik else {}
        if (wd.get("description") or "").strip().lower() in _GENERIC_WIKIDATA_DESC:
            wd = {}  # drop useless "chemical compound"-type descriptions
        grp = refine_pharma_topics(consolidate_grouped(dedup_pubchem_grouped(build_grouped(
            {k: v for k, v in flat.items()
             if not (k and k.strip().lower() == "record description")}
        ))), [name] + [s for s in syn if _name_like_synonym(s)][:5]) if flat else {}
        return {
            "cid": b_cid, "smiles": b_smiles, "name": name, "inchikey": ik,
            "flat": flat, "record": record, "synonyms": syn, "wiki": wk,
            "wikidata": wd, "chembl": _chembl_for(ik),
            "xrefs": cached_unichem(ik) if ik else [], "grouped": grp,
        }

    inchikey = inchikey_for(smiles)
    chembl = _chembl_for(inchikey)
    wikidata = cached_wikidata(inchikey) if (inchikey and full) else {}
    if (wikidata.get("description") or "").strip().lower() in _GENERIC_WIKIDATA_DESC:
        wikidata = {}
    xrefs = cached_unichem(inchikey) if (inchikey and full) else []

    # Tautomer/protonation recovery: when the searched depiction's own InChIKey
    # misses ChEMBL (typical for PubChem records stored as an unusual tautomer,
    # e.g. the "carboximidic acid" form the bare name 'doxycycline' resolves to),
    # find the mainstream record by name and VERIFY it is the same substance via
    # canonical-tautomer InChIKey before adopting it. That recovered key then
    # unlocks the ChEMBL card, UniChem cross-refs (DrugBank, …) and Wikidata that
    # structure-matching alone could never reach.
    # A salt's own InChIKey often misses ChEMBL while its free base hits
    # (Lipitor → atorvastatin *calcium*). Try that cheap structure lookup before
    # the slow name-search recovery below; the result feeds the clinical
    # summary (drug detection, At-a-Glance), while the Pharmacology tab still
    # shows it under the parent-compound block.
    frag_chembl: Dict[str, Any] = {}
    if full and inchikey and not chembl.get("chembl_id") and is_multifragment(smiles):
        frag_ik = inchikey_for(largest_fragment_smiles(smiles))
        if frag_ik and frag_ik != inchikey:
            frag_chembl = _chembl_for(frag_ik)

    canonical_ik = ""
    if full and recover_tautomer and inchikey and not chembl.get("chembl_id") \
            and not frag_chembl.get("chembl_id"):
        my_taut = canonical_tautomer_inchikey(smiles)
        if my_taut:
            fallback_ik = ""
            # ChEMBL name search is slow (3–25 s per query) — run the synonyms
            # concurrently under one time budget instead of one after another,
            # then check results in synonym-priority order. Searches still
            # running at the deadline finish in the background (into the cache).
            _names = [s for s in synonyms if _name_like_synonym(s)][:6]
            _by_name: Dict[str, List[Dict[str, str]]] = {}
            if _names and not chembl_timed_out and not cs.host_degraded(cs.CHEMBL_BASE):
                _ex = ThreadPoolExecutor(max_workers=len(_names))
                _nf = dict(zip(_names, submit_all(
                    _ex, *[(cached_chembl_by_name, nm) for nm in _names])))
                futures_wait(list(_nf.values()), timeout=_CHEMBL_NAME_BUDGET_S)
                _ex.shutdown(wait=False, cancel_futures=True)
                _by_name = {nm: _quiet_result(f, []) or [] for nm, f in _nf.items()
                            if f.done()}
            for nm in _names:
                for cand in _by_name.get(nm, []):
                    cand_smi, cand_ik = cand.get("smiles"), cand.get("inchikey")
                    if not (cand_smi and cand_ik):
                        continue
                    if canonical_tautomer_inchikey(cand_smi) == my_taut:
                        if not is_multifragment(cand_smi):   # prefer the free base
                            canonical_ik = cand_ik
                            break
                        fallback_ik = fallback_ik or cand_ik
                if canonical_ik:
                    break
            canonical_ik = canonical_ik or fallback_ik
        if canonical_ik:
            chembl = cached_chembl(canonical_ik) or chembl
            # Merge cross-refs from both keys (same compound): the canonical form
            # carries the drug databases (DrugBank, DrugCentral, ChEBI, …) the
            # tautomer depiction lacks; keep any primary-only links too.
            canon_xrefs = cached_unichem(canonical_ik)
            if canon_xrefs:
                seen = {x["db"] for x in canon_xrefs}
                xrefs = canon_xrefs + [x for x in xrefs if x["db"] not in seen]
            if not wikidata.get("description"):
                wd2 = cached_wikidata(canonical_ik)
                if (wd2.get("description") or "").strip().lower() not in _GENERIC_WIKIDATA_DESC:
                    wikidata = wd2 or wikidata

    grouped = build_grouped(
        {k: v for k, v in flat_summary.items() if not (k and k.strip().lower() == "record description")}
    ) if flat_summary else {}
    # Drop sentences PubChem repeats across headings (keeping each fact in its
    # most-pertinent heading) before Wikipedia is merged on top.
    grouped = dedup_pubchem_grouped(grouped)

    # Full Wikipedia article, split into sections, to backfill tabs PubChem left
    # thin. Try titles most-reliable-first and stop at the first that yields an
    # article: structure-grounded InChIKey -> Wikidata sitelink, then the
    # redirect-resolved lead-summary title, the record name, and finally the
    # compound's common/brand synonyms (which catch cases where the primary CID
    # is an obscure stereoisomer whose own name has no page — e.g. a PubChem
    # record titled "Alpha-6-Deoxyoxytetracycline" that is really doxycycline).
    wiki_secs: Dict[str, Any] = {}
    if full:
        _title_candidates: List[str] = []
        for _cand in [(cached_wiki_title(inchikey) if inchikey else ""),
                      (wiki.get("title") if wiki else ""), compound_name]:
            if _cand and _cand not in _title_candidates:
                _title_candidates.append(_cand)
        for _syn in synonyms:
            if _name_like_synonym(_syn) and _syn not in _title_candidates:
                _title_candidates.append(_syn)
            if len(_title_candidates) >= 9:
                break
        for _title in _title_candidates:
            wiki_secs = cached_wiki_sections(_title)
            if wiki_secs:
                break
        wiki_secs, wiki = _prefer_medication_article(wiki_secs, wiki)
        _assign_wiki_tabs(wiki_secs)
    # Merge the Wikipedia sections into the PubChem group structure so they
    # render as peer, point-form headings inside the group expanders — not as a
    # separate "From Wikipedia" block.
    grouped = merge_wiki_into_grouped(grouped, wiki_secs)
    # Fold the pharmacology group's many synonymous headings (Mechanism/PD,
    # ADME/Metabolism/Half-life, Uses/Indication) into canonical topics and
    # collapse bare identifier headings into one section.
    grouped = consolidate_grouped(grouped)
    # Keep pharmacology headings on-topic: journal abstracts, disease
    # background, trial results, side effects and PK re-homed out of Mechanism.
    grouped = refine_pharma_topics(
        grouped, [compound_name, resolved_name or ""]
        + [s for s in synonyms if _name_like_synonym(s)][:5])
    # If the lead-paragraph summary (Overview) missed but section resolution
    # found the real page, refresh the summary from that canonical title so the
    # Overview lead isn't blank while the tabs are full.
    if wiki_secs and not wiki.get("extract"):
        wiki = cached_wiki(wiki_secs.get("title") or "") or wiki

    # Parent (free base/acid) bundle — only when the primary is a salt/mixture.
    parent = None
    if parent_smiles:
        p_cid = parent_cid
        if not p_cid:  # salt had no CID (e.g. raw SMILES input) — resolve via the parent SMILES
            try:
                p_cid = cached_compound_by_smiles(parent_smiles).get("cid")
            except Exception:
                p_cid = None
        parent = _lookup_bundle(p_cid, parent_smiles, "parent compound")
    has_parent = bool(parent and (parent["record"] or parent["wiki"].get("extract")
                                  or parent["chembl"].get("chembl_id")
                                  or parent["grouped"] or parent["xrefs"]))

    # ---------- Clinical / regulatory enrichment ----------
    formula = pruned.get("molecular_formula") or pruned.get("formula") or ""
    drug_hit = find_library_drug(compound_name, resolved_name, input_text)
    # ChEMBL record for the clinical summary: the exact compound's, else its free base's.
    clin_chembl = chembl if chembl.get("chembl_id") else frag_chembl
    if full:
        clinical = _clinical_enrichment(
            drug_hit=drug_hit, compound_name=compound_name, resolved_name=resolved_name,
            synonyms=synonyms, clin_chembl=clin_chembl, region=region, inchikey=inchikey,
            extras=extras)
    else:
        clinical = {"drug_signal": bool(drug_hit), "openfda": {}, "adverse_events": [],
                    "market": {}, "gtopdb": {}, "is_drug": bool(drug_hit), "label": {}}
    # Mechanism renders summary-first: ChEMBL + FDA-label key points on top.
    attach_mechanism_key_points(grouped, clin_chembl, clinical["label"])
    if parent:
        attach_mechanism_key_points(parent["grouped"], parent["chembl"] or clin_chembl,
                                    clinical["label"])

    return {
        "level": level, "name_only": False, "errors": errors, "region": region,
        "input_text": input_text, "smiles": smiles, "cid": cid, "source": source,
        "resolved_name": resolved_name, "rawinfo": rawinfo, "general_info": general_info,
        "connectivity_info": connectivity_info, "flat_summary": flat_summary,
        "record_text": record_text, "compound_name": compound_name, "synonyms": synonyms,
        "wiki": wiki, "inchikey": inchikey, "chembl": chembl, "frag_chembl": frag_chembl,
        "canonical_ik": canonical_ik, "wikidata": wikidata, "xrefs": xrefs,
        "grouped": grouped, "wiki_secs": wiki_secs, "parent": parent,
        "has_parent": has_parent, "formula": formula, "drug_hit": drug_hit,
        "clin_chembl": clin_chembl, "chembl_timed_out": chembl_timed_out, **clinical,
    }


def _prefer_medication_article(secs: Dict[str, Any], wiki: Dict[str, Any]
                               ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Wikipedia splits some substances into a physiology article and a drug
    article — 'Glucagon' (the hormone) vs 'Glucagon (medication)', likewise
    insulin, oxytocin, testosterone. The drug article is the one with medical
    uses, dosing and e.g. why glucagon treats beta-blocker overdose, so use it
    (sections and lead summary) whenever it exists."""
    title = (secs or {}).get("title") or ""
    if not title or title.endswith("(medication)"):
        return secs, wiki
    med = cached_wiki_sections(f"{title} (medication)")
    if not (med and med.get("sections")) or med.get("title") == title:
        return secs, wiki
    return med, (cached_wiki(med["title"]) or wiki)


def load_name_only(name: str, *, region: str = "US", drug=None) -> Dict[str, Any]:
    """Fallback context for a medicine with no small-molecule structure (a
    biologic such as insulin or adalimumab) or a name PubChem can't resolve:
    everything keyed by NAME — FDA label, Wikipedia, RxNorm/HSA brands — with
    no PubChem record, ChEMBL or structure. Same keys as load_compound()."""
    drug = drug or find_library_drug(name)
    title = (drug or {}).get("name") or name
    with st.spinner(f"Fetching FDA label and Wikipedia for {title}…"):
        wiki = cached_wiki(title) or (cached_wiki(name) if name != title else {})
        wiki_secs = cached_wiki_sections(wiki.get("title") or title) or {}
        wiki_secs, wiki = _prefer_medication_article(wiki_secs, wiki)
        clinical = _clinical_enrichment(
            drug_hit=drug, compound_name=title, resolved_name=name, synonyms=[],
            clin_chembl={}, region=region)
    _assign_wiki_tabs(wiki_secs)
    grouped = refine_pharma_topics(consolidate_grouped(merge_wiki_into_grouped({}, wiki_secs)),
                                   [title, name])
    attach_mechanism_key_points(grouped, {}, clinical["label"])
    return {
        "level": "full", "name_only": True, "errors": [], "region": region,
        "input_text": name, "smiles": "", "cid": None, "source": None,
        "resolved_name": name, "rawinfo": {}, "general_info": {}, "connectivity_info": {},
        "flat_summary": {}, "record_text": "", "compound_name": title, "synonyms": [],
        "wiki": wiki, "inchikey": "", "chembl": {}, "frag_chembl": {}, "canonical_ik": "",
        "wikidata": {}, "xrefs": [], "grouped": grouped, "wiki_secs": wiki_secs,
        "parent": None, "has_parent": False, "formula": "", "drug_hit": drug,
        "clin_chembl": {}, "chembl_timed_out": False, **clinical,
    }


def render_full_profile(ctx: Dict[str, Any], *, show_atoms: bool = False,
                        show_bonds: bool = False, enable_2d_download: bool = True,
                        enable_3d_download: bool = False, debug_mode: bool = False):
    """The Explore page for one compound: header, At-a-Glance card and tabs."""
    smiles, cid, source = ctx["smiles"], ctx["cid"], ctx["source"]
    resolved_name, input_text = ctx["resolved_name"], ctx["input_text"]
    rawinfo, general_info = ctx["rawinfo"], ctx["general_info"]
    connectivity_info = ctx["connectivity_info"]
    record_text, compound_name, synonyms = ctx["record_text"], ctx["compound_name"], ctx["synonyms"]
    wiki, inchikey, chembl = ctx["wiki"], ctx["inchikey"], ctx["chembl"]
    canonical_ik, wikidata, xrefs = ctx["canonical_ik"], ctx["wikidata"], ctx["xrefs"]
    grouped, parent, has_parent = ctx["grouped"], ctx["parent"], ctx["has_parent"]
    formula, drug_hit, clin_chembl = ctx["formula"], ctx["drug_hit"], ctx["clin_chembl"]
    openfda, adverse_events, market = ctx["openfda"], ctx["adverse_events"], ctx["market"]
    gtopdb, is_drug, label = ctx["gtopdb"], ctx["is_drug"], ctx["label"]
    chembl_timed_out, region = ctx["chembl_timed_out"], ctx["region"]
    local_route = market_route(region, (drug_hit or {}).get("name"), compound_name)
    # ---------- Header banner ----------
    src_label = {"pubchem": "PubChem", "opsin": "OPSIN", "cir": "CIR/CACTUS",
                 "smiles": "direct SMILES"}.get(source, source or "—")
    h1, h2 = st.columns([4, 1])
    with h1:
        title = compound_name
        if label:
            gen = (label.get("generic_names") or [""])[0]
            if gen and gen.lower() != compound_name.lower():
                title += f"  ·  *{gen.lower()}*"
        st.subheader(title)
        meta = []
        if formula:
            meta.append(f"**{formula}**")
        if cid:
            meta.append(f"[PubChem CID {cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
        if clin_chembl.get("chembl_id") and clin_chembl.get("url"):
            meta.append(f"[{clin_chembl['chembl_id']}]({clin_chembl['url']})")
        if label.get("url"):
            meta.append(f"[DailyMed label]({label['url']})")
        meta.append(f"resolved via *{src_label}*")
        st.markdown(" · ".join(meta))
        if has_parent:
            st.caption("🧂 This is a **salt/multi-component** structure. Data "
                       "below is split: 📍 *specific to this compound* vs ↩ "
                       f"*extrapolated from the parent* ({parent['name']}, "
                       f"CID {parent['cid']}).")
    with h2:
        img_hdr = render_2d_pil(smiles, size=(220, 220))
        if img_hdr is not None:
            st.image(img_hdr, width='content')

    # ---------- At-a-Glance card (synthesis above the tabs) ----------
    render_at_a_glance(
        compound_name=compound_name, formula=formula, smiles=smiles,
        dm=derived_metrics(smiles), chembl=clin_chembl, wikidata=wikidata,
        wiki=wiki, record_text=record_text, drug=drug_hit,
        openfda=openfda, adverse_events=adverse_events, market=market,
        region=region, record_use=record_indication_sentence(grouped),
    )
    down = [nm for nm, url in (("PubChem", cs.PUBCHEM_BASE), ("ChEMBL", cs.CHEMBL_BASE),
                               ("openFDA", cs.OPENFDA_LABEL), ("RxNorm", cs.RXNAV_BASE))
            if cs.host_failing(url) or (nm == "ChEMBL" and chembl_timed_out)]
    if down:
        st.caption(f"⚠️ {', '.join(down)} didn't respond just now — some clinical "
                   "details may be missing. Reload in a few minutes to retry.")

    # ---------- Tabs ----------
    # Medicines get a monograph-style order (what it's for → how to take it →
    # safety → how it works) with chemistry after; other compounds keep the
    # chemistry-first order. The Clinical tab only exists with an FDA label.
    if is_drug:
        tab_order = ["overview"] + (["clinical"] if label else []) + \
            ["safety", "pharma", "props", "uses", "structure", "spectra", "refs"]
        tab_names = dict(TAB_LABELS, clinical="Uses & Dosing", safety="Side Effects & Safety",
                         props="Chemistry", uses="Background & Regulatory")
    else:
        tab_order = [k for k, _ in TAB_LABELS]
        tab_names = dict(TAB_LABELS)
    tab_objs = st.tabs([tab_names[k] for k in tab_order])
    tabs = dict(zip(tab_order, tab_objs))
    tab_overview = tabs["overview"]
    tab_props = tabs["props"]
    tab_safety = tabs["safety"]
    tab_pharma = tabs["pharma"]
    tab_uses = tabs["uses"]
    tab_spectra = tabs["spectra"]
    tab_3d = tabs["structure"]
    tab_all = tabs["refs"]

    def _groups_for(tab_key):
        return [g for g, t in GROUP_TO_TAB.items() if t == tab_key]

    def render_group_tab(tab_key, empty_msg, groups=None, key_suffix="",
                         expand_first=True):
        """Render a tab's grouped data (PubChem + merged Wikipedia headings) for
        the exact compound, then — when a parent exists — the parent's data in a
        clearly-labelled block. `groups` narrows to a subset of the tab's groups."""
        groups = groups if groups is not None else _groups_for(tab_key)
        kp = f"{tab_key}{key_suffix}"
        primary_has = bool(grouped) and any(grouped.get(g) for g in groups)
        if has_parent and primary_has:
            this_compound_caption(compound_name, cid)
        if primary_has:
            render_group_expanders(grouped, groups, cid, key_prefix=kp,
                                   expand_first=expand_first)
        if not primary_has and not has_parent:
            st.caption(empty_msg)
        if has_parent:
            if primary_has:
                st.markdown("---")
            parent_scope_callout(parent)
            if parent["grouped"] and any(parent["grouped"].get(g) for g in groups):
                render_group_expanders(parent["grouped"], groups, parent["cid"],
                                       key_prefix=f"{kp}_parent",
                                       expand_first=expand_first)
            else:
                st.caption("No data in this category for the parent compound.")

    # ===== Overview =====
    with tab_overview:
        if has_parent:
            this_compound_caption(compound_name, cid)
        # Cross-link to the Drug Library if this compound is a known medicine.
        render_drug_class_banner(compound_name, resolved_name, input_text)
        render_reference_card(input_text, resolved_name, compound_name)
        # One best summary up front; the other sources' descriptions sit behind
        # an expander rather than stacking three overlapping summaries.
        any_primary = render_summaries(wiki, record_text, wikidata, cid, compact=True)
        if not any_primary and not has_parent:
            st.caption("No descriptive summary available for this compound.")
        elif not any_primary and has_parent:
            st.caption("No salt-specific summary — see the parent compound below.")

        # Drug-likeness snapshot — a discovery-chemistry metric, so for a
        # marketed medicine it lives in the Chemistry tab instead.
        dm = derived_metrics(smiles)
        if dm and not is_drug:
            st.markdown("#### Drug-likeness snapshot")
            if has_parent:
                st.caption("📍 Computed from the exact searched structure "
                           "(includes the counterion — e.g. molecular weight).")
            ro5 = dm.get("_ro5_violations", 0)
            badge = "✅ Passes Lipinski Rule of 5" if ro5 == 0 else f"⚠️ {ro5} Lipinski violation(s)"
            st.markdown(badge)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mol. weight", f"{dm['Molecular weight'][0]} Da")
            c2.metric("cLogP", dm["cLogP (Crippen)"][0])
            c3.metric("TPSA", f"{dm['TPSA'][0]} Å²")
            qv = dm["QED (drug-likeness)"][0]
            c4.metric("QED", qv)
            render_druglikeness_chart(dm)

        if has_parent:
            st.markdown("---")
            parent_scope_callout(parent)
            if not render_summaries(parent["wiki"], parent["record"],
                                    parent["wikidata"], parent["cid"], compact=True):
                st.caption("No descriptive summary for the parent compound.")

    # ===== Uses & Dosing (FDA label) — medicines only =====
    if "clinical" in tabs:
        with tabs["clinical"]:
            render_label_provenance(label, region=region, sg=market.get("sg"),
                                    drug_routes=clin_chembl.get("routes") or (),
                                    local_route=local_route)
            st.caption("Each section shows the key points first — tick *Show full "
                       "label text* for the complete wording.")
            _combination_label_note(label)
            if not _render_label_sections(label, CLINICAL_SECTIONS, "clin"):
                st.caption("The FDA label for this drug has no usage/dosing sections.")
            st.caption("For information only — not medical advice. Dosing must be "
                       "individualised by a prescriber.")

    # ===== Properties =====
    with tab_props:
        if has_parent:
            this_compound_caption(compound_name, cid)
        st.markdown("#### Structure")
        st.code(smiles, language="text")
        if is_drug and dm:
            st.markdown("#### Drug-likeness")
            ro5 = dm.get("_ro5_violations", 0)
            st.markdown("✅ Passes Lipinski Rule of 5" if ro5 == 0
                        else f"⚠️ {ro5} Lipinski violation(s)")
            render_druglikeness_chart(dm)
        st.markdown("#### General Compound Data")
        if general_info:
            for prop_key in order_keys(list(general_info.keys())):
                tip = PROPERTY_GLOSSARY.get(_norm_key_for_order(prop_key)) \
                    or PROPERTY_GLOSSARY.get(prop_key) or "No description available."
                src_tag = " `RDKit`" if prop_key in RDKIT_KEYS and prop_key not in rawinfo else ""
                col_label, col_val = st.columns([1.4, 3.6])
                with col_label:
                    st.markdown(f"**{prop_key}**{src_tag}", help=tip)
                with col_val:
                    st.write(_format_value(prop_key, general_info[prop_key]))
        else:
            st.write("No general PubChem metadata available.")

        dm = derived_metrics(smiles)
        if dm:
            st.markdown("#### RDKit-derived descriptors")
            rows = [{"Property": k, "Value": f"{v[0]}{v[1]}",
                     "Basis": DESCRIPTOR_BASIS.get(k, "Derived")}
                    for k, v in dm.items() if not k.startswith("_")]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            st.caption("‘Basis’ shows how each value is obtained — *Derived* is "
                       "exact from the structure; *Estimated* uses a grounded "
                       "group-contribution method; *Heuristic* is a scoring rule.")

        st.markdown("#### Inferred from structure (functional-group reasoning)")
        st.caption("Works for any molecule, however sparsely studied — these are "
                   "deduced from the exact structure searched, not looked up.")
        render_structure_inference(smiles)

        st.markdown("#### Physical & experimental properties (PubChem)")
        render_group_tab("props", "No physical/experimental properties available.")

    # ===== Safety, Toxicity & Environment =====
    with tab_safety:
        if is_drug:
            # Clinical safety first (what a patient/prescriber needs), then the
            # toxicology record; lab/industrial hazard data last and collapsed.
            if label:
                render_label_provenance(label, region=region, sg=market.get("sg"),
                                        drug_routes=clin_chembl.get("routes") or (),
                                        local_route=local_route)
            else:
                st.info("❔ **No FDA label found**, so there is no official list of "
                        "contraindications, warnings or interactions here. Their absence "
                        "below is missing information — not evidence that none exist.")
            _combination_label_note(label)
            if label.get("boxed_warning"):
                st.error("**⚠️ FDA Boxed Warning**\n\n" + label["boxed_warning"])
            if label:
                _render_label_sections(label, SAFETY_LABEL_SECTIONS, "safe")
            if adverse_events:
                st.markdown("#### Most-reported adverse events (FDA FAERS)")
                render_faers_chart(adverse_events)
            clinical_groups = ["Toxicity & Health"]
            other_groups = [g for g in _groups_for("safety") if g not in clinical_groups]
            st.markdown("#### Toxicity & overdose data (PubChem / Wikipedia)")
            render_group_tab("safety", "No toxicity data in the PubChem record.",
                             groups=clinical_groups, expand_first=not label)
            if any(grouped.get(g) for g in other_groups) or has_parent:
                st.markdown("#### Handling, occupational & environmental")
                st.caption("Hazard classification, workplace exposure and "
                           "environmental fate — relevant to handling the bulk "
                           "substance, not to taking the medicine.")
                render_group_tab("safety", "No handling/environmental data.",
                                 groups=other_groups, key_suffix="_env",
                                 expand_first=False)
        else:
            render_group_tab("safety",
                             "No safety/toxicity data available (CID missing or fetch failed).")

    # ===== Pharmacology & Bioactivity =====
    with tab_pharma:
        if has_parent:
            this_compound_caption(compound_name, cid)
        if canonical_ik and chembl.get("chembl_id"):
            canonical_form_callout(canonical_ik)
        primary_chembl = render_chembl_card(chembl)
        if render_gtopdb_targets(gtopdb):
            primary_chembl = True
        primary_pharma = bool(grouped) and any(grouped.get(g) for g in _groups_for("pharma"))
        if primary_pharma:
            if primary_chembl:
                st.markdown("---")
            render_group_expanders(grouped, _groups_for("pharma"), cid, key_prefix="pharma")
        if not primary_chembl and not primary_pharma and not has_parent:
            st.caption("No pharmacology data available (PubChem record thin and "
                       "no ChEMBL entry for this structure).")
        if has_parent:
            if primary_chembl or primary_pharma:
                st.markdown("---")
            parent_scope_callout(parent)
            p_chembl = render_chembl_card(parent["chembl"])
            p_pharma = parent["grouped"] and any(parent["grouped"].get(g) for g in _groups_for("pharma"))
            if p_pharma:
                if p_chembl:
                    st.markdown("---")
                render_group_expanders(parent["grouped"], _groups_for("pharma"),
                                       parent["cid"], key_prefix="pharma_parent")
            if not p_chembl and not p_pharma:
                st.caption("No pharmacology data for the parent compound.")

    # ===== Uses & Regulatory =====
    with tab_uses:
        render_group_tab("uses",
                         "No uses/regulatory data available (CID missing or fetch failed).")

    # ===== Spectra =====
    with tab_spectra:
        render_group_tab("spectra", "No spectral data available.")

    # ===== 3D & Structure =====
    with tab_3d:
        st.markdown("#### 3D Structure (interactive)")
        viewer, molblock, embed_err = render_3d_viewer_and_block(smiles, width=700, height=480)
        if viewer is not None:
            try:
                components.html(viewer._make_html(), height=520)
                if embed_err and debug_mode:
                    st.info(f"3D info: {embed_err}")
            except Exception as e:
                st.write("Interactive 3D viewer unavailable in this environment.")
                if debug_mode:
                    st.exception(e)
            if enable_3d_download:
                try:
                    html_code = viewer._make_html()
                    st.download_button("Download 3D HTML", data=html_code.encode("utf8"),
                                       file_name=f"{cid or 'compound'}_3d.html", mime="text/html")
                except Exception:
                    pass
        else:
            st.write("Could not build 3D model.")
            if embed_err and debug_mode:
                st.write("3D build info:", embed_err)

        st.markdown("#### 2D Structure")
        img2d = render_2d_pil(smiles, size=(512, 512))
        if img2d is not None:
            st.image(img2d, width='content')
            if enable_2d_download:
                buf = io.BytesIO()
                img2d.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("Download 2D PNG", data=buf,
                                   file_name=f"{cid or 'compound'}_2d.png", mime="image/png")

        if connectivity_info:
            st.markdown("#### Connectivity Summary")
            st.write(f"- Atom stereo count: {connectivity_info.get('atom_stereo_count', 0)}")
            st.write(f"- Bond stereo count: {connectivity_info.get('bond_stereo_count', 0)}")
            st.write(f"- Atom count: {len(connectivity_info.get('atoms', []))}")
            st.write(f"- Bond count: {len(connectivity_info.get('bonds', []))}")
            if "atoms" in connectivity_info and show_atoms and isinstance(connectivity_info["atoms"], list):
                st.markdown("**Atom Details (2D coords)**")
                st.dataframe(pd.DataFrame(connectivity_info["atoms"]), width='stretch')
            if "bonds" in connectivity_info and show_bonds and isinstance(connectivity_info["bonds"], list):
                bonds_df = pd.DataFrame(connectivity_info["bonds"])
                if "order" in bonds_df.columns:
                    order_map = {1: "single", 2: "double", 3: "triple", 7: "aromatic"}
                    bonds_df["order_readable"] = bonds_df["order"].map(lambda x: order_map.get(x, str(x)))
                st.markdown("**Bond Details**")
                st.dataframe(bonds_df, width='stretch')

        if has_parent:
            st.caption("📍 Structure shown is the exact searched compound.")
        if grouped and any(grouped.get(g) for g in _groups_for("structure")):
            render_group_expanders(grouped, _groups_for("structure"), cid,
                                   key_prefix="structure")

    # ===== All data =====
    def _render_xref_block(ik, xr, syn, b_cid):
        if ik:
            st.markdown("#### InChIKey")
            st.code(ik, language="text")
        render_xrefs(xr)
        st.markdown("#### Synonyms (PubChem)")
        with st.expander(f"Synonyms — {len(syn)} found", expanded=False):
            if syn:
                st.write(syn)
                st.download_button(
                    "Download synonyms (JSON)",
                    data=json.dumps(syn, ensure_ascii=False, indent=2).encode("utf8"),
                    file_name=f"{b_cid or 'compound'}_synonyms.json",
                    mime="application/json", key=f"dl_syn_{b_cid or ik}")
            else:
                st.write("No synonyms found.")

    with tab_all:
        if has_parent:
            this_compound_caption(compound_name, cid)
        if canonical_ik and xrefs:
            canonical_form_callout(canonical_ik)
        _render_xref_block(inchikey, xrefs, synonyms, cid)

        st.markdown("#### Names, references & related records")
        if grouped:
            refs_groups = _groups_for("refs")
            # safety net: include any group not explicitly mapped to a tab
            unmapped = [g for g in grouped if g not in GROUP_TO_TAB and grouped.get(g)]
            render_group_expanders(grouped, refs_groups + unmapped, cid, key_prefix="refs")
        elif not has_parent:
            st.write("No flattened PUG-View summary available (CID missing or fetch failed).")

        if has_parent:
            st.markdown("---")
            parent_scope_callout(parent)
            _render_xref_block(parent["inchikey"], parent["xrefs"],
                               parent["synonyms"], parent["cid"])
            refs_groups = _groups_for("refs")
            unmapped = [g for g in parent["grouped"]
                        if g not in GROUP_TO_TAB and parent["grouped"].get(g)]
            if parent["grouped"] and any(parent["grouped"].get(g) for g in refs_groups + unmapped):
                st.markdown("#### Names, references & related records")
                render_group_expanders(parent["grouped"], refs_groups + unmapped,
                                       parent["cid"], key_prefix="refs_parent")

        with st.expander("Raw PubChemPy metadata (debug)", expanded=False):
            st.json(rawinfo or {})


def render_report_form(ctx: Dict[str, Any]):
    """'Report a data problem' — saved to a local log (data_reports.py) with
    everything needed to reproduce the lookup: query, CID, the FDA label used,
    market and cache version."""
    label = ctx.get("label") or {}
    with st.expander("🚩 Report incorrect, outdated or mismatched information"):
        with st.form(key=f"report_{_wkey(ctx.get('smiles') or ctx.get('input_text', ''))}",
                     clear_on_submit=True):
            kind = st.selectbox("What's wrong?", drp.PROBLEM_TYPES)
            detail = st.text_area("Details (what you expected, which tab/section)",
                                  max_chars=2000)
            sent = st.form_submit_button("Save report")
        if sent:
            drp.add_report({
                "problem": kind, "detail": detail.strip(),
                "query": ctx.get("input_text"), "compound": ctx.get("compound_name"),
                "cid": ctx.get("cid"), "region": ctx.get("region"),
                "label_set_id": label.get("spl_set_id", ""),
                "label_product": (label.get("brand_names") or [""])[0],
                "label_generic": (label.get("generic_names") or [""])[0],
                "cache_version": globals().get("_CACHE_VERSION", ""),
            })
            st.success("Thanks — report saved to `reports/data_reports.jsonl` on the "
                       "computer running the app. (On a hosted copy, e.g. Streamlit "
                       "Cloud, that file is cleared whenever the app restarts.)")
        n = len(drp.load_reports())
        if n:
            st.caption(f"{n} report{'s' if n != 1 else ''} saved so far.")


# ============================== Page sections ==============================
_MODE_ASK, _MODE_EXPLORE, _MODE_LIBRARY = "💬 Ask", "🔎 Explore", "💊 Drug Library"
_MODE_REFERENCE = "📘 Reference"


def render_sidebar(mode: str) -> Dict[str, Any]:
    """Sidebar shared by Ask and Explore (same widget keys, so settings carry
    over when switching). Returns the options both pages read."""
    opts: Dict[str, Any] = {"input_mode": "Name or SMILES"}
    with st.sidebar:
        if mode == _MODE_EXPLORE:
            st.markdown("### Input method")
            opts["input_mode"] = st.radio("Choose input method",
                                          options=["Name or SMILES", "Draw structure"],
                                          index=0, key="input_mode")
            st.markdown("---")
        region = st.radio(
            "Market (brands & availability)",
            options=["Singapore (HSA)", "United States (FDA)"],
            index=0, key="market_region",
            help="Singapore: local brands and Prescription-only / Pharmacy-only / "
                 "General-sale status from the HSA register. United States: "
                 "RxNorm brands and Rx/OTC from FDA labels. Clinical label text "
                 "(uses, dosing, warnings) is from the FDA label either way.")
        opts["region"] = "SG" if region.startswith("Singapore") and HAS_SG_REGISTER else "US"
        opts["strip_salts"] = st.checkbox(
            "Strip salts/counterions (use parent compound)", value=True, key="strip_salts",
            help="For salts like a hydrochloride, analyse the free base/acid — "
                 "its PubChem record and cross-database data are far richer.")
        with st.expander("Chemistry & display options"):
            opts["show_atoms"] = st.checkbox("Show atom table", value=False, key="opt_atoms")
            opts["show_bonds"] = st.checkbox("Show bond table", value=False, key="opt_bonds")
            opts["include_rdkit"] = st.checkbox("Include RDKit-calculated properties",
                                                value=True, key="opt_rdkit")
            opts["enable_2d_download"] = st.checkbox("Enable 2D PNG download", value=True,
                                                     key="opt_dl2d")
            opts["enable_3d_download"] = st.checkbox("Enable 3D HTML download", value=False,
                                                     key="opt_dl3d")
            opts["debug_mode"] = st.checkbox("Debug mode (show errors)", value=False,
                                             key="opt_debug")
            if st.button("🔄 Refresh cached data", key="clear_caches", width='stretch',
                         help="Fetched data is kept on disk so repeat questions are "
                              "instant, even after a restart. Clear it to re-download "
                              "everything fresh."):
                clear_data_caches()
                st.session_state["_chat_ctx_cache"] = {}
                st.toast("Cached data cleared — the next lookups fetch fresh data.")
        st.markdown("---")
        if mode == _MODE_ASK:
            st.button("🗑 Clear conversation", width='stretch', on_click=_chat_clear,
                      key="chat_clear")
            st.caption("Answers are matched from FDA labels, ChEMBL, PubChem, "
                       "Wikipedia and the HSA register — no AI model, nothing is "
                       "made up. Not medical advice.")
        elif st.session_state["history"]:
            st.markdown("### Recent")
            for h in reversed(st.session_state["history"][-8:]):
                st.button(h, key=f"hist_{h}", width='stretch',
                          on_click=_set_search, args=(h,))
    return opts


def render_explore(opts: Dict[str, Any]):
    """The original single-compound explorer: search box → full profile."""
    # --- top search bar ---
    search_col, btn_col = st.columns([6, 1])
    with search_col:
        st.text_input(
            "Drug or chemical name (generic or brand), or SMILES",
            key="chemical_search_input",
            placeholder="e.g. Lipitor, metformin, aspirin or CC(=O)Oc1ccccc1C(=O)O",
        )
    with btn_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🔍 Lookup", width='stretch'):
            st.session_state["lookup_pressed"] = True
            if st.session_state.get("input_mode") == "Draw structure" and \
                    st.session_state.get("chemical_search_input"):
                st.session_state["input_mode"] = "Name or SMILES"
            st.rerun()

    # --- example chips ---
    # pills wrap onto one or two lines on a phone (a row of columns would
    # stack into six full-width buttons)
    # selection_mode="multi" only so the empty state is [] rather than None
    # (AppTest can't replay a None pills value); a pick is consumed at once
    st.pills("Examples", EXAMPLE_QUERIES, key="example_pick", on_change=_pick_example,
             selection_mode="multi", label_visibility="collapsed")
    timer_slot = st.empty()   # filled with the fetch time once results render

    debug_mode = opts["debug_mode"]
    # --- Draw structure mode ---
    if opts["input_mode"] == "Draw structure":
        st.markdown("### Draw a molecule")
        if HAS_KETCHER:
            drawn = st_ketcher(st.session_state.get("draw_smiles", ""), height=500)
            if drawn:
                st.session_state["draw_smiles"] = drawn
                st.code(drawn, language="text")
                st.button("Use this structure ➜",
                          on_click=_set_search, args=(drawn,))
        elif show_jsme_drawer is not None:
            st.info("Draw below, copy the SMILES, then paste it into the search box above.")
            show_jsme_drawer(height=620)
        else:
            st.error("No drawing component available. Install `streamlit-ketcher`.")
        if not st.session_state.get("lookup_pressed"):
            st.stop()

    input_text = (st.session_state.get("chemical_search_input") or "").strip()
    if st.session_state.get("lookup_pressed"):
        st.session_state["lookup_pressed"] = False

    if not input_text:
        st.info("Type a drug name (generic or brand, e.g. 'Lipitor') or any chemical "
                "name/SMILES, or pick an example above.")
        render_drug_of_the_day()
        return
    t_start, req_start = time.perf_counter(), cs.REQUEST_COUNT

    # ---------- Resolve input to SMILES + CID ----------
    with st.spinner("Resolving name → structure (PubChem → OPSIN → CIR)…"):
        res = resolve_input(input_text, opts["region"])
    if res["exception"] is not None and debug_mode:
        st.exception(res["exception"])
    candidates = res["candidates"]
    if res["brand"] and candidates:
        ings = res["brand"]["ingredients"]
        ing_txt = " + ".join(ings)
        st.info(f"**{input_text}** is a brand name ({res['brand']['source']}) — active "
                f"ingredient{'s' if len(ings) > 1 else ''}: **{ing_txt}**."
                + ("  Pick one below to explore." if len(candidates) > 1 else ""))
    if not candidates:
        st.error(res["error"])
        return
    if len(candidates) > 1:
        def _fmt(i, c=candidates):
            cc = c[i]
            lbl = f"CID {cc['cid']}" if cc.get("cid") else "—"
            if cc.get("name"):
                lbl += f" · {cc['name']}"
            return f"{lbl} · {cc['smiles']}"
        st.warning(f"Multiple matches for '{input_text}'. Pick one:")
        idx = st.selectbox("Candidate", options=list(range(len(candidates))),
                           format_func=_fmt, index=0)
        chosen = candidates[int(idx)]
    else:
        chosen = candidates[0]

    ctx = load_compound(input_text, chosen, res["source"], region=opts["region"],
                        strip_salts=opts["strip_salts"], include_rdkit=opts["include_rdkit"])
    if debug_mode:
        for e in ctx.get("errors") or []:
            st.exception(e)
    if ctx.get("error"):
        st.error(ctx["error"])
        return
    if input_text not in st.session_state["history"]:
        st.session_state["history"].append(input_text)

    render_full_profile(ctx, show_atoms=opts["show_atoms"], show_bonds=opts["show_bonds"],
                        enable_2d_download=opts["enable_2d_download"],
                        enable_3d_download=opts["enable_3d_download"], debug_mode=debug_mode)
    render_report_form(ctx)

    elapsed = time.perf_counter() - t_start
    n_req = cs.REQUEST_COUNT - req_start
    timer_slot.caption(
        f"⏱ Result in **{elapsed:.1f} s** · "
        + (f"{n_req} live request{'s' if n_req != 1 else ''} to online databases"
           if n_req else "served from cache (no network calls)"))


# ============================== Ask (chat) mode ==============================
# A chat front-end over the same data. chat_router (rule-based — no AI model, no
# API) parses each question into (drug, intents, focus words); the matching
# slice of the record is assembled into "blocks" — plain data — that are stored
# in the chat history and replayed on every rerun, so old answers never
# re-fetch and every widget key is namespaced by its message.
_CHAT_MAX_MESSAGES = 60          # history kept (user + assistant messages)
_CHAT_CTX_CACHE = 4              # loaded compounds kept for instant follow-ups
_CHAT_3D_MAX_ATOMS = 150         # larger molecules skip the 3D embed (slow)

# Data each intent needs: 'struct' (no network beyond PubChem's structure),
# 'chem' (+ the PubChem record) or 'full' (every source). Unlisted → full.
_INTENT_LEVEL = {"structure": "struct", "class": "struct", "alternatives": "struct",
                 "identifiers": "chem", "properties": "chem", "spectra": "chem"}
_INTENT_ICON = {
    "overview": "💊", "mechanism": "🧬", "pk": "⏱", "uses": "🎯", "dosing": "📏",
    "side_effects": "⚠️", "warnings": "🚫", "interactions": "🔀", "pregnancy": "🤰",
    "population": "👪", "overdose": "☠️", "brands": "🏷", "class": "🗂",
    "alternatives": "🔁", "identifiers": "🆔", "structure": "🧪", "properties": "📊",
    "spectra": "📈", "history": "📜", "synthesis": "🏭", "legal": "⚖️", "storage": "📦",
    "cost": "💲", "ingredients": "🧾", "search": "🔎",
}
_CHAT_EXAMPLES = [
    "How does dextromethorphan stop me from coughing?",
    "Semaglutide pharmacodynamics",
    "Structure of codeine",
    "Side effects of metformin",
    "Can I take ibuprofen with warfarin?",
    "Half-life of diazepam",
    "Is paracetamol safe in pregnancy?",
    "What can I take for heartburn?",
]
# Chat styling: a readable column, an answer box, coloured source badges,
# small-caps section titles and chips. Colours are translucent so they work on
# both the light and dark Streamlit themes.
_CHAT_CSS = """
<style>
div[data-testid="stChatMessage"] { max-width: 1000px; margin-left: auto; margin-right: auto; }
div[data-testid="stChatMessage"] li { margin-bottom: .3rem; line-height: 1.55; }
.ans-box { border-left: 4px solid #ff4b4b; background: rgba(255,75,75,.07);
  padding: .65rem 1rem .7rem 1rem; border-radius: .5rem; margin: .15rem 0 .9rem 0;
  font-size: 1.04rem; line-height: 1.6; }
.ans-box p { margin: 0 0 .35rem 0; }
.ans-box p:last-child { margin-bottom: 0; }
.ans-label, .sec-title { font-size: .7rem; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; opacity: .62; }
.ans-label { margin-bottom: .3rem; }
.sec-title { margin: .95rem 0 .3rem 0; }
.src { display: inline-block; font-size: .66rem; line-height: 1.35; padding: 0 .45rem;
  margin-left: .35rem; border-radius: 999px; vertical-align: .1rem; white-space: nowrap;
  text-decoration: none !important; color: inherit !important; opacity: .85;
  border: 1px solid rgba(128,128,128,.35); }
.src-fda { background: rgba(59,130,246,.18); } .src-pc { background: rgba(34,197,94,.16); }
.src-wiki { background: rgba(148,163,184,.20); } .src-chembl { background: rgba(249,115,22,.18); }
.src-gtp { background: rgba(168,85,247,.18); } .src-calc { background: rgba(20,184,166,.18); }
.src-atc { background: rgba(234,179,8,.18); }
.chip { display: inline-block; padding: .18rem .7rem; margin: .15rem .35rem .2rem 0;
  border-radius: 999px; font-size: .85rem; background: rgba(128,128,128,.12);
  border: 1px solid rgba(128,128,128,.28); }
.chip small { opacity: .7; }
</style>
"""

_SRC_CLASS = (("FDA", "fda"), ("PubChem", "pc"), ("Wikipedia", "wiki"), ("ChEMBL", "chembl"),
              ("Guide to Pharmacology", "gtp"), ("RDKit", "calc"), ("inferred", "calc"),
              ("WHO ATC", "atc"))


def _chat_ask(question: str):
    """Button callback: queue a question for the next run."""
    st.session_state["chat_pending"] = question
    st.session_state["app_mode"] = _MODE_ASK


def _chat_clear():
    st.session_state["chat"] = []
    st.session_state["chat_context"] = {}
    st.session_state["_chat_ctx_cache"] = {}


# ---- answer blocks: plain data, rendered by render_blocks() ----
def _b(t: str, **kw) -> Dict[str, Any]:
    return {"t": t, **kw}


def _md_safe(s: str) -> str:
    """Source text → safe Markdown: strip/escape HTML and escape '$' (Streamlit
    would otherwise typeset '$5 … $10' as LaTeX)."""
    s = str(s or "")
    s = sanitize_html_fragment(s) if _BLEACH_AVAILABLE else _html.escape(s, quote=False)
    return s.replace("$", "\\$")


def _inline_html(s: str) -> str:
    """Safe text for INSIDE an HTML block (where Markdown isn't parsed): sanitise,
    then turn **bold** / *italic* into tags."""
    s = _md_safe(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)


def _badge(src: str, url: str = "") -> str:
    if not src:
        return ""
    cls = next((c for k, c in _SRC_CLASS if k in src), "other")
    label = _html.escape(src)
    if url:
        return (f" <a class='src src-{cls}' href='{_html.escape(url, quote=True)}' "
                f"target='_blank'>{label}</a>")
    return f" <span class='src src-{cls}'>{label}</span>"


def _tidy(text: str) -> str:
    """Display clean-up of quoted source text: HSDB '... ' elisions, label
    section numbers ('12.2 Pharmacodynamics …'), stray spaces."""
    t = re.sub(r"^\s*\.{3}\s*", "", str(text or ""))
    t = re.sub(r"\s+\.{3}\s+", " … ", t)
    t = re.sub(r"^\d{1,2}\.\d{1,2}\s+(?:Mechanism of Action|Pharmacodynamics|Pharmacokinetics)\s+",
               "", t)
    # a label sub-heading glued on: 'Hypertension Adult Patients: Administration…',
    # 'Hemodynamics Following administration…'
    t = re.sub(r"^(?:[A-Z][\w\-]*\s+){1,4}[A-Z][\w\-]*:\s+(?=[A-Z])", "", t)
    t = re.sub(r"^(?:Hemodynamics|Electrophysiolog\w+(?: Effects)?|Cardiac Electrophysiology|"
               r"Effects on [A-Z]\w+(?: \w+)?)\s+(?=[A-Z])", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _points_md(items: List[Tuple[str, str, str]]) -> str:
    return "\n".join(f"- {_md_safe(_tidy(text))}{_badge(src, url)}" for text, src, url in items)


def _pts(items, title: str = "", cap: int = 6, more: str = "Show more") -> List[Dict[str, Any]]:
    """A points block (or nothing when there are no items)."""
    items = [(t, s, u) for t, s, u in items if t and str(t).strip()]
    if not items:
        return []
    return [_b("points", items=items, title=title, cap=cap, more=more)]


def _answer_box(items: List[Tuple[str, str, str]], label: str = "In short") -> List[Dict[str, Any]]:
    """The direct answer, highlighted at the top of a section."""
    items = [it for it in items if it and it[0]]
    return [_b("answer", items=items, label=label)] if items else []


def _sec(title: str) -> Dict[str, Any]:
    return _b("sec", v=title)


def render_blocks(blocks: List[Dict[str, Any]], key: str, latest: bool = True):
    """Replay an answer's blocks. Interactive blocks (follow-up buttons) only
    render on the latest answer, so old answers stay quiet."""
    for j, b in enumerate(blocks):
        t, k = b.get("t"), f"{key}_{j}"
        try:
            if t == "md":
                st.markdown(b["v"], unsafe_allow_html=b.get("html", False))
            elif t == "caption":
                st.caption(b["v"])
            elif t in ("info", "warning", "error", "success"):
                getattr(st, t)(b["v"])
            elif t == "answer":
                body = "".join(f"<p>{_inline_html(_tidy(txt))}{_badge(src, url)}</p>"
                               for txt, src, url in b["items"])
                st.markdown(f"<div class='ans-box'><div class='ans-label'>"
                            f"{_html.escape(b.get('label') or 'In short')}</div>{body}</div>",
                            unsafe_allow_html=True)
            elif t == "sec":
                st.markdown(f"<div class='sec-title'>{_html.escape(b['v'])}</div>",
                            unsafe_allow_html=True)
            elif t == "chips":
                if b.get("title"):
                    st.markdown(f"<div class='sec-title'>{_html.escape(b['title'])}</div>",
                                unsafe_allow_html=True)
                chips = "".join(
                    f"<span class='chip'>{_inline_html(lbl)}"
                    + (f" <small>{_inline_html(sub)}</small>" if sub else "") + "</span>"
                    for lbl, sub in b["items"])
                st.markdown(chips + (_badge(b.get("src", ""), b.get("url", ""))),
                            unsafe_allow_html=True)
            elif t == "points":
                items, cap = b["items"], b.get("cap", 6)
                if b.get("title") and cap:
                    st.markdown(f"<div class='sec-title'>{_html.escape(b['title'])}</div>",
                                unsafe_allow_html=True)
                if cap:
                    st.markdown(_points_md(items[:cap]), unsafe_allow_html=True)
                if len(items) > cap:
                    label = f"{b.get('more') or 'Show more'} ({len(items) - cap})"
                    with st.expander(label):
                        st.markdown(_points_md(items[cap:]), unsafe_allow_html=True)
            elif t == "cols":
                c1, c2 = st.columns(b.get("spec", [3, 2]))
                with c1:
                    render_blocks(b["left"], f"{k}l", latest)
                with c2:
                    render_blocks(b["right"], f"{k}r", latest)
            elif t == "image":
                st.image(b["v"], caption=b.get("caption"), width=b.get("width", 360))
            elif t == "code":
                st.code(b["v"], language="text")
            elif t == "table":
                st.dataframe(pd.DataFrame(b["rows"]), hide_index=True, width='stretch')
            elif t == "faers":
                render_faers_chart(b["v"])
            elif t == "html":
                components.html(b["v"], height=b.get("height", 420))
            elif t == "buttons" and latest and b.get("items"):
                _render_button_row(b, k)
            elif t == "profile" and latest and b.get("query"):
                st.button(f"📖 Open the full {b.get('name') or ''} profile".replace("  ", " "),
                          key=k, on_click=_open_drug, args=(b["query"],),
                          help="Every tab — label sections, pharmacology, chemistry, "
                               "safety and references — in the Explore view")
        except Exception as e:  # noqa: BLE001 — one bad block mustn't sink the chat
            st.caption(f"(couldn't display part of this answer: {e})")


_PROFILE_PREFIX = "__profile__:"


def _render_button_row(b: Dict[str, Any], k: str):
    """Follow-up suggestions as one compact row of small buttons (wrapping as
    needed); an item whose question starts with _PROFILE_PREFIX opens the
    Explore profile instead of asking."""
    if b.get("title"):
        st.markdown(f"<div class='sec-title'>{_html.escape(b['title'])}</div>",
                    unsafe_allow_html=True)
    try:
        row = st.container(horizontal=True, gap="small")
    except TypeError:                     # older Streamlit: no horizontal containers
        row = None
    cols = None if row is not None else st.columns(min(len(b["items"]), 5))
    for n, (label, question) in enumerate(b["items"]):
        target = row if row is not None else cols[n % len(cols)]
        if question.startswith(_PROFILE_PREFIX):
            target.button(label, key=f"{k}_{n}", on_click=_open_drug,
                          args=(question[len(_PROFILE_PREFIX):],),
                          width="content" if row is not None else "stretch",
                          help="Every tab — label sections, pharmacology, chemistry, safety "
                               "and references — in the Explore view")
        else:
            target.button(label, key=f"{k}_{n}", on_click=_chat_ask, args=(question,),
                          width="content" if row is not None else "stretch",
                          help=question)


# ---- the record as a searchable list of sentences ----
# (label field, section title, topic) — topics match chat_router intents.
_LABEL_TOPICS = [
    ("boxed_warning", "Boxed warning", "warnings"),
    ("indications", "Indications", "uses"), ("purpose", "Purpose", "uses"),
    ("dosage", "Dosage & administration", "dosing"),
    ("dosage_forms", "Dosage forms", "dosing"),
    ("contraindications", "Contraindications", "warnings"),
    ("warnings", "Warnings & precautions", "warnings"),
    ("do_not_use", "Do not use", "warnings"), ("ask_doctor", "Ask a doctor", "warnings"),
    ("stop_use", "Stop use", "warnings"), ("when_using", "When using", "warnings"),
    ("adverse_reactions", "Adverse reactions", "side_effects"),
    ("drug_interactions", "Drug interactions", "interactions"),
    ("specific_populations", "Specific populations", "pregnancy"),
    ("pregnancy", "Pregnancy", "pregnancy"), ("lactation", "Lactation", "pregnancy"),
    ("pediatric_use", "Pediatric use", "children"),
    ("geriatric_use", "Geriatric use", "elderly"),
    ("overdosage", "Overdosage", "overdose"),
    ("mechanism_of_action", "Mechanism of action", "mechanism"),
    ("pharmacodynamics", "Pharmacodynamics", "mechanism"),
    ("pharmacokinetics", "Pharmacokinetics", "pk"),
    ("abuse", "Abuse & dependence", "legal"),
    ("controlled_substance", "Controlled substance", "legal"),
    ("storage", "Storage", "storage"),
]
# Medication Guide headings are patient questions; split at them.
_MEDGUIDE_HEAD_RE = re.compile(
    r"(?:^|(?<=\s))((?:What|How|Who|Should|Can|Do|Does|Is|Are|Where|When|Before)\b"
    r"[^?•.!]{3,150}\?)")      # no full stop inside: a heading is one question
# heading / subsection words → topic (first match wins)
_PATIENT_TOPICS = [
    ("ingredients", re.compile(r"ingredient", re.I)),
    ("storage", re.compile(r"\bstor(?:e|age)|keep .{0,30}out of", re.I)),
    ("side_effects", re.compile(r"side effects?|adverse", re.I)),
    ("interactions", re.compile(r"\bavoid\b|interact|other medicines|alcohol|grapefruit", re.I)),
    ("dosing", re.compile(r"how (?:should|do|to) (?:i |you )?(?:take|use|give)|missed? dose|"
                          r"dosing|administ|how much", re.I)),
    ("pregnancy", re.compile(r"pregnan|breast|nursing|lactat", re.I)),
    ("overdose", re.compile(r"overdos|too much", re.I)),
    ("legal", re.compile(r"abuse|dependen|addict|withdraw", re.I)),
    ("uses", re.compile(r"^what is \S+(?: \S+){0,4}\?$|used (?:for|to)", re.I)),
    ("warnings", re.compile(r"important information|should not|do not|before (?:taking|using|you)|"
                            r"tell your|warning|precaution|driv|machiner", re.I)),
]


def _patient_topic(heading: str) -> Optional[str]:
    for topic, rx in _PATIENT_TOPICS:
        if rx.search(heading or ""):
            return topic
    return None


def medguide_sections(text: str) -> List[Tuple[str, str, str]]:
    """A Medication Guide → [(question heading, body, topic)]. 'General
    information about the safe use…' boilerplate is dropped."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) < 80:
        return []
    marks = [(m.start(1), m.group(1)) for m in _MEDGUIDE_HEAD_RE.finditer(text)]
    out = []
    for i, (pos, head) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos + len(head):end].strip(" •")
        if len(body) < 8 or re.search(r"general information about", head, re.I):
            continue
        out.append((head, body, _patient_topic(head) or "warnings"))
    return out


# Heading text → topic (first match wins), then the group's default topic.
_HEADING_TOPICS = [
    ("history", re.compile(r"histor|discover|etymolog", re.I)),
    ("synthesis", re.compile(r"synthes|manufactur|production|preparation|biosynth|"
                             r"methods? of", re.I)),
    ("legal", re.compile(r"legal|regulat|controlled|recreational|abuse|dependen|society|"
                         r"schedul|doping|misuse", re.I)),
    ("storage", re.compile(r"storage|shelf", re.I)),
    ("interactions", re.compile(r"interaction", re.I)),
    ("pregnancy", re.compile(r"pregnan|lactat|breast", re.I)),
    ("side_effects", re.compile(r"adverse|side[- ]effect|hepatotox|effects? on", re.I)),
    ("overdose", re.compile(r"overdos|toxicity|ld50|lethal|antidote|poison|toxic", re.I)),
    ("warnings", re.compile(r"warning|precaution|contraindic|hazard", re.I)),
    ("pk", re.compile(r"pharmacokinetic|absorption|metabol|excretion|half|distribution|"
                      r"elimination", re.I)),
    ("mechanism", re.compile(r"mechanism|pharmacodynamic|mode of action|pharmacolog|"
                             r"function", re.I)),
    ("uses", re.compile(r"\buses?\b|indication|therapeutic|medical|clinical|treatment|"
                        r"dosage|administration", re.I)),
    ("properties", re.compile(r"point|solubility|density|pressure|logp|pka|dissociation|"
                              r"physical|colou?r|odou?r|refractive|viscosity|taste|propert|"
                              r"stability", re.I)),
]
_GROUP_TOPIC = {
    "Pharmacology & Clinical": "pharma", "Interactions, Pathways & Bioactivity": "pharma",
    "Toxicity & Health": "toxicity", "Toxicology Programs & Studies": "toxicity",
    "Safety & Hazards": "hazard", "Exposure & Occupational": "hazard",
    "Environmental & Fate": "hazard", "Manufacturing, Uses & Commerce": "uses",
    "Regulatory & Transport": "legal", "Experimental Properties & Measurements": "properties",
    "Physical Properties": "properties",
}


def _strip_depth(heading: str) -> str:
    """'› › Pharmacodynamics' → 'Pharmacodynamics' (merged Wikipedia sub-sections)."""
    return re.sub(r"^(?:›\s*)+", "", heading or "").strip()


def _heading_topic(group: str, heading: str) -> str:
    if heading == _RESEARCH_HEADING:
        return "research"
    if heading == _EFFICACY_HEADING:
        return "uses"
    for topic, rx in _HEADING_TOPICS:
        if rx.search(heading):
            return topic
    return _GROUP_TOPIC.get(group, "other")


def _wiki_section_topic(sec: Dict[str, Any]) -> str:
    """Chat topic of a Wikipedia section: its own heading, else the nearest
    parent heading that names a topic."""
    for h in [sec.get("heading") or ""] + list(reversed(sec.get("parents") or [])):
        for topic, rx in _HEADING_TOPICS:
            if rx.search(h):
                return topic
    return "other"


def _value_items(heading: str, val: Any) -> List[Tuple[str, str]]:
    """[(sentence, source)] for any heading value — PubChem text, a Wikipedia
    section or a merged heading — de-duplicated and on-topic first."""
    if isinstance(val, dict) and val.get("__combined__"):
        return _combined_items(heading, val)
    pub, bullets, url = _split_value_parts(val)
    if not pub and not bullets:
        return []
    return _combined_items(heading, {"pubchem": pub, "bullets": bullets, "url": url})


def _ctx_grouped_list(ctx: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str]]:
    """[(grouped, scope)] — the compound's own record, then its parent's."""
    out = [(ctx.get("grouped") or {}, "")]
    if ctx.get("has_parent") and ctx.get("parent"):
        out.append((ctx["parent"].get("grouped") or {}, "parent"))
    return out


def _src_url(ctx: Dict[str, Any], src: str, wiki_url: str = "") -> str:
    if src == "Wikipedia":
        return wiki_url or (ctx.get("wiki") or {}).get("url", "")
    if src == "PubChem" and ctx.get("cid"):
        return f"https://pubchem.ncbi.nlm.nih.gov/compound/{ctx['cid']}"
    if src == "FDA label":
        return (ctx.get("label") or {}).get("url", "")
    if src == "ChEMBL":
        return (ctx.get("clin_chembl") or {}).get("url", "")
    return ""


_SKIP_CORPUS_GROUPS = {"Names & Identifiers", "Literature & References",
                       "Related Records & Vendors", "Patents", "Spectra & Analytical",
                       "Structures & Conformers", "Classification, Standards & Registries",
                       "Computed Properties / Descriptors", "Other"}


def _chat_corpus(ctx: Dict[str, Any]) -> List[Dict[str, str]]:
    """Every sentence the chat can quote for this compound, tagged with source,
    link and topic — built once per loaded compound and cached on the ctx."""
    if ctx.get("_corpus") is not None:
        return ctx["_corpus"]
    docs: List[Dict[str, str]] = []
    of = ctx.get("label") or {}
    for field, section, topic in _LABEL_TOPICS:
        text = label_field(of, field) if field in _LABEL_TOPIC_RE else (of.get(field) or "")
        if len(text) < 20:
            continue
        for title, body in _label_chunks(text):
            for s in _split_sentences(body):
                if len(s) >= 25:
                    docs.append({"text": s, "src": "FDA label", "url": of.get("url", ""),
                                 "topic": topic, "section": f"{section}" + (f" · {title}" if title else "")})
    # plain-language patient text (Medication Guide Q&A, counseling) — what
    # practical questions (missed dose, crushing, alcohol, driving) need
    for heading, body, topic in medguide_sections(of.get("medguide") or ""):
        for s in _split_sentences(body):
            if len(s) >= 20:
                docs.append({"text": s, "src": "FDA label", "url": of.get("url", ""),
                             "topic": topic, "section": f"Medication Guide · {heading}"})
    for title, body in _label_chunks(of.get("patient_info") or ""):
        topic = _patient_topic(title or body[:80]) or "warnings"
        for s in _split_sentences(body):
            if len(s) >= 25:
                docs.append({"text": s, "src": "FDA label", "url": of.get("url", ""),
                             "topic": topic, "section": "Patient counseling" + (f" · {title}" if title else "")})
    if len(of.get("inactive_ingredients") or "") > 3:
        docs.append({"text": "Inactive ingredients: " + of["inactive_ingredients"].strip(),
                     "src": "FDA label", "url": of.get("url", ""), "topic": "ingredients",
                     "section": "Inactive ingredients"})
    lead = (ctx.get("wiki") or {}).get("extract") or ""
    for s in _split_sentences(lead):
        docs.append({"text": s, "src": "Wikipedia", "url": (ctx.get("wiki") or {}).get("url", ""),
                     "topic": "overview", "section": "Summary"})
    # the record description joins several sources' paragraphs with ' | '
    rec = [s for part in re.split(r"\s*\|\s*", ctx.get("record_text") or "")
           for s in _split_sentences(part)]
    for s in rec[:15]:
        docs.append({"text": _clean_record_sentence(s), "src": "PubChem",
                     "url": _src_url(ctx, "PubChem"), "topic": "overview", "section": "Description"})
    ch = ctx.get("clin_chembl") or {}
    for me in ch.get("mechanisms") or []:
        if me.get("mechanism") and me["mechanism"].lower() != "unknown":
            tgt = f" (target: {me['target_name']})" if me.get("target_name") else ""
            docs.append({"text": f"{me['mechanism']}{tgt}.", "src": "ChEMBL", "url": ch.get("url", ""),
                         "topic": "mechanism", "section": "Mechanism"})
    # Wikipedia straight from its sections (not the de-duplicated copy merged
    # into the PubChem groups), topic taken from the heading or, for generic
    # sub-headings, the nearest parent ('Pharmacodynamics' › 'Monoamine
    # releasing agent' is mechanism).
    wsecs = (ctx.get("wiki_secs") or {}).get("sections") or []
    wurl0 = (ctx.get("wiki_secs") or {}).get("url") or (ctx.get("wiki") or {}).get("url", "")
    for sec in wsecs:
        topic = _wiki_section_topic(sec)
        path = " › ".join(list(sec.get("parents") or []) + [sec["heading"]])
        for line in (sec.get("text") or "").split("\n"):
            for s in _split_sentences(line):
                if len(s) >= 25 and not _is_lowvalue_sentence(s):
                    docs.append({"text": s, "src": "Wikipedia", "url": wurl0, "topic": topic,
                                 "section": path})
    for grouped, scope in _ctx_grouped_list(ctx):
        for group, items in grouped.items():
            if group in _SKIP_CORPUS_GROUPS or not items:
                continue
            for heading, val in items.items():
                if heading == "IDs & Classifications" or _is_spectral_heading(heading.lower()):
                    continue
                topic = _heading_topic(group, heading)
                clean_head = _strip_depth(heading)
                if topic == "properties" and isinstance(val, str):
                    # short measured values: keep them whole, labelled by heading
                    for part in [p.strip() for p in val.split(" | ") if p.strip()][:6]:
                        docs.append({"text": f"{clean_head}: {part[:400]}", "src": "PubChem",
                                     "url": _src_url(ctx, "PubChem"), "topic": topic,
                                     "section": clean_head, "scope": scope})
                    continue
                wurl = val.get("url", "") if isinstance(val, dict) else ""
                for s, src in _value_items(heading, val):
                    if src == "Wikipedia" and wsecs and not scope:
                        continue                     # already added from the sections
                    for piece in re.split(r"\s+\|\s+", s):   # stray multi-entry joins
                        if len(piece) >= 15:
                            docs.append({"text": piece, "src": src,
                                         "url": _src_url(ctx, src, wurl), "topic": topic,
                                         "section": clean_head, "scope": scope})
    ctx["_corpus"] = docs
    return docs


def _relevant(ctx: Dict[str, Any], terms: List[str], topics=None, prefer=None,
              k: int = 4) -> List[Tuple[str, str, str]]:
    """The k sentences that best match `terms` (BM25 over light stems, lay words
    expanded to their medical synonyms), restricted to `topics` when given."""
    if not terms or cr is None:
        return []
    weights = cr.expand_terms(terms)
    if not weights:
        return []
    docs = [d for d in _chat_corpus(ctx) if topics is None or d["topic"] in topics]
    ranked = cr.rank([d["text"] for d in docs], weights)
    if not ranked:
        return []
    # A multi-word focus ('weight gain') should match as a phrase: once some
    # sentence covers ≥2 of the terms, drop sentences that hit only one
    # ('…small molecular weight…').
    n_terms = {i: len(cr.matched_terms(docs[i]["text"], weights)) for i, _sc in ranked}
    if len({st for f in terms for st in cr.tokens(f)}) >= 2 and max(n_terms.values()) >= 2:
        ranked = [(i, sc) for i, sc in ranked if n_terms[i] >= 2]
    scored = []
    for i, sc in ranked:
        d = docs[i]
        boost = 1.3 if prefer and d["topic"] in prefer else 1.0
        if d["topic"] == "research":
            boost *= 0.6
        if len(d["text"]) > 450:
            boost *= 0.8
        scored.append((sc * boost, d))
    scored.sort(key=lambda x: -x[0])
    top = scored[0][0]
    out, dd = [], _SentenceDeduper()
    for sc, d in scored:
        if sc < max(0.6, 0.35 * top) or len(out) >= k:
            break
        ws = _content_words(d["text"])
        if dd.is_dup(ws):
            continue
        dd.accept(ws)
        out.append((d["text"], d["src"], d["url"]))
    return out


# GHS precautionary-code lists ('P203, P210, P233, …') — codes, not sentences.
_CODE_LIST_RE = re.compile(r"(?:\bP\d{3}\b.*){5,}|click each P-code", re.S)


def _topic_items(ctx: Dict[str, Any], topics, limit: int = 40) -> List[Tuple[str, str, str]]:
    """Sentences of the given topics in record order (no ranking)."""
    out, dd = [], _SentenceDeduper()
    for d in _chat_corpus(ctx):
        if d["topic"] not in topics or _CODE_LIST_RE.search(d["text"]):
            continue
        ws = _content_words(d["text"])
        if dd.is_dup(ws):
            continue
        dd.accept(ws)
        out.append((d["text"], d["src"], d["url"]))
        if len(out) >= limit:
            break
    return out


def _mentions(ctx: Dict[str, Any], patterns: List[str], topics=None,
              k: int = 6) -> List[Tuple[str, str, str]]:
    """Sentences that literally mention any of `patterns` (whole words), label
    interaction/contraindication text first."""
    pats = [p for p in dict.fromkeys(p.strip() for p in patterns) if p and len(p) >= 3]
    if not pats:
        return []
    rx = re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(p) for p in pats) + r")(?![a-z])", re.I)
    order = {"interactions": 0, "warnings": 1, "side_effects": 2}
    hits = [d for d in _chat_corpus(ctx) if (topics is None or d["topic"] in topics)
            and rx.search(d["text"])]
    hits.sort(key=lambda d: (d["src"] != "FDA label", order.get(d["topic"], 3)))
    out, dd = [], _SentenceDeduper()
    for d in hits:
        ws = _content_words(d["text"])
        if dd.is_dup(ws):
            continue
        dd.accept(ws)
        out.append((d["text"], d["src"], d["url"]))
        if len(out) >= k:
            break
    return out


# Label sentences that are true of every label and say nothing about the drug.
_LABEL_BOILERPLATE_RE = re.compile(
    r"because (?:clinical trials|these reactions) are|"
    r"(?:the following|these) (?:serious |clinically significant )?adverse reactions are "
    r"(?:also )?(?:discussed|described)|"
    r"(?:is|are) (?:discussed|described) (?:in (?:more )?detail )?(?:elsewhere|in other sections)|"
    r"cannot be directly compared to rates|it is not always possible to reliably estimate|"
    r"^see (?:full prescribing information|section)|^\(?see\b", re.I)


def _label_points(of: Dict[str, Any], field: str, n_sent: int = 2,
                  only: Optional[re.Pattern] = None) -> List[Tuple[str, str, str]]:
    """An FDA label section as points: each numbered subsection's title + its
    opening (non-boilerplate) sentences, or one point per sentence for an
    unstructured section. `only` keeps subsections whose TITLE matches (e.g.
    '8.2 Lactation'), else just the sentences that match."""
    text = label_field(of, field) if field in _LABEL_TOPIC_RE else ((of or {}).get(field) or "")
    if len(text) < 20:
        return []
    url = (of or {}).get("url", "")
    chunks = _label_chunks(text)

    def _good(s):
        return len(s) >= 20 and not _LABEL_BOILERPLATE_RE.search(s)

    if any(t for t, _b in chunks):
        if only is not None:
            titled = [(t, b) for t, b in chunks if t and only.search(t)]
            if titled:
                chunks = titled
            else:   # no subsection is about it — fall back to matching sentences
                return [(s, "FDA label", url) for _t, b in chunks for s in _split_sentences(b)
                        if _good(s) and only.search(s)][:8]
        out = []
        for title, body in chunks:
            sents = [s for s in _split_sentences(body) if _good(s)]
            lead = " ".join(sents[:n_sent]).strip()
            title = (title or "").strip(" :—-")
            if lead:
                out.append((f"**{title}** — {lead}" if title else lead, "FDA label", url))
        return out
    return [(s, "FDA label", url) for _t, b in chunks for s in _split_sentences(b)
            if _good(s) and (only is None or only.search(s))]


def _find_heading(ctx: Dict[str, Any], want: str):
    """(heading, value, scope) of the pharmacology heading for a canonical topic
    ('mechanism' | 'pk' | 'uses'), primary record first, then the parent's."""
    for grouped, scope in _ctx_grouped_list(ctx):
        heading, val = _topic_heading(grouped, want)
        if heading is not None:
            return heading, val, scope
    return None, None, ""


_TOPIC_CANON = {"mechanism": "Mechanism of Action & Pharmacodynamics",
                "pk": "Pharmacokinetics (ADME)", "uses": "Uses & Indications"}


def _topic_heading(grouped: Dict[str, Any], want: str):
    """(heading, value) of one bundle's pharmacology heading for a topic."""
    canon = _TOPIC_CANON[want]
    for heading, val in (grouped.get(_REFINE_GROUP) or {}).items():
        if isinstance(val, dict) and val.get(want if want != "uses" else "__none__"):
            return heading, val
        if _canonical_topic(heading) == canon or heading == canon:
            return heading, val
    return None, None


def _points_for(ctx: Dict[str, Any], heading: str, val: Any, scope: str
                ) -> List[Tuple[str, str, str]]:
    """A heading's sentences as (text, source, link). Sentences from the
    parent's record link to the PARENT's PubChem page / article."""
    view = ctx
    if scope == "parent" and ctx.get("parent"):
        par = ctx["parent"]
        view = dict(ctx, cid=par.get("cid"), wiki=par.get("wiki") or ctx.get("wiki"))
    wurl = val.get("url", "") if isinstance(val, dict) else ""
    return [(s, src, _src_url(view, src, wurl)) for s, src in _value_items(heading, val)]


def _heading_points(ctx, want: str) -> Tuple[List[Tuple[str, str, str]], str]:
    heading, val, scope = _find_heading(ctx, want)
    if heading is None:
        return [], ""
    return _points_for(ctx, heading, val, scope), scope


def _dedup_against(items, seen_texts) -> List[Tuple[str, str, str]]:
    dd = _SentenceDeduper()
    for t in seen_texts:
        dd.accept(_content_words(t))
    out = []
    for it in items:
        ws = _content_words(it[0])
        if dd.is_dup(ws) or dd.is_restatement(ws):
            continue
        dd.accept(ws)
        out.append(it)
    return out


def _focus_label(p: Dict[str, Any]) -> str:
    return " ".join(p.get("focus") or [])


def _is_med(ctx) -> bool:
    return bool(ctx.get("is_drug") or ctx.get("label") or ctx.get("drug_hit"))


def _no_label_note(ctx, name: str, what: str) -> List[Dict[str, Any]]:
    """Why clinical detail is missing, and where to look instead."""
    if ctx.get("label") or not _is_med(ctx):
        return []
    sg = (ctx.get("market") or {}).get("sg")
    where = (f"the Singapore package insert on [HSA Infosearch]({HSA_INFOSEARCH_URL})"
             if ctx.get("region") == "SG" and sg else "the local package insert")
    return [_b("info", v=f"There's no US FDA label for {name}, so official {what} "
                          f"information isn't available here — see {where}.")]


# ---- one builder per intent: (ctx, name, parsed) -> blocks ----
def _ctx_glance(ctx) -> Dict[str, Any]:
    """glance_facts for a loaded context (the At-a-Glance card's data)."""
    return glance_facts(compound_name=ctx["compound_name"], formula=ctx["formula"],
                        dm=derived_metrics(ctx["smiles"]) if ctx["smiles"] else {},
                        chembl=ctx["clin_chembl"], wikidata=ctx["wikidata"], wiki=ctx["wiki"],
                        record_text=ctx["record_text"], drug=ctx["drug_hit"],
                        openfda=ctx["openfda"], adverse_events=ctx["adverse_events"],
                        market=ctx["market"], region=ctx["region"],
                        record_use=record_indication_sentence(ctx["grouped"]))


def _ans_overview(ctx, name, p):
    f = _ctx_glance(ctx)
    left: List[Dict[str, Any]] = []
    extract = (ctx.get("wiki") or {}).get("extract") or ""
    what = " ".join(_split_sentences(extract)[:2]) if extract else f["what"]
    src, url = ("Wikipedia", (ctx.get("wiki") or {}).get("url", "")) if extract else ("", "")
    if not what and ctx.get("drug_hit"):
        what, src, url = ctx["drug_hit"].get("blurb") or "", "Drug Library", ""
    left += _answer_box([(what[:600], src, url)], label="In short")
    bits = []
    if f["is_med"]:
        for label, key in ((f.get("brand_label", "Brand names"), "brands"), ("Drug class", "classes"),
                           ("Used for", "use"), ("How it works", "mech"),
                           ("Common side effects", "common"), ("Availability", "avail"),
                           ("Status", "status")):
            val = f.get(key)
            if isinstance(val, list):
                val = (", " if key == "brands" else " · ").join(val)
            if val:
                bits.append(f"- **{label}:** {_md_safe(val)}{_badge(f['src'].get(key, ''))}")
            if key == "common" and not val and f.get("faers_terms"):
                bits.append(f"- **Most-reported side effects:** {_md_safe(f['faers_terms'])} "
                            "<small>(FDA FAERS reports, not incidence)</small>")
    else:
        dm = f.get("dm") or {}
        if ctx["formula"]:
            bits.append(f"- **Formula:** {ctx['formula']}")
        if dm:
            bits.append(f"- **Mol. weight:** {dm['Molecular weight'][0]} Da")
        bits.append("- **Type:** not a recognised medicine — chemical / research compound")
    if bits:
        left.append(_b("md", v="\n".join(bits), html=True))
    right = []
    if ctx["smiles"]:
        png = render_2d_highlight(ctx["smiles"], [], size=(300, 240))
        if png:
            right.append(_b("image", v=png, width=240, caption=ctx["formula"] or None))
    blocks = [_b("cols", left=left, right=right, spec=[3, 1.3])] if right else left
    if f.get("boxed"):
        blocks.append(_b("error", v="**⚠️ FDA Boxed Warning** — " + _md_safe(boxed_summary(f["boxed"], 300))))
    blocks += _no_label_note(ctx, name, "dosing and safety")
    return blocks


# --- "How it works" / pharmacodynamics, answer-first ------------------------
# Every mechanism sentence from every source is scored for being a direct
# answer: it should say what the drug DOES (an action verb) to WHAT (a target /
# messenger) and, ideally, with what RESULT — and not be a hedge, a lab finding
# or a fragment. The best one or two become the "In short" box; sentences about
# effects on the body (appetite, blood pressure, CNS stimulation…) form their own
# section, because that's what pharmacodynamics also means; the rest is detail.
_MECH_ACT_RE = re.compile(
    r"\b(?:releas\w*|reuptake|inhibit\w*|block\w*|bind\w*|agonis\w*|antagonis\w*|activat\w*|"
    r"modulat\w*|stimulat(?:es|ed|ing|ion)?\b|suppress(?:es|ed|ing|ion)?\b|potentiat\w*|"
    r"enhanc\w*|increas\w* (?:the )?(?:release|"
    r"level|concentration|activity|amount)|decreas\w* (?:the )?(?:release|level|production|"
    r"synthesis)|prevent\w* the|interfer\w*|acts? (?:as|on|by|at)|works? by|mimic\w*|depolari\w*|"
    r"hyperpolari\w*|opens?|closes?)\b", re.I)
_MECH_TGT_RE = re.compile(
    r"\b(?:receptors?|transporters?|enzymes?|channels?|norepinephrine|noradrenaline|dopamine|"
    r"serotonin|5-HT\w*|GABA\w*|glutamate|NMDA|histamine|acetylcholine|muscarinic|nicotinic|"
    r"opioid|adrenergic|cyclooxygenase|COX-?\d?|prostaglandin\w*|kinases?|synthase|reductase|"
    r"calcium|sodium|potassium|chloride|insulin|glucagon|GLP-1|incretin|hormones?|"
    r"neurotransmitters?|vesic\w*|TAAR1|sigma|monoamines?|catecholamines?|cell wall|DNA|RNA|"
    r"ribosom\w*|proton pump|H\+/K\+|thrombin|factor X\w*|vitamin K|platelets?)\b", re.I)
_MECH_OUT_RE = re.compile(
    r"\b(?:thereby|resulting in|results in|leading to|leads to|which (?:in turn )?\w+s\b|causing|"
    r"so that|producing|produces?|to (?:reduce|decrease|increase|lower|raise|produce|suppress|"
    r"relieve|prevent|block)|in order to)\b|,\s*(?:increasing|decreasing|reducing|lowering|"
    r"raising|preventing|suppressing|relieving)\b", re.I)
# 'It …' / 'This agent …' at the start refers to the drug itself.
_DRUG_ANAPHOR_RE = re.compile(r"^(?:it|this (?:drug|agent|medication|medicine|compound)|"
                              r"the drug)\b(?!\s+(?:does this|also|then|in turn|is also))", re.I)
# 'Sertraline also shows…' — a secondary action, not the main one.
_SECONDARY_RE = re.compile(r"^.{0,60}?\b(?:also|additionally|furthermore)\b|"
                           r"\b(?:secondary|weak(?:er|ly)?|minor|modest) (?:\w+ )?"
                           r"(?:inhibit\w*|activity|affinity|effects?)\b|"
                           r"\b(?:displays?|shows?|has|have) (?:\w+ )?affinity\b|"
                           r"^it does this\b", re.I)
_MECH_HEDGE_RE = re.compile(
    r"has not been (?:established|determined|elucidated|shown)|not (?:fully |completely |well[- ]?|"
    r"yet |entirely )?(?:known|understood|established|elucidated|clear)|"
    r"(?:is|are) (?:unknown|unclear|not known)|"
    r"remains? (?:unknown|unclear|uncertain)|some reports|may also be involved|is not known|"
    r"poorly understood|other mechanisms should|(?:has|have) been (?:proposed|suggested|"
    r"hypothesi[sz]ed)|it is (?:reported|suggested|thought) that", re.I)
_EFFECT_RE = re.compile(
    r"\b(?:appetite|hunger|satiety|weight (?:loss|gain|reduction)|food intake|blood pressure|"
    r"heart rate|tachycardia|bradycardia|hypertension|central nervous system stimulation|"
    r"CNS stimulation|stimula\w+ (?:of )?the (?:central nervous system|CNS)|euphori\w*|"
    r"alertness|wakefulness|insomnia|sedation|sedative|drowsiness|analges\w*|"
    r"pain(?: relief| threshold)?|"
    r"antipyretic|fever|body temperature|blood glucose|glucose (?:levels?|production|uptake|"
    r"tolerance)|insulin (?:secretion|sensitivity|release)|bronchodilat\w*|vasoconstrict\w*|"
    r"vasodilat\w*|diures\w*|tolerance|tachyphylaxis|dependence|mood|anxiety|anxiolytic|"
    r"energy expenditure|metabolic rate|gastric emptying|acid secretion|cough reflex|"
    r"platelet aggregation|bleeding time|clotting|cholesterol|LDL|respiratory depression|"
    r"muscle relaxation|anticonvulsant|seizures?|inflammation|immune response|bacterial "
    r"growth|cell death|ovulation|fertility|erection|urine output|heart contractility)\b", re.I)
_LEADIN_RE = re.compile(r"^(?:lastly|also|in addition|additionally|furthermore|moreover|"
                        r"however|this|these|such|as well|thus|therefore|conversely|"
                        r"similarly|likewise|in contrast|because|at the same time|in turn|"
                        r"meanwhile|consequently|finally|then)\b", re.I)
# Speculation and indications — fine in the detail, poor as THE answer.
_SPECULATIVE_RE = re.compile(r"\bmay be (?:in part|partly|partially|due|related)\b|"
                             r"\bsuggest(?:s|ed)? that\b|\bpossibl[ey]\b", re.I)
# Lab-pharmacology detail ('…even at low nanomolar concentrations').
_LAB_RE = re.compile(r"\bnanomolar|\bmicromolar|\bat (?:low |high )?concentrations\b|"
                     r"\b\d+(?:\.\d+)?\s*(?:nM|µM|μM|uM)\b", re.I)
# Refers back to something the reader hasn't seen ('…to this receptor complex').
_DANGLING_REF_RE = re.compile(r"\b(?:this|these|such|the same|the latter|the former) "
                              r"(?:receptor|complex|enzyme|channel|mechanism|effect|process|"
                              r"pathway|site|protein|action|interaction)s?\b", re.I)
_CONTRAST_RE = re.compile(r"\brather than\b|\binstead of\b|,\s*however,|\bhowever\b", re.I)
# A sentence about the target itself, not the drug ('The GABAA receptors are…').
_TARGET_SUBJECT_RE = re.compile(r"^(?:the )?[\w\-α-ω]+(?: [\w\-α-ω]+)? (?:receptors?|enzymes?|"
                                r"channels?|transporters?) (?:are|is)\b", re.I)
_USES_CLAUSE_RE = re.compile(r"\bused (?:in|for) the treatment|\bused to treat\b|"
                             r"\bindicated (?:for|in)\b|\bsold under the brand|"
                             r"\bis used for its\b|\bused in the (?:therapy|management) of\b",
                             re.I)
# Sentences that sit in a 'Pharmacology' / 'Mechanism' section but answer a
# different question — how it's metabolised, its formulation or stereochemistry,
# PubChem's ChEBI role boilerplate — kept out of 'how does it work' entirely.
_MECH_OFFTOPIC_RE = re.compile(
    r"\bha(?:s|ve) (?:a )?roles? as\b|\bxenobiotic\b|\benvironmental contaminant\b|"
    r"\bundergoes? (?:\w+ ){0,3}metabolism\b|\bis (?:extensively |mainly |largely )?"
    r"metaboli[sz]ed\b|\bmetaboli[sz]ed (?:by|in|via|to)\b|\bglucuronid\w*|\bhalf-?life\b|"
    r"\bexcreted\b|\bbioavailab\w*|\bpatch(?:es)?\b|\bformulations?\b|\benantiomers?\b|"
    r"\bracemic\b|\bstereogenic\b|\bchiral\b|\bmirror-image\b|\bultraviolet\b|\bUV (?:light|"
    r"exposure|radiation)\b|\bmarketed as\b", re.I)
# Advice or asides that only make sense next to a sentence the reader can't see.
_ORPHAN_RE = re.compile(r"^given\b|\bsuch (?:a |an )?(?:risk|circumstances?|cases?|patients?|"
                        r"situations?)\b|\bin such\b|\bit is better to\b|"
                        r"^this (?:occurred|was|happened)\b", re.I)
# Study write-ups and dose-specific findings: fine as detail, not as 'effects'.
_STUDY_DETAIL_RE = re.compile(r"\bstud(?:y|ies)\b|\b\d+(?:\.\d+)?\s*mg\b|^no effect\b", re.I)
# Consequence sentences that finish the mechanism's story ('This results in
# decreased prostaglandins that mediate pain…').
_CONSEQUENCE_RE = re.compile(r"^(?:this|which|these effects?) (?:results? in|leads? to|causes?)\b|"
                             r"^as a (?:result|consequence)\b", re.I)
# Statements about the drug combined with another, or animal findings — true,
# but not the answer to "how does X work".
_MECH_ASIDE_RE = re.compile(r"\bcombin\w*|\bco-?administ\w*|\btogether with\b|"
                            r"\bin (?:animals|rats|mice|dogs|monkeys)\b(?! and humans)|"
                            r"\banimal models?\b|\bin vitro\b|\bstrains?\b|"
                            r"\belectroencephalograph\w*|\blaboratory\b|\bmodels of\b", re.I)

# ChEMBL action types → readable verbs ('RELEASING AGENT' + 'Sodium-dependent
# noradrenaline transporter' → 'Triggers transmitter release via the
# noradrenaline transporter').
_ACTION_PHRASE = {
    "inhibitor": "Inhibits the {t}", "antagonist": "Blocks the {t}",
    "agonist": "Activates the {t}", "partial agonist": "Partially activates the {t}",
    "inverse agonist": "Is an inverse agonist at the {t}",
    "positive allosteric modulator": "Boosts the {t} (positive allosteric modulator)",
    "negative allosteric modulator": "Dampens the {t} (negative allosteric modulator)",
    "allosteric antagonist": "Blocks the {t} (allosteric site)",
    "blocker": "Blocks the {t}", "opener": "Opens the {t}",
    "releasing agent": "Triggers transmitter release via the {t}",
    "modulator": "Modulates the {t}", "binding agent": "Binds {t}",
    "substrate": "Is carried by the {t}", "activator": "Activates the {t}",
    "stabiliser": "Stabilises the {t}", "chelating agent": "Binds (chelates) {t}",
    "disrupting agent": "Disrupts {t}", "cross-linking agent": "Cross-links {t}",
    "hydrolytic enzyme": "Breaks down {t}", "exogenous gene": "Supplies the gene for {t}",
    "exogenous protein": "Replaces {t}",
}


def _readable_target(t: str) -> str:
    t = re.sub(r"^sodium-dependent\s+", "", (t or "").strip(), flags=re.I)
    t = re.sub(r";\s*(.+)$", r" (\1)", t)
    if len(t) > 1 and t[0].isupper() and t[1].islower():
        t = t[0].lower() + t[1:]
    return t


def _chembl_mechs(ctx) -> List[Tuple[str, str, str]]:
    """ChEMBL mechanism records → [(readable sentence, target, action)]."""
    out = []
    for me in (ctx.get("clin_chembl") or {}).get("mechanisms") or []:
        mech = (me.get("mechanism") or "").strip()
        if not mech or mech.lower() == "unknown":
            continue
        tgt = _readable_target(me.get("target_name") or "")
        act = (me.get("action_type") or "").strip().lower()
        tmpl = _ACTION_PHRASE.get(act)
        if tmpl and tgt:
            text = tmpl.format(t=tgt) + "."
        else:
            text = mech[:1].upper() + mech[1:] + (f" ({tgt})" if tgt and tgt not in mech.lower() else "") + "."
        out.append((text, tgt or mech, act or "mechanism"))
    return out


def _drug_names_re(ctx) -> Optional[re.Pattern]:
    names = {ctx.get("compound_name") or "", (ctx.get("drug_hit") or {}).get("name", "")}
    names |= set(((ctx.get("label") or {}).get("generic_names") or [])[:1])
    names = [re.escape(n.split()[0]) for n in names if n and len(n) > 3]
    return re.compile(r"\b(?:" + "|".join(names) + r")\b", re.I) if names else None


# The molecular site a primary mechanism names (vs downstream biology).
_MECH_CORE_RE = re.compile(
    r"\b(?:receptors?|transporters?|enzymes?|channels?|pumps?|ATPase|reductase|synthase|"
    r"synthetase|kinase|cyclooxygenase|COX-?\d?|oxidase|esterase|carbonic anhydrase|"
    r"penicillin[- ]binding|topoisomerase|gyrase|polymerase|protease|integrase|"
    r"MAO-?[AB]?|VKORC1|HMG-CoA|reuptake)\b", re.I)
_NEGATION_RE = re.compile(r"\b(?:does not|do not|did not|doesn't|don't|lacks?|lack of|"
                          r"no (?:risk|effect|activity|significant)|without|"
                          r"(?:is|are|was|were|has|have|had) not)\b", re.I)
# Words in class names that say nothing about WHICH mechanism.
_CLASS_STOPWORDS = frozenset("""agent agents inhibitor inhibitors receptor receptors agonist
agonists antagonist antagonists blocker blockers activity drug drugs class classes other
products preparations derivatives combinations established pharmacologic mechanism action
actions system systemic effect effects plain selective chemical structure channel channels
pump pumps nervous central interactions neurotransmitter neurotransmitters transporter
transporters membrane symporter dependent cytochrome p450 inducers inducer substrates
substrate protein proteins enzyme enzymes""".split())


def _class_keywords(ctx) -> List[str]:
    """Specific words from the drug's curated classes (FDA EPC/MOA, ChEMBL
    targets): 'Serotonin Reuptake Inhibitor' → serotonin, reuptake."""
    names: List[str] = []
    drug = ctx.get("drug_hit")
    if drug and HAS_DRUG_LIBRARY:
        names += [dl.class_name(s) for s in drug["classes"] if dl.class_system(s) in ("EPC", "MOA")]
    lab = ctx.get("label") or {}
    names += [_strip_class_tag(c) for c in (lab.get("pharm_class_epc") or []) +
              (lab.get("pharm_class_moa") or [])]
    names += [tg for _t, tg, _a in _chembl_mechs(ctx)]
    words = []
    for n in names:
        for w in re.findall(r"[A-Za-z0-9β-]{4,}", n.lower()):
            w = w.strip("-")
            if w and w not in _CLASS_STOPWORDS and w not in words and not _GENERIC_CLASS_RE.match(w):
                words.append(w)
    return words[:12]
_CONTINUATION_START_RE = re.compile(r"^\W*\w+(?:\s+\w+){0,2},?\s+in turn\b|^in turn\b", re.I)


def _mech_score(text: str, src: str, names_re, origin: str = "",
                class_words: Sequence[str] = ()) -> float:
    """How well a sentence works as THE answer to 'how does it work': a primary
    mechanism (action + molecular site) about this drug, stated plainly — and in
    agreement with its curated class (class_words) when there is one."""
    act = bool(_MECH_ACT_RE.search(text))
    tgt = bool(_MECH_TGT_RE.search(text))
    core = act and bool(_MECH_CORE_RE.search(text))
    out = bool(_MECH_OUT_RE.search(text))
    eff = bool(_EFFECT_RE.search(text))
    score = 3 * act + 2 * tgt + 4 * core + 1 * out + 0.5 * eff
    if origin == "label_moa":
        score += 1.5                      # the label's own mechanism section
    anaphor = bool(_DRUG_ANAPHOR_RE.match(text))
    if names_re and names_re.search(text):
        score += 1
        if names_re.search(text[:30]):
            score += 2                    # the drug is the sentence's subject
    elif anaphor:
        score += 2
    elif class_words and any(w in text[:35].lower() for w in class_words):
        score += 1.5                      # its class as subject: 'Benzodiazepines are…'
    else:
        score -= 2                        # about something else (an enzyme, a metabolite…)
    neg = _NEGATION_RE.search(text)
    if neg:
        # 'does not inhibit platelets' isn't how it works; a late 'and does not
        # appreciably inhibit dopamine reuptake' is just a qualifier
        score -= 3 if neg.start() < 60 else 1
    if _CONTRAST_RE.search(text):
        score -= 2
    if _TARGET_SUBJECT_RE.match(text):
        score -= 3
    if _CONTINUATION_START_RE.search(text):
        score -= 3
    if _SECONDARY_RE.search(text):
        score -= 3
    if _SPECULATIVE_RE.search(text):
        score -= 2
    if _LAB_RE.search(text):
        score -= 3
    if _DANGLING_REF_RE.search(text):
        score -= 3
    if _USES_CLAUSE_RE.search(text):
        score -= 2                        # a 'what is it' sentence, not 'how it works'
    if re.match(r"^\S+(?: \S+)? is (?:mainly |mostly |commonly |widely )?used\b", text, re.I):
        score -= 3                        # opens with its uses: 'X is used for its … properties by…'
    if re.search(r"\bselective(?:ly)?\b", text, re.I):
        score += 1                        # 'selectively inhibits…' names the main action
    if class_words:
        low = text.lower()
        hits = sum(1 for w in class_words if w in low)
        score += min(4.5, 1.5 * hits)     # agrees with the curated class = primary action
    if anaphor and _LEADIN_RE.match(text):
        score += 2                        # undo the lead-in penalty below for 'This agent…'
    if _MECH_ASIDE_RE.search(text):
        score -= 3
    if text.rstrip().endswith(":"):
        score -= 4                        # a list's lead-in, not a statement
    score += {"FDA label": 1, "Wikipedia": 1, "ChEMBL": 0.5}.get(src, 0)
    if _MECH_HEDGE_RE.search(text):
        score -= 4
    if _RESEARCH_RE.search(text) or _STRONG_DETAIL_RE.search(text):
        score -= 3
    if _LEADIN_RE.match(text):
        score -= 2
    # a headline should be readable: long, abbreviation-dense sentences are detail
    score -= max(0.0, (len(text) - 180) / 60)
    if len(re.findall(r"\([^)]{1,15}\)", text)) >= 3:
        score -= 1.5
    if len(text) < 45:
        score -= 1
    if not act and not (tgt and out):
        score -= 3
    return score


def _outcome_score(text: str, names_re) -> float:
    """How well a sentence says what the drug's action RESULTS in, for the
    second line of the 'In short' box (appetite suppression, pain relief…)."""
    eff = bool(_EFFECT_RE.search(text))
    out = bool(_MECH_OUT_RE.search(text))
    if not (eff or out):
        return -99
    sc = 3 * eff + 2 * out + 1 * bool(_MECH_ACT_RE.search(text))
    if names_re and names_re.search(text):
        sc += 2
    elif _DRUG_ANAPHOR_RE.match(text):
        sc += 1
    for rx, pen in ((_NEGATION_RE, 3), (_MECH_ASIDE_RE, 3), (_MECH_HEDGE_RE, 4),
                    (_SECONDARY_RE, 2), (_LEADIN_RE, 1), (_SPECULATIVE_RE, 2)):
        if rx.search(text):
            sc -= pen
    if _RESEARCH_RE.search(text) or _STRONG_DETAIL_RE.search(text) \
            or len(_DETAIL_RE.findall(text)) >= 2:
        sc -= 2
    if len(text) > 250:
        sc -= 2
    if _CONSEQUENCE_RE.search(text):
        sc += 3                           # finishes the story: '…which mediate pain, fever'
                                          # (+1 offsets the 'This …' lead-in penalty)
    if re.match(r"^\S+(?: \S+)? is (?:mainly |mostly |commonly |widely )?used\b", text, re.I) \
            or _USES_CLAUSE_RE.search(text):
        sc -= 3                           # a uses sentence, not a result
    return sc


def _mech_material(ctx):
    """Every mechanism / pharmacodynamics sentence, de-duplicated:
    [(text, src, url, origin)] with origin 'chembl' | 'label_moa' | 'label_pd' | 'record'."""
    lab = ctx.get("label") or {}
    lurl = lab.get("url", "")
    curl = (ctx.get("clin_chembl") or {}).get("url", "")
    raw: List[Tuple[str, str, str, str]] = []
    raw += [(t, "ChEMBL", curl, "chembl") for t, _tg, _a in _chembl_mechs(ctx)]
    for field, origin in (("mechanism_of_action", "label_moa"), ("pharmacodynamics", "label_pd")):
        for s in _split_sentences(_tidy(lab.get(field) or "")):
            if len(s) >= 25 and not _LABEL_BOILERPLATE_RE.search(s):
                raw.append((s, "FDA label", lurl, origin))
    # Wikipedia first (plain language), then PubChem (DrugBank/HSDB), each in
    # the record's own order; the article's lead often has the clearest
    # one-line mechanism ('X is a β2 agonist which relaxes airway muscle')
    corpus = [d for d in _chat_corpus(ctx) if d["topic"] == "mechanism"]
    for want in ("Wikipedia", "PubChem"):
        raw += [(d["text"], d["src"], d["url"], "record") for d in corpus if d["src"] == want]
    raw += [(d["text"], d["src"], d["url"], "lead") for d in _chat_corpus(ctx)
            if d["topic"] == "overview" and _MECH_ACT_RE.search(d["text"])
            and _MECH_TGT_RE.search(d["text"])]
    out, dd = [], _SentenceDeduper()
    for text, src, url, origin in raw:
        text = _tidy(text)
        ws = _content_words(text)
        if len(text) < 20 or dd.is_dup(ws) or dd.is_restatement(ws):
            continue
        if origin != "chembl" and (_MECH_OFFTOPIC_RE.search(text) or _ORPHAN_RE.search(text)):
            continue
        dd.accept(ws)
        out.append((text, src, url, origin))
    return out


def _ans_mechanism(ctx, name, p):
    blocks: List[Dict[str, Any]] = []
    names_re = _drug_names_re(ctx)
    material = _mech_material(ctx)
    hedges = [m for m in material if _MECH_HEDGE_RE.search(m[0])]
    usable = [m for m in material if m not in hedges]

    # 1) the direct answer — focus words first ('…stop me from COUGHING')
    rel = []
    cw = _class_keywords(ctx)
    if p.get("focus"):
        # the sentences about the asked-about thing ('coughing'), best
        # explanation first: relevance picks candidates, mechanism quality orders
        cand = _relevant(ctx, p["focus"], topics={"mechanism", "uses", "pharma", "overview",
                                                   "pk", "side_effects"},
                         prefer={"mechanism", "pharma"}, k=6)
        ranked = sorted(cand, key=lambda it: -_mech_score(it[0], it[1], names_re, "", cw))
        rel = [it for it in ranked if _mech_score(it[0], it[1], names_re, "", cw) >= 3][:2] \
            or cand[:2]
    scored = sorted(((_mech_score(t, s, names_re, o, cw), (t, s, u, o)) for t, s, u, o in usable),
                    key=lambda x: -x[0])
    # ChEMBL's curated one-liners are shown as the Targets chips; the headline
    # prefers a readable sentence and falls back to ChEMBL only if none is good
    prose = [x for x in scored if x[1][3] != "chembl"]
    if prose and prose[0][0] >= 3:
        scored = prose + [x for x in scored if x[1][3] == "chembl"]
    summary: List[Tuple[str, str, str, str]] = []
    if scored and scored[0][0] >= 3:
        summary.append(scored[0][1])
        first = scored[0][1][0]
        # add a second line when the first doesn't say what it means for the
        # BODY (less pain, lower glucose…) — and that isn't the first reworded.
        # 'Inhibits COX → fewer prostaglandins' needs '→ less pain and fever'.
        if not _EFFECT_RE.search(first):
            fw = _content_words(first)
            outs = sorted(((_outcome_score(m[0], names_re), m) for _sc, m in scored[1:]
                           if _EFFECT_RE.search(m[0]) and not _ORPHAN_RE.search(m[0])
                           and len(fw & _content_words(m[0])) < 0.5 * max(1, len(_content_words(m[0])))),
                          key=lambda x: -x[0])
            if outs and outs[0][0] >= 4:
                summary.append(outs[0][1])
    # the user's own study notes, when the drug is in their reference
    mine = []
    if HAS_MAINSTREAM_REF:
        ref = mref.find_entry(name, ctx.get("compound_name") or "",
                              (ctx.get("drug_hit") or {}).get("name", ""))
        if ref:
            mine = [(ref["mechanism"], "Your reference", "")]
    if rel:
        blocks += _answer_box(rel, label=f"About “{_focus_label(p)}”")
        blocks += _pts(mine + _dedup_against([(t, s, u) for t, s, u, _o in summary],
                                             [t for t, _, _ in rel]), title="How it works", cap=3)
    else:
        blocks += _answer_box(mine + [(t, s, u) for t, s, u, _o in summary],
                              label="Mechanism of action")
    shown = [t for t, _s, _u in rel] + [m[0] for m in summary]

    # 2) molecular targets as chips (ChEMBL curated + Guide to Pharmacology)
    chips = [(f"**{tg[:1].upper() + tg[1:]}**", act) for _t, tg, act in _chembl_mechs(ctx)]
    gt = ctx.get("gtopdb") or {}
    for it in (gt.get("interactions") or [])[:6]:
        act = " ".join(x for x in (it.get("action") or it.get("type") or "",
                                   f"{it['parameter']} {it['affinity']}" if it.get("affinity") else "")
                       if x).lower()
        chips.append((f"**{it['target']}**", act))
    if chips:
        blocks.append(_b("chips", items=chips[:8], title="Targets",
                         src="ChEMBL" if _chembl_mechs(ctx) else "Guide to Pharmacology",
                         url=(ctx.get("clin_chembl") or {}).get("url", "") or gt.get("url", "")))

    # 3) effects on the body — the other half of pharmacodynamics
    eff = []
    for t, s, u, o in usable:
        if t in shown or not _EFFECT_RE.search(t):
            continue
        if _RESEARCH_RE.search(t) or _STRONG_DETAIL_RE.search(t) or _MECH_ASIDE_RE.search(t):
            continue                      # animal / lab / combination findings → detail
        if not (names_re and names_re.search(t) or _DRUG_ANAPHOR_RE.match(t)
                or _MECH_ACT_RE.search(t) or _MECH_OUT_RE.search(t)):
            continue                      # not about this drug's action at all
        if _USES_CLAUSE_RE.search(t) or _STUDY_DETAIL_RE.search(t):
            continue                      # what it's for / trial detail → not an effect
        sc = 2 + (2 if o == "label_pd" else 0) + (1 if names_re and names_re.search(t) else 0) \
            - (1 if len(t) > 300 else 0) - (2 if _NEGATION_RE.search(t) else 0) \
            - (1 if _LEADIN_RE.match(t) else 0)
        eff.append((sc, (t, s, u)))
    eff = [it for _sc, it in sorted(eff, key=lambda x: -x[0])]
    blocks += _pts(eff, title="Effects on the body", cap=4, more="More effects")
    shown += [t for t, _s, _u in eff]

    # 4) everything else, collapsed (lab/animal findings and list lead-ins
    # last); hedges as a footnote rather than a headline
    # ChEMBL one-liners are already the Targets chips
    detail = [(t, s, u) for t, s, u, o in usable if t not in shown and o != "chembl"]
    detail.sort(key=lambda it: bool(_RESEARCH_RE.search(it[0]) or _MECH_ASIDE_RE.search(it[0])
                                    or _STRONG_DETAIL_RE.search(it[0])
                                    or it[0].rstrip().endswith(":")))
    blocks += _pts(detail, cap=0, more="More on the mechanism")
    # a footnote only when a source says THIS drug's mechanism is uncertain —
    # not any hedge (a cohort study's 'risk is unclear')
    hedges = [h for h in hedges if len(h[0]) < 220
              and re.search(r"\bmechanism|\bmode of action\b|\bhow it works\b", h[0], re.I)
              and (names_re and names_re.search(h[0]) or _DRUG_ANAPHOR_RE.match(h[0]))]
    if hedges:
        note = _tidy(hedges[0][0])
        blocks.append(_b("caption", v=f"Note — {_md_safe(note)} ({hedges[0][1]})"))

    if not summary and not rel:
        drug = ctx.get("drug_hit")
        moa = [dl.class_name(s) for s in (drug or {}).get("classes", [])
               if HAS_DRUG_LIBRARY and dl.class_system(s) in ("MOA", "EPC")
               and not _GENERIC_CLASS_RE.match(dl.class_name(s))]
        if moa:
            blocks.insert(0, _b("answer", items=[(f"FDA classifies {name} as: **{' · '.join(moa[:4])}**.",
                                                 "FDA class", "")], label="In short"))
    return blocks


PARENT_TAG = " ↩ parent"


def _pk_items(ctx) -> Tuple[List[Tuple[str, str, str]], str]:
    """Every pharmacokinetic sentence for a compound — record PK heading, FDA
    label PK section, Wikipedia PK text — de-duplicated, with the record
    heading's scope ('primary' or 'parent'). Sentences taken from the PARENT
    compound's record carry it in their source tag ('PubChem ↩ parent'), so
    the scope stays visible wherever the sentence ends up."""
    # every bundle's PK heading — the exact compound first, then the parent's:
    # a salt's own heading is often one borrowed Wikipedia line, while the
    # free base's record holds the measured ADME data
    items: List[Tuple[str, str, str]] = []
    for grouped, sc in _ctx_grouped_list(ctx):
        heading, val = _topic_heading(grouped, "pk")
        if heading is None:
            continue
        pts = _points_for(ctx, heading, val, sc)
        if sc == "parent":
            pts = [(s, src + PARENT_TAG, u) for s, src, u in pts]
        items += pts
    scope = "parent" if any(PARENT_TAG in src for _s, src, _u in items) else ""
    lbl_url = (ctx["label"] or {}).get("url", "")
    items += [(s, "FDA label", lbl_url)
              for s in _split_sentences(_tidy((ctx["label"] or {}).get("pharmacokinetics") or ""))
              if len(s) > 25]
    items += [(d["text"], d["src"], d["url"]) for d in _chat_corpus(ctx)
              if d["topic"] == "pk" and d["src"] == "Wikipedia"]
    return _dedup_against(items, []), scope


def _ans_pk(ctx, name, p):
    items, scope = _pk_items(ctx)
    if not items:
        return _pts(_relevant(ctx, p.get("focus") or ["half", "life", "metabolism"], topics={"pk"}),
                    title="From the record")
    # (source, url) travel with each sentence through fact extraction
    facts, rest = _pk_key_facts([(s, (src, u)) for s, src, u in items])
    wanted = [w for w in (p.get("pk_facts") or []) if any(f[0] == w for f in facts)
              or w in dict(_pk_detail_groups(rest))]
    blocks: List[Dict[str, Any]] = []
    fact_pts = [(f"**{lbl}:** {s}", src, u) for lbl, s, (src, u) in facts]
    if wanted:
        direct = [fp for fp, (lbl, _s, _src) in zip(fact_pts, facts) if lbl in wanted]
        blocks += _answer_box(direct[:2], label=" · ".join(wanted))
        groups = dict(_pk_detail_groups(rest))
        more = [(s, src, u) for w in wanted for s, (src, u) in groups.get(w, [])]
        more = _dedup_against(more, [s for lbl, s, _src in facts if lbl in wanted])
        blocks += _pts(direct[2:] + more, title=f"More on {', '.join(w.lower() for w in wanted)}",
                       cap=3, more="More detail")
        others = [fp for fp, (lbl, _s, _src) in zip(fact_pts, facts) if lbl not in wanted]
        blocks += _pts(others, cap=0, more="Other pharmacokinetic facts")
    else:
        blocks += _pts(fact_pts, title="At a glance", cap=6)
        detail = [(f"*{lbl}:* {s}", src, u)
                  for lbl, grp in _pk_detail_groups(rest) for s, (src, u) in grp]
        blocks += _pts(detail, cap=0, more="Detailed pharmacokinetics")
    if scope == "parent":
        blocks.append(_b("caption", v="↩ Lines tagged *parent* come from the parent compound "
                                      "(free base/acid), not the exact salt searched."))
    return blocks


def _ans_uses(ctx, name, p):
    blocks: List[Dict[str, Any]] = []
    of = ctx.get("label") or {}
    if p.get("focus"):
        rel = _relevant(ctx, p["focus"], topics={"uses", "overview", "mechanism", "pharma"},
                        prefer={"uses"}, k=3)
        if rel:
            blocks += _answer_box(rel[:2], label=f"About “{_focus_label(p)}”")
            blocks += _pts(rel[2:], cap=2)
        else:
            blocks.append(_b("info", v=f"I couldn't find “{_md_safe(_focus_label(p))}” among "
                                       f"{name}'s documented uses in these sources."))
    pts = _label_points(of, "indications") or _label_points(of, "purpose")
    appr = chembl_approved_indications(ctx.get("clin_chembl") or {})
    items, _scope = _heading_points(ctx, "uses")
    if not p.get("focus"):
        if pts:
            blocks += _answer_box(pts[:1], label="Approved uses (FDA label)")
            pts = pts[1:]
        elif appr:
            blocks += _answer_box([(", ".join(appr[:15]), "ChEMBL",
                                    (ctx.get("clin_chembl") or {}).get("url", ""))],
                                  label="Approved for")
            appr = []
        elif items:
            blocks += _answer_box(items[:1], label="Used for")
            items = items[1:]
    blocks += _pts(pts, title="More approved uses", cap=4)
    if appr and not pts:
        blocks.append(_b("md", v="**Approved indications (ChEMBL):** " + _md_safe(", ".join(appr[:15]))))
    items = _dedup_against(items, [t for t, _, _ in pts])
    if items:
        blocks += _pts(items, title="", cap=0, more="More from PubChem & Wikipedia")
    return blocks


def _label_product_note(of: Dict[str, Any]) -> str:
    """Which product the FDA label describes — dosing is product-specific, and
    openFDA's best-ranked label can be an injection or a combination."""
    if not of:
        return ""
    brand = (of.get("brand_names") or [""])[0]
    routes = [r.lower() for r in of.get("routes") or []]
    gen = (of.get("generic_names") or [""])[0].lower()
    bits = [b for b in (brand, gen if gen and gen != brand.lower() else "") if b]
    txt = "FDA label used: " + " · ".join(bits) + (f" ({', '.join(routes)})" if routes else "")
    if routes and "oral" not in routes:
        txt = ("⚠️ " + txt + " — not an oral product, so tablet/syrup doses differ. "
               "Check your own product's leaflet.")
    return txt


_CHILD_RE = re.compile(r"\b(?:child|children|pediatric|paediatric|infant|adolescent|"
                       r"years? of age|under \d+|kids?)\b", re.I)
_ELDER_RE = re.compile(r"\b(?:geriatric|elderly|older (?:adults?|patients)|65 years)\b", re.I)


def _ans_dosing(ctx, name, p):
    of = ctx.get("label") or {}
    blocks: List[Dict[str, Any]] = []
    pop = p.get("population") or []
    def _lead(items, label, title):
        """First item as the answer box if nothing has claimed it yet."""
        nonlocal blocks
        if items and not any(b.get("t") == "answer" for b in blocks):
            blocks += _answer_box(items[:1], label=label)
            items = items[1:]
        blocks += _pts(items, title=title, cap=3)

    if p.get("focus"):
        _lead(_relevant(ctx, p["focus"], topics={"dosing"}, k=3),
              f"About “{_focus_label(p)}”", "")
    if "children" in pop:
        _lead(_label_points(of, "dosage", only=_CHILD_RE) + _label_points(of, "pediatric_use"),
              "For children (FDA label)", "More for children")
    if "elderly" in pop:
        _lead(_label_points(of, "geriatric_use") + _label_points(of, "dosage", only=_ELDER_RE),
              "For older adults (FDA label)", "More for older adults")
    dose = _label_points(of, "dosage")
    if dose and not any(b.get("t") == "answer" for b in blocks):
        blocks += _answer_box(dose[:1], label="Dosage (FDA label)")
        dose = dose[1:]
    blocks += _pts(dose, title="More dosing detail (FDA label)", cap=4,
                   more="Full dosing section")
    blocks += _pts(_label_points(of, "dosage_forms"), title="Forms & strengths", cap=3)
    note = _label_product_note(of)
    if note and blocks:
        blocks.insert(0, _b("caption", v=note))
    if not of:
        blocks += _pts(_relevant(ctx, ["dose", "dosage", "daily", "mg", "administered"],
                                 topics={"uses", "pharma", "overview"}, k=4),
                       title="Mentions of dosing in PubChem / Wikipedia", cap=4)
        blocks += _no_label_note(ctx, name, "dosing")
    if blocks:
        blocks.append(_b("caption", v="For information only — not medical advice. Follow your "
                                      "prescriber/pharmacist and the package insert."))
    return blocks


def _ans_side_effects(ctx, name, p):
    of = ctx.get("label") or {}
    blocks: List[Dict[str, Any]] = []
    common = _common_side_effects(of)
    if p.get("focus"):
        rel = _relevant(ctx, p["focus"], topics={"side_effects", "warnings", "overdose",
                                                  "toxicity", "pregnancy", "legal"},
                        prefer={"side_effects"}, k=4)
        if rel:
            blocks += _answer_box(rel[:2], label=f"What the sources say about “{_focus_label(p)}”")
            blocks += _pts(rel[2:], cap=2)
        else:
            blocks.append(_b("info", v=f"No mention of “{_md_safe(_focus_label(p))}” in {name}'s "
                                       "FDA label, PubChem or Wikipedia text. That doesn't rule "
                                       "it out — ask a pharmacist."))
    elif common:
        blocks += _answer_box([(common, "FDA label", of.get("url", ""))],
                              label="Most common (clinical trials)")
    if of.get("boxed_warning"):
        blocks.append(_b("error", v="**⚠️ FDA Boxed Warning** — "
                                    + _md_safe(boxed_summary(of["boxed_warning"], 300))))
    if common and p.get("focus"):
        blocks.append(_b("md", v=f"**Most common (clinical trials):** {_md_safe(common)}"))
    ar = _dedup_against(_label_points(of, "adverse_reactions"), [common] if common else [])
    blocks += _pts(ar, title="Adverse reactions (FDA label)", cap=3 if common else 6,
                   more="More adverse reactions")
    if not ar:
        # OTC 'Drug Facts' labels have no adverse-reactions section — use their
        # stop-use / when-using panels and the article's adverse-effects text
        blocks += _pts(_label_points(of, "when_using") + _label_points(of, "stop_use"),
                       title="Label: when using / stop use if", cap=4)
        blocks += _pts(_topic_items(ctx, {"side_effects", "toxicity"}),
                       title="From PubChem / Wikipedia", cap=6, more="More")
    if ctx.get("adverse_events") and not p.get("_compact"):
        blocks.append(_b("md", v="**Most-reported to the FDA (FAERS)**"))
        blocks.append(_b("faers", v=ctx["adverse_events"]))
    return blocks


def _ans_warnings(ctx, name, p):
    of = ctx.get("label") or {}
    blocks: List[Dict[str, Any]] = []
    if p.get("focus"):
        blocks += _pts(_relevant(ctx, p["focus"], topics={"warnings", "side_effects", "pk",
                                                          "toxicity", "interactions", "dosing"},
                                 prefer={"warnings"}, k=4),
                       title=f"About “{_focus_label(p)}”", cap=4)
    if of.get("boxed_warning"):
        blocks.append(_b("error", v="**⚠️ FDA Boxed Warning** — "
                                    + _md_safe(boxed_summary(of["boxed_warning"], 400))))
    blocks += _pts(_label_points(of, "contraindications") or _label_points(of, "do_not_use"),
                   title="Who should not take it", cap=5)
    blocks += _pts(_label_points(of, "warnings"), title="Warnings & precautions", cap=4,
                   more="More warnings")
    blocks += _pts(_label_points(of, "ask_doctor") + _label_points(of, "stop_use"),
                   title="Ask a doctor / stop use if", cap=3)
    if not of:
        blocks += _pts(_topic_items(ctx, {"warnings", "hazard", "toxicity"}),
                       title="Hazards & toxicity (PubChem / Wikipedia)", cap=6, more="More")
        blocks += _no_label_note(ctx, name, "warnings")
    return blocks


_INTERACTANT_TERMS = {
    "alcohol": ["alcohol", "ethanol", "alcoholic"], "beer": ["alcohol", "ethanol"],
    "wine": ["alcohol", "ethanol", "wine"], "liquor": ["alcohol", "ethanol"],
    "grapefruit": ["grapefruit"], "grapefruit juice": ["grapefruit"],
    "food": ["food", "meal", "fed", "fasting", "empty stomach"],
    "meal": ["food", "meal", "fed", "fasting"], "meals": ["food", "meal", "fed"],
    "milk": ["milk", "dairy", "calcium"], "dairy": ["milk", "dairy", "calcium"],
    "coffee": ["caffeine", "coffee"], "caffeine": ["caffeine", "coffee"],
    "antacid": ["antacid", "aluminum", "magnesium"], "antacids": ["antacid", "aluminum", "magnesium"],
    "smoking": ["smoking", "smokers", "tobacco", "nicotine"], "tobacco": ["tobacco", "smoking"],
    "cannabis": ["cannabis", "marijuana", "cannabinoid"], "marijuana": ["cannabis", "marijuana"],
    "contraceptives": ["contraceptive", "oral contraceptives", "estrogen"],
    "birth control": ["contraceptive", "oral contraceptives"],
    "supplements": ["supplement", "herbal"], "herbal": ["herbal", "supplement"],
}
# FDA/ATC class name → the words labels use for that class in interaction text.
_CLASS_KEYWORDS = [
    (r"nonsteroidal anti-?inflammatory|cyclooxygenase inhibitor|anti-inflammatory and "
     r"antirheumatic", ["NSAID", "NSAIDs", "nonsteroidal anti-inflammatory"]),
    (r"serotonin reuptake inhibitor", ["SSRI", "SSRIs", "serotonergic"]),
    (r"serotonin and norepinephrine reuptake", ["SNRI", "SNRIs", "serotonergic"]),
    (r"monoamine oxidase inhibitor", ["MAOI", "MAOIs", "MAO inhibitor"]),
    (r"opioid|opium", ["opioid", "opioids", "CNS depressant", "CNS depressants"]),
    (r"benzodiazepine", ["benzodiazepine", "benzodiazepines", "CNS depressant"]),
    (r"anticoagulant|vitamin k antagonist|factor xa|thrombin inhibitor",
     ["anticoagulant", "anticoagulants"]),
    (r"platelet aggregation inhibitor", ["antiplatelet", "platelet inhibitors"]),
    (r"hmg-?coa reductase", ["statin", "statins", "HMG-CoA reductase inhibitor"]),
    (r"corticosteroid", ["corticosteroid", "corticosteroids"]),
    (r"angiotensin[- ]converting enzyme|ace inhibitor", ["ACE inhibitor", "ACE inhibitors"]),
    (r"angiotensin (?:2|ii) receptor", ["angiotensin receptor blocker", "ARB", "ARBs"]),
    (r"diuretic", ["diuretic", "diuretics"]),
    (r"beta[- ]?adrenergic blocker|beta blocking", ["beta-blocker", "beta-blockers", "beta blocker"]),
    (r"calcium channel blocker", ["calcium channel blocker", "calcium channel blockers"]),
    (r"histamine-?1 receptor antagonist|antihistamine", ["antihistamine", "antihistamines"]),
    (r"proton pump inhibitor", ["proton pump inhibitor", "proton pump inhibitors", "PPI"]),
    (r"sulfonylurea", ["sulfonylurea", "sulfonylureas"]),
    (r"insulin", ["insulin"]),
    (r"macrolide", ["macrolide", "macrolides"]),
    (r"quinolone", ["fluoroquinolone", "fluoroquinolones", "quinolone"]),
    (r"antacid", ["antacid", "antacids"]),
    (r"central nervous system depressant|sedative|hypnotic|barbiturate",
     ["CNS depressant", "CNS depressants", "sedative"]),
    (r"tricyclic|non-?selective monoamine reuptake", ["tricyclic antidepressant", "TCA", "TCAs"]),
    (r"triptan|serotonin 1b", ["triptan", "triptans", "serotonergic"]),
    (r"cardiac glycoside|digitalis", ["digoxin", "digitalis"]),
]


def _interaction_terms(ctx: Dict[str, Any], display: str) -> List[str]:
    """Names a label might use for this drug in another drug's interaction text:
    its own names/aliases plus its class words ('NSAIDs', 'anticoagulants')."""
    names = {display, ctx.get("compound_name") or "", (ctx.get("drug_hit") or {}).get("name", "")}
    names |= {g for g in (ctx.get("label") or {}).get("generic_names", [])[:2]}
    if HAS_SG_REGISTER:
        for n in list(names):
            if n:
                names |= set(sgr._variants(n))
    terms = [n for n in names if n and len(n) >= 3 and len(n) < 40]
    classes = []
    drug = ctx.get("drug_hit")
    if drug and HAS_DRUG_LIBRARY:
        classes += [dl.class_name(s) for s in drug["classes"]]
    classes += [_strip_class_tag(c) for c in (ctx.get("label") or {}).get("pharm_class_epc", [])]
    for rx, words in _CLASS_KEYWORDS:
        if any(re.search(rx, c, re.I) for c in classes):
            terms += words
    return list(dict.fromkeys(terms))


def _ans_interactions(ctx, name, p, other=None):
    """Interactions of one drug; with `other` = (display, ctx_or_None) or a
    named interactant (alcohol, grapefruit…), what the sources say about that
    specific combination."""
    of = ctx.get("label") or {}
    blocks: List[Dict[str, Any]] = []
    target, terms = "", []
    if other is not None:
        target, octx = other
        terms = _interaction_terms(octx, target) if octx else [target]
    elif p.get("interactant"):
        target = p["interactant"]
        terms = _INTERACTANT_TERMS.get(target.lower(), [target])
    elif p.get("focus"):
        target = _focus_label(p)
        terms = [t for f in p["focus"] for t in _INTERACTANT_TERMS.get(f, [f])]
    if target:
        hits = _mentions(ctx, terms, topics={"interactions", "warnings", "side_effects",
                                             "dosing", "pk", "pharma", "toxicity", "overdose"})
        if hits:
            blocks += _answer_box(hits[:1], label=f"{name} + {target}")
            blocks += _pts(hits[1:], title=f"More from {name}'s sources", cap=3,
                           more="More mentions")
        else:
            blocks.append(_b("info", v=f"No specific mention of **{_md_safe(target)}** in "
                                       f"{name}'s FDA label, PubChem or Wikipedia text. That "
                                       "doesn't prove the combination is safe — check with a "
                                       "pharmacist."))
    gen = _label_points(of, "drug_interactions")
    if gen:
        gen = _dedup_against(gen, [])
        blocks += _pts(gen, title="" if target else "Drug interactions (FDA label)",
                       cap=0 if target else 6, more="All interactions listed on the FDA label")
    elif not target:
        blocks += _pts(_topic_items(ctx, {"interactions"}), title="Interactions", cap=6,
                       more="More")
        blocks += _no_label_note(ctx, name, "interaction")
    return blocks


_PREG_RE = re.compile(r"pregnan|fetal|foetal|teratogen|birth defect|reproductive|"
                      r"fertility|conceive", re.I)
_LACT_RE = re.compile(r"lactat|breast|nursing|human milk|infant", re.I)


def _ans_pregnancy(ctx, name, p):
    of = ctx.get("label") or {}
    raw = (p.get("raw") or "").lower()
    lact_first = bool(re.search(r"breast|lactat|nursing|milk", raw))
    preg = _label_points(of, "specific_populations", only=_PREG_RE) or \
        _label_points(of, "pregnancy")
    lact = _label_points(of, "specific_populations", only=_LACT_RE) or \
        _label_points(of, "lactation")
    preg = [x for x in preg if not (_LACT_RE.search(x[0]) and not _PREG_RE.search(x[0]))]
    otc = _label_points(of, "warnings", only=re.compile(r"pregnan|breast", re.I))
    parts = [("Breastfeeding", lact), ("Pregnancy", preg)] if lact_first else \
        [("Pregnancy", preg), ("Breastfeeding", lact)]
    blocks: List[Dict[str, Any]] = []
    for n, (title, pts) in enumerate(parts):
        pts = _dedup_against(pts, [])
        if n == 0 and pts:
            blocks += _answer_box(pts[:1], label=f"{title} (FDA label)")
            pts = pts[1:]
        blocks += _pts(pts, title=f"{title} (FDA label)" if n else "", cap=3,
                       more=f"More on {title.lower()}")
    if otc and not (preg or lact):
        blocks += _pts(otc, title="Label warning", cap=3)
    if not blocks:
        blocks += _pts(_topic_items(ctx, {"pregnancy"}), title="From PubChem / Wikipedia", cap=5)
    if blocks:
        blocks.append(_b("caption", v="Always check with a doctor or pharmacist before taking "
                                      "any medicine while pregnant or breastfeeding."))
    return blocks


def _ans_population(ctx, name, p):
    of = ctx.get("label") or {}
    blocks: List[Dict[str, Any]] = []
    pop = p.get("population") or ["children"]
    if "children" in pop:
        pts = (_label_points(of, "pediatric_use")
               or _label_points(of, "specific_populations", only=_CHILD_RE))
        pts += _label_points(of, "dosage", only=_CHILD_RE)
        pts += _label_points(of, "do_not_use", only=_CHILD_RE) + \
            _label_points(of, "warnings", only=_CHILD_RE)
        blocks += _pts(_dedup_against(pts, []), title="Children (FDA label)", cap=5)
    if "elderly" in pop:
        pts = (_label_points(of, "geriatric_use")
               or _label_points(of, "specific_populations", only=_ELDER_RE))
        blocks += _pts(_dedup_against(pts, []), title="Older adults (FDA label)", cap=5)
    if not blocks:
        blocks += _no_label_note(ctx, name, "age-specific")
    return blocks


def _ans_overdose(ctx, name, p):
    of = ctx.get("label") or {}
    blocks: List[Dict[str, Any]] = []
    if _is_med(ctx):
        blocks.append(_b("warning", v="If someone may have taken an overdose, call emergency "
                                      "services now (995 in Singapore) or a poison centre."))
    od = _label_points(of, "overdosage")
    blocks += _answer_box(od[:1], label="Overdose (FDA label)")
    blocks += _pts(od[1:], cap=3, more="More")
    tox = [it for it in _topic_items(ctx, {"overdose", "toxicity"})
           if not (od and it[1] == "FDA label")]          # label lines already shown
    blocks += _pts(tox, title="Toxicity (PubChem / Wikipedia)", cap=0 if of else 6,
                   more="Toxicity data")
    if not _is_med(ctx):
        blocks += _pts(_topic_items(ctx, {"hazard"}), title="Hazards", cap=4, more="More hazards")
    return blocks if len(blocks) > 1 or not _is_med(ctx) else []


def _ans_structure(ctx, name, p):
    smiles = ctx.get("smiles") or ""
    if not smiles:
        return [_b("info", v=f"No small-molecule structure is available for {name} — it may be "
                             "a biologic (a protein, antibody or mixture) rather than a single "
                             "small molecule.")]
    mol = Chem.MolFromSmiles(smiles)
    raw = (p.get("raw") or "").lower()
    groups = ci.detect_functional_groups(smiles) or []
    want_groups = bool(re.search(r"functional|group|moiet", raw))
    hl = [(g["atom_indices"], _HL_PALETTE[i % len(_HL_PALETTE)])
          for i, g in enumerate(groups[:len(_HL_PALETTE)])] if want_groups else []
    png = render_2d_highlight(smiles, hl, size=(480, 360))
    dm = derived_metrics(smiles)
    facts = []
    if ctx.get("formula"):
        facts.append(f"- **Formula:** {ctx['formula']}")
    if dm:
        facts.append(f"- **Molecular weight:** {dm['Molecular weight'][0]} Da")
    iupac = (ctx.get("rawinfo") or {}).get("iupac_name")
    if iupac:
        facts.append(f"- **IUPAC name:** {_md_safe(iupac)}")
    if mol is not None:
        try:
            centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                                useLegacyImplementation=False)
        except Exception:  # noqa: BLE001
            centres = []
        facts.append(f"- **Rings:** {dm['Rings'][0]} ({dm['Aromatic rings'][0]} aromatic) · "
                     f"**Heavy atoms:** {dm['Heavy atoms'][0]} · "
                     f"**Stereocentres:** {len(centres)}"
                     + (f" ({dm['Undefined stereocenters'][0]} undefined)"
                        if dm['Undefined stereocenters'][0] else ""))
    if groups:
        if hl:
            names = ", ".join(f"{g['name']} ×{g['count']}" for g in groups[:len(hl)])
            facts.append(f"- **Functional groups (highlighted):** {names}")
        else:
            facts.append("- **Functional groups:** "
                         + ", ".join(f"{g['name']} ×{g['count']}" for g in groups))
    right = [_b("md", v="\n".join(facts))] if facts else []
    left = [_b("image", v=png, width=360)] if png else []
    blocks = [_b("cols", left=left, right=right, spec=[1.1, 1])] if left else right
    blocks.append(_b("code", v=smiles))
    if is_multifragment(smiles):
        blocks.append(_b("caption", v="This is a salt / multi-component structure (shown with "
                                      "its counter-ion)."))
    if re.search(r"\b3d\b|3-d|three[- ]dimensional|conformer", raw):
        heavy = mol.GetNumHeavyAtoms() if mol is not None else 0
        if heavy > _CHAT_3D_MAX_ATOMS:
            blocks.append(_b("caption", v=f"3D model skipped — {heavy} heavy atoms is too large "
                                          "to embed quickly. Use the full profile's 3D tab."))
        else:
            viewer, _blk, err = render_3d_viewer_and_block(smiles, width=520, height=380)
            if viewer is not None:
                try:
                    blocks.append(_b("html", v=viewer._make_html(), height=400))
                except Exception:  # noqa: BLE001
                    blocks.append(_b("caption", v="3D viewer unavailable here."))
            else:
                blocks.append(_b("caption", v=f"Couldn't build a 3D model ({err})."))
    return blocks


# (label, query pattern, derived_metrics keys, PubChem heading pattern)
_PROP_QUERIES = [
    ("Molecular weight", r"molecular (?:weight|mass)|mol\.? ?wt|\bmw\b|molar mass",
     ["Molecular weight", "Exact mass"], None),
    ("logP (lipophilicity)", r"log ?p|lipophil|hydrophob|hydrophil", ["cLogP (Crippen)"],
     r"\blogp\b|octanol|partition"),
    ("pKa (acid/base)", r"\bpka|\bpkb|acidic|basic|acid or base|base or acid|a (?:weak |strong )?"
     r"(?:acid|base)\b|ioni[sz]", [], r"dissociation|pka"),
    ("Solubility", r"solub|dissolv", [], r"solubility"),
    ("Melting point", r"melting", [], r"melting point"),
    ("Boiling point", r"boiling", [], r"boiling point"),
    ("Density", r"density", [], r"density"),
    ("Vapour pressure", r"vapou?r pressure", [], r"vapor pressure"),
    ("Polar surface area", r"tpsa|polar surface", ["TPSA"], None),
    ("Hydrogen bonding", r"h[- ]?bond|hydrogen bond", ["H-bond donors", "H-bond acceptors"], None),
    ("Appearance", r"appearance|colou?r|odou?r|solid|liquid|\bgas\b|look like", [],
     r"physical description|colou?r|odou?r"),
    ("Stability", r"stabil|reactiv|shelf", [], r"stability|reactivit|decomposition"),
    ("Flash point", r"flash point", [], r"flash point"),
    ("Refractive index", r"refractive", [], r"refractive"),
]


def _measured(ctx, heading_rx: str, n: int = 3) -> List[Tuple[str, str, str]]:
    rx = re.compile(heading_rx, re.I)
    out = []
    for grouped, scope in _ctx_grouped_list(ctx):
        for group in ("Experimental Properties & Measurements", "Physical Properties",
                      "Computed Properties / Descriptors", "Safety & Hazards"):
            for heading, val in (grouped.get(group) or {}).items():
                if not rx.search(heading) or not isinstance(val, str):
                    continue
                parts = [x.strip() for x in val.split(" | ") if x.strip()]
                head = _strip_depth(heading)
                for part in parts[:n]:
                    out.append((f"{head}: {part[:300]}"
                                + (" *(parent compound)*" if scope else ""),
                                "PubChem", _src_url(ctx, "PubChem")))
        if out:
            break
    return out


def _ans_properties(ctx, name, p):
    smiles = ctx.get("smiles") or ""
    raw = (p.get("raw") or "").lower()
    dm = derived_metrics(smiles) if smiles else {}
    wanted = [q for q in _PROP_QUERIES if re.search(q[1], raw)]
    drug_like = bool(re.search(r"drug[- ]?like|lipinski|rule of", raw))
    blocks: List[Dict[str, Any]] = []
    for label, _qrx, dm_keys, head_rx in wanted:
        pts = []
        for k in dm_keys:
            if k in dm:
                basis = DESCRIPTOR_BASIS.get(k, "Derived")
                pts.append((f"{k}: **{dm[k][0]}{(' ' + dm[k][1]) if dm[k][1] else ''}**",
                            f"RDKit · {basis.lower()}", ""))
        if head_rx:
            pts += _measured(ctx, head_rx)
        if label.startswith("pKa") and smiles:
            ab = ci.infer_acid_base(smiles)
            for c in (ab.get("centres") or [])[:4]:
                lo, hi = c["pka"]
                kind = "acid" if c["role"] == "acid" else "base (pKaH)"
                pts.append((f"{c['name']} — {kind}, pKa ≈ {lo:g}–{hi:g} → {c['state']} at pH 7.4",
                            "inferred from structure", ""))
            if ab.get("net_charge_note"):
                pts.append((ab["net_charge_note"], "inferred from structure", ""))
        if label == "Solubility" and smiles:
            sol = ci.infer_solubility_tendency(smiles)
            if sol:
                pts.append((sol["tendency"], "inferred from structure", ""))
        blocks += _answer_box(pts[:1], label=label)
        blocks += _pts(pts[1:], cap=3, more=f"More {label.lower()} data")
        if not pts:
            blocks.append(_b("caption", v=f"No {label.lower()} value found for {name}."))
    if drug_like and dm:
        ro5 = dm.get("_ro5_violations", 0)
        blocks.append(_b("md", v=("✅ Passes Lipinski's Rule of 5" if ro5 == 0
                                  else f"⚠️ {ro5} Lipinski violation(s)")
                         + f" · QED {dm['QED (drug-likeness)'][0]}"))
    if not wanted and not drug_like and dm:
        rows = [{"Property": k, "Value": f"{v[0]}{(' ' + v[1]) if v[1] else ''}",
                 "Basis": DESCRIPTOR_BASIS.get(k, "Derived")}
                for k, v in dm.items() if not k.startswith("_") and k in (
                    "Molecular weight", "cLogP (Crippen)", "TPSA", "H-bond donors",
                    "H-bond acceptors", "Rotatable bonds", "QED (drug-likeness)",
                    "Lipinski violations")]
        if ctx.get("formula"):
            rows.insert(0, {"Property": "Formula", "Value": ctx["formula"], "Basis": "Derived"})
        blocks.append(_b("table", rows=rows))
        meas = []
        for label, _q, _k, head_rx in _PROP_QUERIES:
            if head_rx:
                meas += _measured(ctx, head_rx, n=1)
        blocks += _pts(meas, title="Measured (PubChem)", cap=6, more="More measured properties")
    if blocks:
        blocks.append(_b("caption", v="Measured values come from PubChem's record (with their "
                                      "original sources); RDKit values are computed from the "
                                      "structure; 'inferred' values are reasoned from functional "
                                      "groups, not measured."))
    return blocks


def _ans_brands(ctx, name, p):
    market = ctx.get("market") or {}
    of = ctx.get("label") or {}
    sg = market.get("sg") or {}
    blocks: List[Dict[str, Any]] = []
    ent = p.get("_entity") or {}
    if ent.get("kind") == "brand" and ent.get("ingredients"):
        blocks.append(_b("md", v=f"**{_md_safe(ent['display'])}** is a brand of "
                                 f"**{_md_safe(', '.join(ent['ingredients']))}**."))
    generic = (of.get("generic_names") or [""])[0]
    if generic and generic.lower() != name.lower():
        blocks.append(_b("md", v=f"**Generic name:** {_md_safe(generic.lower())}"))
    if ctx.get("region") == "SG" and HAS_SG_REGISTER:
        if sg:
            blocks.append(_b("md", v=f"**Brands in Singapore:** {_md_safe(', '.join(sg['brands'][:15]))}"))
            blocks.append(_b("md", v=f"**Availability (HSA):** {_md_safe(sgr.availability_text(sg))}"))
            # every registered product (the table scrolls); combination products
            # are marked, since the class counts above cover single-ingredient ones
            rows = [{"Class": x["cls"], "Product": x["name"], "Strength": x.get("strength", ""),
                     "Form": (x.get("form") or "").title(),
                     "Type": "Combination" if len(x.get("ings") or []) > 1 else "Single"}
                    for x in sg["products"]]
            if rows:
                blocks.append(_b("table", rows=rows))
                n_combo = sum(r["Type"] == "Combination" for r in rows)
                if n_combo and not sg.get("combination_only"):
                    blocks.append(_b("caption", v=f"{len(rows)} products in total — the counts above are "
                                                  f"for the {len(rows) - n_combo} single-ingredient ones; "
                                                  f"{n_combo} combination product(s) also contain it."))
            blocks.append(_b("caption", v="POM = prescription-only · P = pharmacy-only · GSL = "
                                          "general sale. From the HSA Listing of Registered "
                                          "Therapeutic Products (data.gov.sg, Singapore Open Data "
                                          "Licence)."))
        else:
            blocks.append(_b("info", v=f"No HSA-registered product containing {name} was found in "
                                       "Singapore."))
    us = _brand_names(market, of, generic or name)
    if us:
        blocks.append(_b("md", v=f"**US brands:** {_md_safe(', '.join(us))}"))
    avail = _availability_label(market, of)
    if avail:
        blocks.append(_b("md", v=f"**US availability:** {avail}"))
    return blocks


# MED-RT hierarchy nodes the library carries as classes but that say nothing.
_GENERIC_CLASS_RE = re.compile(r"^(?:chemical structure|established pharmacologic class(?:es)?|"
                               r"mechanisms? of action|physiologic(?:al)? effects?|"
                               r"pharmacokinetics?)\s*$", re.I)


def _class_lines(drug) -> List[Tuple[str, str, str]]:
    if not drug or not HAS_DRUG_LIBRARY:
        return []
    order = {"ATC": 0, "EPC": 1, "MOA": 2}
    out = []
    for slug in sorted(drug["classes"], key=lambda s: (order.get(dl.class_system(s), 3),
                                                       bool(_MINOR_ATC_RE.search(dl.class_name(s))),
                                                       dl.class_name(s))):
        if _GENERIC_CLASS_RE.match(dl.class_name(slug)):
            continue
        sysname = dl.class_system(slug)
        desc = dl.class_desc(slug) if sysname == "ATC" else ""
        out.append((f"**{dl.class_name(slug)}**" + (f" — {desc}" if desc else ""),
                    {"ATC": "WHO ATC", "EPC": "FDA class", "MOA": "FDA mechanism"}.get(sysname, sysname),
                    ""))
    return out


def _ans_class(ctx, name, p):
    drug = ctx.get("drug_hit")
    blocks: List[Dict[str, Any]] = []
    asked = p.get("class_slugs") or []
    if asked and drug and HAS_DRUG_LIBRARY:
        mine = set(drug["classes"])
        hit = next((s for s in asked if s in mine or any(m.startswith(s) and
                    dl.class_system(m) == "ATC" and dl.class_system(s) == "ATC" for m in mine)), None)
        phrase = p.get("class_phrase") or dl.class_name(asked[0])
        if hit:
            blocks.append(_b("success", v=f"Yes — {name} is classed as **{dl.class_name(hit)}** "
                                          f"({dl.class_system(hit)})."))
        else:
            blocks.append(_b("info", v=f"{name} isn't listed under “{_md_safe(phrase)}” in the WHO "
                                       "ATC / FDA classes in the Drug Library."))
    lines = _class_lines(drug)
    blocks += _pts(lines, title="Classes" if lines else "", cap=5, more="All classes")
    if not lines:
        epc = [_strip_class_tag(c) for c in (ctx.get("label") or {}).get("pharm_class_epc", [])]
        atc = (ctx.get("clin_chembl") or {}).get("atc") or []
        if epc or atc:
            blocks.append(_b("md", v="**Classes:** " + _md_safe(" · ".join(epc + atc))))
    return blocks


def _ans_alternatives(ctx, name, p):
    drug = ctx.get("drug_hit")
    if not drug or not HAS_DRUG_LIBRARY:
        return []
    order = {"EPC": 0, "ATC": 1, "MOA": 2}
    slugs = sorted(drug["classes"], key=lambda s: (order.get(dl.class_system(s), 3),
                                                   bool(_MINOR_ATC_RE.search(dl.class_name(s)))))
    blocks: List[Dict[str, Any]] = []
    seen = {drug["name"]}
    buttons = []
    for slug in slugs[:4]:
        others = [d for d in dl.drugs_in_class(slug) if d["name"] not in seen]
        if not others or len(others) > 80:
            continue
        seen |= {d["name"] for d in others}
        names = ", ".join(d["name"] for d in others[:20]) + (" …" if len(others) > 20 else "")
        blocks.append(_b("md", v=f"**{_md_safe(dl.class_name(slug))}** ({dl.class_system(slug)}): "
                                 f"{_md_safe(names)}"))
        buttons += [(d["name"], f"What is {d['name']}?") for d in others[:2]]
        if len(blocks) >= 3:
            break
    if blocks:
        blocks.append(_b("caption", v="Same-class drugs aren't automatically interchangeable — "
                                      "switching should be decided with a doctor or pharmacist."))
        blocks.append(_b("buttons", items=buttons[:4], title="Ask about one:"))
    return blocks


def _ans_identifiers(ctx, name, p):
    lines = []
    cid = ctx.get("cid")
    if cid:
        lines.append(f"- **PubChem CID:** [{cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
    cas = next((s for s in ctx.get("synonyms") or [] if _CAS_RE.match(s)), "")
    if cas:
        lines.append(f"- **CAS number:** {cas}")
    if ctx.get("inchikey"):
        lines.append(f"- **InChIKey:** `{ctx['inchikey']}`")
    iupac = (ctx.get("rawinfo") or {}).get("iupac_name")
    if iupac:
        lines.append(f"- **IUPAC name:** {_md_safe(iupac)}")
    if ctx.get("formula"):
        lines.append(f"- **Formula:** {ctx['formula']}")
    ch = ctx.get("clin_chembl") or {}
    if ch.get("chembl_id"):
        lines.append(f"- **ChEMBL:** [{ch['chembl_id']}]({ch.get('url', '')})")
    unii = (ctx.get("label") or {}).get("unii") or []
    if unii:
        lines.append(f"- **UNII:** {unii[0]}")
    if (ctx.get("drug_hit") or {}).get("rxcui"):
        lines.append(f"- **RxNorm RxCUI:** {ctx['drug_hit']['rxcui']}")
    links = [f"[{x['db']}]({x['url']})" for x in (ctx.get("xrefs") or []) if x.get("url")][:12]
    if links:
        lines.append("- **Also in:** " + " · ".join(links))
    blocks = [_b("md", v="\n".join(lines))] if lines else []
    if ctx.get("smiles"):
        blocks.append(_b("md", v="**SMILES**"))
        blocks.append(_b("code", v=ctx["smiles"]))
    return blocks


def _ans_spectra(ctx, name, p):
    cid = ctx.get("cid")
    heads = []
    for grouped, _scope in _ctx_grouped_list(ctx):
        heads += [h for h in (grouped.get("Spectra & Analytical") or {})
                  if _is_spectral_heading(h.strip().lower())]
    if not heads or not cid:
        return [_b("info", v=f"PubChem's record for {name} has no spectra.")] if cid else []
    lines = [f"- [{_md_safe(_strip_depth(h))} ↗](https://pubchem.ncbi.nlm.nih.gov/compound/"
             f"{cid}#section={urlquote(str(h), safe='')}&fullscreen=true)" for h in heads[:12]]
    return [_b("md", v="Spectra in PubChem's record (opens PubChem):\n" + "\n".join(lines))]


def _ans_topic(topics, title, extra=None, search_terms=(), skip_label=False):
    """Builder for record-section topics (history, synthesis, legal, storage):
    the topic's sentences in record order; when the record has no such section,
    the sentences anywhere that best match the topic's keywords."""
    def build(ctx, name, p):
        blocks: List[Dict[str, Any]] = []
        if p.get("focus"):
            blocks += _pts(_relevant(ctx, p["focus"], topics=set(topics), k=3),
                           title=f"About “{_focus_label(p)}”", cap=3)
        if extra:
            blocks += extra(ctx, name, p)
        items = [it for it in _topic_items(ctx, set(topics))
                 if not (skip_label and it[1] == "FDA label")]
        if not items and search_terms:
            items = _relevant(ctx, list(search_terms), k=5)
        blocks += _pts(items, title=title, cap=6, more="More")
        return blocks
    return build


def _history_extra(ctx, name, p):
    ch = ctx.get("clin_chembl") or {}
    if ch.get("first_approval"):
        return [_b("md", v=f"**First approved:** {ch['first_approval']} (ChEMBL)")]
    return []


def _legal_extra(ctx, name, p):
    of = ctx.get("label") or {}
    blocks = _pts(_label_points(of, "controlled_substance") + _label_points(of, "abuse"),
                  title="Abuse & dependence (FDA label)", cap=4, more="More")
    sg = (ctx.get("market") or {}).get("sg")
    if sg and HAS_SG_REGISTER:
        blocks.append(_b("md", v=f"**Singapore (HSA):** {_md_safe(sgr.availability_text(sg))}"))
    return blocks


def _storage_extra(ctx, name, p):
    return _pts(_label_points(ctx.get("label") or {}, "storage"), title="Storage (FDA label)", cap=4)


def _ans_search(ctx, name, p):
    """No recognised question type, but extra words: the sentences that best
    match them — or, when nothing matches, the overview."""
    rel = _relevant(ctx, p.get("focus") or [], k=6)
    if rel:
        # matched words, not a recognised question — offer clearer phrasings
        return (_pts(rel, title=f"Most relevant to “{_focus_label(p)}”", cap=6)
                + _suggest_blocks(name, title="Not quite it? Try asking:"))
    return ([_b("caption", v=f"I wasn't sure what you wanted to know about {name} — here's the "
                             "overview. Tap a question below to ask something specific.")]
            + _ans_overview(ctx, name, p) + _suggest_blocks(name, skip_topics=("overview",)))


# =========================== Short answers ===================================
# A question's FORM decides how the answer opens: a yes/no question gets a
# verdict, 'how long' a duration, 'how much' a dose. Every verdict is derived
# from quoted source text (or computed from the structure) and says which; when
# the sources are silent it says so — never 'safe' by default.
_NEG_BEFORE_RE = re.compile(
    r"\b(?:no|not|never|without|neither|nor|unlikely|rarely|lack(?:s|ing)? of|absence of|free of|"
    r"does not|doesn't|did not|is not|isn't|are not|aren't|was not|were not|has not|have not|"
    r"cannot|can't|won't|will not|minimal|little|slight(?:ly)?|lower risk of)\b[^.;:]{0,45}$",
    re.I)
# symptom words whose lay synonyms are safe to add (direction-neutral)
_EFFECT_SYNONYMS = {
    "sleepy": ("drowsiness", "somnolence", "sedation", "sleep aid", "sleep-aid", "nighttime sleep"),
    "drowsy": ("drowsiness", "somnolence", "sedation"),
    "drowsiness": ("somnolence", "sedation", "drowsy"), "tired": ("fatigue", "somnolence"),
    "tiredness": ("fatigue",), "dizzy": ("dizziness", "vertigo"), "dizziness": ("dizzy", "vertigo"),
    "itchy": ("pruritus", "itching"), "itching": ("pruritus",), "hair": ("alopecia",),
    "nausea": ("nauseous",), "vomiting": ("emesis",), "headache": ("headaches",),
    "insomnia": ("sleeplessness", "difficulty sleeping"), "libido": ("sexual desire",),
    "constipation": ("constipated",), "diarrhea": ("diarrhoea",), "diarrhoea": ("diarrhea",),
}
_GENERIC_EFFECT_WORDS = {"side", "effect", "effects", "cause", "causes", "caused", "causing", "make",
                         "makes", "feel", "get", "give", "lead", "leads", "drugx", "taking", "take",
                         "you", "me", "people", "does", "can", "will", "could", "would", "might",
                         "may", "common", "rare", "normal", "linked", "associated", "risk",
                         "chance", "chances", "percentage", "likely", "possible", "really", "actually",
                         "safe", "okay", "bad", "good", "taken", "using", "use", "medicine", "drug",
                         "tablet", "tablets", "pill", "pills", "anyone", "someone", "body", "my", "your"}
# words dropped from an effect phrase ('blood sugar' ≈ 'glucose', 'levels')
_OPTIONAL_EFFECT_WORDS = {"blood", "level", "levels", "plasma", "serum"}
# direction-preserving substitutes for a word inside the phrase
_PHRASE_SUBS = {
    "raise": ("increas", "elevat", "rais"), "raises": ("increas", "elevat"),
    "increase": ("increas", "elevat", "rais"), "lower": ("lower", "reduc", "decreas"),
    "lowers": ("lower", "reduc", "decreas"), "reduce": ("reduc", "lower", "decreas"),
    "sugar": ("glucose", "glycem", "glycaem"), "gain": ("gain", "increas"),
    "loss": ("loss", "lose", "decreas", "reduc"), "pressure": ("pressure",),
}


def _effect_patterns(words: List[str]) -> List[re.Pattern]:
    """Patterns for an effect phrase: all its words present (as 5-letter
    prefixes, any order: 'weight gain' ≈ 'gained weight'), the same with
    direction-preserving substitutes ('lower blood sugar' ≈ 'reduces glucose'),
    plus direction-neutral medical synonyms ('sleepy' → somnolence)."""
    words = [w.lower() for w in words if len(w) >= 3 and w.lower() not in _GENERIC_EFFECT_WORDS
             and w.lower() not in _OPTIONAL_EFFECT_WORDS]
    if not words:
        return []

    def piece(w: str) -> str:
        """One word as a regex: symptom words exactly ('sleepy' must not
        match 'sleeplessness'), others by their first 5 letters."""
        if w in _EFFECT_SYNONYMS:
            stem_ = w[:-1] if w.endswith("y") else w
            return rf"\b{re.escape(stem_)}(?:y|iness|ness|s|ed|ing)?\b"
        return rf"\b{re.escape(w[:5])}(?!less)\w*"
    variants = [[piece(w) for w in words]]
    for i, w in enumerate(words):
        for sub in _PHRASE_SUBS.get(w, ()):
            for base in list(variants):
                v = list(base)
                v[i] = rf"\b{re.escape(sub)}\w*"
                if v not in variants:
                    variants.append(v)
        if len(variants) > 12:
            break
    pats = []
    for v in variants:
        if len(v) == 1:
            pats.append(v[0])
        elif len(v) == 2:
            # a two-word effect must be a phrase, in either order, ≤3 words apart
            a, b = v
            gap = r"\W+(?:\w+\W+){0,3}?"
            pats.append(rf"{a}{gap}{b}|{b}{gap}{a}")
        else:
            pats.append("".join(rf"(?=.*{p})" for p in v))
    for w in words:
        for syn in _EFFECT_SYNONYMS.get(w, ()):
            pats.append(rf"\b{re.escape(syn)}\w*")
    ws = set(words)
    for verbs, nouns, terms in _DIRECTIONAL_TERMS:
        if ws & verbs and ws & nouns:
            pats += [rf"\b{re.escape(t)}\w*" for t in terms]
    return [re.compile(p, re.I | re.S) for p in pats]


# (direction words, subject words, the single medical term meaning both)
_DIRECTIONAL_TERMS = [
    ({"raise", "raises", "increase", "increases", "high", "spike"}, {"sugar", "glucose"},
     ("hyperglyc",)),
    ({"lower", "lowers", "reduce", "reduces", "low", "drop"}, {"sugar", "glucose"}, ("hypoglyc",)),
    ({"raise", "raises", "increase", "increases", "high"}, {"pressure"}, ("hypertens",)),
    ({"lower", "lowers", "reduce", "reduces", "low", "drop"}, {"pressure"}, ("hypotens",)),
    ({"raise", "raises", "increase", "increases", "high"}, {"potassium"}, ("hyperkal",)),
    ({"lower", "lowers", "reduce", "reduces", "low"}, {"potassium"}, ("hypokal",)),
    ({"lower", "lowers", "reduce", "reduces", "low"}, {"sodium"}, ("hyponatr",)),
    ({"raise", "raises", "increase", "fast", "faster"}, {"heart", "rate"}, ("tachycard",)),
    ({"lower", "slow", "slows", "slower"}, {"heart", "rate"}, ("bradycard",)),
]


def _question_words(p: Dict[str, Any], name: str = "") -> List[str]:
    """The content words of the question itself (not just the router's focus,
    which loses words swallowed by topic phrases — 'weight gain', 'make you
    sleepy', 'cross the placenta'), minus the drug's own name."""
    raw = (p.get("raw") or "").lower()
    ent = p.get("_entity") or {}
    drop = set()
    for s in (name, ent.get("text"), ent.get("display"), ent.get("query")):
        drop |= set(re.findall(r"[a-z0-9]+", (s or "").lower()))
    words = []
    for w in re.findall(r"[a-z][a-z\-]+", raw):
        if w in drop or len(w) < 3 or w in _GENERIC_EFFECT_WORDS:
            continue
        if cr is not None and w in cr.STOPWORDS:
            continue
        if w not in words:
            words.append(w)
    return words


def _negated(text: str, pats: List[re.Pattern]) -> bool:
    """True when the effect is mentioned under a negation ('does not cause
    weight gain', 'no evidence of drowsiness')."""
    for rx in pats:
        m = rx.search(text)
        if not m:
            continue
        # lookahead patterns match at 0 — find the first content word instead
        span_start = m.start()
        if m.end() == m.start():
            first = re.search(r"\b\w", text[m.start():])
            words = [p for p in re.findall(r"\\b(\w+)", rx.pattern)]
            hits = [re.search(rf"\b{re.escape(w)}", text, re.I) for w in words]
            hits = [h.start() for h in hits if h]
            span_start = min(hits) if hits else (first.start() if first else 0)
        if _NEG_BEFORE_RE.search(text[max(0, span_start - 60):span_start]):
            return True
    return False


_USE_WORDING_RE = re.compile(r"\bused (?:as|to|for|in)\b|\bindicated\b|\btreat\w*|\bhelps? (?:with|you)\b|"
                             r"\bsleep aid\b|\bfor the (?:relief|treatment|management)\b", re.I)
_DIRECTION_WORDS = {"raise", "raises", "increase", "increases", "lower", "lowers", "reduce", "reduces",
                    "decrease", "decreases", "lose", "losing", "high", "low", "spike", "drop", "thin",
                    "slow", "speed"}
_COUNTER_RE = re.compile(r"\b(?:counteract\w*|counter(?:s|ing)?|prevent\w*|reduc\w*|treat\w*|against|"
                         r"revers\w*|reliev\w*|lower\w*|decreas\w*|block\w*|inhibit\w*|suppress\w*|"
                         r"protect\w* (?:against|from)|without causing|attenuat\w*|mitigat\w*|"
                         r"limit\w*|offset\w*|minimi[sz]\w*)\b[^.;]{0,45}$", re.I)


def _counters(text: str, pats: List[re.Pattern]) -> bool:
    """True when the effect is named as something the drug works AGAINST
    ('counteracting the weight gain caused by…', 'prevents nausea')."""
    for rx in pats:
        m = rx.search(text)
        if not m:
            continue
        words = re.findall(r"\\b(\w+)", rx.pattern)
        pos = [h.start() for h in (re.search(rf"\b{re.escape(w)}", text, re.I) for w in words) if h]
        start = min(pos) if pos else m.start()
        if _COUNTER_RE.search(text[max(0, start - 60):start]):
            return True
    return False


def _verdict(text: str, quote: Optional[Tuple[str, str, str]] = None) -> List[Dict[str, Any]]:
    items = [(text, "", "")]
    if quote and quote[0]:
        items.append((f"“{_tidy(quote[0])[:320]}”", quote[1], quote[2]))
    return [_b("answer", items=items, label="Short answer")]


def _scan(ctx, pats: List[re.Pattern], topics=None):
    """Corpus docs matching any pattern → (doc, negated), label first."""
    out = []
    for d in _chat_corpus(ctx):
        if topics is not None and d["topic"] not in topics:
            continue
        if any(rx.search(d["text"]) for rx in pats):
            out.append((d, _negated(d["text"], pats)))
    out.sort(key=lambda x: (x[0]["src"] != "FDA label", len(x[0]["text"])))
    return out


def _q(d, pats: Optional[List[re.Pattern]] = None) -> Tuple[str, str, str]:
    """A source quote; with effect patterns, centred on the matched words so a
    long list ('most common side effects include: …') shows the right part."""
    text = d["text"]
    if pats and len(text) > 260:
        words = [w for rx in pats for w in re.findall(r"\\b(\w+)", rx.pattern)]
        pos = [h.start() for h in (re.search(rf"\b{re.escape(w)}", text, re.I) for w in words) if h]
        if pos:
            s = max(0, min(pos) - 110)
            e = min(len(text), min(pos) + 150)
            while s > 0 and text[s - 1] not in " .;•":
                s -= 1
            text = ("…" if s > 0 else "") + text[s:e].strip() + ("…" if e < len(d["text"]) else "")
    return (text, d["src"], d.get("url", ""))


def _effect_verdict(ctx, name, p):
    """'Does X cause Y?' — intended effect / listed on the label / reported
    elsewhere / sources say no / not mentioned."""
    focus = [w for w in _question_words(p, name) if w not in _OPTIONAL_EFFECT_WORDS]
    if not focus:
        return []
    effect = " ".join(focus)
    pats = _effect_patterns(focus)
    if not pats:
        return []
    hits = _scan(ctx, pats)
    # a 'uses' sentence about the effect can mean the drug PRODUCES it ('lowers
    # blood glucose') or COUNTERS it ('counteracting the weight gain caused by
    # antipsychotics') — the direction decides the verdict
    has_dir = bool(set(w.lower() for w in focus) & _DIRECTION_WORDS)
    uses_hits = [d for d, neg in hits if not neg and d["topic"] in ("uses", "mechanism")]
    opposes = [d for d in uses_hits if not has_dir and _counters(d["text"], pats)]
    # 'intended' needs a direction question ('does it lower blood sugar') or
    # real use wording ('used as a sleep aid') — a mechanism sentence that
    # merely mentions the effect ('produces minimal sedation') is not a use
    intended = [d for d in uses_hits if d not in opposes and (
        has_dir or (d["topic"] == "uses" and _USE_WORDING_RE.search(d["text"])))]
    # normal-dose side-effect evidence vs mentions only after an overdose
    se_topics = ("side_effects", "warnings", "children", "elderly", "pregnancy", "toxicity", "legal")
    rank = {"side_effects": 0, "warnings": 1, "pregnancy": 2, "elderly": 2, "children": 3, "legal": 3,
            "toxicity": 4}

    def strength(d):          # the side-effect list (with an incidence %) first
        return (rank.get(d["topic"], 6), "%" not in d["text"], len(d["text"]))

    def usable(d):
        return d["topic"] in se_topics and not _counters(d["text"], pats) and not re.search(
            r"\boverdos\w*|took (?:\d|an? (?:overdose|large))|too much|more than directed", d["text"], re.I)
    label_pos = sorted((d for d, neg in hits if not neg and d["src"] == "FDA label" and usable(d)),
                       key=strength)
    other_pos = sorted((d for d, neg in hits if not neg and d["src"] != "FDA label" and usable(d)),
                       key=strength)
    overdose_only = [d for d, neg in hits if not neg and (d["topic"] == "overdose" or re.search(
        r"\boverdos\w*|too much|more than directed", d["text"], re.I))]
    negs = [d for d, neg in hits if neg]
    # 'produces minimal sedation compared to…' is a RELATIVE statement (weak),
    # not a denial like 'does not cause sedation' (strong)
    weak_negs = [d for d in negs if re.search(r"\b(?:minimal|little|slight(?:ly)?|lower risk|compared|"
                                              r"than|less)\b", d["text"], re.I)
                 and not re.search(r"\b(?:not|no|never|without|doesn't|isn't|aren't|cannot)\b",
                                   d["text"], re.I)]
    negs = [d for d in negs if d not in weak_negs]
    e = _md_safe(effect)
    n = _md_safe(name)
    if other_pos and weak_negs and not negs and not label_pos:
        return _verdict(f"**Yes — reported as a side effect** in {other_pos[0]['src']}, though some "
                        "sources describe it as mild or less than with similar drugs.",
                        _q(other_pos[0], pats))
    if weak_negs and not (label_pos or other_pos or negs):
        return _verdict("**Uncommon or mild, according to the sources:**", _q(weak_negs[0], pats))
    if re.search(r"\bnon[- ]?(?:drowsy|sedating)\b", p.get("raw") or "", re.I):
        ev = (label_pos or other_pos or overdose_only or negs or [None])[0]
        if ev is None:
            return _verdict(f"**Drowsiness isn't mentioned** in {n}'s sources.")
        return _verdict("**What the sources say about drowsiness:**", _q(ev, pats))
    if opposes and not label_pos:
        return _verdict(f"**No — the sources describe {n} as working against this**, not causing it.",
                        _q(opposes[0], pats))
    if intended and (has_dir or not label_pos):
        use = [d for d in intended if d["topic"] == "uses"]
        if use:
            return _verdict(f"**Yes — that's an intended effect** — it's part of what {n} is used for.",
                            _q(use[0], pats))
        # a mechanism sentence: a known pharmacological effect, not the purpose
        return _verdict(f"**Yes — it's a known effect of how {n} works**, which can matter as a side "
                        "effect too.", _q(intended[0], pats))
    if label_pos:
        return _verdict(f"**Yes — {n}'s US FDA label lists it.**", _q(label_pos[0], pats))
    if other_pos and not negs:
        return _verdict(f"**Yes — reported as a side effect** in {other_pos[0]['src']} (it isn't in "
                        "the FDA label text I have).", _q(other_pos[0], pats))
    if overdose_only and not negs and not other_pos:
        return _verdict("**Mainly in overdose** — the sources mention it after taking too much:",
                        _q(overdose_only[0], pats))
    if negs and not other_pos:
        return _verdict(f"**Not usually — the sources describe {_md_safe(name)} as not causing "
                        f"{e}.**", _q(negs[0], pats))
    if negs and other_pos:
        return _verdict(f"**The sources differ on {e}** — some report it, some say it doesn't "
                        "happen. Ask a pharmacist about your situation.", _q(other_pos[0], pats))
    return _verdict(f"**Not mentioned** — {e} isn't listed in {_md_safe(name)}'s FDA label, "
                    "PubChem or Wikipedia text. That doesn't rule it out — ask a pharmacist.")


_AVOID_RE = re.compile(r"\b(?:avoid|do not|don't|should not|must not|not recommended|contraindicat\w*|"
                       r"disulfiram|severe|serious|dangerous|increase[sd]? the risk|may increase|"
                       r"potentiat\w*|enhance[sd]?|discontinue|abstain|refrain)\b", re.I)
_STRONG_AVOID_RE = re.compile(
    r"\b(?:avoid(?:ed)?|discontinue (?:the )?consumption|abstain|refrain|do not (?:drink|consume|take|use|"
    r"eat)|should not (?:drink|consume|be (?:taken|used|consumed))|must not|not be consumed|"
    r"contraindicated)\b", re.I)


def _interactant_verdict(ctx, name, p):
    target = (p.get("interactant") or "").strip()
    if not target:
        return []
    terms = _INTERACTANT_TERMS.get(target.lower(), [target])
    hits = _mentions(ctx, terms, topics={"interactions", "warnings", "side_effects", "dosing", "pk",
                                         "overdose"}, k=6)
    t = _md_safe(target)
    if not hits:
        return _verdict(f"**No warning found** about {t} in {_md_safe(name)}'s FDA label, PubChem "
                        "or Wikipedia text — that isn't proof it's safe. Ask a pharmacist.")
    # an explicit instruction about the substance beats a sentence that merely
    # contains a warning word ('…psychotic reactions with disulfiram…')
    def caution_score(h):
        s = h[0]
        sc = 3 if _STRONG_AVOID_RE.search(s) else (1 if _AVOID_RE.search(s) else 0)
        sc += 1 if re.search(r"beverage|drink|consum|intake", s, re.I) else 0
        return sc
    ranked = sorted(hits, key=lambda h: (-caution_score(h), h[1] != "FDA label", len(h[0])))
    caution = ranked[0] if caution_score(ranked[0]) > 0 else None
    if caution:
        where = "its FDA label" if caution[1] == "FDA label" else caution[1]
        return _verdict(f"**Caution — {where} warns about {t} with {_md_safe(name)}.**", caution)
    return _verdict(f"**The sources mention {t} with {_md_safe(name)}** (no explicit warning):",
                    hits[0])


def _class_member_verdict(ctx, name, p):
    slugs = p.get("class_slugs") or []
    phrase = p.get("class_phrase") or ""
    drug = ctx.get("drug_hit")
    if not slugs or not drug or not HAS_DRUG_LIBRARY:
        return []
    member = [s for s in slugs if s in (drug.get("classes") or [])]
    if member:
        cname = dl.class_name(member[0])
        return _verdict(f"**Yes — {_md_safe(name)} is classed as {_md_safe(cname)}** "
                        f"({dl.class_system(member[0])}).")
    own: List[str] = []
    for s in drug.get("classes") or []:
        cname = dl.class_name(s)
        if re.search(r"established pharmacologic|chemical structure|mechanism of action|physiologic effect|"
                     r"^drugs?$|^other\b", cname, re.I):
            continue
        if not any(cname.lower().rstrip("s") == o.lower().rstrip("s") for o in own):
            own.append(cname)
    own = own[:3]
    return _verdict(f"**No — {_md_safe(name)} isn't in the {_md_safe(phrase or dl.class_name(slugs[0]))} "
                    f"class** in WHO ATC / FDA classifications. It's classed as: "
                    f"{_md_safe(', '.join(own)) or 'not recorded'}.")


_RX_Q_RE = re.compile(r"prescription|over[- ]the[- ]counter|\botc\b|buy|available|availab|"
                      r"pharmacy|get (?:it|drugx)|sold", re.I)


def _availability_verdict(ctx, name, p):
    if not _RX_Q_RE.search(p.get("raw") or ""):
        return []
    market = ctx.get("market") or {}
    n = _md_safe(name)
    if ctx.get("region") == "SG" and HAS_SG_REGISTER:
        sg = market.get("sg")
        if not sg:
            return _verdict(f"**No HSA-registered {n} product found** — it may not be sold in "
                            "Singapore.")
        classes = sg.get("classes") or []
        if classes == ["POM"]:
            head = f"**Prescription-only in Singapore** — every registered {n} product is POM."
        elif "GSL" in classes:
            head = (f"**No prescription needed for some {n} products in Singapore** — "
                    "general-sale packs exist (larger packs/strengths may need one).")
        elif "P" in classes:
            head = (f"**Pharmacy-only in Singapore** — from a pharmacist without a prescription "
                    "(some strengths may be prescription-only).")
        else:
            head = f"**Singapore availability:** {sgr.availability_text(sg)}"
        return _verdict(head + f"  \n<small>{_md_safe(sgr.availability_text(sg))} · HSA register</small>")
    avail = _availability_label(market, ctx.get("label") or {})
    return _verdict(f"**{_md_safe(avail)}.**") if avail else []


_SCHEDULE_RE = re.compile(r"(?i:\bschedule)\s*(?:C-?)?(I{1,3}V?|IV|V|[1-5])\b|\bC-(II|III|IV|V)\b")


def _controlled_verdict(ctx, name, p):
    of = ctx.get("label") or {}
    if not of:
        return []
    ctrl, abuse = of.get("controlled_substance") or "", of.get("abuse") or ""
    n = _md_safe(name)
    m = _SCHEDULE_RE.search(ctrl) or _SCHEDULE_RE.search(abuse[:600])
    if m:
        sch = (m.group(1) or m.group(2)).upper()
        return _verdict(f"**Yes — {n} is a controlled substance in the US (Schedule {sch})**, "
                        "meaning it has recognised abuse/dependence potential.",
                        (_first_sentence(ctrl or abuse, 260), "FDA label", of.get("url", "")))
    if re.search(r"not (?:a )?controlled", ctrl + " " + abuse, re.I):
        extra = ""
        if re.search(r"dependen|withdraw|tolerance", abuse, re.I):
            extra = " The label does discuss dependence/withdrawal — see below."
        return _verdict(f"**No — {n} is not a controlled substance in the US.**{extra}")
    if abuse:
        return _verdict(f"**The label discusses abuse/dependence for {n}:**",
                        (_first_sentence(abuse, 260), "FDA label", of.get("url", "")))
    return _verdict(f"**No controlled-substance or abuse/dependence section** on {n}'s US label.")


def _uses_verdict(ctx, name, p):
    focus = [w for w in (p.get("focus") or []) if w not in ("drugx",)]
    if not focus:
        return []
    cond = " ".join(focus)
    words = focus + [s for w in focus for s in (cr.LAY_SYNONYMS.get(w, ()) if cr else ())
                     if " " not in s][:6]
    pats = [re.compile(rf"\b{re.escape(w[:5])}\w*", re.I) for w in words if len(w) >= 3]
    if not pats:
        return []
    of = ctx.get("label") or {}
    n, c = _md_safe(name), _md_safe(cond)
    for field in ("indications", "purpose"):
        for s in _split_sentences(of.get(field) or ""):
            if any(rx.search(s) for rx in pats):
                return _verdict(f"**Yes — it's covered by {n}'s FDA-approved uses.**",
                                (s, "FDA label", of.get("url", "")))
    ch = ctx.get("clin_chembl") or {}
    for ind in ch.get("indications") or []:
        if any(rx.search(ind.get("condition") or "") for rx in pats):
            if ind.get("phase") == 4:
                return _verdict(f"**Yes — approved for {_md_safe(ind['condition'])}** (ChEMBL).")
            return _verdict(f"**Investigated, not approved** — {_md_safe(ind['condition'])} has been "
                            f"studied (phase {ind.get('phase')} trials, ChEMBL) but isn't an approved use.")
    hits = [d for d, neg in _scan(ctx, pats, topics={"uses", "overview"}) if not neg]
    if hits:
        return _verdict(f"**Sometimes — {hits[0]['src']} mentions it as a use**, but it isn't among "
                        "the FDA-approved uses I have.", _q(hits[0]))
    return _verdict(f"**Not among {n}'s documented uses** in the FDA label, ChEMBL, PubChem or "
                    f"Wikipedia text for “{c}”.")


def _mention_verdict(ctx, name, p, topics, subject: str):
    """Generic yes/no from a mention with negation handling ('does it cross the
    placenta', 'is it a prodrug')."""
    focus = _question_words(p, name)
    pats = _effect_patterns(focus)
    if not pats:
        return []
    hits = _scan(ctx, pats, topics=topics)
    if hits:
        d, neg = hits[0]
        return _verdict("**No, according to the sources:**" if neg
                        else "**Yes, according to the sources:**", _q(d))
    # 'cleared by the liver OR the kidneys' — no sentence names both: report
    # what the sources say about either, without forcing a yes/no
    singles = [rx for w in focus if len(w) >= 4 for rx in _effect_patterns([w])]
    hits = _scan(ctx, singles, topics=topics)
    if hits:
        return _verdict("**What the sources say:**", _q(hits[0][0]))
    return _verdict(f"**Not stated** — {_md_safe(name)}'s sources don't say anything about "
                    f"{_md_safe(subject or ' '.join(focus))}.")


def _chem_verdict(ctx, name, p):
    raw = (p.get("raw") or "").lower()
    smiles = ctx.get("smiles") or ""
    if not smiles:
        return []
    core = largest_fragment_smiles(smiles) or smiles
    mol = Chem.MolFromSmiles(core)
    if mol is None:
        return []
    if re.search(r"chiral|stereo|enantiomer|optical|isomer", raw):
        try:
            centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        except Exception:  # noqa: BLE001
            centres = []
        n = len(centres)
        if n:
            return _verdict(f"**Yes — {n} stereocentre{'s' if n != 1 else ''}**, so it is chiral "
                            "(computed from the structure).  \n<small>Derived · RDKit</small>")
        return _verdict("**No — no stereocentres**, so it's achiral (computed from the structure)."
                        "  \n<small>Derived · RDKit</small>")
    if re.search(r"\b(?:acid|acidic|base|basic|alkaline|ioni[sz]\w*|charge|pka|neutral)\b", raw):
        ab = ci.infer_acid_base(core)
        acids = [c for c in ab["centres"] if c["role"] == "acid"]
        bases = [c for c in ab["centres"] if c["role"] == "base"]
        def desc(cs_):
            return ", ".join(f"{c['name'].lower()} (pKa ≈ {c['pka'][0]:g}–{c['pka'][1]:g})" for c in cs_)
        if acids and bases:
            kind = "Both acidic and basic (amphoteric)"
        elif acids:
            kind = "A (weak) acid"
        elif bases:
            kind = "A (weak) base"
        else:
            kind = "Essentially neutral — no ionisable acid/base groups recognised"
        q = ab.get("net_charge_estimate", 0)
        charge = ("it is mostly uncharged" if q == 0 else
                  f"it is mostly {'negatively' if q < 0 else 'positively'} charged (net ≈ {q:+d})")
        detail = "; ".join(x for x in (desc(acids), desc(bases)) if x)
        measured = [d for d in _chat_corpus(ctx) if d["topic"] == "properties"
                    and re.search(r"\bpka\b|dissociation", d["text"], re.I)]
        lines = [(f"**{kind}.**" + (f" Ionisable groups: {_md_safe(detail)}." if detail else "")
                  + f" At pH 7.4 {charge}.  \n<small>Inferred from functional groups — "
                  "Henderson–Hasselbalch reasoning, not a measurement</small>", "", "")]
        if measured:
            lines.append((f"Measured: {_tidy(measured[0]['text'])[:200]}", measured[0]["src"],
                          measured[0].get("url", "")))
        return [_b("answer", items=lines, label="Short answer")]
    if re.search(r"solub|dissolv|water", raw):
        measured = [d for d in _chat_corpus(ctx) if d["topic"] == "properties"
                    and re.search(r"solubility", d["text"], re.I)]
        if measured:
            return _verdict("**Measured solubility:**", _q(measured[0]))
        tend = ci.infer_solubility_tendency(core)
        if tend:
            return _verdict(f"**{_md_safe(tend['tendency'].capitalize())}** — reasoned from logP, "
                            "polarity and ionisation, not measured.  \n<small>Inferred</small>")
    return []


# ---- durations, doses, frequency, timing ----
_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:(?:to|-|–|and)\s*(\d+(?:\.\d+)?)\s*)?"
                     r"(hours?|hrs?|h\b|minutes?|mins?|days?|weeks?)", re.I)
_ONSET_RE = re.compile(r"\bonset of (?:action|effect|analgesia|response|activity)|\bwithin \d|"
                       r"\bbegins? to work|\bstarts? (?:to )?work|"
                       r"\btakes? (?:about |up to |several |a few |at least )?\d*\s*(?:-|to)?\s*\d*\s*"
                       r"(?:minutes|hours|days|weeks)|\bpeak (?:effect|plasma|serum|concentration)|"
                       r"\bfull (?:effect|benefit|therapeutic)|\bnoticeable\b", re.I)
_SYSTEM_RE = re.compile(r"system|body|stay|leave|clear|drug test|detect|urine|wear off|out of", re.I)
_CLINICAL_ONSET_RE = re.compile(
    r"\bonset of (?:action|effect|analgesia|response|activity)|"
    r"(?:may|can|might|will) take (?:several |a few |up to |about |at least )?[\d\-–to ]*(?:weeks|days)|"
    r"(?:within|after) (?:\d+|several|a few|one|two|three|four) (?:weeks|days|minutes|hours) of "
    r"(?:starting|treatment|therapy|use|taking)|(?:start|begin|starts|begins) (?:to )?(?:work|feel)|"
    r"feel (?:better|the effect)|full (?:effect|benefit|therapeutic effect)|therapeutic (?:effect|response)|"
    r"works? within|effects? (?:begin|start|appear|are felt)|within \d+\s*(?:minutes|min)\b", re.I)
_PK_PEAK_RE = re.compile(r"plasma|serum|C ?max|\bconcentration|bioavailab|AUC\b", re.I)
# washout periods ('within 2 weeks of stopping an MAOI') are not onset
_WASHOUT_RE = re.compile(r"\bMAOI|after stopping|of stopping|discontinu|washout", re.I)


def _duration_lead(ctx, name, p):
    raw = p.get("raw") or ""
    onset = "Onset" in (p.get("pk_facts") or []) or re.search(
        r"take\w* (?:to work|effect)|kick|start\w* work|how (?:fast|quickly|soon)", raw, re.I)
    if onset:
        cands = [d for d in _chat_corpus(ctx) if d["topic"] in ("pk", "dosing", "uses", "mechanism",
                                                                "overview", "warnings")
                 and _ONSET_RE.search(d["text"]) and _DUR_RE.search(d["text"])]

        # 'how long until it works' means the clinical effect (weeks for an
        # antidepressant), not when blood levels peak (hours)
        def onset_score(d):
            s = d["text"]
            sc = 3 if _CLINICAL_ONSET_RE.search(s) else 0
            sc -= 2 if _PK_PEAK_RE.search(s) else 0
            sc -= 4 if _WASHOUT_RE.search(s) else 0
            sc += 0.5 if d["src"] == "FDA label" else 0
            return sc
        cands = [d for d in cands if not _WASHOUT_RE.search(d["text"])]
        cands.sort(key=lambda d: (-onset_score(d), len(d["text"])))
        if cands and onset_score(cands[0]) > 0:
            return _verdict("**How soon it works:**", _q(cands[0]))
        peak = next((d for d in cands if _PK_PEAK_RE.search(d["text"]) or re.search(r"\bpeak|tmax", d["text"], re.I)),
                    None)
        if peak:
            return _verdict("**Time to peak blood levels** (when it starts working isn't stated):", _q(peak))
        return []
    items, _scope = _pk_items(ctx)
    facts, _rest = _pk_key_facts([(s, (src, u)) for s, src, u in items])
    hl = next(((s, payload) for lbl, s, payload in facts if lbl == "Half-life"), None)
    if not hl:
        return []
    s, (src, u) = hl
    blocks = _verdict("**Half-life:**", (s, src, u))
    if _SYSTEM_RE.search(raw):
        m = _DUR_RE.search(s)
        if m:
            lo = float(m.group(1))
            hi = float(m.group(2)) if m.group(2) else lo
            unit = m.group(3).lower()
            unit = {"h": "hours", "hr": "hours", "hrs": "hours", "min": "minutes",
                    "mins": "minutes"}.get(unit, unit if unit.endswith("s") else unit + "s")
            span = (f"{lo * 5:g}" if lo == hi else f"{lo * 5:g}–{hi * 5:g}")
            blocks[0]["items"].append(
                (f"Rule of thumb: a drug is ~97% eliminated after about 5 half-lives — here roughly "
                 f"**{span} {unit}**. Active metabolites, repeated doses and drug tests can differ.",
                 "calculated", ""))
    return blocks


_MAX_Q_RE = re.compile(r"\bmax|\bmost\b|too much|limit|exceed|upper|highest|24 hours|in a day|per day|"
                       r"a day\b", re.I)
_MAX_S_RE = re.compile(r"\bmaximum\b|not (?:to )?exceed|no more than|do not (?:take|use|give) more than|"
                       r"should not exceed|up to \d+(?:\.\d+)?\s*(?:mg|g|mcg|tablets?|capsules?|doses?)"
                       r"[^.]{0,30}(?:per|a|in|each|every)\s*(?:day|24)", re.I)
_DOSE_NUM_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g\b|ml|mL|units?|tablets?|capsules?|puffs?)", re.I)
_MAX_NUM_RE = re.compile(
    r"(?:not (?:to )?exceed|no more than|more than|maximum(?: (?:daily|total))?(?: dose)?(?: is| of)?|"
    r"up to)\s*\d+(?:\.\d+)?\s*(?:mg|g\b|mcg|grams?|tablets?|capsules?|doses?|caplets?|puffs?)"
    r"[^.;]{0,40}", re.I)
_FREQ_EXPLICIT_RE = re.compile(r"every \d+(?:\s*(?:to|-|–)\s*\d+)?\s*hours|once (?:a |per )?day|"
                               r"twice (?:a |per )?day|(?:one|two|three|four|1|2|3|4) times (?:a|per|each) day|"
                               r"once daily|twice daily|\bq\d+h\b|every (?:morning|evening|night)", re.I)
_FREQ_RE = re.compile(_FREQ_EXPLICIT_RE.pattern + r"|\bdaily\b|every day", re.I)
_CRUSH_RE = re.compile(
    r"(?:do not|don't|should not|must not|never|not to) (?:\w+ ){0,3}(?:crush|chew|split|break|cut|open)\w*|"
    r"swallow\w* (?:\w+ ){0,2}whole|(?:may|can|should) be (?:crushed|split|chewed|broken|cut|opened|"
    r"sprinkled|dissolved|halved)|open\w* (?:the )?capsules? and sprinkl\w*|sprinkl\w* (?:the )?(?:contents|"
    r"granules|pellets)", re.I)
_TIMING_RE = re.compile(r"\b(?:morning|evening|bedtime|at night|before (?:meals?|breakfast|bed|food)|"
                        r"after (?:meals?|food|eating)|with (?:food|meals?|a meal|milk)|empty stomach|"
                        r"same time each day|time of day)\b", re.I)


def _label_dose_sentences(ctx):
    of = ctx.get("label") or {}
    sents = [(s, "FDA label", of.get("url", "")) for s in _split_sentences(_tidy(of.get("dosage") or ""))
             if len(s) > 20 and not _LABEL_BOILERPLATE_RE.search(s)]
    sents += [(d["text"], d["src"], d.get("url", "")) for d in _chat_corpus(ctx)
              if d["topic"] == "dosing" and d.get("section", "").startswith(("Medication Guide",
                                                                            "Patient counseling"))]
    return sents


def _clause(text: str, rx: Optional[re.Pattern], before: int = 80, after: int = 150) -> str:
    """The part of a long label passage around the match (OTC 'Directions'
    panels are one unpunctuated run)."""
    text = _tidy(text)
    if len(text) <= 260 or rx is None:
        return text
    m = rx.search(text)
    if not m:
        return text[:260].rsplit(" ", 1)[0] + "…"
    s, e = max(0, m.start() - before), min(len(text), m.end() + after)
    while s > 0 and text[s - 1] not in " .;•":
        s -= 1
    while e < len(text) and text[e] not in " .;•":
        e += 1
    return ("…" if s > 0 else "") + text[s:e].strip(" ,;") + ("…" if e < len(text) else "")


def _dose_lead(ctx, name, p):
    if p.get("population"):
        return []                       # the dosing section answers for the population
    sents = _label_dose_sentences(ctx)
    if not sents:
        return []
    if _MAX_Q_RE.search(p.get("raw") or ""):
        # a stated number ('no more than 10 tablets in 24 hours') beats 'do not
        # take more than directed'
        hit = next((x for x in sents if _MAX_NUM_RE.search(x[0])), None)
        rx = _MAX_NUM_RE
        if not hit:
            hit, rx = next((x for x in sents if _MAX_S_RE.search(x[0])), None), _MAX_S_RE
        if hit:
            return _verdict("**Maximum (FDA label):**",
                            (_clause(hit[0], rx, before=40, after=60), hit[1], hit[2]))
    hit = next((x for x in sents if _DOSE_NUM_RE.search(x[0])), None)
    return (_verdict("**Usual dose (FDA label):**", (_clause(hit[0], _DOSE_NUM_RE), hit[1], hit[2]))
            if hit else [])


def _pattern_lead(ctx, p, rx: re.Pattern, title: str, topics=("dosing",),
                  primary: Optional[re.Pattern] = None, n: int = 1):
    """Best sentence(s) matching `rx` in the given topics — sentences matching
    the `primary` pattern first, label text before other sources."""
    cands = [(d["text"], d["src"], d.get("url", "")) for d in _chat_corpus(ctx)
             if d["topic"] in topics and rx.search(d["text"])]
    if not cands:
        return []
    def where(x):              # 'If you miss a dose, take it…' (match near the start) first
        m = primary.search(x[0]) if primary else None
        return m.start() if m else 9999
    cands.sort(key=lambda x: (where(x) == 9999, x[1] != "FDA label", where(x) > 40, len(x[0])))
    out = _verdict(title, (_clause(cands[0][0], primary or rx), cands[0][1], cands[0][2]))
    seen = {_tidy(cands[0][0])[:60]}
    for c in cands[1:]:
        if len(out[0]["items"]) > n:
            break
        if _tidy(c[0])[:60] in seen or (primary and primary.search(c[0]) and n == 1):
            continue
        seen.add(_tidy(c[0])[:60])
        out[0]["items"].append((f"“{_clause(c[0], rx)[:320]}”", c[1], c[2]))
    return out


_PRACTICAL = [
    # (question cue, sentence pattern, main-instruction pattern, title, topics, quotes)
    (re.compile(r"miss|forg[eo]t|skip|double up|late dose", re.I),
     re.compile(r"miss(?:ed)? (?:a |your |the )?dose|forget|forgot|skip|double (?:the )?dose|two doses", re.I),
     re.compile(r"miss(?:ed)? (?:a |your |the )?dose|if you forget|forget to take", re.I),
     "**If you miss a dose:**", ("dosing", "warnings"), 2),
    (re.compile(r"\bcrush|\bsplit|\bchew|cut (?:\w+ ){0,2}in half|\bbreak (?:it|the|them)\b|"
                r"\bopen (?:it|the|them|\w+ capsules?)\b|\bswallow|\bhalve|dissolve", re.I),
     _CRUSH_RE, _CRUSH_RE, "**How to take it:**", ("dosing", "warnings"), 1),
    (re.compile(r"\bfood|meal|empty stomach|breakfast|\beat\b", re.I),
     re.compile(r"\bfood\b|\bmeals?\b|empty stomach|\bfed\b|fasting|with milk", re.I),
     re.compile(r"with (?:or without )?food|with meals?|empty stomach|without regard to (?:food|meals)|"
                r"(?:before|after) (?:meals?|food|breakfast)|high[- ]fat meal", re.I),
     "**Food:**", ("dosing", "pk", "interactions"), 1),
    (re.compile(r"\bdriv|machiner", re.I),
     re.compile(r"\bdriv|machiner|hazardous (?:tasks|activities)|alertness", re.I),
     re.compile(r"\bdriv\w*|operat\w* (?:heavy )?machinery|hazardous", re.I),
     "**Driving & machinery:**", ("warnings", "side_effects", "interactions", "dosing"), 1),
    (re.compile(r"\bstop|taper|come off|coming off|wean|cold turkey|quit|abrupt|discontinu", re.I),
     re.compile(r"abrupt|discontinu|stop(?:ping)? (?:taking|treatment|therapy)|taper|gradual|"
                r"withdrawal|rebound", re.I),
     re.compile(r"(?:do not|don't) stop|abrupt\w*|taper\w*|gradual\w* (?:reduc|decreas|withdraw)|"
                r"discontinu\w*", re.I),
     "**Stopping:**", ("warnings", "legal", "dosing"), 2),
]


def _practical_lead(ctx, name, p):
    raw = p.get("raw") or ""
    for cue, rx, primary, title, topics, n in _PRACTICAL:
        if not cue.search(raw):
            continue
        lead = _pattern_lead(ctx, p, rx, title, topics, primary=primary, n=n)
        if lead:
            return lead
        if rx is _CRUSH_RE:
            return _verdict(f"**No crushing/splitting instructions on {_md_safe(name)}'s label.** Ask a "
                            "pharmacist first — tablets marked ER, XR, SR (extended-release) or "
                            "enteric-coated usually must not be crushed or chewed.")
    return []


def _allergy_verdict(ctx, name, p):
    allergen = (p.get("allergen") or "").strip()
    n = _md_safe(name)
    if not allergen:
        return []
    a = _md_safe(allergen)
    drug = ctx.get("drug_hit") or {}
    own = [dl.class_name(s).lower() for s in drug.get("classes") or []] if HAS_DRUG_LIBRARY else []
    stem_a = allergen.lower().rstrip("s")[:6]
    same_class = [c for c in own if stem_a and stem_a in c]
    base = allergen.rstrip("s")
    hits = [h for h in _mentions(ctx, [allergen, base, base + "s"],
                                 topics={"warnings", "side_effects"}, k=4)
            if re.search(r"allerg|hypersensitiv|cross|reaction", h[0], re.I)]
    if same_class:
        return _verdict(f"**⚠️ {n} is itself in the {_md_safe(same_class[0])} class** — with a "
                        f"{a} allergy, don't take it unless a doctor says so.",
                        hits[0] if hits else None)
    if hits:
        return _verdict(f"**⚠️ Its label mentions {a} allergy** — check with a doctor or pharmacist "
                        "before taking it.", hits[0])
    return _verdict(f"**Its label doesn't mention {a}**, but allergies can cross-react within a "
                    "drug family — ask a pharmacist or doctor before taking it.")


_INGREDIENT_Q = [
    ("lactose", re.compile(r"lactose", re.I)), ("gelatin", re.compile(r"gelatin", re.I)),
    ("gluten", re.compile(r"gluten|wheat|starch \(wheat\)", re.I)),
    ("sugar", re.compile(r"sucrose|sugar|glucose|dextrose|sorbitol", re.I)),
    ("alcohol", re.compile(r"\balcohol\b|ethanol", re.I)),
    ("dyes", re.compile(r"\bFD&C\b|\bD&C\b|dye|iron oxide|titanium dioxide|lake\b", re.I)),
]


def _ingredients_verdict(ctx, name, p):
    raw = (p.get("raw") or "").lower()
    of = ctx.get("label") or {}
    inact = of.get("inactive_ingredients") or ""
    if not inact:
        mg = next((b for h, b, tp in medguide_sections(of.get("medguide") or "") if tp == "ingredients"), "")
        inact = mg
    n = _md_safe(name)
    if not inact:
        return _verdict(f"**No inactive-ingredient list** in the US label data I have for {n}. "
                        "Check your product's leaflet — ingredients vary by brand.")
    tail = "  \n<small>For this US product — fillers differ between brands and countries; check your pack.</small>"
    for word, rx in _INGREDIENT_Q:
        if word in raw or (word == "dyes" and re.search(r"dye|colou?r", raw)):
            m = rx.search(inact)
            if m:
                return _verdict(f"**Yes — this product lists {_md_safe(m.group(0))}** among its "
                                f"inactive ingredients.{tail}", (inact[:300], "FDA label", of.get("url", "")))
            return _verdict(f"**Not listed** among this product's inactive ingredients.{tail}",
                            (inact[:300], "FDA label", of.get("url", "")))
    if re.search(r"halal|kosher|vegan|vegetarian|animal|pork|porcine|bovine", raw):
        animal = re.findall(r"gelatin\w*|lactose|carmine|shellac|stearic acid|magnesium stearate", inact, re.I)
        if animal:
            return _verdict(f"**Not stated on the label.** This product contains "
                            f"{_md_safe(', '.join(dict.fromkeys(a.lower() for a in animal)))}, which can be "
                            f"animal-derived — ask the pharmacist or manufacturer.{tail}")
        return _verdict(f"**Not stated on the label**, and no commonly animal-derived ingredient is "
                        f"listed — certification isn't recorded.{tail}")
    return _verdict("**Inactive ingredients (FDA label):**", (inact[:400], "FDA label", of.get("url", "")))


def direct_answer(ctx, name, p) -> List[Dict[str, Any]]:
    """The 'Short answer' that opens a reply, chosen by the question's form and
    primary topic. Empty when nothing direct can be said."""
    intents = p.get("intents") or []
    primary = intents[0] if intents else ""
    form = p.get("form") or ""
    try:
        if p.get("allergen") and primary == "warnings":
            return _allergy_verdict(ctx, name, p)
        if primary == "ingredients":
            return _ingredients_verdict(ctx, name, p)
        if primary == "cost":
            return []                                   # the cost section says it
        if primary == "interactions" and p.get("interactant"):
            return _interactant_verdict(ctx, name, p)
        if primary in ("dosing", "warnings", "interactions", "legal"):
            lead = _practical_lead(ctx, name, p)
            if lead:
                return lead
        if form == "yes_no":
            if primary == "side_effects" or (
                    primary == "mechanism" and set(_question_words(p, name)) & _DIRECTION_WORDS):
                return _effect_verdict(ctx, name, p)
            if primary == "interactions" and p.get("interactant"):
                return _interactant_verdict(ctx, name, p)
            if primary == "class":
                return _class_member_verdict(ctx, name, p)
            if primary == "brands":
                return _availability_verdict(ctx, name, p)
            if primary == "legal":
                return _controlled_verdict(ctx, name, p)
            if primary == "uses":
                return _uses_verdict(ctx, name, p)
            if primary == "pk":
                return _mention_verdict(ctx, name, p, {"pk", "mechanism", "pregnancy", "overview"}, "")
            if primary in ("properties", "structure"):
                return _chem_verdict(ctx, name, p)
        if primary in ("properties", "structure") and form in ("what", "statement", "which"):
            return _chem_verdict(ctx, name, p)
        if primary == "brands" and _RX_Q_RE.search(p.get("raw") or ""):
            return _availability_verdict(ctx, name, p)
        if form == "how_long" and primary == "pk":
            return _duration_lead(ctx, name, p)
        if form == "how_much" and primary == "dosing":
            return _dose_lead(ctx, name, p)
        if form == "how_often" and primary == "dosing":
            return _pattern_lead(ctx, p, _FREQ_RE, "**How often (label):**", primary=_FREQ_EXPLICIT_RE)
        if form == "when" and primary == "dosing":
            return _pattern_lead(ctx, p, _TIMING_RE, "**When to take it:**")
        if form == "how_common" and primary == "side_effects":
            return _effect_verdict(ctx, name, p)
    except Exception:  # noqa: BLE001 — a short answer must never break the reply
        return []
    return []


def _ans_cost(ctx, name, p):
    blocks = [_b("info", v=f"I don't have price information for {_md_safe(name)} — prices depend on "
                           "the brand, pack size, clinic/pharmacy and any subsidy. Ask the pharmacy or "
                           "your clinic.")]
    sg = (ctx.get("market") or {}).get("sg")
    if ctx.get("region") == "SG" and sg and HAS_SG_REGISTER:
        blocks.append(_b("md", v=f"**Singapore availability:** {_md_safe(sgr.availability_text(sg))} "
                                 "<small>· HSA register</small>", html=True))
    return blocks


def _ans_ingredients(ctx, name, p):
    of = ctx.get("label") or {}
    items = []
    if of.get("inactive_ingredients"):
        items.append((f"**Inactive ingredients:** {of['inactive_ingredients'][:600]}", "FDA label",
                      of.get("url", "")))
    for head, body, tp in medguide_sections(of.get("medguide") or ""):
        if tp == "ingredients":
            items.append((body[:600], "FDA label", of.get("url", "")))
            break
    if not items:
        return [_b("info", v=f"No ingredient list for {_md_safe(name)} in the US label data I have.")]
    blocks = _pts(items, title="", cap=2)
    blocks.append(_b("caption", v="From one US product's label — inactive ingredients differ between "
                                  "brands and countries, so check your own pack."))
    return blocks


_INTENT_BUILDERS = {
    "overview": _ans_overview, "mechanism": _ans_mechanism, "pk": _ans_pk,
    "uses": _ans_uses, "dosing": _ans_dosing, "side_effects": _ans_side_effects,
    "warnings": _ans_warnings, "interactions": _ans_interactions,
    "pregnancy": _ans_pregnancy, "population": _ans_population, "overdose": _ans_overdose,
    "brands": _ans_brands, "class": _ans_class, "alternatives": _ans_alternatives,
    "identifiers": _ans_identifiers, "structure": _ans_structure,
    "properties": _ans_properties, "spectra": _ans_spectra,
    "history": _ans_topic({"history"}, "History", _history_extra,
                          search_terms=("discovered", "discovery", "history", "invented",
                                        "first synthesized", "developed", "patented")),
    "synthesis": _ans_topic({"synthesis"}, "How it's made",
                            search_terms=("synthesis", "synthesized", "manufactured", "produced",
                                          "prepared", "reaction")),
    "legal": _ans_topic({"legal"}, "Legal status & misuse (Wikipedia / PubChem)", _legal_extra,
                        search_terms=("legal", "controlled", "schedule", "abuse", "recreational"),
                        skip_label=True),
    "storage": _ans_topic({"storage"}, "Stability & storage", _storage_extra,
                          search_terms=("store", "storage", "stable", "refrigerate"),
                          skip_label=True),
    "cost": _ans_cost, "ingredients": _ans_ingredients,
    "search": _ans_search,
}


def _intent_heading(intent: str, name: str) -> str:
    icon = _INTENT_ICON.get(intent, "•")
    titles = {
        "overview": f"{name} at a glance", "mechanism": f"How {name} works",
        "pk": f"{name} in the body (pharmacokinetics)", "uses": f"What {name} is used for",
        "dosing": f"{name} dosing", "side_effects": f"Side effects of {name}",
        "warnings": f"Warnings — who shouldn't take {name}",
        "interactions": f"{name} interactions",
        "pregnancy": f"{name} in pregnancy & breastfeeding",
        "population": f"{name} in children & older adults",
        "overdose": f"{name} overdose & toxicity", "brands": f"{name} brands & availability",
        "class": f"What kind of drug is {name}?", "alternatives": f"Drugs similar to {name}",
        "identifiers": f"{name} identifiers", "structure": f"Structure of {name}",
        "properties": f"Properties of {name}", "spectra": f"Spectra of {name}",
        "history": f"History of {name}", "synthesis": f"How {name} is made",
        "legal": f"{name}: abuse & legal status", "storage": f"Storing {name}",
        "cost": f"Cost of {name}", "ingredients": f"What's in {name}",
        "search": f"{name}: from the record",
    }
    return f"#### {icon} {_md_safe(titles.get(intent, name))}"


def _fallback_blocks(ctx, name, p, intent) -> List[Dict[str, Any]]:
    """When an intent's builder finds nothing: the closest sentences in the
    whole record, else an honest 'not found'."""
    title = cr.INTENT_TITLES.get(intent, intent).lower() if cr else intent
    terms = list(p.get("focus") or []) + [w for w in re.findall(r"[a-z]{4,}", title)]
    rel = _relevant(ctx, terms, k=4)
    if not _is_med(ctx) and intent in ("dosing", "side_effects", "warnings", "interactions",
                                       "pregnancy", "population", "uses", "brands"):
        msg = (f"{name} isn't a recognised medicine, so there's no {title} information "
               "(FDA label / drug databases) for it.")
    else:
        msg = f"I couldn't find {title} information for {name} in the sources I use."
    blocks = [_b("info", v=msg)]
    blocks += _pts(rel, title="Closest matches in the record", cap=4)
    if _is_med(ctx):
        blocks += _suggest_blocks(name, skip_topics=(intent,), title="Other things I can tell you:")
    return blocks


def _chat_name(ctx: Dict[str, Any], entity: Dict[str, Any]) -> str:
    kind = entity.get("kind")
    if kind in ("alias", "library") and entity.get("display"):
        return entity["display"]
    if kind == "brand" and entity.get("ingredients"):
        # 'Panadol' → 'Paracetamol' (the brand itself is named in a caption)
        ing = re.sub(r"\s+", " ", cr._SALT_WORDS_RE.sub("", entity["ingredients"][0].lower())).strip()
        if ing:
            return ing[:1].upper() + ing[1:]
    if (ctx.get("drug_hit") or {}).get("name"):
        return ctx["drug_hit"]["name"]
    if kind == "fuzzy" and entity.get("corrected_display"):
        return entity["corrected_display"]
    name = ctx.get("compound_name") or entity.get("display") or "this compound"
    if kind == "smiles" and (not name or name == entity.get("query")):
        return "this compound"
    return name[:1].upper() + name[1:] if len(name) < 60 else name[:57] + "…"


def _answer_for(ctx, entity, p, opts, compact=False) -> List[Dict[str, Any]]:
    name = _chat_name(ctx, entity)
    p = dict(p, _entity=entity, _compact=compact)
    blocks: List[Dict[str, Any]] = []
    # the direct answer to the question as asked (verdict / duration / dose)
    blocks += direct_answer(ctx, name, p)
    for intent in p["intents"]:
        fn = _INTENT_BUILDERS.get(intent, _ans_search)
        try:
            part = fn(ctx, name, p)
        except Exception as e:  # noqa: BLE001 — degrade per intent, never the whole answer
            part = [_b("caption", v=f"⚠️ Couldn't build this part of the answer ({type(e).__name__}: {e}).")]
            if opts.get("debug_mode"):
                import traceback
                part.append(_b("code", v=traceback.format_exc()))
        if not part:
            part = _fallback_blocks(ctx, name, p, intent)
        if intent == "mechanism" and re.search(r"pharmacodynamic", p.get("raw") or "", re.I):
            blocks.append(_b("md", v=f"#### 🧬 Pharmacodynamics of {_md_safe(name)}"))
        elif intent != "overview" or len(p["intents"]) > 1 or compact:
            blocks.append(_b("md", v=_intent_heading(intent, name)))
        blocks += part
    return blocks


def _entity_ref(entity: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> str:
    """How to name this compound in a follow-up question so the router finds it again."""
    kind = entity.get("kind")
    if kind == "fuzzy":
        return entity.get("resolved_as") or entity.get("corrected_display") or entity.get("query")
    if kind == "brand":
        return entity.get("display") or entity.get("query")
    if kind == "smiles":
        # a SMILES that turned out to be a known compound is asked about by name
        if ctx and (ctx.get("drug_hit") or {}).get("name"):
            return ctx["drug_hit"]["name"]
        nm = (ctx or {}).get("compound_name") or ""
        if nm and _name_like_synonym(nm) and len(nm) < 40:
            return nm
        return entity.get("query")
    return entity.get("display") or entity.get("query") or ""


def _followups(ref: str, ctx: Dict[str, Any], asked: List[str]) -> List[Tuple[str, str]]:
    """Next questions as (short button label, full question)."""
    if _is_med(ctx):
        pool = [("mechanism", "🧬 How it works", f"How does {ref} work?"),
                ("side_effects", "⚠️ Side effects", f"Side effects of {ref}"),
                ("dosing", "📏 Dosing", f"{ref} dosage"),
                ("interactions", "🔀 Interactions", f"{ref} interactions"),
                ("pk", "⏱ Half-life", f"Half-life of {ref}"),
                ("pregnancy", "🤰 Pregnancy", f"Is {ref} safe in pregnancy?"),
                ("structure", "🧪 Structure", f"Structure of {ref}"),
                ("brands", "🏷 Brands", f"Brands of {ref}"),
                ("alternatives", "🔁 Similar drugs", f"Drugs similar to {ref}")]
        if not ctx.get("label"):
            pool = [x for x in pool if x[0] not in ("dosing", "pregnancy")]
    else:
        pool = [("structure", "🧪 Structure", f"Structure of {ref}"),
                ("properties", "📊 Properties", f"Properties of {ref}"),
                ("overdose", "☠️ Toxicity", f"How toxic is {ref}?"),
                ("uses", "🎯 Uses", f"What is {ref} used for?"),
                ("identifiers", "🆔 Identifiers", f"CAS number of {ref}"),
                ("history", "📜 History", f"History of {ref}")]
    if ctx.get("name_only"):
        pool = [x for x in pool if x[0] not in ("structure", "properties", "identifiers")]
    return [(label, q) for intent, label, q in pool if intent not in asked][:5]


def _sources_caption(ctx: Dict[str, Any]) -> str:
    links = []
    if ctx.get("label", {}).get("url"):
        links.append(f"[FDA label (DailyMed)]({ctx['label']['url']})")
    ch = ctx.get("clin_chembl") or {}
    if ch.get("url"):
        links.append(f"[ChEMBL]({ch['url']})")
    if ctx.get("cid"):
        links.append(f"[PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{ctx['cid']})")
    if (ctx.get("wiki") or {}).get("url"):
        links.append(f"[Wikipedia]({ctx['wiki']['url']}) (CC BY-SA)")
    if (ctx.get("market") or {}).get("sg"):
        links.append("HSA register")
    return ("Sources: " + " · ".join(links)) if links else ""


_CHAT_CHEMBL_BUDGET_S = 4.0     # chat answers don't wait longer than this for ChEMBL


def _progress(opts: Dict[str, Any], msg: str):
    """Update the 'working…' status line above the answer being built."""
    fn = opts.get("_progress")
    if fn:
        try:
            fn(msg)
        except Exception:  # noqa: BLE001 — cosmetic only
            pass


def _ctx_cache() -> Dict[Any, Any]:
    return st.session_state.setdefault("_chat_ctx_cache", {})


def _load_rank(level: str, extras: bool) -> int:
    """Order of 'how much was loaded': struct < chem < full < full + FAERS/market."""
    return _LOAD_LEVELS[level] * 2 + (1 if extras and level == "full" else 0)


# Question types that use the slow optional data (FAERS counts, US market counts).
_EXTRAS_INTENTS = {"overview", "side_effects", "brands", "search"}


def _chat_load(entity: Dict[str, Any], opts: Dict[str, Any], level: str,
               extras: bool = True) -> Dict[str, Any]:
    """Resolve one recognised entity and load its data (memoised per session,
    so follow-up questions about the same compound are instant).
    Returns {"ctx", "notes": [blocks], "query", "others": [ingredients]} or
    {"error": str, "suggestions": [names]}."""
    region, strip = opts["region"], opts["strip_salts"]
    notes: List[Dict[str, Any]] = []
    kind = entity.get("kind")
    query = (entity.get("query") or entity.get("display") or "").strip()
    display = entity.get("display") or query
    others: List[str] = []
    if kind == "brand" and entity.get("ingredients"):
        ings = entity["ingredients"]
        notes.append(_b("caption", v=f"**{_md_safe(display)}** is a Singapore brand of "
                                     f"{_md_safe(' + '.join(ings))}"
                                     + (f" — answering for **{_md_safe(ings[0])}**." if len(ings) > 1 else ".")))
        others = ings[1:]
    elif kind == "alias" and entity.get("alias_of") and \
            entity["alias_of"].lower() != display.lower():
        notes.append(_b("caption", v=f"{_md_safe(display)} is also known as {_md_safe(entity['alias_of'])}."))
    elif kind == "fuzzy":
        original = entity.get("original") or ""
        use_original = False
        if not entity.get("certain") and original:
            try:
                use_original = bool(cached_resolve_name(original, False).get("candidates"))
            except Exception:  # noqa: BLE001
                use_original = False
        if use_original:
            query, display = original, original
        else:
            notes.append(_b("caption", v=f"Interpreting “{_md_safe(original)}” as "
                                         f"**{_md_safe(entity.get('corrected_display') or query)}**."))
        entity["resolved_as"] = display

    rank = _load_rank(level, extras)
    cache = _ctx_cache()
    key = (query.lower(), region, bool(strip))
    hit = cache.get(key)
    if hit and hit.get("expires", 0) < time.time():
        cache.pop(key, None)
        hit = None
    if hit and hit["rank"] >= rank:
        cache[key] = cache.pop(key)          # LRU refresh
        return {"ctx": hit["ctx"], "notes": notes + hit["notes"], "query": hit["query"],
                "others": others or hit["others"]}

    _progress(opts, f"Finding “{display}”…")
    res = resolve_input(query, region)
    drug = find_library_drug(query, display)
    unresolved = bool(res["error"] or not res["candidates"])
    pubchem_down = unresolved and cs.host_failing(cs.PUBCHEM_BASE)
    if unresolved and _network_down() and not (drug or kind in ("library", "alias", "brand")):
        # can't tell "unknown name" from "no network" — say so, answer offline
        return {"offline": True, "blocks": notes + _offline_blocks(query, display, drug, opts)}
    if unresolved:
        if drug or kind in ("library", "alias", "brand"):
            _progress(opts, f"Reading the FDA label and Wikipedia for {display}…")
            ctx = load_name_only(query, region=region, drug=drug)
            if ctx.get("label") or (ctx.get("wiki") or {}).get("extract") or ctx.get("grouped"):
                why = ("PubChem isn't responding right now, so the structure and PubChem's "
                       "record are missing" if pubchem_down else
                       f"No small-molecule structure found for {_md_safe(display)} (it may be a "
                       "biologic)")
                n2 = [_b("caption", v=f"{why} — answering from the FDA label and Wikipedia.")]
                if pubchem_down:          # don't remember a degraded answer for long
                    _ctx_store(key, rank=_load_rank("full", True), ctx=ctx, notes=n2,
                               query=query, others=others, expires=time.time() + 60)
                    return {"ctx": ctx, "notes": notes + n2, "query": query, "others": others}
                _ctx_store(key, rank=_load_rank("full", True), ctx=ctx, notes=n2,
                           query=query, others=others,
                           expires=time.time() + 6 * 3600)
                return {"ctx": ctx, "notes": notes + n2, "query": query, "others": others}
        if _network_down():
            return {"offline": True, "blocks": notes + _offline_blocks(query, display, drug, opts)}
        sugg = [cr.lexicon_entry(k)["display"] for k, _d in cr.fuzzy_candidates(query)
                if cr.lexicon_entry(k)] if cr else []
        return {"error": f"I couldn't identify **{_md_safe(display)}** — it isn't in the Drug "
                         "Library, PubChem, OPSIN, CIR or the brand registers.",
                "suggestions": sugg}
    cands = res["candidates"]
    if res["brand"]:
        ings = res["brand"]["ingredients"]
        notes.append(_b("caption", v=f"**{_md_safe(display)}** is a brand name "
                                     f"({res['brand']['source']}) — active ingredient"
                                     f"{'s' if len(ings) > 1 else ''}: **{_md_safe(' + '.join(ings))}**"
                                     + (" — answering for the first." if len(ings) > 1 else ".")))
        others = others or ings[1:]
    _progress(opts, {"struct": f"Drawing {display}…",
                     "chem": f"Reading PubChem's record for {display}…"}.get(
        level, f"Reading the FDA label, PubChem, ChEMBL and Wikipedia for {display}…"))
    # The chat waits less for ChEMBL (the slowest, most erratic source) and
    # skips its slow name-search recovery: the FDA label, PubChem and
    # Wikipedia already carry the mechanism text.
    ctx = load_compound(query, cands[0], res["source"], region=region, strip_salts=strip,
                        include_rdkit=opts.get("include_rdkit", True), level=level,
                        chembl_budget=_CHAT_CHEMBL_BUDGET_S, recover_tautomer=False,
                        extras=extras)
    if ctx.get("error"):
        if drug:
            ctx = load_name_only(query, region=region, drug=drug)
        else:
            return {"error": f"Found **{_md_safe(display)}** but couldn't get its structure.",
                    "suggestions": []}
    # A result loaded while a source was failing is kept only briefly, so a
    # follow-up within a minute is instant but a later one retries the source.
    _ctx_store(key, rank=_load_rank(ctx.get("level", level), extras), ctx=ctx,
               notes=notes[-1:] if res["brand"] else [], query=query, others=others,
               expires=time.time() + (60 if _sources_down(ctx) else 6 * 3600))
    return {"ctx": ctx, "notes": notes, "query": query, "others": others}


_SOURCE_HOSTS = (("PubChem", cs.PUGVIEW_BASE), ("ChEMBL", cs.CHEMBL_BASE),
                 ("FDA labels", cs.OPENFDA_LABEL), ("RxNorm", cs.RXNAV_BASE),
                 ("Wikipedia", cs.WIKI_API))


def _sources_down(ctx: Dict[str, Any]) -> List[str]:
    """Online sources that failed / timed out while this compound loaded."""
    down = [nm for nm, url in _SOURCE_HOSTS if cs.host_failing(url)]
    if ctx.get("chembl_timed_out") and "ChEMBL" not in down:
        down.append("ChEMBL")
    return down


def _network_down() -> bool:
    """PubChem (the resolver) failing, or every source failing at once."""
    return cs.host_failing(cs.PUBCHEM_BASE) or all(cs.host_failing(u) for _n, u in _SOURCE_HOSTS)


def _offline_blocks(query: str, display: str, drug, opts) -> List[Dict[str, Any]]:
    """What can be said with no network: the local Drug Library (classes,
    one-line description) and the Singapore HSA register snapshot."""
    blocks = [_b("warning", v="I can't reach PubChem right now (no internet connection, or the "
                              "service is down), so this is only what I know offline. Ask again "
                              "in a minute for the full answer.")]
    if drug:
        blocks.append(_b("md", v=f"**{_md_safe(drug['name'])}**"
                                 + (f" — {_md_safe(drug['blurb'])}" if drug.get("blurb") else "")))
        blocks += _pts(_class_lines(drug), title="Drug classes (Drug Library)", cap=4,
                       more="All classes")
    if HAS_SG_REGISTER:
        summ = sgr.lookup(*(n for n in ((drug or {}).get("query"), query, display) if n))
        if summ:
            blocks.append(_b("md", v=f"**Brands in Singapore:** {_md_safe(', '.join(summ['brands'][:12]))}"))
            blocks.append(_b("md", v=f"**Availability (HSA):** {_md_safe(sgr.availability_text(summ))}"))
    if len(blocks) == 1:
        blocks.append(_b("caption", v=f"“{_md_safe(display)}” isn't in the offline Drug Library or "
                                      "HSA register."))
    return blocks


def _ctx_store(key, **entry):
    cache = _ctx_cache()
    cache.pop(key, None)
    cache[key] = entry
    while len(cache) > _CHAT_CTX_CACHE:
        cache.pop(next(iter(cache)))


def _level_for(intents: List[str]) -> str:
    rank = max((_LOAD_LEVELS[_INTENT_LEVEL.get(i, "full")] for i in intents), default=2)
    return {0: "struct", 1: "chem", 2: "full"}[rank]


def _replace_entity(raw: str, entity: Dict[str, Any], new: str) -> str:
    span = entity.get("span")
    if span and 0 <= span[0] < span[1] <= len(raw):
        return raw[:span[0]] + new + raw[span[1]:]
    return f"{raw} ({new})"


def _answer_entities(p: Dict[str, Any], opts: Dict[str, Any]
                     ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    blocks: List[Dict[str, Any]] = []
    for note in p.get("notes") or []:
        if note.startswith("Interpreting"):
            continue                      # said once, in _chat_load
        if note == "same question as before":
            blocks.append(_b("caption", v="(same question as before)"))
    ents = p["entities"]
    intents = p["intents"]
    level = _level_for(intents)
    if p.get("same_as"):
        a, b = p["same_as"][:2]
        kinds = (list(p["same_as"]) + ["", ""])[2:4]
        if "brand" in kinds:
            brand, generic = (a, b) if kinds[0] == "brand" else (b, a)
            text = (f"**Yes — {_md_safe(brand)} is a brand of {_md_safe(generic)}**, so it's the "
                    "same active medicine (brands can differ in strength, form and fillers).")
        else:
            text = (f"**Yes — {_md_safe(a)} and {_md_safe(b)} are two names for the same medicine** "
                    "(one is the international/British name, the other the US name).")
        blocks.append(_b("answer", items=[(text, "", "")], label="Short answer"))
    if p["mode"] == "interaction":
        return _answer_interaction(p, opts, blocks)
    if p["mode"] == "compare" and len(ents) == 2:
        out = _answer_compare(p, opts, blocks)
        if out is not None:
            return out
    if p["mode"] == "relation":
        out = _answer_relation(p, opts, blocks)
        if out is not None:
            return out
    multi = len(ents) > 1
    first_ok = None
    for en in ents:
        loaded = _chat_load(en, opts, level, extras=bool(set(intents) & _EXTRAS_INTENTS))
        if multi:
            blocks.append(_b("md", v=f"### {_md_safe(en.get('display') or en.get('query'))}"))
        if loaded.get("offline"):
            blocks += loaded["blocks"]
            continue
        if loaded.get("error"):
            blocks.append(_b("warning", v=loaded["error"]))
            sugg = loaded.get("suggestions") or []
            if sugg:
                blocks.append(_b("buttons", title="Did you mean:",
                                 items=[(s, _replace_entity(p["raw"], en, s)) for s in sugg[:4]]))
            else:
                blocks.append(_b("buttons", title="Questions I can answer, for example:",
                                 items=[(q, q) for q in _examples(4)]))
            continue
        ctx = loaded["ctx"]
        blocks += loaded["notes"]
        if p.get("used_context") and en is ents[0]:
            blocks.append(_b("caption", v=f"(about **{_md_safe(_chat_name(ctx, en))}**, from earlier)"))
        _progress(opts, "Picking out the parts that answer your question…")
        blocks += _answer_for(ctx, en, p, opts, compact=multi)
        down = [d for d in _sources_down(ctx) if d != "ChEMBL" or cs.host_failing(cs.CHEMBL_BASE)]
        if down:
            blocks.append(_b("caption", v=f"⚠️ {', '.join(down)} didn't respond just now — this "
                                          "answer may be incomplete. Ask again in a few minutes."))
        elif ctx.get("chembl_timed_out") and set(intents) & {"mechanism", "overview", "class"}:
            # ChEMBL is merely slow (it keeps loading in the background)
            blocks.append(_b("caption", v="ChEMBL was slow, so curated target data may be "
                                          "missing — ask again shortly to include it."))
        if loaded.get("others"):
            blocks.append(_b("buttons", title="Also in this product:",
                             items=[(f"Ask about {o}", _replace_entity(p["raw"], en, o))
                                    for o in loaded["others"][:3]]))
        if first_ok is None:
            first_ok = (en, ctx, loaded["query"])
    if first_ok is None:
        return blocks, None
    en, ctx, query = first_ok
    ref = _entity_ref(en, ctx)
    src = _sources_caption(ctx)
    if src:
        blocks.append(_b("caption", v=src))
    nxt = _followups(ref, ctx, intents) if not multi else []
    if not ctx.get("name_only"):
        nxt.append(("📖 Full profile", _PROFILE_PREFIX + query))
    blocks.append(_b("buttons", title="Ask next", items=nxt))
    stored = {"kind": "free", "query": query, "display": ref}
    if en.get("kind") in ("library", "alias"):
        stored = {k: en[k] for k in ("kind", "query", "display", "alias_of") if k in en}
    return blocks, {"entity": stored, "intents": intents,
                    "interactant": p.get("interactant") or "", "focus": p.get("focus") or [],
                    "pk_facts": p.get("pk_facts") or [], "population": p.get("population") or []}


_COMPARE_PK = ("Absorption", "Protein binding", "Metabolism", "Half-life", "Excretion")


def _table_cell(text: str, src: str = "", url: str = "", limit: int = 220) -> str:
    """Source text → one Markdown-table cell (no pipes/newlines) + source badge."""
    t = _tidy(text)
    t = t[:1].upper() + t[1:]          # OTC panels start mid-sentence ('temporarily relieves…')
    if len(t) > limit:
        t = t[:limit].rsplit(" ", 1)[0] + "…"
    t = _md_safe(t).replace("|", "\\|").replace("\n", " ")
    return t + _badge(src, url)


def compare_profile(ctx: Dict[str, Any]) -> Dict[str, str]:
    """One drug's column of the comparison table: field → cell ('' when its
    sources don't state it). Every value is quoted from that drug's own sources
    — nothing is computed across the two."""
    f = _ctx_glance(ctx)
    src = f.get("src") or {}
    of = ctx.get("label") or {}
    row: Dict[str, str] = {}

    def put(key: str, val: str, s: str = "", url: str = ""):
        row[key] = _table_cell(val, s, url) if val else ""

    if f["is_med"]:
        put("Drug class", " · ".join(f.get("classes") or []), src.get("classes", ""))
        put("Used for", f.get("use", ""), src.get("use", ""))
        put("How it works", f.get("mech", ""), src.get("mech", ""))
    else:
        row["Drug class"] = "not a recognised medicine"
    items, _scope = _pk_items(ctx)
    # (source, url) ride along with each sentence, so a headline whose
    # 'Half Life:' prefix gets stripped keeps its link
    facts, _rest = _pk_key_facts([(s, (sr, u)) for s, sr, u in items])
    by_label = {lbl: (s, payload) for lbl, s, payload in facts}
    fact_rx = dict(_PK_FACTS)
    for lbl in _COMPARE_PK:
        s, (sr, u) = by_label.get(lbl, ("", ("", "")))
        # a multi-sentence source line → just the sentence that states the fact
        # (split after any end mark, so '…approximately 50%. Less than…' parts)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", s)
        cell = next((x for x in parts if fact_rx[lbl].search(x)), s)
        put(lbl, cell, sr, u)
        if PARENT_TAG in sr:
            row["_parent_pk"] = "1"           # metadata, not a table row
    if f.get("common"):
        put("Common side effects", f["common"], src.get("common", ""), of.get("url", ""))
    elif f.get("faers_terms"):
        put("Common side effects", f"Most reported: {f['faers_terms']} (reports, not incidence)",
            "FDA FAERS")
    else:
        row["Common side effects"] = ""
    if of:
        put("Boxed warning", boxed_status(of), "US FDA label", of.get("url", ""))
    else:
        row["Boxed warning"] = "No US FDA label found"
    put("Availability", f.get("avail", ""), src.get("avail", ""))
    put("Status", f.get("status", ""), src.get("status", ""))
    # which product the label rows describe — doses/warnings are product-specific
    if of:
        brand = (of.get("brand_names") or [""])[0]
        routes = ", ".join(r.lower() for r in of.get("routes") or [])
        put("US label used", " · ".join(x for x in (brand, routes, of.get("effective_date", ""))
                                       if x), "openFDA", of.get("url", ""))
    return row


def boxed_status(of: Dict[str, Any], limit: int = 200) -> str:
    """What THIS label says about a boxed warning — claiming 'none' only when
    the label's structure supports it: a modern (PLR) or OTC label has a
    dedicated boxed-warning section when there is one; an old-format label may
    keep it inside 'Warnings'. 'This label', not 'the drug': another product
    of the same drug can carry one (IV acetaminophen does, oral doesn't)."""
    if of.get("boxed_warning"):
        return boxed_summary(of["boxed_warning"], limit)
    if of.get("has_boxed_section"):
        return "This label has a boxed warning, but its text couldn't be read — see DailyMed"
    form = ", ".join(r.lower() for r in of.get("routes") or [])
    if of.get("plr_format") or of.get("product_type") == "HUMAN OTC DRUG":
        return (f"None on this label{f' ({form})' if form else ''} — other products of "
                "the drug may differ")
    return ("No boxed-warning section found — older-format label, so check its "
            "Warnings section")


_GAP_NOT_STATED = "<span style='opacity:.55'>not stated in sources</span>"
_GAP_SOURCE_DOWN = "<span style='opacity:.55'>not available — a source didn't respond</span>"


def compare_table_md(cols: List[Tuple[str, Dict[str, str]]],
                     gaps: Optional[List[str]] = None) -> str:
    """Side-by-side Markdown table; rows neither drug has are dropped, and a
    gap says 'not stated' rather than implying none/zero — or, for a column
    whose sources failed (`gaps`), that the data is unavailable, not absent.
    Keys starting with '_' are metadata, never rows."""
    gaps = gaps or [_GAP_NOT_STATED] * len(cols)
    fields = list(dict.fromkeys(k for _n, row in cols for k in row if not k.startswith("_")))
    lines = ["| | " + " | ".join(f"**{_md_safe(n)}**" for n, _r in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for fld in fields:
        cells = [row.get(fld, "") for _n, row in cols]
        if not any(cells):
            continue
        lines.append(f"| **{fld}** | " + " | ".join(c or g for c, g in zip(cells, gaps)) + " |")
    return "\n".join(lines)


def compare_notes(ctxs: List[Dict[str, Any]], names: List[str],
                  rows: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Scope/quality notes shown ONCE above the comparison table, as
    [(kind, markdown)] with kind in info/warning/caption: market (US labels
    read in Singapore mode), formulation mismatch per drug, parent-derived PK
    per drug, and sources that didn't respond."""
    notes: List[Tuple[str, str]] = []
    if any(c.get("region") == "SG" and c.get("label") for c in ctxs):
        missing = [n for c, n in zip(ctxs, names)
                   if c.get("label") and not (c.get("market") or {}).get("sg")]
        notes.append(("warning", "🌏 **Label rows (used for, side effects, boxed warning) come "
                                 "from US FDA labels.** Singapore products can differ — the "
                                 f"package insert on [HSA Infosearch]({HSA_INFOSEARCH_URL}) is "
                                 "authoritative there."
                      + (f" No HSA-registered product was found for {', '.join(missing)}."
                         if missing else "")))
    for c, n in zip(ctxs, names):
        of = c.get("label") or {}
        if not of:
            continue
        local = market_route(c.get("region", "US"), (c.get("drug_hit") or {}).get("name"),
                             c.get("compound_name"))
        # region="US": the market note above already covers the jurisdiction
        for w in label_scope_warnings(of, region="US",
                                      drug_routes=(c.get("clin_chembl") or {}).get("routes") or (),
                                      local_route=local):
            notes.append(("warning", f"**{_md_safe(n)}:** {w}"))
    for row, n in zip(rows, names):
        if row.get("_parent_pk"):
            notes.append(("info", f"↩ **{_md_safe(n)}:** some pharmacokinetic facts come from the "
                                  "parent compound (free base/acid), not the exact salt — "
                                  "they're tagged *parent*."))
    down = sorted({d for c in ctxs for d in _sources_down(c)
                   if d != "ChEMBL" or cs.host_failing(cs.CHEMBL_BASE)})
    slow = [n for c, n in zip(ctxs, names) if c.get("chembl_timed_out")]
    if down:
        notes.append(("warning", f"⚠️ {', '.join(down)} didn't respond just now, so empty cells "
                                 "may be missing data rather than absent facts. Ask again in a "
                                 "few minutes."))
    elif slow:
        notes.append(("caption", f"ChEMBL was slow for {', '.join(slow)}, so mechanism/status "
                                 "may be missing — ask again shortly to include them."))
    return notes


def _column_gap(ctx: Dict[str, Any]) -> str:
    """Gap text for one drug's column: unavailable when its sources failed."""
    return _GAP_SOURCE_DOWN if _sources_down(ctx) else _GAP_NOT_STATED


def _answer_compare(p, opts, blocks):
    """'metoprolol vs propranolol' → one side-by-side table of each drug's own
    sourced facts (class, use, mechanism, ADME, side effects, boxed warning,
    availability), then any specific question answered per drug. Returns None
    when either drug can't be loaded (the caller answers them one by one)."""
    ents = p["entities"][:2]
    loaded = [_chat_load(en, opts, "full", extras=False) for en in ents]
    if any(ld.get("offline") or ld.get("error") for ld in loaded):
        return None
    ctxs = [ld["ctx"] for ld in loaded]
    names = [_chat_name(c, en) for c, en in zip(ctxs, ents)]
    for ld in loaded:
        blocks += ld["notes"]
    _progress(opts, "Lining the two up side by side…")
    rows = [compare_profile(c) for c in ctxs]
    blocks.append(_b("md", v=f"#### ⚖️ {_md_safe(names[0])} vs {_md_safe(names[1])}"))
    # scope notes go ABOVE the table, so they're read before the facts
    for kind, text in compare_notes(ctxs, names, rows):
        blocks.append(_b(kind, v=text))
    blocks.append(_b("md", v=compare_table_md(list(zip(names, rows)),
                                              gaps=[_column_gap(c) for c in ctxs]), html=True))
    blocks.append(_b("caption", v="Each cell is quoted from that drug's own sources (the tag says "
                                  "which). “Not stated” means the sources don't say — not that "
                                  "the value is zero or absent. PK figures apply to the population "
                                  "described in the source sentence."))
    specific = [i for i in p["intents"] if i != "overview"]
    if specific:
        for c, en, n in zip(ctxs, ents, names):
            blocks.append(_b("md", v=f"### {_md_safe(n)}"))
            blocks += _answer_for(c, en, p, opts, compact=True)
    refs = [_entity_ref(en, c) for en, c in zip(ents, ctxs)]
    blocks.append(_b("buttons", title="Ask next",
                     items=[("🔀 Take them together?", f"Can I take {refs[0]} with {refs[1]}?"),
                            ("⏱ Half-lives", f"half-life of {refs[0]} vs {refs[1]}"),
                            (f"📖 {names[0]} profile", _PROFILE_PREFIX + loaded[0]["query"]),
                            (f"📖 {names[1]} profile", _PROFILE_PREFIX + loaded[1]["query"])]))
    return blocks, {"entity": {"kind": "free", "query": loaded[0]["query"], "display": refs[0]},
                    "intents": p["intents"]}


# ---- relationship questions: 'why is X the antidote for Y' -----------------
# Words that make a sentence EXPLAIN a link rather than merely mention it.
_EXPLAIN_RE = re.compile(
    r"\b(?:because|due to|by (?:\w+ing)\b|via|through|thereby|bypass\w*|independent(?:ly)? of|"
    r"mechanism|acts? (?:on|by|as|at)|works? by|lead(?:s|ing)? to|result(?:s|ing)? in|"
    r"increas\w*|decreas\w*|restor\w*|activat\w*|inhibit\w*|block\w*|bind\w*|receptors?|"
    r"cAMP|cyclic AMP|second messenger|adenyl\w* cyclase|competitive\w*|antagonis\w*|"
    r"agonis\w*|displac\w*|counteract\w*|revers\w*|stimulat\w*)\b", re.I)
# 'Patients taking beta-blockers may have a transient increase…' — an
# interaction warning, not an explanation of the link.
_REL_NOISE_RE = re.compile(r"\bpatients (?:taking|on|receiving)\b|\bconcomitant\b|"
                           r"\bco-?administ\w*|\bclinical impact\b|\bmay have a transient\b", re.I)
_REL_GENERIC = {"overdose", "overdoses", "overdosage", "toxicity", "poisoning", "toxic",
                "intoxication", "effect", "effects", "treatment", "therapy", "patients",
                "type", "the", "of", "and", "in", "with"}
_REL_QUESTION_CUES = re.compile(r"\b(?:antidote|overdos\w*|toxicity|poisoning|revers\w*|"
                                r"treat\w*)\b", re.I)


def _rel_word_rx(word: str) -> str:
    """One target word → a tolerant pattern ('beta' also matches 'β', a plural
    or inflection also matches)."""
    w = re.escape(word.lower())
    greek = {"beta": "(?:beta|β)", "alpha": "(?:alpha|α)", "gamma": "(?:gamma|γ)"}
    if word.lower() in greek:
        return greek[word.lower()]
    return w[:-1] + r"\w*" if len(word) > 4 else w + r"\w*"


def relation_target_re(target: str) -> Optional[re.Pattern]:
    """Pattern for the core of a relationship target: 'beta blocker overdose'
    → 'beta[- ]?(adrenergic )?blocker…' (the generic 'overdose' is dropped — it
    scores a sentence, but doesn't identify the other side)."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", target) if w.lower() not in _REL_GENERIC]
    if not words:
        words = re.findall(r"[A-Za-z0-9]+", target)
    if not words:
        return None
    joiner = r"[\s\-]*(?:adrenergic[\s\-]*|receptor[\s\-]*)?"
    return re.compile(r"\b" + joiner.join(_rel_word_rx(w) for w in words), re.I)


def relation_core(target: str) -> str:
    """'beta blocker overdose' → 'beta blocker' (for looking up its article)."""
    words = [w for w in target.split() if w.lower() not in _REL_GENERIC]
    return " ".join(words) or target


def score_relation_sentence(text: str, subj_re: Optional[re.Pattern],
                            tgt_re: re.Pattern, question: str) -> float:
    """How well a sentence explains the subject↔target link (0 = unusable).
    Must mention the target; both sides, explanatory wording and the question's
    own cue words (antidote, overdose…) raise it; interaction warnings sink it."""
    if not tgt_re.search(text):
        return 0.0
    score = 1.0
    if subj_re is not None and subj_re.search(text):
        score += 2.0
    score += min(3, len(_EXPLAIN_RE.findall(text))) * 1.2
    for cue in set(m.lower()[:6] for m in _REL_QUESTION_CUES.findall(question)):
        if re.search(re.escape(cue), text, re.I):
            score += 1.0
    if _REL_NOISE_RE.search(text):
        score -= 3.0
    if len(text) > 400:
        score -= 1.0
    return max(score, 0.0)


def _wiki_sentences(secs: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(sentence, section heading) for a whole Wikipedia article."""
    out = []
    for heading, text in [("", secs.get("lead", ""))] + \
            [(s.get("heading", ""), s.get("text", "")) for s in secs.get("sections", [])]:
        for s in _split_sentences(re.sub(r"\s+", " ", text or "")):
            if len(s) > 30:
                out.append((s, heading))
    return out


def relation_evidence(ctx: Dict[str, Any], subject: str, target: str,
                      question: str) -> List[Tuple[float, str, str, str]]:
    """Ranked [(score, sentence, source, url)] connecting the subject drug to
    the target, from the drug's own sources, its full Wikipedia article and
    the TARGET's Wikipedia article (e.g. 'Beta blocker' → its Overdose
    section names glucagon)."""
    tgt_re = relation_target_re(target)
    if tgt_re is None:
        return []
    subj_re = _drug_names_re(ctx) or re.compile(r"\b" + re.escape(subject) + r"\b", re.I)
    cands: List[Tuple[str, str, str]] = []
    cands += [(d["text"], d["src"], d.get("url", "")) for d in _chat_corpus(ctx)]
    secs = ctx.get("wiki_secs") or {}
    cands += [(s, "Wikipedia", secs.get("url", "")) for s, _h in _wiki_sentences(secs)]
    # the other side's article: sentences there must name the subject drug
    core = relation_core(target)
    other = cached_wiki_sections(core) or {}
    if other and other.get("title") != secs.get("title"):
        label = f"Wikipedia: {other.get('title')}"
        cands += [(s, label, other.get("url", "")) for s, _h in _wiki_sentences(other)
                  if subj_re.search(s)]
    scored = []
    dd = _SentenceDeduper()
    for text, src, url in cands:
        # DrugBank/PubChem citation codes glued on: '…pesticides.L42815,​L42835'
        text = re.sub(r"\.?\s*\b[LA]\d{4,6}(?:[,\s​]+[LA]\d{4,6})*\s*$", ".", text).strip()
        sc = score_relation_sentence(text, subj_re, tgt_re, question)
        if sc <= 0:
            continue
        scored.append((sc, text, src, url))
    scored.sort(key=lambda x: -x[0])
    out = []
    for sc, text, src, url in scored:
        ws = _content_words(text)
        if dd.is_dup(ws) or dd.is_restatement(ws):
            continue
        dd.accept(ws)
        out.append((sc, text, src, url))
    return out


def _answer_relation(p, opts, blocks):
    """'why is glucagon the antidote for beta-blocker overdose' → the sentences
    in the sources that explain the link (best first), then how the drug works.
    No reasoning engine: it finds where the sources make the connection."""
    en = p["entities"][0]
    target = p.get("relation_target") or ""
    loaded = _chat_load(en, opts, "full", extras=False)
    if loaded.get("offline") or loaded.get("error"):
        return None
    ctx = loaded["ctx"]
    name = _chat_name(ctx, en)
    blocks += loaded["notes"]
    _progress(opts, f"Looking for where the sources connect {name} and {target}…")
    ev = relation_evidence(ctx, name, target, p.get("raw") or "")
    blocks.append(_b("md", v=f"#### 🔗 {_md_safe(name)} ↔ {_md_safe(target)}"))
    strong = [e for e in ev if e[0] >= 3.0]
    if strong:
        blocks += _answer_box([(t, s, u) for _sc, t, s, u in strong[:2]],
                              label="What the sources say about the link")
        blocks += _pts([(t, s, u) for _sc, t, s, u in strong[2:5]],
                       title="More from the sources", cap=2, more="More")
    elif ev:
        blocks.append(_b("info", v=f"The sources mention {_md_safe(name)} and "
                                   f"{_md_safe(target)} together but don't explain the link:"))
        blocks += _pts([(t, s, u) for _sc, t, s, u in ev[:3]], cap=3)
    else:
        blocks.append(_b("info", v=f"None of {_md_safe(name)}'s sources (FDA label, PubChem, "
                                   f"ChEMBL, Wikipedia) mention {_md_safe(target)} — so there's "
                                   "nothing to quote. How it works is below; a pharmacist or a "
                                   "pharmacology text can explain the connection."))
    # how the drug works — the other half of most 'why' answers
    _h, val, _scope = _find_heading(ctx, "mechanism")
    kp = (val or {}).get("key_points") if isinstance(val, dict) else None
    mech = list(kp or []) or [(t, "ChEMBL", (ctx.get("clin_chembl") or {}).get("url", ""))
                              for t, _tgt, _act in _chembl_mechs(ctx)]
    if mech:
        blocks += _pts(mech[:3], title=f"How {name} works", cap=3)
    blocks.append(_b("caption", v="Ask doesn't reason — it finds the sentences in the sources "
                                  "that connect the two and ranks the ones that explain *how* "
                                  "first. Follow the source tags to read them in context."))
    ref = _entity_ref(en, ctx)
    blocks.append(_b("buttons", title="Ask next",
                     items=[("🧬 How it works", f"How does {ref} work?"),
                            ("⚠️ Side effects", f"Side effects of {ref}"),
                            ("📖 Full profile", _PROFILE_PREFIX + loaded["query"])]))
    return blocks, {"entity": {"kind": "free", "query": loaded["query"], "display": ref},
                    "intents": p.get("intents") or ["overview"]}


def _answer_interaction(p, opts, blocks) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    a, b = p["entities"][0], p["entities"][1]
    la = _chat_load(a, opts, "full", extras=False)
    lb = _chat_load(b, opts, "full", extras=False)
    if la.get("offline") or lb.get("offline"):
        return blocks + (la.get("blocks") or lb.get("blocks") or []), None
    if la.get("error") and lb.get("error"):
        blocks.append(_b("warning", v=la["error"]))
        blocks.append(_b("warning", v=lb["error"]))
        return blocks, None
    if la.get("error"):                    # answer from the one we found
        la, lb, a, b = lb, la, b, a
    ca = la["ctx"]
    na = _chat_name(ca, a)
    blocks += la["notes"]
    cb = None if lb.get("error") else lb["ctx"]
    nb = _chat_name(cb, b) if cb else (b.get("display") or b.get("query"))
    if cb:
        blocks += lb["notes"]
    else:
        blocks.append(_b("caption", v=f"(couldn't identify {_md_safe(nb)} — searching "
                                      f"{_md_safe(na)}'s sources for the name as typed)"))
    blocks.append(_b("md", v=f"#### 🔀 {_md_safe(na)} + {_md_safe(nb)}"))
    blocks += _ans_interactions(ca, na, p, other=(nb, cb))
    if cb:
        hits = _mentions(cb, _interaction_terms(ca, na),
                         topics={"interactions", "warnings", "side_effects", "pk", "pharma"})
        if hits:
            blocks += _pts(hits, title=f"…and what {nb}'s sources say about {na}", cap=3,
                           more="More mentions")
    blocks.append(_b("caption", v="Interaction text is quoted from each drug's own FDA label / "
                                  "PubChem / Wikipedia — no mention isn't proof of safety. "
                                  "Ask a pharmacist before combining medicines."))
    ref = _entity_ref(a, ca)
    blocks.append(_b("buttons", title="Ask next",
                     items=[("🔀 All interactions", f"{ref} interactions"),
                            ("⚠️ Side effects", f"Side effects of {ref}"),
                            ("🧬 How it works", f"How does {ref} work?"),
                            ("📖 Full profile", _PROFILE_PREFIX + la["query"])]))
    return blocks, {"entity": {"kind": "free", "query": la["query"], "display": ref},
                    "intents": ["interactions"]}


def _answer_class_list(p) -> List[Dict[str, Any]]:
    if not HAS_DRUG_LIBRARY:
        return [_b("info", v="The Drug Library isn't available, so I can't list drug classes.")]
    slugs = p.get("class_slugs") or []
    phrase = p.get("class_phrase") or (dl.class_name(slugs[0]) if slugs else "")
    blocks = [_b("md", v=f"#### 🗂 {_md_safe(phrase[:1].upper() + phrase[1:])}")]
    seen: set = set()
    first_drugs = []
    for slug in slugs[:4]:
        drugs = dl.drugs_in_class(slug)
        names = [d["name"] for d in drugs]
        new = [n for n in names if n not in seen]
        if not names or len(new) < max(1, len(names) // 5):
            continue                      # same drugs under another system's name
        seen |= set(names)
        first_drugs = first_drugs or drugs
        desc = dl.class_desc(slug)
        blocks.append(_b("md", v=f"**{_md_safe(dl.class_name(slug))}** — {len(drugs)} drug"
                                 f"{'s' if len(drugs) != 1 else ''}"
                                 + (f" · *{_md_safe(desc)}*" if desc else "")))
        blocks.append(_b("md", v=_md_safe(", ".join(names[:40]) + (" …" if len(names) > 40 else ""))))
    other_intents = [i for i in p.get("intents") or [] if i not in ("overview", "search")]
    if other_intents:
        blocks.append(_b("caption", v=f"{cr.INTENT_TITLES.get(other_intents[0], 'Details')} differ "
                                      "between drugs in a class — ask about one of them:"))
    q = {"mechanism": "How does {} work?", "side_effects": "Side effects of {}",
         "dosing": "{} dosage"}.get(other_intents[0] if other_intents else "", "What is {}?")
    blocks.append(_b("buttons", title="Ask about one:",
                     items=[(d["name"], q.format(d["name"])) for d in first_drugs[:4]]))
    return blocks


def _answer_condition(p, opts) -> List[Dict[str, Any]]:
    if not HAS_DRUG_LIBRARY:
        return [_b("info", v="The Drug Library isn't available for condition lookups.")]
    cond = p["condition"]
    sg = opts["region"] == "SG" and HAS_SG_REGISTER
    blocks = [_b("md", v=f"#### 💡 Medicines used for {cond['label']}"),
              _b("caption", v="Grouped by WHO ATC class from the Drug Library — what's used, not "
                              "a recommendation. A pharmacist or doctor can say what suits you.")]
    buttons = []
    for code in cond["codes"]:
        drugs = dl.drugs_in_class(code)
        if not drugs:
            continue
        rows = []
        for d in drugs:
            tag = ""
            if sg:
                summ = sgr.lookup(d["query"], d["name"])
                if not summ:
                    continue              # not sold in Singapore
                tag = "/".join(summ["classes"])
            rows.append((d["name"], tag))
        if not rows:
            continue
        txt = ", ".join(f"{n} ({t})" if t else n for n, t in rows[:15]) + (" …" if len(rows) > 15 else "")
        blocks.append(_b("md", v=f"**{_md_safe(dl.class_name(code))}:** {_md_safe(txt)}"))
        buttons += [(n, f"How does {n} work?") for n, _t in rows[:2]]
    if sg:
        blocks.append(_b("caption", v="Only medicines registered with HSA are listed. GSL = general "
                                      "sale · P = pharmacy-only · POM = prescription-only."))
    if len(blocks) == 2:
        blocks.append(_b("info", v="No matching medicines found in the library."))
    blocks.append(_b("buttons", title="Ask about one:", items=buttons[:4]))
    return blocks


def _crisis_blocks(opts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Self-harm wording: support first, never toxicity or doses."""
    us = opts.get("region") == "US"
    lines = ("- **US:** call or text **988** (Suicide & Crisis Lifeline) · emergency **911**\n"
             if us else
             "- **Singapore:** Samaritans of Singapore (SOS) 24-hour hotline **1767** "
             "(WhatsApp 9151 1767) · IMH Mental Health Helpline **6389 2222** · emergency **995**\n")
    return [
        _b("error", v="💛 **It sounds like you might be going through something really hard.** You "
                      "don't have to face it alone — please reach out to someone now."),
        _b("md", v=lines + "- **Anywhere else:** your local emergency number, or find a free, "
                           "confidential helpline at [findahelpline.com](https://findahelpline.com)"),
        _b("md", v="**If you or someone else has already taken too much of a medicine, call "
                   + ("911" if us else "995") + " now.**"),
        _b("caption", v="This assistant won't give information that could be used for self-harm."),
    ]


def _refuse_blocks() -> List[Dict[str, Any]]:
    return [
        _b("warning", v="I can't help with making or extracting drugs outside a licensed setting — "
                        "it's illegal and dangerous (toxic reagents, contamination, unknown doses)."),
        _b("md", v="I can explain what a drug is, how it works, its effects and risks, or how "
                   "addiction and withdrawal are treated. In Singapore, the National Addictions "
                   "Management Service (NAMS) helpline is **6732 6837**."),
    ]


def _pet_blocks(p: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = []
    if p.get("urgent"):
        blocks.append(_b("error", v="🐾 **If your pet has eaten a medicine, contact a vet or an "
                                    "animal emergency clinic now** — bring the pack with you."))
    blocks.append(_b("info", v="This assistant covers **human** medicines. Doses and safety are very "
                               "different in animals, and many human medicines are dangerous to "
                               "pets — please don't give one without a vet's advice."))
    return blocks


def _examples(k: int = 6) -> List[str]:
    """Example questions from the question bank, one per category — stable
    within a session (so buttons don't jump on every rerun) until reshuffled."""
    qbm = getattr(cr, "_qb", None) if cr else None
    if qbm is None:
        return _CHAT_EXAMPLES[:k]
    seed = st.session_state.setdefault("showcase_seed", int(time.time()) % 100000)
    try:
        return qbm.showcase(k, seed) or _CHAT_EXAMPLES[:k]
    except Exception:  # noqa: BLE001
        return _CHAT_EXAMPLES[:k]


def _reshuffle_examples():
    st.session_state["showcase_seed"] = st.session_state.get("showcase_seed", 0) + 1


def _suggest_blocks(name: str, skip_topics=(), title: str = "Try asking:") -> List[Dict[str, Any]]:
    """Ready-made questions about this drug — shown when a question wasn't
    understood, so the user can rephrase with one tap."""
    qbm = getattr(cr, "_qb", None) if cr else None
    if qbm is None or not name:
        return []
    qs = qbm.suggestions_for(name, 3, skip_topics)
    return [_b("buttons", title=title, items=[(q, q) for q in qs])] if qs else []


def _help_blocks() -> List[Dict[str, Any]]:
    return [
        _b("md", v="Ask me about any medicine or chemical in plain English. I match your "
                   "question against FDA labels, ChEMBL, PubChem, Wikipedia and the HSA "
                   "register and show the relevant part — there's no AI model, so nothing is "
                   "invented and every line shows its source."),
        _b("md", v="- **Yes/no questions get a straight answer first** — *does metformin cause weight "
                   "gain?*, *is ibuprofen an NSAID?*, *do I need a prescription for amoxicillin?*, *is "
                   "codeine addictive?*\n"
                   "- **How it works** — *how does codeine work?*, *semaglutide pharmacodynamics*\n"
                   "- **Side effects & safety** — *is paracetamol safe in pregnancy?*, *can I drive on "
                   "diazepam?*, *I'm allergic to penicillin — can I take amoxicillin?*\n"
                   "- **Dosing & taking it** — *max dose of paracetamol*, *how often can I take "
                   "ibuprofen?*, *what if I miss a dose?*, *can I crush it?*, *with food?*\n"
                   "- **Half-life, interactions** — *how long does diazepam stay in my system?*, *can I "
                   "drink alcohol on metronidazole?*, *can I take ibuprofen with warfarin?*\n"
                   "- **Chemistry** — *structure of codeine*, *pKa of aspirin*, *3D structure of "
                   "caffeine*\n"
                   "- **Why / how questions** — *why is glucagon the antidote for beta-blocker "
                   "overdose?*, *how does naloxone reverse morphine?* (finds where the sources "
                   "explain the link)\n"
                   "- **Compare** — *metoprolol vs propranolol*\n"
                   "- **Brands & classes** — *brands of atorvastatin in Singapore*, *what are "
                   "statins?*, *what can I take for heartburn?*\n"
                   "- **Follow-ups** — *what about its half-life?*, *and morphine?*"),
        _b("buttons", title="Try:", items=[(q, q) for q in _examples(6)]),
    ]


def chat_respond(text: str, opts: Dict[str, Any]) -> Dict[str, Any]:
    """Answer one chat message → an assistant message ({"role", "blocks"})."""
    t0, req0 = time.perf_counter(), cs.REQUEST_COUNT
    mode = ""
    context = st.session_state.get("chat_context") or {}
    try:
        p = cr.parse_query(text, context=context)
    except Exception as e:  # noqa: BLE001
        return {"role": "assistant", "blocks": _error_blocks(e, opts)}
    try:
        mode = p["mode"]
        new_ctx = None
        if mode == "empty":
            blocks = []
        elif mode == "help":
            blocks = _help_blocks()
        elif mode == "thanks":
            blocks = [_b("md", v="You're welcome! Ask me anything else about a medicine or chemical.")]
        elif mode == "crisis":
            blocks = _crisis_blocks(opts)
        elif mode == "refuse":
            blocks = _refuse_blocks()
        elif mode == "pets":
            blocks = _pet_blocks(p)
        elif mode == "class_list":
            blocks = _answer_class_list(p)
        elif mode == "condition":
            blocks = _answer_condition(p, opts)
        elif mode == "clarify":
            intent = next((i for i in p["intents"] if i != "overview"), "")
            ex = {"side_effects": "side effects of ibuprofen", "dosing": "paracetamol dosage",
                  "mechanism": "how does metformin work", "pk": "half-life of diazepam",
                  "structure": "structure of caffeine"}.get(intent, "how does codeine work")
            blocks = [_b("md", v=f"Which medicine or chemical do you mean? For example: *“{ex}”*.")]
            if p.get("suggestions"):
                blocks.append(_b("buttons", title="Did you mean:",
                                 items=[(s, f"{text} {s}") for s in p["suggestions"][:4]]))
        elif mode == "unknown":
            blocks = [_b("md", v="I couldn't find a medicine or chemical in that question. Try a "
                                 "generic or brand name, a chemical name or a SMILES string.")]
            if p.get("suggestions"):
                blocks.append(_b("buttons", title="Did you mean:",
                                 items=[(s, f"What is {s}?") for s in p["suggestions"][:4]]))
            blocks.append(_b("buttons", title="Questions I can answer, for example:",
                             items=[(q, q) for q in _examples(4)]))
        else:
            blocks, new_ctx = _answer_entities(p, opts)
        if new_ctx:
            st.session_state["chat_context"] = new_ctx
    except Exception as e:  # noqa: BLE001
        blocks = _error_blocks(e, opts)
    n_req = cs.REQUEST_COUNT - req0
    if mode not in ("empty", "help", "thanks"):
        blocks.append(_b("caption", v=f"⏱ {time.perf_counter() - t0:.1f} s · "
                                      + (f"{n_req} live request{'s' if n_req != 1 else ''}"
                                         if n_req else "from cache")))
    return {"role": "assistant", "blocks": blocks}


def _error_blocks(e: Exception, opts: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = [_b("error", v=f"Sorry — something went wrong while answering "
                            f"({type(e).__name__}: {_md_safe(str(e)[:200])}). Try rephrasing, or "
                            "use the 🔎 Explore tab for the full profile.")]
    if opts.get("debug_mode"):
        import traceback
        blocks.append(_b("code", v=traceback.format_exc()))
    return blocks


def render_chat(opts: Dict[str, Any]):
    """The Ask page: a chat over the same data as Explore."""
    if not HAS_CHAT:
        st.error("The chat module (chat_router.py) couldn't be loaded — use 🔎 Explore.")
        return
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)
    msgs: List[Dict[str, Any]] = st.session_state.setdefault("chat", [])
    # Read the question FIRST (the input is docked at the bottom wherever it's
    # called) so history knows whether a new answer is coming. No st.rerun():
    # the new answer renders in place, which keeps every widget in the page
    # tree current (a rerun left stale buttons whose callbacks could fire).
    prompt = st.chat_input("Ask about a drug or chemical — e.g. “how does codeine work?”")
    pending = st.session_state.pop("chat_pending", None)
    question = cr.clip_query(prompt or pending or "")

    if not msgs and not question:
        with st.chat_message("assistant", avatar="💊"):
            st.markdown("Hi! Ask me about a medicine or chemical — what it does, how it works, "
                        "side effects, dosing, interactions, structure or properties.")
            render_blocks([_b("buttons", title="Try one — or ask in your own words:",
                              items=[(q, q) for q in _examples(6)])], "welcome")
            st.button("🔀 Show other examples", key="reshuffle_examples", on_click=_reshuffle_examples)
            st.caption("Answers are quoted from FDA labels, ChEMBL, PubChem, Wikipedia and the "
                       "HSA register — rule-based matching, no AI. For information only; not "
                       "medical advice.")
    if question:
        # keep whole user/assistant pairs when trimming old history
        del msgs[:max(0, len(msgs) - (_CHAT_MAX_MESSAGES - 2))]
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(_md_safe(m["text"]))
        else:
            with st.chat_message("assistant", avatar="💊"):
                # follow-up buttons only on the newest answer, and not while a
                # new question is being answered below it
                render_blocks(m["blocks"], f"m{i}",
                              latest=(not question and i == len(msgs) - 1))
    if not question:
        return
    msgs.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(_md_safe(question))
    with st.chat_message("assistant", avatar="💊"):
        t0 = time.perf_counter()
        # a live status line ('Finding “phentermine”…', 'Reading the FDA label,
        # PubChem…') so a slow first lookup never looks frozen
        with st.status("Reading your question…", expanded=False) as status:
            reply = chat_respond(question, dict(opts, _progress=lambda m: status.update(label=m)))
            status.update(label=f"Answered in {time.perf_counter() - t0:.1f} s", state="complete")
        msgs.append(reply)
        render_blocks(reply["blocks"], f"m{len(msgs) - 1}", latest=True)


def streamlit_app():
    st.set_page_config(page_title="Drug & Chemical Explorer", layout="wide")
    st.title("💊 Drug & Chemical Explorer")
    st.caption("Ask about any medicine or chemical, or look one up — uses, dosing, side "
               "effects, warnings and pharmacology from FDA labels, ChEMBL, PubChem "
               "and Wikipedia, plus full chemistry for any compound.")

    st.session_state.setdefault("chemical_search_input", "")
    st.session_state.setdefault("lookup_pressed", False)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("chat", [])

    modes = ([_MODE_ASK] if HAS_CHAT else []) + [_MODE_EXPLORE] + \
        ([_MODE_LIBRARY] if HAS_DRUG_LIBRARY else []) + \
        ([_MODE_REFERENCE] if HAS_MAINSTREAM_REF else [])
    st.session_state.setdefault("app_mode", modes[0])
    if st.session_state["app_mode"] not in modes:
        st.session_state["app_mode"] = modes[0]
    mode = st.radio("Mode", modes, key="app_mode", horizontal=True,
                    label_visibility="collapsed")
    if mode == _MODE_LIBRARY:
        render_drug_library()
        return
    if mode == _MODE_REFERENCE:
        render_mainstream_reference()
        return
    opts = render_sidebar(mode)
    if mode == _MODE_ASK:
        if os.environ.get("CHEM_DEBUG_GLOBALS"):          # test harness hook only
            st.session_state["_dbg_globals"] = globals()
        if not st.session_state.get("_lexicon_warm"):
            # build the parser's name index while the user is still typing
            threading.Thread(target=cr._lexicon, daemon=True).start()
            st.session_state["_lexicon_warm"] = True
        render_chat(opts)
    else:
        render_explore(opts)

# ---------- Entrypoint -----------------------------------------------------
if __name__ == '__main__':
    if not HAS_STREAMLIT:
        print("This script is intended to be run with Streamlit.")
        print("Use: streamlit run chem_v4.py")
        sys.exit(1)
    streamlit_app()

"""
chem_inference.py
-----------------
Structure-grounded property inference for Chemical Explorer.

This module deliberately contains **no statistical / black-box QSAR models**.
Every value it returns is one of two kinds, and each result is tagged so the UI
can show the user *how* the value was obtained:

  * "derived"  — an exact consequence of the molecular graph (atom counts,
                 ring systems, formula-level facts). Not a prediction.
  * "inferred" — a deduction from recognised functional groups using
                 well-established chemical principles (e.g. a carboxylic acid is
                 acidic with pKa ~3.5-5; a tertiary aliphatic amine is basic).
                 Reported as *ranges with the reasoning attached*, never as a
                 fake-precise single number.

Anything that would require fitting to experimental training data (ESOL-style
solubility regressions, ML ADMET, etc.) is intentionally out of scope — those
belong to a separately-labelled "model estimate" surface, if added at all.

Kept free of Streamlit so it can be unit-tested independently (like
chem_sources.py). Depends only on RDKit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from rdkit import Chem

# Physiological pH used for the ionization-state deduction.
PHYS_PH = 7.4
# How far a pKa must sit from the pH before we call a centre "predominantly"
# (de)protonated rather than "partial". One pH unit ≈ 91% / 9% ionised, a
# defensible cut-off for a qualitative statement.
_PH_MARGIN = 1.0

MolLike = Union[str, "Chem.Mol"]


def _as_mol(mol_or_smiles: MolLike) -> Optional["Chem.Mol"]:
    if mol_or_smiles is None:
        return None
    if isinstance(mol_or_smiles, str):
        return Chem.MolFromSmiles(mol_or_smiles)
    return mol_or_smiles


# ---------------------------------------------------------------------------
# Functional-group catalogue
# ---------------------------------------------------------------------------
# Each entry: a SMARTS pattern plus the chemistry it implies. Patterns are
# written to be reasonably specific (e.g. amide nitrogens are excluded from the
# "amine" patterns) so the downstream acid/base reasoning is sound rather than
# merely pattern-counting.
#
# pka is given as a (low, high) range typical for the group in small organic
# molecules; `role` is "acid" (the group *loses* a proton) or "base" (the
# group's conjugate acid pKa is quoted, i.e. it *gains* a proton). Groups with
# pka=None are non-ionisable under physiological conditions but still carry
# H-bonding / stability meaning.
FUNCTIONAL_GROUPS: List[Dict[str, Any]] = [
    # --- strong acids ------------------------------------------------------
    {"name": "Sulfonic acid", "smarts": "[SX4](=O)(=O)[OX2H1]",
     "role": "acid", "pka": (-2.0, 1.0),
     "note": "Strongly acidic; ionised (anionic) across the whole physiological range."},
    {"name": "Phosphonic/phosphate –OH", "smarts": "[PX4](=O)([OX2H1])",
     "role": "acid", "pka": (1.5, 7.0),
     "note": "Polyprotic acid; typically mono- or di-anionic at physiological pH."},
    # --- carboxylic acids & acidic heterocycles ---------------------------
    {"name": "Carboxylic acid", "smarts": "[CX3](=O)[OX2H1]",
     "role": "acid", "pka": (3.5, 5.0),
     "note": "Acidic; predominantly deprotonated (carboxylate) at pH 7.4. Strong H-bond donor/acceptor; raises water solubility when ionised."},
    {"name": "Tetrazole", "smarts": "c1nnn[nH]1",
     "role": "acid", "pka": (4.5, 4.9),
     "note": "Carboxylic-acid bioisostere; acidic and ionised at physiological pH."},
    # --- weak acids --------------------------------------------------------
    {"name": "Phenol", "smarts": "[c][OX2H1]",
     "role": "acid", "pka": (9.0, 10.5),
     "note": "Weakly acidic; mostly neutral at pH 7.4. H-bond donor/acceptor."},
    {"name": "Thiol", "smarts": "[#6][SX2H1]",
     "role": "acid", "pka": (9.5, 11.0),
     "note": "Weakly acidic; readily oxidised (can form disulfides). Soft nucleophile."},
    {"name": "Sulfonamide N–H", "smarts": "[SX4](=O)(=O)[NX3H]",
     "role": "acid", "pka": (9.5, 11.0),
     "note": "Mildly acidic N–H; largely neutral at physiological pH."},
    {"name": "Imide", "smarts": "[NX3H1]([CX3]=O)[CX3]=O",
     "role": "acid", "pka": (8.0, 10.0),
     "note": "Acidic N–H flanked by two carbonyls; partially ionised near physiological pH."},
    # --- strong/medium bases ----------------------------------------------
    {"name": "Guanidine", "smarts": "[NX3][CX3](=[NX2])[NX3]",
     "role": "base", "pka": (12.0, 13.5),
     "note": "Very strong base; fully protonated (cationic) at physiological pH."},
    {"name": "Amidine", "smarts": "[NX3][CX3]=[NX2]",
     "role": "base", "pka": (10.5, 12.0),
     "note": "Strong base; protonated and cationic at physiological pH."},
    {"name": "Aliphatic amine", "smarts": "[NX3;!$(NC=O);!$(N=*);!$([N+]);!$(N[a]);!$(NS(=O)=O)]",
     "role": "base", "pka": (9.0, 11.0),
     "note": "Basic; predominantly protonated (cationic) at pH 7.4. H-bond donor/acceptor; improves water solubility as the salt."},
    # --- weak bases (aromatic N) ------------------------------------------
    {"name": "Imidazole", "smarts": "c1cnc[nH]1",
     "role": "base", "pka": (6.5, 7.5),
     "note": "Weak base near physiological pH; exists as a mix of protonated and neutral forms (buffering)."},
    {"name": "Aniline (aromatic amine)", "smarts": "[NX3;!$(NC=O);!$([N+])][c]",
     "role": "base", "pka": (4.0, 5.5),
     "note": "Weak base; essentially neutral at pH 7.4. Resonance into the ring lowers basicity."},
    {"name": "Pyridine", "smarts": "[nX2;+0]1ccccc1",
     "role": "base", "pka": (4.5, 6.0),
     "note": "Weak base; largely neutral at physiological pH."},
    # --- non-ionisable groups (H-bonding / stability meaning only) ---------
    {"name": "Ester", "smarts": "[CX3](=O)[OX2H0][#6]",
     "role": None, "pka": None,
     "note": "Hydrolysable to the parent acid + alcohol (chemically and enzymatically, e.g. esterases) — a likely metabolic soft-spot. H-bond acceptor."},
    {"name": "Amide", "smarts": "[NX3][CX3](=[OX1])",
     "role": None, "pka": None,
     "note": "Comparatively stable (resonance-stabilised); neither acidic nor basic at physiological pH. H-bond donor/acceptor. Hydrolysable only under forcing/enzymatic conditions."},
    {"name": "Carbamate", "smarts": "[NX3][CX3](=[OX1])[OX2][#6]",
     "role": None, "pka": None,
     "note": "Intermediate stability between ester and amide; can hydrolyse. H-bond acceptor."},
    {"name": "Nitrile", "smarts": "[NX1]#[CX2]",
     "role": None, "pka": None,
     "note": "Generally metabolically stable; weak H-bond acceptor. Dipolar."},
    {"name": "Nitro", "smarts": "[NX3](=O)=O",
     "role": None, "pka": None,
     "note": "Electron-withdrawing; can be metabolically reduced — a recognised structural-alert motif worth noting."},
    {"name": "Aldehyde", "smarts": "[CX3H1](=O)[#6]",
     "role": None, "pka": None,
     "note": "Electrophilic carbonyl; reactive toward nucleophiles and readily oxidised to the acid."},
    {"name": "Ketone", "smarts": "[#6][CX3](=O)[#6]",
     "role": None, "pka": None,
     "note": "Polar carbonyl; H-bond acceptor. Generally more stable than aldehydes."},
    {"name": "Ether", "smarts": "[OD2;!$(O[CX3]=O);!$(OC=O)]([#6])[#6]",
     "role": None, "pka": None,
     "note": "Weak H-bond acceptor; generally stable and fairly lipophilic."},
    {"name": "Aliphatic hydroxyl", "smarts": "[CX4][OX2H1]",
     "role": None, "pka": None,
     "note": "Effectively non-ionisable at physiological pH (alcohol pKa ~16). Strong H-bond donor/acceptor; aids water solubility."},
    {"name": "Halogen (C–X)", "smarts": "[#6][F,Cl,Br,I]",
     "role": None, "pka": None,
     "note": "Lipophilic; can raise logP and metabolic stability. Aryl-halide can be a metabolic/reactivity site."},
]


def detect_functional_groups(mol_or_smiles: MolLike) -> List[Dict[str, Any]]:
    """
    Return the recognised functional groups present, each with a count and the
    chemistry it implies. Provenance is "derived" — substructure matching is an
    exact reading of the graph, not a prediction.

    Each entry also carries `atom_indices` (the union of atoms across every
    match for that group) so the UI can highlight the group on the depiction.
    """
    mol = _as_mol(mol_or_smiles)
    if mol is None:
        return []
    out: List[Dict[str, Any]] = []
    for grp in FUNCTIONAL_GROUPS:
        patt = Chem.MolFromSmarts(grp["smarts"])
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt, uniquify=True)
        if not matches:
            continue
        atom_indices = sorted({idx for match in matches for idx in match})
        out.append({
            "name": grp["name"],
            "count": len(matches),
            "role": grp["role"],
            "pka": grp["pka"],
            "note": grp["note"],
            "atom_indices": atom_indices,
            "provenance": "derived",
        })
    return out


# ---------------------------------------------------------------------------
# Acid / base & ionization-state inference
# ---------------------------------------------------------------------------
def _state_for_acid(pka_hi: float, ph: float = PHYS_PH) -> tuple[str, int]:
    """Qualitative protonation state of an acidic group, and its charge."""
    # Use the higher end of the pKa range as the conservative threshold.
    if pka_hi <= ph - _PH_MARGIN:
        return "predominantly deprotonated (anionic)", -1
    if pka_hi >= ph + _PH_MARGIN:
        return "predominantly neutral (protonated)", 0
    return "partially ionised (mixed)", 0


def _state_for_base(pkah_lo: float, ph: float = PHYS_PH) -> tuple[str, int]:
    """Qualitative protonation state of a basic group (pKaH given), and charge."""
    # Use the lower end of the conjugate-acid pKa range as the conservative threshold.
    if pkah_lo >= ph + _PH_MARGIN:
        return "predominantly protonated (cationic)", +1
    if pkah_lo <= ph - _PH_MARGIN:
        return "predominantly neutral (free base)", 0
    return "partially ionised (mixed)", 0


def infer_acid_base(mol_or_smiles: MolLike, ph: float = PHYS_PH) -> Dict[str, Any]:
    """
    Deduce ionisable centres, their pKa ranges, and the dominant protonation
    state at the given pH using Henderson–Hasselbalch reasoning.

    Returns a dict with:
      centres: [ {name, count, role, pka, state, charge_each, reasoning} ]
      net_charge_estimate: int      (sum over centres at this pH)
      net_charge_note: str          (qualifier when any centre is "mixed")
      provenance: "inferred"
      ph: float
    """
    groups = detect_functional_groups(mol_or_smiles)
    centres: List[Dict[str, Any]] = []
    net = 0
    any_mixed = False
    for g in groups:
        if g["role"] not in ("acid", "base") or not g["pka"]:
            continue
        lo, hi = g["pka"]
        if g["role"] == "acid":
            state, q = _state_for_acid(hi, ph)
            reasoning = (f"{g['name']} pKa ≈ {lo:g}–{hi:g}; at pH {ph:g} the "
                         f"Henderson–Hasselbalch balance lies toward the "
                         f"{'conjugate base' if q else 'neutral acid'}.")
        else:
            state, q = _state_for_base(lo, ph)
            reasoning = (f"{g['name']} conjugate-acid pKa ≈ {lo:g}–{hi:g}; at "
                         f"pH {ph:g} the centre is "
                         f"{'mostly protonated' if q else 'mostly unprotonated'}.")
        if "mixed" in state:
            any_mixed = True
        net += q * g["count"]
        centres.append({
            "name": g["name"],
            "count": g["count"],
            "role": g["role"],
            "pka": g["pka"],
            "state": state,
            "charge_each": q,
            "atom_indices": g.get("atom_indices", []),
            "reasoning": reasoning,
            "provenance": "inferred",
        })
    note = ""
    if any_mixed:
        note = ("One or more centres have a pKa within ~1 unit of pH "
                f"{ph:g}, so the dominant species is genuinely mixed — treat "
                "the net charge as approximate.")
    return {
        "centres": centres,
        "net_charge_estimate": net,
        "net_charge_note": note,
        "provenance": "inferred",
        "ph": ph,
    }


# ---------------------------------------------------------------------------
# Stability / reactivity flags
# ---------------------------------------------------------------------------
def infer_stability_flags(mol_or_smiles: MolLike) -> List[Dict[str, Any]]:
    """
    Surface chemical-stability / reactivity expectations that follow directly
    from the functional groups present. Each flag carries the group it came
    from so the reasoning is transparent. Provenance: "inferred".
    """
    groups = detect_functional_groups(mol_or_smiles)
    flags: List[Dict[str, Any]] = []
    for g in groups:
        name = g["name"]
        if name in ("Ester", "Carbamate"):
            flags.append({"label": f"Hydrolysable {name.lower()}", "basis": name,
                          "detail": g["note"], "provenance": "inferred"})
        elif name == "Thiol":
            flags.append({"label": "Oxidation-prone thiol", "basis": name,
                          "detail": g["note"], "provenance": "inferred"})
        elif name == "Aldehyde":
            flags.append({"label": "Electrophilic / oxidisable aldehyde", "basis": name,
                          "detail": g["note"], "provenance": "inferred"})
        elif name == "Nitro":
            flags.append({"label": "Reducible nitro (structural alert)", "basis": name,
                          "detail": g["note"], "provenance": "inferred"})
    return flags


# ---------------------------------------------------------------------------
# Qualitative aqueous-solubility expectation (reasoned, not a fitted number)
# ---------------------------------------------------------------------------
def infer_solubility_tendency(mol_or_smiles: MolLike) -> Optional[Dict[str, Any]]:
    """
    A *qualitative* water-solubility expectation argued from first principles:
    lipophilicity (Crippen logP), polar surface area, and the presence of
    ionisable centres (an ionised group dramatically aids aqueous solubility).

    This is explicitly a reasoned tendency with the contributing factors shown
    — NOT a predicted logS value. Provenance: "inferred".
    """
    from rdkit.Chem import Crippen, Descriptors  # local import: RDKit only

    mol = _as_mol(mol_or_smiles)
    if mol is None:
        return None
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    ab = infer_acid_base(mol)
    ionised = any(c["charge_each"] != 0 for c in ab["centres"])

    factors: List[str] = [
        f"cLogP ≈ {logp:.1f} ({'lipophilic, lowers' if logp > 3 else 'moderate' if logp > 1 else 'hydrophilic, raises'} aqueous solubility)",
        f"TPSA ≈ {tpsa:.0f} Å² ({'high polarity favours' if tpsa > 75 else 'moderate polarity' if tpsa > 40 else 'low polarity disfavours'} solubility)",
    ]
    if ionised:
        factors.append("at least one centre is ionised at pH 7.4, which markedly increases water solubility of that species")

    # Combine the directional factors into a coarse, clearly-hedged tendency.
    score = 0
    score += 1 if logp <= 1 else (-1 if logp > 3 else 0)
    score += 1 if tpsa >= 75 else (-1 if tpsa < 40 else 0)
    score += 1 if ionised else 0
    if score >= 2:
        tendency = "likely good aqueous solubility"
    elif score <= -1:
        tendency = "likely poor aqueous solubility"
    else:
        tendency = "intermediate / borderline aqueous solubility"

    return {
        "tendency": tendency,
        "factors": factors,
        "caveat": "Qualitative expectation from physicochemical principles — not a measured or fitted solubility value.",
        "provenance": "inferred",
    }


# ---------------------------------------------------------------------------
# Continuous charge-vs-pH curve (Henderson–Hasselbalch, for plotting)
# ---------------------------------------------------------------------------
def net_charge_at_ph(mol_or_smiles: MolLike, ph: float) -> float:
    """
    Estimated *fractional* net charge at a given pH, summed over all ionisable
    centres using Henderson–Hasselbalch on the midpoint of each pKa range.

    Unlike `infer_acid_base` (which reports a coarse integer charge with a
    ±1-unit "mixed" band for honest qualitative statements), this returns a
    smooth value suitable for drawing a titration-style curve. Still pure
    principle (no fitted data): for an acid the deprotonated fraction is
    1/(1+10^(pKa−pH)); for a base the protonated fraction is 1/(1+10^(pH−pKaH)).
    """
    groups = detect_functional_groups(mol_or_smiles)
    charge = 0.0
    for g in groups:
        if g["role"] not in ("acid", "base") or not g["pka"]:
            continue
        lo, hi = g["pka"]
        pka_mid = (lo + hi) / 2.0
        if g["role"] == "acid":
            frac_deprot = 1.0 / (1.0 + 10 ** (pka_mid - ph))
            charge += -frac_deprot * g["count"]
        else:  # base: pKaH given
            frac_prot = 1.0 / (1.0 + 10 ** (ph - pka_mid))
            charge += frac_prot * g["count"]
    return charge


def charge_curve(mol_or_smiles: MolLike, lo: float = 0.0, hi: float = 14.0,
                 step: float = 0.2) -> List[Dict[str, float]]:
    """Sampled (pH, net_charge) points across a pH range — for a line chart.
    Empty when there are no ionisable centres (nothing to plot)."""
    mol = _as_mol(mol_or_smiles)
    if mol is None:
        return []
    if not any(g["role"] in ("acid", "base") and g["pka"]
               for g in detect_functional_groups(mol)):
        return []
    points: List[Dict[str, float]] = []
    n = int(round((hi - lo) / step))
    for i in range(n + 1):
        ph = lo + i * step
        points.append({"pH": round(ph, 3),
                       "net_charge": round(net_charge_at_ph(mol, ph), 4)})
    return points


# ---------------------------------------------------------------------------
# One-shot bundle
# ---------------------------------------------------------------------------
def infer_all(mol_or_smiles: MolLike) -> Dict[str, Any]:
    """Convenience: run every inference for a SMILES/mol in one call."""
    mol = _as_mol(mol_or_smiles)
    if mol is None:
        return {}
    return {
        "functional_groups": detect_functional_groups(mol),
        "acid_base": infer_acid_base(mol),
        "stability_flags": infer_stability_flags(mol),
        "solubility": infer_solubility_tendency(mol),
    }

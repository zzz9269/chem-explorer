"""
Fast, offline checks of the rules that decide WHICH data a drug page shows —
label selection, salt handling, scope warnings, missing-section notes, the
comparison table, brand registers and the Ask parser. No network needed.

Each regression test names the real bug it pins down, so a failure says what
came back.
"""
import re

import pytest

import chem_sources as cs
import chem_v4 as v
import data_reports as drp


def _label(generic, product_type="HUMAN PRESCRIPTION DRUG", **extra):
    """A minimal openFDA label result, as `_openfda_fetch` returns it."""
    return {"openfda": {"generic_name": [generic], "product_type": [product_type]}, **extra}


# ---------------------------------------------------------------- label choice
class TestLabelIngredientGate:
    def test_word_inside_another_ingredient_is_not_a_match(self):
        # REGRESSION 2026-10-10: 'glucosamine' showed Chromagen (iron + vitamins)
        # because its folate is 'levomefolate glucosamine'.
        chromagen = _label("FERROUS FUMARATE, ASCORBIC ACID, LEVOMEFOLATE GLUCOSAMINE, ZINC")
        assert not cs._names_ingredient(chromagen, "glucosamine")

    def test_salt_form_of_searched_name_matches(self):
        lomotil = _label("DIPHENOXYLATE HYDROCHLORIDE AND ATROPINE SULFATE")
        assert cs._names_ingredient(lomotil, "diphenoxylate")
        assert cs._names_ingredient(lomotil, "atropine")

    def test_searched_salt_matches_bare_label_name(self):
        assert cs._names_ingredient(_label("DOXYCYCLINE"), "doxycycline hyclate")

    def test_exact_and_case_insensitive(self):
        assert cs._names_ingredient(_label("IBUPROFEN"), "Ibuprofen")
        assert not cs._names_ingredient(_label("IBUPROFEN"), "")

    def test_single_ingredient_detection(self):
        assert cs._is_single_ingredient(_label("METFORMIN HYDROCHLORIDE"))
        assert not cs._is_single_ingredient(_label("SITAGLIPTIN AND METFORMIN HYDROCHLORIDE"))
        assert not cs._is_single_ingredient(_label("GLUCOSAMINE SULFATE, CHONDROITIN"))

    def test_rx_detection(self):
        assert cs._is_rx(_label("X"))
        assert not cs._is_rx(_label("X", "HUMAN OTC DRUG"))

    def test_single_ingredient_outranks_combination(self):
        # REGRESSION 2026-10-06: 'metformin' returned a sitagliptin/metformin label
        single = _label("METFORMIN HYDROCHLORIDE")
        combo = _label("SITAGLIPTIN AND METFORMIN HYDROCHLORIDE",
                       warnings_and_cautions=["x"], adverse_reactions=["x"])
        assert cs._label_score(single) > cs._label_score(combo)

    def test_oral_product_outranks_injection(self):
        # REGRESSION 2026-10-11: acetaminophen showed the IV-injection label
        # (IV dosing errors boxed warning) because only it had a boxed warning
        oral = _label("ACETAMINOPHEN", "HUMAN OTC DRUG")
        oral["openfda"]["route"] = ["ORAL"]
        iv = _label("ACETAMINOPHEN", boxed_warning=["x"], warnings_and_cautions=["x"])
        iv["openfda"]["route"] = ["INTRAVENOUS"]
        assert cs._label_score(oral) > cs._label_score(iv)
        # …unless the drug's main form is the injection (vancomycin)
        assert cs._label_score(iv, "INTRAVENOUS") > cs._label_score(oral, "INTRAVENOUS")

    def test_prescription_outranks_otc_for_same_form(self):
        rx = _label("IBUPROFEN")
        otc = _label("IBUPROFEN", "HUMAN OTC DRUG")
        for x in (rx, otc):
            x["openfda"]["route"] = ["ORAL"]
        assert cs._label_score(rx) > cs._label_score(otc)

    def test_allergen_extract_ranks_last(self):
        allergen = _label("DEXTROSE")
        allergen["openfda"]["pharm_class_epc"] = ["Standardized Chemical Allergen [EPC]"]
        assert cs._label_score(_label("DEXTROSE")) > cs._label_score(allergen)


@pytest.mark.parametrize("raw,expected", [
    ("WARNING: ISCHEMIC HEART DISEASE: WARNING: ISCHEMIC HEART DISEASE See full prescribing "
     "information for complete boxed warning.", "Ischemic heart disease"),
    ("WARNING: RISK OF MEDICATION ERRORS AND HEPATOTOXICITY Take care when prescribing to "
     "avoid dosing errors. More text.",
     "Risk of medication errors and hepatotoxicity — Take care when prescribing to avoid "
     "dosing errors."),
    ("Cardiovascular Thrombotic Events NSAIDs cause an increased risk.",
     "Cardiovascular Thrombotic Events NSAIDs cause an increased risk."),
])
def test_boxed_summary(raw, expected):
    assert v.boxed_summary(raw) == expected


def test_pubchem_requests_are_paced(monkeypatch):
    # PubChem blocks clients that exceed 5 requests/s (2026-10-11: a 429 block
    # emptied every lookup for >10 min) — parallel callers must be spaced out
    import threading
    import time
    monkeypatch.setattr(cs, "_next_slot", {})
    stamps = []

    def hit():
        cs._pace("pubchem.ncbi.nlm.nih.gov")
        stamps.append(time.monotonic())

    threads = [threading.Thread(target=hit) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stamps.sort()
    assert stamps[-1] - stamps[0] >= 5 * cs._MIN_INTERVAL["pubchem.ncbi.nlm.nih.gov"] - 0.02
    t0 = time.monotonic()
    cs._pace("api.fda.gov")                       # unlimited hosts never wait
    assert time.monotonic() - t0 < 0.05


def test_iso_date():
    assert cs._iso_date("20240315") == "2024-03-15"
    assert cs._iso_date("") == "" and cs._iso_date(None) == "" and cs._iso_date("2024") == ""


def test_clean_label_text_strips_cross_references():
    out = cs._clean_label_text(["Take with food [see Warnings and Precautions (5.1)]. "
                                "Adjust dose ( 2.1 , 2.2 )."])
    assert "[see" not in out and "2.1" not in out
    assert out.startswith("Take with food.")


# ------------------------------------------------------------ medicine guards
def test_homeopathic_otc_label_is_not_a_medicine():
    # 'Benzinum' (homeopathic benzene) is an OTC label with no class
    benzinum = {"product_type": "HUMAN OTC DRUG", "pharm_class_epc": []}
    assert not v.openfda_is_medicine(benzinum, None, {})
    assert v.openfda_is_medicine({"product_type": "HUMAN PRESCRIPTION DRUG"}, None, {})


def test_availability_label_from_counts():
    assert v._availability_label({"product_types": {"HUMAN PRESCRIPTION DRUG": 3,
                                                    "HUMAN OTC DRUG": 9}}, {}) \
        == "Prescription & over-the-counter (US)"
    assert v._availability_label({}, {"product_type": "HUMAN OTC DRUG"}) == "Over the counter (US)"
    assert v._availability_label({}, {}) == ""


# --------------------------------------------------------------- salt / names
@pytest.mark.parametrize("name,stripped", [
    ("Atorvastatin calcium", "Atorvastatin"),
    ("metformin hydrochloride", "metformin"),
    ("doxycycline hyclate", "doxycycline"),
    # esters are different products (Avamys vs Flonase) — never stripped
    ("fluticasone furoate", "fluticasone furoate"),
    ("fluticasone propionate", "fluticasone propionate"),
])
def test_salt_suffix_stripping(name, stripped):
    assert v._SALT_SUFFIX_RE.sub("", name) == stripped


@pytest.mark.skipif(not v.HAS_DRUG_LIBRARY, reason="drug library data missing")
def test_library_lookup_ignores_salt_suffix():
    hit = v.find_library_drug("Atorvastatin calcium")
    assert hit and hit["name"].lower() == "atorvastatin"


def test_salt_is_multifragment_and_parent_is_largest_fragment():
    salt = "CCOC(=O)C1(CCN(CC1)CCC(C#N)(c1ccccc1)c1ccccc1)c1ccccc1.Cl"  # diphenoxylate HCl
    assert v.is_multifragment(salt)
    assert "Cl" not in v.largest_fragment_smiles(salt).split(".")
    assert not v.is_multifragment("CC(=O)Oc1ccccc1C(=O)O")


def test_tautomers_of_doxycycline_collapse_to_one_identity():
    # PubChem's 'doxycycline' record and ChEMBL's DOXYCYCLINE ANHYDROUS are
    # drawn differently; the canonical-tautomer key must call them the same
    # substance, and must NOT merge doxycycline with tetracycline.
    pubchem = ("C[C@@H]1[C@H]2[C@@H]([C@H]3[C@@H](C(=O)C(=C([C@]3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)"
               "C(=O)N)N(C)C)O")
    chembl = ("C[C@H]1c2cccc(O)c2C(=O)C2=C(O)[C@]3(O)C(=O)C(C(N)=O)=C(O)[C@@H](N(C)C)"
              "[C@@H]3[C@@H](O)[C@@H]21")
    tetracycline = ("C[C@@]1([C@H]2C[C@H]3[C@@H](C(=O)C(=C([C@]3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)"
                    "C(=O)N)N(C)C)O")
    k = v.canonical_tautomer_inchikey
    assert k(pubchem) and k(pubchem) == k(chembl)
    assert k(tetracycline) != k(pubchem)


# ------------------------------------------------- label provenance & scope
LIPITOR = {"brand_names": ["Lipitor"], "generic_names": ["ATORVASTATIN CALCIUM"],
           "routes": ["ORAL"], "product_type": "HUMAN PRESCRIPTION DRUG",
           "effective_date": "2024-03-15", "retrieved": "2026-10-11",
           "url": "https://dailymed.example/x"}


def test_label_identity_line_names_product_route_and_dates():
    line = v.label_identity_line(LIPITOR)
    for part in ("Lipitor", "atorvastatin calcium", "oral", "prescription",
                 "label revised 2024-03-15", "retrieved 2026-10-11", "DailyMed"):
        assert part in line


def test_us_label_in_singapore_mode_is_flagged():
    sg = {"products": [{"name": "LIPITOR"}, {"name": "ATORVASTATIN SANDOZ"}]}
    w = v.label_scope_warnings(LIPITOR, region="SG", sg=sg)
    assert len(w) == 1 and "US label" in w[0] and "2 HSA-registered products" in w[0]
    w = v.label_scope_warnings(LIPITOR, region="SG", sg={})
    assert "No HSA-registered product" in w[0]
    assert v.label_scope_warnings(LIPITOR, region="US") == []


def test_injection_label_for_an_oral_drug_is_flagged():
    inj = dict(LIPITOR, routes=["INTRAVENOUS"])
    w = v.label_scope_warnings(inj, region="US", drug_routes=["oral", "parenteral"])
    assert len(w) == 1 and "Different formulation" in w[0]
    # an injection-only drug (no oral use known) is not a mismatch
    assert v.label_scope_warnings(inj, region="US", drug_routes=["parenteral"]) == []
    assert v.label_scope_warnings(inj, region="US") == []


def test_label_form_not_sold_locally_is_flagged():
    gel = dict(LIPITOR, routes=["TOPICAL"])
    w = v.label_scope_warnings(gel, region="SG", sg={"products": [{}]}, local_route="ORAL")
    assert any("most Singapore products are oral" in x for x in w)
    assert not any("Different formulation" in x for x in
                   v.label_scope_warnings(LIPITOR, region="SG", sg={"products": [{}]},
                                          local_route="ORAL"))


def test_missing_key_sections_are_named():
    of = {"dosage": "Take one tablet daily with water, adjusting by response " * 2,
          "adverse_reactions": "The most common adverse reactions were headache and nausea " * 2}
    clin = v.missing_key_sections(of, v.CLINICAL_SECTIONS)
    assert "Drug interactions" in clin and "Pregnancy & breastfeeding" in clin
    assert "Dosage & administration" not in clin
    assert "Storage" not in clin                      # minor sections never listed
    safe = v.missing_key_sections(of, v.SAFETY_LABEL_SECTIONS)
    assert "Side effects (adverse reactions)" not in safe
    assert "Warnings & precautions" in safe


def test_otc_panel_counts_as_contraindications():
    of = {"do_not_use": "Do not use if you have ever had an allergic reaction to any pain reliever."}
    assert "Contraindications — who should not take it" not in \
        v.missing_key_sections(of, v.SAFETY_LABEL_SECTIONS)


# ------------------------------------------------------ At-a-Glance sources
def _glance(**kw):
    base = dict(compound_name="X", formula="", dm={}, chembl={}, wikidata={}, wiki={},
                record_text="", drug=None, openfda={}, adverse_events=[], market={},
                region="US", record_use="")
    base.update(kw)
    return v.glance_facts(**base)


def test_glance_tags_each_fact_with_its_source():
    f = _glance(wiki={"extract": "Ibuprofen is an NSAID."},
                openfda={"product_type": "HUMAN PRESCRIPTION DRUG",
                         "indications": "Ibuprofen is indicated for relief of mild to moderate pain.",
                         "pharm_class_epc": ["Nonsteroidal Anti-inflammatory Drug [EPC]"]},
                chembl={"max_phase": 4, "max_phase_label": "Approved",
                        "mechanisms": [{"mechanism": "Cyclooxygenase inhibitor"}]})
    assert f["src"]["what"] == "Wikipedia"
    assert f["src"]["use"] == "US FDA label"
    assert f["src"]["classes"] == "FDA class"
    assert f["src"]["mech"] == "ChEMBL" and f["src"]["status"] == "ChEMBL"


def test_glance_without_label_uses_chembl_and_says_so():
    f = _glance(chembl={"max_phase": 4, "max_phase_label": "Approved",
                        "indications": [{"condition": "Osteoarthritis", "phase": 4},
                                        {"condition": "Gout", "phase": 2}]})
    assert f["use"] == "Osteoarthritis"                # trial-phase uses are not 'approved'
    assert f["src"]["use"] == "ChEMBL approved indications"
    assert "<small>" not in f["use"]                   # tag lives in src, not the text


def test_glance_sg_mode_without_registration_says_so():
    f = _glance(chembl={"max_phase": 4}, region="SG", market={"sg": {}})
    assert f["avail"].startswith("No HSA-registered product")
    assert f["brand_label"] == "Brand names (US)"


# --------------------------------------------------------- comparison table
def test_compare_table_drops_empty_rows_and_marks_gaps():
    md = v.compare_table_md([("A", {"Half-life": "2 h", "Excretion": ""}),
                             ("B", {"Half-life": "", "Excretion": ""})])
    assert "Half-life" in md and "Excretion" not in md
    assert "not stated in sources" in md               # B's half-life gap
    assert md.splitlines()[0].count("|") == 4


def test_table_cell_escapes_pipes_and_truncates():
    cell = v._table_cell("a | b " + "x" * 400)
    assert "\\|" in cell and len(cell) < 300


class TestComparisonTrust:
    """Second review (2026-10-11): scope, links, market warnings, 'none'
    claims and outages in the comparison table."""

    GLANCE = {"is_med": True, "src": {}, "classes": [], "use": "", "mech": "",
              "common": "", "faers_terms": "", "avail": "", "status": ""}

    def _profile(self, monkeypatch, pk_items, scope="", label=None):
        monkeypatch.setattr(v, "_ctx_glance", lambda ctx: dict(self.GLANCE))
        monkeypatch.setattr(v, "_pk_items", lambda ctx: (pk_items, scope))
        return v.compare_profile({"label": label or {}})

    def test_link_survives_prefix_stripping(self, monkeypatch):
        # _pk_key_facts strips 'Half-Life:' from the headline; the URL must stay
        row = self._profile(monkeypatch, [("Half-Life: Ibuprofen has a half-life of 2 hours.",
                                           "PubChem", "https://pubchem.example/1")])
        assert "Ibuprofen has a half-life of 2 hours" in row["Half-life"]
        assert "https://pubchem.example/1" in row["Half-life"]

    def test_parent_scope_is_kept_per_fact(self, monkeypatch):
        tag = "PubChem" + v.PARENT_TAG
        row = self._profile(monkeypatch, [("The half-life is 3 hours.", tag, "")])
        assert "parent" in row["Half-life"] and row.get("_parent_pk")
        notes = v.compare_notes([{"region": "US"}], ["Drug A"], [row])
        assert any("parent compound" in t for _k, t in notes)
        # metadata never becomes a table row
        assert "_parent_pk" not in v.compare_table_md([("Drug A", row)])

    def test_pk_answer_keeps_links_and_scope(self, monkeypatch):
        monkeypatch.setattr(v, "_pk_items", lambda ctx: (
            [("Half-Life: Diphenoxylate has a half-life of 2.5 hours.",
              "PubChem" + v.PARENT_TAG, "https://pubchem.example/13505")], "parent"))
        blocks = v._ans_pk({}, "Diphenoxylate HCl", {})
        pts = [it for b in blocks if b.get("t") == "points" for it in b["items"]]
        assert pts and pts[0][2] == "https://pubchem.example/13505"
        assert any("parent" in b.get("v", "") for b in blocks if b.get("t") == "caption")

    @pytest.mark.parametrize("label,expect", [
        ({"boxed_warning": "WARNING: BLEEDING RISK Warfarin can cause major bleeding."},
         "Bleeding risk — Warfarin can cause major bleeding."),
        ({"has_boxed_section": True}, "couldn't be read"),
        ({"plr_format": True, "routes": ["ORAL"]}, "None on this label (oral)"),
        ({"product_type": "HUMAN OTC DRUG", "routes": ["ORAL"]}, "None on this label (oral)"),
        ({"product_type": "HUMAN PRESCRIPTION DRUG"}, "older-format label"),
    ])
    def test_boxed_status_only_claims_none_when_supported(self, label, expect):
        assert expect in v.boxed_status(label)
        assert "the US label" not in v.boxed_status(label)   # never 'the drug has none'

    def test_market_and_formulation_warnings_shown_once(self):
        gel = {"routes": ["TOPICAL"], "brand_names": ["X"]}
        ctxs = [{"region": "SG", "label": gel, "market": {"sg": {"products": [{}]}},
                 "compound_name": "diclofenac", "clin_chembl": {}},
                {"region": "SG", "label": {"routes": ["ORAL"]}, "market": {},
                 "compound_name": "zzznotadrug", "clin_chembl": {}}]
        notes = v.compare_notes(ctxs, ["Diclofenac", "Other"], [{}, {}])
        texts = [t for _k, t in notes]
        assert sum("US FDA labels" in t for t in texts) == 1          # once, not per drug
        assert any("No HSA-registered product was found for Other" in t for t in texts)
        assert any(t.startswith("**Diclofenac:**") and "most Singapore products are oral" in t
                   for t in texts)

    def test_outage_turns_gaps_into_unavailable(self, monkeypatch):
        monkeypatch.setattr(v, "_sources_down", lambda ctx: ["openFDA"])
        ctx = {"region": "US"}
        assert v._column_gap(ctx) == v._GAP_SOURCE_DOWN
        notes = v.compare_notes([ctx], ["A"], [{}])
        assert any("missing data rather than absent facts" in t for _k, t in notes)
        md = v.compare_table_md([("A", {"Half-life": ""}), ("B", {"Half-life": "2 h"})],
                                gaps=[v._GAP_SOURCE_DOWN, v._GAP_NOT_STATED])
        assert "a source didn't respond" in md and "not stated" not in md

    def test_only_one_drug_with_pk(self):
        md = v.compare_table_md([("A", {"Half-life": "2 h"}), ("B", {"Half-life": ""})])
        row = next(l for l in md.splitlines() if "Half-life" in l)
        assert "2 h" in row and "not stated in sources" in row


# ------------------------------------------------------ Singapore register
@pytest.mark.skipif(not v.HAS_SG_REGISTER, reason="HSA snapshot missing")
class TestHsaRegister:
    def test_plain_product_preferred_over_combination(self):
        assert v.sgr.brand_ingredients("Panadol") == ["paracetamol"]

    def test_combination_brand_lists_every_ingredient(self):
        ings = [i.lower() for i in v.sgr.brand_ingredients("Dhamotil")]
        assert any("atropine" in i for i in ings) and any("diphenoxylate" in i for i in ings)

    def test_misspelt_generic_is_not_a_brand(self):
        assert v.sgr.brand_ingredients("glucossamine") == []

    @pytest.mark.parametrize("name,route", [
        ("diclofenac", "ORAL"),          # tablets here, gels on US labels
        ("paracetamol", "ORAL"), ("timolol", "OPHTHALMIC"),
        ("hydrocortisone", "TOPICAL"), ("glucossamine", ""),
    ])
    def test_main_route_in_singapore(self, name, route):
        assert v.sgr.main_route(name) == route

    def test_market_route_only_applies_in_singapore(self):
        assert v.market_route("SG", "Diclofenac sodium") == "ORAL"   # salt stripped
        assert v.market_route("US", "diclofenac") == ""


# ------------------------------------------------ relationship questions
@pytest.mark.skipif(not v.HAS_CHAT, reason="chat_router missing")
@pytest.mark.parametrize("q,subject,target", [
    # REGRESSION 2026-10-11: answered with glucagon's OWN overdose section
    ("why is glucagon the antidote for beta blocker overdose", "Glucagon", "beta blocker overdose"),
    ("how does naloxone reverse morphine overdose", "Naloxone", "morphine overdose"),
    ("why is atropine given for organophosphate poisoning", "Atropine", "organophosphate poisoning"),
    ("why does metformin cause weight loss", "Metformin", "weight loss"),
])
def test_relation_questions_are_recognised(q, subject, target):
    p = v.cr.parse_query(q)
    assert p["mode"] == "relation"
    assert p["entities"][0]["display"] == subject and p["relation_target"] == target


@pytest.mark.skipif(not v.HAS_CHAT, reason="chat_router missing")
@pytest.mark.parametrize("q", ["does metformin cause weight gain?", "how does metformin work",
                               "why is warfarin dangerous", "metoprolol vs propranolol",
                               "can I take ibuprofen with warfarin"])
def test_ordinary_questions_are_not_relations(q):
    assert v.cr.parse_query(q)["mode"] != "relation"


def test_relation_target_pattern_is_tolerant():
    rx = v.relation_target_re("beta blocker overdose")
    for s in ("beta blockers", "beta-blocker", "β-blocker", "beta-adrenergic blocker"):
        assert rx.search(s), s
    assert v.relation_core("beta blocker overdose") == "beta blocker"


def test_explaining_sentence_outranks_interaction_warning():
    rx = v.relation_target_re("beta blocker overdose")
    subj = re.compile(r"\bglucagon\b", re.I)
    q = "why is glucagon the antidote for beta blocker overdose"
    explain = ("Glucagon in beta blocker overdose increases cAMP in the myocardium, in effect "
               "bypassing the β-adrenergic second messenger system.")
    warning = ("Patients taking beta-blockers may have a transient increase in pulse and blood "
               "pressure when given glucagon.")
    unrelated = "Glucagon raises blood sugar."
    assert v.score_relation_sentence(explain, subj, rx, q) > \
        v.score_relation_sentence(warning, subj, rx, q)
    assert v.score_relation_sentence(unrelated, subj, rx, q) == 0


# --------------------------------------------------------------- Ask parser
@pytest.mark.skipif(not v.HAS_CHAT, reason="chat_router missing")
@pytest.mark.parametrize("q,mode,intent", [
    ("compare metoprolol and propranolol", "compare", "overview"),
    ("metoprolol vs propranolol", "compare", "overview"),
    ("half-life of metoprolol vs propranolol", "compare", "pk"),
    ("can I take ibuprofen with warfarin", "interaction", "interactions"),
    ("side effects of ibuprofen", "answer", "side_effects"),
    ("how does metformin work", "answer", "mechanism"),
])
def test_ask_parser_modes(q, mode, intent):
    p = v.cr.parse_query(q)
    assert p["mode"] == mode and intent in p["intents"]


# ------------------------------------------------------------- data reports
def test_report_roundtrip(tmp_path):
    path = str(tmp_path / "r.jsonl")
    drp.add_report({"problem": "Wrong drug or compound matched", "query": "glucosamine"}, path)
    drp.add_report({"problem": "Other"}, path)
    got = drp.load_reports(path)
    assert [r["problem"] for r in got] == ["Wrong drug or compound matched", "Other"]
    assert got[0]["query"] == "glucosamine" and got[0]["time"]
    assert drp.load_reports(str(tmp_path / "missing.jsonl")) == []

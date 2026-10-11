"""
The Mainstream Drug Reference data (parsed from the user's PDF) and the
Reference mode — offline:

    python -m pytest tests/test_mainstream_ref.py
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

import mainstream_ref as mref

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chem_v4.py")
MODE = "📘 Reference"


def test_parsed_counts_match_the_pdf_cover():
    # cover page: 715 entries, 21 sections, 87 drug classes, 533 on a benchmark list
    assert len(mref.entries()) == 715
    assert len(mref.sections()) == 21
    assert sum(len(s["subsections"]) for s in mref.sections()) == 87
    assert sum(1 for e in mref.entries() if e["tag"]) == 533


def test_every_entry_has_all_five_columns():
    for e in mref.entries():
        for col in ("drug", "class", "mechanism", "uses", "cautions", "subsection"):
            assert e[col].strip(), (e["drug"], col)


def test_ibuprofen_row_is_parsed_whole():
    e = mref.find_entry("ibuprofen")
    assert e["tag"] == "both" and e["class"] == "Propionic-acid NSAID"
    assert e["cautions"].endswith("with ACEi plus diuretic.")
    assert e["subsection"] == "NSAIDs — non-selective COX inhibitors"


@pytest.mark.parametrize("row, expected", [
    ("Enalapril / perindopril / benazepril / captopril",
     ["Enalapril", "perindopril", "benazepril", "captopril"]),
    ("Paracetamol (acetaminophen)", ["Paracetamol"]),
    ("Ketamine (recreational)", ["Ketamine"]),
    ("Levodopa with carbidopa or benserazide", ["Levodopa"]),
    ("Antacids (magnesium/ aluminium salts, alginates)", [None]),
    ("BCG vaccine", [None]),
])
def test_drug_links(row, expected):
    assert [q for _, q in mref.drug_links({"drug": row})] == expected


def test_us_name_finds_the_british_entry():
    assert mref.find_entry("acetaminophen")["drug"] == "Paracetamol (acetaminophen)"


def test_drug_in_two_rows_finds_both():
    assert [e["drug"] for e in mref.find_entries("aspirin")] == \
        ["Aspirin (acetylsalicylic acid)", "Aspirin (low dose)"]


def _card_page():
    import streamlit as st
    import chem_v4 as v
    q = st.session_state.get("q", "Aspirin")
    v.render_reference_card(q, None, q)


@pytest.mark.parametrize("name, label", [
    ("Acetaminophen", "Central analgesic and antipyretic"),   # US name → British row
    ("Metformin hydrochloride", "Biguanide"),                 # salt stripped
    ("Water", None),                                          # not in the reference
])
def test_reference_card_on_the_drug_page(name, label):
    at = AppTest.from_function(_card_page, default_timeout=120)
    at.session_state["q"] = name
    at.run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert labels == ([f"📘 In your Mainstream Drug Reference — {label}"] if label else [])


def test_search_ranks_name_matches_first():
    hits = mref.search("tramadol")
    assert hits[0]["drug"] == "Tramadol"
    assert len(hits) > 1          # also mentioned in other rows' cautions


def _reference_page():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.radio(key="app_mode").set_value(MODE)
    at.run()
    return at


def test_every_section_renders():
    at = _reference_page()
    for opt in at.selectbox(key="ref_section").options[1:]:
        at.selectbox(key="ref_section").set_value(opt)
        at.run()
        assert not at.exception, (opt, at.exception[0].value)


@pytest.mark.network
@pytest.mark.slow
def test_search_table_view_and_click_through_to_explore():
    at = _reference_page()
    at.selectbox(key="ref_section").set_value("All sections")
    at.text_input(key="ref_search").set_value("serotonin syndrome")
    at.radio(key="ref_view").set_value("Table")
    at.run()
    assert not at.exception and len(at.dataframe) == 1
    at.radio(key="ref_view").set_value("Cards")
    at.run()
    btn = next(b for b in at.button if (b.key or "").startswith("ref_q_"))
    query = btn.label.rstrip(" ›")
    btn.click()
    at.run(timeout=400)        # Explore then loads the drug (live sources)
    assert at.session_state["app_mode"] == "🔎 Explore"
    assert at.session_state["chemical_search_input"] == query

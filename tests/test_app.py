"""
Whole-app smoke tests through Streamlit's AppTest (no browser): a page must
build with zero exceptions and show the trust signals — label provenance,
market warning, missing-evidence notes — that the other tests only check as
functions. Slow (each page is a full live lookup):

    python -m pytest -m slow tests/test_app.py
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = [pytest.mark.network, pytest.mark.slow]

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chem_v4.py")
TIMEOUT = 400


def _explore(query, region="Singapore (HSA)"):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.session_state["app_mode"] = "🔎 Explore"
    at.session_state["market_region"] = region
    at.session_state["chemical_search_input"] = query
    at.session_state["lookup_pressed"] = True
    at.run()
    return at


def _text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [x.value for kind in (at.info, at.warning, at.error, at.success) for x in kind]
    return "\n".join(str(p) for p in parts)


def test_glucosamine_shows_no_borrowed_label():
    at = _explore("glucosamine")
    assert not at.exception, [e.value for e in at.exception]
    txt = _text(at)
    assert "Chromagen" not in txt and "Joint Revive" not in txt
    assert "Limited clinical data" in txt
    assert "No FDA label found" in txt                 # safety tab: absence ≠ safety
    assert "Uses & Dosing" not in [t.label for t in at.tabs]
    assert any("Report incorrect" in e.label for e in at.expander)


def test_us_label_in_singapore_mode_is_labelled_as_such():
    at = _explore("atorvastatin")
    assert not at.exception, [e.value for e in at.exception]
    txt = _text(at)
    assert "Source: US FDA label" in txt and "label revised" in txt
    assert "This is the US label" in txt
    assert "· HSA register" in txt or "HSA register" in txt   # SG brands/availability tagged


def test_report_is_saved_and_read_back(tmp_path, monkeypatch):
    import data_reports as drp
    log = str(tmp_path / "reports.jsonl")
    monkeypatch.setattr(drp, "REPORT_FILE", log)       # never touch the real log
    at = _explore("glucosamine")
    at.text_area[0].set_value("benchmark test report")
    next(b for b in at.button if "Save report" in (b.label or "")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    saved = drp.load_reports(log)
    assert len(saved) == 1 and saved[0]["detail"] == "benchmark test report"
    assert saved[0]["query"] == "glucosamine" and saved[0]["cid"] == 439213


def test_ask_compare_in_singapore_warns_about_us_labels():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.session_state["app_mode"] = "💬 Ask"
    at.session_state["market_region"] = "Singapore (HSA)"
    at.run()
    at.chat_input[0].set_value("ibuprofen vs paracetamol").run()
    assert not at.exception, [e.value for e in at.exception]
    warns = " ".join(w.value for w in at.warning)
    assert "come from US FDA labels" in warns
    md = "\n".join(m.value for m in at.markdown)
    assert "None on the US label" not in md


def test_ask_compare_builds_a_side_by_side_table():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.session_state["app_mode"] = "💬 Ask"
    at.run()
    at.chat_input[0].set_value("metoprolol vs propranolol").run()
    assert not at.exception, [e.value for e in at.exception]
    md = "\n".join(m.value for m in at.markdown)
    assert "Metoprolol vs Propranolol" in md or "metoprolol vs propranolol" in md.lower()
    table = next((m for m in md.split("####") if "| **Half-life** |" in m
                  or "| **Drug class** |" in m), "")
    assert table, "comparison table missing"
    assert "| **Drug class** |" in table

"""
data_reports.py — local log of user-reported data problems (wrong drug matched,
wrong/outdated label, missing information…). No Streamlit, no network: each
report is one JSON line in reports/data_reports.jsonl next to this file, so the
records can be reviewed (and turned into benchmark cases in tests/) later.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
REPORT_FILE = os.path.join(REPORT_DIR, "data_reports.jsonl")

PROBLEM_TYPES = [
    "Wrong drug or compound matched",
    "Label is for a different product / formulation",
    "Information is wrong",
    "Information is outdated",
    "Important information is missing",
    "Wrong Singapore (HSA) data",
    "Other",
]


def add_report(entry: Dict[str, Any], path: Optional[str] = None) -> Dict[str, Any]:
    """Append one report (stamped with the time) and return what was saved.
    `path` defaults to REPORT_FILE, read at call time (tests repoint it)."""
    path = path or REPORT_FILE
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_reports(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every saved report, oldest first (unreadable lines skipped)."""
    path = path or REPORT_FILE
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out

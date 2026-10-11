"""Parse The_Mainstream_Drug_Reference_3e.pdf → mainstream_reference.json.

Run once (needs PyMuPDF locally; the app itself only reads the JSON):
    python build_mainstream_reference.py

The PDF is a 5-column table (Drug | Class | Mechanism | Main uses | Key
cautions) per sub-section. Rows are separated by thin horizontal rules, so the
parser buckets text spans into columns by x-position and into rows by those
rules, tracking SECTION / sub-section headings by font as it goes.
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
PDF = HERE / "The_Mainstream_Drug_Reference_3e.pdf"
OUT = HERE / "mainstream_reference.json"

COLS = ["drug", "class", "mechanism", "uses", "cautions"]
TAGS = {"B": "both", "E": "who", "T": "top300"}
FOOTER_Y = 560


def _font(s):
    return s["font"], round(s["size"], 1)


def _col_of(x, bounds):
    for i in range(len(bounds) - 1):
        if bounds[i] - 2 <= x < bounds[i + 1] - 2:
            return i
    return None


def _clean(text):
    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # rejoin words hyphen-wrapped at a line end ("renally- cleared" → "renally-cleared")
    text = re.sub(r"(\w)- (?=[a-z])", r"\1-", text)
    return text


def _contents_names(doc):
    """Proper-case sub-section names from the Contents pages (the table
    headings are printed in caps)."""
    names = {}
    for pn in range(3, 5):
        lines = []
        for b in doc[pn].get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    if _font(s) == ("Carlito", 9.0):
                        lines.append(s["text"])
                    elif _font(s) == ("Carlito-Bold", 9.0):
                        lines.append("·")       # section heading = list break
        for part in " ".join(lines).split("·"):
            part = _clean(part)
            if part:
                names[part.casefold()] = part
    return names


def parse(pdf_path=PDF):
    doc = pymupdf.open(pdf_path)
    proper = _contents_names(doc)
    sections, entries = [], []
    section = subsection = None
    row = None
    bounds = None

    def flush():
        nonlocal row
        if row and any(row["cells"][c] for c in COLS):
            cells = {c: _clean(" ".join(row["cells"][c])) for c in COLS}
            if not cells["drug"] and entries and row.get("continuation"):
                prev = entries[-1]           # row split across a page break
                for c in COLS[1:]:
                    if cells[c]:
                        prev[c] = _clean(prev[c] + " " + cells[c])
            elif cells["drug"]:
                entries.append({
                    "drug": cells["drug"],
                    "tag": row["tag"],
                    "section": section["title"],
                    "section_no": section["no"],
                    "subsection": subsection,
                    **{c: cells[c] for c in COLS[1:]},
                })
        row = None

    def new_row(continuation=False):
        nonlocal row
        flush()
        row = {"cells": {c: [] for c in COLS}, "tag": None,
               "continuation": continuation}

    for pn, page in enumerate(doc):
        if pn < 5:
            continue
        rules = sorted({round(dr["rect"].y0, 1) for dr in page.get_drawings()
                        if dr["type"] == "s" and dr["rect"].height < 1
                        and dr["rect"].x0 < 60})
        events = [(y, "rule", None) for y in rules]
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    if s["bbox"][1] >= FOOTER_Y or not s["text"].strip():
                        continue
                    events.append((s["bbox"][1], "span", s))
        events.sort(key=lambda e: (e[0], e[1] != "rule",
                                   e[2]["bbox"][0] if e[2] else 0))
        if row is not None:
            row["continuation"] = True
        first_on_page = True
        for y, kind, s in events:
            if kind == "rule":
                if bounds:
                    new_row()
                continue
            f, text = _font(s), s["text"]
            if f == ("Carlito-Bold", 10.5):            # "SECTION 3"
                flush()
                no = int(re.search(r"\d+", text).group())
                section = {"no": no, "title": None, "subsections": []}
                sections.append(section)
                bounds = None
            elif f == ("Caladea-Bold", 16.5):          # section title
                section["title"] = _clean(text)
            elif f == ("Carlito-Bold", 9.3):           # sub-section heading
                flush()
                raw = _clean(text)
                subsection = proper.get(raw.casefold(), raw.capitalize())
                section["subsections"].append(subsection)
                bounds = None
            elif f == ("Carlito-Bold", 7.0) and text.strip() == "DRUG":
                flush()
                # column x-bounds from this header row's text starts
                hdr = [e[2]["bbox"][0] for e in events
                       if e[1] == "span" and _font(e[2]) == ("Carlito-Bold", 7.0)
                       and abs(e[0] - y) < 2]
                bounds = sorted(hdr) + [page.rect.width]
                new_row(continuation=first_on_page and bool(entries))
            elif f == ("Carlito-Bold", 7.0):
                continue                                # other header labels
            elif f == ("Carlito-Bold", 7.5):
                continue                                # running page header
            elif bounds:
                if row is None:
                    new_row()
                col = _col_of(s["bbox"][0], bounds)
                if col is None:
                    continue
                if f == ("Carlito-Bold", 5.5) and col == 0 and text.strip() in TAGS:
                    row["tag"] = TAGS[text.strip()]
                else:
                    row["cells"][COLS[col]].append(text)
            if kind == "span" and f[0] != "Carlito-Bold" or f == ("Carlito-Bold", 7.8):
                first_on_page = False
    flush()

    for i, e in enumerate(entries):
        e["id"] = i
    return {"source": PDF.name, "sections": sections, "entries": entries}


if __name__ == "__main__":
    data = parse(Path(sys.argv[1]) if len(sys.argv) > 1 else PDF)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(data["sections"]), "sections,",
          sum(len(s["subsections"]) for s in data["sections"]), "subsections,",
          len(data["entries"]), "entries ->", OUT.name)

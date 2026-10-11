"""build_hsa_register.py — snapshot Singapore's register of therapeutic products.

Source: Health Sciences Authority (HSA) "Listing of Registered Therapeutic
Products" on data.gov.sg (dataset d_767279312753558cbf19d48344577084), published
under the Singapore Open Data Licence v1.0. Each registered product carries its
*forensic classification* — Prescription-Only Medicine (POM), Pharmacy-Only
Medicine (P) or General Sale List (GSL) — which is Singapore's equivalent of the
US Rx/OTC split.

The full dataset is small (~5,500 rows), so it is snapshotted into
`hsa_register_data.json` and read locally by `sg_register.py` (no per-search
network calls). Re-run this script to refresh:

    python build_hsa_register.py
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os

import requests

DATASET_ID = "d_767279312753558cbf19d48344577084"
POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsa_register_data.json")

_CLASS_CODE = {"prescription only": "POM", "pharmacy only": "P", "general sale list": "GSL"}


def fetch_rows():
    poll = requests.get(POLL_URL, timeout=60).json()
    url = (poll.get("data") or {}).get("url")
    if not url:
        raise RuntimeError(f"data.gov.sg poll-download failed: {poll}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig"))))


def main():
    rows = fetch_rows()
    products, unknown = [], set()
    for r in rows:
        fc = (r.get("Forensicclassification") or "").strip()
        code = _CLASS_CODE.get(fc.lower())
        if not code:
            unknown.add(fc)
            continue
        products.append({
            "lic": r.get("LicenceNo", "").strip(),
            "name": r.get("Productname", "").strip(),
            "cls": code,
            "atc": r.get("ATCCode", "").strip(),
            "form": r.get("Dosageform", "").strip(),
            "route": r.get("RouteofAdministration", "").strip(),
            # combination products list ingredients joined by '&&'
            "ings": [s.strip() for s in (r.get("Activeingredients") or "").split("&&") if s.strip()],
            "strength": r.get("Strength", "").strip(),
            "holder": r.get("Licenseholder", "").strip(),
            "approved": r.get("Approvaldate", "").strip(),
        })
    meta = {
        "source": "HSA Listing of Registered Therapeutic Products (data.gov.sg)",
        "dataset_url": f"https://data.gov.sg/datasets/{DATASET_ID}/view",
        "licence": "Singapore Open Data Licence v1.0",
        "fetched": dt.date.today().isoformat(),
        "count": len(products),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "products": products}, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {len(products)} products -> {OUT_PATH}")
    if unknown:
        print("skipped unknown classifications:", sorted(unknown))


if __name__ == "__main__":
    main()

"""
chem_sources.py
---------------
All network / data-source logic for Chemical Explorer, kept free of Streamlit
so it can be cached, reused and unit-tested independently.

Provides:
  * a shared requests.Session with retries + a polite User-Agent
  * a multi-source name->structure resolver chain:
        PubChem  (trivial + systematic names, gives CID)
        OPSIN    (systematic IUPAC names, web service; optional local jar)
        CIR/CACTUS (trivial names, CAS numbers, fallback)
  * PubChem SMILES lookup using the *current* property names
    (SMILES / ConnectivitySMILES) with the deprecated names as a fallback
  * PUG-View flattening for the rich record summary
  * a Wikipedia one-paragraph summary
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote as _urlquote

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 ships with requests; guard just in case
    from urllib3.util.retry import Retry
    _HAVE_RETRY = True
except Exception:  # pragma: no cover
    _HAVE_RETRY = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PUG_TIMEOUT = 12
USER_AGENT = "ChemicalExplorer/1.0 (educational; PubChem+RDKit)"

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUGVIEW_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
OPSIN_BASE = "https://opsin.ch.cam.ac.uk/opsin"
CIR_BASE = "https://cactus.nci.nih.gov/chemical/structure"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"
WIKI_API = "https://en.wikipedia.org/w/api.php"

# Cross-reference / measured-data sources, joined by InChIKey. These surface
# real records that exist in other databases when PubChem's own record is thin.
UNICHEM_COMPOUNDS = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# openFDA — FDA-curated structured drug labels (SPL) + spontaneous adverse-event
# reports (FAERS). Public domain, no API key (rate-limited by IP). These are the
# authoritative regulatory facts for approved drugs: boxed warnings, indications,
# contraindications. FAERS counts are raw report tallies, NOT incidence rates.
OPENFDA_LABEL = "https://api.fda.gov/drug/label.json"
OPENFDA_EVENT = "https://api.fda.gov/drug/event.json"

# Guide to Pharmacology (IUPHAR/BPS) — expert-curated ligand↔target interactions
# with measured affinities. Fills the mechanism/target picture for bioactives
# that aren't formal drugs (so aren't in ChEMBL's approved set).
GTOPDB_BASE = "https://www.guidetopharmacology.org/services"

# NLM RxNorm (public domain) — the authoritative US drug vocabulary; used for a
# drug's single-ingredient brand names.
RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"

# High-value databases shown first in the cross-reference list (by UniChem
# longName). Everything else is still listed, just after these.
_XREF_PRIORITY = [
    "DrugBank", "ChEMBL", "ChEBI", "Guide to Pharmacology", "DrugCentral",
    "KEGG LIGAND", "KEGG", "HMDB", "BindingDB", "PharmGKB", "FDA SRS",
    "PubChem", "Wikipedia", "ZINC", "SureChEMBL", "LINCS", "MetaboLights",
    "Rhea", "NMRShiftDB",
]

# Current PubChem property names (the old CanonicalSMILES/IsomericSMILES are
# deprecated). We request the new ones and fall back to the old keys when an
# older PubChem endpoint still returns them.
SMILES_PROPS = "SMILES,ConnectivitySMILES"

# ---------------------------------------------------------------------------
# Shared session
# ---------------------------------------------------------------------------
_SESSION: Optional[requests.Session] = None


def get_session() -> requests.Session:
    """Return a process-wide Session configured with retries and a UA header."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    if _HAVE_RETRY:
        retry = Retry(
            total=3,
            connect=3,
            # one retry on a read timeout: each try can take the full
            # PUG_TIMEOUT, so read=3 let a hung server (ChEMBL) stall a search
            # for ~50 s; connect errors and 5xx keep their retries
            read=1,
            backoff_factor=0.6,            # 0.6s, 1.2s, 2.4s ...
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        # pool sized for the app's parallel prefetch (several hosts × a few
        # concurrent requests each) so threads don't queue for a connection
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    _SESSION = s
    return s


# Per-host circuit breaker. A degraded service (e.g. ChEMBL answering 500s
# after ~45 s of retries) would otherwise stall every lookup that touches it, one
# after another. After _BREAKER_FAILS consecutive failures a host is skipped
# for _BREAKER_COOLDOWN seconds; any success resets it.
_BREAKER_FAILS = 2
_BREAKER_COOLDOWN = 120.0
_host_state: Dict[str, Dict[str, float]] = {}


def _host_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or "")
    return m.group(1).lower() if m else ""


def host_degraded(host_or_url: str) -> bool:
    """True while a host's breaker is open (recent consecutive failures)."""
    host = _host_of(host_or_url) or (host_or_url or "").lower()
    st = _host_state.get(host)
    return bool(st and st["fails"] >= _BREAKER_FAILS
                and time.time() - st["last"] < _BREAKER_COOLDOWN)


def host_failing(host_or_url: str) -> bool:
    """True if the host's most recent request(s) failed within the cooldown —
    callers use it to avoid caching an empty result caused by an outage."""
    host = _host_of(host_or_url) or (host_or_url or "").lower()
    st = _host_state.get(host)
    return bool(st and time.time() - st["last"] < _BREAKER_COOLDOWN)


def _record(host: str, ok: bool):
    if ok:
        _host_state.pop(host, None)
    else:
        st = _host_state.setdefault(host, {"fails": 0, "last": 0.0})
        st["fails"] += 1
        st["last"] = time.time()


# Live (uncached) HTTP requests made through this module — the app shows the
# count next to its timer. A plain counter; increments under the GIL are safe
# enough for a display statistic.
REQUEST_COUNT = 0


def _count():
    global REQUEST_COUNT
    REQUEST_COUNT += 1


# Client-side pacing for hosts with a published request limit. PubChem allows
# ≤5 requests/s (and ~400/min) per client and answers bursts above that with
# 429s — then blocks the client for many minutes, which empties every lookup.
# The app's parallel prefetch can exceed 5/s on its own, so requests to such a
# host are spaced at least this many seconds apart (across all threads).
_MIN_INTERVAL = {"pubchem.ncbi.nlm.nih.gov": 0.22}
_pace_lock = threading.Lock()
_next_slot: Dict[str, float] = {}


def _pace(host: str):
    """Block until `host` may receive another request under its rate limit."""
    gap = _MIN_INTERVAL.get(host)
    if not gap:
        return
    with _pace_lock:
        now = time.monotonic()
        slot = max(now, _next_slot.get(host, 0.0))
        _next_slot[host] = slot + gap
    if slot > now:
        time.sleep(slot - now)


def _get(url: str, timeout: int = PUG_TIMEOUT) -> Optional[requests.Response]:
    host = _host_of(url)
    if host_degraded(host):
        return None
    _pace(host)
    _count()
    try:
        r = get_session().get(url, timeout=timeout)
    except requests.RequestException:
        # a timeout/connection failure has already been retried by urllib3, so
        # one is enough to open the breaker — otherwise a concurrent caller
        # starts another full-length wait on the same dead host
        _record(host, False)
        _host_state[host]["fails"] = max(_host_state[host]["fails"], _BREAKER_FAILS)
        return None
    # 429 = rate-limited: treat like an outage so the breaker sheds load for a
    # bit (callers fall back fast) instead of every call retrying for seconds
    _record(host, r.status_code < 500 and r.status_code != 429)
    return r


def _post_json(url: str, payload: Dict[str, Any],
               timeout: int = PUG_TIMEOUT) -> Optional[requests.Response]:
    _count()
    try:
        return get_session().post(url, json=payload, timeout=timeout)
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# PubChem helpers
# ---------------------------------------------------------------------------
def _smiles_from_props(prop: Dict[str, Any]) -> str:
    """Pull a SMILES from a PUG property dict, new names first, old as fallback."""
    return (
        (prop.get("SMILES")
         or prop.get("IsomericSMILES")
         or prop.get("ConnectivitySMILES")
         or prop.get("CanonicalSMILES")
         or "")
    ).strip()


def pubchem_name_to_candidates(name: str, max_results: int = 12) -> List[Dict[str, Any]]:
    """
    Resolve a name via PubChem -> list of candidate dicts:
        {"cid": int|None, "smiles": str, "name": str}
    Empty list when nothing matches.
    """
    if not name or not name.strip():
        return []
    quoted = _urlquote(name.strip(), safe="")

    # Step 1: name -> CIDs
    r = _get(f"{PUBCHEM_BASE}/compound/name/{quoted}/cids/TXT")
    cids: List[int] = []
    if r is not None and r.status_code == 200 and r.text.strip():
        for ln in r.text.splitlines():
            ln = ln.strip()
            if ln.isdigit():
                cids.append(int(ln))
            if len(cids) >= max_results:
                break

    # Step 2: CIDs -> SMILES + IUPAC name (batched)
    if cids:
        cid_str = ",".join(str(c) for c in cids)
        r = _get(f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/{SMILES_PROPS},IUPACName/JSON")
        if r is not None and r.status_code == 200:
            props = (r.json().get("PropertyTable", {}) or {}).get("Properties", []) or []
            out = []
            for p in props:
                smi = _smiles_from_props(p)
                if not smi:
                    continue
                out.append({
                    "cid": int(p.get("CID")) if p.get("CID") else None,
                    "smiles": smi,
                    "name": p.get("IUPACName") or "",
                })
            if out:
                return out

    # Fallback: name -> SMILES directly (no CID)
    r = _get(f"{PUBCHEM_BASE}/compound/name/{quoted}/property/{SMILES_PROPS}/JSON")
    if r is not None and r.status_code == 200:
        props = (r.json().get("PropertyTable", {}) or {}).get("Properties", []) or []
        out = []
        for p in props:
            smi = _smiles_from_props(p)
            if smi:
                out.append({"cid": int(p["CID"]) if p.get("CID") else None,
                            "smiles": smi, "name": ""})
        if out:
            return out
    return []


def get_smiles_for_cid(cid: int) -> str:
    """Best-effort SMILES for a CID: property JSON -> 3D SDF -> property TXT."""
    if not cid:
        return ""
    cid = int(cid)

    r = _get(f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{SMILES_PROPS}/JSON")
    if r is not None and r.status_code == 200:
        props = (r.json().get("PropertyTable", {}) or {}).get("Properties", []) or []
        if props:
            smi = _smiles_from_props(props[0])
            if smi:
                return smi

    # 3D SDF -> let the caller convert with RDKit if needed; we return the raw
    # molblock only as a last resort via TXT below. (RDKit lives in chem_v4.)
    r = _get(f"{PUBCHEM_BASE}/compound/cid/{cid}/property/SMILES/TXT")
    if r is not None and r.status_code == 200 and r.text.strip():
        for ln in r.text.splitlines():
            ln = ln.strip()
            if ln:
                return ln
    return ""


def pubchem_parent_cid(cid: int) -> Optional[int]:
    """
    Return PubChem's parent CID for a compound (the free base/acid of a salt,
    the neutral form of an ion, etc.), or None if there is no distinct parent.
    Lets us pull the rich parent record instead of a thin salt record.
    """
    if not cid:
        return None
    r = _get(f"{PUBCHEM_BASE}/compound/cid/{int(cid)}/cids/JSON?cids_type=parent")
    if r is None or r.status_code != 200:
        return None
    try:
        cids = (r.json().get("IdentifierList", {}) or {}).get("CID", []) or []
    except Exception:
        return None
    for c in cids:
        if c and int(c) != int(cid):
            return int(c)
    return None


def get_sdf_3d(cid: int) -> str:
    """Return a 3D SDF molblock for a CID (empty string on failure)."""
    if not cid:
        return ""
    r = _get(f"{PUBCHEM_BASE}/compound/cid/{int(cid)}/SDF?record_type=3d")
    if r is not None and r.status_code == 200 and r.text.strip():
        return r.text
    return ""


def fetch_synonyms(cid: int, limit: int = 400) -> List[str]:
    """Return PubChem synonyms for a CID (capped)."""
    if not cid:
        return []
    r = _get(f"{PUBCHEM_BASE}/compound/cid/{int(cid)}/synonyms/JSON")
    if r is not None and r.status_code == 200:
        info = (r.json().get("InformationList", {}) or {}).get("Information", []) or []
        if info:
            syns = info[0].get("Synonym", []) or []
            return syns[:limit]
    return []


# ---------------------------------------------------------------------------
# OPSIN (systematic IUPAC names)
# ---------------------------------------------------------------------------
def opsin_to_smiles(name: str) -> str:
    """
    Resolve a *systematic* IUPAC name to SMILES.
    Uses the local py2opsin jar when Java is available, otherwise the OPSIN
    web service. Returns "" for names OPSIN cannot parse (e.g. trivial names).
    """
    if not name or not name.strip():
        return ""
    name = name.strip()

    # Local jar (fast, offline) — only if Java is present. Suppress the
    # "Java not installed" warning since the web service below is the fallback.
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from py2opsin import py2opsin  # noqa: WPS433
            res = py2opsin(name)
        if isinstance(res, str) and res.strip():
            return res.strip()
    except Exception:
        pass

    # Web service fallback (no Java needed).
    r = _get(f"{OPSIN_BASE}/{_urlquote(name, safe='')}.smi")
    if r is not None and r.status_code == 200 and r.text.strip():
        return r.text.strip()
    return ""


# ---------------------------------------------------------------------------
# CIR / CACTUS (NCI Chemical Identifier Resolver)
# ---------------------------------------------------------------------------
def cir_to_smiles(identifier: str) -> str:
    """Resolve a name / CAS number / identifier to SMILES via NCI CIR."""
    if not identifier or not identifier.strip():
        return ""
    r = _get(f"{CIR_BASE}/{_urlquote(identifier.strip(), safe='')}/smiles")
    if r is not None and r.status_code == 200 and r.text.strip():
        # CIR can return several lines; take the first non-empty token
        for ln in r.text.splitlines():
            ln = ln.strip()
            if ln and " " not in ln:
                return ln
    return ""


# ---------------------------------------------------------------------------
# Unified resolver chain
# ---------------------------------------------------------------------------
def resolve_name(query: str, max_results: int = 12, use_cir: bool = True) -> Dict[str, Any]:
    """
    Resolve a chemical name through PubChem -> OPSIN -> CIR.

    Returns:
        {
          "source": "pubchem"|"opsin"|"cir"|None,
          "candidates": [ {cid, smiles, name}, ... ],
        }
    PubChem may return multiple candidates (caller picks). OPSIN/CIR return a
    single candidate with no CID.
    """
    query = (query or "").strip()
    if not query:
        return {"source": None, "candidates": []}

    cands = pubchem_name_to_candidates(query, max_results=max_results)
    if cands:
        return {"source": "pubchem", "candidates": cands}

    smi = opsin_to_smiles(query)
    if smi:
        return {"source": "opsin", "candidates": [{"cid": None, "smiles": smi, "name": query}]}

    # CIR is the slowest source (often ~9 s just to fail); callers that have a
    # cheaper fallback (brand-name lookup) can skip it and call cir_resolve later.
    if use_cir:
        return cir_resolve(query)
    return {"source": None, "candidates": []}


def cir_resolve(query: str) -> Dict[str, Any]:
    """Last-resort NCI CIR/CACTUS name → structure, in resolve_name's shape."""
    smi = cir_to_smiles(query)
    if smi:
        return {"source": "cir", "candidates": [{"cid": None, "smiles": smi, "name": query}]}
    return {"source": None, "candidates": []}


# ---------------------------------------------------------------------------
# PUG-View flattening (rich record summaries)
# ---------------------------------------------------------------------------
def parse_section(section: Dict[str, Any]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    heading = section.get("TOCHeading") or "Misc"
    for info in section.get("Information", []) or []:
        val = info.get("Value")
        if isinstance(val, dict) and "StringWithMarkup" in val:
            text = " ".join(s.get("String", "") for s in val.get("StringWithMarkup", []))
            if text:
                result.setdefault(heading, []).append(text)
        elif "Synonym" in info:
            syns = info.get("Synonym") or []
            if isinstance(syns, list):
                result.setdefault(heading, []).extend(s for s in syns if isinstance(s, str))
        elif isinstance(val, dict) and "Number" in val:
            num = val.get("Number")
            if num is not None:
                result.setdefault(heading, []).append(str(num))
        elif isinstance(val, dict) and "String" in val:
            s = val.get("String")
            if s:
                result.setdefault(heading, []).append(s)
    for subsection in section.get("Section", []) or []:
        for k, v in parse_section(subsection).items():
            result.setdefault(k, []).extend(v)
    return result


def build_flat_compound_summary(cid: int, exclude_headings=None) -> Dict[str, Any]:
    """Fetch + flatten the PUG-View record for a CID into {heading: text}."""
    if not cid:
        return {}
    r = _get(f"{PUGVIEW_BASE}/data/compound/{int(cid)}/JSON/", timeout=PUG_TIMEOUT)
    if r is None or r.status_code != 200:
        return {}
    try:
        record = (r.json().get("Record", {}) or {})
    except Exception:
        return {}

    summary: Dict[str, Any] = {
        "Name": record.get("RecordTitle"),
        "PubChem CID": record.get("RecordNumber"),
        "Dates": {
            "Created": record.get("RecordDateCreated"),
            "Modified": record.get("RecordDateModified"),
        },
    }
    for section in record.get("Section", []) or []:
        for k, v in parse_section(section).items():
            if exclude_headings and k in exclude_headings and k.strip().lower() != "record description":
                continue
            summary[k] = " | ".join(v)
    return summary


# ---------------------------------------------------------------------------
# Wikipedia summary
# ---------------------------------------------------------------------------
def wikipedia_summary(title: str) -> Dict[str, Any]:
    """
    Return {"extract": str, "url": str} for a Wikipedia page summary, or {}.
    """
    if not title or not title.strip():
        return {}
    r = _get(f"{WIKI_SUMMARY}/{_urlquote(title.strip(), safe='')}")
    if r is None or r.status_code != 200:
        return {}
    try:
        j = r.json()
    except Exception:
        return {}
    extract = (j.get("extract") or "").strip()
    if not extract:
        return {}
    url = (((j.get("content_urls") or {}).get("desktop") or {}).get("page")) or ""
    return {"extract": extract, "url": url, "title": j.get("title") or title}


# Headings on Wikipedia that carry no compound information — skipped when we
# split an article into sections so the tabs never fill with citation cruft.
_WIKI_SKIP_HEADINGS = {
    "see also", "references", "external links", "further reading", "notes",
    "bibliography", "sources", "citations", "footnotes", "gallery",
}

# `== ... ==` / `=== ... ===` section markers, as emitted by the TextExtracts
# extension when exsectionformat=wiki. The backreference keeps the level's
# equals-count symmetric so stray "=" inside prose can't be mistaken for one.
_WIKI_HEADING_RE = re.compile(r'^(={2,6})\s*(.+?)\s*\1$')


def wikipedia_sections(title: str) -> Dict[str, Any]:
    """
    Fetch a full Wikipedia article as clean plain text and split it into its
    sections. Unlike ``wikipedia_summary`` (lead paragraph only), this exposes
    the whole article — Medical uses, Pharmacology, Adverse effects, Chemistry,
    History, … — so a caller can route each section to the right place and fill
    tabs that PubChem left empty.

    Returns:
        {
          "title": <canonical page title after redirects>,
          "url":   <full page URL>,
          "lead":  <intro text before the first heading>,
          "sections": [ {"level": int, "heading": str, "text": str}, ... ],
        }
    or {} when the page does not exist / has no usable text. Section order is
    the article's own order; empty and citation-only sections are dropped.
    """
    if not title or not title.strip():
        return {}
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": "1",
        "exsectionformat": "wiki",   # keep "== Heading ==" markers so we can split
        "redirects": "1",            # follow "Doxycycline hyclate" -> "Doxycycline"
        "inprop": "url",
        "titles": title.strip(),
        "formatversion": "2",
    }
    _count()
    try:
        r = get_session().get(WIKI_API, params=params, timeout=PUG_TIMEOUT)
    except requests.RequestException:
        return {}
    if r is None or r.status_code != 200:
        return {}
    try:
        pages = (r.json().get("query", {}) or {}).get("pages", []) or []
    except Exception:
        return {}
    if not pages:
        return {}
    page = pages[0]
    if page.get("missing") or not page.get("extract"):
        return {}

    extract = page.get("extract") or ""
    lead_lines: List[str] = []
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for raw_line in extract.split("\n"):
        m = _WIKI_HEADING_RE.match(raw_line.strip())
        if m:
            current = {"level": len(m.group(1)),
                       "heading": m.group(2).strip(), "_lines": []}
            sections.append(current)
        elif current is None:
            lead_lines.append(raw_line)
        else:
            current["_lines"].append(raw_line)

    def _clean(lines: List[str]) -> str:
        text = "\n".join(lines).strip()
        # {{abbrlink}} abbreviations flatten to 'PCPTooltip phencyclidine' in
        # plain-text extracts — drop the widget word, keep both names
        text = re.sub(r"(?<=[\w)\-+])Tooltip\s+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text)

    out_sections: List[Dict[str, Any]] = []
    stack: List[Tuple[int, str]] = []      # enclosing headings, outermost first
    for sec in sections:
        while stack and stack[-1][0] >= sec["level"]:
            stack.pop()
        parents = [h for _lvl, h in stack]
        stack.append((sec["level"], sec["heading"]))
        if sec["heading"].strip().lower() in _WIKI_SKIP_HEADINGS:
            continue
        text = _clean(sec.pop("_lines"))
        if not text:
            continue  # heading with only sub-headings under it — its subs carry text
        # `parents` keeps the context the text-less parent headings carried
        # ('Pharmacology › Pharmacodynamics' above 'Monoamine releasing agent')
        out_sections.append({"level": sec["level"], "heading": sec["heading"],
                             "parents": parents, "text": text})

    if not out_sections and not lead_lines:
        return {}
    return {
        "title": page.get("title") or title,
        "url": page.get("fullurl")
        or ("https://en.wikipedia.org/wiki/" + _urlquote((page.get("title") or title).replace(" ", "_"))),
        "lead": _clean(lead_lines),
        "sections": out_sections,
    }


def wikipedia_title_by_inchikey(inchikey: str) -> str:
    """
    Resolve the English Wikipedia article title for a compound via its Wikidata
    item (matched on InChIKey P235 -> enwiki sitelink). Structure-grounded, so
    it can't drift onto a same-named-but-wrong page the way a raw name search
    can. Returns "" when the compound has no linked English article.
    """
    if not inchikey or not inchikey.strip():
        return ""
    query = (
        'SELECT ?article WHERE { '
        '?item wdt:P235 "%s" . '
        '?article schema:about ?item ; '
        'schema:isPartOf <https://en.wikipedia.org/> . '
        '} LIMIT 1' % inchikey.strip()
    )
    _count()
    try:
        r = get_session().get(WIKIDATA_SPARQL,
                              params={"query": query, "format": "json"},
                              timeout=PUG_TIMEOUT)
    except requests.RequestException:
        return ""
    if r is None or r.status_code != 200:
        return ""
    try:
        rows = r.json().get("results", {}).get("bindings", [])
    except Exception:
        return ""
    if not rows:
        return ""
    article_uri = rows[0].get("article", {}).get("value", "")
    if not article_uri:
        return ""
    from urllib.parse import unquote as _urlunquote
    slug = article_uri.rsplit("/", 1)[-1]
    return _urlunquote(slug).replace("_", " ")


# ---------------------------------------------------------------------------
# Cross-references & measured-data sources (joined by InChIKey)
# ---------------------------------------------------------------------------
def unichem_xrefs(inchikey: str) -> List[Dict[str, str]]:
    """
    Map an InChIKey to the same compound in other databases via UniChem.

    Returns a de-duplicated, priority-ordered list of:
        {"db": <database name>, "id": <their id>, "url": <direct link>}
    so the user can jump to the real record wherever fuller data lives. Empty
    list when UniChem has no entry (common for never-deposited molecules).
    """
    if not inchikey or not inchikey.strip():
        return []
    r = _post_json(UNICHEM_COMPOUNDS,
                   {"type": "inchikey", "compound": inchikey.strip()})
    if r is None or r.status_code != 200:
        return []
    try:
        comps = (r.json().get("compounds") or [])
    except Exception:
        return []
    if not comps:
        return []
    seen: set = set()
    out: List[Dict[str, str]] = []
    for src in comps[0].get("sources", []) or []:
        name = src.get("longName") or src.get("shortName") or ""
        if not name or name in seen:
            continue
        seen.add(name)
        out.append({
            "db": name,
            "id": str(src.get("compoundId") or ""),
            "url": src.get("url") or "",
        })

    def _rank(item: Dict[str, str]) -> tuple:
        try:
            return (0, _XREF_PRIORITY.index(item["db"]))
        except ValueError:
            return (1, item["db"].lower())

    return sorted(out, key=_rank)


_MAX_PHASE_LABEL = {
    4: "Approved", 3: "Phase 3", 2: "Phase 2", 1: "Phase 1",
    0: "Preclinical / not in clinical development",
}


def chembl_summary(inchikey: str) -> Dict[str, Any]:
    """
    Fetch ChEMBL's curated record for a compound by InChIKey: development
    status, approval, route flags, ATC class, and mechanism(s) of action with
    resolved target names. All values are ChEMBL-curated, not predicted.

    Returns {} when the molecule is not in ChEMBL.
    """
    if not inchikey or not inchikey.strip():
        return {}
    # ChEMBL's InChIKey filter query is slow and erratic (0.7–8 s, worse in
    # bad spells). UniChem (EBI's own cross-reference, which indexes ChEMBL)
    # answers in ~0.6 s, so:
    #  * if UniChem knows the compound but lists no ChEMBL entry, ChEMBL hasn't
    #    got it — return {} at once instead of waiting out the slow query
    #    (lets the caller's tautomer/name recovery start ~7 s sooner);
    #  * otherwise its ChEMBL id lets the mechanism → target chain run
    #    alongside the molecule query, used only if the ids agree.
    ex = ThreadPoolExecutor(max_workers=2)
    try:
        f_mol = ex.submit(_get, f"{CHEMBL_BASE}/molecule.json"
                                f"?molecule_structures__standard_inchi_key={inchikey.strip()}")
        xrefs = unichem_xrefs(inchikey)
        hint = next((x["id"] for x in xrefs if x["db"] == "ChEMBL" and x["id"]), "")
        if xrefs and not hint:
            return {}
        f_mech = ex.submit(chembl_mechanisms, hint) if hint else None
        f_ind = ex.submit(chembl_indications, hint) if hint else None
        r = f_mol.result()
        if r is None or r.status_code != 200:
            return {}
        try:
            mols = (r.json().get("molecules") or [])
        except Exception:
            return {}
        if not mols:
            return {}
        m = mols[0]
        chembl_id = m.get("molecule_chembl_id")
        mechanisms = f_mech.result() if f_mech and hint == chembl_id else None
        indications = f_ind.result() if f_ind and hint == chembl_id else None
    finally:
        # don't block on an abandoned molecule query (early return above)
        ex.shutdown(wait=False)
    if mechanisms is None:
        mechanisms = chembl_mechanisms(chembl_id) if chembl_id else []
    if indications is None:
        indications = chembl_indications(chembl_id) if chembl_id else []

    try:
        max_phase = int(float(m.get("max_phase"))) if m.get("max_phase") is not None else None
    except (TypeError, ValueError):
        max_phase = None
    routes = [r_ for r_, on in (("oral", m.get("oral")),
                                ("parenteral", m.get("parenteral")),
                                ("topical", m.get("topical"))) if on]
    # atc_classifications is a list of ATC code strings (e.g. "N02BA01").
    atc = [a for a in (m.get("atc_classifications") or []) if isinstance(a, str)]

    summary: Dict[str, Any] = {
        "chembl_id": chembl_id,
        "pref_name": m.get("pref_name"),
        "molecule_type": m.get("molecule_type"),
        "max_phase": max_phase,
        "max_phase_label": _MAX_PHASE_LABEL.get(max_phase) if max_phase is not None else None,
        "first_approval": m.get("first_approval"),
        "withdrawn": bool(m.get("withdrawn_flag")),
        "oral": bool(m.get("oral")),
        "routes": routes,
        "indication_class": m.get("indication_class"),
        "atc": atc,
        "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/" if chembl_id else "",
        "mechanisms": mechanisms,
        "indications": indications,
    }
    return summary


def chembl_molecule_by_name(name: str, limit: int = 8) -> List[Dict[str, str]]:
    """
    Search ChEMBL by a compound name and return lightweight candidate records:
        {"chembl_id", "pref_name", "inchikey", "smiles"}
    Used to recover the mainstream record when a compound's own structure key
    misses ChEMBL — e.g. a PubChem name that resolves to an unusual tautomer
    depiction whose InChIKey no drug database indexes. The CALLER must verify a
    candidate really is the same compound (e.g. by canonical-tautomer InChIKey)
    before trusting it, since name search alone can return look-alikes.
    """
    if not name or not name.strip():
        return []
    r = _get(f"{CHEMBL_BASE}/molecule/search.json"
             f"?q={_urlquote(name.strip(), safe='')}&limit={int(limit)}")
    if r is None or r.status_code != 200:
        return []
    try:
        mols = (r.json().get("molecules") or [])
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    for m in mols:
        struct = m.get("molecule_structures") or {}
        out.append({
            "chembl_id": m.get("molecule_chembl_id") or "",
            "pref_name": m.get("pref_name") or "",
            "inchikey": struct.get("standard_inchi_key") or "",
            "smiles": struct.get("canonical_smiles") or "",
        })
    return out


def chembl_indications(chembl_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """What a drug is approved for / trialled in, from ChEMBL's drug_indication
    table (curated from ATC, EMA, FDA, DailyMed and ClinicalTrials.gov — so it
    covers non-US drugs that have no FDA label). Returns one row per condition,
    highest phase first:
        [{"condition", "phase" (int|None), "sources": [str], "refs": [{type,id,url}]}]
    """
    if not chembl_id:
        return []
    r = _get(f"{CHEMBL_BASE}/drug_indication.json"
             f"?molecule_chembl_id={_urlquote(chembl_id, safe='')}&limit={int(limit)}")
    if r is None or r.status_code != 200:
        return []
    try:
        rows = r.json().get("drug_indications") or []
    except Exception:
        return []
    by_cond: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cond = (row.get("mesh_heading") or row.get("efo_term") or "").strip()
        if not cond:
            continue
        try:
            phase = int(float(row.get("max_phase_for_ind")))
        except (TypeError, ValueError):
            phase = None
        refs = [{"type": x.get("ref_type") or "", "id": x.get("ref_id") or "",
                 "url": x.get("ref_url") or ""}
                for x in (row.get("indication_refs") or []) if x.get("ref_type")]
        cur = by_cond.get(cond.lower())
        if cur is None:
            by_cond[cond.lower()] = {"condition": cond, "phase": phase, "refs": refs}
        else:
            if phase is not None and (cur["phase"] is None or phase > cur["phase"]):
                cur["phase"] = phase
            cur["refs"].extend(refs)
    out = list(by_cond.values())
    for o in out:
        o["sources"] = sorted({x["type"] for x in o["refs"]})
    out.sort(key=lambda o: (-(o["phase"] if o["phase"] is not None else -1),
                            o["condition"].lower()))
    return out


def chembl_mechanisms(chembl_id: str) -> List[Dict[str, str]]:
    """Curated mechanism(s) of action for a ChEMBL molecule, with target names."""
    if not chembl_id:
        return []
    r = _get(f"{CHEMBL_BASE}/mechanism.json?molecule_chembl_id={chembl_id}")
    if r is None or r.status_code != 200:
        return []
    try:
        mechs = (r.json().get("mechanisms") or [])
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    target_cache: Dict[str, str] = {}
    for me in mechs:
        tid = me.get("target_chembl_id")
        tname = ""
        if tid:
            if tid in target_cache:
                tname = target_cache[tid]
            else:
                tname = _chembl_target_name(tid)
                target_cache[tid] = tname
        out.append({
            "action_type": me.get("action_type") or "",
            "mechanism": me.get("mechanism_of_action") or "",
            "target_id": tid or "",
            "target_name": tname,
        })
    return out


def _chembl_target_name(target_id: str) -> str:
    r = _get(f"{CHEMBL_BASE}/target/{target_id}.json")
    if r is None or r.status_code != 200:
        return ""
    try:
        return r.json().get("pref_name") or ""
    except Exception:
        return ""


def wikidata_by_inchikey(inchikey: str) -> Dict[str, str]:
    """
    Look a compound up in Wikidata by InChIKey (property P235). Returns a short
    structured fact set — often present even when the Wikipedia article is a
    stub or absent. {} when nothing matches.
    """
    if not inchikey or not inchikey.strip():
        return {}
    query = (
        'SELECT ?item ?itemLabel ?desc WHERE { '
        '?item wdt:P235 "%s" . '
        'OPTIONAL { ?item schema:description ?desc FILTER(LANG(?desc)="en") } '
        'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } '
        '} LIMIT 1' % inchikey.strip()
    )
    _count()
    try:
        r = get_session().get(WIKIDATA_SPARQL,
                              params={"query": query, "format": "json"},
                              timeout=PUG_TIMEOUT)
    except requests.RequestException:
        return {}
    if r is None or r.status_code != 200:
        return {}
    try:
        rows = r.json().get("results", {}).get("bindings", [])
    except Exception:
        return {}
    if not rows:
        return {}
    row = rows[0]
    item_uri = row.get("item", {}).get("value", "")
    qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
    return {
        "qid": qid,
        "label": row.get("itemLabel", {}).get("value", ""),
        "description": row.get("desc", {}).get("value", ""),
        "url": item_uri,
    }


# ---------------------------------------------------------------------------
# openFDA — FDA structured drug labels (SPL) + FAERS adverse-event reports
# ---------------------------------------------------------------------------
def _clean_label_text(value: Any) -> str:
    """openFDA label fields arrive as a list of long strings that often repeat
    the section title ('8 WARNINGS AND PRECAUTIONS ...'). Join, strip a leading
    section-number/heading, and collapse whitespace into one clean paragraph."""
    if isinstance(value, list):
        text = " ".join(str(v) for v in value if v)
    else:
        text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^BOXED WARNING\s+", "", text).strip()
    # "10 OVERDOSAGE 10.1 Signs…" — a numbered heading straight before a subsection.
    text = re.sub(r"^\d+\s+[A-Z][A-Z /,&-]{2,}[A-Z]\s+(?=\d)", "", text).strip()
    # OTC Drug Facts panel titles ("Uses Temporarily relieves…", "Directions adults…",
    # "Uses ■ for the temporary relief…"); their square bullets read better as •.
    text = text.replace("■", "•").replace("▪", "•")
    text = re.sub(r"^(?:Uses|Directions|Purpose|Warnings)\s+(?:•\s*)?(?=[A-Za-z])", "",
                  text).strip()
    # A known PLR section title ('1 INDICATIONS AND USAGE IDACIO is…') — the
    # generic rule below can't see where it ends when an all-caps brand follows.
    text = re.sub(r"^\d+(?:\.\d+)*\s+(?:INDICATIONS (?:AND|&) USAGE|DOSAGE (?:AND|&) "
                  r"ADMINISTRATION|DOSAGE FORMS (?:AND|&) STRENGTHS|CONTRAINDICATIONS|"
                  r"WARNINGS (?:AND|&) PRECAUTIONS|ADVERSE REACTIONS|DRUG INTERACTIONS|"
                  r"USE IN SPECIFIC POPULATIONS|DRUG ABUSE (?:AND|&) DEPENDENCE|OVERDOSAGE|"
                  r"MECHANISM OF ACTION|PHARMACODYNAMICS|PHARMACOKINETICS|"
                  r"CLINICAL PHARMACOLOGY)\s+", "", text).strip()
    # Drop a leading "<n> <ALL CAPS HEADING>" that duplicates the field name.
    text = re.sub(r"^\d+(?:\.\d+)*\s+(?:[A-Z][A-Z /,&-]{2,}?)(?=[A-Z][a-z])", "", text).strip()
    # ...or an un-numbered one ("ADVERSE REACTIONS The most…", old-format labels).
    text = re.sub(r"^[A-Z][A-Z /,&-]{2,}[A-Z]\s+(?=[A-Z][a-z])", "", text).strip()
    # Cross-reference noise: "[see Warnings and Precautions (5.1)]", "( 2.1 , 2.2 )".
    text = re.sub(r"\s*\[\s*see [^\]]{0,120}\]", "", text, flags=re.I)
    text = re.sub(r"\s*\(\s*\d+(?:\.\d+)*(?:\s*,\s*\d+(?:\.\d+)*)*\s*\)", "", text)
    return re.sub(r"\s+([.,;])", r"\1", text).strip()


_OPENFDA_SAFE = ':"+'   # keep openFDA search operators/quotes unescaped


def _openfda_fetch(search: str, limit: int = 1) -> List[Dict[str, Any]]:
    """One openFDA label query; returns up to `limit` result dicts ([] on miss)."""
    url = (f"{OPENFDA_LABEL}?search={_urlquote(search, safe=_OPENFDA_SAFE)}"
           f"&limit={max(1, limit)}")
    r = _get(url)
    if r is None or r.status_code != 200:
        return []
    try:
        return r.json().get("results") or []
    except Exception:
        return []


def _is_single_ingredient(res: Dict[str, Any]) -> bool:
    """True when every generic name on the label is one ingredient — a phrase
    search for 'metformin' otherwise happily returns a sitagliptin/metformin
    combination label."""
    names = (res.get("openfda") or {}).get("generic_name") or []
    return bool(names) and not any(re.search(r"\band\b|,|/", n, re.I) for n in names)


def _names_ingredient(res: Dict[str, Any], name: str) -> bool:
    """True when `name` is one of the label's ingredients — the whole ingredient
    or its salt form ('diphenoxylate' → 'diphenoxylate hydrochloride'). openFDA's
    phrase search matches a word anywhere, so 'glucosamine' otherwise hits an
    iron/vitamin label whose folate is 'levomefolate glucosamine'."""
    g = re.sub(r"\s+", " ", (name or "").strip().lower())
    if not g:
        return False
    for gn in (res.get("openfda") or {}).get("generic_name") or []:
        for ing in re.split(r"\s*,\s*|\s+and\s+|\s*/\s*", gn.strip().lower()):
            ing = re.sub(r"\s+", " ", ing).strip()
            if ing and (ing == g or ing.startswith(g + " ") or g.startswith(ing + " ")):
                return True
    return False


def _is_rx(res: Dict[str, Any]) -> bool:
    return "HUMAN PRESCRIPTION DRUG" in ((res.get("openfda") or {}).get("product_type") or [])


def _has_route(res: Dict[str, Any], route: str) -> bool:
    return route.upper() in [r.upper() for r in (res.get("openfda") or {}).get("route") or []]


def _label_score(res: Dict[str, Any], main_route: str = "ORAL") -> Tuple[int, ...]:
    """Rank candidate labels: single-ingredient first, then not an allergen /
    skin-test extract (a 'dextrose' search can return one), then the drug's
    MAIN form (`main_route`: the route most of its US labels share — oral for
    acetaminophen, whose best-documented label is otherwise the IV injection;
    intravenous for vancomycin; ophthalmic for timolol), then prescription,
    modern PLR format (structured 'Warnings and Precautions' + dosage forms), a
    boxed warning, completeness."""
    of = res.get("openfda") or {}
    allergen = any(re.search(r"allergen", c, re.I) for c in of.get("pharm_class_epc") or [])
    plr = bool(res.get("warnings_and_cautions") or res.get("dosage_forms_and_strengths"))
    complete = sum(bool(res.get(k)) for k in ("adverse_reactions", "drug_interactions",
                                              "contraindications", "overdosage"))
    return (int(_is_single_ingredient(res)), int(not allergen),
            int(_has_route(res, main_route)), int(_is_rx(res)), int(plr),
            int(bool(res.get("boxed_warning"))), complete)


def _main_route(generic_name: str) -> str:
    """The route most US labels of a drug share ('ORAL', 'OPHTHALMIC'…), from an
    openFDA count — the form people usually mean. 'ORAL' when unknown."""
    rows = _openfda_count(f'openfda.generic_name:"{generic_name}"', "openfda.route.exact")
    return (rows[0]["term"].upper() if rows else "ORAL")


def openfda_label(generic_name: str = "", brand_name: str = "",
                  unii: str = "", rxcui: str = "", main_route: str = "") -> Dict[str, Any]:
    """Fetch the FDA structured product label for a drug from openFDA.

    Tries the most precise identifier first (UNII, then RxCUI), falling back to
    generic and brand name. Within a tier every sampled label is ranked by
    `_label_score` (single ingredient, main form, prescription, complete…).
    `main_route` names the form to prefer (e.g. the route most products share
    in the user's market); by default it's the route most US labels share.
    Returns a normalised dict of the clinically useful sections, or {} when the
    drug has no FDA label (e.g. a non-drug chemical or a non-US product)."""
    # tiers in precision order; each is [(query, generic name each hit must list
    # as an ingredient — name queries only)]. A tier's queries are pooled.
    tiers: List[List[Tuple[str, str]]] = []
    if unii:
        tiers.append([(f'openfda.unii:"{unii}"', "")])
    if rxcui:
        tiers.append([(f'openfda.rxcui:"{rxcui}"', "")])
    if generic_name:
        g = generic_name.strip().lower()
        # the boxed-warning query makes sure Rx labels are sampled even for
        # drugs with hundreds of OTC labels; ranking decides between them
        tiers.append([(f'openfda.generic_name:"{g}"+AND+_exists_:boxed_warning', g),
                      (f'openfda.generic_name:"{g}"', g)])
    if brand_name:
        tiers.append([(f'openfda.brand_name:"{brand_name.strip().lower()}"', "")])
    if not tiers:
        return {}

    # The queries are independent, so run them all concurrently (each ~1 s),
    # then accept the first tier whose best label is single-ingredient, else
    # the best seen overall.
    flat = [a for t in tiers for a in t]
    with ThreadPoolExecutor(max_workers=len(flat) + 1) as ex:
        f_route = (ex.submit(_main_route, generic_name.strip().lower())
                   if generic_name and not main_route else None)
        fetched = dict(zip(flat, ex.map(lambda a: _openfda_fetch(a[0], limit=25), flat)))
        main_route = (main_route.upper() if main_route
                      else f_route.result() if f_route else "ORAL")

    def score(c):
        return _label_score(c, main_route)

    res: Dict[str, Any] = {}
    for tier in tiers:
        cands: Dict[str, Dict[str, Any]] = {}
        for attempt in tier:
            _, must_name = attempt
            for c in fetched[attempt]:
                # a name-matched combination label only stands in for the drug
                # when it's prescription (Lomotil for diphenoxylate) — OTC blends
                # are supplements ('JBA Joint Revive' for glucosamine)
                if must_name and not (_names_ingredient(c, must_name)
                                      and (_is_single_ingredient(c) or _is_rx(c))):
                    continue
                cands.setdefault(c.get("set_id") or c.get("id") or str(len(cands)), c)
        if not cands:
            continue
        best = max(cands.values(), key=score)
        if not res or score(best) > score(res):
            res = best
        if _is_single_ingredient(res):
            break
    if not res:
        return {}

    of = res.get("openfda", {}) or {}
    spl_set = (of.get("spl_set_id") or [""])[0]
    out = {
        "brand_names": of.get("brand_name") or [],
        "generic_names": of.get("generic_name") or [],
        "manufacturer": (of.get("manufacturer_name") or [""])[0],
        "routes": of.get("route") or [],
        "product_type": (of.get("product_type") or [""])[0],
        "pharm_class_epc": of.get("pharm_class_epc") or [],
        "pharm_class_moa": of.get("pharm_class_moa") or [],
        "rxcui": of.get("rxcui") or [],
        "unii": of.get("unii") or [],
        "boxed_warning": _clean_label_text(res.get("boxed_warning")),
        "indications": _clean_label_text(res.get("indications_and_usage")),
        "contraindications": _clean_label_text(res.get("contraindications")),
        "warnings": _clean_label_text(res.get("warnings_and_cautions")
                                      or res.get("warnings")),
        "dosage": _clean_label_text(res.get("dosage_and_administration")),
        "dosage_forms": _clean_label_text(res.get("dosage_forms_and_strengths")),
        "adverse_reactions": _clean_label_text(res.get("adverse_reactions")),
        "drug_interactions": _clean_label_text(res.get("drug_interactions")),
        "pregnancy": _clean_label_text(res.get("pregnancy")
                                       or res.get("pregnancy_or_breast_feeding")),
        "lactation": _clean_label_text(res.get("lactation")
                                       or res.get("nursing_mothers")),
        "specific_populations": _clean_label_text(res.get("use_in_specific_populations")),
        "pediatric_use": _clean_label_text(res.get("pediatric_use")),
        "geriatric_use": _clean_label_text(res.get("geriatric_use")),
        "overdosage": _clean_label_text(res.get("overdosage")),
        "storage": _clean_label_text(res.get("storage_and_handling")),
        # OTC "Drug Facts" panels use their own section names.
        "do_not_use": _clean_label_text(res.get("do_not_use")),
        "ask_doctor": _clean_label_text(res.get("ask_doctor")),
        "stop_use": _clean_label_text(res.get("stop_use")),
        "mechanism_of_action": _clean_label_text(res.get("mechanism_of_action")),
        "purpose": _clean_label_text(res.get("purpose")),
        # Clinical-pharmacology subsections and abuse/dependence (used by the
        # Ask/chat mode to answer pharmacodynamics / PK / addiction questions).
        "pharmacodynamics": _clean_label_text(res.get("pharmacodynamics")),
        "pharmacokinetics": _clean_label_text(res.get("pharmacokinetics")),
        "clinical_pharmacology": _clean_label_text(res.get("clinical_pharmacology")),
        "abuse": _clean_label_text(res.get("drug_abuse_and_dependence")
                                   or [*(res.get("abuse") or []), *(res.get("dependence") or [])]),
        "controlled_substance": _clean_label_text(res.get("controlled_substance")),
        "when_using": _clean_label_text(res.get("when_using")),
        # plain-language patient text: the Medication Guide is written as patient
        # questions ('How should I take…?', 'What should I avoid…?'); counseling
        # covers missed doses, driving, alcohol; inactive ingredients answer
        # lactose/gelatin questions. Kept raw (not cleaned) so headings survive.
        "medguide": " ".join(res.get("spl_medguide") or res.get("spl_patient_package_insert")
                             or res.get("patient_medication_information") or [])[:30000],
        "patient_info": _clean_label_text(res.get("information_for_patients")),
        "inactive_ingredients": " ".join(res.get("inactive_ingredients")
                                         or res.get("inactive_ingredient") or [])[:3000],
        "spl_set_id": spl_set,
        "url": (f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={spl_set}"
                if spl_set else ""),
        # what the label's STRUCTURE says, so absence can be told from unparsed:
        # a boxed-warning section exists (even if its text didn't survive
        # cleaning); modern PLR format (old-format labels sometimes keep the
        # boxed text inside 'Warnings' with no separate section)
        "has_boxed_section": bool(res.get("boxed_warning")),
        "plr_format": bool(res.get("warnings_and_cautions")
                           or res.get("dosage_forms_and_strengths")),
        # provenance: when this label version took effect, and when we fetched it
        "effective_date": _iso_date(res.get("effective_time")),
        "retrieved": time.strftime("%Y-%m-%d"),
    }
    return out


def _iso_date(yyyymmdd: Any) -> str:
    """'20240315' → '2024-03-15' ('' when absent or malformed)."""
    s = str(yyyymmdd or "").strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if re.fullmatch(r"\d{8}", s) else ""


def _openfda_count(search: str, field: str) -> List[Dict[str, Any]]:
    """openFDA label `count=` aggregation → [{"term","count"}] ([] on failure)."""
    url = (f"{OPENFDA_LABEL}?search={_urlquote(search, safe=_OPENFDA_SAFE)}"
           f"&count={field}")
    r = _get(url)
    if r is None or r.status_code != 200:
        return []
    try:
        return [row for row in (r.json().get("results") or []) if row.get("term")]
    except Exception:
        return []


_GENERIC_BRAND_WORDS = {"RELIEF", "PAIN", "ALLERGY", "COLD", "FLU", "SLEEP", "AID",
                        "HEALTH", "CARE", "EQUATE", "GOOD", "SENSE", "LEADER",
                        "TOPCARE", "SIGNATURE", "SUNMARK", "KIRKLAND", "WALGREENS",
                        "CVS", "RITE", "BASICS", "UP"}
_BRAND_MODIFIERS = {"CHILDRENS", "CHILDREN", "JUNIOR", "STRENGTH", "INFANTS", "IB",
                    "XR", "ER", "SR", "XL", "CR", "LIQUI", "GELS", "EXTRA", "PLUS",
                    "MAXIMUM", "REGULAR", "ADULT", "THE", "AND", "FOR"}


def openfda_market_info(generic_name: str, max_brands: int = 6) -> Dict[str, Any]:
    """How a single-ingredient drug is marketed in the US, from FDA label counts:
    {"product_types": {"HUMAN PRESCRIPTION DRUG": 237, "HUMAN OTC DRUG": 693},
     "brands": ["Advil", "Motrin IB", ...]}.

    Matches the generic name EXACTLY (e.g. "METFORMIN HYDROCHLORIDE") so
    combination products don't count. Brands are the most-labelled brand names,
    excluding generic-named store products ("Ibuprofen", "Care One Ibuprofen")."""
    if not generic_name or not generic_name.strip():
        return {}
    gen = generic_name.strip().upper()
    search = f'openfda.generic_name.exact:"{gen}"'
    with ThreadPoolExecutor(max_workers=2) as ex:      # two independent counts
        f_types = ex.submit(_openfda_count, search, "openfda.product_type.exact")
        f_brands = ex.submit(_openfda_count, search, "openfda.brand_name.exact")
        types = {row["term"]: int(row.get("count") or 0) for row in f_types.result()}
        brand_rows = f_brands.result()
    if not types:
        return {}
    core = gen.split()[0]              # "METFORMIN" from "METFORMIN HYDROCHLORIDE"
    brands: List[str] = []
    kept_words: set = set()
    for row in brand_rows:
        b = row["term"].strip()
        key = b.upper()
        words = set(re.findall(r"[A-Z0-9]+", key))
        # Skip generic-named/store products and descriptive OTC names ("Pain
        # Relief", "Allergy Relief"), and variants of a brand already kept
        # ("Junior Strength Advil" after "Advil").
        if core in key or b.islower() or words & _GENERIC_BRAND_WORDS \
                or words & kept_words:
            continue
        kept_words |= words - _BRAND_MODIFIERS
        brands.append(b.title() if b.isupper() else b)
        if len(brands) >= max_brands:
            break
    return {"product_types": types, "brands": brands}


def rxnorm_rxcui(name: str) -> str:
    """RxNorm concept id for a drug name (exact/normalised match), or ''."""
    if not name or not name.strip():
        return ""
    r = _get(f"{RXNAV_BASE}/rxcui.json?name={_urlquote(name.strip(), safe='')}&search=2")
    if r is None or r.status_code != 200:
        return ""
    try:
        ids = (r.json().get("idGroup") or {}).get("rxnormId") or []
    except Exception:
        return ""
    return ids[0] if ids else ""


def rxnorm_brand_ingredients(brand: str) -> List[str]:
    """Active ingredient(s) of a US brand name via RxNorm ('Pepto-Bismol' →
    ['bismuth subsalicylate'], 'Lomotil' → ['atropine', 'diphenoxylate']).
    For brands PubChem doesn't list as a synonym. [] when unknown."""
    rxcui = rxnorm_rxcui(brand)
    if not rxcui:
        return []
    r = _get(f"{RXNAV_BASE}/rxcui/{_urlquote(str(rxcui), safe='')}/related.json?tty=IN")
    if r is None or r.status_code != 200:
        return []
    try:
        groups = (r.json().get("relatedGroup") or {}).get("conceptGroup") or []
    except Exception:
        return []
    return [c["name"] for g in groups for c in (g.get("conceptProperties") or [])
            if c.get("name")]


def rxnorm_brand_names(rxcui: str, max_brands: int = 5) -> List[str]:
    """Current US brand names for a single ingredient, from RxNorm's branded drug
    products (SBD names end in '[Brand]'). Combination products (names with
    ' / ') are skipped so Janumet doesn't show up as a metformin brand; brands
    with more marketed strengths/forms rank first."""
    if not rxcui:
        return []
    r = _get(f"{RXNAV_BASE}/rxcui/{_urlquote(str(rxcui), safe='')}/related.json?tty=SBD")
    if r is None or r.status_code != 200:
        return []
    try:
        groups = (r.json().get("relatedGroup") or {}).get("conceptGroup") or []
    except Exception:
        return []
    counts: Dict[str, int] = {}
    for g in groups:
        for c in g.get("conceptProperties") or []:
            name = c.get("name") or ""
            if " / " in name:
                continue
            m = re.search(r"\[([^\]]+)\]\s*$", name)
            if m:
                # 'Kaopectate Reformulated Aug 2006' → 'Kaopectate'
                b = re.sub(r"\s+Reformulated\b.*$", "", m.group(1), flags=re.I)
                counts[b] = counts.get(b, 0) + 1
    return [b for b, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:max_brands]


def openfda_adverse_events(generic_name: str, top: int = 8) -> List[Dict[str, Any]]:
    """Most-frequently-reported adverse reactions for a drug from the FAERS
    spontaneous-report database, as [{"term","count"}] (highest first).

    IMPORTANT: these are raw report tallies, not rates — a common, widely-used
    drug accrues more reports, and reports are unverified and subject to bias.
    The caller must present them as 'most-reported', never as 'most-likely'."""
    if not generic_name or not generic_name.strip():
        return []
    search = f'patient.drug.openfda.generic_name:"{generic_name.strip().lower()}"'
    url = (f"{OPENFDA_EVENT}?search={_urlquote(search, safe=_OPENFDA_SAFE)}"
           f"&count=patient.reaction.reactionmeddrapt.exact")
    r = _get(url)
    if r is None or r.status_code != 200:
        return []
    try:
        results = (r.json().get("results") or [])[:max(1, top)]
    except Exception:
        return []
    out = []
    for row in results:
        term = (row.get("term") or "").strip()
        if term:
            out.append({"term": term.title(), "count": int(row.get("count") or 0)})
    return out


# ---------------------------------------------------------------------------
# Guide to Pharmacology (IUPHAR/BPS) — curated ligand↔target interactions
# ---------------------------------------------------------------------------
def _gtopdb_get(path: str) -> Any:
    """GET a GtoPdb web-service path, returning parsed JSON (list/dict) or None.
    The service returns bare JSON arrays, not an envelope."""
    r = _get(f"{GTOPDB_BASE}/{path}")
    if r is None or r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


def guidetopharmacology_by_inchikey(inchikey: str,
                                    max_interactions: int = 8) -> Dict[str, Any]:
    """Look a compound up in the Guide to Pharmacology by InChIKey and return its
    curated target interactions with measured affinities.

    Returns {"ligand_id","name","type","approved","url","interactions":[...]}
    where each interaction is {"target","action","type","affinity","parameter"}.
    {} when the compound isn't catalogued. Target interactions are ordered with
    the ligand's primary targets first. All values are IUPHAR/BPS-curated —
    measured, not predicted."""
    if not inchikey or not inchikey.strip():
        return {}
    ligands = _gtopdb_get(f"ligands?inchikey={_urlquote(inchikey.strip(), safe='')}")
    if not isinstance(ligands, list) or not ligands:
        return {}
    lig = ligands[0]
    lig_id = lig.get("ligandId")
    if not lig_id:
        return {}

    interactions = _gtopdb_get(f"ligands/{lig_id}/interactions")
    rows: List[Dict[str, Any]] = []
    target_cache: Dict[Any, str] = {}
    if isinstance(interactions, list):
        # primary targets first, then by presence of an affinity value
        interactions.sort(key=lambda it: (not it.get("primaryTarget"),
                                          not it.get("affinity")))
        for it in interactions[:max(1, max_interactions)]:
            tid = it.get("targetId")
            tname = ""
            if tid is not None:
                if tid in target_cache:
                    tname = target_cache[tid]
                else:
                    tgt = _gtopdb_get(f"targets/{tid}")
                    tname = (tgt or {}).get("name", "") if isinstance(tgt, dict) else ""
                    target_cache[tid] = tname
            rows.append({
                "target": tname or (f"target {tid}" if tid is not None else "—"),
                "action": it.get("action") or "",
                "type": it.get("type") or "",
                "affinity": it.get("affinity") or "",
                "parameter": it.get("affinityParameter") or "",
            })
    return {
        "ligand_id": lig_id,
        "name": lig.get("name") or "",
        "type": lig.get("type") or "",
        "approved": bool(lig.get("approved")),
        "url": f"https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId={lig_id}",
        "interactions": rows,
    }

"""MeSH KG construction — Phase 1, Step 1.

Fetches MeSH heading annotations for a list of PMIDs via the NCBI
E-utilities efetch endpoint, parses the returned XML, and stores the
result as a mapping ``{pmid: [mesh_term, ...]}``.

Typical usage
-------------
>>> from neurovlm.gnn.mesh import fetch_mesh_for_pmids, check_mesh_coverage
>>> pmid_mesh = fetch_mesh_for_pmids(pmids, api_key="YOUR_KEY", out_path="mesh_annotations.json")
>>> check_mesh_coverage(pmid_mesh, pmids)

Rate limits
-----------
* Without an API key : 3 requests / second
* With an NCBI API key: 10 requests / second (free at
  https://www.ncbi.nlm.nih.gov/account/)

At 1.2 M papers with batch_size=200, expect ~6,000 API calls.
With an API key that runs in ~10 minutes; without, ~33 minutes —
plus retries for failed batches.
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _make_session() -> requests.Session:
    """Return a Session with conservative connection settings.

    Disables urllib3's built-in retry so our own exponential-backoff loop
    has full control.  Sets a generous connect/read timeout so the session
    doesn't hang indefinitely on a stalled connection.
    """
    session = requests.Session()
    # Retry(0) disables urllib3's internal retry — we handle retries ourselves
    adapter = HTTPAdapter(max_retries=Retry(total=0, raise_on_status=False))
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# MeSH tree-number prefixes we care about for neuro KG filtering
NEURO_TREE_PREFIXES = {
    "A08",   # Nervous System — anatomical structures
    "F01",   # Behavior and Behavior Mechanisms
    "F02",   # Psychological Phenomena
    "F03",   # Mental Disorders
    "C10",   # Nervous System Diseases
    "C23",   # Pathological Conditions, Signs and Symptoms
    "D12",   # Amino Acids, Peptides, and Proteins (neurotransmitters)
    "D02",   # Organic Chemicals (includes many neuro drugs)
}


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _parse_efetch_xml(xml_text: str) -> dict[str, list[str]]:
    """Extract {pmid: [descriptor_name, ...]} from an efetch XML response.

    Pulls both the ``<DescriptorName>`` and its ``<QualifierName>``
    sub-headings for completeness, returning them as flat term strings in
    the form ``"Descriptor/Qualifier"`` (qualifier-free terms are returned
    as-is).

    Parameters
    ----------
    xml_text:
        Raw XML string returned by the efetch endpoint.

    Returns
    -------
    dict mapping each PMID string to a list of MeSH term strings.
    """
    result: dict[str, list[str]] = {}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("XML parse error — skipping batch: %s", exc)
        return result

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or pmid_el.text is None:
            continue
        pmid = pmid_el.text.strip()

        terms: list[str] = []
        mesh_list = article.find(".//MeshHeadingList")
        if mesh_list is None:
            result[pmid] = terms
            continue

        for heading in mesh_list.findall("MeshHeading"):
            descriptor_el = heading.find("DescriptorName")
            if descriptor_el is None or descriptor_el.text is None:
                continue
            descriptor = descriptor_el.text.strip()

            qualifiers = [
                q.text.strip()
                for q in heading.findall("QualifierName")
                if q.text
            ]
            if qualifiers:
                for qual in qualifiers:
                    terms.append(f"{descriptor}/{qual}")
            else:
                terms.append(descriptor)

        result[pmid] = terms

    return result


# ---------------------------------------------------------------------------
# Single-batch fetch with exponential backoff
# ---------------------------------------------------------------------------

def _fetch_batch(
    pmids: list[str],
    api_key: Optional[str],
    session: Optional[requests.Session] = None,
    max_retries: int = 5,
    initial_backoff: float = 5.0,
) -> dict[str, list[str]]:
    """Fetch MeSH annotations for one batch of PMIDs.

    Retries up to *max_retries* times with exponential backoff on HTTP
    errors or network failures.  Returns an empty dict for a batch that
    fails all retries (those PMIDs are added to the caller's retry queue).

    Parameters
    ----------
    pmids:
        List of PMID strings; should not exceed 200 (NCBI limit).
    api_key:
        NCBI API key; pass ``None`` to use the 3 req/s rate.
    session:
        Reusable requests.Session.  One is created per-call if not provided.
    max_retries:
        Maximum retry attempts before giving up on this batch.
    initial_backoff:
        Starting back-off interval in seconds (doubles each retry).
        Default 5 s is intentionally conservative: connection-refused errors
        return instantly, so a 1 s backoff burns through all retries in < 5 s
        without giving the server time to recover.
    """
    if session is None:
        session = _make_session()

    params: dict[str, str] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            response = session.get(_EFETCH_URL, params=params, timeout=(10, 90))
            response.raise_for_status()
            return _parse_efetch_xml(response.text)

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 429 or (isinstance(status, int) and status >= 500):
                logger.warning(
                    "HTTP %s on attempt %d/%d — backing off %.1fs",
                    status, attempt + 1, max_retries, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 300.0)
            else:
                logger.error("Non-retriable HTTP %s — skipping batch", status)
                return {}

        except requests.exceptions.ConnectionError as exc:
            # ECONNREFUSED / SSL EOF — the server actively refused or dropped
            # the connection.  Use a longer backoff than for normal network
            # errors because the server needs more time to recover.
            conn_backoff = max(backoff * 2, 30.0)
            logger.warning(
                "Connection error on attempt %d/%d: %s — backing off %.1fs",
                attempt + 1, max_retries, exc, conn_backoff,
            )
            time.sleep(conn_backoff)
            backoff = min(backoff * 2, 300.0)

        except requests.exceptions.Timeout as exc:
            logger.warning(
                "Timeout on attempt %d/%d — backing off %.1fs",
                attempt + 1, max_retries, backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 300.0)

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Network error on attempt %d/%d: %s — backing off %.1fs",
                attempt + 1, max_retries, exc, backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 300.0)

    logger.error("Batch failed after %d retries — %d PMIDs lost", max_retries, len(pmids))
    return {}


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

def fetch_mesh_for_pmids(
    pmids: Iterable[str],
    api_key: Optional[str] = None,
    batch_size: int = 200,
    out_path: Optional[str | Path] = None,
    checkpoint_every: int = 500,
    resume: bool = True,
) -> dict[str, list[str]]:
    """Fetch MeSH heading annotations for every PMID in *pmids*.

    Parameters
    ----------
    pmids:
        Iterable of PMID strings (can be a list, set, or generator).
    api_key:
        NCBI API key.  Strongly recommended at scale — raises the rate
        limit from 3 to 10 requests/second and improves reliability.
        Get one free at https://www.ncbi.nlm.nih.gov/account/
    batch_size:
        Number of PMIDs per API request.  NCBI recommends ≤ 200.
    out_path:
        If provided, annotations are written to this JSON file after every
        *checkpoint_every* batches so that the run can be resumed after a
        crash.
    checkpoint_every:
        Flush *out_path* every this many batches (default 500 = 100 K PMIDs).
    resume:
        If *True* and *out_path* exists, load it and skip already-fetched
        PMIDs.  Set to *False* to re-fetch from scratch.

    Returns
    -------
    dict[str, list[str]]
        ``{pmid: [mesh_term, ...]}`` — PMIDs with no MeSH headings map to
        an empty list (not absent from the dict).
    """
    pmids_list = list(pmids)
    total = len(pmids_list)

    # ----- resume from checkpoint -----
    annotations: dict[str, list[str]] = {}
    if resume and out_path is not None and Path(out_path).exists():
        with open(out_path) as fh:
            annotations = json.load(fh)
        already_done = set(annotations.keys())
        pmids_list = [p for p in pmids_list if p not in already_done]
        logger.info(
            "Resuming: %d already fetched, %d remaining",
            len(already_done), len(pmids_list),
        )

    # ----- rate limit -----
    # Stay well under the NCBI limit: 10 req/s with key → use 0.12 s (≈ 8/s),
    # 3 req/s without key → use 0.40 s (2.5/s).  The extra headroom prevents
    # the server from issuing connection-refused errors under bursty conditions.
    min_interval = 0.12 if api_key else 0.40

    # ----- shared session -----
    session = _make_session()

    # ----- batch loop -----
    failed_batches: list[list[str]] = []
    batch_count = 0
    last_request_time = 0.0

    batches = [
        pmids_list[i : i + batch_size]
        for i in range(0, len(pmids_list), batch_size)
    ]

    for batch in tqdm(batches, desc="Fetching MeSH annotations", unit="batch"):
        # Enforce rate limit
        elapsed = time.time() - last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        last_request_time = time.time()

        result = _fetch_batch(batch, api_key=api_key, session=session)

        if not result:
            # Entire batch failed — queue for retry later
            failed_batches.append(batch)
        else:
            # Mark PMIDs with no MeSH as empty list (not missing)
            for pmid in batch:
                annotations[pmid] = result.get(pmid, [])

        batch_count += 1
        if out_path is not None and batch_count % checkpoint_every == 0:
            _save_annotations(annotations, out_path)
            logger.info("Checkpoint saved (%d PMIDs)", len(annotations))

    # ----- retry failed batches (once more, with smaller sub-batches) -----
    permanently_failed: list[str] = []
    if failed_batches:
        n_failed_pmids = sum(len(b) for b in failed_batches)
        logger.info(
            "Retrying %d failed batches (%d PMIDs) with batch_size=50 and longer backoff",
            len(failed_batches), n_failed_pmids,
        )
        # Brief pause before retrying to let the server settle
        time.sleep(30.0)
        retry_interval = 0.25 if api_key else 0.60  # even more conservative on retry
        for batch in tqdm(failed_batches, desc="Retrying failed batches", unit="batch"):
            sub_batches = [batch[i : i + 50] for i in range(0, len(batch), 50)]
            for sub in sub_batches:
                elapsed = time.time() - last_request_time
                if elapsed < retry_interval:
                    time.sleep(retry_interval - elapsed)
                last_request_time = time.time()

                result = _fetch_batch(sub, api_key=api_key, session=session, max_retries=8, initial_backoff=10.0)
                if not result:
                    permanently_failed.extend(sub)
                else:
                    for pmid in sub:
                        annotations[pmid] = result.get(pmid, [])

    if permanently_failed:
        logger.warning(
            "%d PMIDs permanently failed (all retries exhausted) — "
            "these are recorded as empty lists but may have MeSH terms; "
            "re-run with resume=True after network stabilises to fill them in.",
            len(permanently_failed),
        )
        # Still record them as empty so they show as "fetched" in coverage
        for pmid in permanently_failed:
            if pmid not in annotations:
                annotations[pmid] = []

        # Save failed PMIDs separately for targeted re-fetch
        if out_path is not None:
            failed_path = Path(out_path).with_name("mesh_fetch_failed.json")
            with open(failed_path, "w") as fh:
                json.dump(permanently_failed, fh)
            logger.info("Failed PMIDs saved to %s", failed_path)

    # ----- final save -----
    if out_path is not None:
        _save_annotations(annotations, out_path)
        logger.info("Final annotations saved to %s", out_path)

    logger.info("Done. %d / %d PMIDs in annotations dict.", len(annotations), total)
    return annotations


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_annotations(
    annotations: dict[str, list[str]],
    path: str | Path,
) -> None:
    """Write *annotations* to a JSON file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(annotations, fh)
    tmp.replace(path)


def load_annotations(path: str | Path) -> dict[str, list[str]]:
    """Load a previously saved annotations JSON file."""
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def check_mesh_coverage(
    annotations: dict[str, list[str]],
    all_pmids: Iterable[str],
    failed_pmids: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> dict[str, float | int]:
    """Report what fraction of PMIDs received at least one MeSH term.

    Parameters
    ----------
    annotations:
        The dict returned by :func:`fetch_mesh_for_pmids`.
    all_pmids:
        The full list of PMIDs you intended to fetch.
    failed_pmids:
        Optional list of PMIDs that permanently failed all retries.
        Load from ``mesh_fetch_failed.json`` if present.  When provided,
        the summary breaks out fetch-errors vs genuinely unindexed papers.
    verbose:
        Print a summary table.

    Returns
    -------
    dict with keys: ``total``, ``fetched``, ``with_mesh``, ``coverage_pct``,
    ``failed_fetch`` (0 if *failed_pmids* not provided), ``truly_no_mesh``.
    """
    all_pmids = list(all_pmids)
    failed_set = set(failed_pmids) if failed_pmids else set()

    total = len(all_pmids)
    fetched = sum(1 for p in all_pmids if p in annotations)
    with_mesh = sum(1 for p in all_pmids if annotations.get(p))
    failed_fetch = len(failed_set & set(all_pmids))
    truly_no_mesh = total - with_mesh - failed_fetch

    coverage = 100.0 * with_mesh / total if total else 0.0

    stats = {
        "total": total,
        "fetched": fetched,
        "with_mesh": with_mesh,
        "coverage_pct": round(coverage, 2),
        "failed_fetch": failed_fetch,
        "truly_no_mesh": max(truly_no_mesh, 0),
    }

    if verbose:
        print(f"Total PMIDs            : {total:>10,}")
        print(f"Successfully fetched   : {fetched:>10,}  ({100*fetched/total:.1f}%)")
        print(f"With ≥1 MeSH heading   : {with_mesh:>10,}  ({coverage:.1f}%)")
        if failed_set:
            print(f"  └─ failed fetch      : {failed_fetch:>10,}  (connection errors — re-fetchable)")
            print(f"  └─ genuinely no MeSH : {max(truly_no_mesh,0):>10,}  (not in MEDLINE or unindexed)")
        else:
            print(f"No MeSH / not indexed  : {total - with_mesh:>10,}")
            print("   (run with failed_pmids=load_failed_pmids(path) to split error vs unindexed)")
        if coverage < 90:
            print(
                "WARNING: Coverage < 90%. Likely causes: (1) papers not yet indexed "
                "in MEDLINE, (2) journals not covered by MEDLINE, (3) connection errors "
                "during fetch that weren't retried."
            )

    return stats


def load_failed_pmids(out_path: str | Path) -> list[str]:
    """Load the list of PMIDs that permanently failed from a prior run.

    The file is written automatically by :func:`fetch_mesh_for_pmids` when
    batches exhaust all retries.  Pass the result to :func:`check_mesh_coverage`
    to distinguish fetch errors from genuinely unindexed papers.

    Parameters
    ----------
    out_path:
        Path to ``mesh_annotations.json`` (the main output file); this
        function looks for ``mesh_fetch_failed.json`` in the same directory.
    """
    failed_path = Path(out_path).with_name("mesh_fetch_failed.json")
    if not failed_path.exists():
        return []
    with open(failed_path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Flat DataFrame export
# ---------------------------------------------------------------------------

def annotations_to_dataframe(
    annotations: dict[str, list[str]],
) -> "pd.DataFrame":  # noqa: F821
    """Convert the annotations dict to a long-form DataFrame.

    Returns
    -------
    pandas.DataFrame with columns ``pmid``, ``mesh_term``.
    Each row is one (PMID, MeSH term) pair.  PMIDs with no MeSH terms
    are excluded.
    """
    import pandas as pd

    rows = [
        {"pmid": pmid, "mesh_term": term}
        for pmid, terms in annotations.items()
        for term in terms
    ]
    return pd.DataFrame(rows, columns=["pmid", "mesh_term"])


# ---------------------------------------------------------------------------
# MeSH hierarchy (Step 2 — descriptor XML)
# ---------------------------------------------------------------------------

def parse_mesh_descriptor_xml(
    xml_path: str | Path,
) -> "pd.DataFrame":  # noqa: F821
    """Parse the annual MeSH descriptor XML (desc2025.xml) into a DataFrame.

    Downloads available at:
    https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/

    Parameters
    ----------
    xml_path:
        Local path to ``desc2025.xml`` (or any year's descriptor file).

    Returns
    -------
    pandas.DataFrame with columns:
        ``ui``             — DescriptorUI (e.g. ``"D006726"``)
        ``name``           — DescriptorName
        ``tree_numbers``   — list of tree-number strings
        ``synonyms``       — list of entry-term / synonym strings
    """
    import pandas as pd

    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    for desc in root.findall("DescriptorRecord"):
        ui_el = desc.find("DescriptorUI")
        name_el = desc.find("DescriptorName/String")
        if ui_el is None or name_el is None:
            continue

        ui = ui_el.text.strip()
        name = name_el.text.strip()

        tree_numbers = [
            tn.text.strip()
            for tn in desc.findall(".//TreeNumber")
            if tn.text
        ]

        synonyms = [
            st.text.strip()
            for st in desc.findall(".//Term/String")
            if st.text and st.text.strip().lower() != name.lower()
        ]

        records.append(
            {
                "ui": ui,
                "name": name,
                "tree_numbers": tree_numbers,
                "synonyms": synonyms,
            }
        )

    return pd.DataFrame(records, columns=["ui", "name", "tree_numbers", "synonyms"])


def build_mesh_hierarchy_triples(
    descriptor_df: "pd.DataFrame",  # noqa: F821
) -> "pd.DataFrame":  # noqa: F821
    """Derive broader/narrower triples from MeSH tree numbers.

    For every descriptor with at least two tree-number levels, emits:
    ``(child_ui, narrower_term_of, parent_ui)``

    A parent is the descriptor whose tree number is the direct prefix
    (one fewer dot-delimited component) of the child's tree number.

    Parameters
    ----------
    descriptor_df:
        DataFrame returned by :func:`parse_mesh_descriptor_xml`.

    Returns
    -------
    pandas.DataFrame with columns:
        ``subject_id``, ``relation_type``, ``object_id``, ``source``, ``weight``
    """
    import pandas as pd

    # Build tree_number → ui lookup (one entry per tree-number)
    tn_to_ui: dict[str, str] = {}
    for _, row in descriptor_df.iterrows():
        for tn in row["tree_numbers"]:
            tn_to_ui[tn] = row["ui"]

    triples = []
    for _, row in descriptor_df.iterrows():
        child_ui = row["ui"]
        for tn in row["tree_numbers"]:
            parts = tn.split(".")
            if len(parts) < 2:
                continue
            parent_tn = ".".join(parts[:-1])
            parent_ui = tn_to_ui.get(parent_tn)
            if parent_ui and parent_ui != child_ui:
                triples.append(
                    {
                        "subject_id": child_ui,
                        "relation_type": "narrower_term_of",
                        "object_id": parent_ui,
                        "source": "mesh_hierarchy",
                        "weight": 1.0,
                    }
                )

    df = pd.DataFrame(
        triples,
        columns=["subject_id", "relation_type", "object_id", "source", "weight"],
    )
    return df.drop_duplicates(subset=["subject_id", "relation_type", "object_id"])

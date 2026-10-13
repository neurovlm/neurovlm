#!/usr/bin/env python
"""Table-caption audit (comment 3.7): PMC open-access coverage of the corpus and the captions of every table in those papers.

Maps PMIDs to PMCIDs, downloads the open-access articles with pubget (resumable chunks), parses their tables and compares
each caption with the paper's title + abstract (token overlap, MiniLM cosine).
    run_analysis.py --skip-download --n-jobs 12   # analyse the cached download in outputs/pubget_data (CPU, < 1 min)
    run_analysis.py                               # resume the download (network), then analyse
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

HERE = Path(__file__).resolve().parent
IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL = "neurovlm_caption_audit"
SPLITS = ("train", "val", "test", "unassigned")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# (a) corpus
def load_corpus() -> tuple[pd.DataFrame, str]:
    from _hf_pins import hf_file

    def hf(fn: str) -> Path:  # at the revision pinned in scripts/hf_manifest.json, sha256-checked
        return hf_file("neurovlm/neuro_image_papers", fn)

    pubs = pd.read_parquet(hf("publications.parquet"))
    coords = pd.read_parquet(hf("coordinates.parquet"), columns=["pmid"])
    pubs["pmid"] = pubs["pmid"].astype("int64")
    pubs["n_coords_released"] = pubs["pmid"].map(coords["pmid"].astype("int64").value_counts()).fillna(0).astype(int)
    pubs["split"] = "unassigned"
    # the pinned split flags (set-identical to the final run's pmids_split.pt); never the data-dir copy
    for k in ("train", "val", "test"):
        pubs.loc[pubs[k].fillna(False).astype(bool), "split"] = k
    split_source = "publications.parquet flags"
    pubs = pubs.rename(columns={"name": "title", "description": "abstract", "pmcid": "pmcid_released"})
    return pubs[["pmid", "pmcid_released", "doi", "title", "abstract", "split", "n_coords_released"]], split_source


# (b) PMID -> PMCID, open-access status
def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = f"{TOOL}/1.0 (python-requests)"
    return s


def _get(sess, url: str, *, method: str = "GET", tries: int = 5, **kw):
    for i in range(tries):
        try:
            r = sess.request(method, url, timeout=60, **kw)
            if r.status_code == 200:
                return r
            log(f"  HTTP {r.status_code} for {url} (try {i + 1})")
        except Exception as exc:
            log(f"  request error {exc!r} (try {i + 1})")
        time.sleep(2 * 2**i)
    raise RuntimeError(f"giving up on {url}")


def _urlget_json(url: str, params: dict[str, str], tries: int = 6) -> dict[str, Any]:
    """GET with urllib (the PMC web front-end answers 403 to python-requests' client fingerprint; urllib/curl work)."""
    import urllib.parse
    import urllib.request

    full = url + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(full, headers={"User-Agent": f"{TOOL}/1.0"}), timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            log(f"  idconv error {exc!r} (try {i + 1})")
            time.sleep(2 * 2**i)
    raise RuntimeError(f"giving up on {url}")


def map_pmids(pmids: list[int], cache: Path) -> pd.DataFrame:
    """PMID -> PMCID via the PMC ID converter (200 IDs/request); resumable CSV cache."""
    cols = ["pmid", "pmcid", "doi_idconv", "idconv_status"]
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=cols)
    todo = sorted(set(int(p) for p in pmids) - set(done["pmid"].astype("int64")))
    if todo:
        log(f"ID converter: {len(todo)} PMIDs to map ({len(done)} cached)")
        new: list[dict[str, Any]] = []
        for i in range(0, len(todo), 200):
            batch = todo[i:i + 200]
            js = _urlget_json(IDCONV_URL, {"ids": ",".join(map(str, batch)), "idtype": "pmid", "format": "json", "tool": TOOL})
            recs = {str(rec.get("requested-id", rec.get("pmid"))): rec for rec in js.get("records", [])}
            for p in batch:
                rec = recs.get(str(p), {})
                pmc = rec.get("pmcid")
                new.append({"pmid": p, "pmcid": int(pmc[3:]) if pmc else None, "doi_idconv": rec.get("doi"),
                            "idconv_status": rec.get("errmsg", "ok" if pmc else "no record")})
            if (i // 200) % 25 == 24 or i + 200 >= len(todo):
                done = pd.concat([done, pd.DataFrame(new, columns=cols)], ignore_index=True)
                new = []
                done.to_csv(cache, index=False)
                log(f"  mapped {min(i + 200, len(todo))}/{len(todo)}")
            time.sleep(0.34)
    done = done.drop_duplicates("pmid", keep="last")
    done["pmid"] = done["pmid"].astype("int64")
    done["pmcid"] = pd.to_numeric(done["pmcid"], errors="coerce").astype("Int64")
    return done


def oa_status(pmcids: list[int], cache: Path, api_key: str | None) -> pd.DataFrame:
    """PMC open-access membership with EPost + ESearch 'open access[filter]' (pubget's own filter)."""
    from lxml import etree

    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["pmcid", "is_oa"])
    todo = sorted(set(int(p) for p in pmcids) - set(done["pmcid"].astype("int64")))
    if todo:
        log(f"OA status: {len(todo)} PMCIDs to check ({len(done)} cached)")
        sess = _session()
        key = {"api_key": api_key} if api_key else {}
        rows = []
        for i in range(0, len(todo), 2000):
            batch = todo[i:i + 2000]
            r = _get(sess, EUTILS + "epost.fcgi", method="POST", data={"db": "pmc", "id": ",".join(map(str, batch)), "tool": TOOL, **key})
            x = etree.fromstring(r.content)
            we, qk = x.findtext("WebEnv"), x.findtext("QueryKey")
            time.sleep(0.2)
            r = _get(sess, EUTILS + "esearch.fcgi", method="POST",
                     data={"db": "pmc", "term": "open access[filter]", "WebEnv": we, "query_key": qk, "retmode": "json",
                           "retmax": 10000, "tool": TOOL, **key})
            oa = {int(u) for u in r.json()["esearchresult"].get("idlist", [])}
            rows += [{"pmcid": p, "is_oa": p in oa} for p in batch]
            log(f"  OA checked {min(i + 2000, len(todo))}/{len(todo)} ({len(oa)} OA in batch)")
            time.sleep(0.2)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(cache, index=False)
        cache.with_suffix(".json").write_text(json.dumps({"checked_on": str(date.today()), "filter": "open access[filter]"}))
    done["pmcid"] = done["pmcid"].astype("int64")
    done["is_oa"] = done["is_oa"].astype(str).str.lower().isin({"true", "1"})
    return done.drop_duplicates("pmcid", keep="last")


# (c) pubget download
def pubget_bin() -> str:
    b = Path(sys.executable).parent / "pubget"
    return str(b) if b.exists() else "pubget"


def pubget_version() -> str:
    try:
        return subprocess.run([pubget_bin(), "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return "unknown"


def _info(d: Path) -> dict[str, Any]:
    f = d / "info.json"
    return json.loads(f.read_text()) if f.exists() else {}


def chunk_status(chunk_dir: Path) -> dict[str, Any]:
    """Completion of pubget's download / extract_articles / extract_data steps in one chunk directory."""
    st = {"dir": str(chunk_dir), "download": False, "extract_articles": False, "extract_data": False, "n_articles": None}
    if not chunk_dir.exists():
        return st
    st["download"] = bool(_info(chunk_dir / "articlesets").get("is_complete"))
    art = _info(chunk_dir / "articles")
    st["extract_articles"] = bool(art.get("is_complete"))
    st["n_articles"] = art.get("n_articles")
    ed = chunk_dir / "subset_allArticles_extractedData"
    st["extract_data"] = bool(_info(ed).get("is_complete")) and (ed / "coordinates.csv").exists()
    return st


def run_pubget(pmcid_chunks: list[list[int]], prefix: str, pubget_dir: Path, n_jobs: int, keep_articlesets: bool) -> list[dict[str, Any]]:
    """Run ``pubget run`` once per chunk (resumable); returns the per-chunk status records."""
    pubget_dir.mkdir(parents=True, exist_ok=True)
    (pubget_dir / "logs").mkdir(exist_ok=True)
    records = []
    for ci, chunk in enumerate(pmcid_chunks):
        alias = f"{prefix}_{ci:02d}"
        ids_file = pubget_dir / f"{alias}_pmcids.txt"
        ids_file.write_text("\n".join(map(str, chunk)) + "\n")
        cmd = [pubget_bin(), "run", str(pubget_dir), "--pmcids_file", str(ids_file), "--alias", alias,
               "--n_jobs", str(n_jobs), "--log_dir", str(pubget_dir / "logs")]
        link = pubget_dir / alias
        for attempt in range(3):
            st = chunk_status(link.resolve()) if link.exists() else {}
            if st.get("extract_data"):
                break
            log(f"pubget chunk {alias} ({len(chunk)} PMCIDs), attempt {attempt + 1}: {' '.join(cmd)}")
            t = time.perf_counter()
            rc = subprocess.run(cmd).returncode
            log(f"  pubget exit code {rc} after {time.perf_counter() - t:.0f} s")
            st = chunk_status(link.resolve()) if link.exists() else {}
            if st.get("extract_data") and st.get("download"):
                break
            if attempt == 1 and link.exists() and not st.get("download"):
                # an interrupted download resumes from the stored Entrez WebEnv, which expires after a few hours:
                # start the chunk's download afresh (new EPost) on the last attempt.
                adir = link.resolve() / "articlesets"
                log(f"  resetting stale download state in {adir}")
                for f in list(adir.glob("articleset_*.xml")) + [adir / "info.json"]:
                    f.unlink(missing_ok=True)
        st = chunk_status(link.resolve()) if link.exists() else {"dir": None}
        st.update(chunk=alias, n_pmcids_requested=len(chunk))
        if st.get("download") and st.get("extract_articles") and not keep_articlesets and link.exists():
            n = 0
            for f in (link.resolve() / "articlesets").glob("articleset_*.xml"):
                f.unlink()
                n += 1
            if n:
                log(f"  pruned {n} raw articleset batch files of {alias}")
        records.append(st)
    (pubget_dir / f"{prefix}_status.json").write_text(json.dumps(records, indent=1))
    return records


# (d) parse tables
_WS = re.compile(r"\s+")
_LEAD_LABEL = re.compile(r"^\s*(?:supplementary\s+)?(?:table|tab\.)\s*[A-Z]?\d+[A-Za-z]?\s*[.:|\-–—]?\s*", re.I)
_XYZ_CELLS = re.compile(r"^\(?\s*[xyz]\s*\)?$|^[xyz]\s*\(mm\)$", re.I)
_COORD_HEAD = re.compile(r"\bmni\b|talairach|\btal\b|coordinat|\bx\s*[,/;]?\s*y\s*[,/;]?\s*z\b", re.I)


def _txt(el) -> str:
    return _WS.sub(" ", "".join(el.itertext())).strip() if el is not None else ""


def _block_text(el) -> str:
    """Text of a JATS caption/foot with a space between block children (title, p, fn...)."""
    if el is None:
        return ""
    parts = [(el.text or "").strip()]
    for ch in el:
        parts.append(_txt(ch))
        parts.append((ch.tail or "").strip())
    return _WS.sub(" ", " ".join(p for p in parts if p)).strip()


def parse_article(article_dir: str) -> list[dict[str, Any]]:
    from lxml import etree

    d = Path(article_dir)
    pmcid = int(d.name.split("_")[1])
    tx = d / "tables" / "tables.xml"
    if not tx.exists():
        return [{"pmcid": pmcid, "table_nb": -1}]  # article without tables.xml (XSLT failure)
    try:
        root = etree.parse(str(tx)).getroot()
    except Exception:
        return [{"pmcid": pmcid, "table_nb": -1}]
    rows = []
    for nb, t in enumerate(root.iterfind("extracted-table")):
        orig = t.find("original-table")
        tw = None
        if orig is not None:
            tw = next((e for e in orig.iter() if isinstance(e.tag, str) and etree.QName(e).localname == "table-wrap"), None)
        cap_el = foot_el = None
        if tw is not None:
            for ch in tw:
                if not isinstance(ch.tag, str):
                    continue
                ln = etree.QName(ch).localname
                if ln == "caption" and cap_el is None:
                    cap_el = ch
                elif ln == "table-wrap-foot" and foot_el is None:
                    foot_el = ch
        cap_title = ""
        if cap_el is not None:
            ti = next((c for c in cap_el if isinstance(c.tag, str) and etree.QName(c).localname == "title"), None)
            cap_title = _txt(ti)
        caption = _block_text(cap_el) if cap_el is not None else _WS.sub(" ", t.findtext("table-caption") or "").strip()
        foot = _block_text(foot_el) if foot_el is not None else _WS.sub(" ", t.findtext("table-wrap-foot") or "").strip()
        # transformed (XHTML) table geometry
        trs = [e for e in t.iter() if isinstance(e.tag, str) and etree.QName(e).localname == "tr" and _has_ancestor(e, "transformed-table")]
        head_trs = [e for e in trs if _has_ancestor(e, "thead")]
        n_cols = 0
        for tr in trs:
            n_cols = max(n_cols, sum(int(c.get("colspan", "1") or 1) if str(c.get("colspan", "1")).isdigit() else 1
                                     for c in tr if isinstance(c.tag, str) and etree.QName(c).localname in ("td", "th")))
        head_text = " | ".join(_txt(tr) for tr in (head_trs or trs[:1]))
        head_cells = [_txt(c) for tr in (head_trs or trs[:1]) for c in tr if isinstance(c.tag, str)]
        xyz_seq = any(_XYZ_CELLS.match(a or "") and _XYZ_CELLS.match(b or "") and _XYZ_CELLS.match(c or "")
                      for a, b, c in zip(head_cells, head_cells[1:], head_cells[2:]))
        has_graphic = orig is not None and any(isinstance(e.tag, str) and etree.QName(e).localname == "graphic" for e in orig.iter())
        rows.append({
            "pmcid": pmcid, "table_nb": nb, "table_id": (t.findtext("table-id") or "").strip(),
            "table_label": _WS.sub(" ", t.findtext("table-label") or "").strip(),
            "caption": caption, "caption_title": cap_title, "foot": foot,
            "n_rows": len(trs) - len(head_trs), "n_header_rows": len(head_trs), "n_cols": n_cols,
            "table_is_image": bool(has_graphic and not trs), "header_text": head_text[:500],
            "xyz_header": bool(xyz_seq or _COORD_HEAD.search(head_text)),
        })
    if not rows:
        rows.append({"pmcid": pmcid, "table_nb": -1})
    return rows


def _has_ancestor(e, name: str) -> bool:
    from lxml import etree

    p = e.getparent()
    while p is not None:
        if isinstance(p.tag, str) and etree.QName(p).localname == name:
            return True
        p = p.getparent()
    return False


def collect_tables(chunk_dirs: list[Path], n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse all tables of all articles in complete pubget chunks; return (tables, pubget coords per table, metadata)."""
    art_dirs: list[str] = []
    coords, metas = [], []
    for cd in chunk_dirs:
        art_dirs += [str(p) for p in sorted((cd / "articles").glob("*/pmcid_*"))]
        ed = cd / "subset_allArticles_extractedData"
        c = pd.read_csv(ed / "coordinates.csv", dtype={"table_id": str, "table_label": str})
        coords.append(c)
        metas.append(pd.read_csv(ed / "metadata.csv", usecols=["pmcid", "pmid", "doi", "title", "journal", "publication_year", "license"]))
    log(f"parsing tables.xml of {len(art_dirs)} articles ({n_jobs} processes)")
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        rows = [r for rs in ex.map(parse_article, art_dirs, chunksize=64) for r in rs]
    tables = pd.DataFrame(rows)
    coords = pd.concat(coords, ignore_index=True) if coords else pd.DataFrame(columns=["pmcid", "table_id", "table_label", "x", "y", "z"])
    meta = pd.concat(metas, ignore_index=True).drop_duplicates("pmcid") if metas else pd.DataFrame()
    return tables, coords, meta


# (e) caption heuristics
STAT_EXPR = re.compile(r"\b[pPkKtTzZF]\s*(?:\(\w+\))?\s*[<>=≤≥]\s*[-\d.,]+|\b(?:FWE|FDR|uncorrected|corrected)\b", re.I)
# explicit comparison / direction / parametric effect (the strict sub-flag)
COMPARISON_RE = re.compile(
    r"\bvs\.?(?=\s|$|,)|\bv\.(?=\s)|\bversus\b|\s[<>]\s|[A-Za-z)\]]\s?[<>]\s?[A-Za-z(\[]|\bminus\b|\bcompared (?:to|with)\b|\bcomparisons?\b|\brelative to\b"
    r"|\bthan\b|\bdifferen|\bdiffer\b|\bgreater\b|\bhigher\b|\blower\b|\bstronger\b|\bweaker\b|\bincrease|\bdecrease|\breduc|\benhance"
    r"|\binteractions?\b|\bmain effects?\b|\beffects? of\b|\bcorrelat|\bassociated with\b|\bassociations?\b|\brelated to\b|\b\w+-related\b"
    r"|\bpredict|\bmodulat|\balteration|\batroph|\babnormal|\bcontrasts? (?:of|between|for)\b|\bconjunction\b|\bparametric\b|\bdeactivat",
    re.I)
# names a task / condition / stimulus / population / analysis type (counts as "describes contrast/condition")
CONDITION_RE = re.compile(
    r"\bduring\b|\bin response to\b|\btasks?\b|\bconditions?\b|\bstimul|\btrials?\b|\bblocks?\b|\bpatients?\b|\bcontrols?\b|\bgroups?\b"
    r"|\bparticipants?\b|\bsubjects?\b|\bchildren\b|\badolescen|\badults?\b|\bhealthy\b|\bencoding\b|\bretrieval\b|\breward|\bmemory\b"
    r"|\battention|\bemotion|\bpain\b|\blanguage\b|\bmotor\b|\bvisual\b|\bauditory\b|\bfaces?\b|\bwords?\b|\bdecision|\bcognitive\b"
    r"|\bconnectivity\b|\bPPI\b|\bseed\b|\bgr[ae]y matter\b|\bwhite matter\b|\bVBM\b|\bthickness\b|\bresting[- ]state\b|\bmeta-analy"
    r"|\bhabituation\b|\blearning\b|\bprocessing\b|\bperception\b|\bviewing\b|\blistening\b|\breading\b|\bpresentation\b|\bfeedback\b"
    r"|\bonset\b|\bepisod|\bscan\b|\bsession|\btraining\b|\btreatment\b|\bdrug\b|\bplacebo\b|\bdisorder|\bdisease\b|\bsymptom|\bscores?\b",
    re.I)
REGION_RE = re.compile(
    r"\bregions?\b|\bareas?\b|\bstructures?\b|\banatomical\b|\banatomy\b|\bgyr(?:us|i)\b|\bsulc|\bcortex\b|\bcortices\b|\blobes?\b|\bROIs?\b"
    r"|\bnetworks?\b|\bnucle(?:us|i)\b|\bamygdala|\bhippocamp|\binsula|\bstriat|\bthalam|\bcerebell|\bprefrontal|\bparietal|\btemporal"
    r"|\boccipital|\bfrontal|\bcingulate|\bcaudate|\bputamen|\bbasal ganglia|\blocali[sz]ation", re.I)
STATS_RE = re.compile(
    r"\bcoordinates?\b|\bMNI\b|\bTalairach\b|\bthreshold|\bcorrected\b|\bFWE\b|\bFDR\b|\bz[- ]?(?:scores?|values?)\b|\bt[- ]?(?:scores?|values?)\b"
    r"|\bstatistic|\bvoxels?\b|\bsignificant\b|\bresults?\b|\banalys[ie]s\b|\bwhole[- ]brain\b|\bactivations?\b|\bactivated\b|\bclusters?\b"
    r"|\bpeaks?\b|\bfoci\b|\bmaxima\b|\blocations?\b|\bsummary\b|\bcharacteristics\b|\bdemographic", re.I)
QUALITY_ORDER = ["describes contrast/condition", "region list only", "statistics-only / boilerplate", "other", "missing"]
TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")


SW = set(ENGLISH_STOP_WORDS) | {"table", "tab", "et", "al"}


def n_words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-/]*", s or ""))


def strip_label(caption: str) -> str:
    return _LEAD_LABEL.sub("", caption or "").strip()


def explicit_comparison(caption: str) -> bool:
    """Strict sub-flag: the caption states a comparison / direction / parametric effect (vs, >, greater, correlation...)."""
    return bool(COMPARISON_RE.search(STAT_EXPR.sub(" ", strip_label(caption))))


# generic table vocabulary (not study-specific): excluded for the "specific token" overlap
TABLE_GENERIC = set(
    "brain regions region areas area clusters cluster peak peaks coordinates coordinate mni talairach significant significantly activation "
    "activations activated analysis analyses results result location locations local maxima maximum voxel voxels whole-brain wholebrain "
    "effects effect structures structure anatomical anatomic foci list showing shown reported space size values value statistics statistical "
    "threshold thresholds corrected uncorrected fwe fdr level levels contrast contrasts bold response responses data main summary note "
    "abbreviations hemisphere left right lateral medial cortex cortical gyrus montreal neurological institute stereotaxic standard "
    "listed given t-values t-value z-values z-value z-scores z-score p-value p-values comparisons comparison multiple statistically "
    "voxel-wise voxel-by-voxel voxelwise cluster-level clusterwise extent max min xyz mni152 mni305 tal labels label "
    "uncorr unc svc small volume correction family-wise familywise error rate false discovery".split())


def content_tokens(s: str) -> set[str]:
    return {t.strip("-") for t in TOKEN_RE.findall((s or "").lower()) if t.strip("-") not in SW and len(t.strip("-")) > 2}


def quality(caption: str) -> str:
    """Heuristic category of a table caption (label stripped)."""
    s = strip_label(caption)
    if not re.search(r"[A-Za-z]{2,}", s):
        return "missing"
    s_nostat = STAT_EXPR.sub(" ", s)
    if COMPARISON_RE.search(s_nostat) or CONDITION_RE.search(s_nostat):
        return "describes contrast/condition"
    if REGION_RE.search(s_nostat):
        return "region list only"
    content = content_tokens(STATS_RE.sub(" ", s_nostat))
    if STATS_RE.search(s_nostat) or len(content) <= 2:
        return "statistics-only / boilerplate"
    return "other"


# (f) caption vs abstract
def minilm_embed(texts: list[str], model, batch_size: int = 128, cache: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Normalised MiniLM embeddings; ``cache`` maps sha1(text) -> embedding (filled in place)."""
    import hashlib

    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    cache = {} if cache is None else cache
    todo = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items())
    if todo:
        emb = model().encode([t for _, t in todo], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
                             convert_to_numpy=True).astype(np.float32)
        cache.update({k: e for (k, _), e in zip(todo, emb)})
    return np.stack([cache[k] for k in keys]) if keys else np.zeros((0, 384), np.float32)


def load_emb_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    z = np.load(path)
    return dict(zip(z["keys"].tolist(), z["emb"]))


def save_emb_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    if cache:
        ks = sorted(cache)
        np.savez(path, keys=np.array(ks), emb=np.stack([cache[k] for k in ks]).astype(np.float32))


def caption_vs_abstract(cat: pd.DataFrame, corpus: pd.DataFrame, seed: int, n_threads: int,
                        cache_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Token overlap and MiniLM cosine between each captioned table and its publication's title+abstract."""
    import torch
    from _hf_pins import MINILM, sentence_transformer

    torch.set_num_threads(n_threads)
    rng = np.random.default_rng(seed)
    m = cat["caption_present"] & cat["pmid"].notna()
    sub = cat.loc[m, ["pmid", "caption_text"]].copy()
    papers = corpus.set_index("pmid").loc[sorted(set(sub["pmid"].astype("int64")))]
    pap_text = (papers["title"].fillna("") + ". " + papers["abstract"].fillna("")).tolist()
    pap_tok = [content_tokens(t) for t in pap_text]
    pidx = {p: i for i, p in enumerate(papers.index)}
    row_p = sub["pmid"].astype("int64").map(pidx).to_numpy()
    cap_tok = [content_tokens(c) for c in sub["caption_text"]]
    overlap = np.array([len(ct & pap_tok[j]) / len(ct) if ct else np.nan for ct, j in zip(cap_tok, row_p)])
    novel = np.array([len(ct - pap_tok[j]) for ct, j in zip(cap_tok, row_p)])
    spec = [ct - TABLE_GENERIC for ct in cap_tok]
    overlap_s = np.array([len(ct & pap_tok[j]) / len(ct) if ct else np.nan for ct, j in zip(spec, row_p)])
    novel_s = np.array([len(ct - pap_tok[j]) for ct, j in zip(spec, row_p)])
    out = pd.DataFrame({"overlap_frac_title_abstract": overlap, "n_novel_content_tokens": novel,
                        "overlap_frac_specific": overlap_s, "n_novel_specific_tokens": novel_s,
                        "novel_specific_tokens": [" ".join(sorted(ct - pap_tok[j])) for ct, j in zip(spec, row_p)]}, index=sub.index)
    info: dict[str, Any] = {"n_captions": int(len(sub)), "n_papers": int(len(papers))}
    if len(sub) == 0:
        return out, info
    t = time.perf_counter()
    _m: list[Any] = []

    def model():  # lazy: not loaded at all when every text is cached
        if not _m:
            _m.append(sentence_transformer(MINILM, "cpu"))  # pinned revision
        return _m[0]

    cache = load_emb_cache(cache_path) if cache_path else {}
    n_cached = len(cache)
    e_pap = minilm_embed(pap_text, model, cache=cache)
    e_cap = minilm_embed(sub["caption_text"].tolist(), model, cache=cache)
    if cache_path and len(cache) > n_cached:
        save_emb_cache(cache_path, cache)
    info["embedding_cache"] = {"file": str(cache_path) if cache_path else None, "cached_before": n_cached, "texts_now": len(cache)}
    info["minilm_seconds"] = round(time.perf_counter() - t, 1)
    info["minilm_max_seq_length"] = 256  # all-MiniLM-L6-v2 default; longer abstracts are truncated
    out["cos_own_abstract"] = np.einsum("ij,ij->i", e_cap, e_pap[row_p])
    # random other publication (seeded): chance level of caption<->abstract similarity
    rnd = rng.integers(0, len(papers), size=len(sub))
    clash = rnd == row_p
    rnd[clash] = (rnd[clash] + 1) % len(papers)
    out["cos_random_abstract"] = np.einsum("ij,ij->i", e_cap, e_pap[rnd]) if len(papers) > 1 else np.nan
    # caption -> paper retrieval among all audited papers (rank 1 = own abstract is the nearest)
    ranks = np.empty(len(sub), dtype=np.int64)
    for i in range(0, len(sub), 2048):
        s = e_cap[i:i + 2048] @ e_pap.T
        own = s[np.arange(s.shape[0]), row_p[i:i + 2048]]
        ranks[i:i + 2048] = 1 + (s > own[:, None]).sum(1)
    out["rank_own_abstract"] = ranks
    # within-publication caption similarity (do different coordinate tables of one paper get different captions?)
    ccos = []
    coord_sub = cat.loc[m & cat["is_coord_table"], ["pmid"]]
    pos = {ix: k for k, ix in enumerate(sub.index)}
    for _, g in coord_sub.groupby("pmid"):
        if len(g) < 2:
            continue
        E = e_cap[[pos[i] for i in g.index]]
        S = E @ E.T
        ccos.append(float(S[np.triu_indices(len(g), 1)].mean()))
    info["within_paper_coord_caption_cos_mean_per_paper"] = ccos
    return out, info


# summaries
def qstats(v: pd.Series, prefix: str) -> dict[str, float]:
    v = pd.to_numeric(v, errors="coerce").dropna()
    if not len(v):
        return {}
    q = v.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {f"{prefix}_mean": round(float(v.mean()), 3), f"{prefix}_p10": round(float(q[0.1]), 3), f"{prefix}_p25": round(float(q[0.25]), 3),
            f"{prefix}_median": round(float(q[0.5]), 3), f"{prefix}_p75": round(float(q[0.75]), 3), f"{prefix}_p90": round(float(q[0.9]), 3)}


def summarise_scope(name: str, d: pd.DataFrame) -> dict[str, Any]:
    n = len(d)
    r: dict[str, Any] = {"scope": name, "n_tables": int(n), "n_publications": int(d["pmid"].nunique())}
    if not n:
        return r
    cp = d["caption_present"]
    r["pct_caption_present"] = round(100 * cp.mean(), 2)
    r["pct_caption_title_present"] = round(100 * (d["caption_title"].fillna("").str.strip() != "").mean(), 2)
    r["pct_foot_present"] = round(100 * (d["foot_words"] > 0).mean(), 2)
    r["pct_caption_or_foot_present"] = round(100 * (cp | (d["foot_words"] > 0)).mean(), 2)
    r["pct_caption_ge_5_words"] = round(100 * (d["caption_words"] >= 5).mean(), 2)
    r["pct_caption_ge_10_words"] = round(100 * (d["caption_words"] >= 10).mean(), 2)
    r["pct_table_is_image"] = round(100 * d["table_is_image"].fillna(False).astype(bool).mean(), 2)
    r.update(qstats(d.loc[cp, "caption_words"], "caption_words"))
    r.update(qstats(d.loc[cp, "caption_chars"], "caption_chars"))
    r.update(qstats(d.loc[d["foot_words"] > 0, "foot_words"], "foot_words"))
    for k in QUALITY_ORDER:
        r[f"pct_quality_{k}"] = round(100 * (d["quality_heuristic"] == k).mean(), 2)
    r["pct_explicit_comparison"] = round(100 * d["explicit_comparison"].mean(), 2)
    for col, pre in (("overlap_frac_title_abstract", "overlap_frac"), ("n_novel_content_tokens", "novel_tokens"),
                     ("overlap_frac_specific", "overlap_specific"), ("n_novel_specific_tokens", "novel_specific_tokens"),
                     ("cos_own_abstract", "cos_own"), ("cos_random_abstract", "cos_random")):
        if col in d:
            r.update(qstats(d[col], pre))
    if "rank_own_abstract" in d:
        rk = d["rank_own_abstract"].dropna()
        if len(rk):
            r["retrieval_top1_pct"] = round(100 * float((rk <= 1).mean()), 2)
            r["retrieval_top10_pct"] = round(100 * float((rk <= 10).mean()), 2)
            r["retrieval_rank_median"] = float(rk.median())
            r["retrieval_n_candidates"] = int(d.attrs.get("n_candidates", 0)) or None
    if "overlap_frac_title_abstract" in d:
        ov = d["overlap_frac_title_abstract"].dropna()
        if len(ov):
            r["pct_caption_mostly_redundant_overlap_ge_0.8"] = round(100 * float((ov >= 0.8).mean()), 2)
            r["pct_caption_mostly_new_overlap_lt_0.5"] = round(100 * float((ov < 0.5).mean()), 2)
        if "n_novel_specific_tokens" in d:
            nv = d["n_novel_specific_tokens"].dropna()
            r["pct_caption_ge3_novel_specific_tokens"] = round(100 * float((nv >= 3).mean()), 2) if len(nv) else None
    return r


def coverage_table(cov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("all",) + SPLITS:
        d = cov if split == "all" else cov[cov["split"] == split]
        n = len(d)
        if not n:
            continue
        dl = d["downloaded"]
        rows.append({
            "split": split, "n_pmids": n,
            "n_with_pmcid": int(d["pmcid"].notna().sum()), "pct_with_pmcid": round(100 * d["pmcid"].notna().mean(), 2),
            "n_pmc_oa": int(d["is_oa"].sum()), "pct_pmc_oa": round(100 * d["is_oa"].mean(), 2),
            "n_requested_from_pubget": int(d["requested"].sum()),
            "n_downloaded": int(dl.sum()), "pct_downloaded_of_requested_oa": round(100 * dl.sum() / max(1, int((d["requested"] & d["is_oa"]).sum())), 2),
            "n_with_any_table": int((d["n_tables"] > 0).sum()),
            "n_with_coord_table": int((d["n_coord_tables"] > 0).sum()),
            "pct_with_coord_table_of_downloaded": round(100 * (d["n_coord_tables"] > 0).sum() / max(1, int(dl.sum())), 2),
            "n_coord_tables": int(d["n_coord_tables"].sum()),
            "n_coord_tables_captioned": int(d["n_coord_tables_captioned"].sum()),
            "pct_coord_tables_captioned": round(100 * d["n_coord_tables_captioned"].sum() / max(1, int(d["n_coord_tables"].sum())), 2),
            "n_pmids_with_captioned_coord_table": int((d["n_coord_tables_captioned"] > 0).sum()),
            "pct_corpus_with_captioned_coord_table_observed": round(100 * (d["n_coord_tables_captioned"] > 0).mean(), 2),
            # extrapolation to the whole split when only part of the OA set has been downloaded
            "pct_corpus_with_captioned_coord_table_est": round(
                100 * d["is_oa"].mean() * ((d.loc[dl, "n_coord_tables_captioned"] > 0).mean() if dl.any() else np.nan), 2),
            "coord_tables_per_pub_median": float(d.loc[d["n_coord_tables"] > 0, "n_coord_tables"].median()) if (d["n_coord_tables"] > 0).any() else np.nan,
            "coord_tables_per_pub_mean": round(float(d.loc[d["n_coord_tables"] > 0, "n_coord_tables"].mean()), 3) if (d["n_coord_tables"] > 0).any() else np.nan,
            "pct_pubs_with_ge2_coord_tables": round(100 * float((d.loc[d["n_coord_tables"] > 0, "n_coord_tables"] >= 2).mean()), 2) if (d["n_coord_tables"] > 0).any() else np.nan,
        })
    return pd.DataFrame(rows)


def write_tex(summ: pd.DataFrame, cov: pd.DataFrame, path: Path) -> None:
    def f(x, nd=1):
        return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}")

    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Tables & $n$ & caption (\%) & words (median [IQR]) & contrast (\%) & overlap w/ abstract & cos own / random & top-10 (\%) \\",
             r"\midrule"]
    for _, r in summ.iterrows():
        lines.append(f"{r['scope']} & {f(int(r['n_tables']))} & {f(r.get('pct_caption_present'))} & "
                     f"{f(r.get('caption_words_median'), 0)} [{f(r.get('caption_words_p25'), 0)}--{f(r.get('caption_words_p75'), 0)}] & "
                     f"{f(r.get('pct_quality_describes contrast/condition'))} & {f(r.get('overlap_frac_median'), 2)} & "
                     f"{f(r.get('cos_own_median'), 2)} / {f(r.get('cos_random_median'), 2)} & {f(r.get('retrieval_top10_pct'))} \\\\")
    lines += [r"\midrule", r"Split & PMIDs & PMCID (\%) & PMC-OA (\%) & downloaded & w/ coord.\ table & coord.\ tables & captioned (\%) \\", r"\midrule"]
    for _, r in cov.iterrows():
        lines.append(f"{r['split']} & {f(int(r['n_pmids']))} & {f(r['pct_with_pmcid'])} & {f(r['pct_pmc_oa'])} & {f(int(r['n_downloaded']))} & "
                     f"{f(int(r['n_with_coord_table']))} & {f(int(r['n_coord_tables']))} & {f(r['pct_coord_tables_captioned'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None, help="Output dir (default outputs/, outputs/smoke with --smoke).")
    p.add_argument("--pubget-dir", type=Path, default=HERE / "outputs" / "pubget_data", help="pubget data dir and ID caches.")
    p.add_argument("--smoke", action="store_true", help="Seeded sample of 200 open-access papers.")
    p.add_argument("--max-articles", type=int, default=None, help="Seeded sample of N open-access papers.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=2500, help="PMCIDs per pubget run.")
    p.add_argument("--n-jobs", type=int, default=8, help="Processes for pubget and table parsing.")
    p.add_argument("--n-review", type=int, default=100, help="Size of the manual-review sample.")
    p.add_argument("--skip-download", action="store_true", help="Only analyse the complete pubget chunks on disk.")
    p.add_argument("--download-only", action="store_true", help="Map IDs and run pubget, no analysis.")
    p.add_argument("--keep-articlesets", action="store_true", help="Keep pubget's raw articleset XML files.")
    p.add_argument("--no-embed", action="store_true", help="Skip the MiniLM similarity.")
    p.add_argument("--no-embed-cache", action="store_true", help="Do not use minilm_embeddings_cache.npz.")
    args = p.parse_args(argv)
    out: Path = args.out or (HERE / "outputs" / ("smoke" if args.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    pg: Path = args.pubget_dir
    pg.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("NCBI_API_KEY")
    timings: dict[str, float] = {}
    T0 = t = time.perf_counter()

    # (a) corpus
    corpus, split_source = load_corpus()
    log(f"corpus: {len(corpus)} PMIDs; split source {split_source}; "
        + ", ".join(f"{k}={int((corpus['split'] == k).sum())}" for k in SPLITS))
    timings["load_corpus"] = round(time.perf_counter() - t, 1)

    # (b) mapping + OA status (whole corpus, always: coverage is a key number)
    t = time.perf_counter()
    mp = map_pmids(corpus["pmid"].tolist(), pg / "pmid_pmcid_map.csv")
    corpus = corpus.merge(mp[["pmid", "pmcid", "idconv_status"]], on="pmid", how="left")
    # PMCIDs present in the released parquet but missing from the converter (should be rare)
    rel = pd.to_numeric(corpus["pmcid_released"], errors="coerce").astype("Int64")
    corpus["pmcid"] = corpus["pmcid"].fillna(rel)
    oa = oa_status([int(x) for x in corpus["pmcid"].dropna()], pg / "pmc_oa_status.csv", api_key)
    corpus["is_oa"] = corpus["pmcid"].map(oa.set_index("pmcid")["is_oa"]).fillna(False).astype(bool)
    log(f"PMCID for {corpus['pmcid'].notna().sum()} / {len(corpus)} PMIDs; PMC-OA {corpus['is_oa'].sum()}")
    timings["id_mapping_and_oa"] = round(time.perf_counter() - t, 1)

    # (c) pubget
    t = time.perf_counter()
    if args.smoke or args.max_articles:
        n = args.max_articles or 200
        pool = corpus.loc[corpus["is_oa"], "pmcid"].astype("int64").to_numpy()
        sel = np.sort(np.random.default_rng(args.seed).choice(pool, size=min(n, len(pool)), replace=False))
        chunks, prefix = [sel.tolist()], f"neurovlm_sample{len(sel)}_seed{args.seed}"
    else:
        allp = sorted(corpus["pmcid"].dropna().astype("int64").unique().tolist())  # all PMCIDs; pubget keeps the OA ones
        chunks = [allp[i:i + args.chunk_size] for i in range(0, len(allp), args.chunk_size)]
        prefix = f"neurovlm_corpus_c{args.chunk_size}"
    requested = {int(x) for c in chunks for x in c}
    corpus["requested"] = corpus["pmcid"].isin(requested).fillna(False).astype(bool)
    pg_cmd_example = (f"{pubget_bin()} run {pg} --pmcids_file {pg}/{prefix}_NN_pmcids.txt --alias {prefix}_NN "
                      f"--n_jobs {args.n_jobs} --log_dir {pg}/logs")
    if not args.skip_download:
        status = run_pubget(chunks, prefix, pg, args.n_jobs, args.keep_articlesets)
    else:
        status = []
        for ci in range(len(chunks)):
            link = pg / f"{prefix}_{ci:02d}"
            st = chunk_status(link.resolve()) if link.exists() else {"dir": None}
            st.update(chunk=link.name, n_pmcids_requested=len(chunks[ci]))
            status.append(st)
    timings["pubget"] = round(time.perf_counter() - t, 1)
    complete = [Path(s["dir"]) for s in status if s.get("extract_data")]
    log(f"pubget chunks complete: {len(complete)}/{len(chunks)}")
    if args.download_only:
        (out / "timings_download.json").write_text(json.dumps(timings, indent=1))
        return 0 if len(complete) == len(chunks) else 1

    # (d) parse tables
    t = time.perf_counter()
    tables, pcoords, meta = collect_tables(complete, args.n_jobs)
    timings["parse_tables"] = round(time.perf_counter() - t, 1)
    downloaded = set(tables["pmcid"].astype("int64")) if len(tables) else set()
    pmcid2pmid = corpus.dropna(subset=["pmcid"]).drop_duplicates("pmcid").set_index("pmcid")["pmid"]
    tables = tables[tables["table_nb"] >= 0].copy() if len(tables) else tables
    tables["pmid"] = tables["pmcid"].map(pmcid2pmid).astype("Int64")
    tables["split"] = tables["pmid"].map(corpus.set_index("pmid")["split"])
    # pubget coordinate tables: match on (pmcid, table_id), else (pmcid, table_label)
    pcoords["table_id"] = pcoords["table_id"].fillna("").astype(str).str.strip()
    pcoords["table_label"] = pcoords["table_label"].fillna("").astype(str).str.strip()
    by_id = pcoords[pcoords["table_id"] != ""].groupby(["pmcid", "table_id"]).size()
    by_lab = pcoords[pcoords["table_id"] == ""].groupby(["pmcid", "table_label"]).size()
    k_id = list(zip(tables["pmcid"], tables["table_id"].fillna("")))
    k_lab = list(zip(tables["pmcid"], tables["table_label"].fillna("")))
    tables["n_coords_pubget"] = [int(by_id.get(a, 0)) if a[1] else int(by_lab.get(b, 0)) for a, b in zip(k_id, k_lab)]
    tables["is_coord_table"] = tables["n_coords_pubget"] > 0
    tables["is_coord_table_or_xyz_header"] = tables["is_coord_table"] | tables["xyz_header"].fillna(False).astype(bool)
    # caption fields
    for c in ("caption", "caption_title", "foot", "table_label"):
        tables[c] = tables[c].fillna("").astype(str)
    tables["caption_text"] = tables["caption"].map(strip_label)
    tables["caption_present"] = tables["caption_text"].str.contains(r"[A-Za-z]{2,}", regex=True)
    tables["caption_chars"] = tables["caption_text"].str.len()
    tables["caption_words"] = tables["caption_text"].map(n_words)
    tables["foot_words"] = tables["foot"].map(n_words)
    tables["quality_heuristic"] = tables["caption"].map(quality)
    tables["explicit_comparison"] = tables["caption"].map(explicit_comparison)
    log(f"{len(tables)} tables in {tables['pmcid'].nunique()} articles; coordinate tables (pubget) {int(tables['is_coord_table'].sum())}")

    # (f) caption vs abstract
    t = time.perf_counter()
    sim_info: dict[str, Any] = {}
    if len(tables) and not args.no_embed:
        sim, sim_info = caption_vs_abstract(tables, corpus, args.seed, n_threads=max(1, args.n_jobs),
                                            cache_path=None if args.no_embed_cache else out / "minilm_embeddings_cache.npz")
        for c in sim:
            tables[c] = sim[c]
    timings["caption_vs_abstract"] = round(time.perf_counter() - t, 1)

    # (e) coverage + summaries
    t = time.perf_counter()
    tables["coord_captioned"] = tables["is_coord_table"] & tables["caption_present"]
    g = tables.groupby("pmcid")
    per = pd.DataFrame({"n_tables": g.size(), "n_coord_tables": g["is_coord_table"].sum(),
                        "n_coord_tables_captioned": g["coord_captioned"].sum(),
                        "n_coords_pubget": g["n_coords_pubget"].sum()})
    cov = corpus[["pmid", "pmcid", "split", "is_oa", "requested", "n_coords_released", "idconv_status"]].copy()
    cov["downloaded"] = cov["pmcid"].isin(downloaded).fillna(False).astype(bool)
    for c in per.columns:
        cov[c] = cov["pmcid"].map(per[c]).fillna(0).astype(int)
    if len(meta):
        mm = meta.set_index("pmcid")["pmid"]
        cov["pmid_pubget_metadata"] = cov["pmcid"].map(mm).astype("Int64")
    cov.to_csv(out / "pmid_coverage.csv", index=False)
    cov_tab = coverage_table(cov)
    cov_tab.to_csv(out / "coverage_summary.csv", index=False)

    n_cand = int(sim_info.get("n_papers", 0))
    scopes = [("all tables", tables), ("coordinate tables (pubget)", tables[tables["is_coord_table"]]),
              ("coord. tables (pubget or x/y/z header)", tables[tables["is_coord_table_or_xyz_header"]]),
              ("non-coordinate tables", tables[~tables["is_coord_table_or_xyz_header"]])]
    for _, d in scopes:
        d.attrs["n_candidates"] = n_cand
    summ = pd.DataFrame([summarise_scope(nm, d) for nm, d in scopes])
    for split in ("train", "val", "test"):
        d = tables[tables["is_coord_table"] & (tables["split"] == split)]
        d.attrs["n_candidates"] = n_cand
        summ = pd.concat([summ, pd.DataFrame([summarise_scope(f"coordinate tables (pubget), {split}", d)])], ignore_index=True)
    summ.to_csv(out / "caption_summary.csv", index=False)
    write_tex(summ.iloc[:3], cov_tab, out / "caption_summary.tex")

    # seeded manual-review sample of coordinate-table captions
    ct = tables[tables["is_coord_table"]]
    n_rev = min(args.n_review, len(ct))
    samp = ct.sample(n=n_rev, random_state=args.seed) if n_rev else ct
    samp = samp.merge(corpus[["pmid", "title", "abstract"]], on="pmid", how="left")
    keep = ["pmid", "pmcid", "split", "table_id", "table_label", "caption_text", "caption_words", "foot", "n_rows", "n_cols",
            "n_coords_pubget", "quality_heuristic", "explicit_comparison", "overlap_frac_title_abstract", "overlap_frac_specific",
            "novel_specific_tokens", "cos_own_abstract", "title", "abstract"]
    samp = samp[[c for c in keep if c in samp]].copy()
    samp["abstract"] = samp["abstract"].fillna("").str.slice(0, 600)
    samp["manual_category (contrast/condition | statistics-only/boilerplate | region list only | other)"] = ""
    samp["manual_adds_info_beyond_abstract (y/n)"] = ""
    samp["manual_notes"] = ""
    samp.to_csv(out / "manual_review_sample.csv", index=False)

    # pubget vs released coordinates (sanity check that pubget finds the corpus' coordinate tables)
    dl = cov[cov["downloaded"]]
    agree: dict[str, Any] = {}
    if len(dl):
        both = dl[(dl["n_coords_pubget"] > 0) & (dl["n_coords_released"] > 0)]
        agree = {"downloaded_pubs": int(len(dl)), "pct_with_pubget_coords": round(100 * float((dl["n_coords_pubget"] > 0).mean()), 2),
                 "pct_released_coords_count_within_20pct_of_pubget": round(100 * float(
                     (np.abs(both["n_coords_pubget"] - both["n_coords_released"]) <= 0.2 * both["n_coords_released"]).mean()), 2) if len(both) else None,
                 "spearman_n_coords_pubget_vs_released": round(float(both[["n_coords_pubget", "n_coords_released"]].corr("spearman").iloc[0, 1]), 3) if len(both) > 2 else None}
        if "pmid_pubget_metadata" in dl:
            agree["pct_pmid_mapping_confirmed_by_pubget_metadata"] = round(100 * float((dl["pmid_pubget_metadata"] == dl["pmid"]).mean()), 2)

    # catalog
    cat_cols = ["pmid", "pmcid", "split", "table_nb", "table_id", "table_label", "caption_text", "caption_title", "caption_present",
                "caption_chars", "caption_words", "foot", "foot_words", "n_rows", "n_header_rows", "n_cols", "table_is_image", "xyz_header",
                "header_text", "n_coords_pubget", "is_coord_table", "is_coord_table_or_xyz_header", "quality_heuristic", "explicit_comparison",
                "overlap_frac_title_abstract", "n_novel_content_tokens", "overlap_frac_specific", "n_novel_specific_tokens",
                "novel_specific_tokens", "cos_own_abstract", "cos_random_abstract", "rank_own_abstract"]
    tables[[c for c in cat_cols if c in tables]].to_parquet(out / "tables_catalog.parquet", index=False)
    timings["summaries"] = round(time.perf_counter() - t, 1)
    timings["total"] = round(time.perf_counter() - T0, 1)

    allrow = cov_tab[cov_tab["split"] == "all"].iloc[0].to_dict()
    n_oa_requested = int((corpus["requested"] & corpus["is_oa"]).sum())
    within = sim_info.pop("within_paper_coord_caption_cos_mean_per_paper", [])
    summary = {
        "date": str(date.today()), "smoke": bool(args.smoke), "max_articles": args.max_articles, "seed": args.seed,
        "pubget_version": pubget_version(), "pubget_command_template": pg_cmd_example, "pubget_chunks_total": len(chunks),
        "pubget_chunks_complete": len(complete), "pubget_chunk_status": status,
        "ncbi_api_key_used": bool(api_key), "split_source": split_source,
        "corpus_pmids": int(len(corpus)), "pmids_with_pmcid": int(corpus["pmcid"].notna().sum()),
        "pmids_pmc_oa": int(corpus["is_oa"].sum()), "pmc_oa_requested": n_oa_requested,
        "articles_downloaded": int(len(downloaded)),
        "download_complete": bool(len(complete) == len(chunks)),
        "pct_of_requested_oa_downloaded": round(100 * len(downloaded) / max(1, n_oa_requested), 2),
        "coverage_all": allrow,
        "abstract_words_median": float(corpus["abstract"].fillna("").map(n_words).median()),
        "title_words_median": float(corpus["title"].fillna("").map(n_words).median()),
        "abstract_words_median_downloaded": float(corpus.loc[corpus["pmcid"].isin(downloaded).fillna(False).astype(bool), "abstract"].fillna("").map(n_words).median()) if downloaded else None,
        "tables": {nm: summarise_scope(nm, d) for nm, d in scopes},
        "quality_heuristic_coord_tables_counts": {k: int(v) for k, v in tables.loc[tables["is_coord_table"], "quality_heuristic"].value_counts().items()},
        "manual_review_sample_quality_counts": {k: int(v) for k, v in samp["quality_heuristic"].value_counts().items()} if len(samp) else {},
        "coord_tables_per_downloaded_pub_hist": {str(k): int(v) for k, v in cov.loc[cov["downloaded"], "n_coord_tables"].clip(upper=6)
                                                 .value_counts().sort_index().items()},
        "within_paper_coord_caption_cos": qstats(pd.Series(within), "cos") | {"n_papers_ge2_captioned_coord_tables": len(within)},
        "caption_vs_abstract_info": sim_info,
        "pubget_vs_released_coordinates": agree,
        "timings_s": timings,
    }
    (out / "caption_summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (out / "timings.json").write_text(json.dumps(timings, indent=1))
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(cov_tab.to_string(index=False))
        print(summ[["scope", "n_tables", "pct_caption_present", "caption_words_median", "pct_quality_describes contrast/condition",
                    "overlap_frac_median", "cos_own_median", "cos_random_median", "retrieval_top10_pct"]].to_string(index=False))
    log(f"done in {timings['total']} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

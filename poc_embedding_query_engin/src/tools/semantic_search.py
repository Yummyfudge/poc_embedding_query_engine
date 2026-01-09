#!/usr/bin/env python3
"""
READ-ONLY / DEBUG-ONLY TOOL — NOT PART OF NORMAL QUERY FLOW
===========================================================

This script performs *read-only semantic search* over an already-built
embedding index for a single document.

IMPORTANT SCOPE BOUNDARY
------------------------
This script does **NOT**:
- Parse PDFs
- Chunk documents
- Generate or update embeddings
- Write any files
- Modify the vector store
- Participate in ingestion or indexing

It assumes that *all ingestion steps have already completed*.

REQUIRED PRE-EXISTING ARTIFACTS
-------------------------------
Created by `scripts/embed_chunks.py`:

  out/embeddings/<document_id>/
    ├── embeddings.npy        # shape: (num_chunks, embedding_dim)
    └── chunks_meta.jsonl     # per-chunk metadata (page, index, path, etc.)

If these files do not exist, this script will fail.

INTENDED USE
------------
This file exists strictly as:
- A debugging / inspection tool for semantic retrieval quality
- A way to experiment with similarity, reranking, and filtering logic
- A reference implementation for retrieval math (cosine similarity, rerank)

It is *not* called by `run_query.py` and is *not* part of the production
RAG execution path.

NORMAL QUERY FLOW
-----------------
The canonical query engine is:

  scripts/run_query.py
      → scripts/answer_with_evidence.py

That engine handles:
- Planner LLM calls
- Query embedding via Infinity (through LiteLLM)
- Vector search + filtering
- Evidence prompt construction
- Answer generation

This script should be treated as a standalone, read-only analyzer.

------------------------------------------------------------

Original functionality summary (unchanged):
1) Embed the query via LiteLLM /v1/embeddings (OpenAI-compatible)
2) Load precomputed chunk embeddings matrix (.npy) + chunk metadata (jsonl)
3) Compute cosine similarity to get top-K candidates
4) Optional reranking using page-level refined_metadata stored in Postgres
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import psycopg
from psycopg.rows import dict_row


# ----------------------------
# Config helpers
# ----------------------------

def _schema() -> str:
    return os.getenv("PDF_POC_SCHEMA", "pdf_emergency")


def _project_root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()


def _connect() -> psycopg.Connection:
    """Connect to Postgres using an explicit DSN from the environment.

    Required: SSL full verification only (sslmode=verify-full).

    DSN precedence:
      1) PDF_POC_PG_DSN (preferred)
      2) DATABASE_URL
      3) PG_DSN

    Note: We intentionally do NOT print the DSN (it may contain secrets).
    """

    dsn_sources = ["PDF_POC_PG_DSN", "DATABASE_URL", "PG_DSN"]
    dsn = None
    used = None

    for key in dsn_sources:
        val = os.getenv(key)
        if val and val.strip():
            dsn = val.strip()
            used = key
            break

    if not dsn:
        raise RuntimeError(
            "No Postgres DSN found. Set one of: PDF_POC_PG_DSN, DATABASE_URL, PG_DSN"
        )

    # Enforce verify-full explicitly
    if "sslmode=" not in dsn:
        raise RuntimeError(
            f"{used} must include sslmode=verify-full (no implicit SSL allowed)"
        )

    print(f"pg_dsn_source={used}")

    return psycopg.connect(dsn, row_factory=dict_row)


# ----------------------------
# LiteLLM embeddings client
# ----------------------------

def call_embeddings(
    *,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    timeout_s: int,
    retries: int = 2,
    backoff_s: float = 0.6,
) -> List[float]:
    """Call LiteLLM OpenAI-compatible /v1/embeddings.

    Note: We send input as a list to match the verified working curl.
    """

    url = base_url.rstrip("/") + "/v1/embeddings"
    payload = {"model": model, "input": [text]}
    data = json.dumps(payload).encode("utf-8")

    last_err: Optional[Exception] = None

    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            obj = json.loads(raw)

            if isinstance(obj, dict) and "data" in obj:
                return obj["data"][0]["embedding"]

            raise RuntimeError(
                f"Unexpected embeddings response shape from {url}: keys={list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
            )

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff_s * (attempt + 1))
                continue
            break

    raise RuntimeError(f"Embeddings request failed for {url}. Last error: {last_err}")


# ----------------------------
# Data structures
# ----------------------------


@dataclass(frozen=True)
class ChunkMeta:
    row_index: int
    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    chunk_text_path: str
    token_estimate: int


@dataclass
class SearchHit:
    row_index: int
    sim: float
    score: float
    chunk_meta: ChunkMeta
    snippet: str
    rerank_notes: str


# ----------------------------
# Loading embeddings + metadata
# ----------------------------


def load_meta(meta_path: Path) -> List[ChunkMeta]:
    metas: List[ChunkMeta] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            metas.append(
                ChunkMeta(
                    row_index=int(obj["row_index"]),
                    chunk_id=str(obj["chunk_id"]),
                    document_id=str(obj["document_id"]),
                    page_number=int(obj["page_number"]),
                    chunk_index=int(obj["chunk_index"]),
                    chunk_text_path=str(obj["chunk_text_path"]),
                    token_estimate=int(obj.get("token_estimate", 0)),
                )
            )

    # Ensure metas indexed by row_index
    metas.sort(key=lambda m: m.row_index)
    return metas


def load_embeddings_matrix(path: Path) -> np.ndarray:
    mat = np.load(path)
    # Force float32 for speed/memory
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32)
    return mat


# ----------------------------
# Text snippets
# ----------------------------


def _read_text(path_str: str, base: Path) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base / p).resolve()
    try:
        return p.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
    except Exception:
        return ""


def _make_snippet(text: str, query_terms: List[str], *, window: int = 220) -> str:
    if not text:
        return ""

    # Prefer first occurrence of any non-trivial term
    tl = text.lower()
    best_idx: Optional[int] = None
    best_term: Optional[str] = None

    for t in query_terms:
        if len(t) < 3:
            continue
        idx = tl.find(t)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_term = t

    if best_idx is None:
        # fallback: first 240 chars
        snip = text[:240]
        snip = re.sub(r"\s+", " ", snip).strip()
        return snip + ("…" if len(text) > 240 else "")

    start = max(0, best_idx - window)
    end = min(len(text), best_idx + window)
    snip = text[start:end]
    snip = snip.replace("\n", " ")
    snip = re.sub(r"\s{2,}", " ", snip).strip()
    if start > 0:
        snip = "…" + snip
    if end < len(text):
        snip = snip + "…"
    return snip


def _normalize_for_dedupe(s: str) -> str:
    s = s.lower()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedupe_key(snippet: str) -> str:
    """Stable key to collapse near-duplicate results.

    We hash a normalized snippet prefix to avoid storing large strings.
    """
    norm = _normalize_for_dedupe(snippet)
    norm = norm[:600]  # cap
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()


# ----------------------------
# Reranking helpers (uses refined_metadata)
# ----------------------------


def fetch_page_metadata(conn: psycopg.Connection, schema: str, document_id: str, page_numbers: List[int]) -> Dict[int, dict]:
    if not page_numbers:
        return {}

    sql = f"""
        SELECT page_number, refined_metadata
        FROM {schema}.pdf_page
        WHERE document_id = %s AND page_number = ANY(%s)
    """

    out: Dict[int, dict] = {}
    with conn.cursor() as cur:
        cur.execute(sql, (document_id, page_numbers))
        for r in cur.fetchall():
            pn = int(r["page_number"])
            md = r["refined_metadata"] or {}
            out[pn] = md
    return out


def rerank_score(
    *,
    base_sim: float,
    mode: str,
    page_md: dict,
    boost_flags: List[str],
) -> Tuple[float, str]:
    """Return (final_score, notes).

    base_sim is cosine similarity.

    Modes:
      - global: no rerank
      - appeal: boost admin/decision/appeal pages; lightly penalize high clinical density
      - clinical: boost high clinical density (labs/meds) lightly
    """

    if mode == "global" or not page_md:
        return base_sim, ""

    notes: List[str] = []
    score = base_sim

    # Extract flags
    def _b(key: str) -> bool:
        v = page_md.get(key)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() == "true"
        return False

    clin = float(page_md.get("clinical_term_density", 0.0) or 0.0)

    if mode == "appeal":
        # Boost admin/appeal/decision-ish pages
        if _b("has_admin_audit_terms"):
            score += 0.12
            notes.append("+admin")
        if _b("has_decision_terms"):
            score += 0.08
            notes.append("+decision")
        if _b("has_appeal_terms"):
            score += 0.05
            notes.append("+appeal")

        # User-specified refined_metadata boolean boosts (avoid hard-coding topic keys in code).
        for k in boost_flags:
            if not k:
                continue
            if _b(k):
                score += 0.06
                notes.append(f"+{k}")

        # Downweight strongly clinical pages slightly
        if clin >= 0.02:
            score -= min(0.10, clin * 2.5)
            notes.append(f"-clinical({clin:.3f})")

    elif mode == "clinical":
        # Slight boost for clinical density
        if clin > 0.0:
            bonus = min(0.10, clin * 2.5)
            score += bonus
            notes.append(f"+clinical({clin:.3f})")

    return score, ",".join(notes)


# ----------------------------
# Cosine similarity
# ----------------------------


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)


# ----------------------------
# Main
# ----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Semantic search across chunk embeddings + optional rerank.")
    ap.add_argument("query", help="Query text")
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--schema", default=_schema())
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--candidates", type=int, default=200, help="How many top cosine candidates to consider before rerank")
    ap.add_argument("--mode", choices=["global", "appeal", "clinical"], default="global")
    ap.add_argument(
        "--boost-flag",
        action="append",
        default=[],
        help="Repeatable. Name of a boolean key in refined_metadata to boost when true (e.g., --boost-flag has_q1_2024_hint).",
    )
    ap.add_argument("--snippet-chars", type=int, default=220)
    ap.add_argument("--show-path", action="store_true")
    ap.add_argument("--max-per-page", type=int, default=0, help="Max results per page (0 = unlimited)")
    ap.add_argument("--dedupe", action="store_true", help="Collapse near-duplicate results (recommended)")
    ap.add_argument("--sleep-ms", type=int, default=0, help="Optional sleep after embedding query (debug)")
    args = ap.parse_args()

    # Embeddings config
    base_url = os.getenv("LITELLM_BASE_URL")
    api_key = os.getenv("LITELLM_API_KEY")
    model = os.getenv("EMBEDDINGS_MODEL", "local-embed")

    if not base_url or not api_key:
        print("❌ Missing LITELLM_BASE_URL or LITELLM_API_KEY in environment")
        return 2

    project_root = _project_root()

    emb_dir = project_root / "out" / "embeddings" / args.document_id
    emb_path = emb_dir / "embeddings.npy"
    meta_path = emb_dir / "chunks_meta.jsonl"

    if not emb_path.exists():
        print(f"❌ Missing embeddings file: {emb_path}")
        return 2
    if not meta_path.exists():
        print(f"❌ Missing metadata file: {meta_path}")
        return 2

    # Load
    metas = load_meta(meta_path)
    mat = load_embeddings_matrix(emb_path)

    if mat.shape[0] != len(metas):
        print("⚠️  embeddings row count != metas count")
        print(f"embeddings_rows={mat.shape[0]} metas={len(metas)}")

    # Normalize matrix once
    matn = normalize_rows(mat)

    # Embed query
    q_emb = call_embeddings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        text=args.query,
        timeout_s=60,
    )

    if int(args.sleep_ms) > 0:
        time.sleep(int(args.sleep_ms) / 1000.0)

    qv = np.asarray(q_emb, dtype=np.float32)
    qv = normalize_vec(qv)

    # Cosine similarity = dot with normalized vectors
    sims = matn @ qv

    # Take top-N candidates
    cand_n = min(int(args.candidates), sims.shape[0])
    top_idx = np.argpartition(-sims, cand_n - 1)[:cand_n]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    # Prepare query terms for snippets
    q_terms = re.findall(r"\w+", args.query.lower())

    # Optional rerank metadata fetch
    page_md_map: Dict[int, dict] = {}
    if args.mode != "global":
        unique_pages = sorted({metas[i].page_number for i in top_idx if i < len(metas)})
        conn = _connect()
        try:
            page_md_map = fetch_page_metadata(conn, args.schema, args.document_id, unique_pages)
        finally:
            conn.close()

    hits: List[SearchHit] = []

    for i in top_idx:
        if i >= len(metas):
            continue
        m = metas[i]
        sim = float(sims[i])

        md = page_md_map.get(m.page_number, {}) if page_md_map else {}
        score, notes = rerank_score(base_sim=sim, mode=args.mode, page_md=md, boost_flags=list(args.boost_flag))

        text = _read_text(m.chunk_text_path, project_root)
        snippet = _make_snippet(text, q_terms, window=int(args.snippet_chars))

        hits.append(
            SearchHit(
                row_index=m.row_index,
                sim=sim,
                score=float(score),
                chunk_meta=m,
                snippet=snippet,
                rerank_notes=notes,
            )
        )

    # Final sort by score desc, then similarity desc
    hits.sort(key=lambda h: (-h.score, -h.sim, h.chunk_meta.page_number, h.chunk_meta.chunk_index))

    # Optional: collapse near-duplicates and/or limit results per page.
    if args.dedupe or int(args.max_per_page) > 0:
        seen_keys = set()
        per_page_counts: Dict[int, int] = {}
        filtered: List[SearchHit] = []

        for h in hits:
            pn = h.chunk_meta.page_number

            if int(args.max_per_page) > 0:
                per_page_counts[pn] = per_page_counts.get(pn, 0)
                if per_page_counts[pn] >= int(args.max_per_page):
                    continue

            if args.dedupe:
                key = _dedupe_key(h.snippet)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

            filtered.append(h)
            if int(args.max_per_page) > 0:
                per_page_counts[pn] += 1

        hits = filtered

    print(f"document_id={args.document_id}")
    print(f"mode={args.mode}")
    print(f"top={args.top} candidates={cand_n}")

    for h in hits[: int(args.top)]:
        m = h.chunk_meta
        header = f"p.{m.page_number:04d} chunk={m.chunk_index:02d}  score={h.score:.4f}  sim={h.sim:.4f}"
        if h.rerank_notes:
            header += f"  [{h.rerank_notes}]"
        if args.show_path:
            header += f"  path={m.chunk_text_path}"
        print(header)
        print(f"  {h.snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
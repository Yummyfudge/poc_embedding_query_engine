#!/usr/bin/env python3

"""
embed_chunks.py
================

ROLE IN PROJECT
---------------
This script performs **offline embedding generation** for document chunks
that have already been parsed and stored in Postgres.

It is responsible for:
  • Fetching chunk metadata (chunk_id, page, text path, etc.) from Postgres
  • Loading chunk text from disk
  • Calling the embeddings endpoint (via LiteLLM → Infinity)
  • Writing:
      - embeddings.npy  (NumPy matrix of embeddings)
      - chunks_meta.jsonl (row-aligned metadata)

This script is **NOT used during live query answering**.

WHEN IT IS USED
---------------
• Initial document ingestion
• Rebuilding embeddings after:
    - Chunking changes
    - Embedding model changes
    - Dimension mismatches
• One-off maintenance or recovery tasks

This is a **batch / offline job** and may be slow by design.

WHEN IT IS NOT USED
-------------------
• Not part of run_query.py
• Not part of answer_with_evidence.py
• Not invoked during interactive or streaming queries

DEPENDENCIES & ASSUMPTIONS
--------------------------
• Chunks already exist in Postgres (pdf_page_chunk table)
• Chunk text files exist on disk
• EMBEDDINGS_MODEL points to a valid embedding model (e.g. local-embed)
• Embedding dimensionality must remain consistent with downstream search

CONFIGURATION
-------------
• Postgres connection via PDF_POC_PG_DSN (sslmode=verify-full required)
• Embedding endpoint via:
    - LITELLM_BASE_URL
    - LITELLM_API_KEY
• Output written under:
    out/embeddings/<document_id>/

STATUS
------
This script is **actively used**, but intentionally isolated.
It should remain stable and boring.

All retrieval, ranking, and answering logic lives elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import psycopg
from psycopg import sql
from psycopg.rows import dict_row


def _schema() -> str:
    return _validate_schema_name(os.getenv("PDF_POC_SCHEMA", "pdf_emergency"))


def _validate_schema_name(schema: str) -> str:
    s = (schema or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
        raise ValueError(f"Invalid schema name: {schema!r}")
    return s


def _project_root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()


def _connect() -> psycopg.Connection:
    """Connect to Postgres using an explicit DSN from the environment.

    Required: SSL full verification only (sslmode=verify-full).

    DSN precedence:
      1) PDF_POC_PG_DSN (preferred)
      2) DATABASE_URL
      3) PG_DSN
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
            "Missing Postgres DSN. Set PDF_POC_PG_DSN (preferred) or DATABASE_URL/PG_DSN. "
            "DSN must include sslmode=verify-full."
        )

    if "sslmode=verify-full" not in dsn:
        raise RuntimeError(
            f"Postgres DSN from {used} must include sslmode=verify-full (full certificate verification)."
        )

    print(f"pg_dsn_source={used}")
    return psycopg.connect(dsn, row_factory=dict_row)


def _read_text(path_str: str, base: Path) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base / p).resolve()
    try:
        return p.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
    except Exception:
        return ""


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    # MVP token estimate: ~chars/4
    max_chars = max_tokens * 4
    return text if len(text) <= max_chars else text[:max_chars]


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    token_estimate: int
    chunk_text_path: str


def fetch_chunks(conn: psycopg.Connection, schema: str, document_id: str) -> Iterable[ChunkRow]:
    query = sql.SQL("""
        SELECT chunk_id, document_id, page_number, chunk_index, token_estimate, chunk_text_path
        FROM {schema}.pdf_page_chunk
        WHERE document_id = %s
        ORDER BY page_number ASC, chunk_index ASC
    """).format(schema=sql.Identifier(schema))
    with conn.cursor() as cur:
        cur.execute(query, (document_id,))
        for r in cur.fetchall():
            r = dict(r)
            yield ChunkRow(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                page_number=int(r["page_number"]),
                chunk_index=int(r["chunk_index"]),
                token_estimate=int(r["token_estimate"]),
                chunk_text_path=str(r["chunk_text_path"]),
            )


def call_embeddings(
    *,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    timeout_s: int,
    retries: int = 2,
    backoff_s: float = 0.6,
) -> list[float]:
    # LiteLLM embedding endpoint expects input as a list (matches your verified curl)
    payload = {"model": model, "input": [text]}
    data = json.dumps(payload).encode("utf-8")

    last_err: Optional[Exception] = None

    url = base_url.rstrip("/") + "/v1/embeddings"

    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            obj = json.loads(raw)

            # OpenAI-style response (LiteLLM)
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

    raise RuntimeError(
        f"Embeddings request failed for {base_url.rstrip('/') + '/v1/embeddings'}. Last error: {last_err}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Embed chunks via LiteLLM /v1/embeddings and write .npy + metadata."
    )
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--schema", default=_schema())
    ap.add_argument(
        "--max-tokens", type=int, default=480, help="Defensive truncation cap (backend ~512)"
    )
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument(
        "--sleep-ms",
        type=int,
        default=1000,
        help="Sleep between embedding calls (ms). Helps avoid 429s.",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="For testing: embed only first N chunks (0=all)"
    )
    ap.add_argument(
        "--use-existing-meta",
        action="store_true",
        help="Reuse existing out/embeddings/<document_id>/chunks_meta.jsonl instead of querying Postgres (rebuild embeddings.npy only).",
    )
    args = ap.parse_args()
    args.schema = _validate_schema_name(args.schema)

    base_url = os.getenv("LITELLM_BASE_URL")
    api_key = os.getenv("LITELLM_API_KEY")
    model = os.getenv("EMBEDDINGS_MODEL", "local-embed")

    if not base_url:
        print("❌ Missing env var: LITELLM_BASE_URL (e.g., http://192.168.1.172:4000)")
        return 2
    if not api_key:
        print("❌ Missing env var: LITELLM_API_KEY (your Bearer token, e.g., POMPY)")
        return 2

    project_root = _project_root()
    out_dir = (project_root / "out" / "embeddings" / args.document_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "chunks_meta.jsonl"
    emb_path = out_dir / "embeddings.npy"

    def _load_rows_from_existing_meta(p: Path) -> list[ChunkRow]:
        rows: list[ChunkRow] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(
                    ChunkRow(
                        chunk_id=str(obj.get("chunk_id")),
                        document_id=str(obj.get("document_id")),
                        page_number=int(obj.get("page_number")),
                        chunk_index=int(obj.get("chunk_index")),
                        token_estimate=int(obj.get("token_estimate")),
                        chunk_text_path=str(obj.get("chunk_text_path")),
                    )
                )
        return rows

    use_existing = bool(args.use_existing_meta) or (os.getenv("EMBED_USE_EXISTING_META", "0") == "1")

    if use_existing and meta_path.exists() and meta_path.stat().st_size > 0:
        print(f"[embed_chunks] Using existing metadata: {meta_path}")
        rows = _load_rows_from_existing_meta(meta_path)
    else:
        conn = _connect()
        try:
            rows = list(fetch_chunks(conn, args.schema, args.document_id))
        finally:
            conn.close()

    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    vectors: list[np.ndarray] = []

    mf = None
    if not (use_existing and meta_path.exists() and meta_path.stat().st_size > 0):
        mf = meta_path.open("w", encoding="utf-8")

    try:
        for i, row in enumerate(rows):
            text = _read_text(row.chunk_text_path, project_root)
            if not text.strip():
                continue

            text = _truncate_to_tokens(text, int(args.max_tokens))
            emb = call_embeddings(
                base_url=base_url,
                api_key=api_key,
                model=model,
                text=text,
                timeout_s=int(args.timeout),
            )

            # Pace requests to keep LiteLLM/Infinity happy (your curl used sleep 1)
            if int(args.sleep_ms) > 0:
                time.sleep(int(args.sleep_ms) / 1000.0)

            v = np.asarray(emb, dtype=np.float32)
            vectors.append(v)

            if mf is not None:
                mf.write(
                    json.dumps(
                        {
                            "row_index": len(vectors) - 1,
                            "chunk_id": row.chunk_id,
                            "document_id": row.document_id,
                            "page_number": row.page_number,
                            "chunk_index": row.chunk_index,
                            "chunk_text_path": row.chunk_text_path,
                            "token_estimate": row.token_estimate,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if (i + 1) % 50 == 0:
                print(f"... embedded {i + 1}/{len(rows)}")
    finally:
        if mf is not None:
            mf.close()

    mat = np.vstack(vectors)
    np.save(emb_path, mat)

    print(f"document_id={args.document_id}")
    print(f"chunks_embedded={mat.shape[0]}")
    print(f"embedding_dim={mat.shape[1]}")
    print(f"embeddings_npy={emb_path}")
    print(f"chunks_meta={meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

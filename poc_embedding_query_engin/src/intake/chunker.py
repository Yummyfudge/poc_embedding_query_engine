#!/usr/bin/env python3
"""Build paragraph-aware chunks from page text and register them in Postgres.

Chunking strategy:
- Prefer paragraph boundaries (blank lines)
- Target max_tokens (approx, chars/4)
- Enforce min_tokens to avoid tiny chunks
- Add overlap_tokens to reduce boundary misses

Outputs:
- Writes chunk text files to: out/chunks_txt/<document_id>/
- Inserts rows into pdf_emergency.pdf_page_chunk

Run:
  python scripts/chunk_pages.py --document-id <UUID>

ROLE IN PROJECT:
- This script is a **core, first-class pipeline component**.
- It is responsible for deterministic chunk creation and authoritative registration of chunk metadata in Postgres.
- It is **not a debug or legacy script**.
- It is intended to be run during document ingestion / preprocessing, not during query-time RAG execution.
- Postgres is treated as the authoritative metadata store; text files are secondary artifacts.

STATUS:
- This script is active and maintained.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import psycopg
from psycopg.rows import dict_row


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


def _schema() -> str:
    return _validate_schema_name(os.getenv("PDF_POC_SCHEMA", "pdf_emergency"))


def _validate_schema_name(schema: str) -> str:
    s = (schema or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
        raise ValueError(f"Invalid schema name: {schema!r}")
    return s


def _project_root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _read_text(path_str: str, base: Path) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base / p).resolve()
    try:
        return p.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
    except Exception:
        return ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _token_estimate(text: str) -> int:
    # Fast approximation; good enough for chunk sizing.
    return max(1, int(len(text) / 4))


def _split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """Return list of (char_start, char_end, paragraph_text) using blank-line boundaries."""
    if not text:
        return []

    # Normalize excessive newlines while preserving paragraph breaks
    norm = re.sub(r"\n{3,}", "\n\n", text)
    parts = norm.split("\n\n")

    paras: list[tuple[int, int, str]] = []
    cursor = 0
    for part in parts:
        idx = norm.find(part, cursor)
        if idx == -1:
            idx = cursor
        start = idx
        end = idx + len(part)
        cursor = end + 2  # skip delimiter

        ptxt = part.strip()
        if ptxt:
            paras.append((start, end, ptxt))
    return paras


@dataclass(frozen=True)
class Chunk:
    char_start: int
    char_end: int
    text: str
    token_est: int


def build_chunks_for_page(
    page_text: str,
    *,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    paras = _split_paragraphs(page_text)
    if not paras:
        return []

    chunks: list[Chunk] = []
    cur_text = ""
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    def flush() -> None:
        nonlocal cur_text, cur_start, cur_end
        if cur_start is None or cur_end is None:
            cur_text = ""
            cur_start = None
            cur_end = None
            return
        txt = cur_text.strip()
        if not txt:
            cur_text = ""
            cur_start = None
            cur_end = None
            return
        chunks.append(
            Chunk(
                char_start=int(cur_start),
                char_end=int(cur_end),
                text=txt,
                token_est=_token_estimate(txt),
            )
        )
        cur_text = ""
        cur_start = None
        cur_end = None

    for (p_start, p_end, p_txt) in paras:
        if cur_start is None:
            cur_start = p_start
        proposed = (cur_text + "\n\n" + p_txt) if cur_text else p_txt

        if _token_estimate(proposed) <= max_tokens:
            cur_text = proposed
            cur_end = p_end
            continue

        # Avoid tiny chunks: allow slight overflow if still below min_tokens
        if _token_estimate(cur_text) < min_tokens and cur_text:
            cur_text = proposed
            cur_end = p_end
            flush()
            continue

        flush()
        cur_start = p_start
        cur_text = p_txt
        cur_end = p_end

    flush()

    # Add overlap by prepending tail of previous chunk
    if overlap_tokens > 0 and len(chunks) > 1:
        overlap_chars = overlap_tokens * 4
        new_chunks: list[Chunk] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = new_chunks[-1]
            cur = chunks[i]
            tail = prev.text[-overlap_chars:] if len(prev.text) > overlap_chars else prev.text
            merged = (tail + "\n\n" + cur.text).strip()
            new_chunks.append(
                Chunk(
                    char_start=cur.char_start,
                    char_end=cur.char_end,
                    text=merged,
                    token_est=_token_estimate(merged),
                )
            )
        chunks = new_chunks

    return chunks


def fetch_page_paths(conn: psycopg.Connection, schema: str, document_id: str) -> Iterable[dict]:
    sql = f"""
        SELECT page_number, text_path
        FROM {schema}.pdf_page
        WHERE document_id = %s
        ORDER BY page_number ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (document_id,))
        yield from cur.fetchall()


def insert_chunk_row(
    conn: psycopg.Connection,
    schema: str,
    *,
    chunk_id: str,
    document_id: str,
    page_number: int,
    chunk_index: int,
    char_start: int,
    char_end: int,
    token_estimate: int,
    chunk_text_path: str,
    text_sha256: str,
) -> None:
    sql = f"""
        INSERT INTO {schema}.pdf_page_chunk (
            chunk_id, document_id, page_number, chunk_index,
            char_start, char_end, token_estimate,
            chunk_text_path, text_sha256
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (document_id, page_number, chunk_index) DO UPDATE SET
            chunk_id = EXCLUDED.chunk_id,
            char_start = EXCLUDED.char_start,
            char_end = EXCLUDED.char_end,
            token_estimate = EXCLUDED.token_estimate,
            chunk_text_path = EXCLUDED.chunk_text_path,
            text_sha256 = EXCLUDED.text_sha256
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                chunk_id,
                document_id,
                int(page_number),
                int(chunk_index),
                int(char_start),
                int(char_end),
                int(token_estimate),
                chunk_text_path,
                text_sha256,
            ),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build paragraph-aware chunks and register them in Postgres.")
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--schema", default=_schema())
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--min-tokens", type=int, default=250)
    ap.add_argument("--overlap-tokens", type=int, default=200)
    ap.add_argument("--commit-every", type=int, default=200)
    args = ap.parse_args()

    args.schema = _validate_schema_name(args.schema)

    base = _project_root()
    out_dir = (base / "out" / "chunks_txt" / args.document_id).resolve()

    conn = _connect()
    conn.autocommit = False
    try:
        total_chunks = 0
        pages_seen = 0

        for row in fetch_page_paths(conn, args.schema, args.document_id):
            pages_seen += 1
            page_number = int(row["page_number"])
            page_text = _read_text(row["text_path"], base)
            if not page_text.strip():
                continue

            chunks = build_chunks_for_page(
                page_text,
                max_tokens=int(args.max_tokens),
                min_tokens=int(args.min_tokens),
                overlap_tokens=int(args.overlap_tokens),
            )

            for idx, ch in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                chunk_file = out_dir / f"page_{page_number:04d}_chunk_{idx:02d}.txt"
                _write_text(chunk_file, ch.text)

                rel_path = str(chunk_file.relative_to(base))
                insert_chunk_row(
                    conn,
                    args.schema,
                    chunk_id=chunk_id,
                    document_id=args.document_id,
                    page_number=page_number,
                    chunk_index=idx,
                    char_start=ch.char_start,
                    char_end=ch.char_end,
                    token_estimate=ch.token_est,
                    chunk_text_path=rel_path,
                    text_sha256=_sha256_text(ch.text),
                )
                total_chunks += 1

            if pages_seen % int(args.commit_every) == 0:
                conn.commit()
                print(f"... committed chunks through page {page_number} (pages_seen={pages_seen}, chunks={total_chunks})")

        conn.commit()
        print(f"document_id={args.document_id}")
        print(f"pages_seen={pages_seen}")
        print(f"chunks_created={total_chunks}")
        print(f"chunks_dir={out_dir}")
        return 0

    except Exception as e:
        conn.rollback()
        print("❌ chunk_pages failed; transaction rolled back")
        print(f"{type(e).__name__}: {e}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
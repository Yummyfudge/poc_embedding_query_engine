#!/usr/bin/env python3
"""Prepare a clean re-chunking run for a document.

ROLE IN PROJECT
---------------
Active maintenance / hygiene utility for ingestion artifacts.

This script resets *derived* chunk state (DB rows + on-disk chunk files) for a single
`document_id` so that chunking and embedding can be re-run deterministically.

It is part of the ingestion toolchain and is NOT used during query-time or answer generation.

STATUS: ACTIVE (maintained)

This script does TWO intentional cleanup steps for a given document_id:

1) Deletes existing chunk rows from Postgres (pdf_emergency.pdf_page_chunk)
   - Prevents stale chunk_index rows when chunking parameters change

2) Removes existing chunk text files on disk:
   out/chunks_txt/<document_id>/

This makes re-running scripts/chunk_pages.py deterministic and safe.

USAGE
-----
  python scripts/re-chunk_prep.py --document-id <UUID>

NOTES
-----
- Safe to run multiple times.
- Requires DELETE privileges on pdf_emergency.pdf_page_chunk (joe_admin is sufficient).
- Does NOT touch:
    * pdf_document
    * pdf_page
    * page text files
    * embeddings (those should be regenerated separately)

Recommended workflow when changing chunk parameters:
----------------------------------------------------
  1) python scripts/re-chunk_prep.py --document-id <UUID>
  2) python scripts/chunk_pages.py --document-id <UUID> [new params]
  3) python scripts/embed_chunks.py --document-id <UUID>
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import psycopg


def _schema() -> str:
    return _validate_schema_name(os.getenv("PDF_POC_SCHEMA", "pdf_emergency"))


def _project_root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()


def _validate_schema_name(schema: str) -> str:
    s = (schema or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
        raise ValueError(f"Invalid schema name: {schema!r}")
    return s


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

    # Enforce full SSL verification.
    # We do not print the DSN (it may contain secrets); only the env var name used.
    if "sslmode=verify-full" not in dsn:
        raise RuntimeError(
            f"Postgres DSN from {used} must include sslmode=verify-full (full certificate verification)."
        )

    print(f"pg_dsn_source={used}")
    return psycopg.connect(dsn)


def delete_chunks_from_db(conn: psycopg.Connection, schema: str, document_id: str) -> int:
    sql = f"""
        DELETE FROM {schema}.pdf_page_chunk
        WHERE document_id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (document_id,))
        return cur.rowcount


def delete_chunk_files(project_root: Path, document_id: str) -> bool:
    chunk_dir = project_root / "out" / "chunks_txt" / document_id
    if chunk_dir.exists() and chunk_dir.is_dir():
        shutil.rmtree(chunk_dir)
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean DB + filesystem state before re-chunking.")
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--schema", default=_schema())
    ap.add_argument("--dry-run", action="store_true", help="Show what would be deleted, but do not delete")
    args = ap.parse_args()

    args.schema = _validate_schema_name(args.schema)

    project_root = _project_root()

    print(f"document_id={args.document_id}")
    print(f"schema={args.schema}")

    # --- Filesystem ---
    chunk_dir = project_root / "out" / "chunks_txt" / args.document_id
    if chunk_dir.exists():
        print(f"chunk_dir_exists={chunk_dir}")
    else:
        print("chunk_dir_exists=False")

    # --- Database ---
    conn = _connect()
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) FROM {args.schema}.pdf_page_chunk WHERE document_id = %s",
                (args.document_id,),
            )
            (count_before,) = cur.fetchone()

        print(f"db_chunks_before={count_before}")

        if args.dry_run:
            print("dry_run=True (no changes made)")
            conn.rollback()
            return 0

        deleted = delete_chunks_from_db(conn, args.schema, args.document_id)
        conn.commit()
        print(f"db_chunks_deleted={deleted}")

    except Exception as e:
        conn.rollback()
        print("❌ Failed to delete chunk rows from DB")
        print(f"{type(e).__name__}: {e}")
        return 1
    finally:
        conn.close()

    removed = delete_chunk_files(project_root, args.document_id)
    print(f"chunk_files_removed={removed}")

    print("✅ Re-chunk prep complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
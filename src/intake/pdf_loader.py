# =============================================================================
# PDF TO PARSE POC: PDF LOADER SCRIPT
# -----------------------------------------------------------------------------
# ROLE IN PROJECT:       Core intake / ingestion step
# WHAT IT DOES:          PDF → per-page text files + Postgres metadata rows
# WHEN IT RUNS:          Offline / batch, not query-time
# WHAT IT DOES NOT DO:   No OCR, no embeddings, no querying
# STATUS:                Active / maintained
# =============================================================================

from __future__ import annotations

import hashlib
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class LoadConfig:
    schema: str
    pdf_path: Path
    out_dir: Path
    pages_dir: Path
    min_chars_ok: int
    store_paths_relative_to: Path


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


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
            "Missing Postgres DSN. Set PDF_POC_PG_DSN (preferred) or DATABASE_URL/PG_DSN. "
            "DSN must include sslmode=verify-full."
        )

    if "sslmode=verify-full" not in dsn:
        raise RuntimeError(
            f"Postgres DSN from {used} must include sslmode=verify-full (full certificate verification)."
        )

    print(f"pg_dsn_source={used}")
    return psycopg.connect(dsn, row_factory=dict_row)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _clean_text(text: str) -> str:
    # Keep it simple; avoid aggressive “refactors” of the text.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def ensure_schema_exists(conn: psycopg.Connection, schema: str) -> None:
    # Safety: don’t auto-create tables here; you already created them in DBeaver.
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,))
        if cur.fetchone() is None:
            raise RuntimeError(f"Schema not found in DB: {schema}")


def doc_already_loaded(conn: psycopg.Connection, schema: str, source_path: str) -> Optional[dict]:
    sql = f"""
        SELECT document_id, source_filename, source_path, page_count, created_at_utc
        FROM {schema}.pdf_document
        WHERE source_path = %s
        ORDER BY created_at_utc DESC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (source_path,))
        return cur.fetchone()


def insert_document(conn: psycopg.Connection, schema: str, document_id: uuid.UUID, pdf_path: Path, page_count: int) -> None:
    sql = f"""
        INSERT INTO {schema}.pdf_document (document_id, source_filename, source_path, page_count)
        VALUES (%s, %s, %s, %s)
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (str(document_id), pdf_path.name, str(pdf_path.resolve()), int(page_count)),
        )


def upsert_page(
    conn: psycopg.Connection,
    schema: str,
    document_id: uuid.UUID,
    page_number: int,
    text_path: str,
    text_sha256: str,
    char_count: int,
    needs_ocr: bool,
    has_unreadable_content: bool,
    extraction_warning: Optional[str],
) -> None:
    sql = f"""
        INSERT INTO {schema}.pdf_page (
            document_id, page_number, text_path, text_sha256, char_count,
            has_unreadable_content, extraction_warning, needs_ocr, ocr_completed
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (document_id, page_number) DO UPDATE SET
            text_path = EXCLUDED.text_path,
            text_sha256 = EXCLUDED.text_sha256,
            char_count = EXCLUDED.char_count,
            has_unreadable_content = EXCLUDED.has_unreadable_content,
            extraction_warning = EXCLUDED.extraction_warning,
            needs_ocr = EXCLUDED.needs_ocr
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                str(document_id),
                int(page_number),
                text_path,
                text_sha256,
                int(char_count),
                bool(has_unreadable_content),
                extraction_warning,
                bool(needs_ocr),
                False,  # ocr_completed default
            ),
        )


def main() -> int:
    # Defaults from envrc/.env
    schema = _validate_schema_name(os.getenv("PDF_POC_SCHEMA", "pdf_emergency"))
    out_dir = Path(os.getenv("PDF_POC_OUT_DIR", "./out")).resolve()
    pages_dir = Path(os.getenv("PDF_POC_PAGES_DIR", str(out_dir / "pages_txt"))).resolve()

    # You can pass the PDF path as first arg, or set PDF_PATH env var.
    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else os.getenv("PDF_PATH")
    if not pdf_arg:
        print("Usage: python scripts/load_pdf.py /full/path/to/file.pdf")
        print("   or:  PDF_PATH=/full/path/to/file.pdf python scripts/load_pdf.py")
        return 2

    pdf_path = Path(pdf_arg).expanduser().resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return 2

    # Tuneable threshold: below this, mark page as likely needing OCR
    min_chars_ok = int(os.getenv("PDF_MIN_CHARS_OK", "80"))

    # We store text_path as relative to project root if possible (cleaner)
    project_root = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()

    cfg = LoadConfig(
        schema=schema,
        pdf_path=pdf_path,
        out_dir=out_dir,
        pages_dir=pages_dir,
        min_chars_ok=min_chars_ok,
        store_paths_relative_to=project_root,
    )

    cfg.pages_dir.mkdir(parents=True, exist_ok=True)

    # Connect and validate schema
    conn = _connect()
    conn.autocommit = False
    try:
        ensure_schema_exists(conn, cfg.schema)

        existing = doc_already_loaded(conn, cfg.schema, str(cfg.pdf_path))
        if existing:
            print("⚠️  This PDF path already exists in pdf_document (most recent):")
            print(existing)
            print("Proceeding will create a NEW document_id entry anyway (by design).")

        # Open PDF
        doc = fitz.open(str(cfg.pdf_path))
        page_count = len(doc)

        # Generate UUID in Python (as requested)
        document_id = uuid.uuid4()
        insert_document(conn, cfg.schema, document_id, cfg.pdf_path, page_count)

        print(f"✅ document_id={document_id} pages={page_count}")
        print(f"Writing text files to: {cfg.pages_dir}")

        # Extract per page
        for idx in range(page_count):
            page_number = idx + 1
            warn: Optional[str] = None

            try:
                raw_text = doc[idx].get_text("text") or ""
                text = _clean_text(raw_text)
            except Exception as e:
                text = ""
                warn = f"PyMuPDF text extraction error: {type(e).__name__}: {e}"

            char_count = len(text)
            needs_ocr = char_count < cfg.min_chars_ok
            has_unreadable_content = needs_ocr and char_count == 0

            # Write page text to file (even if empty, so you have a full set)
            txt_filename = f"page_{page_number:04d}.txt"
            txt_path = (cfg.pages_dir / txt_filename).resolve()
            txt_path.write_text(text, encoding="utf-8")

            text_sha = _sha256_text(text)

            upsert_page(
                conn=conn,
                schema=cfg.schema,
                document_id=document_id,
                page_number=page_number,
                text_path=_relpath(txt_path, cfg.store_paths_relative_to),
                text_sha256=text_sha,
                char_count=char_count,
                needs_ocr=needs_ocr,
                has_unreadable_content=has_unreadable_content,
                extraction_warning=warn,
            )

            if page_number % 50 == 0:
                conn.commit()
                print(f"... committed through page {page_number}")

        conn.commit()
        print("✅ Done. All pages loaded.")
        print(f"document_id={document_id}")
        return 0

    except Exception as e:
        conn.rollback()
        print("❌ Loader failed. Transaction rolled back.")
        print(f"{type(e).__name__}: {e}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

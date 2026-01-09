#!/usr/bin/env python3
"""refine_pages.py — OFFLINE / PRE-PROCESSING ENRICHMENT (optional)

ROLE IN PROJECT
---------------
Optional, offline metadata enrichment step for ingestion artifacts.

This script reads page rows from Postgres (`{schema}.pdf_page`), loads the corresponding
page text files (`text_path`), computes cheap-but-useful signals, and stores them back into
Postgres as JSONB in `pdf_page.refined_metadata`.

This is NOT part of the live query / RAG execution path. It is intended to be run during
pre-processing to improve downstream retrieval and filtering.

WHAT IT DOES
------------
- Computes lightweight signals (flags, year counts, first line, density heuristics)
- Writes the enrichment back to Postgres (`refined_metadata` JSONB)

WHAT IT DOES NOT DO
-------------------
- No OCR
- No embeddings
- No LLM calls
- No query-time ranking/answering

WHY THIS EXISTS
---------------
- Improves filtering/ranking for keyword search and later embeddings
- Helps quickly find "appeal / decision / determination" type pages
- Avoids heavy NLP (scispaCy/UMLS) until later

STATUS
------
OPTIONAL / AS-NEEDED (not part of the default query path)

NOTES
-----
- Safe to run multiple times; updates are idempotent at the row level.
- Designed to stay lightweight; heavy NLP enrichment (scispaCy/UMLS) may be added later.

Examples:
  python scripts/refine_pages.py --document-id dffea446-18a7-48ab-9322-0cae071cb201

  # Dry run (no DB updates)
  python scripts/refine_pages.py --document-id dffea446-18a7-48ab-9322-0cae071cb201 --dry-run

  # Only refine pages that were flagged needs_ocr=true
  python scripts/refine_pages.py --document-id dffea446-18a7-48ab-9322-0cae071cb201 --only-needs-ocr
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import psycopg
from psycopg.rows import dict_row



@dataclass(frozen=True)
class PageRow:
    page_number: int
    text_path: str
    char_count: int
    needs_ocr: bool


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


def _schema_default() -> str:
    return _validate_schema_name(os.getenv("PDF_POC_SCHEMA", "pdf_emergency"))


def _project_root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()


def _read_text(text_path: str, base: Path) -> str:
    p = Path(text_path)
    if not p.is_absolute():
        p = (base / p).resolve()
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        # Postgres JSON/JSONB rejects strings containing NUL (\u0000). Strip them early.
        return text.replace("\x00", "")
    except Exception:
        return ""


def _first_nonempty_line(text: str, *, max_len: int = 200) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            # Keep it bounded; avoids dumping a whole paragraph into metadata.
            return s[:max_len]
    return ""


def _count_years(text: str) -> dict[str, int]:
    # Capture years like 2023, 2024, 2025, etc.
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    out: dict[str, int] = {}
    for y in years:
        out[y] = out.get(y, 0) + 1
    return out


def _has_q1_2024_hint(text: str) -> bool:
    # Covers common representations:
    # - Q1 2024 / Q1-2024 / Q1/2024
    # - Jan/Feb/Mar 2024
    # - 1/xx/2024, 2/xx/2024, 3/xx/2024
    pattern = re.compile(
        r"(\bQ\s*1\s*[-/]?\s*2024\b)"
        r"|(\b(Jan|January|Feb|February|Mar|March)\b[^\n]{0,30}\b2024\b)"
        r"|(\b(1|2|3)\s*/\s*\d{1,2}\s*/\s*2024\b)",
        re.IGNORECASE,
    )
    return bool(pattern.search(text))


def _flag_terms(text: str, patterns: list[str]) -> bool:
    # Fast OR-based indicator.
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def _clinical_density(text: str) -> float:
    # Cheap heuristic: count clinical-ish token hits / word count.
    # This is NOT medical NLP; it just helps separate "med list / labs" pages from admin pages.
    if not text:
        return 0.0

    words = re.findall(r"\w+", text)
    if not words:
        return 0.0

    token_patterns = [
        r"\bMRN\b",
        r"\bICD\b",
        r"\bDiagnosis\b",
        r"\bAssessment\b",
        r"\bPlan\b",
        r"\bmg\b",
        r"\bMG\b",
        r"\btablet\b",
        r"\bcapsule\b",
        r"\bRx\b",
        r"\bRefill\b",
        r"\bOrdered\b",
        r"\bPerform:\b",
        r"\bLab\b",
        r"\bTherapy\b",
        r"\bMedication\b",
        r"\bDose\b",
        r"\bRoute\b",
        r"\bSig\b",
    ]

    hits = 0
    for pat in token_patterns:
        hits += len(re.findall(pat, text, re.IGNORECASE))

    return float(hits) / float(len(words))


def _ensure_refined_metadata_column(conn: psycopg.Connection, schema: str) -> None:
    # Safe, additive schema change.
    sql = f"""
        ALTER TABLE {schema}.pdf_page
        ADD COLUMN IF NOT EXISTS refined_metadata JSONB;
    """
    with conn.cursor() as cur:
        cur.execute(sql)


def _fetch_latest_document_id(conn: psycopg.Connection, schema: str) -> str:
    sql = f"""
        SELECT document_id
        FROM {schema}.pdf_document
        ORDER BY created_at_utc DESC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        if not row:
            raise RuntimeError(f"No documents found in {schema}.pdf_document")
        return str(row["document_id"])


def _fetch_pages(conn: psycopg.Connection, schema: str, document_id: str, only_needs_ocr: bool) -> Iterable[PageRow]:
    where_extra = "AND needs_ocr = true" if only_needs_ocr else ""
    sql = f"""
        SELECT page_number, text_path, char_count, needs_ocr
        FROM {schema}.pdf_page
        WHERE document_id = %s
        {where_extra}
        ORDER BY page_number ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (document_id,))
        for r in cur.fetchall():
            yield PageRow(
                page_number=int(r["page_number"]),
                text_path=str(r["text_path"]),
                char_count=int(r["char_count"]),
                needs_ocr=bool(r["needs_ocr"]),
            )


def _strip_nuls(value):
    """Recursively remove NUL characters from strings (Postgres JSONB disallows \u0000)."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {k: _strip_nuls(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_nuls(v) for v in value]
    return value


def _update_page_metadata(conn: psycopg.Connection, schema: str, document_id: str, page_number: int, metadata: dict) -> None:
    sql = f"""
        UPDATE {schema}.pdf_page
        SET refined_metadata = %s::jsonb
        WHERE document_id = %s AND page_number = %s
    """
    with conn.cursor() as cur:
        safe_metadata = _strip_nuls(metadata)
        cur.execute(sql, (json.dumps(safe_metadata, ensure_ascii=False), document_id, int(page_number)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Refine pages with lightweight metadata and store JSONB in Postgres.")
    ap.add_argument("--document-id", help="Document UUID (defaults to latest)")
    ap.add_argument("--schema", default=_schema_default(), help="DB schema (default: env PDF_POC_SCHEMA or pdf_emergency)")
    ap.add_argument("--dry-run", action="store_true", help="Compute stats only; do not update DB")
    ap.add_argument("--only-needs-ocr", action="store_true", help="Only refine pages where needs_ocr=true")
    ap.add_argument("--commit-every", type=int, default=200, help="Commit every N pages (default 200)")
    ap.add_argument(
        "--skip-ddl",
        action="store_true",
        help="Skip attempting to ALTER TABLE to add refined_metadata (use if running as non-owner).",
    )

    args = ap.parse_args()
    args.schema = _validate_schema_name(args.schema)

    project_root = _project_root()

    # Term groups (regex snippets)
    appeal_terms = [
        r"\bappeal\b",
        r"\breconsideration\b",
        r"\breview\b",
    ]
    decision_terms = [
        r"\bdecision\b",
        r"\bdetermination\b",
        r"\badverse\b",
        r"\bdenied\b",
        r"\bdenial\b",
        r"\bapproved\b",
        r"\bupheld\b",
        r"\boverturn\w*\b",
    ]
    claim_terms = [
        r"\bclaim\b",
        r"\bclaim\s*#\b",
        r"\bbenefit\b",
        r"\berisa\b",
        r"\bstd\b",
        r"\bltd\b",
    ]
    admin_audit_terms = [
        r"\baction\s+changed\b",
        r"\bdecision\s+made\s+date\b",
        r"\bstatus\s+changed\b",
        r"\boverride\b",
        r"\bcase\s+owner\b",
        r"\brecord\s+type\b",
    ]

    conn = _connect()
    conn.autocommit = False
    try:
        # Ensure destination column exists (best-effort). This requires table ownership.
        if not args.skip_ddl:
            try:
                _ensure_refined_metadata_column(conn, args.schema)
                conn.commit()
            except Exception as e:
                conn.rollback()
                print("⚠️  Could not apply DDL to add refined_metadata (likely due to privileges).")
                print(f"    {type(e).__name__}: {e}")
                print("    Fix by running this once as the table owner/superuser:")
                print(f"    ALTER TABLE {args.schema}.pdf_page ADD COLUMN IF NOT EXISTS refined_metadata JSONB;")
                print("    Then re-run this script (or re-run with --skip-ddl if the column already exists).")
                return 1

        document_id = args.document_id or _fetch_latest_document_id(conn, args.schema)

        total = 0
        flagged_admin = 0
        flagged_appeal = 0
        flagged_decision = 0
        flagged_claim = 0
        flagged_q1_2024 = 0

        for row in _fetch_pages(conn, args.schema, document_id, only_needs_ocr=bool(args.only_needs_ocr)):
            total += 1
            text = _read_text(row.text_path, project_root)

            first_line = _first_nonempty_line(text)
            years_present = _count_years(text)
            q1_2024 = _has_q1_2024_hint(text)

            has_appeal = _flag_terms(text, appeal_terms)
            has_decision = _flag_terms(text, decision_terms)
            has_claim = _flag_terms(text, claim_terms)
            has_admin = _flag_terms(text, admin_audit_terms)

            clin_density = _clinical_density(text)

            if has_admin:
                flagged_admin += 1
            if has_appeal:
                flagged_appeal += 1
            if has_decision:
                flagged_decision += 1
            if has_claim:
                flagged_claim += 1
            if q1_2024:
                flagged_q1_2024 += 1

            metadata = {
                "first_line": first_line,
                "years_present": years_present,
                "has_q1_2024_hint": bool(q1_2024),
                "has_appeal_terms": bool(has_appeal),
                "has_decision_terms": bool(has_decision),
                "has_claim_terms": bool(has_claim),
                "has_admin_audit_terms": bool(has_admin),
                "clinical_term_density": round(float(clin_density), 6),
            }

            if not args.dry_run:
                _update_page_metadata(conn, args.schema, document_id, row.page_number, metadata)

            if not args.dry_run and (total % int(args.commit_every) == 0):
                conn.commit()
                print(f"... committed refined_metadata through page batch count={total}")

        if not args.dry_run:
            conn.commit()

        print(f"document_id={document_id}")
        print(f"pages_refined={total}")
        print(f"flag_admin_audit_pages={flagged_admin}")
        print(f"flag_appeal_pages={flagged_appeal}")
        print(f"flag_decision_pages={flagged_decision}")
        print(f"flag_claim_pages={flagged_claim}")
        print(f"flag_q1_2024_hint_pages={flagged_q1_2024}")
        print(f"dry_run={bool(args.dry_run)}")
        return 0

    except Exception as e:
        conn.rollback()
        print("❌ refine_pages failed; transaction rolled back")
        print(f"{type(e).__name__}: {e}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

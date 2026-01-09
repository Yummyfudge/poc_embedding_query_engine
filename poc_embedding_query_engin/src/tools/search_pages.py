#!/usr/bin/env python3
"""Search extracted page text and return page numbers + snippets.

Default mode: uses Postgres (pdf_emergency) to get the ordered list of pages for a document_id,
then reads the extracted text files on disk (text_path) to score matches.

This keeps PHI/PII out of the database while still letting Postgres track metadata.

Examples:
  python scripts/search_pages.py "appeal claim Q1 2024" \
    --document-id dffea446-18a7-48ab-9322-0cae071cb201 --top 30

  python scripts/search_pages.py "WTW Leave Administration" --top 20

  # regex mode (Python regex)
  python scripts/search_pages.py "Q[1-4]\\s+2024" --regex --top 50

Notes:
- Connection uses libpq env vars: PGHOST, PGPORT, PGDATABASE, PGUSER, PGSSLMODE, PGSSLROOTCERT, PGSSLCERT, PGSSLKEY
- Schema defaults to env PDF_POC_SCHEMA, else pdf_emergency
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class PageHit:
    page_number: int
    score: float
    snippet: str
    text_path: str
    term_counts: dict[str, int]


def _connect() -> psycopg.Connection:
    """
    Connect to Postgres using a standardized DSN pattern.
    DSN is read from, in order:
      - PDF_POC_PG_DSN
      - DATABASE_URL
      - PG_DSN
    Raises clear error if none are set.
    Prints which DSN source is used (not the DSN itself).
    Enforces sslmode=verify-full.
    Uses row_factory=dict_row.
    """
    dsn_sources = [
        ("PDF_POC_PG_DSN", os.getenv("PDF_POC_PG_DSN")),
        ("DATABASE_URL", os.getenv("DATABASE_URL")),
        ("PG_DSN", os.getenv("PG_DSN")),
    ]
    for src, dsn in dsn_sources:
        if dsn:
            print(f"Using Postgres DSN from {src}")
            if "sslmode=verify-full" not in dsn:
                raise RuntimeError(f"DSN from {src} must include sslmode=verify-full")
            return psycopg.connect(dsn, row_factory=dict_row)
    raise RuntimeError(
        "No Postgres DSN found. Set PDF_POC_PG_DSN, DATABASE_URL, or PG_DSN environment variable."
    )


def _validate_schema_name(schema: str) -> None:
    """
    Only allow lowercase letters, numbers, and underscores.
    Raise ValueError if invalid.
    """
    if not re.fullmatch(r"[a-z0-9_]+", schema):
        raise ValueError(f"Invalid schema name: {schema!r}")


def _schema() -> str:
    schema = os.getenv("PDF_POC_SCHEMA", "pdf_emergency")
    _validate_schema_name(schema)
    return schema


def _project_root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()


def _read_text(path_str: str, base: Path) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base / p).resolve()
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _make_snippet(text: str, match_span: tuple[int, int], window: int = 140) -> str:
    if not text:
        return ""
    start, end = match_span
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    snippet = snippet.replace("\n", " ")
    snippet = re.sub(r"\s{2,}", " ", snippet).strip()
    if left > 0:
        snippet = "…" + snippet
    if right < len(text):
        snippet = snippet + "…"
    return snippet


def _score_terms(
    text_cmp: str,
    terms: list[str],
    *,
    require_all: bool,
    numeric_weight: float,
) -> tuple[float, Optional[tuple[int, int]], dict[str, int]]:
    """Return (score, first_match_span, per_term_counts).

    - `require_all=True` implements AND semantics across all terms.
    - Purely-numeric terms (e.g., years like 2024) are down-weighted via `numeric_weight`
      because they appear on many pages and can dominate ranking.
    """
    counts: dict[str, int] = {}
    for t in terms:
        if not t:
            continue
        counts[t] = text_cmp.count(t)

    if require_all and any(counts.get(t, 0) == 0 for t in terms if t):
        return 0.0, None, counts

    # Weighted score: cap each term's contribution to avoid a single repeated token dominating.
    score = 0.0
    for t, c in counts.items():
        if c <= 0:
            continue
        contrib = float(min(c, 5))
        if t.isdigit():
            contrib *= float(numeric_weight)
        score += contrib

    # Snippet: prefer earliest match among NON-numeric terms; fall back to numeric if needed.
    def _earliest_span(term_list: list[str]) -> Optional[tuple[int, int]]:
        earliest_idx: Optional[int] = None
        earliest_term: Optional[str] = None
        for tt in term_list:
            if not tt:
                continue
            idx = text_cmp.find(tt)
            if idx != -1 and (earliest_idx is None or idx < earliest_idx):
                earliest_idx = idx
                earliest_term = tt
        if earliest_idx is None or earliest_term is None:
            return None
        return (earliest_idx, earliest_idx + len(earliest_term))

    non_numeric_terms = [t for t in terms if t and not t.isdigit()]
    span = _earliest_span(non_numeric_terms) or _earliest_span([t for t in terms if t])

    return score, span, counts


def _score_regex(text: str, pattern: re.Pattern) -> tuple[int, Optional[tuple[int, int]]]:
    matches = list(pattern.finditer(text))
    if not matches:
        return 0, None
    first = matches[0]
    return len(matches), (first.start(), first.end())


def fetch_latest_document_id(conn: psycopg.Connection, schema: str) -> str:
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


def fetch_pages(conn: psycopg.Connection, schema: str, document_id: str) -> Iterable[dict]:
    sql = f"""
        SELECT page_number, text_path, char_count, needs_ocr
        FROM {schema}.pdf_page
        WHERE document_id = %s
        ORDER BY page_number ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (document_id,))
        yield from cur.fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(description="Search extracted page text and return page hits.")
    ap.add_argument("query", help="Search query (space-separated terms by default)")
    ap.add_argument("--document-id", help="UUID of the document to search (defaults to latest)")
    ap.add_argument("--schema", default=_schema(), help="DB schema (default: env PDF_POC_SCHEMA or pdf_emergency)")
    ap.add_argument("--top", type=int, default=30, help="How many results to show")
    ap.add_argument("--min-score", type=int, default=1, help="Minimum score to include")
    ap.add_argument(
        "--require-all-terms",
        action="store_true",
        help="If set, require every query term to appear at least once on the page (AND semantics).",
    )
    ap.add_argument(
        "--numeric-weight",
        type=float,
        default=0.15,
        help="Weight applied to purely-numeric terms (e.g., years like 2024). Default 0.15.",
    )
    ap.add_argument(
        "--debug-terms",
        action="store_true",
        help="If set, print per-term counts for each hit.",
    )
    ap.add_argument("--regex", action="store_true", help="Treat query as a Python regex")
    ap.add_argument("--ignore-case", action="store_true", default=True, help="Case-insensitive search (default on)")
    ap.add_argument("--show-path", action="store_true", help="Print the page text_path for each hit")
    ap.add_argument("--snippet-window", type=int, default=140, help="Characters around first match")

    args = ap.parse_args()

    project_root = _project_root()

    conn = _connect()
    try:
        document_id = args.document_id or fetch_latest_document_id(conn, args.schema)

        # Prepare search
        if args.regex:
            flags = re.IGNORECASE if args.ignore_case else 0
            pattern = re.compile(args.query, flags)
        else:
            # normalize terms
            terms = re.findall(r"\w+", args.query.lower() if args.ignore_case else args.query)

        hits: list[PageHit] = []

        for row in fetch_pages(conn, args.schema, document_id):
            page_number = int(row["page_number"])
            text_path = str(row["text_path"])

            text = _read_text(text_path, project_root)
            if not text:
                continue

            if args.regex:
                score, span = _score_regex(text, pattern)
                if score < args.min_score or span is None:
                    continue
                snippet = _make_snippet(text, span, window=args.snippet_window)
                hits.append(
                    PageHit(
                        page_number=page_number,
                        score=float(score),
                        snippet=snippet,
                        text_path=text_path,
                        term_counts={},
                    )
                )
            else:
                text_cmp = text.lower() if args.ignore_case else text
                score, span, counts = _score_terms(
                    text_cmp,
                    terms,
                    require_all=bool(args.require_all_terms),
                    numeric_weight=float(args.numeric_weight),
                )
                if score < float(args.min_score) or span is None:
                    continue
                snippet = _make_snippet(text, span, window=args.snippet_window)
                hits.append(
                    PageHit(
                        page_number=page_number,
                        score=float(score),
                        snippet=snippet,
                        text_path=text_path,
                        term_counts=counts if not args.regex else {},
                    )
                )

        # Sort: score desc, then page asc
        hits.sort(key=lambda h: (-h.score, h.page_number))

        print(f"document_id={document_id}")
        print(f"hits={len(hits)}")

        for h in hits[: args.top]:
            line1 = f"p.{h.page_number:04d}  score={h.score:.2f}"
            if args.show_path:
                line1 += f"  path={h.text_path}"
            print(line1)
            print(f"  {h.snippet}")
            if args.debug_terms and h.term_counts:
                print(f"  term_counts={h.term_counts}")

        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
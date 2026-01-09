#!/usr/bin/env python3
"""
Render the most recent parsing_engine trace into a readable report (PDF + optional MD).

Reads:
  out/traces/parsing_engine_*.json   (JSONL; one JSON object per line with {ts_utc, stream, text})

Finds:
  - document_id from filename
  - fused retrieval chunk_ids from lines like: "chunk_id=<uuid>"

Then loads:
  out/embeddings/<document_id>/chunks_meta.jsonl
and reads each chunk's text file via chunk_text_path/text_path.

Outputs:
  out/parsed_query_runs/<UTCSTAMP>_<document_id>/
    - report.pdf
    - report.md  (optional, kept for grepping/debugging)

Usage:
  python3 scripts/render_latest_query_run.py

Notes:
  - Expects you synced Linux out/ to Mac out/ (your logs_to_mac.sh).
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- Optional PDF backend ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
    from reportlab.lib.styles import getSampleStyleSheet
except Exception:
    SimpleDocTemplate = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "out"
TRACES_DIR = OUT_DIR / "traces"
PARSED_DIR = OUT_DIR / "parsed_query_runs"

TRACE_GLOB = "parsing_engine_*.json"  # JSONL but currently named .json


UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


@dataclass(frozen=True)
class ChunkMeta:
    row_index: int
    chunk_id: str
    page_number: Optional[int]
    chunk_index: Optional[int]
    text_path: Optional[str]
    raw: Dict[str, Any]


def _latest_trace_file() -> Path:
    if not TRACES_DIR.exists():
        raise SystemExit(f"Missing traces dir: {TRACES_DIR}")
    files = sorted(TRACES_DIR.glob(TRACE_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit(f"No trace files found in {TRACES_DIR} matching {TRACE_GLOB}")
    return files[0]


def _parse_document_id_from_filename(path: Path) -> str:
    # expected: parsing_engine_<UTC>_<document_id>.json
    m = UUID_RE.search(path.name)
    if not m:
        raise SystemExit(f"Could not extract document_id UUID from trace filename: {path.name}")
    return m.group(0)


def _read_jsonl_trace(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _reconstruct_console_text(trace_rows: List[Dict[str, Any]]) -> str:
    # Preserve exact console output order
    return "".join(str(r.get("text", "")) for r in trace_rows)


def _extract_fused_chunk_ids(console_text: str) -> List[str]:
    # Look for the fused section lines: "chunk_id=<uuid>"
    ids: List[str] = []
    for m in re.finditer(r"chunk_id=([0-9a-fA-F-]{36})", console_text):
        ids.append(m.group(1))
    # de-dupe preserve order
    seen = set()
    out = []
    for cid in ids:
        if cid not in seen:
            out.append(cid)
            seen.add(cid)
    return out


def _extract_chunk_hit_details(console_text: str) -> Dict[str, Dict[str, Any]]:
    """Extract best-effort fused hit details from the engine transcript.

    We look for lines like:
      "score=0.7542 page=... chunk_id=<uuid> probes=['Probe A', 'Probe B']"

    Returns mapping: chunk_id -> {"score": float|None, "probes": list[str]}
    """

    details: Dict[str, Dict[str, Any]] = {}

    # Regex is intentionally permissive; we parse probes with ast.literal_eval when possible.
    line_re = re.compile(
        r"score=(?P<score>[0-9.]+).*?chunk_id=(?P<cid>[0-9a-fA-F-]{36}).*?probes=(?P<probes>\[[^\]]*\])"
    )

    for m in line_re.finditer(console_text):
        cid = m.group("cid")
        score_s = m.group("score")
        probes_s = m.group("probes")

        try:
            score = float(score_s)
        except Exception:
            score = None

        probes: List[str] = []
        try:
            parsed = ast.literal_eval(probes_s)
            if isinstance(parsed, list):
                probes = [str(x) for x in parsed if str(x).strip()]
        except Exception:
            # fallback: keep raw probes string
            probes = [probes_s]

        # Keep the best score observed for this chunk_id
        prev = details.get(cid)
        if prev is None or (score is not None and (prev.get("score") is None or score > prev.get("score"))):
            details[cid] = {"score": score, "probes": probes}
        else:
            # merge probes
            if probes:
                existing = prev.get("probes") or []
                for p in probes:
                    if p not in existing:
                        existing.append(p)
                prev["probes"] = existing

    return details


def _load_chunks_meta(document_id: str) -> Dict[str, ChunkMeta]:
    meta_path = OUT_DIR / "embeddings" / document_id / "chunks_meta.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"Missing chunks_meta.jsonl: {meta_path}")

    by_id: Dict[str, ChunkMeta] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            chunk_id = str(obj.get("chunk_id") or obj.get("id") or "").strip()
            if not chunk_id:
                continue

            # Common field names seen in your outputs
            row_index = int(obj.get("row_index", obj.get("index", -1)))
            page_number = obj.get("page_number", obj.get("page"))
            chunk_index = obj.get("chunk_index", obj.get("chunk_no"))
            text_path = obj.get("chunk_text_path") or obj.get("text_path")

            cm = ChunkMeta(
                row_index=row_index,
                chunk_id=chunk_id,
                page_number=int(page_number) if page_number is not None else None,
                chunk_index=int(chunk_index) if chunk_index is not None else None,
                text_path=str(text_path) if text_path else None,
                raw=obj,
            )
            by_id[chunk_id] = cm

    return by_id


def _resolve_text_path(p: str) -> Path:
    # Many of your paths are stored relative; treat relative as relative to PROJECT_ROOT
    pp = Path(p)
    if not pp.is_absolute():
        pp = PROJECT_ROOT / p
    return pp


def _read_chunk_text(cm: ChunkMeta) -> str:
    if not cm.text_path:
        return "(No text_path/chunk_text_path in metadata for this chunk.)"
    p = _resolve_text_path(cm.text_path)
    if not p.exists():
        return f"(Missing chunk text file: {p})"
    return p.read_text(encoding="utf-8", errors="replace").strip()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_markdown(out_dir: Path, document_id: str, trace_file: Path, console_text: str, chunks: List[Tuple[ChunkMeta, str]], hit_details: Dict[str, Dict[str, Any]]) -> Path:
    md_path = out_dir / "report.md"
    lines: List[str] = []

    lines.append(f"# Parsed Query Run\n")
    lines.append(f"- Document ID: `{document_id}`\n")
    lines.append(f"- Trace file: `{trace_file}`\n")
    lines.append(f"- Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`\n")
    lines.append("\n---\n\n")
    lines.append("## Engine Transcript\n\n")
    lines.append("```text\n")
    lines.append(console_text)
    lines.append("\n```\n\n")
    lines.append("---\n\n")
    lines.append("## Fused Chunks (Full Text)\n\n")

    for i, (cm, text) in enumerate(chunks, 1):
        lines.append(f"### {i}. Page {cm.page_number} — chunk_id {cm.chunk_id}\n")
        lines.append(f"- row_index: {cm.row_index}\n")
        if cm.chunk_index is not None:
            lines.append(f"- chunk_index: {cm.chunk_index}\n")
        hd = hit_details.get(cm.chunk_id, {})
        score = hd.get("score")
        probes = hd.get("probes") or []
        if score is not None:
            lines.append(f"- score: {score:.4f}\n")
        if probes:
            lines.append("- probes:\n")
            for p in probes:
                lines.append(f"  - {p}\n")
        if cm.text_path:
            lines.append(f"- text_path: `{cm.text_path}`\n")
        lines.append("\n```text\n")
        lines.append(text)
        lines.append("\n```\n\n")

    md_path.write_text("".join(lines), encoding="utf-8")
    return md_path


def _write_pdf(out_dir: Path, document_id: str, trace_file: Path, console_text: str, chunks: List[Tuple[ChunkMeta, str]], hit_details: Dict[str, Dict[str, Any]]) -> Optional[Path]:
    if SimpleDocTemplate is None:
        return None

    pdf_path = out_dir / "report.pdf"

    styles = getSampleStyleSheet()
    story: List[Any] = []

    story.append(Paragraph("Parsed Query Run", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Document ID: <code>{document_id}</code>", styles["Normal"]))
    story.append(Paragraph(f"Trace file: <code>{str(trace_file)}</code>", styles["Normal"]))
    story.append(Paragraph(f"Generated (UTC): <code>{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}</code>", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Engine Transcript", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Preformatted(console_text, styles["Code"]))
    story.append(PageBreak())

    story.append(Paragraph("Fused Chunks (Full Text)", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))

    for idx, (cm, text) in enumerate(chunks, 1):
        story.append(Paragraph(f"{idx}. Page {cm.page_number} — chunk_id {cm.chunk_id}", styles["Heading3"]))
        meta_bits = [f"row_index={cm.row_index}"]
        if cm.chunk_index is not None:
            meta_bits.append(f"chunk_index={cm.chunk_index}")
        if cm.text_path:
            meta_bits.append(f"text_path={cm.text_path}")
        story.append(Paragraph(", ".join(meta_bits), styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))
        hd = hit_details.get(cm.chunk_id, {})
        score = hd.get("score")
        probes = hd.get("probes") or []
        if score is not None:
            story.append(Paragraph(f"score={score:.4f}", styles["Normal"]))
        if probes:
            story.append(Paragraph("probes:", styles["Normal"]))
            story.append(Preformatted("\n".join([f"- {p}" for p in probes]), styles["Code"]))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Preformatted(text, styles["Code"]))
        story.append(PageBreak())

    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
    doc.build(story)
    return pdf_path


def main() -> int:
    trace = _latest_trace_file()
    document_id = _parse_document_id_from_filename(trace)

    trace_rows = _read_jsonl_trace(trace)
    console_text = _reconstruct_console_text(trace_rows)
    hit_details = _extract_chunk_hit_details(console_text)

    chunk_ids = _extract_fused_chunk_ids(console_text)
    if not chunk_ids:
        print("No chunk_id=... entries found in trace transcript. Nothing to render.")
        print(f"Trace: {trace}")
        return 2

    by_id = _load_chunks_meta(document_id)

    chunks: List[Tuple[ChunkMeta, str]] = []
    missing: List[str] = []
    for cid in chunk_ids:
        cm = by_id.get(cid)
        if not cm:
            missing.append(cid)
            continue
        text = _read_chunk_text(cm)
        chunks.append((cm, text))

    stamp = _utc_stamp()
    out_dir = PARSED_DIR / f"{stamp}_{document_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = _write_markdown(out_dir, document_id, trace, console_text, chunks, hit_details)
    pdf_path = _write_pdf(out_dir, document_id, trace, console_text, chunks, hit_details)

    print("== render_latest_query_run ==")
    print("trace:", trace)
    print("output dir:", out_dir)
    print("md:", md_path)
    if pdf_path:
        print("pdf:", pdf_path)
    else:
        print("pdf: (reportlab not available; only markdown written)")

    if missing:
        print("\nWARNING: Some chunk_ids were referenced in trace but missing in chunks_meta.jsonl:")
        for cid in missing:
            print(" -", cid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
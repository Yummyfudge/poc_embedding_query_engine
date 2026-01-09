#!/usr/bin/env python3
"""run_intake_and_query.py

INTAKE-ONLY runner (no query execution).

This script standardizes the end-to-end intake pipeline into a single command:
  1) Load config/runtime.env into process environment
  2) Generate a fresh run_id (UUID) for every run
  3) Copy the input PDF to out/intake_runs/<run_id>/source.pdf
  4) Run intake stages:
       - PDF -> pages/text (pdf_loader)
       - pages -> chunks (chunker)
       - chunks -> embeddings (embedder)

Non-goals:
  - Does NOT run query_engine / retrieval / answer generation.

Usage:
  python scripts/run_intake_and_query.py --pdf /full/path/to/file.pdf

Outputs:
  out/intake_runs/<run_id>/source.pdf
  out/embeddings/<document_id>/embeddings.npy
  out/embeddings/<document_id>/chunks_meta.jsonl

"""

from __future__ import annotations

import argparse
import re
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_file_path(repo_root: Path) -> Path:
    return repo_root / "config" / "runtime.env"


def _parse_runtime_env(path: Path) -> Dict[str, str]:
    """Parse a bash-style KEY=VALUE env file.

    Supports:
      - blank lines
      - full-line comments (# ...)
      - inline comments after values
      - quoted values: KEY="..." or KEY='...'

    Note: This is intentionally minimal and conservative.
    """

    env: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"runtime env not found: {path}")

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()

        # Strip inline comments unless inside quotes
        if val and val[0] in ("\"", "'"):
            q = val[0]
            if val.endswith(q) and len(val) >= 2:
                val = val[1:-1]
            else:
                # Unterminated quote; keep as-is
                val = val.lstrip(q)
        else:
            if "#" in val:
                val = val.split("#", 1)[0].strip()

        if not key:
            continue
        env[key] = val

    return env


def _load_runtime_env_into_process(repo_root: Path) -> Path:
    env_path = _env_file_path(repo_root)
    loaded = _parse_runtime_env(env_path)

    # Do not clobber existing process env unless runtime.env explicitly sets it.
    for k, v in loaded.items():
        os.environ[k] = v

    # Always set PROJECT_ROOT to repo root (helps relative artifact resolution)
    os.environ["PROJECT_ROOT"] = str(repo_root)

    return env_path


def _run_capture(cmd: list[str], cwd: Path, extra_env: Dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess and capture output for diagnosis."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print("\n[run]", " ".join(cmd))
    cp = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if cp.stdout:
        print(cp.stdout, end="" if cp.stdout.endswith("\n") else "\n")
    if cp.stderr:
        # keep stderr visible; many scripts print usage/errors here
        print(cp.stderr, end="" if cp.stderr.endswith("\n") else "\n")
    return cp


def _run_checked(cmd: list[str], cwd: Path, extra_env: Dict[str, str] | None = None) -> None:
    cp = _run_capture(cmd, cwd=cwd, extra_env=extra_env)
    if cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)


def _run_pdf_loader(repo_root: Path, document_id: str, pdf_path: Path) -> str:
    """Run pdf_loader with a few compatible invocation styles.

    The pdf_loader CLI has varied across iterations (flags vs positional/env).
    We always generate a document_id here; we never require the caller to pass one.
    """

    loader = str(repo_root / "src" / "intake" / "pdf_loader.py")

    attempts: list[tuple[list[str], Dict[str, str] | None]] = [
        # Preferred (argparse-style)
        ([sys.executable, loader, "--document-id", document_id, "--pdf-path", str(pdf_path)], None),

        # Env-driven (most compatible): positional pdf path + env provides DOCUMENT_ID
        ([sys.executable, loader, str(pdf_path)], {"DOCUMENT_ID": document_id, "PDF_PATH": str(pdf_path)}),

        # Positional (pdf then doc_id)
        ([sys.executable, loader, str(pdf_path), document_id], None),

        # Positional (pdf only) + env doc id (no PDF_PATH)
        ([sys.executable, loader, str(pdf_path)], {"DOCUMENT_ID": document_id}),
    ]

    last_err: Exception | None = None
    for cmd, extra_env in attempts:
        try:
            cp = _run_capture(cmd, cwd=repo_root, extra_env=extra_env)
            if cp.returncode != 0:
                raise subprocess.CalledProcessError(cp.returncode, cp.args)

            # Validate that pdf_loader used the document_id we generated.
            combined = (cp.stdout or "") + "\n" + (cp.stderr or "")
            m = re.search(r"\bdocument_id=([0-9a-fA-F\-]{36})\b", combined)
            if m:
                used_id = m.group(1)
                if used_id != document_id:
                    print(
                        f"[runner] NOTE: pdf_loader used document_id={used_id} (runner generated {document_id}). "
                        "Treating pdf_loader document_id as authoritative for downstream stages."
                    )
                return used_id

            print("[runner] WARNING: pdf_loader did not print document_id; assuming it honored the provided DOCUMENT_ID")
            return document_id
        except Exception as exc:
            last_err = exc
            print("[runner] pdf_loader attempt failed; trying next style...")

    raise RuntimeError("pdf_loader failed for all invocation styles")


def main() -> int:
    ap = argparse.ArgumentParser(description="Intake-only runner (PDF -> chunks -> embeddings).")
    ap.add_argument("--pdf", required=True, help="Path to the source PDF file")
    ap.add_argument("--sleep-ms", type=int, default=50, help="Sleep between embedding calls (ms)")
    ap.add_argument("--skip-embeddings", action="store_true", help="Run loader+chunker but skip embeddings")
    ap.add_argument("--skip-chunking", action="store_true", help="Run loader only (no chunking/embeddings)")

    args = ap.parse_args()

    repo_root = _repo_root()
    env_path = _load_runtime_env_into_process(repo_root)

    pdf_src = Path(args.pdf).expanduser().resolve()
    if not pdf_src.exists():
        print(f"❌ PDF not found: {pdf_src}")
        return 2

    run_id = str(uuid.uuid4())
    run_dir = repo_root / "out" / "intake_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pdf_dst = run_dir / "source.pdf"
    shutil.copy2(pdf_src, pdf_dst)

    print("== run_intake_and_query.py (INTAKE ONLY) ==")
    print("repo_root   :", repo_root)
    print("runtime.env :", env_path)
    print("run_id      :", run_id)
    print("document_id : (from pdf_loader)")
    print("pdf_src     :", pdf_src)
    print("pdf_dst     :", pdf_dst)

    # Stage 1: PDF -> pages/text
    document_id = _run_pdf_loader(repo_root=repo_root, document_id=run_id, pdf_path=pdf_dst)

    if args.skip_chunking:
        print("\n[runner] skip_chunking enabled; stopping after pdf_loader.")
        return 0

    # Stage 2: pages -> chunks
    _run_checked(
        [sys.executable, str(repo_root / "src" / "intake" / "chunker.py"), "--document-id", document_id],
        cwd=repo_root,
    )

    if args.skip_embeddings:
        print("\n[runner] skip_embeddings enabled; stopping after chunker.")
        return 0

    # Stage 3: chunks -> embeddings
    _run_checked(
        [
            sys.executable,
            str(repo_root / "src" / "intake" / "embedder.py"),
            "--document-id",
            document_id,
            "--sleep-ms",
            str(args.sleep_ms),
        ],
        cwd=repo_root,
    )

    # Final artifact sanity
    emb_dir = repo_root / "out" / "embeddings" / document_id
    emb_npy = emb_dir / "embeddings.npy"
    meta_jsonl = emb_dir / "chunks_meta.jsonl"

    print("\n== artifacts ==")
    print("embeddings dir:", emb_dir)
    print("embeddings.npy :", "OK" if emb_npy.exists() else "MISSING", emb_npy)
    print("chunks_meta    :", "OK" if meta_jsonl.exists() else "MISSING", meta_jsonl)

    if not emb_npy.exists() or not meta_jsonl.exists():
        print("\n❌ Intake completed but embeddings artifacts are missing.")
        return 3

    print("\n✅ Intake complete.")
    print("run_id     =", run_id)
    print("document_id=", document_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
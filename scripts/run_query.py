#!/usr/bin/env python3

"""
run_query.py — Canonical CLI entrypoint (thin wrapper around src/engine/query_engine.py)

Purpose and scope
-----------------
This file is intentionally *not* the primary entrypoint or the place where
RAG behavior, prompting, or retrieval logic is defined.

Its sole purpose is to provide a safe, predictable, low-friction CLI
for running ad‑hoc document queries while respecting runtime configuration.

Specifically, this wrapper is responsible for:

• Loading config/runtime.env and injecting it into the subprocess environment
• Validating that required embeddings artifacts exist for the given document_id
• Selecting streaming vs non-streaming execution based on config + CLI flags
• Applying guardrails for known model-specific constraints (e.g. DeepSeek reasoning)
• Delegating all real work to src/engine/query_engine.py

What this file does NOT do
--------------------------
This wrapper does *not*:
• Construct planner / refiner / answer prompts
• Modify or interpret model outputs
• Control retrieval, embeddings, scoring, or chunk selection
• Define prompt semantics or formatting
• Contain any business logic

If you are debugging planner output, retrieval behavior, or answer correctness,
you should be working in src/engine/query_engine.py (or src/engine/* + src/query/*) — not here.

Streaming behavior
------------------
Streaming is controlled by a combination of environment variables and CLI flags:

Environment:
• STREAM_PROGRESS_SECONDS = 0
    → Streaming disabled (single blocking response)
• STREAM_PROGRESS_SECONDS > 0
    → Streaming enabled automatically, printing partial output every N seconds

CLI overrides:
• --stream
    → Force streaming ON (overrides env)
• --no-stream
    → Force streaming OFF (overrides env)

Model safety guardrail:
If LLAMA_ARG_THINK=deepseek is set, streaming is automatically disabled,
because DeepSeek-style reasoning extraction is unreliable in streaming mode.

Embeddings artifact expectations
--------------------------------
Both this wrapper and src/engine/query_engine.py expect embeddings artifacts at:

  <PROJECT_ROOT>/out/embeddings/<DOCUMENT_ID>/
    ├── embeddings.npy
    └── chunks_meta.jsonl

PROJECT_ROOT resolution:
• Taken from config/runtime.env if set
• Otherwise defaults to the repository root (directory above /scripts)

If you see:
  "Missing embeddings artifacts for document_id"

It means one of the following:
• The document-id is incorrect
• Embeddings were never generated
• PROJECT_ROOT is pointing at the wrong repo

This file exists to make experimentation safer and more repeatable —
not to hide or abstract away the real system.
"""


from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description="Canonical query runner (calls src/engine/query_engine.py)")
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--env-file", default="config/runtime.env")

    # Streaming controls:
    # - Default behavior remains env-driven via STREAM_PROGRESS_SECONDS
    # - But you can force/disable streaming explicitly for troubleshooting
    stream_group = ap.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        action="store_true",
        help="Force streaming on (overrides STREAM_PROGRESS_SECONDS)",
    )
    stream_group.add_argument(
        "--no-stream",
        action="store_true",
        help="Force streaming off (overrides STREAM_PROGRESS_SECONDS)",
    )

    ap.add_argument("query", help="User question")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env_path = (repo_root / args.env_file).resolve()
    if not env_path.exists():
        print(f"❌ Missing env file: {env_path}")
        return 2

    env_overrides = load_env_file(env_path)

    # Build subprocess env (inherit + override)
    child_env = os.environ.copy()
    child_env.update(env_overrides)

    # STREAM_PROGRESS_SECONDS is the default knob:
    #   0  -> streaming disabled
    #   >0 -> streaming enabled (unless overridden by CLI)
    try:
        stream_seconds = int(child_env.get("STREAM_PROGRESS_SECONDS", "0") or "0")
    except ValueError:
        stream_seconds = 0

    if args.stream:
        enable_stream = True
    elif args.no_stream:
        enable_stream = False
    else:
        enable_stream = stream_seconds > 0

    # Guardrail: DeepSeek-style reasoning extraction is safest when NOT streaming.
    # If LLAMA_ARG_THINK=deepseek is set in runtime.env, force streaming off.
    if str(child_env.get("LLAMA_ARG_THINK", "")).strip().lower() == "deepseek" and enable_stream:
        print("[runner] NOTE: LLAMA_ARG_THINK=deepseek detected; forcing streaming OFF for safer reasoning extraction")
        enable_stream = False

    # Ensure PROJECT_ROOT defaults to repo root if not set
    child_env.setdefault("PROJECT_ROOT", str(repo_root))

    # Validate embeddings artifacts exist before running the engine
    emb_dir = Path(child_env["PROJECT_ROOT"]) / "out" / "embeddings" / args.document_id
    emb_path = emb_dir / "embeddings.npy"
    meta_path = emb_dir / "chunks_meta.jsonl"

    if not emb_path.exists() or not meta_path.exists() or meta_path.stat().st_size == 0:
        print("❌ Missing embeddings artifacts for document_id")
        print(f"  {emb_path}")
        print(f"  {meta_path}")
        return 2

    cmd = [
        sys.executable,
        str(repo_root / "src" / "engine" / "query_engine.py"),
    ]
    if enable_stream:
        cmd.append("--stream")
    cmd += [
        "--document-id",
        args.document_id,
        args.query,
    ]

    print(f"[runner] env_file={env_path}")
    print(f"[runner] document_id={args.document_id}")
    if enable_stream:
        if args.stream:
            print("[runner] streaming enabled (forced by --stream)")
        else:
            print(f"[runner] streaming enabled, interval={stream_seconds} seconds")
    else:
        if args.no_stream:
            print("[runner] streaming disabled (forced by --no-stream)")
        else:
            print("[runner] streaming disabled")
    print(f"[runner] engine cmd: {' '.join(cmd)}")
    return subprocess.call(cmd, env=child_env)


if __name__ == "__main__":
    raise SystemExit(main())
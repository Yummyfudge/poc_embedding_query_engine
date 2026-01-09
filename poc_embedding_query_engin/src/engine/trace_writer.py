

"""src.engine.trace_writer

PURPOSE
-------
Centralized trace writer for the query engine.

This module provides a lightweight "tee" that writes all stdout/stderr text to:
  1) the console (original stream)
  2) a single JSONL trace file (one record per write)

Design goals:
- Never fail a run due to tracing errors
- Record UTC timestamps
- Keep the surface area tiny so the engine can evolve around it

"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Optional, Tuple


class JsonTraceTee(io.TextIOBase):
    """A TextIO wrapper that tees writes to both console and a JSONL trace file."""

    def __init__(self, original: io.TextIOBase, trace_path: str, stream_name: str):
        self._orig = original
        self._path = trace_path
        self._stream = stream_name

    def write(self, s: str) -> int:
        # Always write to console first
        n = self._orig.write(s)
        try:
            self._orig.flush()
        except Exception:
            pass

        # Best-effort trace append
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            rec = {"ts_utc": ts, "stream": self._stream, "text": s}
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
        except Exception:
            # Never fail the run due to tracing
            pass

        return n

    def flush(self) -> None:
        try:
            self._orig.flush()
        except Exception:
            pass


def enable_process_trace(trace_path: str) -> Tuple[io.TextIOBase, io.TextIOBase]:
    """Enable stdout/stderr teeing for the current process.

    Returns the original (stdout, stderr) so callers can restore them.
    """

    import sys

    orig_out = sys.stdout
    orig_err = sys.stderr
    sys.stdout = JsonTraceTee(orig_out, trace_path, "stdout")
    sys.stderr = JsonTraceTee(orig_err, trace_path, "stderr")
    return orig_out, orig_err


def restore_process_streams(orig_out: io.TextIOBase, orig_err: io.TextIOBase) -> None:
    """Restore stdout/stderr previously returned by enable_process_trace."""

    import sys

    sys.stdout = orig_out
    sys.stderr = orig_err
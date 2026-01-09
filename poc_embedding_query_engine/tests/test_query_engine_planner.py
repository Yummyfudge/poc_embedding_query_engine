"""Minimal planner + wiring tests for the canonical src engine.

Purpose:
- Lock down planner parsing behavior (Option D tolerant parsing)
- Avoid external deps (no LiteLLM / Postgres)

NOTE:
- We intentionally do NOT lock the planner to any specific reasoning format.
- We only assert that expansions/keywords can be extracted.
"""

import json

import pytest

import sys
from pathlib import Path

# Ensure src/ is importable for tests without requiring installation.
_repo_root = Path(__file__).resolve().parents[1]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from engine.planner_parser import parse_planner_option_d  # type: ignore
import engine.query_engine as query_engine  # type: ignore


def _wrap_planner_content(content: str) -> dict:
    """Wrap planner text as an OpenAI-style chat completion response."""
    return {"choices": [{"message": {"content": content}}]}


def test_parse_planner_option_d_json():
    content = json.dumps(
        {
            "expansions": [
                "Claim decision date",
                "Adverse benefit determination date",
            ],
            "keyword_mode": "ANY",
            "require_keywords": ["determination", "adverse"],
            "exclude_keywords": [],
        }
    )

    result = parse_planner_option_d(_wrap_planner_content(content))

    assert result.expansions == [
        "Claim decision date",
        "Adverse benefit determination date",
    ]
    assert result.keyword_mode == "ANY"
    assert "determination" in result.require_keywords


def test_parse_planner_option_d_plaintext_requires_keywords_header():
    """Plaintext contract requires both Expansions and Keywords headers."""
    content = """
    Expansions
    1. Claim decision date
    2. Adverse benefit determination date

    Keywords
    Require
    - determination
    - adverse
    """

    result = parse_planner_option_d(_wrap_planner_content(content))

    assert result.expansions == [
        "Claim decision date",
        "Adverse benefit determination date",
    ]
    assert "determination" in result.require_keywords


def test_parse_planner_option_d_empty_raises():
    with pytest.raises(RuntimeError):
        parse_planner_option_d(_wrap_planner_content("nonsense with no usable content"))


def test_engine_wiring_smoke(monkeypatch):
    """Smoke test: canonical modules import and key entrypoints exist."""

    # Ensure the orchestration entrypoint exists.
    assert hasattr(query_engine, "main"), "query_engine is expected to expose main()"

    # Ensure planner parse function exists.
    assert callable(parse_planner_option_d), "planner_parser must expose parse_planner_option_d()"

    # Ensure retrieval hook exists (used by the engine pipeline).
    from query.content_query import run_content_query  # type: ignore
    assert callable(run_content_query), "content_query must expose run_content_query()"

    # Ensure the planner call hook exists (even if tests stub it later).
    from engine.inference_worker import call_planner_llm  # type: ignore
    assert callable(call_planner_llm), "inference_worker must expose call_planner_llm()"

    _ = monkeypatch

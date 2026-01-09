"""Unit tests for the Round-1 content query layer.

These tests are intentionally deterministic and offline:
- No LiteLLM / Infinity calls
- No Postgres

They validate:
- env parsing helper tolerates inline comments
- embeddings/meta artifact loader behavior (shape alignment + clear errors)
- cosine similarity top-k ranking (basic sanity)

Fixtures used:
- tests/fixtures/embeddings_sample.npy  (N x 384)
- tests/fixtures/chunks_meta_sample.jsonl (N lines)
- tests/fixtures/chunks/*.txt
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from query import content_query as cq  # type: ignore

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
EMB_SAMPLE = FIXTURES_DIR / "embeddings_sample.npy"
META_SAMPLE = FIXTURES_DIR / "chunks_meta_sample.jsonl"


def _stage_artifacts(tmp_path: Path, document_id: str) -> Path:
    out_dir = tmp_path / "out" / "embeddings" / document_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "embeddings.npy").write_bytes(EMB_SAMPLE.read_bytes())
    (out_dir / "chunks_meta.jsonl").write_text(
        META_SAMPLE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return out_dir


def test_env_int_allow_comment_basic(monkeypatch):
    monkeypatch.setenv("RAG_PER_QUERY_CANDIDATES", "80")
    assert cq._env_int("RAG_PER_QUERY_CANDIDATES", 10) == 80


def test_env_int_allow_comment_with_inline_comment(monkeypatch):
    monkeypatch.setenv("RAG_PER_QUERY_CANDIDATES", "80  # UNUSED")
    assert cq._env_int("RAG_PER_QUERY_CANDIDATES", 10) == 80


def test_env_int_allow_comment_empty_or_missing(monkeypatch):
    monkeypatch.delenv("RAG_PER_QUERY_CANDIDATES", raising=False)
    assert cq._env_int("RAG_PER_QUERY_CANDIDATES", 10) == 10
    monkeypatch.setenv("RAG_PER_QUERY_CANDIDATES", "   # comment only")
    assert cq._env_int("RAG_PER_QUERY_CANDIDATES", 10) == 10


def test_load_embeddings_artifacts_happy_path(tmp_path, monkeypatch):
    document_id = "11111111-1111-1111-1111-111111111111"
    _stage_artifacts(tmp_path, document_id)

    monkeypatch.chdir(tmp_path)

    mat, meta = cq.load_embeddings(document_id=document_id)

    assert isinstance(mat, np.ndarray)
    assert mat.ndim == 2
    assert mat.shape[1] == 384
    assert len(meta) == mat.shape[0]


def test_load_embeddings_artifacts_mismatch_raises(tmp_path, monkeypatch):
    document_id = "22222222-2222-2222-2222-222222222222"
    out_dir = _stage_artifacts(tmp_path, document_id)

    meta_path = out_dir / "chunks_meta.jsonl"
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    meta_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(Exception):
        _ = cq.load_embeddings(document_id=document_id)


def test_cosine_topk_prefers_identical_vector(tmp_path, monkeypatch):
    document_id = "33333333-3333-3333-3333-333333333333"
    _stage_artifacts(tmp_path, document_id)
    monkeypatch.chdir(tmp_path)

    mat, _meta = cq.load_embeddings(document_id=document_id)

    qv = mat[0].copy()

    idxs, sims = cq.cosine_topk(mat, qv, k=5)

    assert idxs.shape[0] == 5
    assert int(idxs[0]) == 0
    assert float(sims[0]) == pytest.approx(1.0, abs=1e-4)
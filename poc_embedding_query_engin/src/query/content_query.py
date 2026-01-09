"""
content_query.py

PURPOSE
-------
Deterministic content‑side retrieval engine for the project.

This module owns *all* embedding‑based and vector‑math retrieval logic.
It is intentionally LLM‑free.

It is called by higher‑level orchestration layers (e.g. answer_with_evidence.py)
and returns ranked, explainable evidence candidates.

STATUS
------
ACTIVE (greenfield)

This file was split out of answer_with_evidence.py to keep retrieval
as a first‑class bounded context with its own knobs and metrics.

No answering, no prompting, no refinement lives here.

"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np



# ---------------------------
# Embedding helpers
# ---------------------------

def _normalize_litellm_obj(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    try:
        return obj.model_dump()  # pydantic v2
    except Exception:
        try:
            return obj.dict()  # pydantic v1
        except Exception:
            try:
                return json.loads(obj.json())
            except Exception:
                return {"_raw_response": str(obj)}


def embed_texts(texts: List[str]) -> List[List[float]]:
    import litellm  # local import: keeps module importable for unit tests
    model = os.environ.get("EMBEDDINGS_MODEL", "").strip()
    prefix = os.environ.get("LITELLM_MODEL_PREFIX", "litellm_proxy/").rstrip("/")
    api_base = os.environ.get("LITELLM_BASE_URL", "").strip()
    api_key = os.environ.get("LITELLM_API_KEY", "").strip()

    if not model:
        raise RuntimeError("EMBEDDINGS_MODEL not set")
    if not api_base or not api_key:
        raise RuntimeError("LiteLLM base or key missing")

    resp = litellm.embedding(
        model=f"{prefix}/{model}",
        input=texts,
        api_base=api_base,
        api_key=api_key,
    )

    resp_obj = _normalize_litellm_obj(resp)
    data = resp_obj.get("data", [])
    if len(data) != len(texts):
        raise RuntimeError("Embedding count mismatch")

    return [row["embedding"] for row in data]


# ---------------------------
# Artifact loading
# ---------------------------

def load_embeddings(document_id: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    base = Path("out/embeddings") / document_id
    emb_path = base / "embeddings.npy"
    meta_path = base / "chunks_meta.jsonl"

    if not emb_path.exists():
        raise RuntimeError(f"Missing embeddings: {emb_path}")
    if not meta_path.exists():
        raise RuntimeError(f"Missing metadata: {meta_path}")

    mat = np.load(str(emb_path))
    meta: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))

    if len(mat) != len(meta):
        raise RuntimeError("Embedding/meta row mismatch")

    return mat, meta


# ---------------------------
# Vector math
# ---------------------------

def cosine_topk(mat: np.ndarray, qv: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    qv = np.asarray(qv, dtype=np.float32)
    mat = np.asarray(mat, dtype=np.float32)

    qv /= (np.linalg.norm(qv) + 1e-8)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)

    sims = mat @ qv
    if len(sims) < k:
        k = len(sims)

    idxs = np.argpartition(-sims, range(k))[:k]
    order = idxs[np.argsort(-sims[idxs])]
    return order, sims[order]


def elbow_cutoff(sorted_scores: np.ndarray) -> Tuple[int, float]:
    if len(sorted_scores) < 2:
        return len(sorted_scores), 0.0

    gaps = sorted_scores[:-1] - sorted_scores[1:]
    i = int(np.argmax(gaps))
    return i + 1, float(gaps[i])


# ---------------------------
# Env helper for robust integer parsing
# ---------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    # Allow inline comments in .env values: e.g. "80  # UNUSED"
    raw2 = raw.split("#", 1)[0].strip()
    if not raw2:
        return default
    return int(raw2)

# ---------------------------
# Retrieval round 1
# ---------------------------

def retrieval_round_one(
    *,
    document_id: str,
    probes: List[str],
) -> Dict[str, Any]:
    """
    Perform Round‑1 retrieval across all probes independently,
    then fuse via MAX score.

    Returns a dict containing:
      - per_probe_metrics
      - fused_results (sorted)
    """

    emb_mat, meta = load_embeddings(document_id)
    probe_vecs = embed_texts(probes)

    per_query_k = _env_int("RAG_PER_QUERY_CANDIDATES", 80)
    fused: Dict[int, Dict[str, Any]] = {}
    probe_metrics: List[Dict[str, Any]] = []
    per_probe_results: List[Dict[str, Any]] = []

    for name, qv in zip(probes, probe_vecs):
        idxs, sims = cosine_topk(emb_mat, qv, per_query_k)
        kstar, gap = elbow_cutoff(sims)

        kept_idxs = idxs[:kstar]
        kept_sims = sims[:kstar]

        probe_metrics.append({
            "probe": name,
            "best_score": float(sims[0]) if len(sims) else None,
            "elbow_k": kstar,
            "gap": gap,
            "kept": len(kept_idxs),
        })

        # Preserve full per-probe top-k for observability (kept + dropped).
        topk_rows: List[Dict[str, Any]] = []
        for rank, (row_i, row_score) in enumerate(zip(idxs.tolist(), sims.tolist()), 1):
            topk_rows.append(
                {
                    "rank": rank,
                    "row_index": int(row_i),
                    "score": float(row_score),
                    "kept": bool(rank <= kstar),
                    "chunk": meta[int(row_i)],
                }
            )

        per_probe_results.append(
            {
                "probe": name,
                "top_k": int(per_query_k),
                "elbow_k": int(kstar),
                "gap": float(gap),
                "rows": topk_rows,
            }
        )

        for i, score in zip(kept_idxs, kept_sims):
            i = int(i)
            if i not in fused or score > fused[i]["score"]:
                fused[i] = {
                    "score": float(score),
                    "chunk": meta[i],
                    "via": [name],
                }
            else:
                fused[i]["via"].append(name)

    fused_items = sorted(
        fused.values(),
        key=lambda x: -x["score"]
    )

    return {
        "per_probe_metrics": probe_metrics,
        "per_probe_results": per_probe_results,
        "fused_results": fused_items,
    }


# ---------------------------
# Retrieval dispatcher (adapter)
# ---------------------------

def run_content_query(
    *,
    document_id: str,
    probes: List[str],
    variant: str = "round1",
) -> Dict[str, Any]:
    """Dispatch to a retrieval variant.

    Variants are intentionally named ("round1", "round2", etc.) so higher layers can
    switch retrieval strategies cleanly via configuration.

    Round 2 is not implemented yet; we fail fast with a clear error.
    """

    v = (variant or "round1").strip().lower()
    if v in {"round1", "r1", "1"}:
        out = retrieval_round_one(document_id=document_id, probes=probes)
        out["variant"] = "round1"
        out["probe_count"] = len(probes)
        return out

    if v in {"round2", "r2", "2"}:
        raise NotImplementedError("content_query variant 'round2' is not implemented yet")

    raise ValueError(f"Unknown content_query variant: {variant!r}")
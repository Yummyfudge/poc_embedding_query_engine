

"""src.engine.inference_worker

PURPOSE
-------
Worker module for all LLM-facing calls and response normalization.

This is the "inference" bounded context:
- Calls LiteLLM proxy for planner + answer stages
- Normalizes LiteLLM response objects into plain dicts
- Applies minimal, deterministic validation of required environment

NON-GOALS
---------
- No retrieval math (see: src/query/content_query.py)
- No orchestration / pipeline control flow (see: src/engine/query_engine.py)
- No planner parsing logic move yet (kept in scripts/answer_with_evidence.py for now to avoid churn)

"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _normalize_litellm_obj(obj: Any) -> Dict[str, Any]:
    """Normalize LiteLLM / ModelResponse objects to plain JSON dicts."""
    if isinstance(obj, dict):
        return obj
    try:
        return obj.model_dump()  # type: ignore[attr-defined]
    except Exception:
        try:
            return obj.dict()  # type: ignore[attr-defined]
        except Exception:
            try:
                return json.loads(obj.json())  # type: ignore[attr-defined]
            except Exception:
                return {"_raw_response": str(obj)}


def _require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def configure_litellm_client_from_env() -> None:
    """Configure LiteLLM client globals for talking to the local LiteLLM proxy."""
    import litellm

    base_url = _require_env("LITELLM_BASE_URL")
    api_key = _require_env("LITELLM_API_KEY")

    litellm.api_base = base_url
    litellm.api_key = api_key


def _litellm_model_name(model_group: str) -> str:
    prefix = os.environ.get("LITELLM_MODEL_PREFIX", "litellm_proxy/").rstrip("/")
    return f"{prefix}/{model_group}".lstrip("/")


def call_planner_llm(*, document_id: str, user_question: str, planner_prompt_text: str) -> Dict[str, Any]:
    """Planner call (non-streaming). Returns normalized dict response."""
    import litellm

    base_url = _require_env("LITELLM_BASE_URL")
    api_key = _require_env("LITELLM_API_KEY")

    model_group = (os.environ.get("PLANNER_MODEL", "").strip() or os.environ.get("CHAT_MODEL", "").strip())
    if not model_group:
        raise RuntimeError("PLANNER_MODEL/CHAT_MODEL not set")

    temperature = float(os.environ.get("PLANNER_TEMPERATURE", "0.6").split("#", 1)[0].strip() or "0.6")
    timeout_s = int(os.environ.get("PLANNER_TIMEOUT", "90").split("#", 1)[0].strip() or "90")
    max_tokens = int(os.environ.get("PLANNER_MAX_TOKENS", "300").split("#", 1)[0].strip() or "300")

    user_message = planner_prompt_text.rstrip() + "\n\nORIGINAL_QUESTION\n\n" + user_question.strip()

    resp = litellm.completion(
        model=_litellm_model_name(model_group),
        messages=[{"role": "user", "content": user_message}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        timeout=timeout_s,
        api_base=base_url,
        api_key=api_key,
        metadata={"document_id": document_id, "stage": "planner"},
    )

    resp_obj = _normalize_litellm_obj(resp)
    return resp_obj


def call_answer_llm(*, document_id: str, prompt_text: str, stream: bool) -> Dict[str, Any]:
    """Answer call. Returns normalized dict response."""
    import litellm

    base_url = _require_env("LITELLM_BASE_URL")
    api_key = _require_env("LITELLM_API_KEY")

    model_group = (os.environ.get("ANSWER_MODEL", "").strip() or os.environ.get("CHAT_MODEL", "").strip())
    if not model_group:
        raise RuntimeError("ANSWER_MODEL/CHAT_MODEL not set")

    temperature = float(os.environ.get("ANSWER_TEMPERATURE", "0.2").split("#", 1)[0].strip() or "0.2")
    timeout_s = int(os.environ.get("ANSWER_TIMEOUT", "180").split("#", 1)[0].strip() or "180")
    max_tokens = int(os.environ.get("ANSWER_MAX_TOKENS", "600").split("#", 1)[0].strip() or "600")

    resp = litellm.completion(
        model=_litellm_model_name(model_group),
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=bool(stream),
        timeout=timeout_s,
        api_base=base_url,
        api_key=api_key,
        metadata={"document_id": document_id, "stage": "answer"},
    )

    return _normalize_litellm_obj(resp)
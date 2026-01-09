"""src.engine.planner_parser

PURPOSE
-------
Deterministic, model-agnostic parsing for the planner stage.

The planner stage produces a structured "Option D" payload describing:
  - query expansions (strings)
  - keyword gating inputs (mode + require/exclude lists)

This module:
  - normalizes and parses the planner response into a PlannerOptionD
  - tolerates different planner output styles (JSON-first, then plaintext contract)

NON-GOALS
---------
- No LLM calls (handled by src/engine/inference_worker.py)
- No retrieval math (handled by src/query/content_query.py)
- No orchestration (handled by src/engine/query_engine.py)

NOTES
-----
- Parsing is intentionally model-agnostic.
- If a model emits <think>...</think> (or similar), we treat it as best-effort removable noise
  for parsing only; we do not require or depend on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class PlannerOptionD:
    """Planner output schema (Option D)."""

    expansions: List[str]
    keyword_mode: str
    require_keywords: List[str]
    exclude_keywords: List[str]


def _get_finish_reason_and_usage(raw_llm_response: Any) -> Tuple[str, Dict[str, Any]]:
    """Best-effort extraction of finish_reason + usage from a LiteLLM/OpenAI-style response."""
    finish = ""
    usage: Dict[str, Any] = {}
    try:
        if isinstance(raw_llm_response, dict):
            finish = str(raw_llm_response.get("choices", [{}])[0].get("finish_reason", "") or "")
            usage = dict(raw_llm_response.get("usage", {}) or {})
    except Exception:
        finish = ""
        usage = {}
    return finish, usage


def _strip_think_block(text: str) -> str:
    """Remove a <think>...</think> block if present. Best-effort only."""
    s = text or ""
    start = s.find("<think>")
    end = s.find("</think>")
    if start != -1 and end != -1 and end > start:
        end2 = end + len("</think>")
        return (s[:start] + s[end2:]).strip()
    return s.strip()


def _strip_code_fences(text: str) -> str:
    """Remove surrounding Markdown code fences if present."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from a string.

    Tolerates code fences and leading/trailing text.
    """
    t = _strip_code_fences(_strip_think_block(text))
    decoder = json.JSONDecoder()

    # Try direct parse first
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: scan forward to the first '{' and raw_decode
    idx = t.find("{")
    if idx == -1:
        raise ValueError("Planner output did not contain a JSON object")

    obj, _end = decoder.raw_decode(t[idx:])
    if not isinstance(obj, dict):
        raise ValueError("Planner output JSON was not an object")
    return obj


def parse_planner_option_d(raw_llm_response: Any) -> PlannerOptionD:
    """Parse the planner response into Option D.

    Supported formats:
      1) JSON-first (preferred): contains keys expansions/keyword_mode/require_keywords/exclude_keywords
      2) Plaintext contract:
         Expansions
         1. ...
         2. ...
         Keywords
         Require
         - ...

    Raises RuntimeError with helpful diagnostics if parsing fails.
    """

    # Normalize response shape to pull content
    try:
        if isinstance(raw_llm_response, dict):
            content = raw_llm_response["choices"][0]["message"]["content"]
        else:
            # Best-effort normalization for ModelResponse-like objects
            try:
                d = raw_llm_response.model_dump()  # type: ignore[attr-defined]
            except Exception:
                try:
                    d = raw_llm_response.dict()  # type: ignore[attr-defined]
                except Exception:
                    d = None
            if not isinstance(d, dict):
                raise TypeError("planner response is not dict-like")
            content = d["choices"][0]["message"]["content"]
            raw_llm_response = d
    except Exception as exc:
        raise RuntimeError("Unexpected planner response shape (missing choices/message/content)") from exc

    finish_reason, usage = _get_finish_reason_and_usage(raw_llm_response)

    raw_text = str(content or "")
    parse_text = _strip_think_block(raw_text)

    def _fail(reason: str) -> None:
        # Print diagnostics to stdout for trace capture.
        print("\n[planner-parse] ERROR:", reason)
        if finish_reason:
            print("[planner-parse] finish_reason:", finish_reason)
        if usage:
            print("[planner-parse] usage:", json.dumps(usage, ensure_ascii=False))
        print("[planner-parse] raw_content (full):")
        print(raw_text)
        raise RuntimeError(reason)

    # 1) JSON-first parse
    try:
        obj = _extract_first_json_object(parse_text)
        expansions = obj.get("expansions")
        keyword_mode = str(obj.get("keyword_mode", "NONE")).upper().strip()
        require_keywords = obj.get("require_keywords") or []
        exclude_keywords = obj.get("exclude_keywords") or []

        if not isinstance(expansions, list) or not all(isinstance(x, str) for x in expansions):
            raise ValueError("Option D JSON: expansions must be list[str]")

        if keyword_mode not in {"NONE", "ANY", "ALL"}:
            keyword_mode = "NONE"

        if not isinstance(require_keywords, list):
            require_keywords = []
        if not isinstance(exclude_keywords, list):
            exclude_keywords = []

        expansions = [x.strip() for x in expansions if x and str(x).strip()]
        require_keywords = [str(x).strip() for x in require_keywords if x and str(x).strip()]
        exclude_keywords = [str(x).strip() for x in exclude_keywords if x and str(x).strip()]

        if not expansions:
            _fail("Planner JSON contained empty expansions list")

        return PlannerOptionD(
            expansions=expansions,
            keyword_mode=keyword_mode,
            require_keywords=require_keywords,
            exclude_keywords=exclude_keywords,
        )
    except Exception:
        pass

    # 2) Plaintext contract parse
    lines = [ln.strip() for ln in parse_text.splitlines() if ln.strip()]

    try:
        exp_i = next(i for i, ln in enumerate(lines) if ln.lower() == "expansions")
    except StopIteration:
        _fail("Planner output missing 'Expansions' header")

    try:
        kw_i = next(i for i, ln in enumerate(lines) if ln.lower() == "keywords")
    except StopIteration:
        _fail("Planner output missing required 'Keywords' header")

    if kw_i <= exp_i:
        _fail("Planner output 'Keywords' section appears before 'Expansions'")

    expansions: List[str] = []
    for ln in lines[exp_i + 1 : kw_i]:
        if len(ln) >= 3 and ln[0].isdigit() and ln[1] == ".":
            item = ln[2:].strip()
            if item:
                expansions.append(item)
        elif ln.startswith("-"):
            item = ln[1:].strip()
            if item:
                expansions.append(item)

    if not expansions:
        _fail("Planner expansions list is empty")

    require_keywords: List[str] = []
    mode = "ANY"

    for ln in lines[kw_i + 1 :]:
        low = ln.lower()
        if low in {"require", "required", "must", "must include"}:
            mode = "ANY"
            continue
        if low in {"exclude", "excluded", "must not", "must not include"}:
            # We currently do not support a rich exclude block in plaintext; skip header.
            continue

        if ln.startswith("-"):
            item = ln[1:].strip()
            if item:
                require_keywords.append(item)
        else:
            if "," in ln and not require_keywords:
                require_keywords.extend([p.strip() for p in ln.split(",") if p.strip()])

    if not require_keywords:
        _fail("Planner required keywords list is empty")

    return PlannerOptionD(
        expansions=expansions,
        keyword_mode=mode,
        require_keywords=require_keywords,
        exclude_keywords=[],
    )


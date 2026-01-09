

"""src.engine.query_engine

PURPOSE
-------
Canonical end-to-end orchestrator for this project.

This module is intentionally the *thin* control-plane for the query pipeline.
It wires together:
  1) Planner (LLM) — proposes query expansions + keyword gates (Option D)
  2) Retrieval (deterministic) — embedding + vector search + fusion
  3) Answer (LLM) — evidence-grounded answer with citations

It does NOT own:
  - retrieval math (see: src/query/content_query.py)
  - low-level embedding/vector operations

STATUS
------
ACTIVE (greenfield)

ENTRYPOINT
----------
This module is intended to be invoked via the thin wrapper:
  scripts/run_query.py

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Ensure src/ is importable when invoked from scripts/ wrappers.
# (scripts/run_query.py already sets PROJECT_ROOT for deterministic imports.)
# Ensure src/ is importable when invoked from scripts/ wrappers.
# (scripts/run_query.py already sets PROJECT_ROOT for deterministic imports.)
_project_root = os.environ.get("PROJECT_ROOT")
if _project_root:
    _src_dir = os.path.join(_project_root, "src")
else:
    _src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from query.content_query import run_content_query  # type: ignore

# Use centralized trace writer
from engine.trace_writer import enable_process_trace, restore_process_streams  # type: ignore




# -----------------------------------------------------------------------------
# Planner Option D schema + tolerant parsing
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannerOptionD:
    expansions: List[str]
    keyword_mode: str
    require_keywords: List[str]
    exclude_keywords: List[str]


def _split_think_block(text: str) -> Tuple[str, str]:
    s = text or ""
    start = s.find("<think>")
    end = s.find("</think>")
    if start != -1 and end != -1 and end > start:
        end2 = end + len("</think>")
        think = s[start:end2]
        remainder = (s[:start] + s[end2:]).strip()
        return think, remainder
    return "", s.strip()


def _strip_code_fences(text: str) -> str:
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
    _, remainder = _split_think_block(text)
    t = _strip_code_fences(remainder)

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    idx = t.find("{")
    if idx == -1:
        raise ValueError("Planner output did not contain a JSON object")

    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(t[idx:])
    if not isinstance(obj, dict):
        raise ValueError("Planner output JSON was not an object")
    return obj


def parse_planner_option_d(raw_llm_response: Any) -> PlannerOptionD:
    """Parse the planner response.

    Accepts either:
      - strict Option D JSON (preferred)
      - plaintext contract (Expansions + Keywords)

    IMPORTANT: We preserve the <think> block in raw_text for debugging.
    """

    # Accept dict-like LiteLLM responses.
    if isinstance(raw_llm_response, dict):
        obj = raw_llm_response
    else:
        # Best-effort normalization for model response objects.
        try:
            obj = raw_llm_response.model_dump()  # type: ignore[attr-defined]
        except Exception:
            try:
                obj = raw_llm_response.dict()  # type: ignore[attr-defined]
            except Exception:
                try:
                    obj = json.loads(raw_llm_response.json())  # type: ignore[attr-defined]
                except Exception:
                    raise RuntimeError("Planner response could not be normalized to JSON")

    try:
        content = obj["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError("Unexpected planner response shape (missing choices/message/content)") from exc

    raw_text = str(content or "")
    think_block, remainder = _split_think_block(raw_text)

    # Prefer JSON extraction.
    try:
        j = _extract_first_json_object(remainder)
        expansions = j.get("expansions")
        keyword_mode = str(j.get("keyword_mode", "NONE")).upper().strip()
        require_keywords = j.get("require_keywords") or []
        exclude_keywords = j.get("exclude_keywords") or []

        if not isinstance(expansions, list) or not all(isinstance(x, str) for x in expansions):
            raise ValueError("Option D JSON: expansions must be list[str]")

        if keyword_mode not in {"NONE", "ANY", "ALL"}:
            keyword_mode = "NONE"

        expansions = [x.strip() for x in expansions if x and str(x).strip()]
        require_keywords = [str(x).strip() for x in require_keywords if x and str(x).strip()]
        exclude_keywords = [str(x).strip() for x in exclude_keywords if x and str(x).strip()]

        if not expansions:
            raise RuntimeError("Planner JSON contained empty expansions list")

        return PlannerOptionD(
            expansions=expansions,
            keyword_mode=keyword_mode,
            require_keywords=require_keywords,
            exclude_keywords=exclude_keywords,
        )
    except Exception:
        pass

    # Plaintext contract parsing.
    lines = [ln.strip() for ln in remainder.splitlines() if ln.strip()]

    def _fail(reason: str) -> None:
        print("\n[planner-parse] ERROR:", reason)
        print("[planner-parse] raw_content (full):")
        print(raw_text)
        raise RuntimeError(reason)

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
    for ln in lines[kw_i + 1 :]:
        if ln.startswith("-"):
            item = ln[1:].strip()
            if item:
                require_keywords.append(item)
        else:
            # tolerate comma-separated
            if "," in ln and not require_keywords:
                require_keywords.extend([p.strip() for p in ln.split(",") if p.strip()])

    if not require_keywords:
        _fail("Planner required keywords list is empty")

    return PlannerOptionD(
        expansions=expansions,
        keyword_mode="ANY",
        require_keywords=require_keywords,
        exclude_keywords=[],
    )


# -----------------------------------------------------------------------------
# LiteLLM calls (planner + answer)
# -----------------------------------------------------------------------------


def _normalize_litellm_obj(obj: Any) -> Dict[str, Any]:
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


# --------------------------------------------------------------------------
# Human-friendly planner output printing (model-agnostic, preserves line breaks)
# --------------------------------------------------------------------------

def _print_planner_raw_response(planner_raw: Dict[str, Any]) -> None:
    """Print the full planner response in a human-friendly way.

    We want BOTH:
      - full JSON (for trace/debug)
      - rendered content + reasoning_content with real line breaks

    This is for observability only; parsing is handled separately.
    """

    print("\n===== PLANNER RAW (full JSON) =====")
    try:
        print(json.dumps(planner_raw, indent=2, ensure_ascii=False))
    except Exception:
        print(str(planner_raw))

    msg: Dict[str, Any] = {}
    try:
        msg = (planner_raw.get("choices", [{}])[0].get("message", {}) or {})
    except Exception:
        msg = {}

    content = msg.get("content")
    if not isinstance(content, str):
        content = ""

    # Prefer message.reasoning_content, but fall back to provider_specific_fields.reasoning_content if present
    reasoning = msg.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        psf = msg.get("provider_specific_fields") or {}
        if isinstance(psf, dict):
            r2 = psf.get("reasoning_content")
            if isinstance(r2, str):
                reasoning = r2

    if not isinstance(reasoning, str):
        reasoning = ""

    print("\n===== PLANNER MESSAGE.content (rendered) =====")
    print(content.rstrip())

    if reasoning.strip():
        print("\n===== PLANNER MESSAGE.reasoning_content (rendered) =====")
        print(reasoning.rstrip())


def _require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _litellm_model_name(model_group: str) -> str:
    prefix = os.environ.get("LITELLM_MODEL_PREFIX", "litellm_proxy/").rstrip("/")
    return f"{prefix}/{model_group}".lstrip("/")


def call_planner_llm(*, document_id: str, user_question: str, planner_prompt_text: str) -> Dict[str, Any]:
    import litellm

    base_url = _require_env("LITELLM_BASE_URL")
    api_key = _require_env("LITELLM_API_KEY")

    model_group = os.environ.get("PLANNER_MODEL", "").strip() or os.environ.get("CHAT_MODEL", "").strip()
    if not model_group:
        raise RuntimeError("PLANNER_MODEL/CHAT_MODEL not set")

    temperature = float(os.environ.get("PLANNER_TEMPERATURE", "0.6"))
    timeout_s = int(os.environ.get("PLANNER_TIMEOUT", "90"))
    max_tokens = int(os.environ.get("PLANNER_MAX_TOKENS", "300"))

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
    import litellm

    base_url = _require_env("LITELLM_BASE_URL")
    api_key = _require_env("LITELLM_API_KEY")

    model_group = os.environ.get("ANSWER_MODEL", "").strip() or os.environ.get("CHAT_MODEL", "").strip()
    if not model_group:
        raise RuntimeError("ANSWER_MODEL/CHAT_MODEL not set")

    temperature = float(os.environ.get("ANSWER_TEMPERATURE", "0.2"))
    timeout_s = int(os.environ.get("ANSWER_TIMEOUT", "180"))
    max_tokens = int(os.environ.get("ANSWER_MAX_TOKENS", "600"))

    resp = litellm.completion(
        model=_litellm_model_name(model_group),
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        timeout=timeout_s,
        api_base=base_url,
        api_key=api_key,
        metadata={"document_id": document_id, "stage": "answer"},
    )

    return _normalize_litellm_obj(resp)


# -----------------------------------------------------------------------------
# Prompt loading
# -----------------------------------------------------------------------------

def load_planner_prompt() -> str:
    p = os.environ.get("PLANNER_PROMPT_PATH", "").strip()
    if not p:
        raise RuntimeError("PLANNER_PROMPT_PATH not set")
    return Path(p).read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# Orchestration (E2E)
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evidence-first query engine (canonical orchestrator)")
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--stream", action="store_true", help="Enable streaming answer progress (planner remains non-streaming)")
    ap.add_argument("query", help="User question")
    args = ap.parse_args(argv)

    # Trace file: one artifact per run
    ts_file = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    traces_dir = os.path.join("out", "traces")
    os.makedirs(traces_dir, exist_ok=True)
    trace_path = os.path.join(traces_dir, f"parsing_engine_{ts_file}_{args.document_id}.json")

    _orig_stdout, _orig_stderr = enable_process_trace(trace_path)

    print(f"[engine] trace_file={trace_path}")
    print(f"[engine] start document_id={args.document_id}")

    # Show config (non-secret)
    print(f"[engine] litellm_base={os.environ.get('LITELLM_BASE_URL','').strip()}")
    print(f"[engine] litellm_model_prefix={os.environ.get('LITELLM_MODEL_PREFIX','').strip()}")
    print(f"[engine] planner_model={os.environ.get('PLANNER_MODEL','').strip() or os.environ.get('CHAT_MODEL','').strip()}")
    print(f"[engine] answer_model={os.environ.get('ANSWER_MODEL','').strip() or os.environ.get('CHAT_MODEL','').strip()}")
    print(f"[engine] embeddings_model={os.environ.get('EMBEDDINGS_MODEL','').strip()}")
    if args.stream:
        print("[engine] --stream requested (planner is always non-streaming)")

    # 1) Planner
    planner_prompt = load_planner_prompt()
    planner_raw = call_planner_llm(document_id=args.document_id, user_question=args.query, planner_prompt_text=planner_prompt)
    _print_planner_raw_response(planner_raw)

    parsed = parse_planner_option_d(planner_raw)
    print("\n===== PLANNER OUTPUT (Option D) =====")
    print(json.dumps({
        "expansions": parsed.expansions,
        "keyword_mode": parsed.keyword_mode,
        "require_keywords": parsed.require_keywords,
        "exclude_keywords": parsed.exclude_keywords,
    }, indent=2, ensure_ascii=False))

    # 2) Retrieval
    retrieval_variant = os.environ.get("CONTENT_QUERY_VARIANT", "round1").strip()
    print("\n===== RETRIEVAL (E2E-2) =====")
    print(f"[retrieval] variant={retrieval_variant}")

    # Probes: original + expansions (dedup preserve order)
    probes: List[str] = []
    seen: set[str] = set()
    uq = str(args.query).strip()
    if uq and uq not in seen:
        probes.append(uq)
        seen.add(uq)
    for exp in parsed.expansions:
        exp = exp.strip()
        if exp and exp not in seen:
            probes.append(exp)
            seen.add(exp)

    print(f"[retrieval] total probes: {len(probes)}")
    for i, t in enumerate(probes):
        print(f"  [{i}] probe: {repr(t)}")

    retrieval_out = run_content_query(document_id=args.document_id, probes=probes, variant=retrieval_variant)

    fused = retrieval_out.get("fused_results", []) or []
    print(f"\n[retrieval] FUSION: total unique kept chunks: {len(fused)}")
    print("\n[retrieval] Top 25 fused chunks:")
    for rank, item in enumerate(fused[:25], 1):
        chunk = item.get("chunk", {})
        page = chunk.get("page_number", chunk.get("page", "NA"))
        chunk_id = chunk.get("chunk_id", chunk.get("id", "NA"))
        score = float(item.get("score", 0.0))
        via = item.get("via", [])
        print(f"  #{rank:2d} score={score:.4f} page={page} chunk_id={chunk_id} probes={via}")

    # 3) Answer (minimal baseline)
    print("\n===== ANSWER (E2E-3) =====")
    # Build an evidence prompt from fused chunks (simple concatenation baseline)
    max_chars = int(os.environ.get("RAG_MAX_EXCERPTS_CHARS", "4500").split("#", 1)[0].strip() or "4500")
    max_excerpts = int(os.environ.get("RAG_MAX_EXCERPTS", "6").split("#", 1)[0].strip() or "6")

    excerpts: List[str] = []
    used_chars = 0
    for item in fused:
        if len(excerpts) >= max_excerpts:
            break
        chunk = item.get("chunk", {})
        text_path = chunk.get("chunk_text_path") or chunk.get("text_path")
        snippet = ""
        if isinstance(text_path, str) and text_path:
            p = Path(text_path)
            if not p.is_absolute() and os.environ.get("PROJECT_ROOT"):
                p = Path(os.environ["PROJECT_ROOT"]) / text_path
            if p.exists():
                snippet = p.read_text(encoding="utf-8", errors="replace").strip()
        if not snippet:
            snippet = json.dumps(chunk, ensure_ascii=False)

        header = f"[chunk_id={chunk.get('chunk_id','NA')} page={chunk.get('page_number', chunk.get('page','NA'))} score={item.get('score')}]"
        block = header + "\n" + snippet
        if used_chars + len(block) > max_chars:
            break
        excerpts.append(block)
        used_chars += len(block)

    evidence_blob = "\n\n---\n\n".join(excerpts)
    answer_prompt = (
        "Answer the user question using ONLY the provided evidence. "
        "If the evidence does not contain the answer, say so.\n\n"
        f"USER QUESTION:\n{args.query.strip()}\n\n"
        f"EVIDENCE:\n{evidence_blob}\n"
    )

    answer_resp = call_answer_llm(document_id=args.document_id, prompt_text=answer_prompt, stream=bool(args.stream))

    # Print a minimal normalized answer
    try:
        answer_text = answer_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        answer_text = ""
    print("\n===== ANSWER (model output) =====")
    print(str(answer_text).strip())

    # Restore stdout/stderr
    restore_process_streams(_orig_stdout, _orig_stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
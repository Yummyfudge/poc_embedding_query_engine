# Changelog

All notable changes to this project will be documented here.

This project is a research / systems prototype. Entries focus on architectural
changes, capability shifts, and correctness improvements rather than polish.

---

## v0.3.3-alpha —29 December 2025

### Added
- Unified PDF intake runner (`run_intake_and_query.py`) for deterministic, one‑command ingestion
- Fully modularized intake pipeline under `src/intake/` (PDF loading, chunking, embedding, refinement)
- Canonical query orchestration under `src/engine/` with clear worker vs. orchestrator separation
- Dedicated planner parsing module (`planner_parser.py`) with tolerant, inspectable parsing
- Centralized trace writer producing single per‑run artifacts for end‑to‑end inspection
- Markdown/PDF query run rendering pipeline for human review of evidence selection
- Stable pytest coverage for planner parsing and retrieval logic (greenfield modules)

### Changed
- Migrated all active logic out of `scripts/` into `src/` bounded contexts
- Separated **intake** (one‑time, expensive) from **querying** (repeatable, cheap)
- Replaced ad‑hoc runners with explicit, reproducible pipelines
- Standardized environment configuration via `example.runtime.env`
- Clarified naming and responsibilities across intake, retrieval, and inference layers

### Deprecated
- All `scripts/LEGACY-*` files
  - Quarantined outside the active codepath
  - Retained only temporarily for reference during transition
- Direct orchestration inside `answer_with_evidence.py`
  - Now frozen and no longer evolved

### Fixed
- Multiple silent failure modes in retrieval and planner parsing
- Mismatched embedding / metadata edge cases
- Test brittleness caused by implicit imports and duplicated modules

### Notes
- This milestone marks the transition from exploratory scripts to a coherent, testable system
- The project is now structured to support deeper retrieval research without architectural churn

---

## v0.3.0 — 28 December 2025

### Notes
- Architectural refactor milestone (.sh → .py, bounded contexts)
- No distinct feature set beyond groundwork for v0.3.3-alpha

---

## v0.2.6 — 27 December 2025

### Added
- First-class retrieval engine (`src/query/content_query.py`) with explicit probe-based vector search
- Deterministic retrieval round 1 with per-probe scoring, elbow-based cutoff, and fusion
- Engine-level orchestration split (`query_engine`, `planner_parser`, `trace_writer`)
- Comprehensive unit tests for planner parsing and retrieval logic
- Human-readable query run inspection artifacts (markdown/PDF pipeline groundwork)

### Changed
- Separated orchestration from retrieval and inference concerns
- Migrated retrieval logic out of legacy scripts into bounded contexts under `src/`
- Hardened planner parsing to support JSON and structured plaintext outputs
- Improved trace fidelity with full per-probe and per-chunk visibility
- Reduced reliance on implicit behavior; all major knobs now config-driven

### Deprecated
- Direct use of `scripts/answer_with_evidence.py` as an orchestrator
  - Now treated as legacy and frozen for reference
  - Orchestration moving fully into `src/engine/`

### Fixed
- Multiple retrieval correctness issues uncovered via trace inspection
- Test coverage gaps that previously masked parsing and fusion errors

---

## v0.2.2 — December 2025

### Added
- End-to-end evidence-bound query pipeline (`run_query.py` → `answer_with_evidence.py`)
- Planner → retrieval → answer architecture with explicit phase separation
- Streaming query execution with rolling progress output
- Full per-run trace artifacts written to `out/traces/` (JSONL)
- Post-run human-readable query inspection via `render_latest_query_run.py`
- Config-driven behavior via `example.runtime.env` (no hardcoded knobs)
- Support for multiple retrieval “probes” per query (original + planner expansions)
- Embedding backend abstraction using Infinity via LiteLLM
- Chunk-level scoring visibility (probe → score → chunk mapping)

### Changed
- Replaced monolithic `ask_rag.py` with `answer_with_evidence.py` (greenfield engine)
- Refactored retrieval logic into explicit, inspectable stages
- Reduced LLM context size to improve cold-start performance and memory pressure
- Moved toward single-run artifacts instead of scattered debug prints
- Tightened PostgreSQL connection handling (explicit DSN, SSL verification)

### Deprecated
- `scripts/LEGACY-ask_rag.py`
- `scripts/LEGACY-answer_with_llm.py`
  - Retained temporarily for reference and comparison
  - Will be removed once greenfield engine reaches full parity

### Removed
- Ad-hoc bootstrap scripts and implicit environment coupling
- Implicit trust in LLM output formatting (now tolerant, defensive parsing)

### Known Limitations
- Retrieval quality degrades for long-range semantic context
- Vector similarity alone is insufficient for nuanced legal / medical language
- No entity-level normalization or domain-specific embeddings yet
- Answer correctness still depends heavily on chunk segmentation quality

---

## v0.1.0 — Mid December 2025

### Added
- PDF ingestion and page chunking pipeline
- Basic embedding generation and similarity search
- Early RAG proof-of-concept

### Notes
- This phase intentionally favored speed over correctness
- Many scripts from this era are retained only for historical context
# Evidence‑First Document Query Engine (POC)

## What this is

This project began as a response to a very real problem: finding specific, legally relevant facts inside a single document that was over 1,500 pages long.

Not summarizing it.
Not “asking questions” of it.
Finding *exactly* where a fact appears, with evidence you can point to.

That constraint — evidence first, reasoning second — shaped everything that followed.

What started as a narrowly successful proof‑of‑concept has since evolved into a generalized query engine designed to operate in adversarial, high‑stakes document environments such as insurance claims, medical records, and appeals.

This repository represents the midpoint of that evolution: a working system being refactored into a reusable, principled architecture.


It is intentionally not polished. It is intentionally honest.

## Context: Empowering Self‑Advocacy for Insurance Denials & Appeals

This work sits inside a larger, long‑running effort focused on **empowering self‑advocacy for insurance claim denials and appeals**.

For roughly the past 8–9 months, I’ve been building systems to help individuals navigate opaque, adversarial insurance processes—where decisions are buried across hundreds or thousands of pages, timelines are fragmented, and outcomes hinge on locating precise language and dates.

That broader project includes intake pipelines, document provenance tracking, and audit‑friendly storage. **This repository is the technical highlight of that effort**: the query engine that makes large, hostile documents *inspectable*—so a person can point to evidence, not just conclusions.

While the broader project provides the real‑world motivation, this repo stands on its own as a reusable system for evidence‑first querying in high‑stakes domains.

## Why this exists

Most modern “AI document search” systems collapse three very different concerns into one:

- retrieval
- reasoning
- trust

This project refuses to do that.

Instead, it treats each concern as a separate system with explicit boundaries and measurable behavior.

## This project is not “RAG”

It is a query engine that uses:

- **LLMs for reasoning**
- **ML for representation**
- **Math for ranking**
- **Code for control**

That separation is what makes it reliable.

LLMs are used to *understand intent and propose search strategy*, not to invent answers.
Vector search is used to *surface candidates*, not to declare relevance.
Mathematical thresholds and clustering determine what survives.
Code enforces what is allowed to pass downstream.

## Limitations (by design)

This system is very good at locating words, phrases, and short passages that align with a query.

It is **not** yet good at fully understanding long‑range narrative context.

Embedding‑based representations flatten meaning:
- the broader story a paragraph participates in is partially lost,
- overloaded terms (“claim”, “determination”, “approval”) bleed across domains,
- trust requires more than cosine similarity.

Solving that — via richer metadata, entity mapping, and domain‑specific representations — is an explicit future goal, not an unacknowledged flaw.

Until then, this engine is designed to be *inspectable*, *auditable*, and *conservative*.

## Minimal Demonstration (End‑to‑End)

> This section demonstrates the smallest complete path from a raw PDF to an evidence‑bound answer.
>
> Nothing here is meant to be magical or automatic. Each step exists so the system’s behavior can be inspected, repeated, and challenged.

There are two distinct phases: Intake (one‑time, per document) and Query (repeatable, evidence‑first).

### 1) Create the environment

```bash
conda env create -f environment.yml
conda activate pdf_to_parse_poc
pip install -e .
```

Optional extras:

```bash
pip install -e ".[render]"   # enables PDF generation for rendered run reports
pip install -e ".[dev]"      # developer tools
```

### 2) Create local-only config files

This repository does **not** track secrets or machine-specific paths. Copy the examples:

```bash
cp config/example.runtime.env config/runtime.env
cp example.gitignore .gitignore
```

Edit `config/runtime.env` for your machine (LiteLLM, Postgres DSN, paths).

### 3) Intake a PDF (one‑time, per document)

To ingest a new PDF for querying, use:

```bash
python scripts/run_intake_and_query.py --pdf /path/to/file.pdf
```

This will:
- Generate a new `document_id` for the ingested PDF
- Copy the PDF into `out/intake_runs/<DOCUMENT_ID>/source.pdf`
- Load pages, create chunks, and compute embeddings
- Print the resulting `document_id` to use for queries

> **Note:** Intake is required once for each PDF document. After a PDF is ingested, you can run queries against it repeatedly using its `document_id`.

### 4) Run queries (repeatable, evidence‑first)

Once your PDF has been intaken and you have its `document_id`, run queries like:

```bash
python scripts/run_query.py --document-id <DOCUMENT_UUID> "Your question"
```

Replace `<DOCUMENT_UUID>` with the `document_id` printed during the intake step.

### 5) Render the latest run into something readable

After syncing `out/` to your Mac, you can generate a human-readable report:

```bash
python scripts/render_latest_query_run.py
```

This writes `report.md` (and `report.pdf` if ReportLab is installed) under `out/parsed_query_runs/`.

## Artifacts and Outputs

- `out/traces/` — per-run trace transcripts (JSONL/NDJSON). These are meant to be inspectable and diffable.
- `out/parsed_query_runs/` — rendered run reports (`report.md` / `report.pdf`) that gather the selected chunks into a readable artifact.
- `out/pages_txt/` — page text files
- `out/pages.jsonl` — page index (one JSON record per page)

> `out/` is intentionally ignored and should not be committed.

## Query engine overview

The current greenfield query engine lives in:

```
scripts/answer_with_evidence.py
```

This engine implements a **multi‑stage, evidence‑bound retrieval pipeline**:

1. **Planner (LLM)**
   - Interprets the user’s question
   - Preserves intent exactly
   - Proposes retrieval parameters and query refinements

2. **Embedding & vector search**
   - Query text is embedded via Infinity
   - Local NumPy similarity search is performed against precomputed embeddings

3. **Post‑filtering & budgeting**
   - Per‑page caps
   - Score thresholds
   - Deduplication
   - Excerpt size limits to keep prompts bounded

4. **Evidence‑bound answer**
   - Only retrieved excerpts are passed to the answer model
   - Page citations are required

Streaming output is supported end‑to‑end and controlled via a single environment variable.

## High-level architecture (current state)

```mermaid
flowchart LR
  User[User Question]
  Runner[run_query.py]
  Planner[Planner LLM]
  CQ[content_query.py]
  Emb[Embeddings (Infinity)]
  Vec[Vector Math + Clustering]
  Evidence[Selected Evidence Chunks]
  Answer[Answer LLM]
  Trace[Trace Writer]

  User --> Runner
  Runner --> Planner
  Planner --> CQ
  CQ --> Emb
  Emb --> Vec
  Vec --> Evidence
  Evidence --> Answer
  Planner --> Trace
  CQ --> Trace
  Answer --> Trace
```

This architecture ensures that evidence is selected before answering, so the LLMs do not see the full document. Every stage emits trace data to support inspection and debugging. The design is intentionally transparent and auditable to maintain reliability in high-stakes environments.

## Legacy files

Some older scripts are retained **only for reference** during the transition:

- `scripts/LEGACY-ask_rag.py`
- `scripts/LEGACY-answer_with_llm.py`

These are no longer the source of truth and will be removed once the greenfield engine reaches full parity.

## Future work

Planned improvements and cleanup:

- **Retrieval tuning**
  - Multi‑query aggregation strategies (max / sum / RRF)
  - Keyword‑aware gating derived from planner intent
  - Additional retrieval diagnostics and trace outputs

- **Script reorganization**
  - Move scripts into clearer domains, e.g.:
    - `bin/intake/`
    - `bin/query/`
    - `bin/tools/`
  - Keep thin wrappers for backwards compatibility during transition

  - **Multi-domain embeddings**
  - Run multiple embedding models side-by-side (e.g. general, medical, legal)
  - Allow planner intent to select embedding domain or query multiple domains
  - Support fusion strategies across embedding spaces (max, sum, RRF)
  - Keep embedding domain selection fully configurable via runtime.env

- **NLP enrichment**
  - Optional one‑time enrichment with scispaCy / UMLS
  - Store entity and concept metadata in Postgres
  - Make enrichment opt‑in at query time

- **Cleanup**
  - Remove legacy scripts once the new engine is stable
  - Reduce POC‑era duplication

## Safety

This project may process PHI/PII. Do not commit extracted text, images, or `.env` files.

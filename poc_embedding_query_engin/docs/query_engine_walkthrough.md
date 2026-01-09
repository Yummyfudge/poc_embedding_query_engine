# Query Engine Walkthrough

This document explains the **end‑to‑end query engine** used in this project.  
It is intentionally structured to highlight how *multiple forms of AI* are used together:
- Large Language Models (LLMs)
- Embedding models (ML)
- Classical vector math (linear algebra)
- Deterministic filtering and budgeting logic

The goal is **traceability, controllability, and evidence‑bound answers** — not “magic”.

---

## High‑Level Philosophy

We treat each phase of the query pipeline as a **first‑class, tunable component**.

Key design principles:

- **Separation of concerns**
- **Explicit hand‑offs between phases**
- **Config‑driven behavior (env > code)**
- **Streaming visibility into long‑running operations**
- **Evidence‑bound answers with citations**

The pipeline is deliberately *not* a single opaque “RAG call”.

---

## Pipeline Overview (M‑Stages)

```
User Question
   ↓
M1: Planner (LLM – intent + retrieval strategy)
   ↓
M2: Query Expansion & Keyword Extraction (LLM output)
   ↓
M3: Embedding & Vector Search (ML + math)
   ↓
M4: Post‑Filtering & Budgeting (deterministic)
   ↓
M5: Evidence Prompt Assembly
   ↓
M6: Answer Generation (LLM, evidence‑bound)
   ↓
(Optional) M7: Refinement Loop
```

Each stage is described below.

---

## M1 — Planner (LLM)

**Purpose:**  
Understand *what the user is actually asking* and propose how to search for it.

**Characteristics:**
- Preserves the **original intent** exactly
- Does *not* answer the question
- Emits a **structured JSON plan**, not prose

**Typical planner output includes:**
- Query expansions (re‑phrasings)
- Candidate counts
- Chunk limits
- Keyword gates (required / excluded terms)

This is where “understanding” happens — not retrieval.

---

## M2 — Query Expansion & Keyword Extraction (LLM Output)

**Purpose:**  
Convert user intent into **multiple retrieval signals**.

Planner output is interpreted as:

- **Query expansions**  
  → Used for embedding + vector similarity

- **Keyword gates**  
  → Used for lexical filtering (not embeddings)

These are treated differently on purpose.

> Infinity only embeds text — keyword logic lives *outside* the embedding model.

---

## M3 — Embedding & Vector Search (ML + Math)

**Purpose:**  
Numerically score document chunks against the expanded queries.

Steps:
1. Each query expansion is embedded via **Infinity**
2. Stored document embeddings are loaded (NumPy)
3. Similarity is computed using matrix multiplication
4. Results are aggregated across queries

This stage is:
- Pure ML + linear algebra
- Fast, deterministic, inspectable
- Completely model‑agnostic once embeddings exist

---

## M4 — Post‑Filtering & Budgeting (Deterministic)

**Purpose:**  
Enforce **precision, diversity, and size limits**.

Examples:
- Per‑page caps
- Minimum similarity score
- De‑duplication
- Total excerpt character budget
- Required / excluded keyword checks

This stage prevents:
- Over‑representation from one page
- Prompt blow‑ups
- Hallucination‑friendly noise

No AI is used here — by design.

---

## Evaluating Retrieval Quality (Evidence‑First)

Retrieval quality is explicitly evaluated *before* answer generation.  
The key question is: **“Did it find the right evidence, even if the answer isn’t generated yet?”**

Early pipeline iterations may stop after M4 to inspect evidence quality.  
Trace artifacts and rendered reports are used for manual inspection and validation.

### Primary Tuning Knobs

- **per‑probe top_k:** Controls how many top candidates are selected per query expansion probe.
- **elbow sensitivity (gap‑based cutoff):** Determines the threshold for cutting off results based on score gaps.
- **fusion behavior (e.g., max score vs. aggregation across probes):** Defines how scores from multiple probes are combined.

These knobs affect *retrieval behavior only* and do not change LLM prompting.

### Why Tuning Happens Here

- This aligns with the project’s evidence‑first philosophy.  
- Incorrect retrieval cannot be fixed by better prompting.  
- Observability at this stage prevents hallucination downstream.

---

## M5 — Evidence Prompt Assembly

**Purpose:**  
Construct the *only* information the answer model is allowed to see.

The prompt contains:
- Selected excerpts
- Page numbers / chunk IDs
- Explicit instruction to **cite evidence**

The answer model never sees:
- The full document
- Unfiltered embeddings
- Planner reasoning
- Hidden metadata

---

## M6 — Answer Generation (LLM)

**Purpose:**  
Generate a final answer **strictly bounded by evidence**.

Key constraints:
- No new facts
- No unstated assumptions
- Explicit page citations
- Failure is allowed (e.g., “not found”)

Streaming is enabled here to:
- Observe progress
- Capture partial output
- Avoid silent timeouts

---

## M7 — Optional Refinement Loop

**Purpose:**  
Retry with adjusted retrieval parameters *only if needed*.

Examples:
- Increase candidate count
- Expand excerpt budget
- Adjust similarity threshold

Refinement is **explicit and bounded** — not recursive guessing.

---

## Why This Architecture Matters

This system demonstrates that “AI” is not one thing:

- **LLMs** handle interpretation and language
- **Embedding models** handle semantic proximity
- **Linear algebra** does the heavy lifting
- **Rules & budgets** enforce correctness

Each part is observable, testable, and tunable.

That is the point.

---

## Current Focus

The current development focus is:
- Completing and stabilizing M3–M4 retrieval
- Instrumenting traces and diagnostics
- Deferring answer fluency optimization until retrieval is trustworthy

---

## TL;DR

This project is not “RAG”.

It is a **query engine** that uses:
- LLMs for reasoning
- ML for representation
- Math for ranking
- Code for control

And that is what makes it reliable.
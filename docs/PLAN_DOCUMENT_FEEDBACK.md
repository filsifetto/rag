# Plan: Document Feedback and Reference-Suggestion System

**Status:** Implemented (loader, models, feedback service, query strategy, CLI, tests).

This document describes how we added the **document feedback** feature: upload a file related to a subject, have the LLM read it, query the subject RAG database for relevant content, and return feedback with proposed changes and **specific suggestions for where to insert references** from the subject material.

---

## 1. Goal

- **User action:** Upload a file (e.g. essay, report) and choose a subject whose material is already in the RAG database.
- **System behaviour:**
  1. Read the uploaded file’s text.
  2. Query the subject’s vector collection for content relevant to the file.
  3. Use the LLM to produce:
     - **Feedback** on the document (quality, clarity, gaps).
     - **Proposed changes** (edits, improvements).
     - **Reference suggestions:** concrete places in the document where the author should add citations, with the exact APA reference and the supporting snippet from the subject material.

- **Non-goals for this step:** Web UI (can be CLI-only first); modifying ingested subject documents; real-time collaboration.

---

## 2. What We Reuse (Existing Interfaces)

We keep existing components as-is and call them from new orchestration code. No changes to their public APIs unless necessary.

| Component | Location | How we use it |
|-----------|----------|----------------|
| **Search** | `core/services/search_engine.py` | `search_engine.search(query, limit=..., filters=...)` → `List[SearchResult]`. We call it with one or more queries derived from the user’s document. |
| **Response generation (Q&A)** | `core/services/response_generator.py` | Unchanged. Still used for interactive “ask” in the CLI. |
| **Citation** | `core/citation.py` | `format_apa_inline`, `format_apa_reference`, `build_citation_key` to format suggested references. |
| **Config / subject** | `core/config.py` | `apply_subject(settings, subject)` so we target the correct Qdrant collection for the chosen subject. |
| **Embedding** | `core/services/embedding_service.py` | Used indirectly by the search engine for query embedding. |
| **Document loading** | See §4 | We introduce a **shared** loader so ingestion and document feedback both use the same “file → text + metadata” path. |

---

## 3. New Components and Where They Live

### 3.1 Shared document loading (clean interface)

- **Problem:** Today “load a single file to text” lives in `scripts/ingest_documents.py` (`load_single_file`). Document feedback should use the same behaviour without depending on the ingest script.
- **Approach:** Move or mirror the “load one file → text + metadata (and optional page_list)” logic into **core**, so both ingestion and document feedback depend on one interface.
- **Options:**
  - **A)** Add `core/loaders/document_loader.py`: functions like `load_file_to_text(path) -> Tuple[str, dict]` (and optionally `page_list` for PDFs/TXT with markers). Ingest script and feedback script both call this.
  - **B)** Keep parsing in `core/parsers/`, add a thin `core/loaders/` (or a single module) that composes parsers and returns a simple “user document” type (text + metadata). Same idea: one place for “file → content we can work with”.
- **Interface:** `load_user_document(file_path: Path) -> UserDocument` (or similar) where `UserDocument` has `content: str`, `metadata: dict`, and optionally `page_list` or sections. No dependency on ingestion-specific types beyond what’s needed for parsing.

### 3.2 Document feedback service (new)

- **Location:** `core/services/document_feedback_service.py`.
- **Responsibility:** Given the user’s document text and a set of retrieved subject-matter excerpts (from search), call the LLM to produce structured feedback and reference suggestions. **Does not** perform search or file I/O; it receives text and search results (or a pre-built context string).
- **Inputs:** `document_text: str`, `search_results: List[SearchResult]` (or a single context string built from them). Optional: `custom_instructions: str`.
- **Output:** A Pydantic model, e.g. `DocumentFeedback` with:
  - `feedback_summary: str`
  - `suggested_edits: List[SuggestedEdit]` (optional)
  - `suggested_references: List[SuggestedReference]` where each has:
    - `place_in_document: str` (e.g. quote or sentence where to add the citation)
    - `citation_apa: str` (full or inline APA)
    - `source_snippet: str` (excerpt from the subject material)
    - `reason: str` (why this citation fits here)
- **Implementation:** Use the same LLM client as `ResponseGenerator` (or a shared base) and the same citation helpers to format source info in the prompt. Different system prompt and `response_model=DocumentFeedback` (with Instructor or manual parsing). This keeps “Q&A response” and “document feedback” separate but consistent (same config, same citation style).

### 3.3 Document feedback models

- **Location:** `core/models/document_feedback.py` (new file).
- **Contents:** Pydantic models: `DocumentFeedback`, `SuggestedReference`, and optionally `SuggestedEdit`. Used by the document feedback service and by the script that displays or saves output.

### 3.4 Query strategy for “document → search queries”

- **Responsibility:** Turn the user’s document into one or more search queries so we retrieve relevant subject material. Options:
  - **Simple:** One query = first N characters of the document (or whole doc if short). Single search call.
  - **Richer:** Split document into sections (e.g. by headings or by paragraph chunks), run one search per section, aggregate and deduplicate results. Better for “suggest a reference for this paragraph”.
- **Where it lives:** Either a small helper in `document_feedback_service.py` or a dedicated module (e.g. `core/services/document_query_strategy.py`). Prefer a single place so we can change strategy (e.g. “one query” vs “per-section”) without touching the rest of the pipeline.

### 3.5 Orchestration script (CLI)

- **Location:** `scripts/document_feedback.py` (or `feedback_on_document.py`).
- **Flow:**
  1. Parse CLI: `--file <path>`, `--subject <subject_id>` (required for subject-scoped search).
  2. Load file using the shared loader (§3.1).
  3. Apply subject: `settings = apply_subject(settings, args.subject)`; create search engine (and embedding service) with these settings.
  4. Build search query/queries from the document (using the chosen strategy).
  5. Call `search_engine.search(query, limit=...)` (and aggregate if multiple queries).
  6. Call document feedback service with document text and search results.
  7. Print or write the structured feedback and suggested references (e.g. to stdout or to a file).
- **Dependencies:** Only core: loaders, config, search engine, embedding service, document feedback service, models. No tight coupling to other scripts.

---

## 4. Interface Boundaries (Keep Clean)

- **core/loaders (or parsers):** File path → text + metadata (and optional structure). Used by ingest and by document feedback. **Ingestion** may still add its own logic (e.g. chunking, embedding, storing); the loader only does “read file → content”.
- **core/services/search_engine.py:** No change. Continues to accept `query: str`, optional `filters`, and return `List[SearchResult]`. Subject is selected via config (collection name) before constructing the engine.
- **core/services/response_generator.py:** No change. Used for Q&A only.
- **core/services/document_feedback_service.py:** Depends only on: settings, LLM client (or shared base), citation helpers, and `SearchResult` / `DocumentFeedback` models. Does not depend on document store, ingestion, or CLI.
- **core/citation.py:** No change; used as a library.
- **Scripts:** Thin orchestration: parse args → load file (core) → search (core) → feedback service (core) → output.

This allows:
- Replacing the “document feedback” LLM prompt or output schema without touching search or ingestion.
- Changing search (e.g. different ranking) without touching feedback logic.
- Adding a web UI later that calls the same core services (loader + search + document feedback service).

---

## 5. Implementation Order

1. **Shared loader**  
   Add `core/loaders/document_loader.py` (or equivalent) with `load_user_document(path)` using existing parsing (PDF, TXT, etc.). Refactor `scripts/ingest_documents.py` to use it so we don’t duplicate logic. Tests: load a sample TXT and PDF.

2. **Models**  
   Add `core/models/document_feedback.py` with `DocumentFeedback`, `SuggestedReference`, and optionally `SuggestedEdit`.

3. **Document feedback service**  
   Implement `DocumentFeedbackService` in `core/services/document_feedback_service.py`: build context from `List[SearchResult]` (reuse citation formatting), call LLM with a dedicated system prompt, return `DocumentFeedback`. Use same config (e.g. `OPENAI_MODEL`) and citation helpers.

4. **Query strategy**  
   Implement a simple strategy first: one query = document text truncated to a safe length (e.g. 2000 chars). Optionally add a “per-section” strategy later. Place in service or small helper module.

5. **CLI script**  
   Implement `scripts/document_feedback.py`: `--file`, `--subject`, load file, apply subject, run search, run feedback service, print/save result. Document in README.

6. **Docs and README**  
   Update README with current project state, subject-based workflow, and the new “Document feedback” section. Add docstrings and, if useful, a short `docs/ARCHITECTURE.md` or extend this plan with an “Implemented” section.

---

## 7. Documentation and Code Style

- **Docstrings:** All public functions and classes in core (loaders, feedback service, new models) get clear docstrings (purpose, args, returns, and any subject/collection assumptions).
- **README:** Reflect current state: hybrid search, subject-based collections, page-aware chunking, interactive search CLI, and the new document feedback CLI. Keep “Getting Started” and “Configuration” accurate.
- **This plan:** Keep `docs/PLAN_DOCUMENT_FEEDBACK.md` as the single place for the design; update it when we deviate or complete steps (e.g. “§5.1 Done: loader in core/loaders/document_loader.py”).

---

### Implemented (current state)

- Shared loader: `core/loaders/document_loader.py`; ingest uses it.
- Models: `core/models/document_feedback.py`; tests in `tests/test_document_loader.py`, `tests/test_document_feedback_models.py`.
- Document feedback service and query strategy; tests in `tests/test_document_feedback_service.py`, `tests/test_document_query_strategy.py`.
- CLI: `scripts/document_feedback.py` (--file, --subject). README updated.

---

## 8. Optional Later Steps

- **Web UI:** File upload form, subject dropdown, display of feedback and suggested references (and optionally export).
- **Per-section search:** Split document by headings or paragraphs, query per section, merge results and pass section-aware context to the LLM for more precise “add reference here” suggestions.
- **Export:** Write feedback and suggested references to a structured file (e.g. JSON or Markdown) for use outside the CLI.

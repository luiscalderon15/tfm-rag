# 🎯 Job Hunter — RAG Recruiting Assistant

A Retrieval-Augmented Generation system that screens anonymized candidate CVs against a job description, ranks and explains the shortlist, answers free-form questions about individual candidates, and drafts outreach emails — all through Streamlit UIs backed by a hybrid (BM25 + FAISS) retriever and a swappable LLM layer (Ollama / Azure OpenAI).

> 🎓 MSc Data Science TFM project.

---

## ✨ Apps

Three independent Streamlit front-ends sit on top of the same retrieval + LLM backend. Each covers a different interaction shape:

| App | Purpose | Entry point |
|---|---|---|
| 🖥️ **Screening Assistant** | Paste a job description → retrieves, ranks, and evaluates the whole candidate pool, writes a narrative summary per candidate, drafts outreach emails. | [`app.py`](app.py) |
| 💬 **Candidate Chat** | Pick one candidate by ID → free-form Q&A grounded strictly in *their* full CV (no ranking involved). | [`candidate_chat.py`](candidate_chat.py) |
| 🤖 **Agentic Router** *(experimental)* | Single free-text chat box; an LLM with tool-calling picks **one** of the four underlying tools per message (`screen_candidates`, `get_candidate`, `answer_about_candidate`, `draft_outreach_email`) and tracks the "current candidate" across turns for follow-ups. | [`router_app.py`](router_app.py) / [`src/router.py`](src/router.py) |

### ▶️ Running an app

```bash
streamlit run app.py                                              # port 8501
streamlit run candidate_chat.py --server.port 8502                # port 8502
streamlit run router_app.py --server.port 8503                    # port 8503
```

(See [`.claude/launch.json`](.claude/launch.json) for the exact launch configs, including the conda interpreter path used locally.)

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
   PDF CVs  ──────▶ │  Extraction          │  src/extraction.py (docling / PyMuPDF)
                    └─────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Anonymization       │  src/anonimization.py (Presidio + spaCy)
                    │                       │  src/patterns.py (regex PII rules)
                    └─────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Embedding + Index    │  src/embedding.py (BAAI/bge-m3)
                    │                       │  src/vectorstore.py (FAISS, per-pool)
                    └─────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Hybrid Retrieval     │  src/search/hybrid_retriever.py
                    │  BM25 + FAISS → RRF   │  src/search/fussion.py, keywords.py
                    │  facet rollup, rerank │  src/rollup.py, search/reranker.py
                    └─────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  LLM layer            │  src/llm.py (Ollama / Azure OpenAI)
                    │  screening, narrative,│  src/bot.py
                    │  QA, email draft      │
                    └─────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Streamlit UIs        │  app.py / candidate_chat.py / router_app.py
                    └─────────────────────┘
```

### 🔍 Retrieval — `src/search/hybrid_retriever.py`

`HybridCandidateRetriever` runs a **two-stage** pipeline:

1. **Chunk-level recall per facet** — BM25 (`keywords.py`) and FAISS semantic search are each run independently, then fused with **Reciprocal Rank Fusion** (`fussion.py`).
2. **Candidate-level ranking** — per-facet chunk scores are rolled up to the best-matching chunk per candidate (MaxSim, `rollup.py`), fused again across facets via RRF, and (optionally) reranked with a cross-encoder.

✅ **Current justified defaults** (from the retrieval evaluation phase — see `config/models.yml` comments): `retrieval_mode: hybrid` and `use_rerank: false` — hybrid was never the worst signal across 3 pools × 2 ground truths, and cross-encoder reranking *worsened* results in that evaluation.

`get_candidate(candidate_id)` bypasses ranking entirely and returns the candidate's **full anonymized CV** as a single evidence block — used by Candidate Chat and the router, not by the screening flow.

### 🧠 LLM layer — `src/llm.py` + `src/bot.py`

- `src/llm.py` builds a LangChain chat client (`ChatOllama` or `ChatOpenAI` pointed at an Azure endpoint) per model config, resolves per-model temperature rules (e.g. reasoning models that reject custom temperature), and exposes:
  - `chat_structured(...)` — schema-constrained JSON output (Pydantic) for every screening/narrative/QA/email call.
  - `get_tool_calling_client(...)` — raw client for `.bind_tools(...)`, used only by the router.
- `src/bot.py` holds every prompt + Pydantic schema:
  - `evaluate_candidates` — ranks the shortlist, matched requirements / gaps per candidate.
  - `generate_candidate_narratives` — a **separate stage-2 LLM call** producing a "Candidate Summary" + "Recruiter Assessment" per candidate; this is the only place `about_me` is used in the screening flow (known limitation: batch narrative generation has a self-consistency limit across candidates, accepted trade-off).
  - `answer_about_candidate` — grounded, single-candidate free-form Q&A; explicitly instructed to say "I don't know / not stated in the CV" instead of inventing when the CV lacks the answer.
  - `draft_outreach_email` — drafts subject + body for one evaluated candidate (never sends anything).

### 🧭 Router — `src/router.py`

Wraps the four pipeline functions above as LangChain tools and lets an LLM pick **at most one** per user message (no chaining). Tracks `current_candidate_id` as explicit, code-managed state so follow-ups ("¿y su experiencia?") don't need to repeat the candidate ID — the LLM never guesses this ID itself.

---

## ⚙️ Configuration — `config/models.yml`

```yaml
llm_provider: llm_gpt        # which llm_* block is active by default

llm_opensource:               # free, local
  provider: ollama
  deployment_name: qwen2.5:7b-instruct

llm_nano:                     # Azure — reasoning model, no custom temperature
  provider: azure
  deployment_name: o4-mini

llm_gpt:                      # Azure — general purpose
  provider: azure
  deployment_name: gpt-4.1-mini
  temperature: 0

retrieval:
  retrieval_mode: hybrid      # semantic | keyword | hybrid
  use_rerank: false           # cross-encoder rerank — off by default (see above)
  ...
```

Every `llm_*` block is auto-discovered (e.g. by the router's provider picker) — no code changes needed to add a new one, as long as:
- the `deployment_name` matches an existing Azure deployment under the same endpoint/key, **or** an Ollama model tag pulled locally.
- `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` are set once, globally, in `.env` (shared by *all* Azure deployments — there's no per-model endpoint config).

---

## 📦 Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (never commit this) with:

```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
```

Ollama models (e.g. `qwen2.5:7b-instruct`) must be pulled locally beforehand via `ollama pull <model>`.

---

## 📁 Project structure

```
├── app.py                    # 🖥️ Screening Assistant (main flow)
├── candidate_chat.py         # 💬 Single-candidate chat
├── router_app.py             # 🤖 Agentic router (experimental)
├── config/
│   ├── models.yml             # LLM + retrieval configuration
│   └── loader.py
├── src/
│   ├── bot.py                 # Prompts, schemas, LLM-facing pipeline functions
│   ├── llm.py                 # LangChain client construction (Ollama/Azure)
│   ├── router.py               # Tool-calling router for router_app.py
│   ├── eval.py                 # Facet generation, synthetic-JD generation for evaluation
│   ├── benchmark.py            # PDF-extraction benchmarking (PyMuPDF/docling)
│   ├── extraction.py           # CV field extraction
│   ├── anonimization.py         # Presidio/spaCy PII anonymization
│   ├── patterns.py              # Regex PII patterns
│   ├── embedding.py              # Embedding model wrapper (BAAI/bge-m3)
│   ├── vectorstore.py            # FAISS build/load helpers
│   ├── rollup.py                  # Chunk→candidate MaxSim rollup, facet fusion
│   └── search/
│       ├── hybrid_retriever.py     # HybridCandidateRetriever (see above)
│       ├── fussion.py               # Reciprocal Rank Fusion
│       ├── keywords.py               # BM25 index
│       └── reranker.py                # Cross-encoder reranker
├── notebooks/                 # Exploration: extraction, evaluation, retrieval ablations
└── test/                      # Test suite
```

---

## 🧪 Evaluation

- Retrieval was benchmarked across 3 candidate pools × 2 ground truths, comparing `semantic` / `keyword` / `hybrid` modes and rerank on/off — see `notebooks/retrieve-evaluation.ipynb` and the conclusions baked into `config/models.yml`'s defaults above.
- PDF-extraction methods (PyMuPDF vs. docling) were timed/compared in `src/benchmark.py` and `notebooks/extraction_times.ipynb`.
- Synthetic job descriptions for evaluation are generated per-candidate via `src/eval.generate_synthetic_jd` (deliberately paraphrased/generalized so the JD doesn't leak exact CV wording).

---

## ⚠️ Known limitations

- **Narrative batch generation**: `generate_candidate_narratives` writes each candidate's summary in one batched LLM call for the whole shortlist — accepted self-consistency limit across candidates in the same batch (see `src/bot.py`).
- **`router_app.py` is explicitly labelled "test"**: single tool call per message, no multi-step chaining, no second LLM pass to compose a final natural-language answer over the tool result.
- Testing must force `provider="ollama"` to avoid real LLM API costs — mocks must be verified to actually intercept calls, never assume.

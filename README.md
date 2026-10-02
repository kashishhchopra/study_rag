# StudyMate — a Local RAG Study Assistant

## The real-world problem

Students revising for exams often have notes scattered across multiple
topics (ML basics, neural nets, NLP, computer vision) and waste time
re-reading entire files to find one specific fact ("what does L1
regularization do again?"). The knowledge exists in their own notes — it's
just slow to search. This app lets a student ask a plain question and get an
answer grounded in their actual notes, with the source section cited, instead
of re-skimming everything or risking a hallucinated answer from a generic
chatbot that isn't tied to their material.

## Requirements → implementation map

| Requirement          | Implementation |
|-----------------------|----------------|
| Document input         | `documents/*.txt` — 4 sample lecture-note files (ML fundamentals, neural networks, NLP, computer vision); PDF support + file-upload in the UI |
| Document chunking      | `chunking.py` — paragraph-aware splitter with configurable size/overlap |
| Embeddings              | `sentence-transformers` (`all-MiniLM-L6-v2`), local, no API key |
| Local vector store / FAISS | `faiss.IndexFlatIP` over normalized vectors (cosine similarity), saved to `faiss_index/` |
| Semantic retrieval      | `query.py: retrieve()` — top-K nearest chunks |
| LLM generation           | OpenAI chat API if `OPENAI_API_KEY` is set, else a local `flan-t5-base` model (fully offline fallback) |
| Strict RAG prompt        | `config.py: SYSTEM_PROMPT` — context-only answers, exact refusal string when the notes don't cover it, cites the relevant concept |
| Simple UI                 | `app.py` — Streamlit app: ask questions, upload your own notes, rebuild index, view sources |
| ≥ 5 test questions        | `test_questions.py` — 7 questions, including one deliberately outside the notes (general trivia) to test the refusal path |

## Architecture

```
documents/*.txt (lecture notes)
      │
      ▼
 chunking.py  ──►  overlapping text chunks
      │
      ▼
 ingest.py    ──►  sentence-transformers embeddings ──► FAISS index (faiss_index/)
                                                              │
Student's question ──► query.py: retrieve() ─ top-K chunks ◄─┘
      │
      ▼
 strict RAG prompt (context + question)
      │
      ▼
 LLM (OpenAI or local flan-t5)  ──►  answer + cited source notes
      │
      ▼
   app.py (Streamlit UI)  /  test_questions.py (CLI)
```

## Setup

```bash
pip install -r requirements.txt
```

(First run downloads the embedding model and, if no OpenAI key is set, the
local flan-t5 model — both from Hugging Face — so you'll need internet access
once.)

### 1. Build the index

```bash
python ingest.py
```

### 2. Ask a question from the command line

```bash
python query.py "What is the bias-variance tradeoff?"
```

### 3. Run the automated test questions

```bash
python test_questions.py
```

### 4. Run the UI

```bash
streamlit run app.py
```

Type a question, see the answer and which note file it came from, and drop
in your own `.txt` notes from the sidebar (then click "Rebuild index").

### Optional: use OpenAI instead of the local model

```bash
export OPENAI_API_KEY=sk-...
```

Without this, generation falls back to the local `flan-t5-base` model —
slower and lower quality, but works fully offline with no key.

## Design notes

- **Same core engine as a general-purpose RAG pipeline** — `chunking.py` and
  `query.py` are domain-agnostic; only the sample `documents/`, the system
  prompt tone, and the UI copy changed to fit the study-assistant use case.
  This demonstrates the pipeline generalizes across knowledge domains just by
  swapping the document folder.
- **Why the refusal test question matters:** asking something unrelated to
  any of the notes (e.g., a general trivia question) confirms the model
  isn't quietly falling back on its own training knowledge — a common
  failure mode in naive RAG implementations that undermines trust in the
  citations.
- **Extending it:** drop in your own `.txt` lecture notes, textbook chapter
  exports, or PDF slides and re-run `python ingest.py` (or use the sidebar
  controls) to build a study assistant for any subject.

## Limitations

- Local `flan-t5-base` generation quality is noticeably weaker than a hosted
  LLM — included as a no-API-key fallback, not the recommended path if you
  want high-quality explanations.
- Retrieval quality depends on chunk size/overlap tuning (`config.py`) for
  your specific notes; defaults are reasonable for prose-style lecture notes
  but may need adjustment for bullet-heavy slide exports.
- No spaced-repetition or quiz generation yet — this is pure Q&A retrieval,
  not a full study-planning tool.

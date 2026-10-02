"""
Query-time RAG pipeline: semantic retrieval + LLM generation.

generate_answer() is the single entry point used by both the CLI test
script and the Streamlit UI. It:
  1. Embeds the user's question with the same local embedding model used
     at ingestion time.
  2. Retrieves the top-K most similar chunks from the FAISS index.
  3. Builds a strict, context-only prompt.
  4. Calls an LLM to generate the answer:
       - OpenAI's chat API if OPENAI_API_KEY is set in the environment
         (better quality), or
       - a local HuggingFace flan-t5 model otherwise (fully offline,
         no API key needed).
  5. Returns the answer plus the source chunks used, for transparency.
"""

import os
import json
import functools

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL_NAME, TOP_K,
    OPENAI_MODEL, LOCAL_LLM_MODEL_NAME, SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Lazy-loaded singletons so the (relatively slow) models/index are only
# loaded once per process, not once per query.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@functools.lru_cache(maxsize=1)
def _get_index_and_metadata():
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(
            "No FAISS index found. Run `python ingest.py` first to build it."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


@functools.lru_cache(maxsize=1)
def _get_local_llm():
    """Fallback generator used when no OPENAI_API_KEY is set."""
    from transformers import pipeline
    return pipeline("text2text-generation", model=LOCAL_LLM_MODEL_NAME)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(question: str, top_k: int = TOP_K):
    """Return the top_k most relevant chunks for `question`, with scores."""
    embedder = _get_embedder()
    index, metadata = _get_index_and_metadata()

    q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk_meta = metadata[idx]
        results.append({
            "source": chunk_meta["source"],
            "chunk_id": chunk_meta["chunk_id"],
            "text": chunk_meta["text"],
            "score": float(score),
        })
    return results


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_prompt(question: str, chunks: list) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


def _generate_with_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from environment
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _generate_with_local_llm(prompt: str) -> str:
    llm = _get_local_llm()
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    output = llm(full_prompt, max_new_tokens=200, do_sample=False)
    return output[0]["generated_text"].strip()


def generate_answer(question: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline for a single question.
    Returns: {"question", "answer", "sources": [...]}
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "question": question,
            "answer": "I don't have enough information in the provided documents to answer that.",
            "sources": [],
        }

    prompt = _build_prompt(question, chunks)

    if os.environ.get("OPENAI_API_KEY"):
        answer = _generate_with_openai(prompt)
    else:
        answer = _generate_with_local_llm(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"source": c["source"], "score": round(c["score"], 3)} for c in chunks
        ],
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How many days of annual leave do I get?"
    result = generate_answer(q)
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print("Sources:", result["sources"])

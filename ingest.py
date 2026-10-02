"""
Ingestion pipeline.

1. Loads every .txt (and .pdf, if present) file from DOCS_DIR.
2. Splits each document into overlapping chunks.
3. Embeds every chunk with a local sentence-transformers model.
4. Builds a local FAISS index (cosine similarity via inner product on
   normalized vectors) and saves it to disk, along with chunk metadata
   (source file + chunk text) needed at query time.

Run directly to (re)build the index:
    python ingest.py
"""

import os
import json
import glob

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    DOCS_DIR, INDEX_DIR, INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL_NAME,
)
from chunking import chunk_text


def load_documents(docs_dir: str) -> list:
    """Return a list of {"source": filename, "text": full_text} dicts."""
    docs = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})

    # Optional: also ingest PDFs if any are dropped into documents/
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.pdf"))):
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append({"source": os.path.basename(path), "text": text})
        except ImportError:
            print(f"Skipping {path}: install `pypdf` to ingest PDF files.")
    return docs


def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print(f"Loading documents from {DOCS_DIR} ...")
    docs = load_documents(DOCS_DIR)
    if not docs:
        raise RuntimeError(f"No documents found in {DOCS_DIR}. Add .txt or .pdf files first.")
    print(f"Loaded {len(docs)} document(s).")

    # 1. Chunk every document
    all_chunks = []       # list of chunk text
    all_metadata = []     # list of {"source":..., "chunk_id":..., "text":...}
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk,
            })
    print(f"Created {len(all_chunks)} chunks.")

    # 2. Embed all chunks
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding chunks ...")
    embeddings = model.encode(
        all_chunks, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    # 3. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors (dim={dim}).")

    # 4. Persist index + metadata
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    build_index()

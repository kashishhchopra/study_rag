"""
Central configuration for the RAG application.
Change these values to tune chunking, retrieval, and model choice.
"""

import os

# --- Paths ---
DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

# --- Chunking ---
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks

# --- Embeddings ---
# Local, free, runs on CPU. Downloaded automatically on first run.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 4  # number of chunks to retrieve per query

# --- Generation ---
# If an OPENAI_API_KEY environment variable is set, the app uses OpenAI's
# chat completion API for generation (higher quality). Otherwise it falls
# back to a small local HuggingFace text2text model (flan-t5-base), which
# runs fully offline with no API key required.
OPENAI_MODEL = "gpt-4o-mini"
LOCAL_LLM_MODEL_NAME = "google/flan-t5-base"

# --- Strict RAG prompt ---
SYSTEM_PROMPT = (
    "You are a study assistant that helps a student review lecture notes on "
    "Machine Learning, Neural Networks, NLP, and Computer Vision. Answer "
    "ONLY using the context provided below, which is extracted from the "
    "student's own lecture notes.\n"
    "Rules:\n"
    "1. Only use facts stated in the context. Do not use outside knowledge, "
    "even if you know the answer from general training.\n"
    "2. If the answer is not contained in the context, respond exactly with: "
    "\"I don't have enough information in the provided notes to answer that.\"\n"
    "3. Be concise and precise, using the same terminology as the notes.\n"
    "4. When useful, briefly mention which concept/section the answer relates "
    "to, so the student can go back and re-read that part."
)

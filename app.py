"""
Simple Streamlit UI for the study assistant RAG app.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st

from config import DOCS_DIR, INDEX_PATH
from query import generate_answer

st.set_page_config(page_title="StudyMate — AI/ML Notes Assistant", page_icon="🧠")

st.title("🧠 StudyMate — AI/ML Lecture Notes Assistant")
st.caption(
    "Ask questions about your Machine Learning, Neural Networks, NLP, and "
    "Computer Vision lecture notes. Answers are generated only from the "
    "notes in `documents/` — great for quick review before an exam."
)

# --- Sidebar: document management + index rebuild ---
with st.sidebar:
    st.header("Your notes")
    existing = sorted(os.listdir(DOCS_DIR)) if os.path.isdir(DOCS_DIR) else []
    st.write(f"**{len(existing)} note file(s) loaded:**")
    for name in existing:
        st.write(f"- {name}")

    uploaded = st.file_uploader(
        "Add your own notes (.txt)", type=["txt"], accept_multiple_files=True
    )
    if uploaded:
        for file in uploaded:
            save_path = os.path.join(DOCS_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s). Click 'Rebuild index' below.")

    if st.button("🔄 Rebuild index"):
        with st.spinner("Chunking, embedding, and indexing notes..."):
            from ingest import build_index
            build_index()
        st.success("Index rebuilt.")

    if not os.path.exists(INDEX_PATH):
        st.warning("No index found yet. Click 'Rebuild index' to build one.")

# --- Main: question input ---
question = st.text_input("Ask a question about your notes:", "")
top_k = st.slider("Chunks to retrieve", min_value=1, max_value=8, value=4)

if st.button("Ask") and question.strip():
    if not os.path.exists(INDEX_PATH):
        st.error("No index found. Build one from the sidebar first.")
    else:
        with st.spinner("Retrieving relevant notes and generating an answer..."):
            try:
                result = generate_answer(question, top_k=top_k)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                result = None

        if result:
            st.subheader("Answer")
            st.write(result["answer"])

            with st.expander("📚 Sources used"):
                for s in result["sources"]:
                    st.write(f"- **{s['source']}** (similarity score: {s['score']})")

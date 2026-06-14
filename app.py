import os
from pathlib import Path
from typing import List

import streamlit as st

from src.config import APP_TITLE, APP_SUBTITLE, ALLOWED_EXTS, INDEX_DIR
from src.db import init_db, list_documents, list_topics
from src.ingest import ingest_files
from src.rag import answer_query

st.set_page_config(page_title=APP_TITLE, layout="wide")

# Initialize DB
init_db()

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Upload Materials")
    subject = st.text_input("Subject (e.g., DBMS)")
    doc_type = st.selectbox("Document Type", ["textbook", "lecture_notes", "lab_manual", "question_paper", "other"], index=0)
    year = st.number_input("Year (optional)", min_value=1990, max_value=2100, step=1, value=2024)
    tags = st.text_input("Tags (comma-separated)")

    uploaded = st.file_uploader(
        "Upload PDFs, DOCX, PPTX",
        type=[e[1:] for e in ALLOWED_EXTS],
        accept_multiple_files=True,
    )
    if st.button("Process & Index"):
        if not uploaded:
            st.error("Please upload at least one file.")
        else:
            uploads_dir = Path("data/uploads")
            uploads_dir.mkdir(parents=True, exist_ok=True)
            tmp_files: List[Path] = []
            for f in uploaded:
                tmp_path = uploads_dir / f.name
                with open(tmp_path, "wb") as out:
                    out.write(f.getbuffer())
                tmp_files.append(tmp_path)
            res = ingest_files(tmp_files, subject=subject.strip() or None, doc_type=doc_type, year=year, tags=tags.strip() or None)
            if res.get("ok"):
                st.success(f"Indexed {res['indexed']} chunks from {res['documents']} document(s).")
            else:
                st.error(res.get("error", "Unknown error during ingestion."))

st.divider()

# Uploaded Documents Section
st.subheader("📚 Uploaded Study Materials")

docs = list_documents()

if not docs:
    st.info("No documents uploaded yet. Upload materials from the sidebar.")
else:
    for d in docs:
        with st.expander(f"📄 {d['name']}"):
            st.write(f"**Type:** {d['doc_type']}")
            st.write(f"**Subject:** {d.get('subject') or 'N/A'}")
            st.write(f"**Year:** {d.get('year') or 'N/A'}")

st.divider()

# Main Query Interface
st.subheader("🔍 Ask Questions From Uploaded Materials")

mode = st.radio(
    "Mode",
    ["Topic Explanation", "Question Solving"],
    horizontal=True
)

query = st.text_area(
    "Ask a question",
    placeholder="""
Explain a topic from my notes
Summarize Chapter 3
Generate important exam questions
Create viva questions
Solve a previous year question
"""
)

if st.button("Get Answer"):
    if not (INDEX_DIR / "index.faiss").exists():
        st.error("No index found. Please upload and process documents first.")
    elif not query.strip():
        st.error("Please enter a query.")
    else:
        try:
            with st.spinner("Reasoning across your materials..."):
                res = answer_query(query)

            st.markdown("### Answer")
            st.write(res["answer"])

            st.markdown("### Sources")
            for s in res["sources"]:
                st.write(
                    f"- {s.get('name')} | page {s.get('page')} | "
                    f"{s.get('doc_type')} | {s.get('subject')} | {s.get('year')}"
                )

        except Exception as e:
            st.error(
                f"LLM error: {e}. Please configure "
                f"LLM_PROVIDER and API key in your environment."
            )

st.divider()

# Study Tools Section
st.subheader("📝 Study Tools")

c1, c2, c3 = st.columns(3)

with c1:
    st.button("Generate Notes")

with c2:
    st.button("Generate MCQs")

with c3:
    st.button("Generate Question Bank")
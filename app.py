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
cols = st.columns([1, 1, 2])

with cols[0]:
    st.subheader("Subject Dashboard")
    docs = list_documents()
    if not docs:
        st.info("No documents yet. Upload from the sidebar.")
    else:
        for d in docs[:10]:
            st.markdown(f"**{d['name']}** — {d['doc_type']} | {d.get('subject') or 'N/A'} | {d.get('year') or ''}")

with cols[1]:
    st.subheader("Topics (auto-extracted)")
    subject_filter = st.text_input("Filter subject", value=subject)
    topics = list_topics(subject=subject_filter.strip() or None)
    if topics:
        for t in topics[:20]:
            st.write(f"- {t['name']} ({t['score']:.2f}) — {t['subject']}")
    else:
        st.caption("Topics will appear after document processing.")

with cols[2]:
    st.subheader("Comprehensive Query Interface")
    mode = st.radio("Mode", ["Topic Explanation", "Question Solving"], horizontal=True)
    query = st.text_area("Ask a question", placeholder="Explain Database Normalization with examples OR How to solve Q3 from 2023 DBMS paper")
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
                st.write(res["answer"])  # Streamlit will render markdown
                st.markdown("### Sources")
                for s in res["sources"]:
                    st.write(f"- {s.get('name')} | page {s.get('page')} | {s.get('doc_type')} | {s.get('subject')} | {s.get('year')}")
            except Exception as e:
                st.error(f"LLM error: {e}. Please configure LLM_PROVIDER and API key in your environment.")

st.divider()
st.subheader("Content Browser & Export")
left, right = st.columns([2, 1])
with left:
    docs = list_documents()
    for d in docs:
        st.write(f"- {d['name']} — {d['doc_type']} | {d.get('subject') or ''} | {d.get('year') or ''}")
with right:
    st.caption("Export study guides (coming next)")
    st.button("Export Topic Summary (soon)")

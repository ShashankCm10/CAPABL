# Comprehensive Academic Learning Assistant

An AI-powered academic assistant that combines multi-source RAG across lecture notes, textbooks, lab manuals, and previous year question papers. Ask questions like “Explain Database Normalization with examples” or “How to solve Q3 from 2023 DBMS paper” and get comprehensive answers blending theory, examples, and practice guidance, with citations.

## Features
- Multi-document upload (PDF/DOCX/PPTX) and processing
- FAISS vector index for fast similarity search
- Topic extraction via spaCy for quick study organization
- RAG-based answers with source citations
- SQLite for content metadata and (extensible) progress tracking
- Streamlit academic dashboard and query interface

## Stack
- Frontend: Streamlit
- Backend: Python + LangChain + FAISS + spaCy
- Database: SQLite
- LLM: OpenAI GPT-4o-mini or Google Gemini (configurable)

## Setup

### 1) Clone and install dependencies

```bash
# From the project root
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure LLM provider

Choose one provider and set env vars:

```bash
# Option A: OpenAI
set LLM_PROVIDER=OPENAI
set OPENAI_API_KEY=your_openai_key

# Option B: Google Gemini
set LLM_PROVIDER=GEMINI
set GOOGLE_API_KEY=your_google_key
```

### 3) Prepare NLP resources

spaCy model (required for topic extraction):

```bash
python -m spacy download en_core_web_sm
```

Optionally (for better text preprocessing), NLTK stopwords:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 4) Run the app

```bash
streamlit run app.py
```

Upload your materials from the sidebar, set subject and document type (e.g., textbook, lecture_notes, lab_manual, question_paper), then click “Process & Index”. Use the query interface for explanations or question solving.

## Usage Tips
- For question papers, include subject and year in the upload for better filtering (e.g., DBMS, 2023).
- Use queries like: "Explain Database Normalization with examples" or "How to solve Q3 from 2023 DBMS paper".
- Answers include citations with document name and page where relevant content was retrieved.

## Roadmap (Extensible)
- Export study guides (Markdown/PDF) with topic-wise summaries
- Weak area identification and targeted recommendations
- Topic-wise practice question generation
- Progress tracking dashboard
- Hindi technical term recognition for bilingual support

## Troubleshooting
- No LLM configured: Ensure `LLM_PROVIDER` and the corresponding API key are set.
- spaCy error: Run `python -m spacy download en_core_web_sm`.
- Index not found: Upload and process files first.

## Notes on Indian Educational Context
The assistant is tuned for semester-based syllabi and common examination patterns. You can tag uploads with subject names like DBMS, OS, DSA, etc., and years for question papers, aiding targeted retrieval.

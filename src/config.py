import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "storage"
EXPORTS_DIR = BASE_DIR / "exports"
INDEX_DIR = DATA_DIR / "faiss_index"

load_dotenv(BASE_DIR / ".env")

# Ensure directories exist
for d in [DATA_DIR, DB_DIR, EXPORTS_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# LLM provider: OPENAI | GEMINI | OLLAMA
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OPENAI").upper()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Embeddings: default to local sentence-transformers for portability
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# SQLite DB path
SQLITE_PATH = DB_DIR / "academic_assistant.db"

# App configuration
APP_TITLE = "Comprehensive Academic Learning Assistant"
APP_SUBTITLE = "Multi-source RAG for study guides and question solutions"

# Allowed upload types
ALLOWED_EXTS = {".pdf", ".docx", ".pptx"}

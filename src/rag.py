from typing import Dict, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from .config import INDEX_DIR, EMBEDDING_MODEL, LLM_PROVIDER, OPENAI_API_KEY, GOOGLE_API_KEY

# LLMs
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


ANSWER_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "You are an academic assistant for Indian university curricula.\n"
        "Use the provided context from lecture notes, textbooks, lab manuals, and question papers to answer comprehensively.\n"
        "Combine theory, examples, and practice guidance. Show step-wise reasoning for solutions, but keep it concise.\n"
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer (include citations as [Doc: name | page]):"
    ),
)


def get_llm():
    provider = (LLM_PROVIDER or "OPENAI").upper()
    if provider == "OPENAI" and ChatOpenAI and OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    if provider == "GEMINI" and ChatGoogleGenerativeAI and GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    # Fallback: raise
    raise RuntimeError("No LLM configured. Set LLM_PROVIDER and API key.")


def load_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def build_context(docs: List[Document]) -> str:
    buf: List[str] = []
    for d in docs:
        meta = d.metadata
        cite = f"[Doc: {meta.get('name')} | page {meta.get('page')}]" if meta.get("name") else ""
        buf.append(f"{cite}\n{d.page_content}\n")
    return "\n".join(buf)


def parse_question_reference(q: str) -> Dict:
    q = q.lower()
    result: Dict[str, Optional[str]] = {"year": None, "number": None, "subject": None}
    # Year
    import re
    y = re.search(r"(20\d{2})", q)
    if y:
        result["year"] = y.group(1)
    # Question number
    n = re.search(r"q\s*([0-9]{1,2})", q)
    if not n:
        n = re.search(r"question\s*([0-9]{1,2})", q)
    if n:
        result["number"] = n.group(1)
    # Subject (simple heuristic)
    subj = re.search(r"dbms|database|os|dsa|math|physics|chemistry", q)
    if subj:
        result["subject"] = subj.group(0).upper()
    return result


def retrieve(query: str, filters: Optional[Dict] = None, k: int = 6) -> List[Document]:
    vs = load_vectorstore()
    docs = vs.similarity_search(query, k=k)
    if filters:
        def ok(meta: Dict) -> bool:
            for key, val in filters.items():
                if val is None:
                    continue
                if str(meta.get(key)).lower() != str(val).lower():
                    return False
            return True
        docs = [d for d in docs if ok(d.metadata)]
        if not docs:  # if filters too strict, fallback
            docs = vs.similarity_search(query, k=k)
    return docs


def answer_query(query: str) -> Dict:
    llm = get_llm()
    filters: Dict = {}
    fq = parse_question_reference(query)
    if fq.get("year"):
        filters["year"] = fq["year"]
    if fq.get("subject"):
        filters["subject"] = fq["subject"]
    docs = retrieve(query, filters=filters, k=8)
    context = build_context(docs)
    prompt = ANSWER_PROMPT.format(question=query, context=context)
    resp = llm.invoke(prompt)
    return {
        "answer": resp.content if hasattr(resp, "content") else str(resp),
        "sources": [d.metadata for d in docs],
    }

from typing import List, Tuple

import os
import spacy
from collections import Counter

from .db import record_topics

# Attempt to load spaCy model; instruct user to install if missing
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        # Lazy import/download suggestion (handled in README)
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
        )


def extract_topics(texts: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
    nlp = load_spacy()
    noun_chunks: List[str] = []
    for t in texts:
        doc = nlp(t)
        noun_chunks.extend([nc.text.lower().strip() for nc in doc.noun_chunks if len(nc.text) > 2])
    freq = Counter(noun_chunks)
    total = sum(freq.values()) or 1
    topics = [(term, count / total) for term, count in freq.most_common(top_n)]
    return topics


def process_and_store_topics(doc_id: int, texts: List[str], top_n: int = 20):
    topics = extract_topics(texts, top_n=top_n)
    record_topics(doc_id, topics)
    return topics

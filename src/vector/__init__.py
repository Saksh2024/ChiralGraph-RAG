"""
src/vector/__init__.py
"""
from .store import (
    LiteratureChunk,
    ScoredChunk,
    InMemoryVectorStore,
    get_vector_store,
    get_default_literature_corpus,
)

__all__ = [
    "LiteratureChunk",
    "ScoredChunk",
    "InMemoryVectorStore",
    "get_vector_store",
    "get_default_literature_corpus",
]

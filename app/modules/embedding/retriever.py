"""Query side of the vector store, used by the Conversation Info Advisor.

Per the workflow diagram the vector store is touched only after the Info
Advisor has decided "Info Needed" - retrieval is never unconditional, so a
turn that asks no question costs nothing.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ..config.settings import CHROMA_DIR, get_settings
from .build_index import COLLECTION


class JobDescriptionRetriever:
    def __init__(self, k: int = 3):
        settings = get_settings()
        self.k = k
        self._store = Chroma(
            collection_name=COLLECTION,
            embedding_function=OpenAIEmbeddings(
                model=settings.embed_model, api_key=settings.openai_api_key
            ),
            persist_directory=str(CHROMA_DIR),
        )

    def search(self, query: str, k: int | None = None) -> list[str]:
        docs = self._store.similarity_search(query, k=k or self.k)
        return [d.page_content for d in docs]

    def is_ready(self) -> bool:
        try:
            return bool(self._store.get().get("ids"))
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_retriever(k: int = 3) -> JobDescriptionRetriever:
    """Cached so the Chroma client and embedding function are built once."""
    return JobDescriptionRetriever(k=k)

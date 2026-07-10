"""Retrieval layer for diabetes RAG using Chroma MMR with fallback keyword scoring."""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Model embedding HF (env HF_EMBED_MODEL, default all-MiniLM). WAJIB sama dengan yang dipakai
# saat ingest (knowledge_base.py) agar embedding query & dokumen sebanding.
_HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "all-MiniLM-L6-v2")


def _build_embeddings(
    embed_provider: str,
    ollama_base_url: str = "http://localhost:11434",
    embed_model: str = "nomic-embed-text",
) -> Any:
    """Return a LangChain-compatible embedding object for the requested provider."""
    if embed_provider == "sentence-transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=_HF_EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    if embed_provider == "google":
        import os

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    # Default: Ollama
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=embed_model, base_url=ollama_base_url)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class SimpleKeywordRetriever:
    """Fallback retriever used when vector infrastructure is unavailable."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_tokens = Counter(_tokenize(query))
        results: List[Dict[str, Any]] = []

        for chunk in self.chunks:
            meta = dict(chunk.get("metadata", {}))
            if metadata_filter and not _metadata_matches(meta, metadata_filter):
                continue

            text = chunk.get("text", "")
            chunk_tokens = Counter(_tokenize(text))
            overlap = sum(min(query_tokens[token], chunk_tokens[token]) for token in query_tokens)
            similarity = overlap / max(len(query_tokens), 1)
            results.append(
                {
                    "rank": 0,
                    "text": text,
                    "source": chunk.get("source", "manual_kb"),
                    "similarity": float(similarity),
                    "metadata": meta,
                }
            )

        results.sort(key=lambda item: item["similarity"], reverse=True)
        for idx, row in enumerate(results[:top_k], start=1):
            row["rank"] = idx
        return results[:top_k]


class MMRRetriever:
    """ChromaDB retriever using MMR search strategy with configurable embeddings.

    Supports three embedding providers:
    - ``"sentence-transformers"`` — CPU-only, no server needed (default, recommended)
    - ``"google"``               — Google Generative AI embeddings (requires GOOGLE_API_KEY)
    - ``"ollama"``               — local Ollama server (legacy)
    """

    def __init__(
        self,
        persist_dir: str = "models/chroma_db",
        collection_name: str = "diabetes_kb",
        embed_provider: str = "sentence-transformers",
        ollama_base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embed_provider = embed_provider
        self.ollama_base_url = ollama_base_url
        self.embed_model = embed_model

        self._vector_store = None
        self._init_error: Optional[str] = None

        try:
            from langchain_chroma import Chroma

            embeddings = _build_embeddings(embed_provider, ollama_base_url, embed_model)
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_dir,
            )
            logger.info(
                "MMRRetriever connected — collection=%s embed=%s",
                self.collection_name,
                embed_provider,
            )
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning(
                "MMRRetriever unavailable (embed=%s), keyword fallback required: %s",
                embed_provider,
                exc,
            )

    @property
    def is_ready(self) -> bool:
        return self._vector_store is not None

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._vector_store:
            raise RuntimeError(f"MMR retriever is not ready: {self._init_error}")

        kwargs: Dict[str, Any] = {
            "k": top_k,
            "fetch_k": max(top_k * 3, 12),
            "lambda_mult": 0.5,
        }
        if metadata_filter:
            kwargs["filter"] = metadata_filter

        retriever = self._vector_store.as_retriever(search_type="mmr", search_kwargs=kwargs)
        docs = retriever.invoke(query)

        results: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            metadata = dict(doc.metadata or {})
            results.append(
                {
                    "rank": idx,
                    "text": doc.page_content,
                    "source": metadata.get("source", "manual_kb"),
                    "similarity": 0.0,
                    "metadata": metadata,
                }
            )

        return results

    def retrieve_with_context(
        self,
        query: str,
        patient_state: Dict[str, Any],
        top_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        enhanced_query = self._enhance_query(query, patient_state)
        return self.retrieve(query=enhanced_query, top_k=top_k, metadata_filter=metadata_filter)

    def _enhance_query(self, query: str, patient_state: Dict[str, Any]) -> str:
        tags: List[str] = []
        glucose = float(patient_state.get("current_glucose", 0.0))
        stress = int(patient_state.get("stress_level", 0))

        if glucose < 70:
            tags.append("hipoglikemia")
        elif glucose > 180:
            tags.append("hiperglikemia")

        if stress >= 7:
            tags.append("stress tinggi")

        if not tags:
            return query
        return f"{query}. Konteks pasien: {', '.join(tags)}."


def _metadata_matches(metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
    for key, expected in metadata_filter.items():
        value = metadata.get(key)
        if isinstance(value, list):
            if expected not in value:
                return False
        else:
            if value != expected:
                return False
    return True


# Backward compatibility for existing imports.
DocumentRetriever = MMRRetriever

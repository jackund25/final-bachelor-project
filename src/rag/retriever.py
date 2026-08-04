"""Retrieval layer for diabetes RAG using Chroma MMR with fallback keyword scoring."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Pembangun embedding dipakai bersama knowledge_base.py — memakai SATU implementasi
# menjamin model embedding kueri dan dokumen tidak mungkin menyimpang. Bila keduanya
# berbeda, retrieval merosot menjadi derau tanpa error apa pun.
from .knowledge_base import _build_embeddings  # noqa: E402


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class SimpleKeywordRetriever:
    """Fallback retriever used when vector infrastructure is unavailable."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if top_k is None:
            from src.config import load_rag_config
            top_k = load_rag_config().top_k
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
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embed_provider: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        embed_model: Optional[str] = None,
        top_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        config: Optional[Any] = None,
    ):
        from src.config import load_rag_config

        cfg = config or load_rag_config()
        self.cfg = cfg

        self.persist_dir = persist_dir or cfg.persist_dir
        self.collection_name = collection_name or cfg.collection_name
        self.embed_provider = embed_provider or cfg.embedding_provider
        self.ollama_base_url = ollama_base_url or cfg.ollama_base_url
        self.embed_model = embed_model or cfg.ollama_embed_model
        self.top_k = top_k if top_k is not None else cfg.top_k
        self.fetch_k = fetch_k if fetch_k is not None else cfg.fetch_k
        self.lambda_mult = lambda_mult if lambda_mult is not None else cfg.lambda_mult

        self._vector_store = None
        self._embeddings = None
        self._init_error: Optional[str] = None

        try:
            from langchain_chroma import Chroma

            embeddings = _build_embeddings(
                self.embed_provider, self.ollama_base_url, self.embed_model,
                hf_model=cfg.embedding_model, google_model=cfg.google_embedding_model,
            )
            self._embeddings = embeddings
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

    @staticmethod
    def _doc_key(doc: Any) -> str:
        """Kunci identitas dokumen untuk menggabungkan hasil MMR dengan skornya."""
        meta = dict(getattr(doc, "metadata", None) or {})
        chunk_id = meta.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        return f"__text__{hash(getattr(doc, 'page_content', ''))}"

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Ambil dokumen dengan MMR, sertai skor relevansi kueri-terhadap-chunk.

        MMR memilih k dokumen dari kolam kandidat fetch_k yang sama dengan yang
        dienumerasi kueri penilaian, sehingga setiap dokumen terpilih dijamin ada
        di dalam kolam skor. Vektor kueri dihitung SEKALI dan dipakai ulang oleh
        kedua pemanggilan, jadi tidak ada biaya embedding tambahan.

        CATATAN: skor yang dikembalikan adalah relevansi kueri-terhadap-chunk,
        BUKAN objektif MMR (yang mengurangi penalti redundansi). Karena itu urutan
        rank tidak selalu menurun monoton terhadap skor.
        """
        if not self._vector_store:
            raise RuntimeError(f"MMR retriever is not ready: {self._init_error}")

        top_k = top_k if top_k is not None else self.top_k
        fetch_k = self.fetch_k
        lambda_mult = self.lambda_mult

        kwargs: Dict[str, Any] = {"k": top_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
        if metadata_filter:
            kwargs["filter"] = metadata_filter

        score_by_key: Dict[str, float] = {}
        docs = None

        if self._embeddings is not None:
            try:
                qvec = self._embeddings.embed_query(query)
                pool = self._vector_store.similarity_search_by_vector_with_relevance_scores(
                    embedding=qvec, k=fetch_k, filter=metadata_filter
                )
                score_by_key = {self._doc_key(d): float(s) for d, s in pool}
                docs = self._vector_store.max_marginal_relevance_search_by_vector(
                    embedding=qvec, k=top_k, fetch_k=fetch_k,
                    lambda_mult=lambda_mult, filter=metadata_filter,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Penilaian skor gagal, kembali ke MMR tanpa skor: %s", exc)
                docs = None

        if docs is None:
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
                    "similarity": score_by_key.get(self._doc_key(doc)),
                    "metadata": metadata,
                }
            )

        return results

    def retrieve_with_context(
        self,
        query: str,
        patient_state: Dict[str, Any],
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        enhanced_query = self._enhance_query(query, patient_state)
        return self.retrieve(query=enhanced_query, top_k=top_k, metadata_filter=metadata_filter)

    def _enhance_query(self, query: str, patient_state: Dict[str, Any]) -> str:
        from src.constants import CLASS_NORMAL, classify_glucose_3class

        tags: List[str] = []
        glucose = float(patient_state.get("current_glucose", 0.0))
        stress = int(patient_state.get("stress_level", 0))

        kondisi = classify_glucose_3class(glucose)
        if kondisi != CLASS_NORMAL:
            tags.append(kondisi)

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

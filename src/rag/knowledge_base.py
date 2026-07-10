"""Knowledge base management for the diabetes RAG pipeline."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 900
_DEFAULT_CHUNK_OVERLAP = 120

# Model embedding HF. Default all-MiniLM (Inggris, ~80MB CPU). Override via env HF_EMBED_MODEL
# untuk model multilingual (mis. paraphrase-multilingual-MiniLM-L12-v2). WAJIB sama antara
# ingest & query — knowledge_base & retriever membaca env yang sama agar konsisten.
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


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Ubah metadata agar kompatibel ChromaDB (hanya skalar str/int/float/bool/None).

    list/tuple → gabung jadi string; dict/lainnya → str(); None/skalar dipertahankan.
    """
    clean: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = ", ".join(str(v) for v in value)
        else:
            clean[key] = str(value)
    return clean


class MedicalKnowledgeBase:
    """Load, chunk, and persist medical knowledge for retrieval."""

    def __init__(
        self,
        kb_dir: str = "data/knowledge_base",
        persist_dir: str = "models/chroma_db",
        collection_name: str = "diabetes_kb",
        embed_provider: str = "sentence-transformers",
        ollama_base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
    ):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.embed_provider = embed_provider
        self.ollama_base_url = ollama_base_url
        self.embed_model = embed_model

        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []

    def load_manual_kb(self, file_name: str = "manual_kb.json") -> List[Dict[str, Any]]:
        """Load manual knowledge entries from JSON file in kb directory."""
        manual_path = self.kb_dir / file_name
        if not manual_path.exists():
            # Fallback to repository default path for convenience.
            repo_default = Path("data/knowledge_base") / file_name
            if repo_default.exists():
                manual_path = repo_default
            else:
                logger.warning("Manual KB file not found: %s", manual_path)
                return []

        with manual_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        docs: List[Dict[str, Any]] = []
        for idx, item in enumerate(payload):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            metadata = self._build_metadata(
                source=item.get("source", "manual_kb"),
                topic=item.get("topic", f"topic_{idx}"),
                doc_id=item.get("doc_id", f"manual_doc_{idx}"),
                index=idx,
            )
            docs.append(
                {
                    "text": text,
                    "source": item.get("source", "manual_kb"),
                    "topic": item.get("topic", f"topic_{idx}"),
                    "metadata": metadata,
                }
            )

        self.documents = docs
        logger.info("Loaded %d manual KB entries from %s", len(docs), manual_path)
        return docs

    def load_documents(self, file_pattern: str = "*.txt") -> None:
        """Compatibility helper to load text files from knowledge base directory."""
        files = list(self.kb_dir.glob(file_pattern))
        loaded_docs: List[Dict[str, Any]] = []

        for idx, path in enumerate(files):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            metadata = self._build_metadata(
                source=path.name,
                topic=path.stem,
                doc_id=f"txt_{path.stem}",
                index=idx,
            )
            loaded_docs.append(
                {
                    "text": text,
                    "source": path.name,
                    "topic": path.stem,
                    "metadata": metadata,
                }
            )

        self.documents = loaded_docs
        logger.info("Loaded %d text documents from %s", len(loaded_docs), self.kb_dir)

    def chunk_documents(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> List[Dict[str, Any]]:
        """Chunk documents using LangChain splitter with a safe fallback splitter."""
        docs = documents if documents is not None else self.documents
        if not docs:
            logger.warning("No documents available to chunk")
            self.chunks = []
            return []

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
            )

            chunk_rows: List[Dict[str, Any]] = []
            for doc_idx, doc in enumerate(docs):
                pieces = splitter.split_text(doc["text"])
                for part_idx, part in enumerate(pieces):
                    metadata = dict(doc.get("metadata", {}))
                    metadata.update(
                        {
                            "chunk_index": part_idx,
                            "chunk_id": f"{metadata.get('doc_id', f'doc_{doc_idx}')}_ch_{part_idx:03d}",
                        }
                    )
                    chunk_rows.append(
                        {
                            "text": part,
                            "source": doc.get("source", "manual_kb"),
                            "topic": doc.get("topic", "general"),
                            "metadata": metadata,
                        }
                    )

            self.chunks = chunk_rows
            logger.info("Chunked %d docs into %d chunks", len(docs), len(chunk_rows))
            return chunk_rows

        except Exception as exc:
            logger.warning("LangChain splitter unavailable, fallback splitter active: %s", exc)
            chunk_rows = self._fallback_chunk_documents(docs, chunk_size=chunk_size, overlap=chunk_overlap)
            self.chunks = chunk_rows
            return chunk_rows

    def _fallback_chunk_documents(
        self,
        docs: List[Dict[str, Any]],
        chunk_size: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        """Simple token-window chunking fallback when text splitter is unavailable."""
        chunks: List[Dict[str, Any]] = []
        step = max(chunk_size - overlap, 1)

        for doc_idx, doc in enumerate(docs):
            words = doc["text"].split()
            for start in range(0, len(words), step):
                part = " ".join(words[start : start + chunk_size]).strip()
                if not part:
                    continue
                metadata = dict(doc.get("metadata", {}))
                chunk_number = len(chunks)
                metadata.update(
                    {
                        "chunk_index": chunk_number,
                        "chunk_id": f"{metadata.get('doc_id', f'doc_{doc_idx}')}_ch_{chunk_number:03d}",
                    }
                )
                chunks.append(
                    {
                        "text": part,
                        "source": doc.get("source", "manual_kb"),
                        "topic": doc.get("topic", "general"),
                        "metadata": metadata,
                    }
                )

        logger.info("Fallback chunker created %d chunks", len(chunks))
        return chunks

    def _build_metadata(self, source: str, topic: str, doc_id: str, index: int) -> Dict[str, Any]:
        """Build normalized metadata fields for each document/chunk."""
        return {
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_ch_000",
            "domain": "manual",
            "subdomain": topic.lower().replace(" ", "_"),
            "sumber": "Manual KB",
            "judul": topic,
            "tahun": 2024,
            "versi": "v1",
            "url_sumber": "",
            "halaman": "N/A",
            "jenis_dm": ["dm_tipe2"],
            "setting": ["fktp", "fkrtl"],
            "sasaran": ["dokter_umum", "sppd"],
                "populasi_khusus": ["umum"],
            "tipe_konten": "panduan_klinis",
            "bahasa": "id",
            "level_bukti": "guideline",
            "topik_terkait": [topic],
            "perlu_update_sebelum": "2026-12",
            "status": "aktif",
            "source": source,
            "created_at": datetime.utcnow().isoformat(),
            "index": index,
        }

    def save_to_chroma(self, chunks: Optional[List[Dict[str, Any]]] = None, reset_collection: bool = False) -> bool:
        """Persist chunks into ChromaDB using the configured embedding provider."""
        chunk_rows = chunks if chunks is not None else self.chunks
        if not chunk_rows:
            logger.warning("No chunks available for Chroma ingestion")
            return False

        try:
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
        except Exception as exc:
            logger.error("Chroma dependencies unavailable: %s", exc)
            return False

        try:
            embeddings = _build_embeddings(
                self.embed_provider, self.ollama_base_url, self.embed_model
            )
        except Exception as exc:
            logger.error("Embedding initialisation failed (%s): %s", self.embed_provider, exc)
            return False

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_dir),
        )

        if reset_collection:
            try:
                vector_store.delete_collection()
                vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=embeddings,
                    persist_directory=str(self.persist_dir),
                )
            except Exception as exc:
                logger.warning("Could not reset existing collection: %s", exc)

        documents: List[Document] = []
        for row in chunk_rows:
            raw_meta = {
                **dict(row.get("metadata", {})),
                "source": row.get("source", "manual_kb"),
                "topic": row.get("topic", "general"),
            }
            # ChromaDB hanya menerima metadata skalar (str/int/float/bool/None);
            # list/dict di-flatten ke string agar tidak ditolak saat upsert.
            documents.append(
                Document(page_content=row["text"], metadata=_sanitize_metadata(raw_meta))
            )

        vector_store.add_documents(documents)
        logger.info("Saved %d chunks to ChromaDB at %s", len(documents), self.persist_dir)
        return True

    def save_chunks(self, output_path: Optional[Path] = None) -> None:
        """Save prepared chunks into JSON for traceability and tests."""
        target = output_path or (self.kb_dir / "chunks.json")
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.chunks, handle, ensure_ascii=False, indent=2)

    def load_chunks(self, filepath: Optional[Path] = None) -> None:
        """Load precomputed chunks from JSON file."""
        source_path = filepath or (self.kb_dir / "chunks.json")
        with source_path.open("r", encoding="utf-8") as handle:
            self.chunks = json.load(handle)

    def process_all_documents(self, chunk_size: int = _DEFAULT_CHUNK_SIZE, overlap: int = _DEFAULT_CHUNK_OVERLAP) -> None:
        """Compatibility wrapper used by existing code paths."""
        self.chunk_documents(documents=self.documents, chunk_size=chunk_size, chunk_overlap=overlap)

    def create_manual_kb(self) -> None:
        """Compatibility helper expected by existing tests and pipeline calls."""
        docs = self.load_manual_kb("manual_kb.json")
        if not docs:
            docs = [
                {
                    "text": "Hiperglikemia adalah kondisi glukosa darah di atas 180 mg/dL dan perlu pemantauan.",
                    "source": "manual_kb",
                    "topic": "Hiperglikemia",
                    "metadata": self._build_metadata("manual_kb", "Hiperglikemia", "manual_doc_0", 0),
                },
                {
                    "text": "Hipoglikemia adalah kondisi glukosa darah di bawah 70 mg/dL dan perlu tatalaksana segera.",
                    "source": "manual_kb",
                    "topic": "Hipoglikemia",
                    "metadata": self._build_metadata("manual_kb", "Hipoglikemia", "manual_doc_1", 1),
                },
            ]
            self.documents = docs

        self.chunk_documents(documents=docs, chunk_size=350, chunk_overlap=40)
        self.save_chunks(self.kb_dir / "manual_kb.json")

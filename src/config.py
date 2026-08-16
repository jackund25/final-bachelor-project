"""Pemuat konfigurasi bersama — config.yaml menjadi otoritatif.

Sebelum Tugas 4, sebagian besar blok `rag:` di config.yaml TIDAK PERNAH dibaca kode.
Nilai efektifnya datang dari konstanta hardcoded, sehingga config menyesatkan:
`temperature: 0.2` tertulis di config sementara `0.1` yang benar-benar dipakai, dan
`max_tokens: 300` dibuang dengan `del` tanpa pernah diteruskan ke LLM.

URUTAN PRESEDENSI (satu aturan, berlaku di mana pun):

    argumen konstruktor eksplisit  >  variabel environment  >  config.yaml  >  default kode

Disertai kebijakan: JANGAN membuat variabel environment baru untuk tombol perilaku
(chunk_size, chunk_overlap, top_k, fetch_k, lambda_mult, temperature, max_tokens,
embedding_model). Karena env semacam itu tidak ada, config.yaml menjadi otoritatif
de facto untuk parameter tersebut, sementara env tetap otoritatif untuk pengaturan
per-mesin dan rahasia yang memang sudah memakainya (GOOGLE_API_KEY, LLM_PROVIDER,
EMBED_PROVIDER, GEMINI_MODEL).

Bila env dan config berbeda untuk parameter yang sama, resolve() menuliskan
peringatan — kalau tidak, laporan bisa mendokumentasikan model yang tidak dijalankan.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Diresolusi relatif terhadap __file__, BUKAN cwd — kalau tidak, pytest (yang rootdir-nya
# berubah-ubah) dan Streamlit yang diluncurkan dari direktori lain akan diam-diam jatuh
# ke nilai default alih-alih membaca config.yaml.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

_SENTINEL = object()


@lru_cache(maxsize=None)
def _load_config_cached(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        logger.warning("config.yaml tidak ditemukan di %s — memakai default kode.", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Muat config.yaml (di-cache). Path diresolusi relatif terhadap akar proyek."""
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return _load_config_cached(str(resolved))


def cfg_get(dotted: str, default: Any = None, *, path: str | Path | None = None) -> Any:
    """Ambil nilai dengan kunci bertitik, mis. cfg_get("rag.llm.temperature", 0.2)."""
    node: Any = load_config(path)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node if node is not None else default


def resolve(
    name: str,
    ctor_value: Optional[T],
    env_var: Optional[str],
    cfg_key: Optional[str],
    default: T,
    *,
    cast: Any = None,
    path: str | Path | None = None,
) -> T:
    """Terapkan urutan presedensi tunggal dan catat lapisan mana yang menang."""
    cfg_value = cfg_get(cfg_key, _SENTINEL, path=path) if cfg_key else _SENTINEL
    env_value = os.getenv(env_var) if env_var else None

    if env_value is not None and cfg_value is not _SENTINEL and str(cfg_value) != str(env_value):
        logger.warning(
            "%s: env %s=%r menimpa config %s=%r. "
            "Pastikan laporan mendokumentasikan nilai yang benar-benar dijalankan.",
            name, env_var, env_value, cfg_key, cfg_value,
        )

    if ctor_value is not None:
        return ctor_value
    if env_value is not None:
        return cast(env_value) if cast else env_value  # type: ignore[return-value]
    if cfg_value is not _SENTINEL:
        return cast(cfg_value) if cast else cfg_value  # type: ignore[return-value]
    return default


# ──────────────────────────────────────────────────────────────
# Konfigurasi RAG
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RagConfig:
    """Seluruh tombol RAG dalam satu objek.

    Meneruskan satu dataclass jauh lebih baik daripada meneruskan sembilan skalar:
    menambah tombol ke-sepuluh nanti hanya menyentuh berkas ini.
    """

    knowledge_base_dir: str
    persist_dir: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    # Potongan berhenti di batas kalimat, bukan di sembarang kata (Gao dkk. 2023
    # Bagian V.A.1). Hanya berpengaruh saat INDEXING; mengubahnya menuntut indeks ulang.
    pemisah_kalimat: bool
    manual_chunk_size: int
    manual_chunk_overlap: int
    embedding_provider: str
    embedding_model: str
    # 0 = pakai bawaan model (256 untuk all-MiniLM-L6-v2). Lihat _build_embeddings.
    embedding_max_seq_length: int
    google_embedding_model: str
    ollama_embed_model: str
    ollama_base_url: str
    top_k: int
    fetch_k: int
    lambda_mult: float
    llm_provider: str
    llm_model: str
    ollama_llm_model: str
    temperature: float
    max_tokens: int


@lru_cache(maxsize=None)
def _load_rag_config_cached(path_str: Optional[str]) -> RagConfig:
    path = path_str
    return RagConfig(
        knowledge_base_dir=resolve(
            "knowledge_base_dir", None, None, "rag.knowledge_base_dir",
            "data/knowledge_base", cast=str, path=path),
        persist_dir=resolve(
            "persist_dir", None, "CHROMA_PERSIST_DIR", "rag.persist_dir",
            "models/chroma_db", cast=str, path=path),
        collection_name=resolve(
            "collection_name", None, "CHROMA_COLLECTION_NAME", "rag.collection_name",
            "diabetes_kb", cast=str, path=path),
        chunk_size=resolve(
            "chunk_size", None, None, "rag.chunk_size", 900, cast=int, path=path),
        chunk_overlap=resolve(
            "chunk_overlap", None, None, "rag.chunk_overlap", 120, cast=int, path=path),
        # Default False = perilaku sebelum T7, sehingga membaca config lama tidak
        # diam-diam mengubah cara korpus dipecah.
        pemisah_kalimat=resolve(
            "pemisah_kalimat", None, "PEMISAH_KALIMAT", "rag.pemisah_kalimat",
            False, cast=bool, path=path),
        # Entri manual_kb adalah prosa pendek, bukan halaman buku — sengaja memakai
        # potongan yang lebih kecil. Bila diarahkan ke rag.chunk_size (900), jumlah
        # chunk-nya anjlok dan tests/test_kb.py gagal.
        manual_chunk_size=resolve(
            "manual_chunk_size", None, None, "rag.manual_kb.chunk_size", 350, cast=int, path=path),
        manual_chunk_overlap=resolve(
            "manual_chunk_overlap", None, None, "rag.manual_kb.chunk_overlap", 40, cast=int, path=path),
        embedding_provider=resolve(
            "embedding_provider", None, "EMBED_PROVIDER", "rag.embedding_provider",
            "sentence-transformers", cast=str, path=path),
        embedding_model=resolve(
            "embedding_model", None, "HF_EMBED_MODEL", "rag.embedding_model",
            "all-MiniLM-L6-v2", cast=str, path=path),
        # 0 = ikut bawaan model. Menaikkannya HARUS diikuti indeks ulang: vektor
        # lama dibuat dengan jendela lain dan tidak sebanding.
        embedding_max_seq_length=resolve(
            "embedding_max_seq_length", None, "EMBED_MAX_SEQ_LENGTH",
            "rag.embedding_max_seq_length", 0, cast=int, path=path),
        # CATATAN: nilai di config.yaml ("models/embedding-001") sudah TIDAK ADA lagi
        # pada API Google per 16 Agustus 2026 — diperiksa lewat models.list dan tidak
        # muncul di antara 53 model yang terlihat. Jalur google karena itu akan gagal
        # apa adanya. Slot environment ditambahkan agar percobaan dapat menimpanya
        # tanpa menyentuh config.yaml; penggantian nilai di config.yaml sendiri
        # menunggu keputusan (kandidat: models/gemini-embedding-2, jendela 8192).
        google_embedding_model=resolve(
            "google_embedding_model", None, "GOOGLE_EMBED_MODEL",
            "rag.google_embedding_model", "models/embedding-001", cast=str, path=path),
        ollama_embed_model=resolve(
            "ollama_embed_model", None, "OLLAMA_EMBED_MODEL", "rag.ollama.embed_model",
            "nomic-embed-text", cast=str, path=path),
        ollama_base_url=resolve(
            "ollama_base_url", None, "OLLAMA_BASE_URL", "rag.ollama.base_url",
            "http://localhost:11434", cast=str, path=path),
        # Nama kunci top_k_retrieval DIPERTAHANKAN karena benchmark_deployability.py
        # membacanya langsung dari config.
        top_k=resolve(
            "top_k", None, None, "rag.top_k_retrieval", 4, cast=int, path=path),
        fetch_k=resolve(
            "fetch_k", None, None, "rag.fetch_k", 12, cast=int, path=path),
        lambda_mult=resolve(
            "lambda_mult", None, None, "rag.lambda_mult", 0.5, cast=float, path=path),
        llm_provider=resolve(
            "llm_provider", None, "LLM_PROVIDER", "rag.llm.provider",
            "gemini", cast=str, path=path),
        llm_model=resolve(
            "llm_model", None, "GEMINI_MODEL", "rag.llm.model",
            "gemini-2.5-flash-lite", cast=str, path=path),
        ollama_llm_model=resolve(
            "ollama_llm_model", None, "OLLAMA_LLM_MODEL", "rag.llm.ollama_model",
            "llama3.1:8b", cast=str, path=path),
        temperature=resolve(
            "temperature", None, None, "rag.llm.temperature", 0.2, cast=float, path=path),
        max_tokens=resolve(
            "max_tokens", None, None, "rag.llm.max_tokens", 700, cast=int, path=path),
    )


def load_rag_config(path: str | Path | None = None) -> RagConfig:
    """Muat konfigurasi RAG dengan presedensi ctor > env > config.yaml > default."""
    return _load_rag_config_cached(str(path) if path is not None else None)


def clear_cache() -> None:
    """Kosongkan cache config — dipakai tes yang menulis config sementara."""
    _load_config_cached.cache_clear()
    _load_rag_config_cached.cache_clear()

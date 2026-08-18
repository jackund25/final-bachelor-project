"""config.yaml harus benar-benar dibaca, dan presedensinya harus dapat diprediksi."""

import os
from pathlib import Path

import pytest
import yaml

from src.config import DEFAULT_CONFIG_PATH, cfg_get, clear_cache, load_config, load_rag_config, resolve


@pytest.fixture(autouse=True)
def _bersihkan_cache():
    clear_cache()
    yield
    clear_cache()


def test_config_resolved_relative_to_project_not_cwd(tmp_path, monkeypatch):
    """Berjalan dari direktori lain tidak boleh diam-diam jatuh ke nilai default."""
    monkeypatch.chdir(tmp_path)
    clear_cache()
    cfg = load_config()
    assert cfg, "config.yaml harus tetap ditemukan dari cwd mana pun"
    # Nilainya dibaca dari config.yaml, bukan ditulis tetap di sini. Yang diuji adalah
    # bahwa berkasnya tetap ditemukan dari cwd mana pun, sehingga uji ini tidak boleh
    # ikut gagal setiap kali sebuah parameter penyetelan diubah.
    assert cfg_get("rag.chunk_size") == cfg["rag"]["chunk_size"]
    assert isinstance(cfg_get("rag.chunk_size"), int)


def test_default_config_path_points_at_repo_config():
    assert DEFAULT_CONFIG_PATH.name == "config.yaml"
    assert DEFAULT_CONFIG_PATH.exists()


def test_rag_config_values_come_from_yaml():
    """Nilai efektif harus sama dengan yang tertulis di config.yaml."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    rag = raw["rag"]
    cfg = load_rag_config()

    assert cfg.chunk_size == rag["chunk_size"]
    assert cfg.chunk_overlap == rag["chunk_overlap"]
    assert cfg.top_k == rag["top_k_retrieval"]
    assert cfg.fetch_k == rag["fetch_k"]
    assert cfg.lambda_mult == rag["lambda_mult"]
    assert cfg.temperature == rag["llm"]["temperature"]
    assert cfg.max_tokens == rag["llm"]["max_tokens"]
    assert cfg.embedding_model == rag["embedding_model"]
    assert cfg.manual_chunk_size == rag["manual_kb"]["chunk_size"]


def test_temperature_matches_config_not_old_hardcoded_value():
    """Regresi: temperature dulu 0.1 hardcoded padahal config menulis 0.2."""
    cfg = load_rag_config()
    assert cfg.temperature == 0.2
    assert cfg.temperature != 0.1


def test_max_tokens_is_applied_not_discarded():
    """Regresi: max_tokens dulu diterima lalu langsung dibuang dengan `del`."""
    cfg = load_rag_config()
    assert cfg.max_tokens >= 300
    from src.rag.generator import RAGGenerator
    import inspect

    sig = inspect.signature(RAGGenerator.generate_explanation)
    assert "temperature" not in sig.parameters
    assert "max_tokens" not in sig.parameters


def test_precedence_ctor_beats_env_beats_config(monkeypatch):
    monkeypatch.setenv("UJI_PRESEDENSI", "dari_env")

    # ctor menang atas semuanya
    assert resolve("x", "dari_ctor", "UJI_PRESEDENSI", "rag.chunk_size", "default") == "dari_ctor"
    # env menang atas config
    assert resolve("x", None, "UJI_PRESEDENSI", "rag.chunk_size", "default") == "dari_env"
    # config menang atas default
    monkeypatch.delenv("UJI_PRESEDENSI")
    dari_berkas = load_config()["rag"]["chunk_size"]
    assert resolve("x", None, "UJI_PRESEDENSI", "rag.chunk_size", 1, cast=int) == dari_berkas
    assert dari_berkas != 1, "nilai default tidak boleh sama dengan nilai config"
    # default dipakai bila kunci tidak ada
    assert resolve("x", None, None, "rag.tidak_ada", "default") == "default"


def test_env_overriding_config_emits_warning(monkeypatch, caplog):
    """Selisih env-vs-config harus terlihat, bukan senyap.

    Kalau tidak, laporan bisa mendokumentasikan model yang tidak dijalankan.
    """
    monkeypatch.setenv("UJI_BEDA", "999")
    with caplog.at_level("WARNING"):
        nilai = resolve("chunk_size", None, "UJI_BEDA", "rag.chunk_size", 900, cast=int)
    assert nilai == 999
    assert any("menimpa config" in r.getMessage() for r in caplog.records), \
        "peringatan selisih env/config tidak muncul"


def test_retriever_reads_retrieval_params_from_config():
    from src.rag.retriever import MMRRetriever

    cfg = load_rag_config()
    r = MMRRetriever(persist_dir="/tmp/tidak-ada-koleksi-uji")
    assert r.top_k == cfg.top_k
    assert r.fetch_k == cfg.fetch_k
    assert r.lambda_mult == cfg.lambda_mult


def test_embedding_model_shared_between_ingest_and_query():
    """Model embedding ingest dan query WAJIB identik.

    Bila berbeda, retrieval merosot menjadi derau tanpa error apa pun. Sebelum
    Tugas 4 keduanya adalah konstanta modul terpisah yang dibaca saat import.
    """
    from src.rag import knowledge_base, retriever

    assert retriever._build_embeddings is knowledge_base._build_embeddings

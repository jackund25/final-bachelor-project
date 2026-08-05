from src.rag.knowledge_base import MedicalKnowledgeBase
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import SimpleKeywordRetriever


def _build_chunks_from_manual_kb(tmp_path):
    kb = MedicalKnowledgeBase(kb_dir=str(tmp_path))
    kb.create_manual_kb()
    return kb.chunks


def test_chunk_documents_from_manual_kb_produces_chunks(tmp_path):
    chunks = _build_chunks_from_manual_kb(tmp_path)

    assert chunks
    assert len(chunks) >= 6
    assert all(isinstance(item.get("text"), str) and item["text"].strip() for item in chunks)


def test_simple_keyword_retriever_returns_hypoglycemia_content(tmp_path):
    chunks = _build_chunks_from_manual_kb(tmp_path)
    retriever = SimpleKeywordRetriever(chunks)

    results = retriever.retrieve("hipoglikemia gula darah rendah", top_k=1)

    assert results
    assert "hipoglikemia" in results[0]["text"].lower()


def test_simple_keyword_retriever_metadata_filter_works(tmp_path):
    chunks = _build_chunks_from_manual_kb(tmp_path)
    retriever = SimpleKeywordRetriever(chunks)

    results = retriever.retrieve(
        "aktivitas fisik untuk diabetes",
        top_k=3,
        metadata_filter={"subdomain": "aktivitas_fisik"},
    )

    assert results
    assert all(item["metadata"].get("subdomain") == "aktivitas_fisik" for item in results)


def test_pipeline_builds_retriever_and_returns_docs(tmp_path):
    pipeline = RAGPipeline(kb_dir=str(tmp_path), llm_provider="template")
    pipeline.build()

    result = pipeline.answer(
        patient_state={
            "current_glucose": 165.0,
            "stress_level": 5,
            "activity_level": 20,
            "insulin_on_board": 0.0,
            "carbs_on_board": 20.0,
        },
        prediction=170.0,
        query="Apa edukasi dasar pemantauan gula darah?",
        top_k=2,
    )

    assert result["retrieved_docs"]
    assert len(result["retrieved_docs"]) <= 2


# ── Sumber pengondisian kueri (A1) ────────────────────────────────────────────
# `_enhance_query` dulu SELALU membaca patient_state["current_glucose"], sehingga
# setiap kueri produksi diam-diam membawa kondisi yang sedang berlaku meski sistem
# mengklaim mengondisikan pada kondisi TERPREDIKSI. Tes-tes berikut mengunci
# perilaku barunya: sumber kondisi harus datang dari pemanggil.

def _enhancer():
    """Instance MMRRetriever tanpa memuat ChromaDB/embedding — hanya butuh methodnya."""
    from src.rag.retriever import MMRRetriever

    return MMRRetriever.__new__(MMRRetriever)


DIVERGEN = {"current_glucose": 112.0, "stress_level": 2}  # normal sekarang, hipo nanti


def test_enhance_query_tidak_membocorkan_current_glucose():
    """Tanpa sumber eksplisit, TIDAK ada tag kondisi — walau current_glucose hipo."""
    q = _enhancer()._enhance_query("Tindakan apa?", {"current_glucose": 55.0, "stress_level": 2})

    assert "hipoglikemia" not in q
    assert q == "Tindakan apa?"


def test_enhance_query_memakai_sumber_eksplisit_bukan_current():
    """Pada kasus divergen, tag mengikuti angka yang DIBERIKAN, bukan current_glucose."""
    e = _enhancer()

    pred = e._enhance_query("Tindakan apa?", DIVERGEN, condition_glucose=58.0)
    cur = e._enhance_query("Tindakan apa?", DIVERGEN, condition_glucose=DIVERGEN["current_glucose"])

    assert "hipoglikemia" in pred
    assert "hipoglikemia" not in cur


def test_enhance_query_kedua_mode_simetris_secara_struktur():
    """Struktur kueri identik untuk kedua mode; hanya nama kondisinya yang berbeda."""
    e = _enhancer()

    hipo = e._enhance_query("Tindakan apa?", DIVERGEN, condition_glucose=58.0)
    hiper = e._enhance_query("Tindakan apa?", DIVERGEN, condition_glucose=230.0)

    assert hipo.replace("hipoglikemia", "X") == hiper.replace("hiperglikemia", "X")


def test_enhance_query_tag_stres_berlaku_pada_kedua_mode():
    """Stres tidak bergantung pada prediksi, jadi muncul apa pun sumber kondisinya."""
    e = _enhancer()
    stres = {"current_glucose": 112.0, "stress_level": 8}

    for src in (None, 58.0, 112.0, 230.0):
        assert "stress tinggi" in e._enhance_query("Tindakan apa?", stres, condition_glucose=src)


def test_pipeline_meneruskan_prediksi_sebagai_sumber_kondisi(tmp_path):
    """Jalur produksi harus mengirim glukosa TERPREDIKSI ke retriever, bukan yang sekarang."""
    pipeline = RAGPipeline(kb_dir=str(tmp_path), llm_provider="template")
    pipeline.build()

    terekam = {}
    asli = pipeline.retriever.retrieve_with_context if hasattr(
        pipeline.retriever, "retrieve_with_context") else None

    def spy(query, patient_state, top_k=None, metadata_filter=None, condition_glucose=None):
        terekam["condition_glucose"] = condition_glucose
        return asli(query, patient_state, top_k, metadata_filter, condition_glucose)

    if asli is None:  # retriever fallback tanpa retrieve_with_context
        import pytest

        pytest.skip("retriever aktif tidak punya retrieve_with_context")

    pipeline.retriever.retrieve_with_context = spy
    pipeline.answer(patient_state=dict(DIVERGEN), prediction=58.0, query="Tindakan apa?", top_k=2)

    assert terekam["condition_glucose"] == 58.0


def test_build_ablation_query_simetris_antar_mode():
    """Kueri ablasi kedua lengan hanya boleh berbeda pada angka + frasa kondisinya."""
    from src.rag.ablation_query import CONDITION_PHRASE, build_ablation_query

    std = build_ablation_query(112.0)     # lengan standard  -> current
    pc = build_ablation_query(58.0)       # lengan PC-RAG    -> predicted

    assert std == f"Kadar glukosa darah 112 mg/dL. {CONDITION_PHRASE['normal']}"
    assert pc == f"Kadar glukosa darah 58 mg/dL. {CONDITION_PHRASE['hipoglikemia']}"
    # Tidak ada penanda horizon yang hanya melekat pada satu lengan.
    assert "menit ke depan" not in std and "menit ke depan" not in pc

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

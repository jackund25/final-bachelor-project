from src.rag.generator import RAGGenerator
from src.rag.pipeline import RAGPipeline


def test_template_generator_includes_doctor_disclaimer():
    generator = RAGGenerator(provider="template")

    answer = generator.generate_explanation(
        context_docs=["Panduan hiperglikemia dan monitoring rutin"],
        patient_state={"current_glucose": 200, "stress_level": 8, "activity_level": 5},
        prediction=210.0,
    )

    assert "dokter" in answer.lower()


def test_generate_advisory_returns_answer_and_sources_structure():
    generator = RAGGenerator(provider="template")

    payload = generator.generate_advisory(
        query="Apa langkah awal untuk hiperglikemia ringan?",
        retrieved_docs=[{"text": "Pantau glukosa dan hidrasi.", "source": "manual_kb", "metadata": {}}],
        patient_state={"current_glucose": 195.0, "stress_level": 7, "activity_level": 10},
        prediction=205.0,
    )

    assert payload.get("answer")
    assert "sources" in payload


def test_pipeline_answer_contains_advisory_and_citations(tmp_path):
    pipeline = RAGPipeline(kb_dir=str(tmp_path), llm_provider="template")

    result = pipeline.answer(
        patient_state={
            "current_glucose": 205.0,
            "stress_level": 8,
            "activity_level": 10,
            "insulin_on_board": 0.0,
            "carbs_on_board": 25.0,
        },
        prediction=210.0,
    )

    assert result["advisory"]["doctor_review_required"] is True
    assert result["advisory"]["source_count"] >= 1
    assert "citations" in result


def test_pipeline_risk_levels_are_classified_correctly(tmp_path):
    pipeline = RAGPipeline(kb_dir=str(tmp_path), llm_provider="template")

    low = pipeline.answer(
        patient_state={"current_glucose": 90.0, "stress_level": 4, "activity_level": 25},
        prediction=65.0,
        query="Evaluasi risiko",
    )
    high = pipeline.answer(
        patient_state={"current_glucose": 190.0, "stress_level": 8, "activity_level": 5},
        prediction=220.0,
        query="Evaluasi risiko",
    )

    assert low["risk_level"].startswith("BAHAYA")
    assert high["risk_level"].startswith("HATI-HATI")

from src.rag import RAGPipeline


def test_rag_pipeline_builds_and_answers_with_template_fallback(tmp_path):
	pipeline = RAGPipeline(kb_dir=str(tmp_path), llm_provider="groq")
	result = pipeline.answer(
		patient_state={
			"current_glucose": 205.0,
			"stress_level": 8,
			"activity_level": 10,
			"carbs_on_board": 25.0,
			"insulin_on_board": 0.0,
		},
		prediction=210.0,
	)

	assert result["risk_level"] == "HATI-HATI - Hiperglikemia"
	assert result["explanation"]
	assert result["retrieved_docs"]
	assert result["retrieved_docs"][0]["text"]


def test_rag_pipeline_keyword_retrieval_prioritizes_relevant_chunk(tmp_path):
	pipeline = RAGPipeline(kb_dir=str(tmp_path), llm_provider="template")
	pipeline.build()

	results = pipeline.retriever.retrieve("hipoglikemia gula darah rendah", top_k=1)

	assert results
	assert "hipoglikemia" in results[0]["text"].lower()
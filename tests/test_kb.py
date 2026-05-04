from src.rag.knowledge_base import MedicalKnowledgeBase


def test_manual_kb_creation_has_valid_chunks(tmp_path):
	kb = MedicalKnowledgeBase(kb_dir=str(tmp_path))
	kb.create_manual_kb()

	assert kb.chunks
	assert len(kb.chunks) >= 6
	assert all(isinstance(chunk.get("text"), str) and chunk["text"].strip() for chunk in kb.chunks)
	assert all(chunk.get("source") == "manual_kb" for chunk in kb.chunks)
	assert (tmp_path / "manual_kb.json").exists()
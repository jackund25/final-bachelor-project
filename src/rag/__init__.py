"""Public API for RAG components."""

from .generator import RAGGenerator
from .knowledge_base import MedicalKnowledgeBase
from .pipeline import RAGPipeline

try:
	from .retriever import DocumentRetriever
except Exception:  # pragma: no cover - optional dependency fallback
	DocumentRetriever = None

__all__ = ["DocumentRetriever", "MedicalKnowledgeBase", "RAGGenerator", "RAGPipeline"]

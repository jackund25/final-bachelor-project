"""Simple orchestration layer for RAG retrieval and generation."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .generator import RAGGenerator
from .knowledge_base import MedicalKnowledgeBase

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
	return re.findall(r"[a-z0-9]+", text.lower())


def _risk_level_from_prediction(prediction: float) -> str:
	if prediction < 70:
		return "BAHAYA - Hipoglikemia"
	if prediction > 180:
		return "HATI-HATI - Hiperglikemia"
	return "AMAN"


@dataclass
class RetrievedDocument:
	rank: int
	text: str
	source: str
	similarity: float
	metadata: Dict[str, Any]


class SimpleKeywordRetriever:
	"""Fallback retriever that scores chunks by keyword overlap."""

	def __init__(self, chunks: List[Dict[str, Any]]):
		self.chunks = chunks

	def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
		query_tokens = Counter(_tokenize(query))
		results: list[dict[str, Any]] = []

		for chunk in self.chunks:
			chunk_text = chunk.get("text", "")
			chunk_tokens = Counter(_tokenize(chunk_text))
			overlap = sum(min(query_tokens[token], chunk_tokens[token]) for token in query_tokens)
			normalizer = max(len(query_tokens), 1)
			similarity = overlap / normalizer
			results.append(
				{
					"rank": 0,
					"chunk": chunk,
					"text": chunk_text,
					"distance": float(1.0 - similarity),
					"similarity": float(similarity),
				},
			)

		results.sort(key=lambda item: item["similarity"], reverse=True)
		for index, item in enumerate(results[:top_k], start=1):
			item["rank"] = index
		return results[:top_k]


class RAGPipeline:
	"""Minimal end-to-end RAG service for diabetic decision support."""

	def __init__(
		self,
		kb_dir: str = "data/knowledge_base",
		llm_provider: str = "template",
		top_k: int = 3,
		use_semantic_retrieval: bool = False,
	):
		self.kb = MedicalKnowledgeBase(kb_dir)
		self.top_k = top_k
		self.use_semantic_retrieval = use_semantic_retrieval
		self.retriever: Any = None
		self.generator: Optional[RAGGenerator] = None
		self._ready = False
		self._requested_provider = llm_provider

	def build(self) -> None:
		"""Load the knowledge base and prepare retrieval/generation."""
		self.kb.load_documents("*.txt")

		if self.kb.documents:
			self.kb.process_all_documents()
		elif not self.kb.chunks:
			self.kb.create_manual_kb()

		self.retriever = self._build_retriever()
		self.generator = self._build_generator()
		self._ready = True

	def answer(self, patient_state: Dict[str, Any], prediction: float, query: Optional[str] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
		"""Generate a compact RAG response for the current patient context."""
		if not self._ready:
			self.build()

		query_text = query or self._build_query(patient_state, prediction)
		k = top_k or self.top_k
		raw_results = self.retriever.retrieve(query_text, top_k=k)
		retrieved_docs = [
			RetrievedDocument(
				rank=item.get("rank", index + 1),
				text=item.get("text", ""),
				source=item.get("chunk", {}).get("source", "manual_kb"),
				similarity=float(item.get("similarity", 0.0)),
				metadata=item.get("chunk", {}),
			)
			for index, item in enumerate(raw_results)
		]

		context_docs = [doc.text for doc in retrieved_docs]
		explanation = self.generator.generate_explanation(context_docs, patient_state, prediction)

		return {
			"query": query_text,
			"risk_level": _risk_level_from_prediction(prediction),
			"prediction": float(prediction),
			"retrieved_docs": [
				{
					"rank": doc.rank,
					"source": doc.source,
					"similarity": doc.similarity,
					"text": doc.text,
					"metadata": doc.metadata,
				}
				for doc in retrieved_docs
			],
			"explanation": explanation,
		}

	def _build_query(self, patient_state: Dict[str, Any], prediction: float) -> str:
		glucose = patient_state.get("current_glucose", "N/A")
		stress = patient_state.get("stress_level", "N/A")
		activity = patient_state.get("activity_level", 0)
		return (
			f"Pasien diabetes dengan gula darah sekarang {glucose} mg/dL, "
			f"prediksi {prediction:.1f} mg/dL, stres {stress}/10, aktivitas {activity} menit. "
			"Berikan penjelasan singkat dan tindakan aman."
		)

	def _build_retriever(self) -> Any:
		chunks = self.kb.chunks
		if not chunks:
			self.kb.create_manual_kb()
			chunks = self.kb.chunks

		if self.use_semantic_retrieval:
			try:
				from .retriever import DocumentRetriever

				semantic_retriever = DocumentRetriever()
				semantic_retriever.index_documents(chunks)
				return semantic_retriever
			except Exception as exc:
				logger.warning("Semantic retriever unavailable, using keyword fallback: %s", exc)

		return SimpleKeywordRetriever(chunks)

	def _build_generator(self) -> RAGGenerator:
		try:
			return RAGGenerator(provider=self._requested_provider)
		except Exception as exc:
			logger.warning("RAG generator unavailable, using template fallback: %s", exc)
			return RAGGenerator(provider="template")
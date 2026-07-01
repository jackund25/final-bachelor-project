"""LCEL-based advisor chain for diabetes RAG — supports Gemini and Ollama backends."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .prompts import SYSTEM_PROMPT, build_question_payload, format_context_with_citations

logger = logging.getLogger(__name__)


class DiabetesAdvisorChain:
    """Generate clinical advisory text from retrieved context and patient state.

    Supports two LLM backends:
    - ``"gemini"``  — Google Gemini via langchain-google-genai (recommended, free tier)
    - ``"ollama"``  — local Ollama server (fallback for offline use)

    Falls back to a rule-based template if neither backend initialises successfully.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        provider: str = "gemini",
        ollama_base_url: str = "http://localhost:11434",
        gemini_api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.provider = provider
        self.ollama_base_url = ollama_base_url
        self._chain = None
        self._init_error: Optional[str] = None

        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    (
                        "human",
                        "Konteks dokumen:\n{context}\n\n"
                        "Data dan pertanyaan:\n{question_payload}\n\n"
                        "Berikan jawaban klinis ringkas dengan langkah aksi dan disclaimer dokter.",
                    ),
                ]
            )

            if provider == "gemini":
                api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY environment variable not set. "
                        "Dapatkan API key gratis di https://aistudio.google.com/app/apikey"
                    )
                from langchain_google_genai import ChatGoogleGenerativeAI

                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0.1,
                )
            else:
                from langchain_ollama import ChatOllama

                llm = ChatOllama(
                    model=model_name,
                    base_url=ollama_base_url,
                    temperature=0.1,
                )

            self._chain = prompt | llm | StrOutputParser()
            logger.info("Advisor chain initialised — provider=%s model=%s", provider, model_name)

        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("Advisor chain fallback active (provider=%s): %s", provider, exc)

    @property
    def is_ready(self) -> bool:
        return self._chain is not None

    def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        patient_state: Dict[str, Any],
        prediction: float,
    ) -> Dict[str, Any]:
        context_block = format_context_with_citations(retrieved_docs)
        question_payload = build_question_payload(query, patient_state, prediction)

        if self._chain is not None:
            try:
                answer = self._chain.invoke(
                    {
                        "context": context_block,
                        "question_payload": question_payload,
                    }
                )
                return {
                    "answer": answer.strip(),
                    "sources": self._extract_sources(retrieved_docs),
                }
            except Exception as exc:
                logger.warning("Advisor chain invocation failed, using template: %s", exc)

        return {
            "answer": self._template_answer(patient_state, prediction, retrieved_docs),
            "sources": self._extract_sources(retrieved_docs),
        }

    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for row in retrieved_docs:
            metadata = dict(row.get("metadata", {}))
            output.append(
                {
                    "source": metadata.get("sumber") or row.get("source", "Manual KB"),
                    "title": metadata.get("judul", row.get("source", "Manual KB")),
                    "year": metadata.get("tahun", "N/A"),
                    "page": metadata.get("halaman", "N/A"),
                }
            )
        return output

    def _template_answer(
        self,
        patient_state: Dict[str, Any],
        prediction: float,
        retrieved_docs: List[Dict[str, Any]],
    ) -> str:
        glucose = float(patient_state.get("current_glucose", 100.0))
        if prediction < 70:
            risk = "BAHAYA - Hipoglikemia"
            action = "Lakukan aturan 15-15 dan evaluasi medis segera."
        elif prediction > 180:
            risk = "HATI-HATI - Hiperglikemia"
            action = "Perkuat hidrasi, evaluasi asupan, dan pantau ulang glukosa dalam 1 jam."
        else:
            risk = "AMAN"
            action = "Pertahankan pola makan dan monitoring rutin."

        source_count = len(retrieved_docs)
        return (
            f"Status risiko: {risk}. Prediksi glukosa 1 jam adalah {prediction:.1f} mg/dL "
            f"dari kondisi saat ini {glucose:.1f} mg/dL. "
            f"Rekomendasi awal: {action} Konteks yang digunakan: {source_count} sumber. "
            "Catatan: keputusan klinis final tetap memerlukan penilaian dokter."
        )

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
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        config: Optional[Any] = None,
    ):
        from src.config import load_rag_config

        cfg = config or load_rag_config()
        provider = provider or cfg.llm_provider
        self.model_name = model_name or (
            cfg.llm_model if provider == "gemini" else cfg.ollama_llm_model
        )
        self.provider = provider
        self.ollama_base_url = ollama_base_url or cfg.ollama_base_url
        self.temperature = temperature if temperature is not None else cfg.temperature
        self.max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        model_name = self.model_name
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
                        "Berikan jawaban klinis ringkas yang langsung menjawab pertanyaan. "
                        "Berikan langkah aksi hanya jika pertanyaan meminta tindakan atau penanganan, "
                        "atau jika tindakan tersebut diperlukan untuk keselamatan pasien. "
                        "Jangan menambahkan rekomendasi klinis yang tidak diminta. "
                        "Selalu sertakan disclaimer dokter.",
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
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            else:
                from langchain_ollama import ChatOllama

                llm = ChatOllama(
                    model=model_name,
                    base_url=self.ollama_base_url,
                    temperature=self.temperature,
                    num_predict=self.max_tokens,
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
        horizon_minutes: Optional[int] = None,
        clinical_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        context_block = format_context_with_citations(retrieved_docs)
        question_payload = build_question_payload(
            query, patient_state, prediction, horizon_minutes=horizon_minutes
        )
        # Blok kondisi klinis terstruktur (tren, urgensi, IOB/COB) dari
        # PredictionConditionedQueryBuilder._llm_context(). Sebelumnya blok ini dibangun
        # lalu dibuang — tidak pernah sampai ke LLM sama sekali.
        if clinical_context:
            question_payload = f"{clinical_context}\n\n{question_payload}"

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
        """Ringkasan sumber terstruktur.

        Membaca skema metadata halaman yang baru (kb_id/lembaga/halaman_cetak) dengan
        fallback ke kunci lama (sumber/judul/halaman) agar chunk manual_kb tetap jalan.
        """
        from .citations import format_page_label

        output: List[Dict[str, Any]] = []
        for row in retrieved_docs:
            metadata = dict(row.get("metadata", {}))
            fallback_name = row.get("source", "Manual KB")
            output.append(
                {
                    "source": (
                        metadata.get("lembaga")
                        or metadata.get("sumber")
                        or fallback_name
                    ),
                    "title": (
                        metadata.get("judul_lengkap")
                        or metadata.get("judul")
                        or fallback_name
                    ),
                    "year": metadata.get("tahun", "N/A"),
                    "page": format_page_label(metadata),
                    "kb_id": metadata.get("kb_id", ""),
                }
            )
        return output

    def _template_answer(
        self,
        patient_state: Dict[str, Any],
        prediction: float,
        retrieved_docs: List[Dict[str, Any]],
    ) -> str:
        from src.constants import (
            CLASS_HYPER, CLASS_HYPO, classify_glucose_3class, classify_glucose_5zone,
            risk_label_id,
        )

        glucose = float(patient_state.get("current_glucose", 100.0))
        risk = risk_label_id(classify_glucose_5zone(prediction))
        kondisi = classify_glucose_3class(prediction)
        if kondisi == CLASS_HYPO:
            action = "Lakukan aturan 15-15 dan evaluasi medis segera."
        elif kondisi == CLASS_HYPER:
            action = "Perkuat hidrasi, evaluasi asupan, dan pantau ulang glukosa dalam 1 jam."
        else:
            action = "Pertahankan pola makan dan monitoring rutin."

        source_count = len(retrieved_docs)
        return (
            f"Status risiko: {risk}. Prediksi glukosa 1 jam adalah {prediction:.1f} mg/dL "
            f"dari kondisi saat ini {glucose:.1f} mg/dL. "
            f"Rekomendasi awal: {action} Konteks yang digunakan: {source_count} sumber. "
            "Catatan: keputusan klinis final tetap memerlukan penilaian dokter."
        )

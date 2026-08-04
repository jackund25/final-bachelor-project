"""Generation helpers for diabetes RAG advisory outputs."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .advisor_chain import DiabetesAdvisorChain


class RAGGenerator:
    """Generate explanations and advisories from retrieved knowledge chunks.

    Provider options:
    - ``"gemini"``   — Google Gemini (default, free tier via GOOGLE_API_KEY env var)
    - ``"ollama"``   — local Ollama server
    - ``"template"`` — rule-based fallback, no LLM required
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
    ):
        from src.config import load_rag_config

        cfg = config or load_rag_config()
        self.provider = provider or cfg.llm_provider
        self.config = model_config or {}

        if self.provider == "template":
            self.chain = None
            return

        if self.provider == "gemini":
            self.chain = DiabetesAdvisorChain(
                model_name=self.config.get("model"),
                provider="gemini",
                gemini_api_key=self.config.get("api_key") or os.getenv("GOOGLE_API_KEY"),
                config=cfg,
            )
        else:
            self.chain = DiabetesAdvisorChain(
                model_name=self.config.get("model"),
                provider="ollama",
                ollama_base_url=self.config.get("base_url"),
                config=cfg,
            )

    def generate_explanation(
        self,
        context_docs: List[str],
        patient_state: Dict[str, Any],
        prediction: float,
    ) -> str:
        """Penjelasan klinis singkat dari daftar teks konteks.

        temperature dan max_tokens dulu menjadi parameter di sini lalu langsung
        dibuang dengan `del`. Keduanya kini dibaca dari config.yaml oleh
        DiabetesAdvisorChain, sehingga parameternya dihapus agar tidak menyesatkan.
        """
        docs = [{"text": item, "source": "manual_kb", "metadata": {}} for item in context_docs]
        payload = self.generate_advisory(
            query="Berikan penjelasan klinis singkat berdasarkan kondisi pasien.",
            retrieved_docs=docs,
            patient_state=patient_state,
            prediction=prediction,
        )
        return payload["answer"]

    def generate_advisory(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        patient_state: Dict[str, Any],
        prediction: float,
    ) -> Dict[str, Any]:
        if self.chain is None:
            return {
                "answer": self._template_answer(patient_state=patient_state, prediction=prediction),
                "sources": [],
            }

        return self.chain.generate(
            query=query,
            retrieved_docs=retrieved_docs,
            patient_state=patient_state,
            prediction=prediction,
        )

    def _template_answer(self, patient_state: Dict[str, Any], prediction: float) -> str:
        from src.constants import (
            CLASS_HYPER, CLASS_HYPO, classify_glucose_3class, classify_glucose_5zone,
            risk_label_id,
        )

        glucose = float(patient_state.get("current_glucose", 100.0))
        stress = int(patient_state.get("stress_level", 5))
        risk = risk_label_id(classify_glucose_5zone(prediction))
        kondisi = classify_glucose_3class(prediction)
        if kondisi == CLASS_HYPO:
            advice = "Segera lakukan aturan 15-15 dan evaluasi klinis."
        elif kondisi == CLASS_HYPER:
            advice = "Pantau ulang glukosa dalam 1 jam dan tinjau asupan serta aktivitas."
        else:
            advice = "Lanjutkan monitoring rutin dan pertahankan pola sehat."

        stress_note = (
            "Stres tinggi berpotensi meningkatkan glukosa."
            if stress >= 7
            else "Tingkat stres relatif terkontrol."
        )

        return (
            f"Status: {risk}. Prediksi 1 jam ke depan {prediction:.1f} mg/dL "
            f"dari kondisi saat ini {glucose:.1f} mg/dL. "
            f"{stress_note} Rekomendasi: {advice} "
            "Catatan: keputusan medis final tetap pada dokter."
        )

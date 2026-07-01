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

    def __init__(self, provider: str = "gemini", model_config: Optional[Dict[str, Any]] = None):
        self.provider = provider
        self.config = model_config or {}

        if provider == "template":
            self.chain = None
            return

        if provider == "gemini":
            model_name = self.config.get("model", os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
            api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
            self.chain = DiabetesAdvisorChain(
                model_name=model_name,
                provider="gemini",
                gemini_api_key=api_key,
            )
        else:
            model_name = self.config.get("model", "llama3.1:8b")
            base_url = self.config.get("base_url", "http://localhost:11434")
            self.chain = DiabetesAdvisorChain(
                model_name=model_name,
                provider="ollama",
                ollama_base_url=base_url,
            )

    def generate_explanation(
        self,
        context_docs: List[str],
        patient_state: Dict[str, Any],
        prediction: float,
        temperature: float = 0.1,
        max_tokens: int = 300,
    ) -> str:
        del temperature
        del max_tokens

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
        glucose = float(patient_state.get("current_glucose", 100.0))
        stress = int(patient_state.get("stress_level", 5))
        if prediction < 70:
            risk = "BAHAYA - Hipoglikemia"
            advice = "Segera lakukan aturan 15-15 dan evaluasi klinis."
        elif prediction > 180:
            risk = "HATI-HATI - Hiperglikemia"
            advice = "Pantau ulang glukosa dalam 1 jam dan tinjau asupan serta aktivitas."
        else:
            risk = "AMAN"
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

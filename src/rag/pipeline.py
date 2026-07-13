"""RAG orchestration pipeline: ingestion, retrieval, and advisory generation."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .conditioned_query import PredictionConditionedQueryBuilder, QueryStrategy
from .generator import RAGGenerator
from .knowledge_base import MedicalKnowledgeBase
from .retriever import MMRRetriever, SimpleKeywordRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    rank: int
    text: str
    source: str
    similarity: float
    metadata: Dict[str, Any]


def _risk_level_from_prediction(prediction: float) -> str:
    if prediction < 70:
        return "BAHAYA - Hipoglikemia"
    if prediction > 180:
        return "HATI-HATI - Hiperglikemia"
    return "AMAN"


class RAGPipeline:
    """End-to-end prediction-conditioned RAG pipeline.

    Novelty: the ML model's numeric prediction is embedded directly into the
    retrieval query (``_build_query``), so retrieved chunks and the LLM advisory
    are conditioned on the predicted glucose value — not on a static text query.

    LLM provider options (``llm_provider``):
    - ``"gemini"``   — Google Gemini via free-tier API key (recommended)
    - ``"ollama"``   — local Ollama server (offline fallback)
    - ``"template"`` — deterministic rule-based answer, no LLM required

    Embedding provider options (``embed_provider``):
    - ``"sentence-transformers"`` — CPU-only, downloads once (~80 MB), no server (default)
    - ``"google"``                — Google Generative AI embeddings (requires GOOGLE_API_KEY)
    - ``"ollama"``                — Ollama nomic-embed-text (requires Ollama server)
    """

    def __init__(
        self,
        kb_dir: str = "data/knowledge_base",
        chroma_persist_dir: str = "models/chroma_db",
        collection_name: str = "diabetes_kb",
        llm_provider: str = "gemini",
        embed_provider: str = "sentence-transformers",
        top_k: int = 4,
        google_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_llm_model: Optional[str] = None,
        ollama_embed_model: Optional[str] = None,
    ):
        self.kb_dir = kb_dir
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self.llm_provider = llm_provider
        self.embed_provider = embed_provider
        self.top_k = top_k

        # Resolve credentials — env vars are the canonical source; ctor params override.
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_llm_model = ollama_llm_model or os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
        self.ollama_embed_model = ollama_embed_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

        self.kb = MedicalKnowledgeBase(
            kb_dir=self.kb_dir,
            persist_dir=self.chroma_persist_dir,
            collection_name=self.collection_name,
            embed_provider=embed_provider,
            ollama_base_url=self.ollama_base_url,
            embed_model=self.ollama_embed_model,
        )

        self.retriever: Any = None

        if llm_provider == "gemini":
            model_config: Dict[str, Any] = {
                "model": self.gemini_model,
                "api_key": self.google_api_key,
            }
        else:
            model_config = {
                "model": self.ollama_llm_model,
                "base_url": self.ollama_base_url,
            }

        self.generator = RAGGenerator(
            provider=llm_provider,
            model_config=model_config,
        )
        self._ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, reset_collection: bool = False) -> Dict[str, Any]:
        """Load manual KB, chunk it, and persist to Chroma."""
        docs = self.kb.load_manual_kb("manual_kb.json")
        if not docs:
            self.kb.create_manual_kb()
            docs = self.kb.documents

        chunks = self.kb.chunk_documents(documents=docs)
        saved = self.kb.save_to_chroma(chunks=chunks, reset_collection=reset_collection)

        return {
            "documents": len(docs),
            "chunks": len(chunks),
            "saved_to_chroma": bool(saved),
            "persist_dir": self.chroma_persist_dir,
            "collection_name": self.collection_name,
            "embed_provider": self.embed_provider,
        }

    def build(self) -> None:
        """Prepare retriever — prefer Chroma MMR, fall back to keyword if unavailable."""
        mmr_retriever = MMRRetriever(
            persist_dir=self.chroma_persist_dir,
            collection_name=self.collection_name,
            embed_provider=self.embed_provider,
            ollama_base_url=self.ollama_base_url,
            embed_model=self.ollama_embed_model,
        )

        if mmr_retriever.is_ready:
            self.retriever = mmr_retriever
            logger.info("RAGPipeline: using MMR retriever (embed=%s)", self.embed_provider)
        else:
            if not self.kb.chunks:
                docs = self.kb.load_manual_kb("manual_kb.json")
                if docs:
                    self.kb.chunk_documents(documents=docs, chunk_size=350, chunk_overlap=40)
                else:
                    self.kb.create_manual_kb()
            self.retriever = SimpleKeywordRetriever(self.kb.chunks)
            logger.info("RAGPipeline: using keyword retriever (Chroma unavailable)")

        self._ready = True

    def answer(
        self,
        patient_state: Dict[str, Any],
        prediction: float,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a prediction-conditioned clinical advisory.

        The ``prediction`` value (glucose mg/dL from the ML model) is embedded
        into the retrieval query so that both document retrieval and the LLM
        response are conditioned on the numerical forecast — the core novelty
        of this system versus static SPARQL-query approaches.
        """
        if not self._ready:
            self.build()

        # Prediction-conditioned query: numeric prediction → natural language context
        user_query = query or self._build_query(patient_state, prediction)
        k = top_k or self.top_k

        retrieved_rows = self._retrieve(user_query, patient_state=patient_state, top_k=k)
        retrieved_docs = [
            RetrievedDocument(
                rank=row.get("rank", idx + 1),
                text=row.get("text", ""),
                source=row.get("source", "manual_kb"),
                similarity=float(row.get("similarity", 0.0)),
                metadata=dict(row.get("metadata", {})),
            )
            for idx, row in enumerate(retrieved_rows)
        ]

        advisory_payload = self.generator.generate_advisory(
            query=user_query,
            retrieved_docs=[
                {
                    "text": item.text,
                    "source": item.source,
                    "metadata": item.metadata,
                }
                for item in retrieved_docs
            ],
            patient_state=patient_state,
            prediction=prediction,
        )

        explanation = self._ensure_disclaimer(advisory_payload["answer"])
        advisory = self._build_advisory(patient_state, prediction, retrieved_docs, explanation)

        return {
            "query": user_query,
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
            "advisory": advisory,
            "citations": advisory_payload.get("sources", []),
            "llm_provider": self.llm_provider,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, patient_state: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        if hasattr(self.retriever, "retrieve_with_context"):
            return self.retriever.retrieve_with_context(query, patient_state=patient_state, top_k=top_k)
        return self.retriever.retrieve(query=query, top_k=top_k)

    def _build_query(self, patient_state: Dict[str, Any], prediction: float) -> str:
        """Build a prediction-conditioned retrieval query via PredictionConditionedQueryBuilder.

        The query embeds the numeric ML prediction so that retrieved chunks
        and the LLM advisory are conditioned on the FORECASTED glucose
        trajectory — the core novelty vs. static SPARQL-query systems.
        """
        try:
            try:
                from src.patient_state import PatientState
            except ImportError:
                from ..patient_state import PatientState

            def _opt_float(key: str):
                v = patient_state.get(key)
                return float(v) if v is not None else None

            state = PatientState.from_model_output(
                patient_id=str(patient_state.get("patient_id", "unknown")),
                current_glucose=float(patient_state.get("current_glucose", 100.0)),
                predicted_glucose=float(prediction),
                feature_row=patient_state,
                # Kondisi dari pengklasifikasi & batas interval konformal (bila disediakan
                # pemanggil) mengaktifkan pengondisian kueri yang sadar-ketidakpastian.
                predicted_condition=patient_state.get("predicted_condition"),
                predicted_lower=_opt_float("predicted_lower"),
                predicted_upper=_opt_float("predicted_upper"),
            )
            builder = PredictionConditionedQueryBuilder(strategy=QueryStrategy.COMPREHENSIVE)
            cq = builder.build(state)
            return cq.primary_query
        except Exception as exc:
            logger.warning("PredictionConditionedQueryBuilder failed, using fallback: %s", exc)
            # Fallback: simple inline query
            current = patient_state.get("current_glucose", "N/A")
            stress = patient_state.get("stress_level", "N/A")
            activity = patient_state.get("activity_level", 0)
            delta = (
                f"{prediction - float(current):+.1f} mg/dL"
                if isinstance(current, (int, float))
                else "N/A"
            )
            return (
                f"Pasien diabetes dengan glukosa saat ini {current} mg/dL, "
                f"stress {stress}/10, aktivitas {activity} menit. "
                f"Model memprediksi glukosa 1 jam ke depan: {prediction:.1f} mg/dL "
                f"(perubahan {delta}). "
                "Berikan penilaian risiko dan tindakan aman yang perlu dipantau dokter."
            )

    def _build_advisory(
        self,
        patient_state: Dict[str, Any],
        prediction: float,
        retrieved_docs: List[RetrievedDocument],
        explanation: str,
    ) -> Dict[str, Any]:
        current_glucose = float(patient_state.get("current_glucose", 100.0))
        stress = int(patient_state.get("stress_level", 5))
        activity = int(patient_state.get("activity_level", 0))
        risk_level = _risk_level_from_prediction(prediction)

        key_factors: List[str] = []
        if stress >= 7:
            key_factors.append("stres tinggi")
        if activity < 15:
            key_factors.append("aktivitas fisik rendah")
        if current_glucose > 150:
            key_factors.append("glukosa awal tinggi")

        return {
            "risk_level": risk_level,
            "summary": f"Prediksi {prediction:.1f} mg/dL dari baseline {current_glucose:.1f} mg/dL.",
            "key_factors": key_factors or ["kondisi metabolik saat ini"],
            "actions": self._actions_for_risk(risk_level),
            "doctor_review_required": True,
            "source_count": len(retrieved_docs),
            "llm_summary": explanation,
        }

    def _actions_for_risk(self, risk_level: str) -> List[str]:
        if risk_level.startswith("BAHAYA"):
            return [
                "Lakukan tatalaksana segera sesuai protokol klinis.",
                "Pantau ulang glukosa dalam interval singkat.",
                "Segera lakukan evaluasi dokter sebelum keputusan lanjutan.",
            ]
        if risk_level.startswith("HATI-HATI"):
            return [
                "Perkuat hidrasi dan review asupan karbohidrat.",
                "Tambahkan aktivitas ringan bila aman.",
                "Pantau glukosa ulang dan konsultasikan hasil ke dokter.",
            ]
        return [
            "Lanjutkan monitoring rutin dan pola hidup stabil.",
            "Pertahankan aktivitas fisik terjadwal.",
            "Tetap lakukan evaluasi berkala bersama dokter.",
        ]

    def _ensure_disclaimer(self, text: str) -> str:
        disclaimer = "keputusan medis final tetap pada dokter"
        if disclaimer in text.lower():
            return text
        return f"{text.strip()} Catatan: Keputusan medis final tetap pada dokter."


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument("action", choices=["ingest", "query"], help="Pipeline action")
    parser.add_argument("--question", default="Apa rekomendasi awal untuk kondisi ini?")
    parser.add_argument("--prediction", type=float, default=160.0, help="Predicted glucose value")
    parser.add_argument("--glucose", type=float, default=150.0)
    parser.add_argument("--stress", type=int, default=5)
    parser.add_argument("--activity", type=int, default=20)
    parser.add_argument("--reset", action="store_true", help="Reset Chroma collection on ingest")
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", "gemini"),
        choices=["gemini", "ollama", "template"],
        help="LLM provider (default: env LLM_PROVIDER or gemini)",
    )
    parser.add_argument(
        "--embed",
        default=os.getenv("EMBED_PROVIDER", "sentence-transformers"),
        choices=["sentence-transformers", "google", "ollama"],
        help="Embedding provider",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = _build_arg_parser()
    args = parser.parse_args()

    pipeline = RAGPipeline(llm_provider=args.provider, embed_provider=args.embed)

    if args.action == "ingest":
        result = pipeline.ingest(reset_collection=args.reset)
        logger.info("Ingest result: %s", result)
        return 0

    result = pipeline.answer(
        patient_state={
            "current_glucose": args.glucose,
            "stress_level": args.stress,
            "activity_level": args.activity,
            "insulin_on_board": 0.0,
            "carbs_on_board": 0.0,
        },
        prediction=args.prediction,
        query=args.question,
    )
    logger.info("Risk: %s", result["risk_level"])
    logger.info("Provider: %s", result["llm_provider"])
    logger.info("Answer: %s", result["explanation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

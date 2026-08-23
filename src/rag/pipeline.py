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
    # None bila retriever tidak mengembalikan skor (mis. fallback MMR tanpa penilaian).
    similarity: Optional[float]
    metadata: Dict[str, Any]


def _opt_similarity(value: Any) -> Optional[float]:
    """Skor kemiripan boleh tidak tersedia; jangan paksa jadi 0.0 yang menyesatkan."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


from src.timing import STAGE_GENERATE, STAGE_QUERY, STAGE_RETRIEVE, StageTimer  # noqa: E402


def _risk_level_from_prediction(prediction: float) -> str:
    """Label risiko Bahasa Indonesia dari nilai prediksi (ambang: src/constants.py)."""
    from src.constants import classify_glucose_5zone, risk_label_id

    return risk_label_id(classify_glucose_5zone(prediction))


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
        kb_dir: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        embed_provider: Optional[str] = None,
        top_k: Optional[int] = None,
        google_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_llm_model: Optional[str] = None,
        ollama_embed_model: Optional[str] = None,
        prediction_horizon_minutes: Optional[int] = None,
        config: Optional[Any] = None,
    ):
        from src.config import load_rag_config

        # Resolusi konfigurasi dilakukan SEKALI di sini, lalu objek RagConfig yang sama
        # diteruskan ke seluruh komponen. Meneruskan satu dataclass jauh lebih baik
        # daripada meneruskan sembilan skalar: menambah tombol berikutnya hanya
        # menyentuh src/config.py.
        cfg = config or load_rag_config()
        self.rag_cfg = cfg

        self.kb_dir = kb_dir or cfg.knowledge_base_dir
        self.chroma_persist_dir = chroma_persist_dir or cfg.persist_dir
        self.collection_name = collection_name or cfg.collection_name
        self.llm_provider = llm_provider or cfg.llm_provider
        self.embed_provider = embed_provider or cfg.embedding_provider
        self.top_k = top_k if top_k is not None else cfg.top_k

        # Horizon prediksi dalam MENIT, diturunkan dari config:
        #   model.default_horizon (langkah) x data.sampling_interval_min (menit/langkah)
        # Dipakai untuk menyusun kueri retrieval dan payload prompt, supaya keduanya
        # menyatakan horizon yang sama dengan yang benar-benar diprediksi model.
        from src.config import cfg_get
        self.prediction_horizon_minutes = int(
            prediction_horizon_minutes
            if prediction_horizon_minutes is not None
            else cfg_get("model.default_horizon", 6) * cfg_get("data.sampling_interval_min", 5)
        )

        # Kredensial tetap dari environment (rahasia, per-mesin).
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.gemini_model = gemini_model or cfg.llm_model
        self.ollama_base_url = ollama_base_url or cfg.ollama_base_url
        self.ollama_llm_model = ollama_llm_model or cfg.ollama_llm_model
        self.ollama_embed_model = ollama_embed_model or cfg.ollama_embed_model

        self.kb = MedicalKnowledgeBase(
            kb_dir=self.kb_dir,
            persist_dir=self.chroma_persist_dir,
            collection_name=self.collection_name,
            embed_provider=self.embed_provider,
            ollama_base_url=self.ollama_base_url,
            embed_model=self.ollama_embed_model,
            config=cfg,
        )

        self.retriever: Any = None

        if self.llm_provider == "gemini":
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
            provider=self.llm_provider,
            model_config=model_config,
            config=cfg,
        )
        self._ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, reset_collection: bool = False) -> Dict[str, Any]:
        """Indeks ulang korpus dari potongan cadangan.

        CATATAN CAKUPAN (17 Agustus 2026). Jalur ingesti PRODUKSI adalah
        ``scripts/reingest_kb.py``, yang membaca PDF pedoman per halaman beserta
        ``manifest.csv`` sehingga tiap potongan membawa nomor halaman cetaknya. Metode ini
        hanya jalur ringkas untuk pengujian dan pemulihan, dan kini membaca
        ``fallback_chunks.json`` yang diekspor skrip tersebut.

        Sebelumnya metode ini membaca ``manual_kb.json``, yakni prosa yang disusun sendiri
        tanpa nomor halaman. Berkas itu sudah dicabut dari sistem.
        """
        docs = self.kb.muat_potongan_cadangan()
        if not docs:
            raise RuntimeError(
                "fallback_chunks.json tidak ditemukan. Jalankan scripts/reingest_kb.py "
                "lebih dulu; ia yang membangun indeks produksi sekaligus mengekspor "
                "potongan cadangan.")

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
            config=self.rag_cfg,
        )

        if mmr_retriever.is_ready:
            self.retriever = mmr_retriever
            logger.info("RAGPipeline: using MMR retriever (embed=%s)", self.embed_provider)
        else:
            # Cadangan KNF-04. Potongan diambil dari fallback_chunks.json, yang diekspor
            # scripts/reingest_kb.py DARI korpus pedoman, sehingga tiap potongan cadangan
            # tetap membawa identitas dokumen dan nomor halamannya. Kegagalan penelusuran
            # vektor karena itu menurunkan JANGKAUAN, bukan keterlacakan.
            if not self.kb.chunks:
                self.kb.chunks = self.kb.muat_potongan_cadangan()
            self.retriever = SimpleKeywordRetriever(self.kb.chunks)
            if self.kb.chunks:
                logger.warning(
                    "RAGPipeline: ChromaDB tidak tersedia — mundur ke penelusuran kata "
                    "kunci atas %d potongan cadangan. Jangkauan JAUH lebih sempit "
                    "daripada korpus penuh.", len(self.kb.chunks))
            else:
                logger.error(
                    "RAGPipeline: ChromaDB tidak tersedia DAN potongan cadangan tidak "
                    "ditemukan. Penelusuran tidak dapat dilayani; jalankan "
                    "scripts/reingest_kb.py.")

        self._ready = True

    def answer(
        self,
        patient_state: Dict[str, Any],
        prediction: float,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        timer: Optional[StageTimer] = None,
    ) -> Dict[str, Any]:
        """Generate a prediction-conditioned clinical advisory.

        The ``prediction`` value (glucose mg/dL from the ML model) is embedded
        into the retrieval query so that both document retrieval and the LLM
        response are conditioned on the numerical forecast — the core novelty
        of this system versus static SPARQL-query approaches.

        ``timer`` mengumpulkan durasi per tahap. Pemanggil dapat meneruskan timer yang
        sudah berisi tahap sebelum pipeline (rekayasa fitur, prediksi, kalibrasi)
        sehingga hasilnya mencakup seluruh jalur, bukan hanya bagian RAG.
        """
        if not self._ready:
            self.build()

        # Instrumentasi waktu per tahap (KNF-10). Timer selalu dibuat, walau pemanggil
        # tidak memberikannya, supaya `timings` pada hasil tidak pernah kosong dan
        # aplikasi tidak perlu menyalakan apa pun untuk mendapat angkanya.
        timer = timer or StageTimer()

        # Prediction-conditioned query: numeric prediction → natural language context
        self._last_llm_context = None
        with timer.measure(STAGE_QUERY):
            user_query = self._build_query(
                patient_state,
                prediction,
                user_question=query,
            )
        k = top_k or self.top_k

        with timer.measure(STAGE_RETRIEVE):
            retrieved_rows = self._retrieve(
                user_query, patient_state=patient_state, top_k=k,
                condition_glucose=float(prediction),
            )
        retrieved_docs = [
            RetrievedDocument(
                rank=row.get("rank", idx + 1),
                text=row.get("text", ""),
                source=row.get("source", "manual_kb"),
                similarity=_opt_similarity(row.get("similarity")),
                metadata=dict(row.get("metadata", {})),
            )
            for idx, row in enumerate(retrieved_rows)
        ]

        with timer.measure(STAGE_GENERATE):
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
                horizon_minutes=self.prediction_horizon_minutes,
                clinical_context=getattr(self, "_last_llm_context", None),
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
            # Apakah jawaban benar-benar ditopang dokumen. UI wajib memakai ini
            # agar tidak menyajikan rekomendasi tanpa rujukan seolah-olah bersumber.
            "grounded": bool(retrieved_docs),
            # Durasi per tahap (detik) + agregat _total/_lokal/_jaringan.
            "timings": timer.as_dict(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        query: str,
        patient_state: Dict[str, Any],
        top_k: int,
        condition_glucose: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Ambil dokumen. ``condition_glucose`` diteruskan eksplisit.

        Jalur produksi adalah prediction-conditioned, sehingga nilai yang diteruskan
        adalah glukosa TERPREDIKSI. Sebelumnya `_enhance_query` mengambil
        `current_glucose` secara implisit, sehingga setiap kueri produksi diam-diam
        membawa kondisi yang sedang berlaku.
        """
        if hasattr(self.retriever, "retrieve_with_context"):
            return self.retriever.retrieve_with_context(
                query, patient_state=patient_state, top_k=top_k,
                condition_glucose=condition_glucose,
            )
        return self.retriever.retrieve(query=query, top_k=top_k)

    def _build_query(
        self,
        patient_state: Dict[str, Any],
        prediction: float,
        user_question: Optional[str] = None,
    ) -> str:
        """Build retrieval query dengan clinical question sebagai anchor.

        Prediction dan patient state tidak dimasukkan sebagai narasi panjang.
        Keduanya hanya digunakan untuk menghasilkan konteks klinis ringkas
        agar tidak menggeser semantic focus dari pertanyaan pengguna.
        """
        try:
            from src.constants import classify_glucose_3class, CLASS_NORMAL

            # --------------------------------------------------------------
            # 1. Clinical question = PRIMARY QUERY
            # --------------------------------------------------------------
            question = (user_question or "").strip()

            if not question:
                question = "Apa tindakan yang sesuai berdasarkan kondisi pasien?"

            # --------------------------------------------------------------
            # 2. Prediction -> clinical condition
            # --------------------------------------------------------------
            condition = classify_glucose_3class(float(prediction))

            context_parts: List[str] = []

            if condition != CLASS_NORMAL:
                context_parts.append(condition)

            # --------------------------------------------------------------
            # 3. Hanya tambahkan konteks klinis yang benar-benar relevan.
            # --------------------------------------------------------------
            if context_parts:
                retrieval_query = (
                    f"{question} "
                    f"Konteks klinis: {', '.join(context_parts)}."
                )
            else:
                retrieval_query = question

            # --------------------------------------------------------------
            # 4. Tetap bangun structured context untuk LLM.
            #    Ini TIDAK dipakai sebagai retrieval query.
            # --------------------------------------------------------------
            try:
                from src.patient_state import PatientState
            except ImportError:
                from ..patient_state import PatientState

            def _opt_float(key: str):
                value = patient_state.get(key)
                return float(value) if value is not None else None

            state = PatientState.from_model_output(
                patient_id=str(
                    patient_state.get("patient_id", "unknown")
                ),
                current_glucose=float(
                    patient_state.get("current_glucose", 100.0)
                ),
                predicted_glucose=float(prediction),
                feature_row=patient_state,
                prediction_horizon_minutes=self.prediction_horizon_minutes,
                predicted_condition=patient_state.get(
                    "predicted_condition"
                ),
                predicted_lower=_opt_float("predicted_lower"),
                predicted_upper=_opt_float("predicted_upper"),
            )

            builder = PredictionConditionedQueryBuilder(
                strategy=QueryStrategy.COMPREHENSIVE
            )

            cq = builder.build(
                state,
                user_question=user_question,
            )

            # Context terstruktur tetap diteruskan ke LLM.
            self._last_llm_context = cq.llm_context

            logger.debug(
                "Retrieval query simplified: %s",
                retrieval_query,
            )

            return retrieval_query

        except Exception as exc:
            logger.warning(
                "Prediction-conditioned query context failed: %s",
                exc,
            )

            # --------------------------------------------------------------
            # Safe fallback:
            # pertanyaan pengguna tetap menjadi query utama.
            # --------------------------------------------------------------
            question = (user_question or "").strip()

            if question:
                return question

            return (
                "Apa tindakan klinis yang sesuai "
                "berdasarkan kondisi pasien?"
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

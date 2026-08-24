from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.prediction_service import PredictionService
from backend.services.supabase_data_service import SupabaseDataService


router = APIRouter(
    prefix="/api/clinical",
    tags=["Clinical Decision Support"],
)


# ============================================================
# Stage 3 — Clinical Decision Support
#
# Flow:
#
# observation
#     ↓
# modality routing
#     ↓
# prediction service
#     ↓
# ┌───────────────────────────────┐
# │ prediction available?         │
# └───────────────┬───────────────┘
#       YES       │       NO
#        ↓        │        ↓
# prediction      │   current-state mode
#        ↓        │        ↓
# prediction-     │   current glucose
# conditioned RAG │   + safety status
#        ↓        │
# clinical advice │
#
# IMPORTANT:
# If prediction is unavailable, this endpoint NEVER treats the
# current glucose as a future prediction. This prevents the
# current-state fallback from corrupting the prediction-
# conditioned retrieval semantics.
# ============================================================


class GlucoseSource(str, Enum):
    CGM = "CGM"
    FINGER_STICK = "FINGER_STICK"


class Observation(BaseModel):
    patient_id: str
    timestamp: datetime
    glucose: float

    insulin: float = 0.0
    carbs: float = 0.0
    activity: float = 0.0
    stress: float = 0.0

    glucose_source: GlucoseSource | None = None


class ClinicalRequest(BaseModel):
    patient_code: str
    question: str = "Apa rekomendasi klinis untuk kondisi pasien saat ini?"
    glucose_source: GlucoseSource = Field(
        default=GlucoseSource.CGM
    )
    data: List[Observation] = Field(default_factory=list)


_prediction_service = None
_rag_pipeline = None
_data_service = None


def get_prediction_service():
    global _prediction_service

    if _prediction_service is None:
        _prediction_service = PredictionService()

    return _prediction_service


def get_rag_pipeline():
    global _rag_pipeline

    if _rag_pipeline is None:
        from src.rag.pipeline import RAGPipeline

        # Seluruh parameter dibiarkan diambil dari config.yaml.
        #
        # SEBELUMNYA tiga nilai dipatri di sini, dan salah satunya menunjuk indeks
        # yang KOSONG: chroma_persist_dir="models/chroma_db_eval" dengan
        # collection_name="diabetes_kb_eval". Koleksi itu tidak pernah ada, sehingga
        # tiap permintaan melewati rantai kegagalan berikut TANPA satu pun galat
        # sampai ke pemanggil:
        #
        #   koleksi tidak ada -> indeks BM25 gagal dibangun
        #   -> mode turun ke vektor -> ruang vektor pun kosong
        #   -> RAGPipeline mundur ke SimpleKeywordRetriever
        #   -> penelusuran dilayani 36 potongan cadangan, bukan 2.233 potongan korpus
        #
        # Jawabannya tetap keluar, sitasinya tetap membawa nomor halaman sungguhan,
        # dan tidak ada yang tampak salah. Hanya 1,6% korpus yang benar-benar
        # terjangkau. Inilah jenis kegagalan yang paling mahal pada alat klinis:
        # sistem yang keliru tanpa terlihat keliru.
        #
        # Nilai produksi ada di config.yaml (rag.persist_dir = models/chroma_db,
        # rag.collection_name = diabetes_kb). Membiarkannya bersumber dari satu
        # tempat mencegah lingkungan penyajian menyimpang diam-diam dari lingkungan
        # yang membangun indeksnya.
        _rag_pipeline = RAGPipeline()

        _rag_pipeline.build()

        _peringatkan_bila_mundur(_rag_pipeline)

    return _rag_pipeline


def _peringatkan_bila_mundur(pipeline) -> None:
    """Catat dengan keras bila penelusuran tidak dilayani korpus penuh.

    Penurunan ke potongan cadangan bersifat SENYAP menurut rancangan — sistem tetap
    menjawab supaya layanan tidak mati total. Justru karena itu ia harus berteriak di
    log, kalau tidak ia hanya akan ketahuan ketika seorang dokter mempertanyakan
    rujukan yang terasa tidak nyambung.
    """
    import logging

    logger = logging.getLogger(__name__)
    retriever = getattr(pipeline, "retriever", None)
    n_bm25 = len(getattr(retriever, "_bm25_teks", []) or [])

    if type(retriever).__name__ == "SimpleKeywordRetriever":
        logger.error(
            "PENELUSURAN TERDEGRADASI: memakai %d potongan cadangan, bukan korpus "
            "penuh. Periksa keberadaan %s dan koleksi %s.",
            len(getattr(retriever, "chunks", []) or []),
            getattr(pipeline, "chroma_persist_dir", "?"),
            getattr(pipeline, "collection_name", "?"),
        )
    elif n_bm25:
        logger.info("Penelusuran siap atas %d potongan korpus.", n_bm25)


def get_data_service():
    global _data_service
    if _data_service is None:
        _data_service = SupabaseDataService()
    return _data_service


def _validate_request(request: ClinicalRequest):
    patient_ids = {
        item.patient_id
        for item in request.data
    }

    if patient_ids and patient_ids != {request.patient_code}:
        raise ValueError(
            "Semua observation harus memiliki patient_code "
            "yang sama dengan request."
        )


def _classify_current_glucose(glucose: float) -> str:
    """
    Runtime-only current-state classification.

    This classification is deliberately separate from the
    prediction pipeline. It describes the latest observed
    glucose, not a future glucose value.
    """

    if glucose < 70:
        return "BAHAYA"
    if glucose > 180:
        return "WASPADA"
    return "AMAN"


def _current_state_advisory(
    current_glucose: float,
    source: str,
    reason: str,
) -> Dict[str, Any]:
    risk = _classify_current_glucose(
        current_glucose
    )

    if risk == "BAHAYA":
        explanation = (
            "Prediksi glukosa belum tersedia karena data historis "
            "belum memenuhi persyaratan model. Nilai glukosa "
            f"terakhir yang teramati adalah {current_glucose:.1f} mg/dL "
            "dan berada pada rentang yang memerlukan perhatian. "
            "Lakukan konfirmasi dengan pengukuran glukosa terkini "
            "dan penilaian klinis sebelum mengambil keputusan terapi."
        )
    elif risk == "WASPADA":
        explanation = (
            "Prediksi glukosa belum tersedia karena data historis "
            "belum memenuhi persyaratan model. Nilai glukosa "
            f"terakhir yang teramati adalah {current_glucose:.1f} mg/dL "
            "dan berada di atas rentang target yang digunakan sistem. "
            "Gunakan informasi ini sebagai kondisi saat ini dan "
            "lakukan penilaian klinis lebih lanjut."
        )
    else:
        explanation = (
            "Prediksi glukosa belum tersedia karena data historis "
            "belum memenuhi persyaratan model. Nilai glukosa "
            f"terakhir yang teramati adalah {current_glucose:.1f} mg/dL "
            "dan tidak menunjukkan kondisi abnormal berdasarkan "
            "klasifikasi sederhana sistem. Kondisi tetap perlu "
            "dinilai bersama konteks klinis pasien."
        )

    return {
        "mode": "current_state",
        "explanation": explanation,
        "risk_level": risk,
        "grounded": False,
        "citations": [],
        "retrieved_docs": [],
        "recommendation_available": False,
        "source": source,
        "reason": reason,
        "disclaimer": (
            "Tidak ada prediksi masa depan yang tersedia. "
            "Keluaran ini bukan diagnosis dan bukan instruksi "
            "perubahan obat/insulin. Keputusan medis final tetap "
            "pada dokter."
        ),
    }


@router.post("")
def clinical_decision_support(
    request: ClinicalRequest,
):
    try:
        _validate_request(request)

        import pandas as pd

        source = request.glucose_source.value
        patient_df, patient, glucose_source = get_data_service().observations(
            request.patient_code,
            source,
        )

        if not patient:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {request.patient_code} tidak ditemukan.",
            )

        if not glucose_source:
            return {
                "status": "NO_DATA",
                "patient_code": request.patient_code,
                "glucose_source": source,
                "mode": "current_state",
                "prediction": {
                    "status": "NO_DATA",
                    "prediction_available": False,
                    "current_glucose": None,
                    "horizons": [],
                    "condition": None,
                    "reason": (
                        f"Belum ada data {source} untuk pasien ini."
                    ),
                },
                "clinical_advisory": {
                    "mode": "current_state",
                    "explanation": (
                        f"Belum ada data {source} untuk pasien ini. "
                        "Input data melalui Logbook terlebih dahulu."
                    ),
                    "risk_level": "NO_DATA",
                    "grounded": False,
                    "citations": [],
                    "retrieved_docs": [],
                    "recommendation_available": False,
                    "source": source,
                    "disclaimer": (
                        "Keluaran ini adalah clinical decision support, "
                        "bukan diagnosis atau instruksi terapi otomatis. "
                        "Keputusan klinis final tetap pada dokter."
                    ),
                },
            }

        # -----------------------------------------------------
        # 1. Modality-aware prediction
        # -----------------------------------------------------

        prediction_result = (
            get_prediction_service().predict(
                patient_df,
                glucose_source=source,
            )
        )
        prediction_result["glucose_source"] = source

        last = patient_df.sort_values("timestamp").iloc[-1]

        patient_state = {
            "patient_code": request.patient_code,
            "current_glucose": float(last.glucose),
            "stress_level": float(last.stress),
            "activity_level": float(last.activity),
            "insulin_on_board": float(last.insulin),
            "carbs_on_board": float(last.carbs),
        }

        prediction_available = bool(
            prediction_result.get(
                "prediction_available",
                False,
            )
        )

        # -----------------------------------------------------
        # 2A. No sufficient history
        #
        # Do NOT feed current_glucose into rag.answer()
        # because rag.answer() interprets its `prediction`
        # argument as a future predicted value.
        # -----------------------------------------------------

        if not prediction_available:
            current_glucose = prediction_result.get(
                "current_glucose"
            )

            if current_glucose is None:
                raise ValueError(
                    "Tidak ada glukosa valid untuk current-state assessment."
                )

            advisory = _current_state_advisory(
                current_glucose=float(
                    current_glucose
                ),
                source=source,
                reason=str(
                    prediction_result.get(
                        "reason",
                        "Prediction unavailable.",
                    )
                ),
            )

            return {
                "status": prediction_result.get(
                    "status",
                    "INSUFFICIENT_HISTORY",
                ),
                "patient_code": request.patient_code,
                "glucose_source": source,
                "mode": "current_state",
                "prediction": prediction_result,
                "clinical_advisory": advisory,
            }

        # -----------------------------------------------------
        # 2B. Prediction available
        # -----------------------------------------------------

        horizons = prediction_result.get(
            "horizons",
            [],
        )

        # Select the primary horizon returned by the
        # modality-specific prediction service.
        primary = (
            horizons[0]
            if horizons
            else None
        )

        if primary is None:
            raise ValueError(
                "Prediction tersedia tetapi horizon prediksi kosong."
            )

        predicted_glucose = float(
            primary["prediction"]
        )

        prediction_horizon = int(
            primary["horizon_minutes"]
        )

        patient_state[
            "prediction_horizon_minutes"
        ] = prediction_horizon

        # -----------------------------------------------------
        # 3. Prediction-conditioned clinical RAG
        # -----------------------------------------------------

        rag = get_rag_pipeline()

        rag_result = rag.answer(
            patient_state=patient_state,
            prediction=predicted_glucose,
            query=request.question,
            top_k=5,
        )

        get_data_service().save_assessment(
            patient_id=patient["id"],
            glucose_source_id=glucose_source["id"],
            assessment={
                "current_glucose": float(last.glucose),
                "prediction_30m": prediction_result.get("prediction_30m"),
                "prediction_60m": prediction_result.get("prediction_60m"),
                "condition": prediction_result.get("condition"),
                "recommendation": rag_result["explanation"],
                "grounded": bool(rag_result.get("grounded", False)),
                "sources": rag_result.get("citations", []),
            },
        )

        return {
            "status": "success",
            "patient_code": request.patient_code,
            "glucose_source": source,
            "mode": "prediction",
            "prediction": prediction_result,
            "clinical_advisory": {
                "mode": "prediction_conditioned",
                "explanation": rag_result[
                    "explanation"
                ],
                "risk_level": rag_result[
                    "risk_level"
                ],
                "grounded": rag_result[
                    "grounded"
                ],
                # False bila narasinya dari templat karena model bahasa tidak
                # tersedia. Rujukannya tetap sah dan tetap ditampilkan; yang tidak
                # sah adalah membiarkan dokter mengira teksnya hasil penalaran LLM.
                "narasi_llm": bool(
                    rag_result.get("narasi_llm", True)
                ),
                "citations": rag_result.get(
                    "citations",
                    [],
                ),
                "retrieved_docs": rag_result.get(
                    "retrieved_docs",
                    [],
                ),
                "prediction_horizon_minutes":
                    prediction_horizon,
                "prediction": predicted_glucose,
                "recommendation_available": bool(
                    rag_result.get(
                        "grounded",
                        False,
                    )
                ),
                "disclaimer": (
                    "Prediksi dan rekomendasi bersifat "
                    "decision support. Bukan diagnosis atau "
                    "instruksi terapi otomatis. Keputusan medis "
                    "final tetap pada dokter."
                ),
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Clinical decision support failed: "
                f"{exc}"
            ),
        ) from exc
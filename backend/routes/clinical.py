import threading
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


# Stage 3 — Clinical Decision Support
#
# PENTING: bila prediksi tidak tersedia, endpoint ini TIDAK PERNAH
# memperlakukan glukosa terkini sebagai prediksi. Tanpa pemisahan itu, mode
# cadangan akan merusak semantik penelusuran terkondisi-prediksi.


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

    # `stress` DICABUT SEPENUHNYA. Ia tidak pernah menjadi fitur model (tidak ada
    # pada config.model.engineered_features), dan kanal `stressors` OhioT1DM hanya
    # memuat 7 event di seluruh 12 pasien sehingga inheren tak informatif. Meminta
    # dokter mengisinya membuang waktu mereka; lebih buruk lagi, angka bawaan 5
    # sempat dicetak balik sebagai "Tingkat stres: 5/10" seolah fakta terukur
    # (temuan T5). Medannya sempat dipertahankan demi klien lama, lalu dicabut
    # bersama tabel `stress_events`.

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

# Satu-satunya pembangun yang boleh berjalan pada satu waktu.
#
# Tanpa kunci ini, permintaan yang datang saat pemanasan belum tuntas akan
# memakai pipeline setengah jadi, lalu RAGPipeline.answer() membangunnya
# ulang. Dua indeks BM25 atas 2.233 potongan hidup bersamaan pada instans
# 512 MB dan prosesnya dibunuh. Kunci ini beserta pola "bangun dulu,
# publikasikan kemudian" di bawah mencegahnya.
_kunci_bangun = threading.Lock()

# Kunci TERPISAH untuk layanan prediksi. Kalau ia ikut memakai _kunci_bangun, sebuah
# permintaan akan terblokir di belakang pembangunan indeks RAG yang memakan menit,
# padahal artefak prediksi hanya beberapa ratus kilobyte dan siap dalam hitungan
# detik. Menunggu sesuatu yang tidak ada hubungannya persis cara membuat batas waktu
# Render terlampaui, dan itu memunculkan kembali 502 yang sedang diperbaiki.
_kunci_prediksi = threading.Lock()

# Berapa lama permintaan bersedia menunggu pemanasan selesai sebelum menyerah.
# Lebih pendek daripada batas waktu permintaan Render, supaya yang diterima
# dokter adalah pesan yang menjelaskan keadaan, bukan 502 tanpa keterangan.
_BATAS_TUNGGU_DETIK = 75


def get_prediction_service():
    global _prediction_service

    with _kunci_prediksi:
        if _prediction_service is None:
            _prediction_service = PredictionService()

    return _prediction_service


class SedangDisiapkan(Exception):
    """Pemanasan belum selesai dan penantiannya sudah melewati batas."""


def get_rag_pipeline():
    global _rag_pipeline

    if _rag_pipeline is not None:
        return _rag_pipeline

    # MENUNGGU, BUKAN MEMBANGUN SENDIRI. Bila pemanasan sedang berjalan, kunci ini
    # dipegang thread pemanasan; permintaan menunggu sampai ia selesai lalu memakai
    # pipeline yang sama. Membangun sendiri secara paralel adalah yang membunuh
    # instans 512 MB pada deploy pertama.
    if not _kunci_bangun.acquire(timeout=_BATAS_TUNGGU_DETIK):
        raise SedangDisiapkan(
            "Sistem masih menyiapkan basis pengetahuan (indeks penelusuran atas "
            "2.233 potongan pedoman). Proses ini hanya berjalan sekali setelah "
            "server dinyalakan. Silakan coba lagi satu sampai dua menit lagi."
        )

    try:
        if _rag_pipeline is not None:
            return _rag_pipeline

        from src.rag.pipeline import RAGPipeline

        # Seluruh parameter diambil dari config.yaml, tidak dipatri di sini.
        #
        # Nilai yang dipatri pernah menunjuk koleksi yang tidak ada, dan
        # rantai cadangannya menurunkan penelusuran ke 36 potongan tanpa satu
        # pun galat sampai ke pemanggil. Bersumber satu tempat mencegah
        # lingkungan penyajian menyimpang diam-diam dari lingkungan yang
        # membangun indeksnya.
        # DIBANGUN KE VARIABEL LOKAL DULU. Menetapkan global sebelum .build()
        # selesai membuat pemanggil lain menerima pipeline setengah jadi, dan
        # RAGPipeline.answer() akan membangunnya ulang — dua indeks sekaligus di
        # instans 512 MB. Globalnya baru dipublikasikan setelah benar-benar siap.
        pipeline = RAGPipeline()

        pipeline.build()

        _peringatkan_bila_mundur(pipeline)

        _rag_pipeline = pipeline

        return _rag_pipeline

    finally:
        _kunci_bangun.release()


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

        # 1. Modality-aware prediction

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
            "activity_level": float(last.activity),
            # Nilai MENTAH dari observasi terakhir. Keduanya hanya cadangan: bila
            # prediksi tersedia, keduanya DIGANTI di bawah oleh iob/cob terekayasa
            # yang sudah meluruh menurut waktu. Lihat blok "fitur terekayasa".
            "insulin_on_board": float(last.insulin),
            "carbs_on_board": float(last.carbs),
        }

        prediction_available = bool(
            prediction_result.get(
                "prediction_available",
                False,
            )
        )

        # 2A. No sufficient history
        #
        # Do NOT feed current_glucose into rag.answer()
        # because rag.answer() interprets its `prediction`
        # argument as a future predicted value.

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

        # 2B. Prediction available

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

        # 2C. Salurkan seluruh keluaran prediktor ke RAG.
        #
        # condition dan interval keduanya terpakai di hilir: regresi menyusut
        # ke tengah dan jarang melewati ambang 70/180, sedangkan interval
        # konformal menghidupkan pengondisian sadar-ketidakpastian pada
        # _primary_query. Membuangnya di sini membuat advisory dangkal.

        patient_state["predicted_condition"] = prediction_result.get("condition")

        interval = primary.get("interval")

        if interval and len(interval) == 2:
            patient_state["predicted_lower"] = float(interval[0])
            patient_state["predicted_upper"] = float(interval[1])

        # Fitur terekayasa: iob/cob yang MELURUH menurut waktu, plus laju perubahan
        # glukosa dan jarak ke pengukuran sebelumnya. Inilah vektor yang benar-benar
        # dilihat model, jadi inilah pula yang harus dilihat model bahasa. Kolom
        # mentah `last.insulin` di atas hanya dipakai bila prediktor tidak
        # mengembalikan fitur terekayasa sama sekali.
        engineered = prediction_result.get("engineered_last") or {}

        if "iob" in engineered:
            patient_state["insulin_on_board"] = engineered["iob"]
        if "cob" in engineered:
            patient_state["carbs_on_board"] = engineered["cob"]

        for nama in ("glucose_rate", "time_since_prev_glucose"):
            if nama in engineered:
                patient_state[nama] = engineered[nama]

        # 3. Prediction-conditioned clinical RAG

        rag = get_rag_pipeline()

        # top_k dibiarkan bersumber dari config.yaml (rag.top_k_retrieval), sama
        # seperti seluruh parameter pipeline lainnya. Nilai 5 yang dipatri di sini
        # membuat config diam-diam tidak berlaku pada satu-satunya jalur yang
        # benar-benar dipakai dokter.
        rag_result = rag.answer(
            patient_state=patient_state,
            prediction=predicted_glucose,
            query=request.question,
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
                # Pemeriksaan deterministik: tiap ANGKA yang diberi penanda [S..]
                # dicek apakah benar ada pada potongan yang ditunjuknya. Sudah
                # dihitung pipeline sejak 24 Agustus tetapi belum pernah sampai ke
                # antarmuka, sehingga hasilnya tidak pernah terlihat siapa pun.
                "verifikasi_sitasi": rag_result.get(
                    "verifikasi_sitasi",
                    None,
                ),
                # Durasi per tahap. Dipakai antarmuka untuk menunjukkan bahwa
                # penantian sedang mengerjakan sesuatu, bukan menggantung.
                "timings": rag_result.get(
                    "timings",
                    None,
                ),
                "prediction_horizon_minutes":
                    prediction_horizon,
                "prediction": predicted_glucose,
                # Kelas dari pengklasifikasi tiga kelas. Ditampilkan sebagai label
                # utama panel advisory, sejajar dengan angka mg/dL: kondisi adalah
                # keluaran yang paling didukung data ini, sedangkan nilai regresi
                # membawa ketidakpastian yang jauh lebih besar.
                "predicted_condition": prediction_result.get("condition"),
                "prediction_interval": primary.get("interval"),
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

    except SedangDisiapkan as exc:
        # 503, BUKAN 500. Ini keadaan sementara yang akan hilang sendiri, dan
        # pesannya memberi tahu dokter persis apa yang harus dilakukan: tunggu
        # sebentar lalu ulangi. Sebelumnya keadaan ini muncul sebagai 502 tanpa
        # keterangan apa pun karena prosesnya keburu mati.
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

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
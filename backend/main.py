import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.prediction import router as prediction_router
from backend.routes.clinical import router as clinical_router
from backend.routes.logbook import router as logbook_router
from backend.routes.patients import router as patients_router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Pemanasan saat startup
# ---------------------------------------------------------
# SEBELUMNYA seluruh pembangunan berat terjadi DI DALAM permintaan HTTP pertama:
# klien ChromaDB, indeks BM25 atas 2.233 potongan, pemuatan artefak model. Dokter
# yang menekan "Run Clinical Assessment" pertama kali menanggung seluruh biaya itu
# di atas waktu penalaran LLM, dan itulah sebagian besar keluhan "assessment lama".
#
# Pemanasan dijalankan sebagai task LATAR, bukan ditunggu. Alasannya operasional:
# Render menganggap instans gagal bila port tidak dibuka dalam batas waktunya. Bila
# pemanasan ditunggu di dalam lifespan, startup yang lambat berubah menjadi deploy
# yang gagal. Dengan task latar, port terbuka seketika.
#
# Permintaan yang datang SEBELUM pemanasan tuntas akan MENUNGGU pekerjaan yang sama
# itu, bukan memulai pekerjaan kedua. Pembedaan ini bukan detail: pada deploy
# pertama, permintaan yang membangun sendiri secara paralel membuat dua indeks BM25
# atas 2.233 potongan hidup bersamaan di instans 512 MB, dan prosesnya dibunuh
# berulang kali. Penjaganya ada di backend/routes/clinical.py (_kunci_bangun).
_status_pemanasan: dict = {"selesai": False, "berjalan": False, "galat": None}


def _panaskan() -> None:
    from backend.routes.clinical import get_prediction_service, get_rag_pipeline

    _status_pemanasan["berjalan"] = True

    try:
        get_prediction_service()
        get_rag_pipeline()
        _status_pemanasan["selesai"] = True
        logger.info("Pemanasan selesai: pipeline RAG dan layanan prediksi siap.")
    except Exception as exc:  # noqa: BLE001
        # Kegagalan pemanasan TIDAK boleh mematikan proses. Permintaan berikutnya
        # akan mencoba membangun ulang dan memunculkan galat yang sebenarnya.
        _status_pemanasan["galat"] = str(exc)
        logger.exception("Pemanasan gagal; pembangunan ditunda ke permintaan pertama.")

    finally:
        _status_pemanasan["berjalan"] = False


# Jeda sebelum pemanasan dimulai, dalam detik.
#
# SEBAB (24 Agustus 2026, deploy kedua). Render menjalankan instans LAMA dan BARU
# bersamaan demi zero-downtime: instans lama tetap melayani sampai yang baru lolos
# health check. Dua proses Python dengan langchain dan chromadb termuat sudah mepet
# di 512 MB. Pemanasan yang mulai seketika membuat instans baru merebut seluruh
# memori kerjanya TEPAT di jendela tumpang-tindih itu, dan Render melaporkan
# "Ran out of memory (used over 512MB)" satu menit setelah tiap deploy dimulai.
#
# Menunda pemanasan membuat instans baru boot RINGAN, lolos health check, instans
# lama berhenti, dan barulah memori dialokasikan — berurutan, bukan berbarengan.
#
# Menundanya aman karena _kunci_bangun sudah membuat permintaan yang datang lebih
# awal MENUNGGU pembangunan yang sama, bukan memulai yang kedua. Tanpa kunci itu,
# jeda ini justru akan memindahkan pembangunan ke thread permintaan.
_JEDA_PEMANASAN_DETIK = int(os.getenv("JEDA_PEMANASAN_DETIK", "90"))


async def _panaskan_setelah_jeda() -> None:
    if _JEDA_PEMANASAN_DETIK > 0:
        logger.info(
            "Pemanasan ditunda %d detik agar tidak berimpit dengan instans lama "
            "yang masih melayani selama deploy.", _JEDA_PEMANASAN_DETIK)
        await asyncio.sleep(_JEDA_PEMANASAN_DETIK)

    await asyncio.to_thread(_panaskan)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tugas = asyncio.create_task(_panaskan_setelah_jeda())
    try:
        yield
    finally:
        # Instans yang sedang dimatikan tidak boleh meneruskan pemanasan: memorinya
        # justru dibutuhkan instans penggantinya.
        tugas.cancel()


app = FastAPI(
    title="Diabetes Clinical Decision Support API",
    version="1.0.0",
    description="Backend API for glucose prediction and clinical recommendation.",
    lifespan=lifespan,
)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
# Development: Next.js biasanya berjalan di localhost:3000.
# Nanti origin production diganti sesuai domain PWA.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://diabetes-glucose-monitor-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)
app.include_router(clinical_router)
app.include_router(logbook_router)
app.include_router(patients_router)


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
# DIPERLUAS 24 Agustus 2026. Versi sebelumnya hanya menjawab {"status": "ok"},
# dan justru itu yang membuat T1 lolos berminggu-minggu: GOOGLE_API_KEY tidak
# pernah dipasang di Render, advisory diam-diam dilayani templat if/else, dan
# health check tetap hijau. Endpoint ini sekarang melaporkan tiga hal yang bila
# salah membuat sistem KELIRU TANPA TERLIHAT KELIRU:
#
#   llm       - rantai LLM benar-benar terbangun, atau jawaban akan berupa templat
#   retrieval - dilayani korpus penuh, atau mundur ke potongan cadangan (1,6% korpus)
#   pemanasan - apakah biaya pembangunan sudah dibayar di muka
#
# Ia sengaja TIDAK memanggil LLM dan TIDAK membangun apa pun: memeriksa keadaan
# tidak boleh mengubah keadaan, dan health check yang mahal akan ikut menghabiskan
# memori instans 512 MB yang justru sedang dijaga.
@app.get("/health")
def health_check():
    from backend.routes import clinical

    pipeline = clinical._rag_pipeline
    retriever = getattr(pipeline, "retriever", None) if pipeline else None

    if pipeline is None:
        # Globalnya SENGAJA baru dipublikasikan setelah .build() tuntas, jadi None
        # di sini berarti "belum siap" — bukan lagi "mungkin setengah jadi".
        retrieval = {
            "siap": False,
            "keterangan": (
                "sedang dibangun"
                if _status_pemanasan.get("berjalan")
                else "belum dibangun"
            ),
        }
    else:
        nama = type(retriever).__name__
        mundur = nama == "SimpleKeywordRetriever"
        retrieval = {
            "siap": retriever is not None,
            "penelusur": nama,
            "n_potongan": (
                len(getattr(retriever, "chunks", []) or [])
                if mundur
                else len(getattr(retriever, "_bm25_teks", []) or [])
            ),
            # True berarti penelusuran dilayani potongan cadangan, bukan korpus penuh.
            "terdegradasi": mundur,
        }

    generator = getattr(pipeline, "generator", None) if pipeline else None
    chain = getattr(generator, "chain", None) if generator else None

    return {
        "status": "ok",
        "service": "diabetes-cdss-api",
        "pemanasan": dict(_status_pemanasan),
        "llm": {
            "provider": getattr(pipeline, "llm_provider", None) if pipeline else None,
            "model": getattr(pipeline, "gemini_model", None) if pipeline else None,
            # False berarti advisory akan dijawab TEMPLAT, bukan penalaran LLM.
            "siap": bool(getattr(chain, "is_ready", False)) if chain else False,
            "galat_init": getattr(chain, "_init_error", None) if chain else None,
        },
        "retrieval": retrieval,
    }


@app.get("/")
def root():
    return {
        "message": "Diabetes Clinical Decision Support API",
        "status": "running",
    }

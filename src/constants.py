"""Ambang glikemik dan klasifikasi kondisi — SATU SUMBER KEBENARAN.

Sebelum Tugas 3, ambang tersebar di lima berkas dengan dua definisi berbeda:
src/patient_state.py memakai 5 zona (54/70/180/250), sedangkan app/ui.py,
src/rag/pipeline.py, dan scripts/train_condition_classifier.py memakai 3 zona
(70/180). Angka apa pun di sini TIDAK BOLEH ditulis ulang di berkas lain.

Dua fungsi klasifikasi, satu set ambang:

- ``classify_glucose_5zone``  — untuk LABEL RISIKO dan tampilan. Membedakan
  kondisi kritis (<54, >250) yang menuntut urgensi berbeda kepada dokter.
- ``classify_glucose_3class`` — untuk PENGKLASIFIKASI KONDISI terlatih dan
  pengondisian kueri RAG.

Mengapa pengklasifikasi tetap 3 kelas (keputusan Tugas 3): pada data latih hanya
ada 998 sampel hipoglikemia berat (0,72%) dan 157 pada data uji, sehingga estimasi
sensitivitasnya sangat tidak stabil. Lebih penting lagi, memecah hipoglikemia di
ambang 54 tidak mengubah tindakan klinis — keduanya sama-sama menuntut pemberian
karbohidrat. Ambang 54/250 tetap dipakai untuk menaikkan urgensi yang ditampilkan,
bukan untuk melatih model.
"""

from __future__ import annotations

from typing import Final

# ──────────────────────────────────────────────────────────────
# Ambang glikemik (mg/dL) — ADA Standards of Care
# ──────────────────────────────────────────────────────────────

GLUCOSE_CRITICAL_LOW: Final[float] = 54.0    # hipoglikemia berat (ADA Level 2)
GLUCOSE_LOW: Final[float] = 70.0             # ambang hipoglikemia (ADA Level 1)
GLUCOSE_HIGH: Final[float] = 180.0           # ambang hiperglikemia (post-prandial)
GLUCOSE_CRITICAL_HIGH: Final[float] = 250.0  # hiperglikemia berat

# Batas fisiologis untuk clamping masukan
GLUCOSE_MIN_PHYSIOLOGICAL: Final[float] = 20.0
GLUCOSE_MAX_PHYSIOLOGICAL: Final[float] = 600.0

# ──────────────────────────────────────────────────────────────
# Ambang tren (mg/dL, selisih sepanjang horizon prediksi)
# ──────────────────────────────────────────────────────────────

TREND_STABLE_THRESHOLD_MGDL: Final[float] = 10.0  # |delta| < 10  -> stabil
TREND_RAPID_THRESHOLD_MGDL: Final[float] = 30.0   # |delta| >= 30 -> cepat

# ──────────────────────────────────────────────────────────────
# Ambang Clarke Error Grid — SENGAJA TERPISAH
# ──────────────────────────────────────────────────────────────
# Clarke Error Grid adalah metrik terbitan dengan batas zona yang sudah baku.
# Angkanya kebetulan sama dengan ambang klinis di atas, tetapi TIDAK BOLEH ikut
# berubah bila kebijakan risiko diubah — kalau ikut berubah, metriknya berhenti
# menjadi Clarke Error Grid dan seluruh perbandingan dengan literatur batal.
CLARKE_LOW: Final[float] = 70.0
CLARKE_HIGH: Final[float] = 180.0

# ──────────────────────────────────────────────────────────────
# Nama kondisi
# ──────────────────────────────────────────────────────────────

RISK_CRITICAL_HYPO: Final[str] = "critical_hypoglycemia"
RISK_HYPO: Final[str] = "hypoglycemia"
RISK_NORMAL: Final[str] = "normal"
RISK_HYPER: Final[str] = "hyperglycemia"
RISK_CRITICAL_HYPER: Final[str] = "critical_hyperglycemia"

# Tiga kelas untuk pengklasifikasi terlatih (label Bahasa Indonesia)
CLASS_HYPO: Final[str] = "hipoglikemia"
CLASS_NORMAL: Final[str] = "normal"
CLASS_HYPER: Final[str] = "hiperglikemia"
CONDITION_CLASSES: Final[list[str]] = [CLASS_HYPO, CLASS_NORMAL, CLASS_HYPER]

_RISK_LABELS_ID: Final[dict[str, str]] = {
    RISK_CRITICAL_HYPO: "BAHAYA - Hipoglikemia Berat",
    RISK_HYPO: "BAHAYA - Hipoglikemia",
    RISK_NORMAL: "AMAN",
    RISK_HYPER: "HATI-HATI - Hiperglikemia",
    RISK_CRITICAL_HYPER: "BAHAYA - Hiperglikemia Berat",
}

# Pemetaan kelas pengklasifikasi (Indonesia) -> nama kondisi (Inggris)
_CLASS_TO_RISK: Final[dict[str, str]] = {
    CLASS_HYPO: RISK_HYPO,
    CLASS_NORMAL: RISK_NORMAL,
    CLASS_HYPER: RISK_HYPER,
}


# ──────────────────────────────────────────────────────────────
# Fungsi klasifikasi
# ──────────────────────────────────────────────────────────────

def classify_glucose_5zone(glucose: float) -> str:
    """Kondisi 5 zona untuk label risiko dan tampilan.

    Mengembalikan salah satu dari RISK_CRITICAL_HYPO, RISK_HYPO, RISK_NORMAL,
    RISK_HYPER, RISK_CRITICAL_HYPER.
    """
    g = float(glucose)
    if g < GLUCOSE_CRITICAL_LOW:
        return RISK_CRITICAL_HYPO
    if g < GLUCOSE_LOW:
        return RISK_HYPO
    if g > GLUCOSE_CRITICAL_HIGH:
        return RISK_CRITICAL_HYPER
    if g > GLUCOSE_HIGH:
        return RISK_HYPER
    return RISK_NORMAL


def classify_glucose_3class(glucose: float) -> str:
    """Kondisi 3 kelas untuk pengklasifikasi terlatih dan pengondisian kueri RAG.

    Mengembalikan CLASS_HYPO, CLASS_NORMAL, atau CLASS_HYPER (Bahasa Indonesia).
    """
    g = float(glucose)
    if g < GLUCOSE_LOW:
        return CLASS_HYPO
    if g > GLUCOSE_HIGH:
        return CLASS_HYPER
    return CLASS_NORMAL


def risk_label_id(risk_level: str) -> str:
    """Label Bahasa Indonesia untuk nama kondisi 5 zona."""
    return _RISK_LABELS_ID.get(risk_level, "AMAN")


def risk_from_condition_class(condition_class: str) -> str:
    """Ubah kelas pengklasifikasi (Indonesia) menjadi nama kondisi (Inggris)."""
    return _CLASS_TO_RISK.get(condition_class, condition_class)


def is_critical(risk_level: str) -> bool:
    """Apakah kondisi termasuk kritis (menuntut urgensi tertinggi)."""
    return risk_level in (RISK_CRITICAL_HYPO, RISK_CRITICAL_HYPER)


def base_condition(risk_level: str) -> str:
    """Buang awalan 'critical_' sehingga kondisi kritis dan biasa dapat dibandingkan."""
    return risk_level.replace("critical_", "")

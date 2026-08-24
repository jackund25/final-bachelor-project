"""PatientState — canonical bridge between ML model output and RAG pipeline.

Converts raw model predictions + feature windows into a structured, clinically
meaningful state object. This is the data contract between the prediction layer
and the RAG advisory layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Ambang klinis berasal dari SATU sumber kebenaran (src/constants.py).
# Jangan menulis ulang angkanya di berkas ini.
from src.constants import (
    GLUCOSE_HIGH,
    GLUCOSE_LOW,
    GLUCOSE_MAX_PHYSIOLOGICAL,
    GLUCOSE_MIN_PHYSIOLOGICAL,
    RISK_HYPER,
    RISK_HYPO,
    RISK_NORMAL,
    TREND_RAPID_THRESHOLD_MGDL,
    TREND_STABLE_THRESHOLD_MGDL,
    base_condition,
    classify_glucose_5zone,
    is_critical,
    risk_label_id,
)


# ──────────────────────────────────────────────────────────────
# Core dataclass
# ──────────────────────────────────────────────────────────────

def _opsional(nilai) -> Optional[float]:
    """float(nilai), atau None bila tidak ada / tidak dapat dibaca.

    Nilai yang hilang TIDAK diganti 0.0. Nol adalah laju yang sah (glukosa datar),
    sehingga memakainya sebagai penanda "tidak diketahui" membuat kedua keadaan itu
    tidak dapat dibedakan lagi di hilir - mekanisme yang sama dengan temuan T4.
    """
    if nilai is None:
        return None
    try:
        return float(nilai)
    except (TypeError, ValueError):
        return None


@dataclass
class PatientState:
    """Canonical patient state derived from ML model output.

    Acts as the data contract fed to the RAG pipeline's query builder.
    All numeric fields are typed and validated on construction.

    Example usage (from model prediction)::

        state = PatientState.from_model_output(
            patient_id="ohio_559",
            current_glucose=180.0,
            predicted_glucose=195.0,
            feature_row={"insulin": 0.5, "carbs": 30.0, "activity": 0},
        )
        print(state.risk_level)     # "hyperglycemia"
        print(state.trend_label)    # "rising"
        print(state.urgency)        # "high"
    """

    # ── Identity ──────────────────────────────────────────────
    patient_id: str

    # ── Core glucose values ───────────────────────────────────
    current_glucose: float
    predicted_glucose: float
    prediction_horizon_minutes: int = 60

    # ── Active pharmacokinetics ───────────────────────────────
    insulin_on_board: float = 0.0   # units
    carbs_on_board: float = 0.0     # grams

    # ── Lifestyle factors ─────────────────────────────────────
    # SKOR INTENSITAS aktivitas dari kanal `exercise` OhioT1DM (atribut `intensity`,
    # skala ordinal), BUKAN menit. Atribut `duration` tidak diekstrak parser.
    activity_level: int = 0
    # `stress_level` DICABUT 24 Agustus 2026, bersama penanda `stress_diketahui`
    # yang sempat menambalnya. Menandai "tidak diukur" tetap lebih baik daripada
    # mengarang angka, tetapi medan yang TIDAK PERNAH terisi pada jalur mana pun
    # lebih baik tidak ada sama sekali: `stress` bukan bagian dari
    # config.model.engineered_features, dan kanal `stressors` OhioT1DM hanya memuat
    # 7 event di seluruh 12 pasien. Lihat src/data/ohio_parser.py.

    # ── Kondisi masa depan hasil pengklasifikasi (opsional) ────
    # Regresi yang meminimalkan galat kuadrat menyusut ke tengah, sehingga jarang berani
    # melewati ambang 70/180 dan gagal menandai perubahan kondisi (lihat Bab VI). Bila
    # tersedia, kondisi dari pengklasifikasi tiga kelas dipakai sebagai pengganti ambang
    # atas nilai regresi.
    predicted_condition: Optional[str] = None      # "hypoglycemia" | "normal" | "hyperglycemia"

    # ── Batas interval prediksi konformal (opsional) ───────────
    # Dipakai untuk pengondisian kueri yang sadar-ketidakpastian: kondisi berisiko yang
    # tercakup interval tetap diambilkan dokumennya meski prediksi titiknya masih normal.
    # Sinyal laju dari jendela terekayasa (opsional).
    # Keduanya SUDAH dihitung preprocessor dan ikut masuk vektor fitur model, tetapi
    # tidak pernah diteruskan ke model bahasa. `time_since_prev_glucose` penting
    # khusus pada modalitas finger-stick: prediksi dari pengukuran empat jam lalu
    # menanggung ketidakpastian yang sama sekali berbeda dari CGM lima menit lalu,
    # dan itu justru bahan bagi bagian "Yang tidak dapat disimpulkan".
    glucose_rate: Optional[float] = None             # mg/dL per menit
    time_since_prev_glucose: Optional[float] = None  # menit

    predicted_lower: Optional[float] = None        # mg/dL
    predicted_upper: Optional[float] = None        # mg/dL

    # ── Timestamp ─────────────────────────────────────────────
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # ── Derived (computed in __post_init__) ───────────────────
    anticipated_conditions: List[str] = field(init=False)  # kondisi yang perlu diantisipasi
    glucose_delta: float = field(init=False)
    trend_direction: str = field(init=False)   # "rising" | "stable" | "falling"
    trend_label: str = field(init=False)       # human-readable Indonesian label
    trend_rate: str = field(init=False)        # "rapid" | "moderate" | "slow"
    risk_level: str = field(init=False)        # "hypoglycemia" | "normal" | "hyperglycemia"
    risk_label: str = field(init=False)        # Indonesian clinical label
    urgency: str = field(init=False)           # "critical" | "high" | "medium" | "low"

    def __post_init__(self) -> None:
        # Clamp inputs to physiological range
        self.current_glucose = float(
            np.clip(self.current_glucose, GLUCOSE_MIN_PHYSIOLOGICAL, GLUCOSE_MAX_PHYSIOLOGICAL)
        )
        self.predicted_glucose = float(
            np.clip(self.predicted_glucose, GLUCOSE_MIN_PHYSIOLOGICAL, GLUCOSE_MAX_PHYSIOLOGICAL)
        )
        self.insulin_on_board = max(0.0, float(self.insulin_on_board))
        self.carbs_on_board = max(0.0, float(self.carbs_on_board))
        self.activity_level = max(0, int(self.activity_level))

        # Trend
        self.glucose_delta = round(self.predicted_glucose - self.current_glucose, 2)
        if abs(self.glucose_delta) < TREND_STABLE_THRESHOLD_MGDL:
            self.trend_direction = "stable"
            self.trend_label = "stabil"
        elif self.glucose_delta > 0:
            self.trend_direction = "rising"
            self.trend_label = "meningkat"
        else:
            self.trend_direction = "falling"
            self.trend_label = "menurun"

        if abs(self.glucose_delta) >= TREND_RAPID_THRESHOLD_MGDL:
            self.trend_rate = "rapid"
        elif abs(self.glucose_delta) >= TREND_STABLE_THRESHOLD_MGDL:
            self.trend_rate = "moderate"
        else:
            self.trend_rate = "slow"

        # Risk based on PREDICTED glucose (not current) — this is the novelty:
        # interventions are chosen based on WHERE the patient is GOING, not where they are now
        self.risk_level = classify_glucose_5zone(self.predicted_glucose)
        self.risk_label = risk_label_id(self.risk_level)

        # Bila pengklasifikasi kondisi tersedia, ia MENGGANTIKAN kondisi hasil pengambangan
        # nilai regresi — kecuali regresi sudah menandai kondisi kritis, yang tetap dihormati.
        if self.predicted_condition and not is_critical(self.risk_level):
            if self.predicted_condition in (RISK_HYPO, RISK_HYPER, RISK_NORMAL):
                self.risk_level = self.predicted_condition
                self.risk_label = risk_label_id(self.risk_level)

        # Kondisi yang perlu diantisipasi: kondisi terprediksi, DITAMBAH kondisi berisiko
        # yang masih tercakup interval ketidakpastian meski prediksi titiknya normal.
        conditions = [base_condition(self.risk_level)]
        if self.predicted_lower is not None and self.predicted_lower < GLUCOSE_LOW:
            conditions.append(RISK_HYPO)
        if self.predicted_upper is not None and self.predicted_upper > GLUCOSE_HIGH:
            conditions.append(RISK_HYPER)
        # dahulukan kondisi berisiko, buang duplikat, pertahankan urutan
        priority = {RISK_HYPO: 0, RISK_HYPER: 1, RISK_NORMAL: 2}
        self.anticipated_conditions = sorted(set(conditions), key=lambda c: priority.get(c, 3))

        # Urgency.
        # Cabang "moderate DAN bukan normal" pada versi terdahulu TIDAK PERNAH
        # terjangkau: setiap kondisi bukan-normal sudah tertangkap cabang `high` di
        # atasnya. Perbaikan berikutnya membuka `medium` lewat dua jalan — tren
        # moderat, atau stres tinggi.
        #
        # Sejak stres dicabut, TINGGAL SATU jalan menuju `medium`: kondisi masih
        # normal tetapi glukosa bergerak moderat menuju batas. Itu memang arti yang
        # dimaksud dan tetap dapat dicapai, jadi tingkatannya dipertahankan — tetapi
        # jangan membaca cabang ini seolah masih menimbang dua faktor.
        if is_critical(self.risk_level):
            self.urgency = "critical"
        elif self.risk_level in (RISK_HYPO, RISK_HYPER) or self.trend_rate == "rapid":
            self.urgency = "high"
        elif self.trend_rate == "moderate":
            self.urgency = "medium"
        else:
            self.urgency = "low"

    # ──────────────────────────────────────────────────────────
    # Serialisation helpers
    # ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Full state dict — includes all derived fields."""
        return {
            "patient_id": self.patient_id,
            "current_glucose": self.current_glucose,
            "predicted_glucose": self.predicted_glucose,
            "prediction_horizon_minutes": self.prediction_horizon_minutes,
            "glucose_delta": self.glucose_delta,
            "trend_direction": self.trend_direction,
            "trend_label": self.trend_label,
            "trend_rate": self.trend_rate,
            "risk_level": self.risk_level,
            "risk_label": self.risk_label,
            "urgency": self.urgency,
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
            "activity_level": self.activity_level,
            "timestamp": self.timestamp,
        }

    def to_rag_context(self) -> Dict[str, Any]:
        """Minimal dict compatible with RAGPipeline.answer(patient_state=...).

        Uses the key names expected by existing pipeline and UI code.
        """
        return {
            "current_glucose": self.current_glucose,
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
            "activity_level": self.activity_level,
        }

    # ──────────────────────────────────────────────────────────
    # Factory methods
    # ──────────────────────────────────────────────────────────

    @classmethod
    def from_model_output(
        cls,
        patient_id: str,
        current_glucose: float,
        predicted_glucose: float,
        feature_row: Optional[Dict[str, Any]] = None,
        prediction_horizon_minutes: int = 60,
        predicted_condition: Optional[str] = None,
        predicted_lower: Optional[float] = None,
        predicted_upper: Optional[float] = None,
    ) -> "PatientState":
        """Bentuk dari keluaran model prakiraan dan baris terakhir jendela fitur.

        Args:
            patient_id: Patient identifier.
            current_glucose: Observed glucose at prediction time (mg/dL).
            predicted_glucose: Model's predicted glucose (mg/dL).
            feature_row: Dict with keys ``insulin``, ``carbs``, ``activity``
                         from the last row of the feature window.
            prediction_horizon_minutes: Model's forecast horizon (default 60 min / 1 step).
            predicted_condition: Optional condition from the condition classifier
                ("hypoglycemia" | "normal" | "hyperglycemia"). Overrides thresholding
                the regression value, which under-detects condition changes.
            predicted_lower: Optional lower bound of the conformal prediction interval.
            predicted_upper: Optional upper bound of the conformal prediction interval.
        """
        row = feature_row or {}
        return cls(
            patient_id=patient_id,
            current_glucose=current_glucose,
            predicted_glucose=predicted_glucose,
            prediction_horizon_minutes=prediction_horizon_minutes,
            # `iob`/`cob` DIDAHULUKAN: itulah nama fitur pada jalur produksi
            # (config.model.engineered_features; disetel di preprocessor.py). Sebelumnya
            # baris ini hanya membaca "insulin"/"carbs", sehingga kunci tidak pernah cocok
            # dan nilainya jatuh ke 0.0 TANPA error — konteks LLM selalu berbunyi
            # "Insulin on board : 0.00 unit" berapa pun insulin aktif sesungguhnya.
            # Pada kasus hipoglikemia justru insulin aktif pendorong utamanya, jadi
            # sistem menyembunyikan penyebab dari model penalarnya sendiri.
            # "insulin"/"carbs" dipertahankan sebagai cadangan: keduanya fitur MENTAH
            # (dosis dan asupan pada satu observasi), bukan akumulasi meluruh.
            insulin_on_board=float(
                row.get("iob", row.get("insulin_on_board", row.get("insulin", 0.0)))),
            carbs_on_board=float(
                row.get("cob", row.get("carbs_on_board", row.get("carbs", 0.0)))),
            activity_level=int(float(row.get("activity", row.get("activity_level", 0)))),
            glucose_rate=_opsional(row.get("glucose_rate")),
            time_since_prev_glucose=_opsional(row.get("time_since_prev_glucose")),
            predicted_condition=predicted_condition,
            predicted_lower=predicted_lower,
            predicted_upper=predicted_upper,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientState":
        """Deserialise from a previously exported dict."""
        return cls(
            patient_id=str(data.get("patient_id", "unknown")),
            current_glucose=float(data.get("current_glucose", 100.0)),
            predicted_glucose=float(data.get("predicted_glucose", 100.0)),
            prediction_horizon_minutes=int(data.get("prediction_horizon_minutes", 60)),
            insulin_on_board=float(data.get("insulin_on_board", 0.0)),
            carbs_on_board=float(data.get("carbs_on_board", 0.0)),
            activity_level=int(data.get("activity_level", 0)),
            timestamp=str(data.get("timestamp", datetime.now().isoformat())),
        )

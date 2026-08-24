"""Pembentuk kueri penelusuran yang dikondisikan pada kondisi TERPREDIKSI.

Inti kebaruan sistem: kueri disusun dari kondisi yang diprediksi akan terjadi, bukan
dari kondisi yang sedang berlaku, sehingga potongan dokumen yang terambil menjawab
keadaan yang akan dihadapi pasien.

Menerima ``PatientState`` dan menghasilkan dua keluaran: ``primary_query`` untuk
penelusur, dan ``llm_system_context`` untuk prompt model bahasa.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
try:
    from src.patient_state import PatientState
except ImportError:
    from ..patient_state import PatientState  # relative fallback inside package

# Ambang klinis dari SATU sumber kebenaran (src/constants.py).
from src.constants import (
    GLUCOSE_HIGH,
    GLUCOSE_LOW,
    RISK_CRITICAL_HYPER,
    RISK_CRITICAL_HYPO,
    RISK_HYPER,
    RISK_HYPO,
    base_condition,
)


# ──────────────────────────────────────────────────────────────
# Query strategy
# ──────────────────────────────────────────────────────────────

class QueryStrategy(str, Enum):
    """Controls the clinical focus of the generated query."""
    RISK_FOCUSED        = "risk"         # What is the risk and how severe?
    INTERVENTION_FOCUSED = "intervention" # What actions should be taken now?
    MONITORING_FOCUSED  = "monitoring"   # What should be monitored and when?
    COMPREHENSIVE       = "comprehensive" # All of the above (default)


# ──────────────────────────────────────────────────────────────
# Query builder
# ──────────────────────────────────────────────────────────────

class PredictionConditionedQueryBuilder:
    """Build clinical RAG queries conditioned on ML model predictions.

    The core novelty: the query is NOT a static template. It is dynamically
    constructed from the model's numerical output so that both document
    retrieval and LLM generation are conditioned on the forecasted glucose
    trajectory, not on the current observed value alone.
    """

    def __init__(self, strategy: QueryStrategy = QueryStrategy.COMPREHENSIVE):
        self.strategy = strategy

    # ── Public API ────────────────────────────────────────────

    def build(
        self,
        state: PatientState,
        user_question: Optional[str] = None,
    ) -> "ConditionedQuery":
        """Build a complete conditioned query bundle from a PatientState.

        Returns a ``ConditionedQuery`` object with both the retrieval query
        (ChromaDB) and the LLM context string (Gemini system context).
        """
        primary = self._primary_query(state, user_question)
        llm_ctx = self._llm_context(state)
        metadata_tags = self._metadata_tags(state)
        return ConditionedQuery(
            primary_query=primary,
            llm_context=llm_ctx,
            metadata_filter_tags=metadata_tags,
            patient_state=state,
        )

    # ── Internal builders ──────────────────────────────────────

    def _primary_query(self, state: PatientState, user_question: Optional[str]) -> str:
        """Build the retrieval query string sent to ChromaDB/MMR retriever.

        The query embeds:
        - The predicted glucose value (not just current)
        - Trend direction and rate
        - Contributing clinical factors
        - Time horizon of the prediction
        """
        parts: List[str] = []

        # 1. Prediction-conditioned core statement
        parts.append(
            f"Prediksi glukosa {state.prediction_horizon_minutes} menit ke depan: "
            f"{state.predicted_glucose:.1f} mg/dL "
            f"(dari {state.current_glucose:.1f} mg/dL, "
            f"perubahan {state.glucose_delta:+.1f} mg/dL, tren {state.trend_label})."
        )

        # 2. Clinical risk classification
        parts.append(f"Status risiko prediksi: {state.risk_label}.")

        # 2b. Pengondisian sadar-ketidakpastian: kondisi berisiko yang masih tercakup
        # interval prediksi tetap dimunculkan pada kueri meski prediksi TITIK-nya normal,
        # agar retrieval tidak buta terhadap bahaya yang mungkin terjadi (lihat Bab VI).
        risk_terms = {
            RISK_HYPO: f"hipoglikemia (glukosa di bawah {GLUCOSE_LOW:.0f} mg/dL)",
            RISK_HYPER: f"hiperglikemia (glukosa di atas {GLUCOSE_HIGH:.0f} mg/dL)",
        }
        extra = [
            risk_terms[c]
            for c in state.anticipated_conditions
            if c in risk_terms and c != base_condition(state.risk_level)
        ]
        if extra and state.predicted_lower is not None and state.predicted_upper is not None:
            parts.append(
                f"Interval prediksi {state.predicted_lower:.0f}-{state.predicted_upper:.0f} mg/dL "
                f"masih mencakup risiko {', '.join(extra)}; sertakan penanganannya."
            )

        # 3. Contributing factors (ordered by clinical significance)
        factors = self._contributing_factors(state)
        if factors:
            parts.append(f"Faktor kontribusi: {', '.join(factors)}.")

        # 4. Strategy-specific question
        if user_question:
            parts.append(user_question)
        else:
            parts.append(self._strategy_question(state))

        return " ".join(parts)

    def _llm_context(self, state: PatientState) -> str:
        """Build the structured context block injected into the LLM system prompt.

        This block gives Gemini explicit quantitative context to reason about
        the patient's predicted trajectory — the key differentiator from
        systems that only pass current measurements.
        """
        delta_sign = "+" if state.glucose_delta >= 0 else ""
        trend_urgency = {
            "rapid": "PERUBAHAN CEPAT — intervensi mungkin diperlukan segera",
            "moderate": "perubahan moderat — pantau lebih sering",
            "slow": "perubahan lambat — monitoring rutin",
        }.get(state.trend_rate, "")

        lines = [
            "=== KONTEKS PREDIKSI GLUKOSA (PREDICTION-CONDITIONED RAG) ===",
            f"  Pasien ID        : {state.patient_id}",
            f"  Glukosa sekarang : {state.current_glucose:.1f} mg/dL",
            f"  Glukosa prediksi : {state.predicted_glucose:.1f} mg/dL "
            f"  (+{state.prediction_horizon_minutes} menit)",
            f"  Perubahan        : {delta_sign}{state.glucose_delta:.1f} mg/dL "
            f"({state.trend_label}, {trend_urgency})",
            f"  Status risiko    : {state.risk_label}",
            f"  Urgensi          : {state.urgency.upper()}",
            "---",
            f"  Insulin on board : {state.insulin_on_board:.2f} unit",
            f"  Carbs on board   : {state.carbs_on_board:.1f} g",
            f"  Skor aktivitas   : {state.activity_level}/10 (skala intensitas, bukan menit)",
            # Laju dan jarak ke pengukuran sebelumnya. Keduanya sudah masuk vektor
            # fitur model; barisnya HANYA dicetak bila benar-benar ada, sebab 0.0
            # adalah laju yang sah dan tidak boleh dipakai sebagai penanda "tidak
            # diketahui".
            *([f"  Laju perubahan   : {state.glucose_rate:+.2f} mg/dL per menit"]
              if state.glucose_rate is not None else []),
            *([f"  Jarak ukur       : {state.time_since_prev_glucose:.0f} menit sejak "
               f"pengukuran sebelumnya"]
              if state.time_since_prev_glucose is not None else []),
            "=============================================================",
        ]
        return "\n".join(lines)

    def _contributing_factors(self, state: PatientState) -> List[str]:
        """Identify the active clinical factors contributing to the prediction."""
        factors: List[str] = []

        if state.trend_rate == "rapid":
            label = "tren cepat meningkat" if state.trend_direction == "rising" else "tren cepat menurun"
            factors.append(label)

        if state.insulin_on_board >= 1.0:
            factors.append(f"insulin aktif {state.insulin_on_board:.1f} unit")
        if state.carbs_on_board >= 10.0:
            factors.append(f"karbohidrat belum terserap {state.carbs_on_board:.0f} g")
        # AMBANG DIPERBAIKI 24 Agustus 2026. Sebelumnya `< 15` dan `>= 60`, warisan
        # asumsi "aktivitas = menit" yang tidak pernah benar: kanal `activity`
        # OhioT1DM adalah skor intensitas ordinal dengan HANYA 10 nilai unik pada
        # 166.533 baris (results/eval_prediksi/feature_importance.json).
        #
        # Akibat ambang lama, setiap pasien selalu jatuh ke cabang pertama dan
        # advisory selalu memuat kalimat "aktivitas fisik rendah" — satu kalimat
        # generik yang sama untuk semua orang, persis yang dikeluhkan pada evaluasi
        # dokter. Pita di bawah mengikuti legenda yang dibaca dokter di Logbook:
        # 1-3 ringan, 4-6 sedang, 7-8 berat.
        if state.activity_level < 2:
            factors.append("aktivitas fisik rendah")
        elif state.activity_level >= 7:
            factors.append(f"aktivitas berat (skor {state.activity_level}/10)")

        return factors

    def _strategy_question(self, state: PatientState) -> str:
        """Return the clinical question suffix based on query strategy and patient state."""
        if self.strategy == QueryStrategy.RISK_FOCUSED:
            return (
                f"Apa risiko klinis utama dari prediksi glukosa {state.predicted_glucose:.0f} mg/dL "
                f"dengan tren {state.trend_label}? Seberapa mendesak penanganannya?"
            )
        if self.strategy == QueryStrategy.INTERVENTION_FOCUSED:
            return (
                f"Tindakan apa yang harus dilakukan segera untuk kondisi prediksi "
                f"{state.risk_label} ini sebelum glukosa mencapai {state.predicted_glucose:.0f} mg/dL?"
            )
        if self.strategy == QueryStrategy.MONITORING_FOCUSED:
            return (
                f"Parameter apa yang harus dipantau dalam {state.prediction_horizon_minutes} menit "
                f"ke depan untuk kondisi prediksi {state.predicted_glucose:.0f} mg/dL ini?"
            )
        # COMPREHENSIVE — used for general clinical advisory
        return (
            f"Berikan penilaian risiko, tindakan pencegahan, dan protokol pemantauan "
            f"untuk kondisi prediksi glukosa {state.predicted_glucose:.0f} mg/dL "
            f"({state.risk_label}) dalam {state.prediction_horizon_minutes} menit ke depan. "
            "Sertakan rekomendasi yang bisa dilakukan dokter maupun pasien."
        )

    def _metadata_tags(self, state: PatientState) -> Dict[str, Any]:
        """Build ChromaDB metadata filter hints based on patient risk."""
        # DM TIPE 1. Tag sebelumnya berbunyi "dm_tipe2", peninggalan pembingkaian
        # lama yang sudah dicabut dari seluruh naskah. Sistem ini dilatih pada
        # OhioT1DM dan menyasar DM Tipe 1.
        #
        # CATATAN: `metadata_filter_tags` dibangun dan disimpan pada ConditionedQuery
        # tetapi TIDAK dikonsumsi penelusur mana pun — tak ada satu pun pemanggil di
        # luar berkas ini. Jadi perbaikan ini tidak mengubah hasil retrieval; ia
        # mencegah pembaca berikutnya mengira sistem menyaring ke pedoman tipe 2.
        # Bila kelak penyaringan metadata benar-benar dipasang, periksa dulu nilai
        # `jenis_dm` yang ada pada metadata potongan sebelum mengandalkannya.
        tags: Dict[str, Any] = {"jenis_dm": "dm_tipe1"}
        if state.risk_level in (RISK_HYPO, RISK_CRITICAL_HYPO):
            tags["topik"] = "hipoglikemia"
        elif state.risk_level in (RISK_HYPER, RISK_CRITICAL_HYPER):
            tags["topik"] = "hiperglikemia"
        return tags


# ──────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────

class ConditionedQuery:
    """Immutable bundle produced by PredictionConditionedQueryBuilder.

    Carries everything the RAGPipeline needs to retrieve documents and
    generate a prediction-conditioned advisory.
    """

    def __init__(
        self,
        primary_query: str,
        llm_context: str,
        metadata_filter_tags: Dict[str, Any],
        patient_state: PatientState,
    ):
        self.primary_query = primary_query
        self.llm_context = llm_context
        self.metadata_filter_tags = metadata_filter_tags
        self.patient_state = patient_state

    def to_pipeline_kwargs(self) -> Dict[str, Any]:
        """Return kwargs ready to be unpacked into ``RAGPipeline.answer()``."""
        return {
            "patient_state": self.patient_state.to_rag_context(),
            "prediction": self.patient_state.predicted_glucose,
            "query": self.primary_query,
        }

    def __repr__(self) -> str:
        return (
            f"ConditionedQuery("
            f"risk={self.patient_state.risk_level!r}, "
            f"pred={self.patient_state.predicted_glucose:.1f} mg/dL, "
            f"urgency={self.patient_state.urgency!r}"
            f")"
        )


# ──────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────

def build_conditioned_query(
    patient_id: str,
    current_glucose: float,
    predicted_glucose: float,
    feature_row: Optional[Dict[str, Any]] = None,
    user_question: Optional[str] = None,
    strategy: QueryStrategy = QueryStrategy.COMPREHENSIVE,
    prediction_horizon_minutes: int = 60,
) -> ConditionedQuery:
    """One-liner helper: model output → ConditionedQuery.

    Typical call from the Streamlit prediction page::

        cq = build_conditioned_query(
            patient_id=selected_patient,
            current_glucose=float(window_df.iloc[-1]["glucose"]),
            predicted_glucose=pred,
            feature_row=dict(window_df.iloc[-1]),
        )
        result = rag_pipeline.answer(**cq.to_pipeline_kwargs())
    """
    state = PatientState.from_model_output(
        patient_id=patient_id,
        current_glucose=current_glucose,
        predicted_glucose=predicted_glucose,
        feature_row=feature_row,
        prediction_horizon_minutes=prediction_horizon_minutes,
    )
    builder = PredictionConditionedQueryBuilder(strategy=strategy)
    return builder.build(state, user_question=user_question)

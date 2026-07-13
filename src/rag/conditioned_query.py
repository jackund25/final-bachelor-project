"""Prediction-Conditioned Query Builder for the Diabetes RAG pipeline.

NOVELTY STATEMENT
-----------------
Rad et al. (2024) — "Personalized Diabetes Management with Digital Twins:
A Patient-Centric Knowledge Graph Approach" (J. Personalized Medicine, MDPI)
— uses **static SPARQL queries** against a pre-built ontology. Query predicates
are fixed templates; the knowledge graph cannot reason about a *future* glucose
value.

This module implements **prediction-conditioned RAG**: the numeric glucose
prediction output from the ML model (Random Forest / future LSTM) is used to
dynamically construct the retrieval query. This means:

1. The retrieved knowledge chunks are selected based on WHERE the patient
   is GOING (predicted state), not where they currently are.
2. The LLM prompt includes the quantitative forecast context, enabling
   temporally-grounded clinical reasoning.
3. Contributing factors (trend rate, IOB, COB, stress, activity) are
   prioritised according to their likely influence on the predicted outcome.

Architecture
------------
    PatientState  ──►  PredictionConditionedQueryBuilder
                              │
                    ┌─────────┴──────────┐
                    │                    │
              primary_query        llm_system_context
             (for ChromaDB)         (for Gemini prompt)
                    │                    │
              MMRRetriever          DiabetesAdvisorChain
                    │                    │
                    └────────┬───────────┘
                         RAGPipeline.answer()
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from src.patient_state import PatientState
except ImportError:
    from ..patient_state import PatientState  # relative fallback inside package


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
            "hypoglycemia": "hipoglikemia (glukosa di bawah 70 mg/dL)",
            "hyperglycemia": "hiperglikemia (glukosa di atas 180 mg/dL)",
        }
        extra = [
            risk_terms[c]
            for c in state.anticipated_conditions
            if c in risk_terms and c != state.risk_level.replace("critical_", "")
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
            f"  Aktivitas hari ini: {state.activity_level} menit",
            f"  Tingkat stres    : {state.stress_level}/10",
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
        if state.stress_level >= 7:
            factors.append(f"stres tinggi ({state.stress_level}/10)")
        if state.activity_level < 15:
            factors.append("aktivitas fisik rendah")
        elif state.activity_level >= 60:
            factors.append(f"aktivitas tinggi ({state.activity_level} menit)")

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
        tags: Dict[str, Any] = {"jenis_dm": "dm_tipe2"}
        if state.risk_level in ("hypoglycemia", "critical_hypoglycemia"):
            tags["topik"] = "hipoglikemia"
        elif state.risk_level in ("hyperglycemia", "critical_hyperglycemia"):
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

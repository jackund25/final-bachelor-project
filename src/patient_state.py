"""PatientState — canonical bridge between ML model output and RAG pipeline.

Converts raw model predictions + feature windows into a structured, clinically
meaningful state object. This is the data contract between the prediction layer
and the RAG advisory layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

GLUCOSE_LOW = 70.0    # mg/dL — hypoglycemia threshold (ADA)
GLUCOSE_HIGH = 180.0  # mg/dL — hyperglycemia threshold (ADA postprandial)
GLUCOSE_CRITICAL_LOW = 54.0
GLUCOSE_CRITICAL_HIGH = 250.0

TREND_STABLE_THRESHOLD_MGDL = 10.0  # delta < 10 mg/dL → stable
TREND_RAPID_THRESHOLD_MGDL = 30.0   # delta > 30 mg/dL → rapid change


# ──────────────────────────────────────────────────────────────
# Core dataclass
# ──────────────────────────────────────────────────────────────

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
            feature_row={"insulin": 0.5, "carbs": 30.0, "activity": 0, "stress": 8},
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
    activity_level: int = 0         # minutes today
    stress_level: int = 5           # 1-10 Likert

    # ── Timestamp ─────────────────────────────────────────────
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # ── Derived (computed in __post_init__) ───────────────────
    glucose_delta: float = field(init=False)
    trend_direction: str = field(init=False)   # "rising" | "stable" | "falling"
    trend_label: str = field(init=False)       # human-readable Indonesian label
    trend_rate: str = field(init=False)        # "rapid" | "moderate" | "slow"
    risk_level: str = field(init=False)        # "hypoglycemia" | "normal" | "hyperglycemia"
    risk_label: str = field(init=False)        # Indonesian clinical label
    urgency: str = field(init=False)           # "critical" | "high" | "medium" | "low"

    def __post_init__(self) -> None:
        # Clamp inputs to physiological range
        self.current_glucose = float(np.clip(self.current_glucose, 20, 600))
        self.predicted_glucose = float(np.clip(self.predicted_glucose, 20, 600))
        self.insulin_on_board = max(0.0, float(self.insulin_on_board))
        self.carbs_on_board = max(0.0, float(self.carbs_on_board))
        self.activity_level = max(0, int(self.activity_level))
        self.stress_level = int(np.clip(self.stress_level, 1, 10))

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
        if self.predicted_glucose < GLUCOSE_CRITICAL_LOW:
            self.risk_level = "critical_hypoglycemia"
            self.risk_label = "BAHAYA - Hipoglikemia Berat"
        elif self.predicted_glucose < GLUCOSE_LOW:
            self.risk_level = "hypoglycemia"
            self.risk_label = "BAHAYA - Hipoglikemia"
        elif self.predicted_glucose > GLUCOSE_CRITICAL_HIGH:
            self.risk_level = "critical_hyperglycemia"
            self.risk_label = "BAHAYA - Hiperglikemia Berat"
        elif self.predicted_glucose > GLUCOSE_HIGH:
            self.risk_level = "hyperglycemia"
            self.risk_label = "HATI-HATI - Hiperglikemia"
        else:
            self.risk_level = "normal"
            self.risk_label = "AMAN"

        # Urgency
        if self.risk_level.startswith("critical"):
            self.urgency = "critical"
        elif self.risk_level in ("hypoglycemia", "hyperglycemia") or self.trend_rate == "rapid":
            self.urgency = "high"
        elif self.stress_level >= 8 or (self.trend_rate == "moderate" and self.risk_level != "normal"):
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
            "stress_level": self.stress_level,
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
            "stress_level": self.stress_level,
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
    ) -> "PatientState":
        """Create from RF/LSTM model output + last feature window row.

        Args:
            patient_id: Patient identifier.
            current_glucose: Observed glucose at prediction time (mg/dL).
            predicted_glucose: Model's predicted glucose (mg/dL).
            feature_row: Dict with keys ``insulin``, ``carbs``, ``activity``, ``stress``
                         from the last row of the feature window.
            prediction_horizon_minutes: Model's forecast horizon (default 60 min / 1 step).
        """
        row = feature_row or {}
        return cls(
            patient_id=patient_id,
            current_glucose=current_glucose,
            predicted_glucose=predicted_glucose,
            prediction_horizon_minutes=prediction_horizon_minutes,
            insulin_on_board=float(row.get("insulin", row.get("insulin_on_board", 0.0))),
            carbs_on_board=float(row.get("carbs", row.get("carbs_on_board", 0.0))),
            activity_level=int(float(row.get("activity", row.get("activity_level", 0)))),
            stress_level=int(float(row.get("stress", row.get("stress_level", 5)))),
        )

    @classmethod
    def from_digital_twin(
        cls,
        twin: Any,
        predicted_glucose: float,
        prediction_horizon_minutes: int = 60,
    ) -> "PatientState":
        """Create from a PatientDigitalTwin instance + model prediction."""
        s = twin.state
        return cls(
            patient_id=twin.patient_id,
            current_glucose=float(s.get("current_glucose", 100.0)),
            predicted_glucose=predicted_glucose,
            prediction_horizon_minutes=prediction_horizon_minutes,
            insulin_on_board=float(s.get("insulin_on_board", 0.0)),
            carbs_on_board=float(s.get("carbs_on_board", 0.0)),
            activity_level=int(s.get("activity_level", 0)),
            stress_level=int(s.get("stress_level", 5)),
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
            stress_level=int(data.get("stress_level", 5)),
            timestamp=str(data.get("timestamp", datetime.now().isoformat())),
        )

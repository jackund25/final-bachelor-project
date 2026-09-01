from datetime import datetime
from enum import Enum
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.prediction_service import PredictionService


router = APIRouter(
    prefix="/api/prediction",
    tags=["Prediction"],
)


# Stage 2 — Prediction API
# Modality-aware request contract


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

    # Optional at observation level.
    # The request-level glucose_source is the authoritative
    # routing value for this endpoint.
    glucose_source: GlucoseSource | None = None


class PredictionRequest(BaseModel):
    patient_id: str

    # Explicit modality selection.
    # Default CGM preserves compatibility with the previous
    # frontend while the UI is being migrated.
    glucose_source: GlucoseSource = Field(
        default=GlucoseSource.CGM,
        description=(
            "Glucose modality used for prediction: "
            "CGM or FINGER_STICK."
        ),
    )

    data: List[Observation]


_service = None


def get_service():
    global _service

    if _service is None:
        _service = PredictionService()

    return _service


def _validate_request(request: PredictionRequest):
    if not request.data:
        raise ValueError(
            "Observation data tidak boleh kosong."
        )

    # Prevent accidentally mixing multiple patients in one
    # prediction request.
    patient_ids = {
        observation.patient_id
        for observation in request.data
    }

    if patient_ids != {request.patient_id}:
        raise ValueError(
            "Semua observation harus memiliki patient_id "
            "yang sama dengan patient_id pada request."
        )


@router.post("")
def predict(request: PredictionRequest):

    try:
        _validate_request(request)

        # Request-level source is authoritative.
        #
        # We intentionally do not let one observation silently
        # route the request to another model.

        source = request.glucose_source.value

        records = []

        for observation in request.data:
            record = observation.model_dump()

            # Normalize source into the dataframe so that
            # PredictionService can filter the selected modality.
            record["glucose_source"] = source

            records.append(record)

        df = pd.DataFrame(records)

        result = get_service().predict(
            df,
            glucose_source=source,
        )

        return {
            "status": "success",
            "patient_id": request.patient_id,
            "glucose_source": source,
            "mode": result.get(
                "mode",
                "prediction",
            ),
            "prediction_available": result.get(
                "prediction_available",
                False,
            ),
            "data": result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc
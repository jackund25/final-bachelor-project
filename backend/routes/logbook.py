from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.supabase_data_service import SupabaseDataService


router = APIRouter(prefix="/api/logbook", tags=["Logbook"])
_data_service = None


class LogbookRequest(BaseModel):
    patient_code: str
    glucose_source: str
    timestamp: datetime
    glucose: float
    insulin: float = 0.0
    carbs: float = 0.0
    activity: float = 0.0
    # DICABUT 24 Agustus 2026 — bukan fitur model, dan kanal sumbernya di OhioT1DM
    # praktis kosong (7 event / 12 pasien). Medannya tetap diterima agar klien lama
    # tidak putus, tetapi tidak lagi disimpan. Lihat backend/routes/clinical.py.
    stress: float | None = None
    insulin_type: str | None = None
    meal_type: str | None = None
    duration_min: float | None = None
    activity_type: str | None = None


def get_data_service():
    global _data_service
    if _data_service is None:
        _data_service = SupabaseDataService()
    return _data_service


@router.post("")
def create_logbook_entry(request: LogbookRequest):
    if request.glucose_source not in {"CGM", "FINGER_STICK"}:
        raise HTTPException(status_code=400, detail="Invalid glucose source.")

    try:
        return {
            "status": "success",
            "patient_code": request.patient_code,
            "glucose_source": request.glucose_source,
            "data": get_data_service().save_logbook(
                request.patient_code,
                request.glucose_source,
                request.model_dump(mode="json"),
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Logbook save failed.") from exc
"""Pendaftaran pasien untuk sesi evaluasi dokter.

Sebelum modul ini ada, satu-satunya cara menambah pasien adalah menyisipkan baris
langsung ke Supabase. Umpan balik dokter pada sesi evaluasi menyebut hal itu
sebagai hambatan pertama: aplikasi tidak dapat menerima pasien baru sama sekali,
sehingga dokter hanya bisa menelusuri pasien yang sudah ada.

Penulisan dilakukan lewat backend, bukan langsung dari peramban, karena kunci
publik Supabase tunduk pada RLS sedangkan backend memegang SUPABASE_SECRET_KEY.
"""

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.supabase_data_service import SupabaseDataService


router = APIRouter(prefix="/api/patients", tags=["Patients"])
_data_service = None

SUMBER_SAH = ("CGM", "FINGER_STICK")

# Kode pasien ikut menyusun kunci penyimpanan lokal dan tampil pada tiap tangkapan
# layar evaluasi, jadi bentuknya dibatasi supaya tidak ada spasi, tanda baca,
# maupun nama yang dapat mengidentifikasi orang.
POLA_KODE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{1,31}$")


class PatientCreateRequest(BaseModel):
    patient_code: str = Field(min_length=2, max_length=32)
    glucose_sources: list[str] = Field(
        default_factory=lambda: list(SUMBER_SAH),
        min_length=1,
    )


def get_data_service():
    global _data_service
    if _data_service is None:
        _data_service = SupabaseDataService()
    return _data_service


@router.get("")
def list_patients():
    try:
        pasien = get_data_service().list_patients()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Daftar pasien gagal dimuat.",
        ) from exc

    return {
        "status": "success",
        "patients": [item["patient_code"] for item in pasien],
    }


@router.post("", status_code=201)
def create_patient(request: PatientCreateRequest):
    kode = request.patient_code.strip().upper()
    if not POLA_KODE.match(kode):
        raise HTTPException(
            status_code=400,
            detail=(
                "Kode pasien hanya boleh memuat huruf, angka, tanda hubung, dan "
                "garis bawah (2-32 karakter). Contoh: P013, DEMO-01."
            ),
        )

    sumber: list[str] = []
    for item in request.glucose_sources:
        nilai = item.strip().upper()
        if nilai not in SUMBER_SAH:
            raise HTTPException(
                status_code=400,
                detail=f"Sumber glukosa tidak dikenal: {item}.",
            )
        if nilai not in sumber:
            sumber.append(nilai)

    try:
        data = get_data_service().create_patient(kode, sumber)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pasien baru gagal dibuat: {exc}",
        ) from exc

    return {"status": "success", "data": data}

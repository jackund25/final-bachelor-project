"""Supabase data adapter for the normalized clinical schema."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class SupabaseDataService:
    """Resolve public patient/source codes into normalized model observations."""

    def __init__(self) -> None:
        self.url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.key = os.getenv("SUPABASE_SECRET_KEY", "")
        if not self.url or not self.key:
            raise RuntimeError(
                "Supabase backend belum dikonfigurasi. Set SUPABASE_URL "
                "dan SUPABASE_SECRET_KEY."
            )

    def _request(
        self,
        table: str,
        params: list[tuple[str, str]],
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query = urlencode(params)
        request = Request(
            f"{self.url}/rest/v1/{table}?{query}",
            method=method,
            headers={
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            data=json.dumps(payload).encode("utf-8") if payload else None,
        )
        try:
            with urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Supabase request failed for {table}: {exc}") from exc

    def patient(self, patient_code: str) -> dict[str, Any] | None:
        rows = self._request(
            "patients",
            [("select", "id,patient_code"), ("patient_code", f"eq.{patient_code}")],
        )
        return rows[0] if rows else None

    def source(
        self,
        patient_id: int | str,
        source_type: str,
    ) -> dict[str, Any] | None:
        rows = self._request(
            "glucose_sources",
            [
                ("select", "id,patient_id,source_type"),
                ("patient_id", f"eq.{patient_id}"),
                ("source_type", f"eq.{source_type}"),
            ],
        )
        return rows[0] if rows else None

    def observations(
        self,
        patient_code: str,
        source_type: str,
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
        patient = self.patient(patient_code)
        if patient is None:
            return pd.DataFrame(), {}, {}

        source = self.source(patient["id"], source_type)
        if source is None:
            return pd.DataFrame(), patient, {}

        readings = self._request(
            "glucose_readings",
            [
                ("select", "timestamp,glucose"),
                ("glucose_source_id", f"eq.{source['id']}"),
                ("order", "timestamp.asc"),
            ],
        )
        event_specs = {
            "insulin": ("insulin_events", "insulin_units"),
            "carbs": ("meal_events", "carbs_grams"),
            "activity": ("activity_events", "activity_level"),
            "stress": ("stress_events", "stress_level"),
        }
        events: dict[str, list[dict[str, Any]]] = {}
        for name, (table, value_column) in event_specs.items():
            events[name] = self._request(
                table,
                [
                    ("select", f"timestamp,{value_column}"),
                    ("patient_id", f"eq.{patient['id']}"),
                    ("order", "timestamp.asc"),
                ],
            )

        def latest_before(rows: list[dict[str, Any]], timestamp: str) -> Any:
            latest = None
            for row in rows:
                if row["timestamp"] <= timestamp:
                    latest = row
                else:
                    break
            return latest

        normalized: list[dict[str, Any]] = []
        for reading in readings:
            timestamp = reading["timestamp"]
            insulin = latest_before(events["insulin"], timestamp)
            carbs = latest_before(events["carbs"], timestamp)
            activity = latest_before(events["activity"], timestamp)
            stress = latest_before(events["stress"], timestamp)
            normalized.append({
                "patient_id": patient_code,
                "timestamp": timestamp,
                "glucose": float(reading["glucose"]),
                "insulin": float((insulin or {}).get("insulin_units", 0) or 0),
                "carbs": float((carbs or {}).get("carbs_grams", 0) or 0),
                "activity": float((activity or {}).get("activity_level", 0) or 0),
                "stress": float((stress or {}).get("stress_level", 0) or 0),
                "glucose_source": source_type,
            })

        return pd.DataFrame(normalized), patient, source

    def save_assessment(
        self,
        patient_id: int | str,
        glucose_source_id: int | str,
        assessment: dict[str, Any],
    ) -> dict[str, Any]:
        rows = self._request(
            "clinical_assessments",
            [],
            method="POST",
            payload={
                "patient_id": patient_id,
                "glucose_source_id": glucose_source_id,
                **assessment,
            },
        )
        return rows[0] if rows else {}

    def save_logbook(
        self,
        patient_code: str,
        source_type: str,
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        patient = self.patient(patient_code)
        if patient is None:
            raise ValueError(f"Patient {patient_code} tidak ditemukan.")

        source = self.source(patient["id"], source_type)
        if source is None:
            raise ValueError(
                f"Source {source_type} tidak tersedia untuk pasien {patient_code}."
            )

        reading = self._request(
            "glucose_readings",
            [],
            method="POST",
            payload={
                "glucose_source_id": source["id"],
                "timestamp": entry["timestamp"],
                "glucose": entry["glucose"],
            },
        )
        patient_id = patient["id"]
        timestamp = entry["timestamp"]
        event_payloads = [
            ("insulin_events", {
                "patient_id": patient_id,
                "timestamp": timestamp,
                "insulin_units": entry["insulin"],
                "insulin_type": entry.get("insulin_type"),
            }),
            ("meal_events", {
                "patient_id": patient_id,
                "timestamp": timestamp,
                "carbs_grams": entry["carbs"],
                "meal_type": entry.get("meal_type"),
            }),
            ("activity_events", {
                "patient_id": patient_id,
                "timestamp": timestamp,
                "activity_level": entry["activity"],
                "duration_min": entry.get("duration_min"),
                "activity_type": entry.get("activity_type"),
            }),
            # `stress_events` TIDAK LAGI DITULIS sejak 24 Agustus 2026. Stres bukan
            # fitur model dan tidak lagi diminta dari dokter, jadi menulisnya hanya
            # menumpuk baris bernilai bawaan yang kelak disalahartikan sebagai
            # pengukuran.
            #
            # TABELNYA SENGAJA DIBIARKAN UTUH. Baris lama tetap dapat dibaca dan
            # tidak ada migrasi yang menghapus apa pun; yang berhenti hanyalah
            # penulisan baru. Menghapus tabel akan membuang data historis demi
            # kerapian, dan itu pertukaran yang salah arah.
        ]
        for table, payload in event_payloads:
            self._request(table, [], method="POST", payload=payload)

        return reading[0] if reading else {}

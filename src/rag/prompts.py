"""Prompt templates and context formatting utilities for diabetes RAG."""

from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT = """Anda adalah asisten klinis berbasis panduan medis Indonesia untuk mendukung keputusan dokter dalam penanganan diabetes.

Aturan:
1. Jawab berdasarkan konteks yang diberikan.
2. Sertakan sitasi ringkas dari metadata sumber jika tersedia.
3. Jika konteks tidak cukup, katakan informasi belum tersedia pada knowledge base saat ini.
4. Untuk kondisi berisiko tinggi, sarankan evaluasi dokter segera.
5. Gunakan Bahasa Indonesia yang ringkas, jelas, dan actionable.
"""


def format_context_with_citations(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a readable context block with citations."""
    if not retrieved_docs:
        return "(Tidak ada konteks dokumen yang ditemukan)"

    lines: List[str] = []
    for idx, row in enumerate(retrieved_docs, start=1):
        metadata = dict(row.get("metadata", {}))
        sumber = metadata.get("sumber") or row.get("source", "Manual KB")
        tahun = metadata.get("tahun", "N/A")
        halaman = metadata.get("halaman", "N/A")
        citation = f"[{sumber}, {tahun}, Hal. {halaman}]"
        lines.append(f"{idx}. {row.get('text', '').strip()} {citation}")

    return "\n\n".join(lines)


def build_question_payload(query: str, patient_state: Dict[str, Any], prediction: float) -> str:
    """Build a compact clinician question payload."""
    # Handle prediction dict or numeric
    if isinstance(prediction, dict):
        pred_glucose = prediction.get('glucose_pred', '?')
        pred_risk = prediction.get('risk_level', 'N/A')
    else:
        pred_glucose = f"{float(prediction):.1f}" if prediction else '?'
        pred_risk = 'N/A'
    
    # Support both key conventions
    gluc = patient_state.get('glucose', patient_state.get('current_glucose', 'N/A'))
    stress = patient_state.get('stress', patient_state.get('stress_level', 5))
    activity = patient_state.get('activity', patient_state.get('activity_level', 0))
    insulin = patient_state.get('insulin', patient_state.get('insulin_on_board', 0))
    carbs = patient_state.get('carbs', patient_state.get('carbs_on_board', 0))
    
    return (
        f"Pertanyaan klinisi: {query}\n"
        f"Data pasien: glukosa={gluc} mg/dL, "
        f"stress={stress}/10, "
        f"aktivitas={activity} menit, "
        f"insulin_on_board={insulin} unit, "
        f"carbs_on_board={carbs} gram.\n"
        f"Prediksi 1 jam: {pred_glucose} mg/dL, Risk: {pred_risk}."
    )

"""Prompt templates and context formatting utilities for diabetes RAG."""

from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT = """Anda adalah asisten klinis berbasis panduan medis Indonesia untuk mendukung keputusan dokter dalam penanganan diabetes.

Aturan:
1. Jawab berdasarkan konteks yang diberikan.
2. JANGAN menulis nomor halaman, nomor bab, nomor tabel, atau tautan.
   Rujuk sumber HANYA dengan penanda [S1], [S2], ... sesuai nomor blok konteks.
   Nomor halaman ditampilkan oleh sistem dari metadata dokumen, bukan oleh Anda.
3. Jika blok konteks kosong, nyatakan secara eksplisit bahwa tidak ada rujukan
   panduan yang relevan pada knowledge base, dan JANGAN mengarang rujukan.
4. Jika konteks tidak cukup, katakan informasi belum tersedia pada knowledge base saat ini.
5. Untuk kondisi berisiko tinggi, sarankan evaluasi dokter segera.
6. Gunakan Bahasa Indonesia yang ringkas, jelas, dan actionable.
"""


def format_context_with_citations(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a readable context block with source markers.

    Nomor halaman SENGAJA TIDAK dimasukkan ke blok konteks. Aturan pada system prompt
    saja hanyalah jaminan lunak; tidak memberikan angkanya sama sekali adalah jaminan
    keras — model tidak dapat menyalin nomor halaman yang tak pernah dilihatnya.
    Penomoran halaman ditangani lapisan UI dari metadata chunk (src/rag/citations.py).
    """
    if not retrieved_docs:
        return "(Tidak ada konteks dokumen yang ditemukan)"

    from .citations import document_title

    lines: List[str] = []
    for idx, row in enumerate(retrieved_docs, start=1):
        metadata = dict(row.get("metadata", {}))
        fallback_name = row.get("source", "Manual KB")
        judul = document_title(metadata, fallback_name)
        lembaga = metadata.get("lembaga") or metadata.get("sumber") or fallback_name
        tahun = metadata.get("tahun", "N/A")
        marker = f"[S{idx}] {judul} — {lembaga}, {tahun}"
        lines.append(f"{marker}\n{row.get('text', '').strip()}")

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

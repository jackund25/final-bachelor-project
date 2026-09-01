"""Pembentuk kueri simetris untuk ablasi standard vs terkondisi-prediksi.

Agar selisih metrik dapat dikaitkan murni pada sumber pengondisian, satu-satunya
yang boleh berbeda antara kedua lengan adalah ANGKA yang dipakai. Sufiks penanda
horizon karena itu tidak dipakai pada lengan mana pun.

Modul ini juga menjadi satu-satunya sumber kebenaran untuk CONDITION_PHRASE.
"""

from __future__ import annotations

from typing import Optional

from src.constants import classify_glucose_3class

# Frasa kondisi yang menerjemahkan kelas glukosa menjadi teks yang dapat di-embed.
# Sengaja verbose: korpus adalah pedoman klinis, bukan katalog angka, sehingga kueri
# berupa angka telanjang hampir tidak beririsan dengan kosakata dokumen.
CONDITION_PHRASE = {
    "hipoglikemia": (
        "Hipoglikemia, gula darah rendah di bawah 70 mg/dL. "
        "Penyebab, gejala, dan penanganan segera (aturan 15-15)."
    ),
    "hiperglikemia": (
        "Hiperglikemia, gula darah tinggi di atas 180 mg/dL. "
        "Penyebab, gejala, dan penanganan."
    ),
    "normal": (
        "Gula darah dalam rentang normal/target. "
        "Target kontrol glikemik dan pemantauan rutin diabetes."
    ),
}


def build_ablation_query(glucose: float, cond: Optional[str] = None) -> str:
    """Bentuk kueri ablasi dari satu nilai glukosa.

    Pemanggil yang menentukan angka mana yang masuk: ``current`` untuk lengan
    standard, ``predicted`` untuk lengan prediction-conditioned. Fungsi ini
    sengaja TIDAK menerima parameter mode — supaya secara struktural mustahil
    menghasilkan dua bentuk kueri yang berbeda untuk kedua lengan.

    ``cond`` menimpa kelas hasil ambang; dipakai mode yang mengondisikan pada
    keluaran classifier alih-alih pada regresor.
    """
    c = cond or classify_glucose_3class(float(glucose))
    return f"Kadar glukosa darah {glucose:.0f} mg/dL. {CONDITION_PHRASE[c]}"

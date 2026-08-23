"""Pembentuk kueri SIMETRIS untuk ablasi standard vs prediction-conditioned.

Ablasi ini menguji satu hal saja: apakah mengondisikan retrieval pada glukosa
TERPREDIKSI lebih baik daripada pada glukosa SAAT INI. Agar kesimpulannya sah,
satu-satunya yang boleh berbeda antara kedua lengan adalah ANGKA yang dipakai.

Sebelumnya tidak demikian. Lengan prediction-conditioned menambahkan sufiks
" (prediksi 30 menit ke depan)" yang tidak ada pada lengan standard::

    standard : "Kadar glukosa darah 112 mg/dL. Gula darah dalam rentang normal..."
    predicted: "Kadar glukosa darah 58 mg/dL (prediksi 30 menit ke depan). Hipoglikemia..."

Kedua kueri itu berbeda pada dua hal sekaligus (angka DAN panjang/isi teks), sehingga
selisih metriknya tidak dapat dikaitkan murni pada sumber pengondisian. Sufiks itu juga
tidak bisa sekadar disalin ke lengan standard, karena di sana angkanya memang bukan
prediksi. Jalan keluarnya: menghapusnya dari keduanya.

Modul ini juga menjadi SATU sumber kebenaran untuk CONDITION_PHRASE, yang tadinya
tersalin di dua skrip dan diimpor silang oleh tiga skrip lain.
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

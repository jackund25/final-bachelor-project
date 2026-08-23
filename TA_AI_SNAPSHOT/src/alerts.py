"""Peringatan divergensi kondisi — logika, bukan tampilan.

Divergensi berarti kategori glukosa SEKARANG berbeda dari kategori glukosa
TERPREDIKSI. Inilah situasi tempat sistem antisipatif punya nilai: angka yang
terlihat sekarang tidak memberi tahu ke mana pasien menuju.

Mengapa modul ini ada di ``src/`` dan bukan di ``app/``
-------------------------------------------------------
Sebelumnya logikanya tertanam di ``app/streamlit_app.py`` sebagai satu ``if`` yang
membandingkan **string label tampilan**::

    if cur_label == "Dalam Target" and pred_label != "Dalam Target":
        st.warning(...)

Dua akibatnya. Pertama, logika keputusan klinis tidak dapat diuji tanpa menjalankan
Streamlit. Kedua, peringatan itu **asimetris**: hanya menyala bila kondisi sekarang
normal. Perpindahan hipoglikemia ke hiperglikemia -- ayunan paling berbahaya, dan
justru yang paling sering muncul setelah koreksi berlebihan -- tidak tertangkap sama
sekali, padahal Tabel III.1 laporan mendefinisikan divergensi sebagai perbedaan
kategori tanpa menyebut kategori awal mana pun.

Modul ini menangani **keenam** perpindahan kategori yang mungkin, dan pesannya
menyebut kedua kategori sehingga arah perpindahannya terbaca.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

from src.constants import (
    CLASS_HYPER,
    CLASS_HYPO,
    CLASS_NORMAL,
    classify_glucose_3class,
    condition_label_id,
)

# Tingkat kegentingan, diurutkan dari paling genting.
SEVERITY_CRITICAL: Final[str] = "kritis"    # ayunan melewati rentang target
SEVERITY_WARNING: Final[str] = "waspada"    # keluar dari rentang target
SEVERITY_INFO: Final[str] = "membaik"       # kembali ke rentang target


@dataclass(frozen=True)
class DivergenceAlert:
    """Hasil evaluasi divergensi untuk SATU horizon prediksi."""

    from_class: str
    to_class: str
    from_label: str
    to_label: str
    severity: str
    message: str
    horizon_minutes: int
    current_glucose: float
    predicted_glucose: float

    @property
    def is_deteriorating(self) -> bool:
        """Apakah perpindahannya menjauhi rentang target."""
        return self.severity in (SEVERITY_CRITICAL, SEVERITY_WARNING)


# Pesan per perpindahan. Kunci = (dari, ke). Keenam pasangan kategori yang berbeda
# tercakup; pasangan yang sama (tidak divergen) tidak ada di sini.
#
# Nada pesan mengikuti kegentingan klinisnya, bukan sekadar "berubah":
#   - ayunan hipo<->hiper : kritis, karena melewati seluruh rentang target dan
#     biasanya menandakan koreksi berlebihan
#   - keluar dari target  : waspada, tindakan pencegahan masih sempat
#   - kembali ke target   : informasi, supaya dokter tahu tren membaik dan tidak
#     mengoreksi dua kali
_TRANSITIONS: Final[dict[tuple[str, str], tuple[str, str]]] = {
    (CLASS_NORMAL, CLASS_HYPO): (
        SEVERITY_WARNING,
        "Kondisi saat ini {from_label}, namun glukosa diprediksi menuju {to_label} "
        "({pred:.0f} mg/dL) dalam {horizon} menit. Pertimbangkan karbohidrat pencegahan.",
    ),
    (CLASS_NORMAL, CLASS_HYPER): (
        SEVERITY_WARNING,
        "Kondisi saat ini {from_label}, namun glukosa diprediksi menuju {to_label} "
        "({pred:.0f} mg/dL) dalam {horizon} menit. Pertimbangkan tindakan pencegahan.",
    ),
    (CLASS_HYPO, CLASS_HYPER): (
        SEVERITY_CRITICAL,
        "Ayunan {from_label} ke {to_label}: dari {cur:.0f} mg/dL menuju {pred:.0f} mg/dL "
        "dalam {horizon} menit. Perpindahan melewati seluruh rentang target dan sering "
        "menandakan koreksi karbohidrat berlebihan. Pantau ketat.",
    ),
    (CLASS_HYPER, CLASS_HYPO): (
        SEVERITY_CRITICAL,
        "Ayunan {from_label} ke {to_label}: dari {cur:.0f} mg/dL menuju {pred:.0f} mg/dL "
        "dalam {horizon} menit. Perpindahan melewati seluruh rentang target dan sering "
        "menandakan koreksi insulin berlebihan. Pantau ketat.",
    ),
    (CLASS_HYPO, CLASS_NORMAL): (
        SEVERITY_INFO,
        "Kondisi saat ini {from_label} ({cur:.0f} mg/dL), diprediksi kembali {to_label} "
        "({pred:.0f} mg/dL) dalam {horizon} menit. Hindari koreksi ganda.",
    ),
    (CLASS_HYPER, CLASS_NORMAL): (
        SEVERITY_INFO,
        "Kondisi saat ini {from_label} ({cur:.0f} mg/dL), diprediksi kembali {to_label} "
        "({pred:.0f} mg/dL) dalam {horizon} menit. Hindari koreksi ganda.",
    ),
}


def evaluate_divergence(
    current_glucose: float,
    predicted_glucose: float,
    horizon_minutes: int,
) -> Optional[DivergenceAlert]:
    """Kembalikan peringatan bila kategori sekarang berbeda dari kategori prediksi.

    Mengembalikan ``None`` bila kedua kategori sama -- termasuk ketika angkanya
    berubah jauh tetapi tetap dalam satu kategori (mis. 80 ke 170 keduanya normal).
    Yang diuji adalah perpindahan KATEGORI, sesuai definisi Tabel III.1, bukan
    besarnya selisih angka.
    """
    from_class = classify_glucose_3class(current_glucose)
    to_class = classify_glucose_3class(predicted_glucose)

    entry = _TRANSITIONS.get((from_class, to_class))
    if entry is None:  # kategori sama -> tidak divergen
        return None

    severity, template = entry
    from_label = condition_label_id(from_class)
    to_label = condition_label_id(to_class)

    return DivergenceAlert(
        from_class=from_class,
        to_class=to_class,
        from_label=from_label,
        to_label=to_label,
        severity=severity,
        message=template.format(
            from_label=from_label, to_label=to_label,
            cur=float(current_glucose), pred=float(predicted_glucose),
            horizon=int(horizon_minutes),
        ),
        horizon_minutes=int(horizon_minutes),
        current_glucose=float(current_glucose),
        predicted_glucose=float(predicted_glucose),
    )

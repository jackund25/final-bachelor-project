"""Interval prediksi konformal — faktor dibaca dari hasil kalibrasi, bukan dihardcode.

Sebelumnya ``app/streamlit_app.py`` menuliskan ``CONFORMAL_K = 3.3`` di dua tempat
berbeda (baris 193 dan 221). Tiga akibatnya:

1. **Terduplikasi.** Mengubah satu tanpa yang lain membuat dua interval yang ditampilkan
   pada halaman yang sama saling bertentangan, tanpa error apa pun.
2. **Kedaluwarsa.** Angka 3,3 berasal dari kalibrasi 9 Juli, sedangkan model dilatih ulang
   5 Agustus pada Tugas 5 (segmentasi jeda sensor). Faktor itu mengkalibrasi model yang
   sudah tidak ada.
3. **Dipakai lintas horizon.** Aplikasi kini menampilkan +30 dan +60 menit. Prediksi 60
   menit jauh lebih tidak pasti, sehingga satu faktor untuk keduanya pasti keliru pada
   salah satunya.

Modul ini membaca faktor dari ``results/eval_prediksi/conformal_h{N}.json`` per horizon.
**Tidak ada nilai cadangan yang dihardcode**: bila berkas kalibrasi untuk horizon tersebut
tidak ada, fungsi mengembalikan ``None`` dan pemanggil wajib menyatakan intervalnya belum
terkalibrasi. Menampilkan interval dengan faktor tebakan lebih berbahaya daripada tidak
menampilkan interval sama sekali — dokter tidak punya cara membedakan keduanya.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

CALIBRATION_DIR = Path("results/eval_prediksi")

# Varian ternormalisasi dipakai produksi: lebarnya mengikuti ketidakpastian per sampel
# (q * std), bukan seragam. Untuk dukungan keputusan klinis itu lebih informatif -- jendela
# yang modelnya ragu terlihat lebih lebar.
VARIANT_NORMALIZED = "conformal_normalized"
VARIANT_ABSOLUTE = "conformal_absolute"


@lru_cache(maxsize=8)
def load_calibration(horizon_steps: int) -> Optional[dict]:
    """Baca berkas kalibrasi untuk satu horizon. ``None`` bila belum dikalibrasi."""
    path = CALIBRATION_DIR / f"conformal_h{int(horizon_steps)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def conformal_factor(
    horizon_steps: int,
    level: int = 95,
    variant: str = VARIANT_NORMALIZED,
) -> Optional[float]:
    """Faktor q untuk horizon dan tingkat cakupan tertentu. ``None`` bila tak ada."""
    cal = load_calibration(horizon_steps)
    if cal is None:
        return None
    entry = cal.get("levels", {}).get(str(level)) or cal.get("levels", {}).get(level)
    if not entry:
        return None
    q = entry.get(variant, {}).get("q")
    return float(q) if q is not None else None


def coverage_achieved(
    horizon_steps: int,
    level: int = 95,
    variant: str = VARIANT_NORMALIZED,
) -> Optional[float]:
    """Cakupan yang BENAR-BENAR tercapai pada set uji, untuk ditampilkan apa adanya."""
    cal = load_calibration(horizon_steps)
    if cal is None:
        return None
    entry = cal.get("levels", {}).get(str(level)) or cal.get("levels", {}).get(level)
    if not entry:
        return None
    cov = entry.get(variant, {}).get("coverage%")
    return float(cov) if cov is not None else None


def prediction_interval(
    prediction: float,
    prediction_std: Optional[float],
    horizon_steps: int,
    level: int = 95,
) -> Optional[Tuple[float, float]]:
    """Interval konformal ternormalisasi. ``None`` bila std atau kalibrasi tak tersedia."""
    if prediction_std is None:
        return None
    q = conformal_factor(horizon_steps, level=level)
    if q is None:
        return None
    margin = q * float(prediction_std)
    return float(prediction) - margin, float(prediction) + margin


def clear_cache() -> None:
    """Kosongkan cache — dipakai tes yang menulis berkas kalibrasi sementara."""
    load_calibration.cache_clear()

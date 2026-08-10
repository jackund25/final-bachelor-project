"""Protokol penyetelan parameter (Bagian C) — pembagian set dan pencatatan konfigurasi.

Alasan modul ini ada: menyetel parameter sampai angkanya bagus lalu melaporkan angka itu
sebagai hasil adalah overfitting pada set evaluasi. Modul ini memaksa pemisahan set
penyetelan dari set pelaporan, dan memaksa SELURUH konfigurasi tercatat, bukan hanya yang
terbaik.

Ketentuan yang diberlakukan di sini:

1. Set penyetelan dan set pelaporan **tidak beririsan**, dibagi dengan benih tetap
   sebelum satu pun hasil dilihat.
2. Kriteria pemilihan **ditetapkan di muka**: MRR. Ditulis sebagai konstanta di bawah,
   bukan sebagai argumen, supaya tidak dapat diubah setelah melihat hasil.
3. Setiap konfigurasi yang diuji dicatat ke ``results/tuning_log.json``, termasuk yang
   kalah.
4. Set evaluasi tidak boleh diubah setelah hasil terlihat. Skrip pemanggil hanya menerima
   pembagian yang dikembalikan modul ini.

Bila jumlah skenario terlalu sedikit untuk dibagi dua, ``bagi_dua()`` mengembalikan
``None`` dan pemanggil WAJIB melaporkan bahwa penyetelan tidak dapat dilakukan pada
himpunan itu. Melaporkan "penyetelan tidak dilakukan" lebih baik daripada melaporkan
angka yang tidak sah.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Optional

# Kriteria utama ditetapkan DI MUKA (Bagian C butir 3). Hit@k jenuh pada mode
# prediction-conditioned, sehingga tidak dapat membedakan konfigurasi.
KRITERIA = "mrr"

# Minimum skenario per bagian agar angkanya bermakna. Di bawah ini, penyetelan
# dinyatakan tidak dapat dilakukan alih-alih dilaporkan dengan n yang terlalu kecil.
MIN_PER_BAGIAN = 30

BENIH = 42
LOG = Path("results/tuning_log.json")


def bagi_dua(pasangan: list, benih: int = BENIH) -> Optional[tuple[list, list]]:
    """Bagi skenario menjadi (set_penyetelan, set_pelaporan) yang tidak beririsan.

    Mengembalikan ``None`` bila salah satu bagian akan berisi kurang dari
    ``MIN_PER_BAGIAN`` skenario.
    """
    if len(pasangan) < 2 * MIN_PER_BAGIAN:
        return None
    urut = list(pasangan)
    random.Random(benih).shuffle(urut)
    tengah = len(urut) // 2
    return urut[:tengah], urut[tengah:]


def catat(parameter: str, nilai: Any, metrik_penyetelan: dict,
          catatan: str = "", log_path: Path = LOG,
          kriteria: Optional[str] = None) -> None:
    """Tambahkan satu konfigurasi ke tuning_log.json. Dipanggil untuk SETIAP nilai diuji.

    ``kriteria`` boleh ditimpa karena :data:`KRITERIA` (MRR) hanya berlaku bagi penyetelan
    RETRIEVAL. Percobaan pada sisi prediksi memakai kriteria lain — T1.1b memilih
    konfigurasi TERKECIL yang RMSE-nya tidak berbeda signifikan. Sebelum parameter ini
    ada, entri semacam itu tetap terstempel ``"mrr"``, sehingga log penyetelan —
    justru artefak yang seharusnya membuktikan protokol dipatuhi — mencatat kriteria
    yang tidak pernah dipakai.

    Timpaan ini TIDAK melonggarkan Bagian C: kriterianya tetap harus ditetapkan sebagai
    konstanta di skrip pemanggil sebelum hasil terlihat, dan yang berubah hanya
    kejujuran pencatatannya.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    riwayat = []
    if log_path.exists():
        try:
            riwayat = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            riwayat = []
    riwayat.append({
        "parameter": parameter,
        "nilai": nilai,
        "kriteria": kriteria or KRITERIA,
        "metrik_set_penyetelan": metrik_penyetelan,
        "waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
        "catatan": catatan,
    })
    log_path.write_text(json.dumps(riwayat, indent=2, ensure_ascii=False),
                        encoding="utf-8")


def pilih_terbaik(kandidat: dict[Any, dict]) -> Any:
    """Pilih nilai parameter dengan KRITERIA tertinggi pada set penyetelan.

    Seri diputus dengan memilih nilai terkecil, supaya pemilihannya deterministik dan
    tidak bergantung urutan iterasi dict.
    """
    return min(kandidat, key=lambda v: (-kandidat[v][KRITERIA], str(v)))

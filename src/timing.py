"""Pengukur waktu per tahap untuk jalur ujung-ke-ujung.

KNF-10 menuntut waktu tanggap dari masukan sampai rekomendasi tampil. Tolok ukur per
komponen yang terpisah tidak sama dengan waktu yang benar-benar dirasakan dokter,
sehingga jalur ujung-ke-ujung diinstrumentasi sendiri di sini.

Modul ini sengaja sangat ringan: satu ``perf_counter`` per tahap, tanpa dependensi,
tanpa I/O. Overhead-nya di bawah satu mikrodetik per tahap, sehingga aman dinyalakan
di jalur produksi -- pengukuran yang hanya bisa dijalankan di bangku uji cenderung
tidak pernah mewakili keadaan sebenarnya.

Tahap dibedakan menjadi dua golongan, karena keduanya berperilaku sangat berbeda:

- **lokal**  : rekayasa fitur, prediksi, kalibrasi, penyusunan kueri, retrieval.
  Bergantung CPU mesin, dapat diprediksi, tidak pernah gagal karena jaringan.
- **jaringan**: generasi LLM. Bergantung layanan luar, variansinya besar, dan tunduk
  pada kuota. Proporsi kedua golongan inilah yang menentukan apakah sistem masih
  dapat dipakai saat LLM lambat atau kuotanya habis.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Final, Iterator, List, Optional

# Nama tahap baku. Dipakai bersama oleh pipeline, aplikasi, dan skrip benchmark
# supaya laporan dari ketiganya dapat disandingkan.
STAGE_FEATURES: Final[str] = "rekayasa_fitur"
STAGE_PREDICT: Final[str] = "prediksi"
STAGE_CALIBRATE: Final[str] = "kalibrasi_interval"
STAGE_QUERY: Final[str] = "penyusunan_kueri"
STAGE_RETRIEVE: Final[str] = "retrieval"
STAGE_GENERATE: Final[str] = "generasi_llm"

# Tahap yang menunggu layanan luar. Sisanya dihitung sebagai komputasi lokal.
NETWORK_STAGES: Final[frozenset] = frozenset({STAGE_GENERATE})

STAGE_ORDER: Final[List[str]] = [
    STAGE_FEATURES, STAGE_PREDICT, STAGE_CALIBRATE,
    STAGE_QUERY, STAGE_RETRIEVE, STAGE_GENERATE,
]


@dataclass
class StageTimer:
    """Kumpulkan durasi per tahap untuk SATU permintaan ujung-ke-ujung."""

    stages: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Ukur satu tahap. Durasi tetap tercatat walau tahapnya melempar.

        Ini disengaja: permintaan yang gagal di tengah justru yang paling menarik
        waktunya, dan mencatatnya hanya bila sukses membuat p95 terlihat lebih baik
        daripada kenyataan.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[stage] = self.stages.get(stage, 0.0) + (time.perf_counter() - t0)

    def record(self, stage: str, seconds: float) -> None:
        """Catat durasi yang sudah diukur di tempat lain."""
        self.stages[stage] = self.stages.get(stage, 0.0) + float(seconds)

    @property
    def total(self) -> float:
        return float(sum(self.stages.values()))

    @property
    def network_total(self) -> float:
        return float(sum(v for k, v in self.stages.items() if k in NETWORK_STAGES))

    @property
    def local_total(self) -> float:
        return float(sum(v for k, v in self.stages.items() if k not in NETWORK_STAGES))

    def network_share(self) -> Optional[float]:
        """Proporsi waktu yang dihabiskan menunggu LLM (0..1). ``None`` bila kosong."""
        t = self.total
        return (self.network_total / t) if t > 0 else None

    def as_dict(self) -> Dict[str, float]:
        """Durasi per tahap, urut sesuai alur, ditambah agregatnya."""
        ordered = {s: self.stages[s] for s in STAGE_ORDER if s in self.stages}
        for s in self.stages:  # tahap tak terduga tetap dilaporkan, tidak dibuang
            ordered.setdefault(s, self.stages[s])
        ordered["_total"] = self.total
        ordered["_lokal"] = self.local_total
        ordered["_jaringan"] = self.network_total
        return ordered


def percentile(values: List[float], p: float) -> float:
    """Persentil dengan interpolasi linear. p dalam 0..100.

    Ditulis sendiri alih-alih memakai numpy agar modul ini tetap tanpa dependensi
    dan dapat diimpor aplikasi tanpa biaya muat tambahan.
    """
    if not values:
        raise ValueError("percentile() atas daftar kosong")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))

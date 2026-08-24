"""Verifikasi sitasi: memeriksa apakah angka ber-penanda [S..] benar-benar ada pada
potongan yang ditunjuknya.

MENGAPA ADA
-----------
Aturan prompt yang melarang sitasi palsu TIDAK CUKUP. Diuji pada 24 Agustus 2026:
setelah aturan eksplisit ditambahkan, model tetap melekatkan [S4] pada ambang
hipoglikemia "Level 1 <= 70 mg/dL" dan "Level 2 < 54 mg/dL", padahal potongan S4
(PERKENI hal. 94, penghentian glukokortikoid) tidak memuat "54" maupun "Level 1"
di seluruh teksnya. Angkanya benar secara klinis — tetapi berasal dari ingatan
model, bukan dari dokumen yang dirujuk.

Selama penelusuran mengembalikan dokumen yang tidak relevan, model akan mengisi
lubang dari ingatannya lalu memberi penanda karena diminta memberi penanda.
Karena itu penjagaannya harus DETERMINISTIK dan tidak bergantung kepatuhan model.

MENGAPA HANYA ANGKA
-------------------
Verifikasi makna kalimat menuntut penilaian semantik, dan itu memperkenalkan
sumber galat baru yang harus dibela sendiri. Angka tidak: ia ada atau tidak ada.

Angka juga yang paling berbahaya bila dikarang (ambang, dosis, interval, laju),
paling mungkin diperiksa dokter, dan persis yang diatur Aturan 6 pada
SYSTEM_PROMPT. Klaim naratif yang keliru merugikan; ANGKA yang keliru berbahaya.

TIDAK MENGUBAH JAWABAN
----------------------
Modul ini MELAPORKAN, tidak menyunting. Mencopot penanda atau menghapus kalimat
secara otomatis akan mengubah teks klinis tanpa seorang pun meninjau hasilnya —
risiko yang lebih besar daripada yang ditutupnya. Keputusan penyajian diserahkan
ke lapisan tampilan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Angka: bilangan bulat atau desimal, koma maupun titik sebagai pemisah desimal.
# Pemisah ribuan tidak ditangani karena tidak muncul pada besaran klinis di korpus ini.
_ANGKA = re.compile(r"\d+(?:[.,]\d+)?")

# Penanda sitasi. Menerima [S1] dan bentuk gabungan [S1, S4] yang dipakai model.
_PENANDA = re.compile(r"\[S\s*(\d+(?:\s*,\s*S?\s*\d+)*)\s*\]", re.IGNORECASE)

# Toleransi pembandingan. Nilai dianggap sama bila selisihnya di bawah ini —
# menampung 3.4 vs 3.40 dan pembulatan penyajian, tanpa menyamakan 54 dengan 55.
_TOLERANSI = 1e-6


def _angka_dari(teks: str) -> List[float]:
    """Seluruh angka pada teks, sebagai float. Koma desimal disamakan dengan titik."""
    keluar: List[float] = []
    for m in _ANGKA.finditer(teks or ""):
        try:
            keluar.append(float(m.group(0).replace(",", ".")))
        except ValueError:  # pragma: no cover — regex sudah menjamin bentuknya
            continue
    return keluar


def _sama(a: float, b: float) -> bool:
    return abs(a - b) <= _TOLERANSI


def _nomor_penanda(fragmen: str) -> List[int]:
    """Nomor S pada satu penanda. "[S1, S4]" -> [1, 4]."""
    return [int(n) for n in re.findall(r"\d+", fragmen)]


@dataclass
class TemuanAngka:
    """Satu angka pada kalimat yang membawa penanda sitasi."""

    nilai: float
    kalimat: str
    penanda: List[int]
    status: str          # "dokumen" | "data_pasien" | "TIDAK_TERVERIFIKASI"
    ditemukan_pada: Optional[int] = None   # nomor S tempat angka itu benar-benar ada

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nilai": self.nilai,
            "status": self.status,
            "penanda_diklaim": self.penanda,
            "ditemukan_pada": self.ditemukan_pada,
            "kalimat": self.kalimat.strip(),
        }


@dataclass
class HasilVerifikasi:
    temuan: List[TemuanAngka] = field(default_factory=list)

    @property
    def tidak_terverifikasi(self) -> List[TemuanAngka]:
        return [t for t in self.temuan if t.status == "TIDAK_TERVERIFIKASI"]

    @property
    def n_diperiksa(self) -> int:
        return len(self.temuan)

    @property
    def proporsi_terverifikasi(self) -> float:
        """Porsi angka ber-penanda yang dapat ditelusuri ke dokumen atau data pasien.

        Bernilai 1.0 bila tidak ada angka ber-penanda sama sekali — tidak ada klaim,
        tidak ada yang gagal diverifikasi.
        """
        if not self.temuan:
            return 1.0
        lolos = sum(1 for t in self.temuan if t.status != "TIDAK_TERVERIFIKASI")
        return lolos / len(self.temuan)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_angka_diperiksa": self.n_diperiksa,
            "n_tidak_terverifikasi": len(self.tidak_terverifikasi),
            "proporsi_terverifikasi": round(self.proporsi_terverifikasi, 4),
            "temuan": [t.as_dict() for t in self.temuan],
        }


def _pecah_kalimat(teks: str) -> List[str]:
    """Pecah kasar menjadi kalimat.

    Pemecahan tidak perlu sempurna: unit yang dinilai adalah "angka dan penanda yang
    berdekatan". Memecah terlalu halus berisiko memisahkan angka dari penandanya dan
    menghasilkan lapor palsu, sehingga pemecahan sengaja dibuat KONSERVATIF —
    hanya pada akhir kalimat yang diikuti spasi dan huruf besar, atau baris baru.
    """
    if not teks:
        return []
    kasar = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+", teks)
    return [k for k in (s.strip() for s in kasar) if k]


def verifikasi_angka_bersitasi(
    jawaban: str,
    potongan: Sequence[Dict[str, Any]],
    konteks_pasien: Optional[str] = None,
    nilai_pasien: Optional[Iterable[float]] = None,
) -> HasilVerifikasi:
    """Periksa tiap angka pada kalimat ber-penanda terhadap potongan yang ditunjuk.

    Args:
        jawaban: teks advisory dari model bahasa.
        potongan: dokumen terambil, URUT sesuai penomoran [S1], [S2], ... Tiap
            elemen minimal memiliki kunci ``text``.
        konteks_pasien: blok konteks prediksi yang diberikan ke model. Angka di
            sini SAH walau tidak ada di dokumen — ia data pasien, bukan klaim
            pedoman.
        nilai_pasien: angka tambahan yang dianggap berasal dari data pasien.

    Status tiap angka:
        ``dokumen``              ada pada salah satu potongan yang ditandai
        ``data_pasien``          ada pada konteks pasien / nilai yang diberikan
        ``TIDAK_TERVERIFIKASI``  tidak ada di keduanya — kandidat sitasi palsu
    """
    hasil = HasilVerifikasi()
    if not jawaban:
        return hasil

    teks_potongan = [str(d.get("text", "") or "") for d in (potongan or [])]
    angka_potongan = [_angka_dari(t) for t in teks_potongan]

    angka_pasien = list(_angka_dari(konteks_pasien or ""))
    if nilai_pasien:
        angka_pasien.extend(float(v) for v in nilai_pasien)

    for kalimat in _pecah_kalimat(jawaban):
        penanda_mentah = _PENANDA.findall(kalimat)
        if not penanda_mentah:
            # Tanpa penanda tidak ada klaim penelusuran yang dibuat. Aturan 8
            # SYSTEM_PROMPT memang membolehkan menulis tanpa penanda.
            continue

        nomor: List[int] = []
        for frag in penanda_mentah:
            nomor.extend(_nomor_penanda(frag))
        nomor = sorted(set(nomor))

        # Angka pada penanda itu sendiri ([S4] -> 4) tidak boleh ikut diperiksa.
        tanpa_penanda = _PENANDA.sub(" ", kalimat)

        for nilai in _angka_dari(tanpa_penanda):
            sumber: Optional[int] = None
            for n in nomor:
                idx = n - 1
                if 0 <= idx < len(angka_potongan) and any(
                    _sama(nilai, a) for a in angka_potongan[idx]
                ):
                    sumber = n
                    break

            if sumber is not None:
                status = "dokumen"
            elif any(_sama(nilai, a) for a in angka_pasien):
                status = "data_pasien"
            else:
                status = "TIDAK_TERVERIFIKASI"

            hasil.temuan.append(
                TemuanAngka(
                    nilai=nilai,
                    kalimat=kalimat,
                    penanda=nomor,
                    status=status,
                    ditemukan_pada=sumber,
                )
            )

    return hasil

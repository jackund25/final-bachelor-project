"""T4.2 — menyambungkan logbook manual ke jalur prediksi.

Sebelum ini halaman "Input Logbook" menulis ``data/raw/manual_logbook.csv`` yang **tidak
pernah dibaca siapa pun**: ``streamlit_app.py`` hanya memuat ``ohio_t1dm``. Akibatnya
KF-01 dan KF-02 berstatus SEBAGIAN pada audit — kelima variabel dapat diisi, tetapi tidak
satu pun sampai ke model.

Modul ini melakukan penggabungan dan, yang sama pentingnya, **memeriksa apakah hasil
gabungan itu layak diprediksi**. Pemeriksaan tidak bisa dilewati, karena ada satu
ketidakcocokan mendasar:

    Model produksi dilatih HANYA pada jendela yang jarak antar-barisnya <= ``max_gap_steps``
    langkah (Tugas 5). Catatan logbook dimasukkan dokter pada waktu bebas — bisa berjarak
    berjam-jam. Jendela yang memuat catatan manual karena itu sangat mungkin berada di
    LUAR sebaran pelatihan.

Ada dua sikap yang mungkin diambil: menginterpolasi jeda supaya jendela "terlihat" rapat,
atau menolak dan mengatakan alasannya. Modul ini memilih yang kedua. Menginterpolasi jeda
delapan jam akan menghasilkan dua belas baris yang tampak sah bagi model dan bagi dokter,
padahal sebelas di antaranya karangan — persis kekeliruan yang membuat Tugas 5 diperlukan.

Logikanya ditaruh di ``src/`` dan bukan di lapisan antarmuka supaya dapat diuji tanpa
menjalankan aplikasi, mengikuti pola yang sama dengan ``src/alerts.py`` dan ``src/conformal.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

LOGBOOK_PATH = Path("data/raw/manual_logbook.csv")

#: Kolom yang benar-benar dipakai model. Kolom logbook lain (stress, sleep, work,
#: illness, meal_type, notes) tetap tersimpan untuk rekam jejak klinis tetapi TIDAK
#: menjadi fitur: model produksi dilatih tanpa kolom-kolom itu, dan menambahkannya di
#: waktu inferensi akan membuat bentuk masukan tidak cocok dengan scaler.
KOLOM_MODEL = ["timestamp", "patient_id", "glucose", "carbs", "insulin", "activity"]

SUMBER_DATASET = "dataset"
SUMBER_MANUAL = "manual"

# Verdict kelayakan jendela
LAYAK = "layak"
LUAR_SEBARAN = "luar_sebaran"
TIDAK_CUKUP = "tidak_cukup"
# Observasi terlalu berdekatan sehingga dianggap near-duplicate. Pelatihan membuang
# jendela semacam ini lewat min_history_interval_min; tanpa penjaga yang sama di sisi
# penyajian, model diberi jendela yang tidak pernah ia lihat.
TERLALU_RAPAT = "terlalu_rapat"


@dataclass
class KelayakanJendela:
    """Hasil pemeriksaan apakah satu jendela boleh dimasukkan ke model produksi."""

    verdict: str
    n_baris: int
    n_dibutuhkan: int
    n_manual: int = 0
    jeda_maks_langkah: Optional[float] = None
    batas_langkah: Optional[int] = None
    alasan: str = ""
    # Satuan menit — dipakai jalur modalitas (CGM/finger-stick) yang cadence-nya
    # berbeda, sehingga "langkah" tidak lagi bermakna sama di kedua jalur.
    jeda_maks_menit: Optional[float] = None
    batas_jeda_menit: Optional[float] = None
    interval_min_menit: Optional[float] = None
    batas_interval_menit: Optional[float] = None

    @property
    def boleh_diprediksi(self) -> bool:
        return self.verdict == LAYAK


@dataclass
class HasilGabung:
    """Deret waktu gabungan beserta asal-usulnya, supaya dapat dilaporkan apa adanya."""

    deret: pd.DataFrame
    n_dataset: int = 0
    n_manual: int = 0
    n_manual_menimpa: int = 0
    kolom_diabaikan: list = field(default_factory=list)

    @property
    def ada_manual(self) -> bool:
        return self.n_manual > 0


def baca_logbook(path: Path | str = LOGBOOK_PATH) -> pd.DataFrame:
    """Baca logbook manual. Mengembalikan DataFrame KOSONG bila berkas belum ada.

    Berkas yang belum ada adalah keadaan normal (dokter belum mencatat apa pun), bukan
    kesalahan — karena itu tidak melempar exception.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=KOLOM_MODEL)
    try:
        df = pd.read_csv(p)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=KOLOM_MODEL)
    if df.empty:
        return pd.DataFrame(columns=KOLOM_MODEL)
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    return df[df["timestamp"].notna()].reset_index(drop=True)


def gabung_dengan_dataset(
    dataset_df: pd.DataFrame,
    logbook_df: pd.DataFrame,
    patient_id: str,
) -> HasilGabung:
    """Gabungkan catatan manual satu pasien ke deret dataset pasien tersebut.

    Aturan yang dipakai, semuanya dilaporkan kembali lewat :class:`HasilGabung`:

    * Hanya baris dengan ``patient_id`` yang sama yang digabungkan.
    * Baris manual tanpa ``glucose`` dibuang — glukosa adalah fitur jangkar dan tidak
      dapat ditebak dari kolom lain.
    * Bila stempel waktu manual bertabrakan dengan baris dataset, **baris manual menang**.
      Catatan dokter adalah pengamatan yang lebih baru terhadap keadaan yang sama.
    * Kolom logbook di luar :data:`KOLOM_MODEL` diabaikan dan dicatat namanya, supaya
      terlihat bahwa stress/sleep/work/illness memang tidak masuk model.
    """
    ds = dataset_df[dataset_df["patient_id"] == patient_id].copy()
    ds["timestamp"] = pd.to_datetime(ds["timestamp"], errors="coerce")
    ds = ds[ds["timestamp"].notna()]
    ds["sumber"] = SUMBER_DATASET

    if logbook_df is None or logbook_df.empty:
        ds = ds.sort_values("timestamp").reset_index(drop=True)
        return HasilGabung(deret=ds, n_dataset=len(ds))

    lb = logbook_df[logbook_df["patient_id"].astype(str) == str(patient_id)].copy()
    diabaikan = sorted(set(lb.columns) - set(KOLOM_MODEL) - {"sumber"})
    if lb.empty:
        ds = ds.sort_values("timestamp").reset_index(drop=True)
        return HasilGabung(deret=ds, n_dataset=len(ds), kolom_diabaikan=diabaikan)

    lb["timestamp"] = pd.to_datetime(lb["timestamp"], errors="coerce")
    lb = lb[lb["timestamp"].notna()]
    if "glucose" not in lb.columns:
        ds = ds.sort_values("timestamp").reset_index(drop=True)
        return HasilGabung(deret=ds, n_dataset=len(ds), kolom_diabaikan=diabaikan)
    lb["glucose"] = pd.to_numeric(lb["glucose"], errors="coerce")
    lb = lb[lb["glucose"].notna()]

    for kol in ("carbs", "insulin", "activity"):
        lb[kol] = pd.to_numeric(lb.get(kol), errors="coerce").fillna(0.0)

    lb = lb[[c for c in KOLOM_MODEL if c in lb.columns]].copy()
    lb["sumber"] = SUMBER_MANUAL
    lb = lb.drop_duplicates(subset=["timestamp"], keep="first")

    tabrakan = int(ds["timestamp"].isin(set(lb["timestamp"])).sum())
    ds = ds[~ds["timestamp"].isin(set(lb["timestamp"]))]

    gabungan = (pd.concat([ds, lb], ignore_index=True)
                .sort_values("timestamp")
                .reset_index(drop=True))
    gabungan["patient_id"] = patient_id
    return HasilGabung(
        deret=gabungan,
        n_dataset=int((gabungan["sumber"] == SUMBER_DATASET).sum()),
        n_manual=int((gabungan["sumber"] == SUMBER_MANUAL).sum()),
        n_manual_menimpa=tabrakan,
        kolom_diabaikan=diabaikan,
    )


def periksa_kelayakan_menit(
    jendela_df: pd.DataFrame,
    sequence_length: int,
    max_gap_min: Optional[float],
    min_interval_min: float = 0.0,
) -> KelayakanJendela:
    """Apakah jendela ini sepadan dengan jendela yang dipakai melatih model produksi?

    Bekerja dalam MENIT, sehingga berlaku untuk semua modalitas: CGM yang cadence-nya
    padat maupun finger-stick yang tidak teratur. Kriterianya harus IDENTIK dengan yang
    dipakai ``create_time_horizon_sequences`` saat membentuk jendela pelatihan:

    * ``max_gap_min``       — jeda terpanjang di dalam jendela riwayat;
    * ``min_interval_min``  — jarak minimum antar-observasi, untuk membuang pembacaan
      near-duplicate yang lazim pada finger-stick.

    Kalau kriteria di sini lebih longgar daripada kriteria pelatihan, aplikasi akan
    menyajikan prediksi atas jendela yang modelnya tidak pernah lihat — tanpa tanda apa
    pun bagi dokter. Itu kegagalan senyap, jenis yang paling mahal pada alat klinis.
    """
    n = len(jendela_df)
    if n < sequence_length:
        return KelayakanJendela(
            verdict=TIDAK_CUKUP, n_baris=n, n_dibutuhkan=sequence_length,
            alasan=f"jendela hanya berisi {n} dari {sequence_length} baris yang dibutuhkan",
        )

    w = jendela_df.tail(sequence_length)
    n_manual = int((w.get("sumber") == SUMBER_MANUAL).sum()) if "sumber" in w else 0

    if max_gap_min is None:
        return KelayakanJendela(
            verdict=LUAR_SEBARAN, n_baris=n, n_dibutuhkan=sequence_length,
            n_manual=n_manual,
            alasan="batas jeda tidak diketahui, kelayakan jendela tidak dapat dipastikan",
        )

    ts = pd.to_datetime(w["timestamp"])
    selisih = ts.diff().dt.total_seconds().div(60.0).dropna()

    jeda_maks = float(selisih.max()) if len(selisih) else 0.0
    interval_min = float(selisih.min()) if len(selisih) else 0.0

    if jeda_maks > float(max_gap_min):
        return KelayakanJendela(
            verdict=LUAR_SEBARAN, n_baris=n, n_dibutuhkan=sequence_length,
            n_manual=n_manual,
            jeda_maks_menit=round(jeda_maks, 2), batas_jeda_menit=float(max_gap_min),
            alasan=(f"jeda terpanjang {jeda_maks:.0f} menit melampaui batas "
                    f"{float(max_gap_min):.0f} menit yang dipakai saat pelatihan"),
        )

    if min_interval_min > 0 and len(selisih) and interval_min < float(min_interval_min):
        return KelayakanJendela(
            verdict=TERLALU_RAPAT, n_baris=n, n_dibutuhkan=sequence_length,
            n_manual=n_manual,
            jeda_maks_menit=round(jeda_maks, 2), batas_jeda_menit=float(max_gap_min),
            interval_min_menit=round(interval_min, 2),
            batas_interval_menit=float(min_interval_min),
            alasan=(f"terdapat observasi berjarak {interval_min:.0f} menit, lebih rapat "
                    f"daripada batas {float(min_interval_min):.0f} menit saat pelatihan"),
        )

    return KelayakanJendela(
        verdict=LAYAK, n_baris=n, n_dibutuhkan=sequence_length, n_manual=n_manual,
        jeda_maks_menit=round(jeda_maks, 2), batas_jeda_menit=float(max_gap_min),
        interval_min_menit=round(interval_min, 2) if len(selisih) else None,
        batas_interval_menit=float(min_interval_min) if min_interval_min else None,
        alasan="jarak antar-baris sepadan dengan jendela pelatihan",
    )


def periksa_kelayakan(
    jendela_df: pd.DataFrame,
    sequence_length: int,
    max_gap_steps: Optional[int],
    cadence_min: float = 5.0,
) -> KelayakanJendela:
    """Pembungkus berbasis LANGKAH atas ``periksa_kelayakan_menit``.

    Dipertahankan karena jalur berbasis langkah masih dipakai pengujian dan skrip era
    sebelum parser disatukan. Ia hanya mengubah satuan lalu mendelegasikan, sehingga
    tidak ada dua salinan aturan kelayakan yang dapat menyimpang diam-diam.
    """
    hasil = periksa_kelayakan_menit(
        jendela_df,
        sequence_length=sequence_length,
        max_gap_min=(None if max_gap_steps is None
                     else float(max_gap_steps) * float(cadence_min)),
    )

    # Medan berbasis langkah diisi ulang supaya pemanggil lama tetap terlayani.
    if hasil.jeda_maks_menit is not None:
        hasil.jeda_maks_langkah = round(hasil.jeda_maks_menit / float(cadence_min), 2)
    if max_gap_steps is not None:
        hasil.batas_langkah = int(max_gap_steps)
    if hasil.verdict == LUAR_SEBARAN and max_gap_steps is None:
        hasil.alasan = (
            "max_gap_steps tidak diketahui, kelayakan jendela tidak dapat dipastikan"
        )

    return hasil

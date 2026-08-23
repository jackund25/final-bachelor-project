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

Logikanya ditaruh di ``src/`` dan bukan di ``app/`` supaya dapat diuji tanpa menjalankan
Streamlit, mengikuti pola yang sama dengan ``src/alerts.py`` dan ``src/conformal.py``.
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


def periksa_kelayakan(
    jendela_df: pd.DataFrame,
    sequence_length: int,
    max_gap_steps: Optional[int],
    cadence_min: float = 5.0,
) -> KelayakanJendela:
    """Apakah jendela ini sepadan dengan jendela yang dipakai melatih model produksi?

    Memakai kriteria yang IDENTIK dengan ``create_sequences(max_gap_steps=...)``: jarak
    antar-baris berturut-turut tidak boleh melampaui ``max_gap_steps`` langkah. Kalau
    kriteria di sini lebih longgar daripada kriteria pelatihan, aplikasi akan menyajikan
    prediksi atas jendela yang modelnya tidak pernah lihat, tanpa tanda apa pun.
    """
    n = len(jendela_df)
    if n < sequence_length:
        return KelayakanJendela(
            verdict=TIDAK_CUKUP, n_baris=n, n_dibutuhkan=sequence_length,
            alasan=f"jendela hanya berisi {n} dari {sequence_length} baris yang dibutuhkan",
        )

    w = jendela_df.tail(sequence_length)
    n_manual = int((w.get("sumber") == SUMBER_MANUAL).sum()) if "sumber" in w else 0

    if max_gap_steps is None:
        return KelayakanJendela(
            verdict=LUAR_SEBARAN, n_baris=n, n_dibutuhkan=sequence_length,
            n_manual=n_manual,
            alasan="max_gap_steps tidak diketahui, kelayakan jendela tidak dapat dipastikan",
        )

    ts = pd.to_datetime(w["timestamp"])
    selisih_menit = ts.diff().dt.total_seconds().div(60.0).dropna()
    jeda_langkah = (selisih_menit / float(cadence_min)) if len(selisih_menit) else selisih_menit
    jeda_maks = float(jeda_langkah.max()) if len(jeda_langkah) else 0.0

    if jeda_maks > float(max_gap_steps):
        return KelayakanJendela(
            verdict=LUAR_SEBARAN, n_baris=n, n_dibutuhkan=sequence_length,
            n_manual=n_manual, jeda_maks_langkah=round(jeda_maks, 2),
            batas_langkah=int(max_gap_steps),
            alasan=(f"jeda terpanjang {jeda_maks * cadence_min:.0f} menit melampaui batas "
                    f"{max_gap_steps * cadence_min:.0f} menit yang dipakai saat pelatihan"),
        )

    return KelayakanJendela(
        verdict=LAYAK, n_baris=n, n_dibutuhkan=sequence_length, n_manual=n_manual,
        jeda_maks_langkah=round(jeda_maks, 2), batas_langkah=int(max_gap_steps),
        alasan="jarak antar-baris sepadan dengan jendela pelatihan",
    )

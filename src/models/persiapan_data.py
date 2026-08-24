"""Persiapan data bersama untuk seluruh lengan model prediksi glukosa.

LATAR
-----
Perbandingan tiga model (RF, GBM, LSTM) hanya sah bila ketiganya menerima data yang
disiapkan dengan cara yang PERSIS sama. Sebelum modul ini, tiap modul model menyalin
sendiri langkah pemuatan, pembersihan, rekayasa fitur, pembagian, dan pembentukan
jendela — sehingga perbedaan angka dapat berasal dari perbedaan persiapan, bukan dari
modelnya, dan itu tidak akan terlihat dari angka mana pun.

Modul ini menjadi satu-satunya sumber kebenaran untuk langkah tersebut.

PERUBAHAN YANG DIWADAHI
-----------------------
Parser OhioT1DM kini menghasilkan SATU berkas gabungan ``data/raw/ohio_t1dm.csv``
dengan metadata ``glucose_source`` (CGM / FINGER_STICK) dan ``dataset_split``
(train / test, mengikuti pembagian resmi berkas XML). Akibatnya:

* jendela dibentuk berdasarkan WAKTU NYATA (menit), bukan jumlah langkah — API
  ``create_sequences(prediction_horizon=<langkah>)`` sudah dicabut dan digantikan
  ``create_time_horizon_sequences(horizon_min=<menit>)``;
* tiap modalitas punya profil sendiri pada ``config.model.source_profiles``;
* pembagian latih/uji mengikuti berkas resmi, bukan rasio maupun pemotongan pasien.

CATATAN PENTING TENTANG PEMBAGIAN RESMI
---------------------------------------
Pembagian resmi OhioT1DM bersifat TEMPORAL DALAM-PASIEN: seluruh 12 pasien muncul di
train MAUPUN test, pada periode waktu yang berbeda. Ia BUKAN pembagian lintas-pasien.
Angka yang dihasilkan modul ini karena itu tidak dapat disebut "pasien uji tak pernah
dilihat model". Uji generalisasi lintas-pasien adalah percobaan terpisah; lihat
skrip crossval pada ``scripts/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.loader import DiabetesDataLoader
from src.data.preprocessor import DataPreprocessor

SUMBER_SAH = {"CGM", "FINGER_STICK"}

METADATA_WAJIB = ["glucose_source", "dataset_split"]


@dataclass
class ProfilSumber:
    """Parameter temporal satu modalitas, dibaca dari config.model.source_profiles."""

    sumber: str
    sequence_length: int
    horizons_min: List[float]
    target_tolerance_min: float
    max_history_gap_min: Optional[float]
    min_history_interval_min: float
    min_target_horizon_min: Optional[float]
    max_target_horizon_min: Optional[float]

    def sebagai_kwargs(self, horizon_min: float) -> Dict[str, Any]:
        """Argumen create_time_horizon_sequences untuk satu horizon."""
        return {
            "sequence_length": self.sequence_length,
            "horizon_min": float(horizon_min),
            "target_tolerance_min": self.target_tolerance_min,
            "max_history_gap_min": self.max_history_gap_min,
            "min_target_horizon_min": self.min_target_horizon_min,
            "max_target_horizon_min": self.max_target_horizon_min,
            "min_history_interval_min": self.min_history_interval_min,
        }


@dataclass
class DataSiap:
    """Hasil persiapan: bingkai yang sudah dibagi beserta alat dan parameternya."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    preprocessor: DataPreprocessor
    profil: ProfilSumber
    feature_list: List[str]
    predict_delta: bool
    feature_engineering: Dict[str, Any] = field(default_factory=dict)
    sumber_data: str = "ohio_t1dm_unified"


def resolve_profil_sumber(
    config: Dict[str, Any],
    sumber: str,
    horizon_min: Optional[float] = None,
) -> ProfilSumber:
    """Baca profil modalitas dari config, dengan cadangan ke nilai global."""
    profil_cfg = (
        config.get("model", {})
        .get("source_profiles", {})
        .get(sumber, {})
    )

    if horizon_min is None:
        horizons = [
            float(h)
            for h in profil_cfg.get("prediction_horizons_min", [30])
        ]
    else:
        horizons = [float(horizon_min)]

    return ProfilSumber(
        sumber=sumber,
        sequence_length=int(
            profil_cfg.get(
                "sequence_length",
                config.get("model", {}).get("sequence_length", 12),
            )
        ),
        horizons_min=horizons,
        target_tolerance_min=float(
            profil_cfg.get("target_tolerance_min", 0.0)
        ),
        max_history_gap_min=profil_cfg.get("max_history_gap_min"),
        min_history_interval_min=float(
            profil_cfg.get("min_history_interval_min", 0.0)
        ),
        min_target_horizon_min=profil_cfg.get("min_target_horizon_min"),
        max_target_horizon_min=profil_cfg.get("max_target_horizon_min"),
    )


@dataclass
class DatasetModalitas:
    """Satu modalitas yang sudah bersih dan berfitur, TANPA dibagi.

    Dipakai percobaan yang menyusun pembagiannya sendiri — terutama lengan
    generalisasi LINTAS-PASIEN pada ``scripts/``, yang menyisihkan pasien utuh alih-alih
    memakai pembagian resmi. Keduanya sah dan menjawab pertanyaan yang berbeda:

    * pembagian resmi menjawab "seberapa baik model memprakirakan kelanjutan deret
      pasien yang riwayatnya sudah pernah dilihat";
    * pembagian lintas-pasien menjawab "seberapa baik model menghadapi pasien baru".

    Yang tidak boleh adalah mencampur angka keduanya dalam satu tabel tanpa menyebut
    rancangannya.
    """

    df: pd.DataFrame
    preprocessor: DataPreprocessor
    profil: ProfilSumber
    feature_list: List[str]
    predict_delta: bool
    feature_engineering: Dict[str, Any] = field(default_factory=dict)
    sumber_data: str = "ohio_t1dm_unified"


def muat_dataset_modalitas(
    config: Dict[str, Any],
    sumber: str = "CGM",
    data_source: str = "auto",
    horizon_min: Optional[float] = None,
) -> DatasetModalitas:
    """Muat, bersihkan, dan rekayasa fitur satu modalitas — tanpa membagi.

    Langkah-langkahnya PERSIS sama dengan yang dipakai jalur produksi; yang tidak
    dilakukan hanyalah pembagian latih/uji, supaya pemanggil bebas menyusun
    pembagiannya sendiri.
    """
    if sumber not in SUMBER_SAH:
        raise ValueError(
            f"sumber harus salah satu dari {sorted(SUMBER_SAH)}, bukan {sumber!r}"
        )

    loader = DiabetesDataLoader(config["data"]["output_dir"])

    if data_source in ("auto", "ohio_t1dm"):
        df = loader.load_csv(
            config["data"].get("unified_dataset", "ohio_t1dm.csv")
        )
        sumber_data = "ohio_t1dm_unified"
    else:
        df = loader.load_latest_dataset()
        sumber_data = "latest_generated"

    hilang = [c for c in METADATA_WAJIB if c not in df.columns]
    if hilang:
        raise ValueError(
            "Dataset gabungan OhioT1DM kehilangan metadata: "
            f"{hilang}. Jalankan `python -m src.data.ohio_parser` lebih dulu."
        )

    df = df[df["glucose_source"] == sumber].copy()
    if df.empty:
        raise ValueError(f"Tidak ada baris untuk glucose_source={sumber}")

    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    preprocessor = DataPreprocessor(config)

    df = preprocessor.handle_missing_values(
        df,
        max_interpolate_steps=config["model"].get("max_interpolate_steps"),
    )

    fe_params = config["model"].get("feature_engineering", {})
    df = preprocessor.engineer_features(df, **fe_params)

    feature_list = list(
        config["model"].get(
            "engineered_features",
            config["model"]["features"],
        )
    )
    preprocessor.feature_columns = list(feature_list)

    return DatasetModalitas(
        df=df,
        preprocessor=preprocessor,
        profil=resolve_profil_sumber(config, sumber, horizon_min),
        feature_list=feature_list,
        predict_delta=bool(config["model"].get("predict_delta", True)),
        feature_engineering=fe_params,
        sumber_data=sumber_data,
    )


def siapkan_data(
    config: Dict[str, Any],
    sumber: str = "CGM",
    data_source: str = "auto",
    horizon_min: Optional[float] = None,
) -> DataSiap:
    """Muat, bersihkan, rekayasa fitur, dan bagi menurut pembagian resmi OhioT1DM.

    Urutan langkahnya SENGAJA identik dengan train_gbm_from_config, sebab lengan
    pembanding yang menerima perlakuan berbeda tidak dapat dibandingkan.
    """
    modal = muat_dataset_modalitas(config, sumber, data_source, horizon_min)
    df = modal.df
    preprocessor = modal.preprocessor
    feature_list = modal.feature_list
    fe_params = modal.feature_engineering
    sumber_data = modal.sumber_data

    train_df = df[df["dataset_split"] == "train"].copy()
    test_df = df[df["dataset_split"] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Pembagian resmi train dan test keduanya harus berisi baris."
        )

    return DataSiap(
        train_df=train_df,
        test_df=test_df,
        preprocessor=preprocessor,
        profil=modal.profil,
        feature_list=feature_list,
        predict_delta=modal.predict_delta,
        feature_engineering=fe_params,
        sumber_data=sumber_data,
    )


def berkas_prediksi(nama_model: str, sumber: str, horizon_min: float) -> Path:
    """Lokasi baku vektor prediksi satu lengan pada satu horizon."""
    return (
        Path("results")
        / "eval_prediksi"
        / f"prediksi_{nama_model.lower()}_{sumber.lower()}_h{int(horizon_min)}m.npz"
    )


def simpan_prediksi(
    nama_model: str,
    sumber: str,
    horizon_min: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anchor: Optional[np.ndarray] = None,
) -> Path:
    """Simpan vektor prediksi supaya gambar dapat dibuat tanpa melatih ulang.

    LATAR. Gambar Clarke Error Grid pada Bab VI dahulu dihasilkan skrip yang MELATIH
    ULANG ketiga model, terpisah dari pelatihan yang menghasilkan tabel metriknya.
    Dua jalur pelatihan yang berbeda untuk satu klaim membuka celah yang pernah
    benar-benar terjadi di proyek ini: gambar memperlihatkan model yang bukan model
    produksi, dan tidak ada yang menyadarinya karena angkanya tidak pernah
    dibandingkan langsung.

    Dengan menyimpan vektor prediksi pada saat pelatihan, gambar dan tabel dijamin
    berasal dari model yang SAMA, dan gambar dapat digambar ulang kapan pun tanpa
    biaya komputasi.

    Berkasnya kecil: 30 ribu prediksi float64 sekitar 240 KB.
    """
    jalur = berkas_prediksi(nama_model, sumber, horizon_min)
    jalur.parent.mkdir(parents=True, exist_ok=True)

    isi = {
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
        "horizon_min": np.asarray([float(horizon_min)]),
    }
    if anchor is not None:
        isi["anchor"] = np.asarray(anchor, dtype=float)

    np.savez_compressed(jalur, **isi)
    return jalur


def bentuk_jendela_bagian(
    modal: DatasetModalitas,
    bagian: pd.DataFrame,
    horizon_min: float,
    return_anchor: bool = True,
) -> Tuple[np.ndarray, ...]:
    """Bentuk jendela dari SEBAGIAN baris — untuk pembagian yang disusun pemanggil.

    Dipakai lengan generalisasi lintas-pasien: pemanggil menyisihkan pasien utuh,
    lalu meminta jendela untuk tiap lipatan. Parameter temporalnya tetap berasal dari
    profil modalitas, sehingga perlakuannya identik dengan jalur produksi dan selisih
    angka hanya berasal dari rancangan pembagiannya.

    Contoh:

        modal = muat_dataset_modalitas(cfg, "CGM")
        latih = modal.df[modal.df.patient_id.isin(pasien_latih)]
        X, y, anc = bentuk_jendela_bagian(modal, latih, 30.0)
    """
    return modal.preprocessor.create_time_horizon_sequences(
        bagian,
        return_anchor=return_anchor,
        **modal.profil.sebagai_kwargs(horizon_min),
    )


def bentuk_jendela(
    siap: DataSiap,
    horizon_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bentuk jendela latih dan uji untuk satu horizon.

    Mengembalikan (X_train, y_train, anc_train, X_test, y_test, anc_test).
    Jangkar (anchor) selalu diminta karena seluruh lengan memakai target delta bila
    ``predict_delta`` menyala, dan rekonstruksinya membutuhkan jangkar.
    """
    kwargs = siap.profil.sebagai_kwargs(horizon_min)

    X_train, y_train, anc_train = siap.preprocessor.create_time_horizon_sequences(
        siap.train_df, return_anchor=True, **kwargs
    )
    X_test, y_test, anc_test = siap.preprocessor.create_time_horizon_sequences(
        siap.test_df, return_anchor=True, **kwargs
    )

    return X_train, y_train, anc_train, X_test, y_test, anc_test

"""Kontrak persiapan data bersama untuk seluruh lengan model.

Yang dijaga berkas ini:

1. Ketiga lengan (RF, GBM, LSTM) menerima jendela yang PERSIS sama. Perbandingan
   model yang lengannya disiapkan berbeda tidak sah, dan perbedaan semacam itu tidak
   akan terlihat dari angka mana pun.
2. Profil modalitas dibaca dari config, bukan dipatri — CGM dan finger-stick memiliki
   karakteristik temporal yang berbeda dan tidak boleh diperlakukan sama.
3. Pembagian yang dipakai adalah pembagian resmi OhioT1DM, dan sifatnya TEMPORAL
   DALAM-PASIEN. Ini didokumentasikan sebagai tes supaya tidak ada yang melaporkannya
   sebagai pembagian lintas-pasien.
"""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _butuh_dataset():
    berkas = ROOT / "data" / "raw" / "ohio_t1dm.csv"
    if not berkas.exists():
        pytest.skip(
            "data/raw/ohio_t1dm.csv tidak ada (tunduk Data Use Agreement); "
            "jalankan `python -m src.data.ohio_parser`"
        )


def test_profil_modalitas_dibaca_dari_config(config):
    """CGM dan finger-stick tidak boleh berbagi parameter temporal."""
    from src.models.persiapan_data import resolve_profil_sumber

    cgm = resolve_profil_sumber(config, "CGM")
    smbg = resolve_profil_sumber(config, "FINGER_STICK")

    assert cgm.sequence_length == 12
    assert 30.0 in cgm.horizons_min
    assert cgm.target_tolerance_min > 0, (
        "CGM memakai toleransi simetris di sekitar horizon"
    )

    assert smbg.sequence_length == 8
    assert smbg.min_target_horizon_min is not None, (
        "finger-stick memakai jendela target eksplisit, bukan toleransi simetris"
    )
    assert smbg.max_target_horizon_min > smbg.min_target_horizon_min


def test_sumber_tidak_dikenal_ditolak(config):
    from src.models.persiapan_data import siapkan_data

    with pytest.raises(ValueError, match="sumber harus"):
        siapkan_data(config, sumber="CGM_PALSU")


def test_pembagian_resmi_bersifat_temporal_dalam_pasien(config):
    """Bukan lintas-pasien. Klaim sebaliknya pernah beredar di README.

    Seluruh 12 pasien muncul pada train MAUPUN test dengan periode waktu berbeda.
    Angka yang dihasilkan pembagian ini karena itu TIDAK dapat disebut "pasien uji
    tak pernah dilihat model".
    """
    _butuh_dataset()

    from src.models.persiapan_data import siapkan_data

    siap = siapkan_data(config, sumber="CGM")

    pasien_latih = set(siap.train_df["patient_id"].unique())
    pasien_uji = set(siap.test_df["patient_id"].unique())

    assert pasien_latih == pasien_uji, (
        "pembagian resmi OhioT1DM seharusnya memuat pasien yang sama di kedua sisi"
    )
    assert len(pasien_uji) == 12


def test_persiapan_bersama_mereproduksi_jendela_gbm(config):
    """Bukti perlakuan identik, bukan asumsi.

    Jendela yang dibentuk modul bersama harus sama persis dengan yang menghasilkan
    metrik GBM produksi. Bila jumlah sampelnya menyimpang, lengan pembanding sedang
    diukur pada himpunan yang berbeda dan seluruh perbandingan model gugur.
    """
    _butuh_dataset()

    acuan_path = ROOT / "models" / "gbm_cgm_h30m_metrics.json"
    if not acuan_path.exists():
        pytest.skip("metrik acuan GBM belum ada; latih GBM lebih dulu")

    with open(acuan_path, encoding="utf-8") as f:
        acuan = json.load(f)

    from src.models.persiapan_data import bentuk_jendela, siapkan_data

    siap = siapkan_data(config, sumber="CGM", horizon_min=30.0)
    X_train, y_train, _, X_test, y_test, _ = bentuk_jendela(siap, 30.0)

    assert len(y_train) == acuan["train_samples"]
    assert len(y_test) == acuan["test_samples"]
    assert X_train.shape[1] == acuan["sequence_length"]
    assert X_train.shape[2] == len(siap.feature_list)

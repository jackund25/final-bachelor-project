"""Baseline *persistence* pada konfigurasi FINGER_STICK produksi (~4 jam).

Motivasi
--------
Model FINGER_STICK produksi (``models/gbm_finger_stick_h4h_seq8_metrics.json``)
mencatat RMSE 68,24 mg/dL dengan Clarke A+B 52,52% dan zona E 23,02%. Angka itu
tidak dapat ditafsirkan sendirian: tanpa pembanding, ia tidak membedakan antara

    (a) model gagal belajar, dan
    (b) kadar glukosa 4 jam ke depan memang tidak dapat diprediksi dari
        pembacaan tusuk jari.

*Persistence* — dugaan naif bahwa glukosa tidak berubah dari pembacaan terakhir —
adalah pembanding yang memisahkan keduanya. Ia tidak menyentuh model sama sekali,
sehingga berlaku sebagai kontrol: bila persistence juga jatuh di kisaran yang sama,
yang terbatas adalah informasinya, bukan modelnya.

Kesetaraan yang dijaga
----------------------
Skrip ini TIDAK membangun ulang jendelanya sendiri. Ia memakai jalur penyiapan
data yang sama persis dengan ``train_gbm_from_config`` — pemuat, penanganan nilai
hilang, rekayasa fitur, profil sumber, dan ``create_time_horizon_sequences`` —
lalu berhenti tepat sebelum pelatihan. Dengan demikian himpunan sampel, split, dan
penyaring temporalnya dijamin identik, dan selisih angka yang muncul benar-benar
berasal dari prediktornya, bukan dari cara data disiapkan.

Prediksi persistence adalah nilai *anchor* itu sendiri, yaitu pembacaan glukosa
terakhir pada jendela riwayat. Pada jalur produksi nilai itu pula yang dipakai
merekonstruksi prediksi absolut dari target delta (Persamaan V.4), sehingga
persistence setara dengan model yang selalu menduga delta nol.

Keluaran: results/eval_prediksi/smbg_persistence_4h.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import DiabetesDataLoader  # noqa: E402
from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.utils.metrics import calculate_all_metrics  # noqa: E402

DEST = ROOT / "results/eval_prediksi/smbg_persistence_4h.json"

# Metrik model produksi, dibaca dari artefak agar perbandingannya tidak
# bersandar pada angka yang disalin tangan.
ARTEFAK = {
    "FINGER_STICK": ROOT / "models/gbm_finger_stick_h4h_seq8_metrics.json",
    "CGM": {
        30.0: ROOT / "models/gbm_cgm_h30m_metrics.json",
        60.0: ROOT / "models/gbm_cgm_h60m_metrics.json",
    },
}


def siapkan(config: dict, source: str):
    """Ulangi jalur penyiapan data train_gbm_from_config, tanpa melatih apa pun.

    URUTANNYA MENGIKAT. Produksi menyaring ``glucose_source`` lebih dulu
    (``src/models/gbm_model.py:391``), BARU merekayasa fitur. Membalik urutan itu
    membuat ``iob``, ``cob``, dan ``glucose_delta`` dihitung dari baris tetangga
    yang berbeda, sehingga angkanya tidak lagi setara dengan model produksi.
    """
    loader = DiabetesDataLoader(config["data"]["output_dir"])
    df = loader.load_csv(
        config["data"].get("unified_dataset", "ohio_t1dm.csv")
    )

    for kolom in ("glucose_source", "dataset_split"):
        if kolom not in df.columns:
            raise SystemExit(f"Dataset tidak memuat metadata {kolom}")

    df = df[df["glucose_source"] == source].copy()
    if df.empty:
        raise SystemExit(f"Tidak ada baris untuk glucose_source={source}")

    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    pre = DataPreprocessor(config)
    df = pre.handle_missing_values(
        df,
        max_interpolate_steps=config["model"].get("max_interpolate_steps"),
    )
    df = pre.engineer_features(
        df, **config["model"].get("feature_engineering", {})
    )
    pre.feature_columns = list(
        config["model"].get("engineered_features", config["model"]["features"])
    )
    return pre, df


def jendela(pre, df, source_cfg, sequence_length, horizon_min, split):
    """Bentuk jendela untuk satu split, kembalikan (target, anchor).

    Kanal sudah disaring di ``siapkan``; di sini tinggal memisahkan split resmi.
    """
    bagian = df[df["dataset_split"] == split]
    _, y, anchor = pre.create_time_horizon_sequences(
        bagian,
        sequence_length=sequence_length,
        horizon_min=float(horizon_min),
        target_tolerance_min=float(source_cfg.get("target_tolerance_min", 2.5)),
        max_history_gap_min=source_cfg.get("max_history_gap_min"),
        return_anchor=True,
        min_target_horizon_min=source_cfg.get("min_target_horizon_min"),
        max_target_horizon_min=source_cfg.get("max_target_horizon_min"),
        min_history_interval_min=float(
            source_cfg.get("min_history_interval_min", 0.0)
        ),
    )
    return y, anchor


def bulatkan(m: dict) -> dict:
    return {k: round(float(v), 2) for k, v in m.items()}


def evaluasi(config, source: str, horizon_min: float) -> dict:
    pre, d = siapkan(config, source)
    source_cfg = (
        config["model"].get("source_profiles", {}).get(source, {})
    )
    seq = int(
        source_cfg.get(
            "sequence_length", config["model"].get("sequence_length", 12)
        )
    )

    hasil = {
        "sequence_length": seq,
        "horizon_min": float(horizon_min),
        "min_target_horizon_min": source_cfg.get("min_target_horizon_min"),
        "max_target_horizon_min": source_cfg.get("max_target_horizon_min"),
    }

    for split in ("train", "test"):
        y, anchor = jendela(pre, d, source_cfg, seq, horizon_min, split)
        if len(y) == 0:
            hasil[split] = {"n": 0, "catatan": "tidak ada jendela yang layak"}
            continue

        m = calculate_all_metrics(y, anchor)          # persistence: y_pred = anchor
        delta = y - anchor
        hasil[split] = {
            "n": int(len(y)),
            "persistence": bulatkan(m),
            "perubahan_sebenarnya_mg_dl": {
                "mean_abs": round(float(np.abs(delta).mean()), 2),
                "median_abs": round(float(np.median(np.abs(delta))), 2),
                "p90_abs": round(float(np.percentile(np.abs(delta), 90)), 2),
                "maks_abs": round(float(np.abs(delta).max()), 2),
                "sd": round(float(delta.std()), 2),
            },
        }
    return hasil


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    out = {
        "catatan": (
            "Persistence = glukosa tidak berubah dari pembacaan terakhir "
            "(prediksi = nilai anchor). Jendela dibentuk lewat jalur penyiapan "
            "data yang sama persis dengan train_gbm_from_config, sehingga "
            "himpunan sampel dan splitnya identik dengan model produksi."
        ),
        "sumber_data": config["data"].get("unified_dataset", "ohio_t1dm.csv"),
        "split": "official_train_test",
        "hasil": {},
        "pembanding_model_produksi": {},
    }

    # --- FINGER_STICK pada horizon produksinya (~4 jam) --------------------
    fs_cfg = config["model"]["source_profiles"]["FINGER_STICK"]
    for h in fs_cfg.get("prediction_horizons_min", [240]):
        kunci = f"FINGER_STICK_+{int(h)}"
        out["hasil"][kunci] = evaluasi(config, "FINGER_STICK", h)

    # --- CGM sebagai titik acuan (opsional) -------------------------------
    # Bukan sasaran tahap ini. Berguna sebagai skala pembanding, tetapi mahal:
    # kanal CGM menghasilkan ~130 ribu jendela berukuran 12x9, dan
    # create_time_horizon_sequences menyusunnya sebagai daftar Python sebelum
    # dijadikan array — cukup untuk menghabiskan memori pada mesin kelas
    # konsumen. Karena itu dijadikan opt-in.
    if "--dengan-cgm" in sys.argv:
        cgm_cfg = config["model"]["source_profiles"]["CGM"]
        for h in cgm_cfg.get("prediction_horizons_min", [30, 60]):
            kunci = f"CGM_+{int(h)}"
            out["hasil"][kunci] = evaluasi(config, "CGM", h)

    # --- metrik model produksi, dibaca dari artefak ------------------------
    p = ARTEFAK["FINGER_STICK"]
    if p.exists():
        out["pembanding_model_produksi"]["FINGER_STICK_+240"] = json.loads(
            p.read_text(encoding="utf-8")
        )
    for h, p in ARTEFAK["CGM"].items():
        if p.exists():
            out["pembanding_model_produksi"][f"CGM_+{int(h)}"] = json.loads(
                p.read_text(encoding="utf-8")
            )

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- ringkas
    print(f"{'konfigurasi':22s} {'n uji':>7s} {'RMSE':>8s} {'A+B %':>8s} {'E %':>7s}")
    print("-" * 56)
    for kunci, h in out["hasil"].items():
        uji = h.get("test", {})
        if not uji.get("n"):
            print(f"{kunci:22s} {'-':>7s}  tidak ada jendela")
            continue
        pers = uji["persistence"]
        print(f"{kunci + ' persistence':22s} {uji['n']:7d} "
              f"{pers['RMSE']:8.2f} {pers['Clarke_A+B']:8.2f} {pers['Clarke_E']:7.2f}")
        model = out["pembanding_model_produksi"].get(kunci.replace("_+", "_+"))
        if model:
            print(f"{kunci + ' model':22s} {model['test_samples']:7d} "
                  f"{model['RMSE']:8.2f} {model['Clarke_A+B']:8.2f} "
                  f"{model.get('Clarke_E', float('nan')):7.2f}")

    print(f"\nDisimpan ke {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

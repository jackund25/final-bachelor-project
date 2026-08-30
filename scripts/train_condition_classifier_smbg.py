"""Pengklasifikasi kondisi FINGER_STICK pada horizon produksi (~4 jam).

Motivasi
--------
Sapuan konfigurasi (``results/eval_prediksi/smbg_sweep.json``) menunjukkan regresi
nilai glukosa pada horizon 4 jam dari pembacaan tusuk jari mengungguli baseline
*persistence* secara meyakinkan, tetapi keamanan klinis absolutnya berhenti di
sekitar 52% zona Clarke A+B — belum layak dipakai.

Namun RAG terkondisi-prediksi sebenarnya tidak membutuhkan angka mg/dL. Yang
dibutuhkannya adalah **kondisi**: hipoglikemia, normal, atau hiperglikemia. Menduga
kelas secara langsung adalah tugas yang jauh lebih ringan daripada menduga nilai,
dan pada kanal CGM langkah yang sama sudah menaikkan sensitivitas hipoglikemia dari
17,3% ke 67,2% (``results/eval_prediksi/condition_classifier_9features.json``).

Skrip ini menguji apakah keuntungan yang sama berlaku pada skenario tusuk jari.

Tiga pendekatan dibandingkan pada himpunan uji yang sama
-------------------------------------------------------
1. ``persistence``   — kondisi masa depan = kondisi pembacaan terakhir.
2. ``regresi``       — nilai hasil GBM regresi, lalu dikelaskan.
3. ``pengklasifikasi`` — kelas diduga langsung, sadar-biaya lewat class_weight.

Perbandingan (2) lawan (3) adalah inti skrip ini: keduanya memakai fitur, split,
dan jendela yang identik, sehingga selisihnya benar-benar berasal dari apa yang
dijadikan sasaran belajar.

Split
-----
Memakai ``dataset_split`` resmi OhioT1DM, sama dengan jalur regresi FINGER_STICK,
supaya perbandingan (2) lawan (3) setara. Ini BERBEDA dari
``scripts/train_condition_classifier.py`` untuk CGM yang memakai split per pasien;
karena itu angka kedua berkas tidak boleh diperbandingkan langsung.

Keluaran: results/eval_prediksi/condition_classifier_smbg_4h.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import CONDITION_CLASSES, classify_glucose_3class  # noqa: E402
from src.data.preprocessor import DataPreprocessor  # noqa: E402

SUMBER = "FINGER_STICK"
SEED = 42
DEST = ROOT / "results/eval_prediksi/condition_classifier_smbg_4h.json"

kelaskan = np.vectorize(classify_glucose_3class)


def siapkan(cfg: dict):
    """Jalur penyiapan data yang sama dengan train_gbm_from_config.

    Penyaringan kanal dilakukan SEBELUM rekayasa fitur — urutan ini mengikat,
    sebab iob/cob/glucose_delta dihitung dari baris tetangga.
    """
    mc = cfg["model"]
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm.csv", parse_dates=["timestamp"])
    df = df[df["glucose_source"].astype(str).str.upper() == SUMBER].copy()
    if df.empty:
        raise SystemExit("Tidak ada baris FINGER_STICK.")
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(
        df, max_interpolate_steps=mc.get("max_interpolate_steps")
    )
    df = pre.engineer_features(df, **mc.get("feature_engineering", {}))
    pre.feature_columns = list(mc["engineered_features"])

    profil = mc["source_profiles"][SUMBER]
    seq = int(profil.get("sequence_length", 8))
    horizon = float(profil["prediction_horizons_min"][0])

    def jendela(split):
        return pre.create_time_horizon_sequences(
            df[df["dataset_split"] == split],
            sequence_length=seq,
            horizon_min=horizon,
            target_tolerance_min=float(profil.get("target_tolerance_min", 0.0)),
            max_history_gap_min=profil.get("max_history_gap_min"),
            return_anchor=True,
            min_target_horizon_min=profil.get("min_target_horizon_min"),
            max_target_horizon_min=profil.get("max_target_horizon_min"),
            min_history_interval_min=float(
                profil.get("min_history_interval_min", 0.0)
            ),
        )

    return jendela("train"), jendela("test"), pre, seq, horizon, list(mc["engineered_features"])


def per_kelas(y_benar, y_duga) -> dict:
    """Sensitivitas, PPV, dan cacah untuk tiap kelas kondisi."""
    cm = confusion_matrix(y_benar, y_duga, labels=CONDITION_CLASSES)
    keluar = {}
    for i, k in enumerate(CONDITION_CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        keluar[k] = {
            "n": int(cm[i].sum()),
            "sensitivitas_%": round(100 * tp / (tp + fn), 1) if tp + fn else None,
            "PPV_%": round(100 * tp / (tp + fp), 1) if tp + fp else None,
            "TP": tp, "FN": fn, "FP": fp,
        }
    keluar["akurasi_%"] = round(100 * float(np.trace(cm)) / cm.sum(), 1)
    keluar["matriks_kekeliruan"] = cm.tolist()
    return keluar


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    (Xtr, ytr, atr), (Xte, yte, ate), pre, seq, horizon, fitur = siapkan(cfg)

    print(f"sumber {SUMBER} | seq {seq} | horizon {horizon:g} menit "
          f"| {len(fitur)} fitur")
    print(f"n latih {len(ytr)} | n uji {len(yte)}")

    kelas_benar = kelaskan(yte)
    print("\nsebaran kelas sebenarnya pada data uji:")
    for k, n in zip(*np.unique(kelas_benar, return_counts=True)):
        print(f"  {k:12s} {n:4d}  ({100 * n / len(kelas_benar):.1f}%)")

    # Skala sendiri, di-fit hanya pada data latih.
    n_tr = len(Xtr)
    scaler = StandardScaler().fit(Xtr.reshape(n_tr, -1))
    Xtr_s = scaler.transform(Xtr.reshape(n_tr, -1))
    Xte_s = scaler.transform(Xte.reshape(len(Xte), -1))

    hasil = {
        "sumber": SUMBER,
        "horizon_menit": horizon,
        "sequence_length": seq,
        "n_features": len(fitur),
        "features": fitur,
        "split": "official_train_test",
        "catatan": (
            "Split resmi OhioT1DM dipakai agar sebanding dengan jalur regresi "
            "FINGER_STICK. Berbeda dari classifier CGM yang memakai split per "
            "pasien, sehingga angka keduanya tidak dapat diperbandingkan langsung."
        ),
        "n_train": int(len(ytr)),
        "n_uji": int(len(yte)),
        "sebaran_kelas_uji": {
            str(k): int(v) for k, v in zip(*np.unique(kelas_benar, return_counts=True))
        },
        "pendekatan": {},
    }

    # --- 0. kelas mayoritas: tebakan trivial ------------------------------
    # Pembanding terendah yang wajib ada (Aturan 35). Tanpa ini, akurasi 54,7%
    # terbaca sebagai capaian, padahal itu persis proporsi kelas terbanyak pada
    # data uji — artinya dapat dicapai tanpa melihat masukan sama sekali.
    mayoritas = pd.Series(kelaskan(ytr)).value_counts().idxmax()
    hasil["kelas_mayoritas_latih"] = str(mayoritas)
    hasil["pendekatan"]["kelas_mayoritas"] = per_kelas(
        kelas_benar, np.full(len(kelas_benar), mayoritas)
    )

    # --- 1. persistence: kondisi tidak berubah dari pembacaan terakhir -----
    hasil["pendekatan"]["persistence"] = per_kelas(kelas_benar, kelaskan(ate))

    # --- 2. regresi lalu dikelaskan ---------------------------------------
    # Target delta, direkonstruksi ke nilai absolut — persis jalur produksi.
    reg = HistGradientBoostingRegressor(
        random_state=cfg["model"].get("gradient_boosting", {}).get("random_state", SEED)
    )
    reg.fit(Xtr_s, ytr - atr)
    nilai_duga = reg.predict(Xte_s) + ate
    hasil["pendekatan"]["regresi"] = per_kelas(kelas_benar, kelaskan(nilai_duga))

    # --- 3. pengklasifikasi langsung, sadar-biaya -------------------------
    # class_weight="balanced": tanpa ini kelas hipoglikemia yang jarang akan
    # dikorbankan demi akurasi keseluruhan, dan justru kelas itulah yang paling
    # penting secara klinis.
    klf = HistGradientBoostingClassifier(
        random_state=SEED, class_weight="balanced"
    )
    klf.fit(Xtr_s, kelaskan(ytr))
    hasil["pendekatan"]["pengklasifikasi"] = per_kelas(kelas_benar, klf.predict(Xte_s))

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------- ringkas
    print(f"\n{'pendekatan':16s} {'akurasi':>8s}", end="")
    for k in CONDITION_CLASSES:
        print(f" {k[:9] + ' sens':>15s} {k[:9] + ' PPV':>15s}", end="")
    print()
    print("-" * 110)
    for nama, m in hasil["pendekatan"].items():
        print(f"{nama:16s} {m['akurasi_%']:7.1f}%", end="")
        for k in CONDITION_CLASSES:
            s, p = m[k]["sensitivitas_%"], m[k]["PPV_%"]
            print(f" {('-' if s is None else f'{s:.1f}%'):>15s}"
                  f" {('-' if p is None else f'{p:.1f}%'):>15s}", end="")
        print()

    print(f"\nDisimpan ke {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

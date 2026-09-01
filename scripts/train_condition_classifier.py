"""Train CGM condition classifier with the current 9-feature schema.

This version is aligned with the current modality-aware preprocessing:
- source: data/raw/ohio_t1dm.csv
- modality: CGM
- features: config.model.engineered_features (9 features)
- temporal builder: create_time_horizon_sequences()
- classifier gets its OWN StandardScaler fitted on training data

Dua lengan dilaporkan pada himpunan uji yang sama: ``regresi_lalu_ambang``
(regresi glukosa lalu diambang menjadi kondisi) dan ``pengklasifikasi_kondisi``
(pengklasifikasi tiga kelas langsung). Keduanya menjadi sumber Gambar VI.7.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import CONDITION_CLASSES, classify_glucose_3class
from src.data.preprocessor import DataPreprocessor


SOURCE = "CGM"
CLASSIFIER_HORIZON_MIN = 30.0
SEED = 42
CLASSES = CONDITION_CLASSES
classify_glucose = classify_glucose_3class


def build(cfg: dict):
    mc = cfg["model"]

    data_path = ROOT / "data/raw/ohio_t1dm.csv"
    if not data_path.exists():
        raise SystemExit(f"Dataset tidak ditemukan: {data_path}")

    df = pd.read_csv(
        data_path,
        parse_dates=["timestamp"],
    )

    required_source_cols = {
        "patient_id",
        "timestamp",
        "glucose",
        "glucose_source",
    }
    missing = required_source_cols - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset unified belum memiliki kolom: "
            + ", ".join(sorted(missing))
        )

    # Condition classifier ini adalah classifier CGM.
    df = df[
        df["glucose_source"]
        .astype(str)
        .str.upper()
        .eq(SOURCE)
    ].copy()

    if df.empty:
        raise ValueError("Tidak ada data CGM pada dataset unified.")

    pre = DataPreprocessor(cfg)

    df = pre.handle_missing_values(
        df,
        max_interpolate_steps=mc.get(
            "max_interpolate_steps"
        ),
    )

    df = pre.engineer_features(
        df,
        **mc["feature_engineering"],
    )

    feature_columns = list(
        mc["engineered_features"]
    )
    pre.feature_columns = feature_columns

    missing_features = [
        f for f in feature_columns
        if f not in df.columns
    ]
    if missing_features:
        raise ValueError(
            "Engineered features tidak tersedia: "
            + ", ".join(missing_features)
        )

    # Keep patient split identical to the previous classifier script.
    patients = sorted(
        df["patient_id"].unique()
    )

    if len(patients) < 3:
        raise ValueError(
            "Minimal 3 pasien diperlukan untuk split train/test."
        )

    test_patients = patients[-2:]

    train_df, test_df = pre.split_by_patient(
        df,
        test_patients,
    )

    try:
        profile = mc["source_profiles"][SOURCE]
    except KeyError as exc:
        raise ValueError(
            f"Source profile tidak ditemukan untuk modality {SOURCE}."
        ) from exc

    configured_horizons = profile.get(
        "prediction_horizons_min",
        [],
    )
    if CLASSIFIER_HORIZON_MIN not in configured_horizons:
        raise ValueError(
            f"Horizon classifier {CLASSIFIER_HORIZON_MIN:g} menit "
            f"tidak tersedia pada profile {SOURCE}: {configured_horizons}"
        )

    sequence_length = int(
        profile.get(
            "sequence_length",
            mc.get("sequence_length", 12),
        )
    )

    target_tolerance_min = float(
        profile.get(
            "target_tolerance_min",
            2.5,
        )
    )

    max_history_gap_min = profile.get(
        "max_history_gap_min",
        30,
    )

    min_history_interval_min = float(
        profile.get(
            "min_history_interval_min",
            0,
        )
    )

    def seqs(d):
        X, y, anc = (
            pre.create_time_horizon_sequences(
                d,
                sequence_length=sequence_length,
                horizon_min=CLASSIFIER_HORIZON_MIN,
                target_tolerance_min=target_tolerance_min,
                max_history_gap_min=max_history_gap_min,
                return_anchor=True,
                min_history_interval_min=min_history_interval_min,
            )
        )
        return X, y, anc

    Xtr, ytr, atr = seqs(train_df)
    Xte, yte, ate = seqs(test_df)

    return (
        (Xtr, ytr, atr),
        (Xte, yte, ate),
        mc,
        feature_columns,
        sequence_length,
        CLASSIFIER_HORIZON_MIN,
        target_tolerance_min,
    )


def metrics_for(
    y_true_lbl: np.ndarray,
    y_pred_lbl: np.ndarray,
) -> dict:
    cm = confusion_matrix(
        y_true_lbl,
        y_pred_lbl,
        labels=CLASSES,
    )

    out = {
        "akurasi_keseluruhan_%": round(
            100
            * float(
                (y_true_lbl == y_pred_lbl).mean()
            ),
            1,
        )
    }

    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        sens = (
            100 * tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        ppv = (
            100 * tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        out[c] = {
            "n": int(cm[i, :].sum()),
            "sensitivitas_%": round(
                float(sens),
                1,
            ),
            "PPV_%": round(
                float(ppv),
                1,
            ),
        }

    return out


def main() -> None:
    cfg = yaml.safe_load(
        (
            ROOT / "config.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    ap = argparse.ArgumentParser(
        description=(
            "Latih condition classifier "
            "CGM dengan 9 engineered features."
        )
    )

    ap.add_argument(
        "--model",
        choices=["gbm", "rf"],
        default=(
            "rf"
            if cfg["model"].get(
                "name"
            )
            == "RandomForest"
            else "gbm"
        ),
    )

    args = ap.parse_args()
    prefix = args.model

    (
        (Xtr, ytr, atr),
        (Xte, yte, ate),
        mc,
        feature_columns,
        sequence_length,
        horizon_min,
        target_tolerance_min,
    ) = build(cfg)

    n_features = len(feature_columns)

    print(
        f"SOURCE              : {SOURCE}"
    )
    print(
        f"FEATURES            : {n_features}"
    )
    print(
        f"FEATURE NAMES       : {feature_columns}"
    )
    print(
        f"SEQUENCE LENGTH     : {Xtr.shape[1]}"
    )
    print(
        f"HORIZON             : +{horizon_min:g} menit"
    )
    print(
        f"TRAIN SAMPLES       : {len(ytr)}"
    )
    print(
        f"TEST SAMPLES        : {len(yte)}"
    )

    if n_features != 9:
        raise RuntimeError(
            "Config saat ini tidak menghasilkan 9 engineered features. "
            f"Ditemukan {n_features}: {feature_columns}"
        )

    if Xtr.shape[1] != sequence_length:
        raise RuntimeError(
            "Sequence length hasil builder tidak sesuai profile: "
            f"{Xtr.shape[1]} vs {sequence_length}."
        )

    lbl_tr = np.array([
        classify_glucose(v)
        for v in ytr
    ])

    lbl_te = np.array([
        classify_glucose(v)
        for v in yte
    ])

    cur_te = np.array([
        classify_glucose(v)
        for v in ate
    ])

    divergent = cur_te != lbl_te

    print(
        f"Kasus divergen di data uji: "
        f"{int(divergent.sum())}"
    )

    # Regression baseline dihitung setelah classifier, karena kedua lengan
    # memakai representasi terskala yang sama. Lihat blok "Regression baseline"
    # di bawah.

    # ---------------------------------------------------------
    # NEW condition classifier
    #
    # IMPORTANT:
    # Do NOT reuse the regression scaler object.
    # The classifier gets its own scaler fitted on the same
    # 9-feature representation, preventing the old 7-feature
    # classifier artifact from leaking into this pipeline.
    # ---------------------------------------------------------

    classifier_scaler = StandardScaler()

    Xtr_flat = Xtr.reshape(
        -1,
        n_features,
    )

    Xte_flat = Xte.reshape(
        -1,
        n_features,
    )

    Xtr_s = classifier_scaler.fit_transform(
        Xtr_flat
    ).reshape(
        len(Xtr),
        Xtr.shape[1],
        n_features,
    )

    Xte_s = classifier_scaler.transform(
        Xte_flat
    ).reshape(
        len(Xte),
        Xte.shape[1],
        n_features,
    )

    if classifier_scaler.n_features_in_ != n_features:
        raise RuntimeError(
            "Classifier scaler tidak sesuai schema fitur: "
            f"{classifier_scaler.n_features_in_} vs {n_features}."
        )

    Xtr_model = Xtr_s.reshape(
        len(Xtr_s),
        -1,
    )

    Xte_model = Xte_s.reshape(
        len(Xte_s),
        -1,
    )

    if prefix == "gbm":
        clf = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=(
                mc.get(
                    "gradient_boosting",
                    {},
                ).get(
                    "random_state",
                    SEED,
                )
            ),
        )
    else:
        rf_cfg = mc["random_forest"]

        clf = RandomForestClassifier(
            n_estimators=rf_cfg[
                "n_estimators"
            ],
            max_depth=rf_cfg[
                "max_depth"
            ],
            min_samples_split=rf_cfg[
                "min_samples_split"
            ],
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )

    clf.fit(
        Xtr_model,
        lbl_tr,
    )

    lbl_clf = clf.predict(
        Xte_model
    )

    # ---------------------------------------------------------
    # Regression baseline from the SAME model family.
    #
    # Lengan ini memprediksi kadar glukosa lalu mengambangnya menjadi kondisi,
    # sedangkan lengan di atas memprediksi kondisi secara langsung. Keduanya
    # WAJIB memakai himpunan uji, panjang jendela, dan representasi terskala
    # yang sama; kalau tidak, selisih sensitivitas yang dilaporkan bercampur
    # dengan selisih data. Karena itu regresor di bawah memakai Xtr_model dan
    # Xte_model yang sama persis dengan pengklasifikasi, bukan bundle produksi
    # yang punya scaler dan split sendiri.
    #
    # Sebelum Agustus 2026 lengan ini pernah hilang saat pipeline berpindah ke
    # 9 fitur, sehingga condition_classifier_9features.json hanya memuat lengan
    # pengklasifikasi dan Gambar VI.7 terpaksa memakai angka 7 fitur yang lama.
    # ---------------------------------------------------------

    if prefix == "gbm":
        reg = HistGradientBoostingRegressor(
            random_state=(
                mc.get(
                    "gradient_boosting",
                    {},
                ).get(
                    "random_state",
                    SEED,
                )
            ),
        )
    else:
        rf_cfg = mc["random_forest"]

        reg = RandomForestRegressor(
            n_estimators=rf_cfg[
                "n_estimators"
            ],
            max_depth=rf_cfg[
                "max_depth"
            ],
            min_samples_split=rf_cfg[
                "min_samples_split"
            ],
            random_state=SEED,
            n_jobs=-1,
        )

    reg.fit(
        Xtr_model,
        ytr,
    )

    yhat_reg = reg.predict(
        Xte_model
    )

    lbl_reg = np.array([
        classify_glucose(v)
        for v in yhat_reg
    ])

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    res = {
        "source": SOURCE,
        "horizon_menit": int(
            horizon_min
        ),
        "sequence_length": int(
            sequence_length
        ),
        "n_features": n_features,
        "features": feature_columns,
        "keluarga_model": type(
            clf
        ).__name__,
        "regresor_pembanding": type(
            reg
        ).__name__,
        "class_weight": "balanced",
        "classifier_scaler": (
            "StandardScaler fitted on training data"
        ),
        "catatan": (
            "Kondisi masa depan diprediksi langsung "
            "oleh pengklasifikasi tiga kelas dengan "
            "9 engineered features yang sama dengan "
            "pipeline GBM regression, dibandingkan "
            "terhadap kondisi yang diturunkan dari "
            "regresi pada keluarga model, himpunan uji, "
            "dan representasi terskala yang SAMA."
        ),
        "n_train": int(
            len(ytr)
        ),
        "n_uji": int(
            len(yte)
        ),
        "n_divergen": int(
            divergent.sum()
        ),
        "regresi_lalu_ambang": metrics_for(
            lbl_te,
            lbl_reg,
        ),
        "pengklasifikasi_kondisi": metrics_for(
            lbl_te,
            lbl_clf,
        ),
        "pada_kasus_divergen": {
            "akurasi_kondisi_regresi_%": (
                round(
                    100
                    * float(
                        (
                            lbl_reg[divergent]
                            == lbl_te[divergent]
                        ).mean()
                    ),
                    1,
                )
                if divergent.any()
                else 0.0
            ),
            "akurasi_kondisi_pengklasifikasi_%": (
                round(
                    100
                    * float(
                        (
                            lbl_clf[divergent]
                            == lbl_te[divergent]
                        ).mean()
                    ),
                    1,
                )
                if divergent.any()
                else 0.0
            ),
        },
    }

    print(
        "\n--- Kondisi dari REGRESI lalu AMBANG ---"
    )

    for c in CLASSES:
        d = res[
            "regresi_lalu_ambang"
        ][c]

        print(
            f"  {c:15s} "
            f"n={d['n']:6d} "
            f"sensitivitas={d['sensitivitas_%']:5.1f}% "
            f"PPV={d['PPV_%']:5.1f}%"
        )

    print(
        "  akurasi keseluruhan:",
        res[
            "regresi_lalu_ambang"
        ][
            "akurasi_keseluruhan_%"
        ],
        "%",
    )

    print(
        "\n--- Kondisi dari CLASSIFIER 9-FEATURE ---"
    )

    for c in CLASSES:
        d = res[
            "pengklasifikasi_kondisi"
        ][c]

        print(
            f"  {c:15s} "
            f"n={d['n']:6d} "
            f"sensitivitas={d['sensitivitas_%']:5.1f}% "
            f"PPV={d['PPV_%']:5.1f}%"
        )

    print(
        "  akurasi keseluruhan:",
        res[
            "pengklasifikasi_kondisi"
        ][
            "akurasi_keseluruhan_%"
        ],
        "%",
    )

    print("\n--- Pada kasus divergen ---")
    print(
        "  regresi         :",
        res["pada_kasus_divergen"][
            "akurasi_kondisi_regresi_%"
        ],
        "%",
    )
    print(
        "  classifier 9F   :",
        res["pada_kasus_divergen"][
            "akurasi_kondisi_pengklasifikasi_%"
        ],
        "%",
    )

    # ---------------------------------------------------------
    # Persist bundle
    # ---------------------------------------------------------

    clf_path = (
        ROOT
        / f"models/{prefix}_condition_classifier_h6.pkl"
    )

    classifier_bundle = {
        "model": clf,
        "scaler": classifier_scaler,
        "classes": CLASSES,
        "features": feature_columns,
        "n_features": n_features,
        "sequence_length": sequence_length,
        "prediction_horizon": int(
            horizon_min / 5
        ),
        "prediction_horizon_min": int(
            horizon_min
        ),
        "glucose_source": SOURCE,
        "model_family": type(
            clf
        ).__name__,
        "feature_engineering": mc[
            "feature_engineering"
        ],
        "target_tolerance_min": (
            target_tolerance_min
        ),
    }

    with open(
        clf_path,
        "wb",
    ) as f:
        pickle.dump(
            classifier_bundle,
            f,
        )

    print(
        f"\nPengklasifikasi -> "
        f"{clf_path.name} "
        f"({clf_path.stat().st_size / 1e6:.2f} MB)"
    )

    dest = (
        ROOT
        / "results/eval_prediksi/"
        "condition_classifier_9features.json"
    )

    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dest.write_text(
        json.dumps(
            res,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Disimpan ke {dest}"
    )


if __name__ == "__main__":
    main()
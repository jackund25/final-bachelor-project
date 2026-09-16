"""Waktu tanggap UJUNG-KE-UJUNG per tahap: masukan pasien -> rekomendasi siap tampil.

Menjawab KNF-10. Tolok ukur per komponen yang terpisah (dahulu `benchmark_deployability.py`,
kini diarsipkan) BUKAN hal yang sama -- ia tidak pernah menjumlahkan satu permintaan utuh
sebagaimana dirasakan dokter.

Tahap yang diukur, berurutan sesuai alur aplikasi:

  rekayasa_fitur     -> build_window + engineer_features
  prediksi           -> RandomForest.predict (per horizon)
  kalibrasi_interval -> std antar-pohon + faktor konformal (per horizon)
  penyusunan_kueri   -> PredictionConditionedQueryBuilder
  retrieval          -> embed kueri + MMR atas ChromaDB
  generasi_llm       -> panggilan Gemini (atau template bila offline)

Dilaporkan median dan p95, ditambah proporsi waktu LLM terhadap komputasi lokal.
Proporsi itu yang menentukan apakah sistem masih terpakai saat LLM lambat atau kuota
habis -- pertanyaan penerapan yang nyata, bukan sekadar angka total.

Contoh:
    python scripts/benchmark_latency.py --n 20
    python scripts/benchmark_latency.py --n 20 --provider template   # tanpa kuota LLM

Keluaran: results/benchmark/latency_endtoend.json
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy/pandas — WinError 1114)
import os

# WAJIB sebelum membangun pipeline. Tanpa ini GOOGLE_API_KEY tidak terbaca, dan sistem
# TIDAK melempar error — ia diam-diam menjawab dengan template sambil tetap melaporkan
# llm_provider "gemini". Benchmark lalu mencatat "waktu LLM" 0,0001 detik yang sebenarnya
# waktu template. Lihat penjagaan lolos_uji_llm() di bawah.
from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.conformal import prediction_interval  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.timing import (  # noqa: E402
    STAGE_CALIBRATE, STAGE_FEATURES, STAGE_GENERATE, STAGE_ORDER, STAGE_PREDICT,
    StageTimer, percentile,
)

OUT_DIR = ROOT / "results/benchmark"
HORIZON_BUNDLES = ["models/rf_inference_bundle_h6.pkl", "models/rf_inference_bundle_h12.pkl"]


def load_bundle(path: Path):
    if not path.exists():
        return None
    b = pickle.load(open(path, "rb"))
    return {"model": b["model"], "scaler": b.get("scaler"),
            "features": b.get("features", ["glucose", "carbs", "insulin", "activity"]),
            "sequence_length": int(b.get("sequence_length", 12)),
            "horizon": int(b.get("prediction_horizon", 6)),
            "use_engineered": bool(b.get("use_engineered", False)),
            "predict_delta": bool(b.get("predict_delta", False)),
            "feature_engineering": dict(b.get("feature_engineering", {}))}


def build_window(patient_df, art):
    if art["use_engineered"]:
        from src.data.preprocessor import DataPreprocessor
        feat_df = DataPreprocessor({}).engineer_features(patient_df, **art["feature_engineering"])
    else:
        feat_df = patient_df
    return feat_df.tail(art["sequence_length"]).reset_index(drop=True)


def predict_one(window_df, art):
    X = window_df[art["features"]].values.astype(float)
    if art["scaler"] is not None:
        X = art["scaler"].transform(X)
    out = float(art["model"].predict(X.reshape(1, -1))[0])
    if art["predict_delta"]:
        out += float(window_df["glucose"].iloc[-1])
    return out


def predict_std(window_df, art):
    est = getattr(art["model"], "estimators_", None)
    if not est:
        return None
    import numpy as np
    X = window_df[art["features"]].values.astype(float)
    if art["scaler"] is not None:
        X = art["scaler"].transform(X)
    Xf = X.reshape(1, -1)
    return float(np.array([t.predict(Xf)[0] for t in est]).std())


def one_request(pat_df, arts, pipeline) -> StageTimer:
    """Satu permintaan utuh, persis urutan yang dijalankan aplikasi."""
    timer = StageTimer()

    windows = []
    with timer.measure(STAGE_FEATURES):
        for a in arts:
            windows.append(build_window(pat_df, a))

    preds = []
    with timer.measure(STAGE_PREDICT):
        for a, w in zip(arts, windows):
            preds.append(predict_one(w, a))

    with timer.measure(STAGE_CALIBRATE):
        for a, w, p in zip(arts, windows, preds):
            prediction_interval(p, predict_std(w, a), a["horizon"], level=95)

    current = float(windows[0]["glucose"].iloc[-1])
    patient_state = {
        "current_glucose": current,
        "stress_level": int(windows[0].get("stress", pd.Series([5])).iloc[-1])
        if "stress" in windows[0].columns else 5,
        "activity_level": float(windows[0]["activity"].iloc[-1]),
        "insulin_on_board": float(windows[0]["insulin"].iloc[-1]),
        "carbs_on_board": float(windows[0]["carbs"].iloc[-1]),
    }
    # Tahap penyusunan_kueri, retrieval, dan generasi_llm diukur DI DALAM pipeline,
    # memakai timer yang sama, sehingga totalnya benar-benar ujung-ke-ujung.
    pipeline.answer(patient_state=patient_state, prediction=preds[0], timer=timer)
    return timer


def ringkas(samples: list[dict], key: str) -> dict:
    vals = [s[key] for s in samples if key in s]
    if not vals:
        return {}
    return {
        "median_dtk": round(statistics.median(vals), 4),
        "p95_dtk": round(percentile(vals, 95), 4),
        "min_dtk": round(min(vals), 4),
        "max_dtk": round(max(vals), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="jumlah permintaan yang diukur")
    ap.add_argument("--warmup", type=int, default=2,
                    help="permintaan pemanasan yang DIBUANG (muat model, cache embedding)")
    ap.add_argument("--provider", default=None,
                    help="paksa provider LLM, mis. 'template' untuk mengukur tanpa kuota")
    ap.add_argument("--delay", type=float, default=0.0,
                    help=("jeda antar-permintaan (detik). Tanpa jeda, n>10 menembus batas "
                          "free-tier gemini-2.5-flash-lite (terukur 10 permintaan/menit) "
                          "sehingga p95 mengukur waktu tunggu retry, bukan waktu model. "
                          "Pakai --delay 7 untuk tetap di bawah batas."))
    args = ap.parse_args()

    arts = [a for a in (load_bundle(ROOT / p) for p in HORIZON_BUNDLES) if a is not None]
    if not arts:
        raise SystemExit("Bundle model tidak ditemukan. Latih model lebih dulu.")
    arts.sort(key=lambda a: a["horizon"])

    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])
    pids = sorted(df["patient_id"].unique().tolist())

    kwargs = {"llm_provider": args.provider} if args.provider else {}
    pipeline = RAGPipeline(**kwargs)
    pipeline.build()

    # Penjagaan: bila provider bukan "template" tetapi rantai LLM tidak siap, sistem akan
    # menjawab dengan template TANPA error apa pun. Angka "waktu LLM" yang dihasilkan
    # benchmark lalu menjadi waktu template — lebih berbahaya daripada gagal, karena
    # terlihat seperti hasil yang sangat cepat.
    if args.provider != "template":
        chain = getattr(pipeline.generator, "chain", None)
        siap = bool(chain is not None and getattr(chain, "is_ready", False))
        if not siap:
            raise SystemExit(
                f"Provider '{pipeline.llm_provider}' diminta, tetapi rantai LLM TIDAK siap "
                f"(kunci API tidak terbaca atau paket tidak terpasang). Sistem akan menjawab "
                f"dengan template dan angka latensinya tidak sah. Perbaiki dulu, atau "
                f"jalankan dengan --provider template bila memang ingin mengukur tanpa LLM."
            )

    # Pemanasan dibuang: permintaan pertama menanggung pemuatan model embedding dan
    # koneksi ChromaDB, yang di aplikasi nyata hanya terjadi sekali per sesi.
    print(f"Pemanasan {args.warmup} permintaan ...")
    for i in range(args.warmup):
        pat = df[df["patient_id"] == pids[i % len(pids)]].sort_values("timestamp")
        one_request(pat, arts, pipeline)

    print(f"Mengukur {args.n} permintaan (provider={pipeline.llm_provider}) ...")
    samples, t0 = [], time.time()
    for i in range(args.n):
        if args.delay and i:
            time.sleep(args.delay)  # jeda TIDAK ikut terukur; timer hanya jalan di dalam permintaan
        pat = df[df["patient_id"] == pids[i % len(pids)]].sort_values("timestamp")
        # Geser titik akhir jendela agar tiap permintaan memakai data berbeda.
        pat = pat.iloc[: len(pat) - (i * 7 % 500)] if len(pat) > 600 else pat
        samples.append(one_request(pat, arts, pipeline).as_dict())
        print(f"  {i + 1}/{args.n}  total {samples[-1]['_total']:.3f} dtk", flush=True)
    durasi_total = time.time() - t0

    per_tahap = {s: ringkas(samples, s) for s in STAGE_ORDER if any(s in x for x in samples)}
    total = ringkas(samples, "_total")
    lokal = ringkas(samples, "_lokal")
    jaringan = ringkas(samples, "_jaringan")

    share = [s["_jaringan"] / s["_total"] for s in samples if s["_total"] > 0]
    out = {
        "catatan": ("Waktu tanggap ujung-ke-ujung: rekayasa fitur -> prediksi -> kalibrasi "
                    "-> penyusunan kueri -> retrieval -> generasi LLM. Pemanasan dibuang."),
        "n": args.n, "warmup_dibuang": args.warmup,
        "jeda_antar_permintaan_dtk": args.delay,
        "llm_provider": pipeline.llm_provider,
        "horizon_steps": [a["horizon"] for a in arts],
        "per_tahap": per_tahap,
        "total": total, "komputasi_lokal": lokal, "menunggu_llm": jaringan,
        "proporsi_llm": {
            "median": round(statistics.median(share), 4) if share else None,
            "p95": round(percentile(share, 95), 4) if share else None,
        },
        "durasi_benchmark_dtk": round(durasi_total, 1),
        "sampel": samples,
    }
    # Keluaran per provider agar hasil dengan dan tanpa LLM tidak saling menimpa.
    out_path = OUT_DIR / f"latency_endtoend_{pipeline.llm_provider}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== WAKTU TANGGAP UJUNG-KE-UJUNG (n={args.n}, provider={pipeline.llm_provider}) ===")
    print(f"{'tahap':<20} {'median':>10} {'p95':>10}")
    for s, v in per_tahap.items():
        print(f"{s:<20} {v['median_dtk']:>9.3f}s {v['p95_dtk']:>9.3f}s")
    print(f"{'-' * 42}")
    for nama, v in (("TOTAL", total), ("  komputasi lokal", lokal), ("  menunggu LLM", jaringan)):
        print(f"{nama:<20} {v['median_dtk']:>9.3f}s {v['p95_dtk']:>9.3f}s")
    if share:
        print(f"\nProporsi waktu menunggu LLM: median {100 * statistics.median(share):.1f}%, "
              f"p95 {100 * percentile(share, 95):.1f}%")

    # Deteksi permintaan yang kemungkinan besar kena batas kuota. Tanpa ini, satu
    # permintaan yang menunggu retry 30+ detik akan menaikkan p95 dan terbaca seolah-olah
    # model memang selambat itu.
    gen = [s.get(STAGE_GENERATE, 0.0) for s in samples]
    if gen and max(gen) > 0:
        med = statistics.median(gen)
        pencilan = [(i + 1, v) for i, v in enumerate(gen) if med > 0 and v > 5 * med]
        if pencilan:
            print(f"\nPERINGATAN: {len(pencilan)} dari {len(gen)} permintaan memakan waktu "
                  f">5x median generasi ({med:.2f} dtk). Hampir pasti menunggu retry batas "
                  f"kuota, bukan waktu model:")
            for i, v in pencilan:
                print(f"  permintaan #{i}: {v:.1f} dtk")
            print("  p95 di atas TIDAK mewakili kecepatan model. Pakai median, atau jalankan "
                  "ulang saat kuota tersedia dengan --delay.")
    print(f"\nDisimpan ke {out_path}")


if __name__ == "__main__":
    main()

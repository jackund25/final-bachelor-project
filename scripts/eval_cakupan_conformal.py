"""T12b — penaksir cakupan conformal yang JUJUR, dengan tiga himpunan terpisah.

MENGAPA SKRIP INI ADA.
scripts/conformal_calibration.py kini melatih pada 10 pasien yang sama dengan
bundel produksi, sehingga `q` yang dipakai runtime SAH. Konsekuensinya himpunan
kalibrasi dan himpunan pelaporan menjadi sama, sehingga angka "coverage%" pada
keluarannya OPTIMISTIS — terlihat dari nilainya yang tepat 90,0% dan 95,0%, tanda
khas kalibrasi in-sample. Angka itu tidak boleh dikutip sebagai cakupan tercapai.

Skrip ini menghasilkan angka yang BOLEH dikutip, dengan memisahkan tiga peran:

    latih 10 pasien  ->  kalibrasi 1 pasien  ->  ukur cakupan pada 1 pasien LAIN

Pasien pengukur tidak pernah dipakai melatih MAUPUN mengkalibrasi, sehingga
cakupannya jujur. Jumlah pasien latih tetap 10, sama dengan produksi, sehingga
yang ditaksir memang prosedur yang benar-benar dipakai — bukan model lain.

RANCANGAN PUTARAN. Dua belas pasien dipasangkan menjadi enam pasangan lepas, dan
setiap pasangan dijalankan DUA ARAH (A mengkalibrasi lalu B mengukur, kemudian
sebaliknya), menghasilkan 12 taksiran per horizon. Kedua arah dijalankan karena
kalibrasi dengan satu pasien berderau; menjalankan satu arah saja membuat hasilnya
bergantung pada pasien mana yang kebetulan terpilih.

YANG TIDAK DIKLAIM. Ini penaksir cakupan PROSEDUR, bukan cakupan bundel produksi
yang spesifik itu. Untuk satu bundel tertentu, jaminan split conformal bersifat
teoretis dan berlaku atas titik baru yang dapat dipertukarkan
(Angelopoulos & Bates 2021, Bagian 2).

Keluaran: results/eval_prediksi/cakupan_conformal.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import DiabetesDataLoader  # noqa: E402
from src.data.preprocessor import DataPreprocessor  # noqa: E402


def conformal_q(scores, alpha):
    """Kuantil conformal dengan koreksi sampel-hingga.

    SALINAN dari scripts/conformal_calibration.py:64. Sengaja disalin, bukan
    diimpor: modul itu mengawali dengan `import torch` (penata urutan DLL), dan
    pada mesin pengembangan ini torch kadang gagal memuat c10.dll sehingga
    mengimpornya membuat skrip ini ikut gagal padahal tidak memerlukan torch
    sama sekali.

    Keduanya dijaga tetap sama oleh tests/test_cakupan_conformal.py.
    """
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))

OUT = ROOT / "results/eval_prediksi/cakupan_conformal.json"
EPS = 1e-6
KUANTIL = (0.025, 0.975)
GAUSS_95_WIDTH = 3.92  # (q0.975 - q0.025) untuk normal baku; lihat gbm_model.py


def _model(seed: int = 42, **kw):
    return HistGradientBoostingRegressor(random_state=seed, **kw)


def jalankan_satu(df, feats, cfg_penuh, horizon, cal_p, test_p, train_p, seq_len,
                  max_gap_steps, cadence_min, predict_delta, fe, use_eng):
    """Satu putaran: latih pada train_p, kalibrasi pada cal_p, ukur pada test_p."""
    prep = DataPreprocessor(cfg_penuh)
    prep.feature_columns = list(feats)

    def seqs(sub):
        return prep.create_sequences(
            df[df["patient_id"].isin(sub)], seq_len, horizon, return_anchor=True,
            max_gap_steps=max_gap_steps, source_interval_min=cadence_min,
        )

    Xtr, ytr, atr = seqs(train_p)
    Xca, yca, aca = seqs(cal_p)
    Xte, yte, ate = seqs(test_p)
    if min(len(ytr), len(yca), len(yte)) == 0:
        return None

    p2 = DataPreprocessor(cfg_penuh)
    Xtr_s, _ = p2.normalize_data(Xtr, None)
    f = lambda X: p2.scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape).reshape(len(X), -1)
    Xtr_f, Xca_f, Xte_f = Xtr_s.reshape(len(ytr), -1), f(Xca), f(Xte)

    ytr_fit = (ytr - atr) if predict_delta else ytr
    titik = _model().fit(Xtr_f, ytr_fit)
    lo_m = _model(loss="quantile", quantile=KUANTIL[0]).fit(Xtr_f, ytr_fit)
    hi_m = _model(loss="quantile", quantile=KUANTIL[1]).fit(Xtr_f, ytr_fit)

    def duga(Xf, anc):
        out = titik.predict(Xf)
        return (out + anc) if predict_delta else out

    def sigma(Xf):
        return np.maximum(hi_m.predict(Xf) - lo_m.predict(Xf), EPS) / GAUSS_95_WIDTH

    yca_p, yte_p = duga(Xca_f, aca), duga(Xte_f, ate)
    std_ca, std_te = sigma(Xca_f), sigma(Xte_f)
    res_ca = np.abs(yca - yca_p)

    hasil = {"n_cal": int(len(yca)), "n_test": int(len(yte))}
    for alpha, tgt in [(0.10, 90), (0.05, 95)]:
        q = conformal_q(res_ca / (std_ca + EPS), alpha)
        lo, hi = yte_p - q * std_te, yte_p + q * std_te
        hasil[str(tgt)] = {
            "q": round(float(q), 3),
            "cakupan_%": round(float(np.mean((yte >= lo) & (yte <= hi)) * 100), 1),
            "lebar_rata2": round(float(np.mean(hi - lo)), 1),
        }
    return hasil


def analisis(putaran) -> dict:
    """Mengapa cakupannya berayun — diuji, bukan ditafsirkan begitu saja.

    Dugaannya: `q` sepenuhnya ditentukan oleh sebaran residual PASIEN PENGKALIBRASI.
    Pasien yang glukosanya bergejolak menghasilkan q besar sehingga terlalu banyak
    menutup pasien lain; pasien yang stabil menghasilkan q kecil sehingga kurang
    menutup. Bila benar, q dan cakupan akan berkorelasi positif kuat — dan itu
    berarti asumsi EXCHANGEABILITY conformal tertekan oleh keragaman antar-pasien,
    bukan sekadar derau sampel kecil.
    """
    from scipy.stats import pearsonr

    q = [x["95"]["q"] for x in putaran]
    c = [x["95"]["cakupan_%"] for x in putaran]
    r, pv = pearsonr(q, c)
    return {
        "korelasi_q_dengan_cakupan_r": round(float(r), 3),
        "korelasi_p": round(float(pv), 5),
        "n_putaran_di_bawah_nominal_95": int(sum(1 for x in c if x < 95)),
        "n_putaran": len(c),
        "q_min": round(float(min(q)), 3),
        "q_maks": round(float(max(q)), 3),
        "tafsir": (
            "Korelasi positif kuat antara q dan cakupan menunjukkan cakupan yang "
            "terwujud ditentukan oleh SIAPA yang mengkalibrasi, bukan oleh mutu "
            "modelnya. Itu tanda asumsi exchangeability conformal tertekan oleh "
            "keragaman antar-pasien: residual satu pasien tidak dapat dipertukarkan "
            "dengan residual pasien lain."),
    }


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    m = cfg["model"]
    d = cfg["data"]
    seq_len = int(m.get("sequence_length", 12))
    max_gap_steps = int(m.get("max_gap_steps", 6))
    cadence_min = float(d.get("sampling_interval_min", 5))
    predict_delta = bool(m.get("predict_delta", True))
    fe = m.get("feature_engineering", {}) or {}
    use_eng = bool(m.get("use_engineered_features", True))

    # Konstruksi disamakan PERSIS dengan scripts/conformal_calibration.py:96-98:
    # loader menerima direktori keluaran, dan DataPreprocessor menerima config
    # PENUH (bukan blok model). Menyimpang di sini akan menghasilkan fitur atau
    # jendela yang berbeda, sehingga cakupan yang ditaksir bukan milik prosedur
    # yang sebenarnya dipakai.
    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = (loader.load_csv("ohio_t1dm_merged.csv")
          .sort_values(["patient_id", "timestamp"]).reset_index(drop=True))
    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(df)
    if use_eng:
        df = prep.engineer_features(df, **fe)
    feats = list(prep.feature_columns)

    pids = sorted(df["patient_id"].unique().tolist())
    pasangan = [(pids[i], pids[i + 1]) for i in range(0, len(pids) - 1, 2)]

    hasil = {
        "percobaan": "T12b — penaksir cakupan conformal dengan tiga himpunan terpisah",
        "mengapa": ("conformal_h*.json mengkalibrasi dan melaporkan pada himpunan yang "
                    "SAMA, sehingga cakupannya optimistis. Di sini pasien pengukur tidak "
                    "pernah dipakai melatih maupun mengkalibrasi."),
        "rancangan": ("latih 10 pasien (sama dengan produksi) -> kalibrasi 1 pasien -> "
                      "ukur pada 1 pasien lain; enam pasangan lepas, dua arah"),
        "yang_tidak_diklaim": ("Ini cakupan PROSEDUR, bukan cakupan satu bundel tertentu. "
                               "Bagi satu bundel, jaminannya teoretis atas titik baru yang "
                               "dapat dipertukarkan (Angelopoulos & Bates 2021 Bagian 2)."),
        "n_pasien": len(pids),
        "horizon": {},
    }

    for horizon in [6, 12]:
        t0 = time.time()
        putaran = []
        for a, b in pasangan:
            for cal_p, test_p in [([a], [b]), ([b], [a])]:
                train_p = [p for p in pids if p not in (a, b)]
                r = jalankan_satu(df, feats, cfg, horizon, cal_p, test_p, train_p,
                                  seq_len, max_gap_steps, cadence_min,
                                  predict_delta, fe, use_eng)
                if r is None:
                    continue
                r.update({"kalibrasi": cal_p[0], "uji": test_p[0],
                          "n_pasien_latih": len(train_p)})
                putaran.append(r)
                print(f"  h{horizon} cal={cal_p[0]:<10} uji={test_p[0]:<10} "
                      f"cakupan95={r['95']['cakupan_%']:5.1f}%  q={r['95']['q']:.2f}",
                      flush=True)
        ringkas = {}
        for tgt in ["90", "95"]:
            c = [p[tgt]["cakupan_%"] for p in putaran]
            q = [p[tgt]["q"] for p in putaran]
            ringkas[tgt] = {
                "cakupan_rerata_%": round(float(np.mean(c)), 1),
                "cakupan_sd_%": round(float(np.std(c)), 1),
                "cakupan_min_%": round(float(np.min(c)), 1),
                "cakupan_maks_%": round(float(np.max(c)), 1),
                "q_rerata": round(float(np.mean(q)), 3),
                "n_putaran": len(c),
            }
        hasil["horizon"][str(horizon)] = {
            "menit": horizon * int(cadence_min),
            "ringkasan": ringkas,
            "analisis_sebaran": analisis(putaran),
            "per_putaran": putaran,
            "durasi_detik": round(time.time() - t0, 1),
        }
        print(f"  -> h{horizon}: cakupan95 rerata {ringkas['95']['cakupan_rerata_%']}% "
              f"(SD {ringkas['95']['cakupan_sd_%']}, rentang "
              f"{ringkas['95']['cakupan_min_%']}-{ringkas['95']['cakupan_maks_%']})\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Disimpan ke {OUT}")


if __name__ == "__main__":
    main()

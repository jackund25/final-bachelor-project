import torch  # noqa: F401 — dimuat lewat run_app.py sebelum Streamlit (hindari WinError 1114 c10.dll)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root (src)
sys.path.insert(0, str(Path(__file__).parent))          # app (ui)

# Muat .env dari root proyek (kredensial Gemini) — robust terhadap cwd
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import json
import pickle
from datetime import datetime
from time import perf_counter

import pandas as pd
import streamlit as st

from src.alerts import evaluate_divergence
from src.clinical_state import ClinicalDecisionLog
from src.conformal import coverage_achieved, prediction_interval
from src.timing import STAGE_CALIBRATE, STAGE_FEATURES, STAGE_PREDICT, StageTimer
from src.constants import GLUCOSE_LOW, RISK_HYPO, risk_from_condition_class
from src.data.loader import DiabetesDataLoader
from src.rag import RAGPipeline
from src.rag.citations import build_source_list
from ui import (app_header, risk_badge, glucose_zone_chart, zone_legend,
                disclaimer_footer, classify_glucose, render_divergence_alert)

st.set_page_config(page_title="Konsultasi — Pendukung Keputusan Diabetes", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")


# ── Loaders (cache) ───────────────────────────────────────────
@st.cache_data
def load_dataset():
    return DiabetesDataLoader("data/raw").load_preferred_dataset("ohio_t1dm", "latest_generated")[0] \
        .sort_values(["patient_id", "timestamp"])


@st.cache_resource
def load_condition_classifier():
    """Pengklasifikasi kondisi masa depan (hipo/normal/hiper).

    Regresi yang meminimalkan galat kuadrat menyusut ke tengah sehingga jarang melewati
    ambang 70/180: sensitivitas hipoglikemia hanya 14%. Pengklasifikasi sadar-biaya
    menaikkannya ke 44% pada ambang standar (lihat scripts/train_condition_classifier.py).
    """
    cf = Path("models/rf_condition_classifier_h6.pkl")
    if not cf.exists():
        return None
    return pickle.load(open(cf, "rb"))


def predict_condition(window_df, clf, art):
    """Kondisi masa depan menurut pengklasifikasi; None bila model tak tersedia."""
    if clf is None:
        return None
    X = window_df[art["features"]].values.astype(float)
    if clf.get("scaler") is not None:
        X = clf["scaler"].transform(X)
    label = str(clf["model"].predict(X.reshape(1, -1))[0])
    return risk_from_condition_class(label)


@st.cache_resource
def load_artifacts(path: str = "models/rf_inference_bundle.pkl"):
    bf = Path(path)
    if not bf.exists():
        return None
    b = pickle.load(open(bf, "rb"))
    return {"model": b["model"], "scaler": b.get("scaler"),
            "features": b.get("features", ["glucose", "carbs", "insulin", "activity"]),
            "sequence_length": int(b.get("sequence_length", 12)),
            "horizon": int(b.get("prediction_horizon", 6)),
            "use_engineered": bool(b.get("use_engineered", False)),
            "predict_delta": bool(b.get("predict_delta", False)),
            "feature_engineering": dict(b.get("feature_engineering", {}))}


# Bundle per horizon. Sebelumnya aplikasi hanya memuat rf_inference_bundle.pkl (= h6),
# padahal bundle h12 sudah ada di models/ dan deskripsi KF-03 menuntut dua horizon.
HORIZON_BUNDLES = ["models/rf_inference_bundle_h6.pkl", "models/rf_inference_bundle_h12.pkl"]


def load_horizons():
    """Muat semua bundle horizon yang tersedia, urut horizon menaik.

    Jatuh kembali ke bundle generik bila berkas per-horizon tidak ada, supaya instalasi
    lama tetap berjalan dengan satu horizon alih-alih gagal total.
    """
    arts = [a for a in (load_artifacts(p) for p in HORIZON_BUNDLES) if a is not None]
    if not arts:
        generik = load_artifacts()
        arts = [generik] if generik is not None else []
    return sorted(arts, key=lambda a: a["horizon"])


@st.cache_resource
def load_rag():
    """Seluruh parameter RAG dibaca dari config.yaml lewat src/config.py (Tugas 4)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    p = RAGPipeline()
    p.build()
    return p


def build_window(patient_df, art):
    """Bangun window fitur; hitung fitur engineered bila bundle memakainya."""
    seq = art["sequence_length"]
    if art["use_engineered"]:
        from src.data.preprocessor import DataPreprocessor
        feat_df = DataPreprocessor({}).engineer_features(patient_df, **art["feature_engineering"])
    else:
        feat_df = patient_df
    return feat_df.tail(seq).reset_index(drop=True)


def predict_next(window_df, art):
    """Prediksi glukosa absolut; rekonstruksi dari delta bila model dilatih delta."""
    X = window_df[art["features"]].values.astype(float)
    if art["scaler"] is not None:
        X = art["scaler"].transform(X)
    out = float(art["model"].predict(X.reshape(1, -1))[0])
    if art["predict_delta"]:
        out += float(window_df["glucose"].iloc[-1])  # anchor = glukosa terakhir window
    return out


def predict_uncertainty(window_df, art):
    """Std prediksi dari sebaran antar-pohon RF (skala absolut; anchor konstan per sampel).
    Mengkomunikasikan keyakinan model ke dokter — relevan karena deteksi hipoglikemia lemah."""
    est = getattr(art["model"], "estimators_", None)
    if not est:
        return None
    import numpy as np
    X = window_df[art["features"]].values.astype(float)
    if art["scaler"] is not None:
        X = art["scaler"].transform(X)
    Xf = X.reshape(1, -1)
    preds = np.array([t.predict(Xf)[0] for t in est])
    return float(preds.std())


# ── Header & sidebar ──────────────────────────────────────────
app_header("Konsultasi Pasien Diabetes",
           "Alat bantu keputusan klinis — prediksi glukosa & rekomendasi antisipatif", "🩺")

horizons = load_horizons()
if not horizons:
    st.error("Model belum siap. Jalankan pelatihan dari terminal:")
    st.code("python -m src.models.rf_model --config config.yaml --data_source ohio_t1dm")
    disclaimer_footer(); st.stop()
# Horizon terpendek menjadi acuan jendela fitur dan pengklasifikasi kondisi.
art = horizons[0]

try:
    data_df = load_dataset()
except FileNotFoundError:
    st.error("Data pasien belum tersedia."); disclaimer_footer(); st.stop()

patient_ids = sorted(data_df["patient_id"].unique().tolist())
with st.sidebar:
    st.markdown("### 👤 Pasien")
    sel = st.selectbox("Pilih pasien", patient_ids,
                       index=patient_ids.index(st.session_state.get("patient_id", patient_ids[0]))
                       if st.session_state.get("patient_id") in patient_ids else 0)
    st.session_state["patient_id"] = sel
    horizon_min = art["horizon"] * 5
    daftar_horizon = ", ".join(f"+{a['horizon'] * 5} mnt" for a in horizons)
    st.caption(f"Horizon prediksi: **{daftar_horizon}**")
    st.markdown("---")
    st.caption("Alur: tinjau status → prediksi & risiko → rekomendasi → simulasi/keputusan.")

pat = data_df[data_df["patient_id"] == sel].sort_values("timestamp")
seq_len = art["sequence_length"]
if len(pat) < seq_len:
    st.warning(f"Data pasien {sel} belum cukup ({len(pat)}/{seq_len} pembacaan)."); disclaimer_footer(); st.stop()

window_df = build_window(pat, art)
current = float(window_df["glucose"].iloc[-1])
try:
    # Satu baris ringkasan per horizon. Setiap horizon memakai faktor konformalnya
    # SENDIRI: prediksi 60 menit jauh lebih tidak pasti daripada 30 menit, sehingga
    # satu faktor untuk keduanya pasti keliru pada salah satunya.
    ramalan = []
    t_fitur = t_prediksi = t_kalibrasi = 0.0
    for a in horizons:
        _t = perf_counter(); w = build_window(pat, a); t_fitur += perf_counter() - _t
        _t = perf_counter(); p = predict_next(w, a); t_prediksi += perf_counter() - _t
        _t = perf_counter()
        s = predict_uncertainty(w, a)
        interval = prediction_interval(p, s, a["horizon"], level=95)
        t_kalibrasi += perf_counter() - _t
        ramalan.append({
            "art": a, "window": w, "horizon": a["horizon"],
            "menit": a["horizon"] * 5, "pred": p, "std": s,
            "interval": interval,
            "cakupan": coverage_achieved(a["horizon"], level=95),
        })
    utama = ramalan[0]
    pred, pred_std = utama["pred"], utama["std"]
except Exception as exc:  # noqa: BLE001
    st.error("Gagal menjalankan model — kemungkinan environment tidak cocok. "
             "Model dilatih dengan scikit-learn 1.3.0; jalankan aplikasi di environment **diabetes-ta**:")
    st.code("conda activate diabetes-ta\nset PYTHONPATH=.\nstreamlit run app/streamlit_app.py")
    st.caption(f"Detail teknis: {exc}")
    disclaimer_footer(); st.stop()
delta = pred - current
_, cur_label, _ = classify_glucose(current)
_, pred_label, _ = classify_glucose(pred)

# ── SECTION 1: Status + Prediksi ──────────────────────────────
st.subheader(f"Pasien: {sel}")
c1, c2 = st.columns([1.15, 1])
with c1:
    fig = glucose_zone_chart(
        x=list(range(1, len(window_df) + 1)), y=window_df["glucose"].tolist(),
        predicted_value=pred, predicted_x=len(window_df) + 1,
        title=f"Glukosa terkini → prediksi +{horizon_min} menit", height=330)
    st.plotly_chart(fig, use_container_width=True)
    zone_legend()
with c2:
    risk_badge(pred, prefix=f"Prediksi +{horizon_min} mnt")
    st.markdown("")
    st.metric("Glukosa sekarang", f"{current:.0f} mg/dL", help=cur_label)

    # Satu blok per horizon, masing-masing dengan intervalnya sendiri.
    for r in ramalan:
        st.metric(f"Prediksi +{r['menit']} mnt", f"{r['pred']:.0f} mg/dL",
                  delta=f"{r['pred'] - current:+.0f}")
        if r["interval"] is not None:
            lo, hi = r["interval"]
            cov = f"{r['cakupan']:.1f}%" if r["cakupan"] is not None else "?"
            st.caption(f"Rentang 95% terkalibrasi conformal: **{lo:.0f}–{hi:.0f}** mg/dL "
                       f"(cakupan terukur {cov})")
        else:
            # Sengaja TIDAK memakai faktor cadangan. Interval dengan faktor tebakan tidak
            # dapat dibedakan dokter dari interval yang benar-benar terkalibrasi.
            st.caption(f"Interval +{r['menit']} mnt belum terkalibrasi — jalankan "
                       f"`python scripts/conformal_calibration.py --horizon {r['horizon']}`")

    trend = "↑ Meningkat" if delta > 10 else ("↓ Menurun" if delta < -10 else "→ Stabil")
    st.metric("Tren", trend)
    # ringkasan kondisi aktif
    st.markdown(
        f'<div class="card"><h4>Kondisi aktif</h4><p>'
        f'Insulin aktif: {float(window_df["insulin"].iloc[-1]):.2f} u &nbsp;·&nbsp; '
        f'Karbohidrat: {float(window_df["carbs"].iloc[-1]):.0f} g &nbsp;·&nbsp; '
        f'Skor aktivitas: {int(float(window_df["activity"].iloc[-1]))}</p></div>',
        unsafe_allow_html=True)

# Peringatan divergen — nilai jual sistem.
# Logikanya ada di src/alerts.py, bukan di sini: keputusan klinis harus dapat diuji
# tanpa menjalankan Streamlit. Versi lama hanya menyala bila kondisi kini "Dalam
# Target", sehingga ayunan hipo<->hiper tidak pernah tertangkap.
#
# Dievaluasi pada SETIAP horizon dan disebutkan horizon mana yang memicunya. Divergensi
# bisa muncul hanya di +60 menit sementara +30 menit masih terlihat aman; kalau hanya
# horizon pendek yang diperiksa, justru peringatan paling awal yang hilang.
for r in ramalan:
    d = evaluate_divergence(current, r["pred"], r["menit"])
    if d is not None:
        render_divergence_alert(d, horizon_note=f" (+{r['menit']} mnt)")

# Peringatan HIPOGLIKEMIA DINI.
# Prediksi titik regresi menyusut ke tengah: pada ambang <70 ia hanya menangkap 14% kejadian
# hipoglikemia. Dua sinyal tambahan dipakai (lihat scripts/train_condition_classifier.py dan
# scripts/eval_retrieval_realcases.py):
#   (a) pengklasifikasi kondisi sadar-biaya  -> sensitivitas hipoglikemia 44% pada ambang standar
#   (b) batas bawah interval konformal       -> menandai risiko yang masih tercakup ketidakpastian
cond_clf = load_condition_classifier()
pred_condition = predict_condition(window_df, cond_clf, art)
# Interval yang sama dengan yang ditampilkan di atas — dulu faktor 3,3 ditulis ulang di
# sini, sehingga dua interval pada halaman yang sama bisa berbeda tanpa error apa pun.
lo95, hi95 = utama["interval"] if utama["interval"] is not None else (None, None)

if pred_condition == RISK_HYPO and pred >= GLUCOSE_LOW:
    st.warning(f"🔻 **Waspada hipoglikemia:** prediksi titik **{pred:.0f} mg/dL** masih di atas {GLUCOSE_LOW:.0f}, "
               f"namun pengklasifikasi kondisi menandai risiko **hipoglikemia** dalam {horizon_min} menit. "
               f"Pertimbangkan karbohidrat pencegahan & pantau ketat.")
elif lo95 is not None and lo95 < GLUCOSE_LOW <= pred:
    st.warning(f"🔻 **Ketidakpastian menyentuh zona hipoglikemia:** prediksi **{pred:.0f} mg/dL**, "
               f"tetapi batas bawah interval 95% mencapai **{lo95:.0f} mg/dL**. Pantau ketat.")

st.divider()

# ── Tabs: Rekomendasi / Keputusan ─────────────────────────────
tab_rec, tab_log = st.tabs(["🧠 Rekomendasi Klinis", "📝 Catat Keputusan"])

with tab_rec:
    st.caption("Rekomendasi antisipatif berbasis panduan medis (PERKENI/ADA), dikondisikan pada nilai prediksi.")
    if st.button("Buat rekomendasi klinis", type="primary"):
        patient_state = {
            "current_glucose": current,
            "insulin_on_board": float(window_df["insulin"].iloc[-1]),
            "carbs_on_board": float(window_df["carbs"].iloc[-1]),
            "activity_level": int(float(window_df["activity"].iloc[-1])),
            "stress_level": int(float(window_df["stress"].iloc[-1])) if "stress" in window_df else 5,
            # Kueri dikondisikan pada KONDISI hasil pengklasifikasi. Pada validasi silang
            # lintas-fold, varian ini setara dengan pengondisian pada nilai regresi untuk mutu
            # retrieval (0,892 vs 0,893; p=0,31) — jadi bukan itu alasannya dipakai. Alasannya:
            # pengklasifikasi menangkap hipoglikemia jauh lebih baik (14% -> 44%), sehingga kueri
            # untuk kasus paling berbahaya lebih sering menargetkan kondisi yang benar.
            #
            # Batas interval SENGAJA tidak diteruskan ke kueri. Memperluas kueri dengan semua
            # kondisi yang tercakup interval memang menaikkan cakupan kondisi sebenarnya
            # (94,2%), tetapi mengencerkan sinyal sehingga MRR justru turun ke 0,753 — lihat
            # scripts/eval_retrieval_realcases.py. Interval tetap dipakai, namun sebagai
            # PERINGATAN klinis kepada dokter (lihat blok peringatan di atas).
            "predicted_condition": pred_condition,
        }
        with st.spinner("Menyusun rekomendasi..."):
            try:
                # Timer sudah berisi tahap sebelum RAG (rekayasa fitur, prediksi,
                # kalibrasi) supaya angkanya benar-benar ujung-ke-ujung, bukan hanya
                # bagian RAG-nya. KNF-10 menuntut waktu yang dirasakan dokter.
                timer = StageTimer()
                timer.record(STAGE_FEATURES, t_fitur)
                timer.record(STAGE_PREDICT, t_prediksi)
                timer.record(STAGE_CALIBRATE, t_kalibrasi)
                res = load_rag().answer(patient_state=patient_state, prediction=pred,
                                        timer=timer)
                st.session_state["last_rec"] = res
            except Exception as exc:  # noqa: BLE001
                st.session_state["last_rec"] = None
                st.warning(f"Layanan rekomendasi (LLM) tidak tersedia: {str(exc)[:90]}")
    res = st.session_state.get("last_rec")
    if res:
        # Nomor halaman DIAMBIL DARI METADATA chunk, tidak pernah dari teks LLM.
        sources = build_source_list(res.get("retrieved_docs", []), snippet_chars=200)
        st.session_state["last_rec_sources"] = sources

        if not sources:
            st.error(
                "**Tidak ditemukan rujukan panduan yang relevan** pada basis pengetahuan. "
                "Teks di bawah TIDAK didukung kutipan panduan dan tidak boleh diperlakukan "
                "sebagai rekomendasi bersumber."
            )

        # Waktu tanggap ujung-ke-ujung (KNF-10) — ditampilkan apa adanya kepada dokter.
        tm = res.get("timings") or {}
        if tm.get("_total"):
            st.caption(f"⏱️ Waktu tanggap: **{tm['_total']:.2f} dtk** "
                       f"(komputasi lokal {tm.get('_lokal', 0):.2f} dtk, "
                       f"menunggu LLM {tm.get('_jaringan', 0):.2f} dtk)")
            with st.expander("Rincian waktu per tahap"):
                st.table({"tahap": list(k for k in tm if not k.startswith('_')),
                          "detik": [round(tm[k], 3) for k in tm if not k.startswith('_')]})

        st.markdown(f'<div class="card">{res["explanation"]}</div>', unsafe_allow_html=True)

        adv = res.get("advisory", {})
        if adv.get("actions"):
            st.markdown("**Tindakan yang disarankan:**")
            for a in adv["actions"]:
                st.markdown(f"- {a}")

        if sources:
            st.caption(
                "Isi rekomendasi bersumber dari dokumen pedoman yang tercantum pada "
                "**Sumber Rujukan** di bawah. Keputusan akhir berada pada dokter."
            )
            with st.expander(f"📚 Sumber Rujukan ({len(sources)} dokumen)", expanded=False):
                st.caption(
                    "Skor kemiripan mengukur kedekatan kueri dengan potongan dokumen. "
                    "Karena urutan dipilih dengan MMR (yang juga menghindari pengulangan "
                    "isi), peringkat tidak selalu urut menurun terhadap skor."
                )
                for s in sources:
                    judul = s["judul_lengkap"] or s["nama_dokumen"]
                    head = f"**#{s['rank']} · {judul}**"
                    if s["similarity"] is not None:
                        head += f" · kemiripan {s['similarity']:.2f}"
                    st.markdown(head)
                    meta_line = " · ".join(
                        p for p in [
                            f"{s['lembaga']} ({s['tahun']})" if s["lembaga"] else "",
                            s["page_label"],
                            f"`{s['nama_dokumen']}`" if s["nama_dokumen"] else "",
                        ] if p
                    )
                    st.caption(meta_line)
                    st.caption(s["snippet"])
                    st.markdown("")

with tab_log:
    st.caption("Catat keputusan/tinjauan dokter untuk audit.")
    sm = ClinicalDecisionLog(storage_file="data/processed/patient_states.json")
    sm.load()
    init = {"current_glucose": current, "insulin_on_board": float(window_df["insulin"].iloc[-1]),
            "carbs_on_board": float(window_df["carbs"].iloc[-1]),
            "activity_level": int(float(window_df["activity"].iloc[-1])),
            "stress_level": int(float(window_df["stress"].iloc[-1])) if "stress" in window_df else 5,
            "timestamp": datetime.now().isoformat()}
    (sm.create_state if sel not in sm.list_patients() else sm.update_state)(sel, init)
    itype = st.selectbox("Jenis keputusan", ["tinjauan", "setujui rekomendasi", "sesuaikan rekomendasi", "tolak"])
    isum = st.text_input("Catatan", value="Dokter meninjau prediksi & rekomendasi")

    rec = st.session_state.get("last_rec") or {}
    rec_sources = st.session_state.get("last_rec_sources", [])
    if rec_sources:
        st.caption(f"Keputusan akan dicatat bersama {len(rec_sources)} sumber rujukan yang ditampilkan.")
    elif rec:
        st.caption("Rekomendasi terakhir tidak memiliki rujukan; hal ini ikut tercatat.")

    if st.button("Simpan keputusan"):
        ev = sm.log_intervention(
            sel, intervention_type=itype, summary=isum,
            payload={
                "current_glucose": current, "predicted": pred, "risk": pred_label,
                # Rekam sumber PERSIS seperti yang dilihat dokter (termasuk page_label
                # yang sudah diresolusi), bukan retrieved_docs mentah, agar keputusan
                # dapat ditelusuri ke halaman dokumen di kemudian hari.
                "rag_sources": rec_sources,
                "rag_grounded": bool(rec_sources),
                "rag_query": rec.get("query"),
                "rag_provider": rec.get("llm_provider"),
                "rag_explanation": rec.get("explanation"),
            })
        sm.save()
        st.success(f"Keputusan tercatat pada {ev['timestamp']}")

disclaimer_footer()

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

import pandas as pd
import streamlit as st

from src.data.loader import DiabetesDataLoader
from src.digital_twin import DigitalTwinStateManager, PatientDigitalTwin, WhatIfSimulator
from src.rag import RAGPipeline
from ui import (app_header, risk_badge, glucose_zone_chart, zone_legend,
                disclaimer_footer, classify_glucose)

st.set_page_config(page_title="Konsultasi — Diabetes Digital Twin", page_icon="🩺",
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
    return {"hipoglikemia": "hypoglycemia", "normal": "normal",
            "hiperglikemia": "hyperglycemia"}.get(label, label)


@st.cache_resource
def load_artifacts():
    bf = Path("models/rf_inference_bundle.pkl")
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


@st.cache_resource
def load_rag():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    p = RAGPipeline(kb_dir="data/knowledge_base",
                    llm_provider=os.getenv("LLM_PROVIDER", "gemini"),
                    embed_provider=os.getenv("EMBED_PROVIDER", "sentence-transformers"))
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

art = load_artifacts()
if art is None:
    st.error("Model belum siap. Jalankan pelatihan dari terminal:")
    st.code("python -m src.models.rf_model --config config.yaml --data_source ohio_t1dm")
    disclaimer_footer(); st.stop()

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
    st.caption(f"Horizon prediksi: **+{horizon_min} menit**")
    st.markdown("---")
    st.caption("Alur: tinjau status → prediksi & risiko → rekomendasi → simulasi/keputusan.")

pat = data_df[data_df["patient_id"] == sel].sort_values("timestamp")
seq_len = art["sequence_length"]
if len(pat) < seq_len:
    st.warning(f"Data pasien {sel} belum cukup ({len(pat)}/{seq_len} pembacaan)."); disclaimer_footer(); st.stop()

window_df = build_window(pat, art)
current = float(window_df["glucose"].iloc[-1])
try:
    pred = predict_next(window_df, art)
    pred_std = predict_uncertainty(window_df, art)
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
    k = st.columns(2)
    k[0].metric("Glukosa sekarang", f"{current:.0f} mg/dL", help=cur_label)
    k[1].metric(f"Prediksi +{horizon_min} mnt", f"{pred:.0f} mg/dL", delta=f"{delta:+.0f}")
    if pred_std:
        # Faktor conformal ternormalisasi (kalibrasi split-conformal → cakupan ~95.5% tervalidasi).
        # ±1.96·std hanya menutup ~86% (falsely confident); lihat scripts/conformal_calibration.py.
        CONFORMAL_K = 3.3
        lo, hi = pred - CONFORMAL_K * pred_std, pred + CONFORMAL_K * pred_std
        st.caption(f"Rentang keyakinan 95% (terkalibrasi conformal): **{lo:.0f}–{hi:.0f}** mg/dL")
    trend = "↑ Meningkat" if delta > 10 else ("↓ Menurun" if delta < -10 else "→ Stabil")
    st.metric("Tren", trend)
    # ringkasan kondisi aktif
    st.markdown(
        f'<div class="card"><h4>Kondisi aktif</h4><p>'
        f'Insulin aktif: {float(window_df["insulin"].iloc[-1]):.2f} u &nbsp;·&nbsp; '
        f'Karbohidrat: {float(window_df["carbs"].iloc[-1]):.0f} g &nbsp;·&nbsp; '
        f'Aktivitas: {int(float(window_df["activity"].iloc[-1]))} mnt</p></div>',
        unsafe_allow_html=True)

# Peringatan divergen (current normal tapi prediksi bahaya) — nilai jual sistem
if cur_label == "Dalam Target" and pred_label != "Dalam Target":
    st.warning(f"⚠️ **Antisipasi:** kondisi saat ini normal, namun glukosa diprediksi menuju "
               f"**{pred_label}** ({pred:.0f} mg/dL) dalam {horizon_min} menit. Pertimbangkan tindakan pencegahan.")

# Peringatan HIPOGLIKEMIA DINI.
# Prediksi titik regresi menyusut ke tengah: pada ambang <70 ia hanya menangkap 14% kejadian
# hipoglikemia. Dua sinyal tambahan dipakai (lihat scripts/train_condition_classifier.py dan
# scripts/eval_retrieval_realcases.py):
#   (a) pengklasifikasi kondisi sadar-biaya  -> sensitivitas hipoglikemia 44% pada ambang standar
#   (b) batas bawah interval konformal       -> menandai risiko yang masih tercakup ketidakpastian
cond_clf = load_condition_classifier()
pred_condition = predict_condition(window_df, cond_clf, art)
lo95 = hi95 = None
if pred_std:
    lo95, hi95 = pred - 3.3 * pred_std, pred + 3.3 * pred_std

if pred_condition == "hypoglycemia" and pred >= 70.0:
    st.warning(f"🔻 **Waspada hipoglikemia:** prediksi titik **{pred:.0f} mg/dL** masih di atas 70, "
               f"namun pengklasifikasi kondisi menandai risiko **hipoglikemia** dalam {horizon_min} menit. "
               f"Pertimbangkan karbohidrat pencegahan & pantau ketat.")
elif lo95 is not None and lo95 < 70.0 <= pred:
    st.warning(f"🔻 **Ketidakpastian menyentuh zona hipoglikemia:** prediksi **{pred:.0f} mg/dL**, "
               f"tetapi batas bawah interval 95% mencapai **{lo95:.0f} mg/dL**. Pantau ketat.")

st.divider()

# ── Tabs: Rekomendasi / What-If / Keputusan ───────────────────
tab_rec, tab_sim, tab_log = st.tabs(["🧠 Rekomendasi Klinis", "🔬 Simulasi What-If", "📝 Catat Keputusan"])

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
                res = load_rag().answer(patient_state=patient_state, prediction=pred)
                st.session_state["last_rec"] = res
            except Exception as exc:  # noqa: BLE001
                st.session_state["last_rec"] = None
                st.warning(f"Layanan rekomendasi (LLM) tidak tersedia: {str(exc)[:90]}")
    res = st.session_state.get("last_rec")
    if res:
        st.markdown(f'<div class="card">{res["explanation"]}</div>', unsafe_allow_html=True)
        adv = res.get("advisory", {})
        if adv.get("actions"):
            st.markdown("**Tindakan yang disarankan:**")
            for a in adv["actions"]:
                st.markdown(f"- {a}")
        with st.expander("📚 Rujukan panduan medis"):
            for doc in res["retrieved_docs"]:
                st.markdown(f"**#{doc['rank']}** · `{doc['source']}`")
                st.caption(doc["text"][:350] + ("…" if len(doc["text"]) > 350 else ""))

with tab_sim:
    st.caption("Simulasikan dampak intervensi untuk konseling pasien (tanpa mengubah data). "
               "Memakai model farmakokinetik mekanistik agar arah kausal (insulin↓, karbohidrat↑) benar.")
    twin = PatientDigitalTwin(patient_id=sel, initial_state={
        "current_glucose": current, "insulin_on_board": float(window_df["insulin"].iloc[-1]),
        "carbs_on_board": float(window_df["carbs"].iloc[-1]),
        "activity_level": int(float(window_df["activity"].iloc[-1])),
        "stress_level": int(float(window_df["stress"].iloc[-1])) if "stress" in window_df else 5})
    sim = WhatIfSimulator(twin)
    sc = st.columns(4)
    add_carbs = sc[0].slider("Karbohidrat (g)", 0, 120, 0, 5)
    add_ins = sc[1].slider("Insulin (unit)", 0.0, 15.0, 0.0, 0.5)
    add_act = sc[2].slider("Aktivitas (mnt)", 0, 120, 0, 5)
    hz = sc[3].slider("Horizon (mnt)", 30, 240, 60, 15)
    if st.button("Jalankan simulasi"):
        r = twin.simulate_scenario({"carbs_delta": float(add_carbs), "insulin_delta": float(add_ins),
                                    "activity_delta": int(add_act), "stress_delta": 0, "time_horizon": int(hz)})
        sp, cg = float(r["predicted_glucose"]), float(r["glucose_change"])
        risk_badge(sp, prefix=f"Simulasi +{hz} mnt")
        o = st.columns(2)
        o[0].metric("Prediksi simulasi", f"{sp:.0f} mg/dL", delta=f"{cg:+.0f}")
        o[1].metric("vs sekarang", f"{current:.0f} → {sp:.0f} mg/dL")

with tab_log:
    st.caption("Catat keputusan/tinjauan dokter untuk audit.")
    sm = DigitalTwinStateManager(storage_file="data/processed/patient_states.json")
    sm.load()
    init = {"current_glucose": current, "insulin_on_board": float(window_df["insulin"].iloc[-1]),
            "carbs_on_board": float(window_df["carbs"].iloc[-1]),
            "activity_level": int(float(window_df["activity"].iloc[-1])),
            "stress_level": int(float(window_df["stress"].iloc[-1])) if "stress" in window_df else 5,
            "timestamp": datetime.now().isoformat()}
    (sm.create_state if sel not in sm.list_patients() else sm.update_state)(sel, init)
    itype = st.selectbox("Jenis keputusan", ["tinjauan", "setujui rekomendasi", "sesuaikan rekomendasi", "tolak"])
    isum = st.text_input("Catatan", value="Dokter meninjau prediksi & rekomendasi")
    if st.button("Simpan keputusan"):
        ev = sm.log_intervention(sel, intervention_type=itype, summary=isum,
                                 payload={"current_glucose": current, "predicted": pred,
                                          "risk": pred_label})
        sm.save()
        st.success(f"Keputusan tercatat pada {ev['timestamp']}")

disclaimer_footer()

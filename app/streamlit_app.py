import torch  # noqa: F401 — dimuat lewat run_app.py sebelum Streamlit (hindari WinError 1114 c10.dll)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root (src)
sys.path.insert(0, str(Path(__file__).parent))          # app (ui)

# Muat .env dari root proyek (kredensial Gemini) — robust terhadap cwd
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import html
import json
import pickle
from datetime import datetime
from time import perf_counter

import pandas as pd
import streamlit as st

from src.alerts import evaluate_divergence
from src.clinical_state import ClinicalDecisionLog
from src.config import cfg_get
from src.conformal import coverage_achieved, prediction_interval
from src.timing import STAGE_CALIBRATE, STAGE_FEATURES, STAGE_PREDICT, StageTimer
from src.constants import GLUCOSE_LOW, RISK_HYPO, risk_from_condition_class
from src.data.loader import DiabetesDataLoader
from src.logbook import (LAYAK, TIDAK_CUKUP, baca_logbook, gabung_dengan_dataset,
                         periksa_kelayakan)
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


@st.cache_data
def load_logbook_df():
    """Logbook manual. Dipisah dari load_dataset supaya cache-nya dapat dibuang sendiri
    setiap kali dokter menyimpan catatan baru, tanpa memuat ulang seluruh OhioT1DM."""
    return baca_logbook()


@st.cache_resource
def load_condition_classifier():
    """Pengklasifikasi kondisi masa depan (hipo/normal/hiper).

    Regresi yang meminimalkan galat kuadrat menyusut ke tengah sehingga jarang melewati
    ambang 70/180: sensitivitas hipoglikemia hanya 14%. Pengklasifikasi sadar-biaya
    menaikkannya ke 44% pada ambang standar (lihat scripts/train_condition_classifier.py).
    """
    # Prioritas keluarga sama dengan bundle regresi: pengklasifikasi dan regresor
    # wajib sekeluarga, karena keduanya memakai scaler yang sama.
    for nama in ("gbm_condition_classifier_h6.pkl", "rf_condition_classifier_h6.pkl"):
        cf = Path("models") / nama
        if cf.exists():
            return pickle.load(open(cf, "rb"))
    return None


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
def load_artifacts(path: str = "models/gbm_inference_bundle_h6.pkl"):
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
            "feature_engineering": dict(b.get("feature_engineering", {})),
            # Cara ketidakpastian per sampel dihitung. Bundle RF lama tidak punya
            # medan ini; bawaannya sebaran antar-pohon, sehingga bundle lama tetap
            # dimuat dengan perilaku yang sama seperti sebelumnya.
            "std_method": str(b.get("std_method", "tree_variance")),
            "std_models": b.get("std_models"),
            "model_family": str(b.get("model_family", type(b["model"]).__name__))}


# Prioritas diberlakukan per KELUARGA model, bukan per berkas: mencampur h6 dari GBM
# dengan h12 dari RF menghasilkan dua horizon dari dua model berbeda pada satu halaman,
# tanpa cara bagi dokter untuk mengetahuinya.
KELUARGA_BUNDLE = [
    ("GBM", ["models/gbm_inference_bundle_h6.pkl", "models/gbm_inference_bundle_h12.pkl"]),
    ("RF", ["models/rf_inference_bundle_h6.pkl", "models/rf_inference_bundle_h12.pkl"]),
]


def load_horizons():
    """Muat bundle horizon dari SATU keluarga model, urut horizon menaik.

    Keluarga pertama yang punya minimal satu bundle dipakai seluruhnya. Jatuh kembali
    ke bundle generik bila tidak ada berkas per-horizon sama sekali, supaya instalasi
    lama tetap berjalan dengan satu horizon alih-alih gagal total.
    """
    for _, berkas in KELUARGA_BUNDLE:
        arts = [a for a in (load_artifacts(p) for p in berkas) if a is not None]
        if arts:
            return sorted(arts, key=lambda a: a["horizon"])
    generik = load_artifacts("models/rf_inference_bundle.pkl")
    return [generik] if generik is not None else []


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
    """Sigma prediksi per sampel, dari dua sumber bergantung keluarga model.

    ``tree_variance`` memakai sebaran antar-pohon hutan acak; ``quantile_spread``
    memakai rentang dua model kuantil GBM dibagi 3,92, sebab
    HistGradientBoostingRegressor tidak punya ``estimators_``.

    Keduanya heuristik dengan status yang sama: jaminan cakupan konformal tidak
    bergantung pada cara sigma dipilih, karena kuantil konformal menyerap skalanya
    (Angelopoulos & Bates, 2021).

    Mengembalikan ``None`` bila sumbernya tidak tersedia; pemanggil wajib menyatakan
    intervalnya belum terkalibrasi alih-alih menampilkan sigma tebakan.
    """
    import numpy as np
    X = window_df[art["features"]].values.astype(float)
    if art["scaler"] is not None:
        X = art["scaler"].transform(X)
    Xf = X.reshape(1, -1)

    if art.get("std_method") == "quantile_spread":
        qm = art.get("std_models") or {}
        lo_m, hi_m = qm.get("low"), qm.get("high")
        if lo_m is None or hi_m is None:
            return None
        from src.models.gbm_model import GAUSS_95_WIDTH
        lebar = float(hi_m.predict(Xf)[0]) - float(lo_m.predict(Xf)[0])
        # Kuantil yang dilatih terpisah tidak dijamin berurutan; lebar negatif
        # dipangkas agar tidak menghasilkan interval terbalik.
        return max(lebar, 0.0) / GAUSS_95_WIDTH

    est = getattr(art["model"], "estimators_", None)
    if not est:
        return None
    preds = np.array([t.predict(Xf)[0] for t in est])
    return float(preds.std())


# ── Header & sidebar ──────────────────────────────────────────
app_header("Konsultasi Pasien Diabetes",
           "Alat bantu keputusan klinis — prediksi glukosa & rekomendasi antisipatif", "🩺")

horizons = load_horizons()
if not horizons:
    st.error("Model belum siap. Jalankan pelatihan dari terminal:")
    st.code("python -m src.models.gbm_model --config config.yaml --data_source ohio_t1dm")
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

    # T4.2 — logbook manual dulu hanya ditulis ke berkas dan tidak pernah dibaca jalur
    # prediksi. Sekarang dapat disertakan, tetapi lewat pilihan sadar: menyertakan
    # catatan bertimestamp bebas mengubah kerapatan jendela, dan itu harus terlihat.
    lb_semua = load_logbook_df()
    n_lb_pasien = int((lb_semua["patient_id"].astype(str) == str(sel)).sum()) if not lb_semua.empty else 0
    st.markdown("---")
    st.markdown("### 📝 Logbook manual")
    pakai_logbook = st.checkbox(
        f"Sertakan catatan logbook ({n_lb_pasien} untuk pasien ini)",
        value=False, disabled=n_lb_pasien == 0,
        help="Catatan manual digabungkan ke deret CGM menurut stempel waktunya. "
             "Jendela hasil gabungan diperiksa terhadap kriteria kerapatan yang sama "
             "dengan yang dipakai saat melatih model.")
    if n_lb_pasien == 0:
        st.caption("Belum ada catatan untuk pasien ini — isi di halaman **Input Logbook**.")

    st.markdown("---")
    st.caption("Alur: tinjau status → prediksi & risiko → rekomendasi → catat keputusan.")

pat = data_df[data_df["patient_id"] == sel].sort_values("timestamp")
seq_len = art["sequence_length"]
if len(pat) < seq_len:
    st.warning(f"Data pasien {sel} belum cukup ({len(pat)}/{seq_len} pembacaan)."); disclaimer_footer(); st.stop()

# Penggabungan logbook + penjaganya. Bila jendela gabungan tidak sepadan dengan sebaran
# pelatihan, aplikasi KEMBALI ke deret dataset saja dan mengatakan alasannya — bukan
# menginterpolasi jeda supaya jendelanya "terlihat" rapat. Menginterpolasi jeda berjam-jam
# menghasilkan baris yang tampak sah bagi dokter padahal karangan, persis kekeliruan yang
# membuat segmentasi jeda sensor (Tugas 5) diperlukan.
catatan_logbook = None
if pakai_logbook and n_lb_pasien:
    hasil = gabung_dengan_dataset(pat, lb_semua, sel)
    # Kriteria kerapatan dibaca dari config.yaml — sumber yang SAMA dengan yang dipakai
    # create_sequences() saat melatih model. Kalau angkanya ditulis ulang di sini, suatu
    # saat keduanya akan berbeda tanpa ada yang gagal.
    kelayakan = periksa_kelayakan(
        hasil.deret, seq_len,
        cfg_get("model.max_gap_steps"),
        cadence_min=float(cfg_get("data.sampling_interval_min", 5)))
    if kelayakan.verdict == LAYAK:
        pat = hasil.deret
        catatan_logbook = ("ok", hasil, kelayakan)
    else:
        catatan_logbook = ("tolak", hasil, kelayakan)

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

# Asal-usul jendela dinyatakan tepat di atas prediksinya. Kalau catatan manual ikut
# membentuk prediksi, dokter harus tahu — dan kalau catatan itu DITOLAK, dokter harus
# tahu bahwa yang dilihatnya bukan prediksi yang menyertakan catatannya.
if catatan_logbook is not None:
    status, hasil, kelayakan = catatan_logbook
    if status == "ok":
        st.success(
            f"✅ Jendela prediksi memakai **{kelayakan.n_manual} catatan logbook** "
            f"dari {seq_len} baris (jeda terpanjang "
            f"{(kelayakan.jeda_maks_langkah or 0) * 5:.0f} menit, batas "
            f"{(kelayakan.batas_langkah or 0) * 5:.0f} menit)."
            + (f" {hasil.n_manual_menimpa} catatan menimpa pembacaan CGM pada waktu yang sama."
               if hasil.n_manual_menimpa else ""))
    elif kelayakan.verdict == TIDAK_CUKUP:
        st.warning(f"⚠️ Catatan logbook tidak dipakai: {kelayakan.alasan}. "
                   "Prediksi di bawah memakai deret CGM saja.")
    else:
        st.error(
            f"⛔ **Catatan logbook tidak dipakai untuk prediksi ini.** {kelayakan.alasan}. "
            f"Model produksi dilatih hanya pada jendela yang jarak antar-barisnya rapat; "
            f"jendela gabungan ini berada di luar sebaran itu, sehingga galatnya tidak "
            f"terwakili oleh angka validasi mana pun. Jeda tidak diinterpolasi dengan "
            f"sengaja — nilai di dalam jeda memang tidak terobservasi. "
            f"Prediksi di bawah memakai **deret CGM saja**.")
    if hasil.kolom_diabaikan:
        st.caption("Kolom logbook yang tidak menjadi fitur model: "
                   + ", ".join(f"`{k}`" for k in hasil.kolom_diabaikan)
                   + " — tetap tersimpan sebagai rekam jejak klinis, tetapi model "
                     "produksi dilatih tanpa kolom-kolom ini.")
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

# Peringatan divergensi. Logikanya di src/alerts.py, bukan di sini, supaya keputusan
# klinis dapat diuji tanpa menjalankan Streamlit. Dievaluasi pada SETIAP horizon:
# divergensi dapat muncul hanya di +60 menit sementara +30 menit masih tampak aman.
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
            # Kueri dikondisikan pada kondisi hasil pengklasifikasi, bukan pada nilai
            # regresi, karena pengklasifikasi jauh lebih sering menangkap hipoglikemia.
            #
            # Batas interval SENGAJA tidak diteruskan ke kueri: memperluas kueri dengan
            # seluruh kondisi yang tercakup interval menaikkan cakupan kondisi sebenarnya
            # tetapi mengencerkan sinyal sehingga mutu penelusuran turun. Interval tetap
            # dipakai, sebagai peringatan klinis kepada dokter.
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
        sources = build_source_list(res.get("retrieved_docs", []))
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
                # Keterangan MENGIKUTI cara menelusur yang benar-benar dipakai.
                # Sebelumnya ia selalu menjelaskan skor kemiripan, padahal pada jalur
                # hibrida skor itu tidak ada sama sekali — dokter akan mencari angka
                # yang tidak pernah muncul.
                if any(s["similarity"] is not None for s in sources):
                    st.caption(
                        "Skor kemiripan mengukur kedekatan kueri dengan potongan dokumen. "
                        "Karena urutan dipilih dengan MMR (yang juga menghindari pengulangan "
                        "isi), peringkat tidak selalu urut menurun terhadap skor."
                    )
                else:
                    st.caption(
                        "Rujukan diurutkan dengan penelusuran gabungan: kemiripan makna "
                        "(vektor) digabung dengan kecocokan istilah persis (BM25) memakai "
                        "Reciprocal Rank Fusion. Karena kedua skornya berbeda skala dan "
                        "digabung menurut PERINGKAT, tidak ada satu angka kemiripan yang "
                        "dapat ditampilkan."
                    )
                # Kutipan panjang dipakai untuk MENCOCOKKAN hasil penelusuran dengan
                # halaman dokumen aslinya. Pilihan "utuh" disediakan karena potongan
                # yang dikirim ke LLM adalah potongan penuh, bukan versi terpotongnya.
                utuh = st.checkbox(
                    "Tampilkan potongan dokumen secara utuh",
                    value=False,
                    help="Menampilkan seluruh isi potongan persis seperti yang dikirim "
                         "ke model bahasa, untuk dicocokkan dengan halaman sumbernya.",
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

                    teks = s["teks_lengkap"] if utuh else s["snippet"]
                    # Di-escape dan dirender sebagai blok kutipan: isi potongan pedoman
                    # memuat karakter yang bermakna di markdown (bullet, angka berimbuh
                    # titik, tanda bintang) sehingga membiarkannya diparse akan mengubah
                    # teks yang justru sedang dibandingkan dengan dokumen aslinya.
                    st.markdown(
                        '<div style="border-left:3px solid rgba(127,127,127,.4);'
                        'padding:.45rem .85rem;margin:.15rem 0 .35rem 0;'
                        'font-size:.9rem;line-height:1.55;opacity:.9;'
                        'white-space:pre-wrap;">'
                        f'{html.escape(teks)}</div>',
                        unsafe_allow_html=True,
                    )
                    # Dua keterangan berikut sengaja dipisah, sebab menjelaskan dua
                    # pemotongan yang berbeda: pemotongan TAMPILAN yang dapat dibatalkan
                    # di sini, dan pemotongan INDEKS yang sudah terjadi saat korpus
                    # dipecah dan tidak dapat dibatalkan dari antarmuka.
                    if s["snippet_terpotong"] and not utuh:
                        cara = {
                            "batas_kalimat": "berhenti di akhir kalimat",
                            "batas_kata": "berhenti di batas kata (tidak ada akhir "
                                          "kalimat pada rentang ini)",
                        }.get(s.get("cara_potong", ""), "dipotong")
                        st.caption(
                            f"Ditampilkan {len(s['snippet'])} dari {s['n_char']} karakter "
                            f"potongan, {cara} — centang kotak di atas untuk melihat utuhnya."
                        )
                    if not s.get("mulai_kalimat_utuh", True) or not s.get(
                        "akhir_kalimat_utuh", True
                    ):
                        tepi = []
                        if not s.get("mulai_kalimat_utuh", True):
                            tepi.append("awal")
                        if not s.get("akhir_kalimat_utuh", True):
                            tepi.append("akhir")
                        st.caption(
                            f"⚠️ Potongan ini bersambung di {' dan '.join(tepi)}: "
                            "kalimatnya berlanjut pada potongan tetangga dokumen yang sama."
                        )
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

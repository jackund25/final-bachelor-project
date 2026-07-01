# Prediction-Conditioned RAG untuk Pengelolaan Diabetes

Tugas Akhir — Institut Teknologi Bandung

Penulis: Daffari Adiyatma (18222003)

Pembimbing:
- Prof. Dr. Ir. Suhono Harso Supangkat, M.Eng.
- Ir. Devi Willieam Anggara S.T., M.Phil., Ph.D

## Ringkasan

Sistem pendukung keputusan klinis (*decision support*) untuk pengelolaan glukosa darah yang
menggabungkan prediksi glukosa jangka pendek dengan *Retrieval-Augmented Generation* (RAG).
Kontribusi utama adalah **Prediction-Conditioned RAG**: kueri pengambilan panduan medis
dibangun dari **nilai glukosa yang diprediksi** (bukan hanya kondisi saat ini), sehingga
rekomendasi bersifat antisipatif terhadap kondisi yang akan datang.

Sistem diposisikan sebagai *simplified data-driven digital twin* (bukan digital twin
fisiologis penuh) dan digunakan secara *doctor-mediated*: keputusan medis final tetap pada
dokter.

## Komponen

- **Prediksi glukosa**: Random Forest (model utama) dengan pembanding LSTM, pada horizon
  standar BGLP Challenge (+30 dan +60 menit).
- **Prediction-Conditioned RAG**: pembangun kueri berbasis prediksi + pengambilan dari basis
  pengetahuan klinis (ChromaDB) + ringkasan oleh LLM (Google Gemini), dengan mode pembanding
  *standard* untuk membuktikan kontribusi.
- **Basis pengetahuan**: dokumen panduan klinis (PERKENI, ADA) di-*ingest* ke ChromaDB.
- **Digital twin sederhana**: proyeksi state metabolik (insulin/karbohidrat aktif) untuk
  simulasi *what-if*.
- **Aplikasi Streamlit**: antarmuka konsultasi klinis (status pasien, prediksi & risiko,
  rekomendasi, simulasi, pencatatan keputusan).

## Dataset

**OhioT1DM** (Marling & Bunescu, 2020) — data CGM Diabetes Tipe 1 dari 12 pasien, digunakan
sebagai *surrogate* untuk pembuktian teknis. Parser menghasilkan dua berkas:
- `data/raw/ohio_t1dm_merged.csv` — timeline CGM 5-menit (pelatihan model).
- `data/raw/ohio_t1dm_smbg.csv` — timeline `finger_stick` (SMBG nyata) untuk skenario SMBG.

Data OhioT1DM tunduk pada *Data Use Agreement* dan tidak disertakan dalam repositori.

## Tech Stack

Python 3.11, scikit-learn 1.3.0 (Random Forest), TensorFlow 2.15 (LSTM), LangChain +
Google Gemini, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Streamlit, pandas, Plotly.

## Penyiapan Lingkungan

Proyek dikembangkan pada environment conda `diabetes-ta` (Python 3.11). Versi paket dikunci
(mis. scikit-learn 1.3.0) agar artefak model yang tersimpan dapat dimuat dengan benar.

```bash
conda create -n diabetes-ta python=3.11 -y
conda activate diabetes-ta
pip install -r requirements.txt
```

Salin `.env.example` menjadi `.env` dan isi `GOOGLE_API_KEY` (Google AI Studio) untuk fitur
advisory LLM.

## Reproduksi

Jalankan dari root proyek dengan environment `diabetes-ta` aktif (`set PYTHONPATH=.` pada
Windows CMD atau `$env:PYTHONPATH="."` pada PowerShell):

```bash
# 1. Parse OhioT1DM XML -> CSV (CGM + SMBG)
python -m src.data.ohio_parser

# 2. Latih model Random Forest (bundle inferensi untuk aplikasi)
python -m src.models.rf_model --config config.yaml --data_source ohio_t1dm

# 3. Evaluasi komparatif RF vs LSTM (RMSE, MAE, Clarke Error Grid) pada +30 & +60 menit
python scripts/eval_rf_lstm.py --config config.yaml

# 4. Ingest basis pengetahuan (manual_kb + PDF panduan) ke ChromaDB (idempoten)
python scripts/reingest_kb.py

# 5. Ablation Prediction-Conditioned vs standard RAG
python scripts/ablation_rag.py

# 6. Jalankan aplikasi
streamlit run app/streamlit_app.py
```

Hasil evaluasi tersimpan di `results/` (CSV dan gambar).

## Pengujian

```bash
pytest tests/ -q
```

## Struktur Proyek

```
app/           Aplikasi Streamlit (konsultasi klinis) + komponen UI
src/data/      Loader, parser OhioT1DM, praproses, kontrak data
src/models/    Random Forest, LSTM, metrik
src/rag/       Pipeline RAG, prediction-conditioned query, retriever, knowledge base
src/digital_twin/  Patient twin, simulator what-if, state manager
scripts/       Skrip evaluasi & utilitas (eval, ablation, re-ingest KB, SUS)
tests/         Uji unit & integrasi
results/       Keluaran evaluasi (metrik, gambar)
```

## Keterbatasan

- OhioT1DM adalah dataset Diabetes Tipe 1 (*surrogate*); diperlukan validasi pada data
  Tipe 2 Indonesia sebelum penggunaan klinis.
- Digital twin bersifat sederhana (berbasis formula), bukan model fisiologis multi-skala.
- Belum ada uji klinis; evaluasi bersifat teknis (akurasi prediksi dan kualitas retrieval).
- Embedding pencarian belum dioptimalkan untuk Bahasa Indonesia.

## Lisensi

Penggunaan akademik — Tugas Akhir Institut Teknologi Bandung.

## Acknowledgments

- Marling & Bunescu (Ohio University) atas dataset OhioT1DM.
- PERKENI dan ADA atas panduan klinis diabetes.

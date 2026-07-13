# Prediction-Conditioned RAG untuk Pengelolaan Diabetes

Sistem pendukung keputusan klinis yang menyatukan prediksi kadar glukosa jangka pendek
dengan *Retrieval-Augmented Generation* (RAG) berbasis pedoman medis.

Tugas Akhir — Program Studi Sistem dan Teknologi Informasi, Institut Teknologi Bandung.

| | |
| --- | --- |
| Penulis | Daffari Adiyatma (18222003) |
| Pembimbing | Prof. Dr. Ir. Suhono Harso Supangkat, M.Eng. |
| | Ir. Devi Willieam Anggara, S.T., M.Phil., Ph.D. |

---

## Ringkasan

Kontribusi utama penelitian ini adalah **Prediction-Conditioned RAG (PC-RAG)**: kueri
pengambilan dokumen dibangun dari **nilai glukosa yang diprediksi**, bukan dari kondisi
pasien saat ini. Akibatnya, dokumen yang diambil dan rekomendasi yang dihasilkan bersifat
antisipatif terhadap kondisi yang akan datang — misalnya mengenali risiko hipoglikemia
sebelum benar-benar terjadi.

Sistem diposisikan sebagai *simplified data-driven digital twin* (bukan *digital twin*
fisiologis penuh) dan dijalankan secara *doctor-mediated*: rekomendasi divalidasi dokter
sebelum diteruskan kepada pasien. Seluruh komponen berjalan pada perangkat standar tanpa
GPU, CGM *real-time*, maupun rekam medis elektronik.

## Komponen

- **Prediksi glukosa** — Random Forest sebagai model utama dengan LSTM sebagai pembanding,
  pada horizon standar BGLP Challenge (+30 dan +60 menit). Fitur berbasis fisiologi:
  *insulin-on-board*, *carbs-on-board*, tren, dan pola diurnal.
- **Prediction-Conditioned RAG** — pembangun kueri terkondisi prediksi, *retriever* MMR di
  atas ChromaDB, dan generator berbasis Google Gemini. Tersedia mode RAG standar sebagai
  pembanding untuk membuktikan kontribusi.
- **Basis pengetahuan** — pedoman klinis PERKENI dan ADA yang di-*ingest* ke ChromaDB.
- **Digital twin sederhana** — proyeksi *state* metabolik dan simulasi *what-if* berbasis
  formula farmakokinetik mekanistik (bukan model prediktif, agar arah kausal intervensi
  tetap benar).
- **Aplikasi Streamlit** — antarmuka konsultasi klinis: status pasien, prediksi dan risiko,
  rekomendasi tertelusur, simulasi *what-if*, serta pencatatan keputusan dokter.

## Hasil Utama

| Evaluasi | Hasil |
| --- | --- |
| Prediksi CGM, +30 menit (Random Forest) | RMSE 22,60 mg/dL — Clarke A+B 94,35% |
| Prediksi CGM, +60 menit (Random Forest) | RMSE 34,24 mg/dL — Clarke A+B 86,94% |
| Prediksi CGM, +30 menit (LSTM, pembanding) | RMSE 22,04 mg/dL — Clarke A+B 94,60% |
| Skenario SMBG (`finger_stick` nyata), +30 menit | RMSE 27,09 mg/dL — Clarke A+B 92,09% |
| Sensitivitas hipoglikemia (ambang standar 70 mg/dL) | 14,0% (regresi) → **44,4%** (pengklasifikasi kondisi) |
| Interval prediksi konformal (nominal 95%) | cakupan empiris 96,5% |

Sebagai konteks, RMSE 22,60 mg/dL berada pada kisaran model *deep learning* terpublikasi
pada OhioT1DM (18,26–22,12 mg/dL), meski penelitian ini memakai pembagian **lintas-pasien**
(pasien uji tak pernah dilihat model) yang lebih berat daripada pembagian temporal yang
umum dipakai literatur.

### Evaluasi *retrieval*: dua tahap

**Demonstrasi terkontrol** (6 kasus terkurasi): MRR 0,225 → 0,889. Angka ini adalah *batas
atas yang optimistis* — kondisi relevannya ditetapkan dari nilai yang diprediksi, sehingga
kesalahan prediksi tidak terhukum.

**Kasus nyata** (120 kasus dari 2 pasien *hold-out*; relevansi ditetapkan dari glukosa yang
**benar-benar terjadi** pada t+30, sehingga kesalahan prediksi dihukum):

| Mode kueri | MRR (natural) | MRR (kasus divergen) |
| --- | --- | --- |
| Acak (*sanity baseline*) | 0,375 | 0,406 |
| RAG standar | 0,863 | 0,166 |
| PC-RAG (regresi) | 0,887 (p = 0,11, tidak signifikan) | 0,340 |
| **PC-RAG + pengklasifikasi kondisi** | **0,902** (p = 0,03) | 0,271 |
| *Oracle* (prediksi sempurna) | 1,000 | **0,994** |

**Temuan utama.** Mekanisme PC-RAG terbukti sahih — dengan kondisi masa depan yang benar,
*retrieval* nyaris sempurna (0,994). Namun manfaatnya di lapangan dibatasi oleh
**prediktornya**: pada kasus divergen, regresi hanya benar menebak kondisi masa depan 15,8%
kali. Ini akar yang sama dengan rendahnya sensitivitas hipoglikemia. Memprediksi *kondisi*
secara langsung (pengklasifikasi sadar-biaya) memperbaiki keduanya sekaligus.

**Catatan pada skenario SMBG.** Pada horizon +30 menit, model praktis setara dengan
*baseline persistence* (RMSE 27,09 vs 27,33 mg/dL): jarak antar-pembacaan SMBG (median
124,8 menit) jauh lebih panjang daripada horizon prediksinya. Penambahan fitur sadar-waktu
diuji dan **tidak membantu** (27,20 mg/dL) — keterbatasannya bersifat informasional, bukan
representasional. Keunggulan yang jelas baru muncul pada +60 menit (42,86 vs 47,87 mg/dL).

## Dataset

**OhioT1DM** (Marling & Bunescu, 2020) — data 12 pasien diabetes tipe 1, digunakan sebagai
*surrogate* untuk pembuktian konsep. Parser menghasilkan dua berkas:

- `data/raw/ohio_t1dm_merged.csv` — timeline CGM 5 menit, dipakai untuk pelatihan model.
- `data/raw/ohio_t1dm_smbg.csv` — timeline `finger_stick` (SMBG nyata), dipakai untuk
  skenario *deployment*. Bukan hasil *downsampling* artifisial dari kanal CGM.

Dataset tunduk pada *Data Use Agreement* dan tidak disertakan dalam repositori ini.

## Teknologi

Python 3.11, scikit-learn 1.3.0 (Random Forest), TensorFlow 2.15 (LSTM), LangChain dengan
Google Gemini, ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`), Streamlit, pandas,
matplotlib, dan Plotly.

## Penyiapan Lingkungan

Proyek dikembangkan pada environment conda `diabetes-ta` (Python 3.11). Versi paket dikunci
— terutama scikit-learn 1.3.0 — agar artefak model yang tersimpan dapat dimuat dan
mereproduksi angka yang dilaporkan.

```bash
conda create -n diabetes-ta python=3.11 -y
conda activate diabetes-ta
pip install -r requirements.txt
```

Salin `.env.example` menjadi `.env`, lalu isi `GOOGLE_API_KEY` (Google AI Studio) untuk
mengaktifkan generator rekomendasi.

## Reproduksi

Jalankan dari root proyek dengan environment `diabetes-ta` aktif (`set PYTHONPATH=.` pada
Windows CMD atau `$env:PYTHONPATH="."` pada PowerShell).

Alur dasar:

```bash
# 1. Parse OhioT1DM XML menjadi CSV (kanal CGM dan finger_stick)
python -m src.data.ohio_parser

# 2. Latih Random Forest dan hasilkan bundle inferensi untuk aplikasi
python -m src.models.rf_model --config config.yaml --data_source ohio_t1dm

# 3. Verifikasi bahwa artefak model mereproduksi angka yang dilaporkan
python scripts/verify_prediction_artifacts.py

# 4. Jalankan aplikasi
streamlit run app/streamlit_app.py
```

Evaluasi yang menghasilkan angka pada tabel Hasil Utama:

```bash
# Prediksi: perbandingan RF vs LSTM, validasi silang lintas-pasien, uji signifikansi
python scripts/eval_rf_lstm.py --config config.yaml
python scripts/crossval_rf_vs_lstm.py

# Keamanan klinis: kalibrasi konformal dan deteksi hipoglikemia
python scripts/conformal_calibration.py
python scripts/eval_hypo_uncertainty.py
python scripts/improve_hypo_detection.py

# Skenario deployment SMBG pada timeline finger_stick nyata (+ varian sadar-waktu)
python scripts/eval_smbg_deployment.py

# Pengklasifikasi kondisi masa depan (hipo 14% -> 44%)
python scripts/train_condition_classifier.py

# Retrieval pada kasus nyata + uji rantai kausal (oracle / noise / acak)
python scripts/eval_retrieval_realcases.py

# Bukti kontribusi fitur multimodal (3,4% -> 22,3%)
python scripts/eval_feature_importance.py

# RAG: ingest basis pengetahuan, lalu bandingkan PC-RAG terhadap RAG standar
python scripts/reingest_kb.py
python scripts/ablation_rag_fullkb.py       # kebaruan pada korpus penuh
python scripts/ablation_rag.py              # ablation terkontrol + contoh kualitatif
python scripts/sensitivity_analysis.py      # sensitivitas top_k dan ukuran chunk
python scripts/eval_generation_novelty.py   # kualitas keluaran LLM
```

Seluruh keluaran evaluasi tersimpan di `results/` dalam bentuk JSON, CSV, dan gambar.

## Pengujian

```bash
pytest tests/ -q
```

## Struktur Proyek

```
app/                Aplikasi Streamlit (konsultasi klinis) dan komponen antarmuka
src/data/           Parser OhioT1DM, praproses, rekayasa fitur, kontrak data
src/models/         Random Forest (utama) dan LSTM (pembanding)
src/rag/            Pipeline RAG, kueri terkondisi prediksi, retriever, basis pengetahuan
src/digital_twin/   Patient twin, simulator what-if, state manager
src/utils/          Metrik evaluasi dan logging
scripts/            Skrip evaluasi yang mereproduksi seluruh angka laporan
tests/              Uji unit dan integrasi
notebooks/          Eksplorasi data (EDA)
results/            Keluaran evaluasi (metrik dan gambar)
models/             Metrik model terlatih (berkas .pkl tidak di-commit)
docs/laporan_TA/    Sumber LaTeX laporan tugas akhir
config.yaml         Konfigurasi terpusat (data, model, RAG, evaluasi)
run_app.py          Entry point aplikasi
```

## Keterbatasan

- OhioT1DM adalah dataset diabetes tipe 1 yang dipakai sebagai *surrogate*; diperlukan
  validasi pada data logbook pasien tipe 2 di Indonesia sebelum penggunaan klinis.
- Pada skenario SMBG, prediksi +30 menit belum melampaui *baseline persistence*; manfaat
  prediktif atas data logbook yang jarang baru nyata pada horizon yang lebih panjang.
- Sensitivitas deteksi hipoglikemia meningkat tiga kali lipat menjadi 44,4% pada ambang
  standar, tetapi masih belum memadai untuk penggunaan klinis mandiri (PPV turun ke 39,4%).
- Manfaat PC-RAG terkonsentrasi pada 13,3% jendela yang divergen; pada distribusi natural
  peningkatannya signifikan tetapi moderat (0,863 → 0,902).
- Digital twin bersifat sederhana (berbasis formula), bukan model fisiologis multi-skala.
- Belum ada uji klinis maupun validasi lapangan; evaluasi bersifat teknis.
- Evaluasi generasi memakai enam kasus dan tanpa penilaian pakar klinis.
- Relevansi dokumen pada evaluasi *retrieval* ditetapkan lewat klasifikasi kata kunci
  berbobot, bukan penilaian pakar — meski risiko sirkularitas ditutup dengan memakai nilai
  kebenaran dari kondisi yang benar-benar terjadi dan diuji terhadap *baseline* acak.
- Model *embedding* belum di-*fine-tune* untuk korpus diabetes berbahasa Indonesia.

## Lisensi

Penggunaan akademik — Tugas Akhir, Institut Teknologi Bandung.

## Penghargaan

- Marling dan Bunescu (Ohio University) atas dataset OhioT1DM.
- PERKENI dan American Diabetes Association atas pedoman klinis yang menjadi basis
  pengetahuan sistem.

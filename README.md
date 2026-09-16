# RAG Terkondisi-Prediksi untuk Dukungan Keputusan Klinis Diabetes Melitus Tipe 1

Sistem pendukung keputusan klinis yang mengondisikan proses *retrieval* pada kadar glukosa
**terprediksi**, bukan pada kadar glukosa terkini, sehingga pedoman yang dirujuk sejalan
dengan kondisi yang akan dihadapi pasien.

Tugas Akhir — Program Studi Sistem dan Teknologi Informasi, Institut Teknologi Bandung.

| | |
| --- | --- |
| Penulis | Daffari Adiyatma (18222003) |
| Pembimbing | Prof. Dr. Ir. Suhono Harso Supangkat, M.Eng. |
| | Ir. Devi Willieam Anggara, S.T., M.Phil., Ph.D. |

---

## Peringatan penggunaan

Perangkat lunak ini adalah **artefak penelitian**, bukan perangkat medis. Sistem belum
pernah diuji pada pasien nyata di layanan klinis, tidak memiliki persetujuan regulator mana
pun, dan tidak boleh digunakan untuk mengambil keputusan terapi. Seluruh keluaran dirancang
untuk ditinjau dokter, dan sistem tidak pernah melakukan tindakan otonom.

## Ringkasan

Sistem RAG konvensional menelusuri korpus memakai kondisi pasien **saat ini**. Pada
pengelolaan diabetes, kondisi dapat berubah arah dalam hitungan puluhan menit, sehingga
pedoman yang ditemukan dapat keliru sasaran ketika keluarannya dibaca. Penelitian ini
mengusulkan **RAG terkondisi-prediksi**: kueri penelusuran dibangun dari kadar glukosa hasil
prediksi jangka pendek beserta label kondisinya, lalu potongan pedoman yang diperoleh dipakai
untuk menyusun rekomendasi berbahasa Indonesia yang dibumikan pada dokumen sumber sampai
nomor halaman.

Artefak terdiri atas tiga bagian: modul prediksi glukosa dengan estimasi ketidakpastian,
pipeline RAG terkondisi-prediksi di atas korpus pedoman klinis, dan antarmuka CDSS termediasi
dokter.

## Kontribusi

1. Perumusan dan realisasi mekanisme pengondisian *retrieval* pada kondisi terprediksi,
   beserta evaluasinya yang memisahkan kasus divergen dari distribusi natural.
2. Protokol evaluasi *retrieval* yang menyertakan lengan *oracle* sebagai batas atas,
   sehingga selisih antara sistem dan *oracle* dapat diatribusikan pada mutu prediktor dan
   bukan pada mekanisme pengondisian.
3. Rangkaian pengaman keluaran: verifikasi keterlacakan setiap angka klinis terhadap dokumen
   yang ditelusuri, dan pemisahan status ketika data atau *evidence* tidak memadai.

## Arsitektur

| Lapisan | Realisasi |
| --- | --- |
| Antarmuka | *Progressive Web Application*, Next.js dan React (`frontend/`) |
| Layanan | FastAPI dan Uvicorn (`backend/`) |
| Modul prediksi | `HistGradientBoostingRegressor`, prediksi delta dengan rekonstruksi (`src/models/`) |
| Ketidakpastian | Prediksi konformal terpisah per horizon (`src/models/`) |
| Pipeline RAG | Transformasi kueri terkondisi-prediksi, BM25, resolusi sitasi (`src/rag/`) |
| Basis pengetahuan | ChromaDB, `all-MiniLM-L6-v2`, potongan 900 karakter dengan tumpang tindih 120 |
| Keadaan klinis | Representasi terstruktur `PatientState` dari keluaran prediktor (`src/patient_state.py`) |
| Penyimpanan aplikasi | Supabase |

## Data

**OhioT1DM** (Marling & Bunescu, 2020) — dua belas pasien diabetes melitus tipe 1 pengguna
pompa insulin. Dataset tunduk pada *Data Use Agreement* dan **tidak disertakan** dalam
repositori ini. Parser menghasilkan dua berkas: `data/raw/ohio_t1dm_merged.csv` (kanal CGM,
interval 5 menit) dan `data/raw/ohio_t1dm_smbg.csv` (kanal *finger-stick* nyata, bukan hasil
penjarangan artifisial dari kanal CGM).

**Korpus pedoman klinis** — 12 dokumen, 543 halaman cetak, terpotong menjadi 2.233 potongan.
Penerbitnya PERKENI, IDAI UKK Endokrinologi Anak dan Remaja, ADA bersama EASD, ISPAD, serta
konsensus internasional ATTD, dengan tahun terbit 2015 sampai 2023. Daftar lengkapnya pada
`data/knowledge_base/manifest.csv`.

## Hasil evaluasi

Seluruh angka di bawah berasal dari berkas pada `results/__Hasil_Akhir__/`, yaitu hasil yang
dipakai laporan dan terpisah dari jejak eksperimen di `results/`. Manifes berikut sidik jari
SHA-256 tiap berkas ada pada `results/__Hasil_Akhir__/README.md`.

### Modul prediksi glukosa

Dua protokol dilaporkan terpisah dan sengaja tidak digabungkan menjadi satu angka, karena
proses pembentukan sampelnya berbeda. Evaluasi *hold-out* adalah pengukuran resmi pada
konfigurasi yang ditetapkan pada artefak; validasi silang lintas-pasien memperlihatkan
variasi kinerja antar-fold ketika model diuji pada pasien yang tidak pernah dilihat saat
pelatihan.

**Hold-out** (Tabel VI.3 laporan):

| Horizon | Model | RMSE (mg/dL) | MAE (mg/dL) | Clarke A+B (%) |
| --- | --- | --- | --- | --- |
| +30 menit | **Gradient Boosting (produksi)** | 22,251 | 14,724 | 94,70 |
| +30 menit | Random Forest | 22,598 | 15,098 | 94,35 |
| +30 menit | LSTM | 21,983 | 14,552 | 94,72 |
| +60 menit | **Gradient Boosting (produksi)** | 33,816 | 24,411 | 87,32 |
| +60 menit | Random Forest | 34,240 | 24,825 | 86,94 |
| +60 menit | LSTM | 34,144 | 24,622 | 87,22 |

**Validasi silang lintas-pasien, enam fold, horizon +30 menit** (Tabel VI.4 laporan),
rerata ± simpangan baku antar-fold:

| Model | RMSE (mg/dL) | Clarke A+B (%) |
| --- | --- | --- |
| **Gradient Boosting (produksi)** | 20,72 ± 1,16 | 94,69 ± 0,50 |
| Random Forest | 21,08 ± 1,15 | 94,35 ± 0,55 |
| LSTM | 20,60 ± 1,33 | 94,89 ± 0,52 |

*Sumber: `T1_prediksi/holdout_semua_horizon.csv` dan `T1_prediksi/crossfold_h30m.json`.*

Selisih antarmodel berada di bawah variasi antar-fold, sehingga ketiganya diperlakukan
sebagai berkinerja sebanding. Gradient Boosting dipertahankan sebagai model produksi atas
pertimbangan gabungan antara kinerja yang sebanding dengan pembanding dan karakteristik
operasional yang sesuai dengan sistem: keterterapan tanpa akselerator grafis dan keluaran
yang sesuai untuk pembentukan kondisi klinis.

### Ketidakpastian dan deteksi hipoglikemia

Cakupan interval konformal diukur dengan pasien pengukur yang tidak pernah dipakai melatih
maupun mengkalibrasi, diulang dua belas kali dengan pasangan pasien berbeda. Pada target 95%,
cakupan empiris rerata **94,7% ± 2,1** pada +30 menit dan **94,5% ± 2,8** pada +60 menit;
rentang antar-putaran melingkupi target pada kedua horizon. Faktor kalibrasi 2,15 dan 2,14
menghasilkan lebar interval rata-rata 73,8 dan 120,6 mg/dL.

Regresi yang diambang menghasilkan sensitivitas hipoglikemia 17,3% dengan PPV 56,6%.
Pengklasifikasi kondisi tiga kelas menaikkan sensitivitas menjadi **67,2%** dengan konsekuensi
PPV turun menjadi 31,9% dan akurasi keseluruhan turun dari 89,4% menjadi 86,1%. Pertukaran ini
dilaporkan apa adanya; sensitivitas tersebut belum memadai untuk penggunaan klinis mandiri.

*Sumber: `T1_prediksi/cakupan_konformal.json`, `T1_prediksi/konformal_h30m.json`, `T1_prediksi/konformal_h60m.json`, `T1_prediksi/pengklasifikasi_kondisi.json`.*

### Penelusuran terkondisi-prediksi

Enam lipatan, 60 kasus per himpunan per lipatan, `top_k` = 5, lengan BM25 sebagaimana
konfigurasi produksi.

| Himpunan | Lengan | MRR | Hit@1 |
| --- | --- | --- | --- |
| Kasus divergen | RAG standar | 0,190 | 0,000 |
| Kasus divergen | **Terkondisi-prediksi** | **0,284** | 0,095 |
| Kasus divergen | Terkondisi + pengklasifikasi | 0,312 | 0,147 |
| Kasus divergen | *Oracle* (batas atas) | 0,778 | 0,667 |
| Distribusi natural | RAG standar | 0,700 | 0,583 |
| Distribusi natural | Terkondisi-prediksi | 0,714 | 0,594 |
| Distribusi natural | *Oracle* (batas atas) | 0,769 | 0,653 |

*Sumber: `T2_retrieval/crossfold.json`.*

Peningkatan terkonsentrasi pada kasus divergen, yaitu ketika kondisi terprediksi berbeda dari
kondisi terkini: MRR naik 0,095 dan unggul pada seluruh enam lipatan (Wilcoxon p = 0,031).
Pada distribusi natural peningkatannya kecil dan tidak signifikan, sebagaimana diharapkan
karena kondisi terkini dan terprediksi sebagian besar berimpit. Jarak menuju *oracle* yang
masih lebar (0,284 berbanding 0,778) menunjukkan batasnya terletak pada ketepatan prediktor,
bukan pada mekanisme pengondisian.

### Keamanan keluaran

Pada enam kasus uji, RAG terkondisi-prediksi memperoleh kemiripan terhadap rujukan
(`sim_ref`) 0,571 dan cakupan tindakan (`action coverage`) 0,700, dibandingkan 0,517 dan
0,667 pada RAG standar. Arahnya meningkat pada kedua ukuran, tetapi ukuran sampel yang
kecil membuat temuan ini bersifat indikatif. Pada dua kasus uji kepatuhan dengan model
terbaru, seluruh aturan keluaran yang diperiksa dipatuhi, termasuk penanda sitasi, larangan
mengarang nomor halaman, dan pernyataan batas keputusan.

Pengujian kestabilan penilai otomatis RAGAS menunjukkan `context_precision` dan
`context_recall` stabil antar-run, sedangkan `answer_relevancy` berubah besar: pada
sepuluh kasus reratanya bergeser dari 0,763 menjadi 0,587, dengan perubahan hingga 0,685
pada satu kasus. Karena itu `answer_relevancy` tidak dipakai sebagai dasar klaim.

*Sumber: `T3_generation/generation_novelty.json`, `T3_generation/kestabilan_ragas.json`.*

### Keterterapan operasional

Diukur seluruhnya pada CPU tanpa akselerator grafis. Waktu tanggap ujung-ke-ujung bermedian
**2,474 detik**, dengan komputasi lokal **0,339 detik** dan sisanya menunggu layanan LLM
(85,0% dari total).

*Sumber: `operasional/latensi_ujung_ke_ujung.json`.*

### Evaluasi ahli

Dua belas responden: 3 dokter spesialis, 1 dokter, 7 dokter muda, dan 1 mahasiswa
kedokteran. Sembilan dari dua belas belum pernah memakai CDSS, dan sepuluh dari dua belas
belum pernah menangani pasien pengguna CGM.

Skor *System Usability Scale* **55,21** (SB 9,20; median 51,25; rentang 40,0 sampai 72,5;
KI 95% 49,36 sampai 61,05). Pada skenario hipoglikemia, 9 dari 12 responden menilai
rekomendasi sesuai pedoman tetapi perlu verifikasi dokter; pada skenario hiperglikemia
11 dari 12 menilai demikian.

*Sumber: `evaluasi_ahli/statistik_n12.json`.*

## Penyiapan lingkungan

Dikembangkan pada environment conda `diabetes-ta` dengan Python 3.11. Versi paket dikunci,
terutama scikit-learn, agar artefak model yang tersimpan dapat dimuat kembali.

```bash
conda create -n diabetes-ta python=3.11
conda activate diabetes-ta
pip install -r requirements.txt
cp .env.example .env    # lalu isi GOOGLE_API_KEY
```

## Reproduksi

Jalankan dari akar proyek dengan environment aktif dan `PYTHONPATH` menunjuk akar proyek.

```bash
python -m src.data.ohio_parser
python -m src.models.rf_model --config config.yaml --data_source ohio_t1dm
python scripts/verify_prediction_artifacts.py
python scripts/reingest_kb.py
```

Menjalankan sistem, backend dan frontend pada dua terminal:

```bash
uvicorn backend.main:app --reload
```

```bash
cd frontend && npm install && npm run dev
```

Evaluasi yang telah diverifikasi menghasilkan ulang angka laporan secara identik dari kode
pada `main` (dijalankan ulang 12 September 2026 di salinan terisolasi, 1.184 angka, nol selisih):

```bash
python scripts/analisis_hasil_form_csv.py
python scripts/crossval_smbg.py
python scripts/conformal_calibration.py --horizon 6 --model gbm
python scripts/conformal_calibration.py --horizon 12 --model gbm
python scripts/eval_cakupan_conformal.py
```

Berkas hasil lainnya di `results/__Hasil_Akhir__/` menyimpan konfigurasi efektif dan sidik
jari eksperimennya, tetapi skrip penghasilnya bergantung pada antarmuka `src/` dan kunci
`config.yaml` yang berubah pada commit `8d88802` (23 Agustus 2026), sesudah hasil itu
dibuat. Menjalankannya ulang memerlukan keadaan kode sebelum commit tersebut; pemulihan
kompatibilitasnya direncanakan pada revisi.

Memeriksa bahwa hasil akhir masih sama dengan berkas sumbernya:

```bash
python scripts/kumpulkan_hasil_akhir.py --periksa
```

## Pengujian

```bash
pytest tests/ -q
```

## Struktur proyek

```
backend/              Layanan FastAPI: rute prediksi, logbook, pasien, klinis
frontend/             Progressive Web Application berbasis Next.js dan React
src/data/             Parser OhioT1DM, praproses, rekayasa fitur, kontrak data
src/models/           Model prediksi dan kalibrasi konformal
src/rag/              Pipeline RAG, kueri terkondisi-prediksi, retriever, sitasi
src/utils/            Metrik evaluasi dan logging
scripts/              Skrip evaluasi yang mereproduksi angka laporan
tests/                Uji unit dan integrasi
notebooks/            Eksplorasi data
data/knowledge_base/  Manifes korpus pedoman klinis
models/               Bundel inferensi dan indeks ChromaDB
results/              Keluaran evaluasi, termasuk __Hasil_Akhir__
docs/laporan_TA/      Sumber LaTeX laporan tugas akhir
config.yaml           Konfigurasi terpusat
```

## Keterbatasan

- Evaluasi bersifat teknis dan berbasis kuesioner ahli; belum ada uji klinis maupun validasi
  lapangan pada pasien nyata.
- OhioT1DM berisi dua belas pasien diabetes melitus tipe 1 pengguna pompa insulin di Amerika
  Serikat. Generalisasi ke populasi Indonesia belum terbukti.
- Sensitivitas deteksi hipoglikemia 67,2% dengan PPV 31,9% belum memadai untuk penggunaan
  klinis mandiri.
- Manfaat pengondisian terkonsentrasi pada kasus divergen; pada distribusi natural
  peningkatannya kecil dan tidak signifikan.
- Relevansi dokumen pada evaluasi *retrieval* ditetapkan melalui klasifikasi kata kunci
  berbobot, bukan penilaian pakar klinis.
- Model *embedding* belum disetel ulang untuk korpus diabetes berbahasa Indonesia.
- Evaluasi generasi memakai enam kasus, dan penilai otomatis `answer_relevancy` terbukti
  tidak stabil antar-run sehingga tidak dijadikan dasar klaim.

## Lisensi

Penggunaan akademik dalam rangka Tugas Akhir, Institut Teknologi Bandung. Dataset OhioT1DM
tunduk pada *Data Use Agreement* tersendiri dan tidak didistribusikan melalui repositori ini.
Dokumen pedoman klinis penyusun basis pengetahuan tetap menjadi hak penerbitnya
masing-masing.

## Sitasi

```
Adiyatma, D. (2026). RAG Terkondisi-Prediksi untuk Dukungan Keputusan Klinis
Diabetes Melitus Tipe 1. Tugas Akhir, Program Studi Sistem dan Teknologi Informasi,
Institut Teknologi Bandung.
```

## Penghargaan

- Marling dan Bunescu, Ohio University, atas dataset OhioT1DM.
- PERKENI, IDAI, ADA bersama EASD, ISPAD, dan konsensus ATTD atas pedoman klinis yang
  menjadi basis pengetahuan sistem.
- Dua belas tenaga medis yang bersedia menjadi responden evaluasi.

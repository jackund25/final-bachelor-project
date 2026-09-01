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
| Keadaan klinis | Representasi terstruktur dan jejak peninjauan dokter (`src/clinical_state/`) |
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

Validasi silang enam lipatan **lintas-pasien**: pasien uji tidak pernah dilihat model saat
pelatihan. Nilai disajikan sebagai rerata antar-lipatan beserta simpangan bakunya.

| Horizon | Model | RMSE (mg/dL) | MAE (mg/dL) | Clarke A+B (%) |
| --- | --- | --- | --- | --- |
| +30 menit | **GBM (produksi)** | **20,72 ± 1,16** | 14,48 ± 0,79 | 94,69 ± 0,50 |
| +30 menit | Random Forest | 21,08 ± 1,15 | 14,78 ± 0,77 | 94,35 ± 0,55 |
| +30 menit | LSTM | 20,60 ± 1,33 | 14,32 ± 0,95 | 94,89 ± 0,52 |
| +60 menit | **GBM (produksi)** | **33,81 ± 1,44** | 24,85 ± 1,12 | 85,83 ± 1,29 |
| +60 menit | Random Forest | 34,78 ± 1,50 | 25,55 ± 1,17 | 85,34 ± 1,25 |
| +60 menit | LSTM | 34,50 ± 1,45 | 25,37 ± 1,19 | 85,50 ± 1,42 |

*Sumber: `T1_prediksi/gbm_produksi_h30m.json` dan `T1_prediksi/gbm_produksi_h60m.json`.*

Keunggulan GBM atas Random Forest **konsisten tetapi tidak bermakna secara klinis**: selisih
RMSE 0,358 mg/dL pada +30 menit dan 0,976 mg/dL pada +60 menit, keduanya konsisten pada uji
Wilcoxon tingkat lipatan (p = 0,031) namun hanya 2,39% dan 6,51% dari ambang orde besaran
15 mg/dL. GBM dipilih sebagai model produksi karena alasan **keterterapan**, bukan ketepatan:
waktu latih 3,4 detik berbanding 708 detik, ukuran artefak 0,485 MB berbanding 322 MB.

### Ketidakpastian dan deteksi hipoglikemia

Interval konformal nominal 95% mencapai cakupan empiris **94,7% ± 2,1** pada dua belas
putaran dengan pasien pengukur yang tidak pernah dipakai melatih maupun mengkalibrasi
(rentang 90,8% sampai 97,4%). Enam dari dua belas putaran berada di bawah nominal, dan
cakupan berkorelasi kuat dengan pasien pengkalibrasi (r = 0,705), sehingga yang dilaporkan
adalah cakupan **prosedur** dan bukan cakupan satu bundel tertentu.

Regresi yang diambang menghasilkan sensitivitas hipoglikemia 17,3% dengan PPV 56,6%.
Pengklasifikasi kondisi tiga kelas menaikkan sensitivitas menjadi **67,2%** dengan konsekuensi
PPV turun menjadi 31,9% dan akurasi keseluruhan turun dari 89,4% menjadi 86,1%. Pertukaran ini
dilaporkan apa adanya; sensitivitas tersebut belum memadai untuk penggunaan klinis mandiri.

*Sumber: `T1_prediksi/cakupan_konformal.json`, `T1_prediksi/pengklasifikasi_kondisi.json`.*

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

Pada enam kasus uji, seluruh **56 angka klinis** yang muncul pada rekomendasi dapat ditelusuri
ke dokumen yang di-*retrieve*, ke keadaan pasien, atau ke ambang klinis baku; tidak satu pun
angka tak tertelusur, tidak ada tindakan salah arah, dan seluruh keluaran menyertakan
pernyataan batas keputusan.

Pengujian kestabilan penilai otomatis bersifat diagnostik: selisih absolut median
`answer_relevancy` antar-dua jalan sebesar 0,056 tanpa pembalikan penuh, sedangkan
`faithfulness` bermedian 0,18 dan karena itu tidak dipakai sebagai dasar klaim.

*Sumber: `T3_generation/generation_safety.json`, `T3_generation/kestabilan_ragas.json`.*

### Keterterapan operasional

Diukur seluruhnya pada CPU tanpa akselerator grafis. Waktu tanggap ujung-ke-ujung bermedian
**2,474 detik**, dengan komputasi lokal **0,339 detik** dan sisanya menunggu layanan LLM
(85,0% dari total). Pemuatan artefak 1,600 detik hanya terjadi sekali saat layanan dijalankan.

*Sumber: `operasional/latensi_ujung_ke_ujung.json`, `operasional/keterterapan.json`.*

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

Evaluasi yang menghasilkan angka pada bagian Hasil:

```bash
python scripts/eval_gradient_boosting.py --config config.yaml
python scripts/crossval_rf_vs_lstm.py --config config.yaml
python scripts/conformal_calibration.py
python scripts/train_condition_classifier.py
python scripts/eval_retrieval_realcases.py --config config.yaml
python scripts/ablation_rag_fullkb.py --config config.yaml
python scripts/eval_smbg_deployment.py
```

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
src/clinical_state/   Keadaan klinis terstruktur dan jejak peninjauan dokter
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
- Cakupan konformal yang terwujud bergantung pada pasien pengkalibrasi, sehingga yang dijamin
  adalah sifat prosedur dan bukan sifat satu bundel model.
- Relevansi dokumen pada evaluasi *retrieval* ditetapkan melalui klasifikasi kata kunci
  berbobot, bukan penilaian pakar klinis.
- Model *embedding* belum disetel ulang untuk korpus diabetes berbahasa Indonesia.
- Evaluasi generasi memakai enam kasus, dan penilai otomatis `faithfulness` terbukti tidak
  stabil sehingga tidak dijadikan dasar klaim.

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

# HANDOFF — status pengerjaan TA

> **Ringkasan status untuk memulai sesi baru. Bukan pengganti rekam jejak.**
> Rekaman lengkap ada di `docs/journey/` (delapan part, lihat `README.md` di sana).
> Berkas ini hanya memuat apa yang perlu diketahui untuk melanjutkan, tanpa perlu
> menelusuri 5.500 baris journey.
>
> **Diperbarui:** 11 Agustus 2026 · **Commit:** `18b9609` · **Cabang:** `refaktor-tujuh-tugas`

---

## 1. Status percobaan

| Kode      | Percobaan                                   | Status                                               | Commit               |
| --------- | ------------------------------------------- | ---------------------------------------------------- | -------------------- |
| **T1.1**  | RF vs LSTM pada jendela tersegmentasi       | ✅ selesai                                           | `da6ab2e`, `2522fa9` |
| **T1.1b** | Perkecil berkas Random Forest               | ✅ selesai                                           | `5136e62`            |
| **T1.2**  | RAGAS 4 metrik + 2 kasus negatif            | ✅ selesai                                           | `c5a170e`, `d76a234` |
| **T1.3**  | Cakupan konformal per rentang glukosa       | ✅ selesai                                           | `b29eef7`            |
| **T1.4**  | Sensitivitas `tau` IOB/COB                  | ✅ selesai — **TIDAK PEKA**                          | `18b9609`            |
| **T2.1**  | Galat khusus hipoglikemia, RF vs LSTM       | ✅ selesai                                           | `cd41fa3`            |
| **T2.2**  | Verifikasi manusia atas pelabel relevansi   | ✅ selesai — **κ 0,2505 LEMAH**                      | `35d5866`, sesi ini  |
| **T3.1**  | Susunan kalimat kueri                       | ✅ selesai                                           | `11c8ac1`            |
| **T3.2**  | Horizon sumber pengondisian retrieval       | ✅ selesai — **tidak berpengaruh**                   | `55bf85a`            |
| **T3.3**  | Jangkauan varian kueri + bentuk aplikasi    | ✅ selesai — **D2 terbantah**                        | `3373dd5`            |
| **T4.1**  | Gradient Boosting, pembanding ketiga (h6+h12)| ✅ selesai — **GBM > RF di tiap dimensi**            | `9141f3a`            |
| **T4.2**  | Logbook manual → jalur prediksi             | ✅ selesai                                           | `5915b2d`            |
| **T5.1**  | Distribusi token per chunk                  | ✅ selesai                                           | `6024159`            |
| **T5.2**  | Ringkasan angka Bab VI                      | 🔄 jalan pertama selesai, **perlu diulang di akhir** | —                    |
| **J7**    | Kontribusi fitur (MDI + permutasi)          | ✅ selesai                                           | `dc51c2f`, `09dc35e` |
| —         | **TEMUAN: jangkauan penelusuran**           | ✅ selesai                                           | `e0b3cca`            |

**Seluruh percobaan Bab VI SELESAI.** Yang tersisa bukan percobaan:

1. **T2.2 SUDAH DIISI, dan hasilnya mengikat.** Cohen's κ = **0,2505 (lemah)**; kesepakatan
   mentah 52,5%, di bawah tebakan konstan kelas mayoritas (77,5%). Aturan κ < 0,40 kini
   **TERPICU**: seluruh angka penelusuran wajib disertai kualifikasi eksplisit **di setiap
   penyebutannya**. Rincian di K1.
2. **Delapan keputusan menggantung** menunggu pembimbing — lihat §7 dan
   `docs/RINGKASAN_KEPUTUSAN_PEMBIMBING.md`.
3. **Revisi naskah** yang sudah ditulis: Bab II subbab II.4.1 membenarkan pemilihan RF atas
   keterjelasan kontribusi fitur dan efisiensi komputasi, dan J7 + T4.1 melemahkan keduanya.

**Temuan baru setelah titik ini masuk daftar keterbatasan atau saran pengembangan, bukan
pekerjaan baru.**

---

## 2. Konfigurasi produksi

### Model prediksi

| Parameter                          | Nilai                                                            |
| ---------------------------------- | ---------------------------------------------------------------- |
| `model.name`                       | RandomForest                                                     |
| `sequence_length`                  | 12 (look-back 1 jam)                                             |
| `prediction_horizons`              | `[6, 12]` (+30 dan +60 menit)                                    |
| `default_horizon`                  | 6                                                                |
| `use_engineered` / `predict_delta` | `true` / `true`                                                  |
| `engineered_features`              | `glucose, glucose_delta, iob, cob, activity, hour_sin, hour_cos` |
| `feature_engineering`              | `insulin_tau_min` 240 · `carbs_tau_min` 180 · `trend_steps` 3    |
| `random_forest`                    | `n_estimators` 200 · `max_depth` 20 · `min_samples_split` 5      |
| **`max_gap_steps`**                | **6** (30 menit — segmentasi jeda sensor, Tugas 5)               |
| `sampling_interval_min` / `seed`   | 5 / 42                                                           |

### RAG

| Parameter                      | Nilai                                                      |
| ------------------------------ | ---------------------------------------------------------- |
| `chunk_size` / `chunk_overlap` | **900** / 120                                              |
| `top_k_retrieval`              | **5**                                                      |
| `fetch_k`                      | **12**                                                     |
| `lambda_mult`                  | **0,0** (diadopsi pada Tahap 0 dari B5)                    |
| `embedding_model`              | `all-MiniLM-L6-v2` — **`max_seq_length` 256 token**        |
| `collection_name`              | `diabetes_kb` — **2.061 chunk terindeks**                  |
| LLM                            | `gemini-3.5-flash-lite` · temperature 0,2 · max_tokens 700 |

### Pembagian pasien — dipakai konsisten di seluruh percobaan

`pids = sorted(unique(patient_id))` dari 12 pasien OhioT1DM:

- **latih** `pids[:-4]` — 8 pasien
- **penyetelan** `pids[-4:-2]` — `ohio_584`, `ohio_588`
- **pelaporan** `pids[-2:]` — `ohio_591`, `ohio_596`

Fold lintas-pasien memakai `pids[i::6]`, 6 fold, aturan identik di
`crossval_rf_vs_lstm.py`, `eval_hipoglikemia.py`, dan `eval_gradient_boosting.py`.

### Kuota Gemini — **per model**, bukan per akun

| Model                              | RPM | RPD    |
| ---------------------------------- | --- | ------ |
| `gemini-3.5-flash-lite` (produksi) | 15  | 500    |
| `gemini-3.1-flash-lite`            | 15  | 500    |
| `gemini-2.5-flash-lite`            | 10  | **20** |
| `gemini-2.5-flash`                 | 5   | **20** |
| `gemini-3-flash`                   | 5   | **20** |

Rentang 25× antar-model. Model tak dikenal diperlakukan konservatif (5 / 20).

---

## 3. Hasil final percobaan yang selesai

### T1.1 — RF vs LSTM (6 fold, jendela tersegmentasi)

|                | RF            | LSTM          |
| -------------- | ------------- | ------------- |
| RMSE +30 mnt   | 21,08 ± 1,15  | 20,60 ± 1,33  |
| RMSE +60 mnt   | 34,78 ± 1,50  | 34,50 ± 1,45  |
| Clarke A+B +30 | 94,35 ± 0,55% | 94,89 ± 0,52% |
| waktu latih    | 705 dtk       | 477 dtk       |
| ukuran model   | 322 MB        | **0,124 MB**  |

+30 mnt signifikan di tingkat fold (Wilcoxon p=0,031); +60 mnt tidak (p=0,313).
**Keputusan:** RF dipertahankan, dibenarkan atas dasar **keterjelasan kontribusi fitur**,
bukan efisiensi. Pemilihan model belum final.

### T1.1b — perkecil RF

Sapuan 11 konfigurasi. `max_depth` 20 tercapai 200/200 pohon → kedalaman mengikat.

Set pelaporan (n=26.445): acuan **200/20 → RMSE 21,251 · 297,15 MB**; terpilih
**50/12 → RMSE 21,332 · 11,09 MB**. Selisih **+0,081 mg/dL untuk −96,3% ukuran**;
Clarke A+B justru naik 94,90% → 94,93%.

**Dua kekeliruan pra-registrasi dicatat:** (a) uji BEDA dipakai sebagai uji KESETARAAN —
gagal menolak H₀ bukan bukti H₀ benar; instrumen yang benar TOST dengan margin yang
ditetapkan di muka, dan margin itu tidak pernah ditetapkan. (b) uji dua sisi mencoret
`200/15` dan `100/15` justru karena RMSE-nya lebih baik.

**Karena itu T1.1b TIDAK membuktikan 50/12 setara dengan 200/20.** Yang berdiri sendiri
tanpa uji hipotesis: pertukaran 0,081 mg/dL untuk 96,3% ukuran.

### T1.2 — RAGAS (n=10, `gemini-3.5-flash-lite`)

| Metrik            | Nilai     | Optimistis?                |
| ----------------- | --------- | -------------------------- |
| faithfulness      | **0,678** | tidak                      |
| answer_relevancy  | **0,763** | tidak                      |
| context_precision | 0,450     | **ya**                     |
| context_recall    | 0,450     | tidak — peringatan dicabut |

Kasus negatif Tahap D: **2/2 lulus**. `context_precision` bernilai **biner** — 0,00 pada
lima kasus, 1,00 pada lima lainnya.

### T1.3 — cakupan konformal per rentang (selang Wilson 95%)

| rentang                 | porsi | +30 mnt                 | +60 mnt                 |
| ----------------------- | ----- | ----------------------- | ----------------------- |
| hipoglikemia berat      | 0,55% | 96,67 [92,4–98,6]       | 94,93 [89,9–97,5]       |
| **hipoglikemia**        | 2,34% | 96,12 [94,3–97,4]       | **90,80 [88,2–92,8]** ⚠ |
| normal                  | 69,2% | 95,18 [94,9–95,5]       | **97,26 [97,0–97,5]**   |
| hiperglikemia           | 23,0% | 96,11 [95,6–96,6]       | 95,76 [95,2–96,2]       |
| **hiperglikemia berat** | 4,95% | **92,97 [91,5–94,2]** ⚠ | **91,76 [90,1–93,1]** ⚠ |
| agregat                 |       | 95,31 [95,1–95,6]       | 96,48 [96,3–96,7]       |
| lebar interval          |       | 99,9 mg/dL              | 154,2 mg/dL             |

⚠ = seluruh selang di bawah nominal 95%.

**Hipotesis (hipo < agregat): SALAH ARAH di +30 mnt, TERBUKTI di +60 mnt.** Pada +30 mnt
kegagalan subkelompok satu-satunya adalah hiperglikemia berat, bukan hipoglikemia.
Pada +60 mnt mekanismenya terlihat: mayoritas 69% kelebihan cakupan menutupi kedua ekor
yang kekurangan.

### T2.1 — galat khusus hipoglikemia (6 fold, RF vs LSTM)

Basis kejadian: **5.206 dari 160.081 jendela (3,25%)** di +30 mnt; 5.138 dari 157.815
(3,26%) di +60 mnt.

| metrik                     | RF +30            | LSTM +30          | RF +60            | LSTM +60          |
| -------------------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| **sensitivitas**           | **21,68 ± 4,72%** | **32,11 ± 6,21%** | **4,36 ± 1,97%**  | **10,56 ± 6,75%** |
| spesifisitas               | 99,70 ± 0,09%     | 99,51 ± 0,07%     | 99,90 ± 0,04%     | 99,71 ± 0,03%     |
| PPV                        | 69,03 ± 11,22%    | 66,34 ± 11,37%    | 57,96 ± 17,53%    | 49,04 ± 17,59%    |
| F1                         | 32,58 ± 5,38%     | 43,09 ± 7,47%     | 7,98 ± 3,42%      | 17,12 ± 10,14%    |
| RMSE pada hipo             | 26,89 ± 2,53      | 24,12 ± 3,72      | 49,71 ± 4,67      | 45,94 ± 7,80      |
| MAE pada hipo              | 21,23 ± 1,11      | 18,28 ± 2,17      | 42,68 ± 2,43      | 38,12 ± 6,17      |
| **bias pada hipo**         | **+21,03 ± 1,07** | +17,81 ± 2,09     | **+42,66 ± 2,42** | +38,03 ± 6,26     |
| bias keseluruhan           | +0,21 ± 1,18      | −0,04 ± 1,61      | +0,21 ± 2,62      | −0,04 ± 2,40      |
| terlewat ke rentang target | 4.095             | 3.512             | 4.921             | 4.558             |
| **hipo BERAT terlewat**    | **588**           | 410               | **933**           | 743               |

**Bias adalah temuan utamanya.** Bias keseluruhan +0,21 mg/dL, tetapi pada kejadian
hipoglikemia **+21,03** di +30 mnt dan **+42,66** di +60 mnt — arah positif berarti model
menduga LEBIH TINGGI daripada kenyataan, yaitu arah yang **menyembunyikan** kejadian
rendah. Ini penyusutan ke tengah pada kelas langka, bukan bias global.

Uji berpasangan tingkat fold, dengan penjaga "selisih harus melampaui SD antar-fold":

|                | +30 mnt                                          | +60 mnt                                          |
| -------------- | ------------------------------------------------ | ------------------------------------------------ |
| sensitivitas   | −10,43 (SD 7,59) p=0,031 → **layak dinarasikan** | −6,20 (SD 5,86) p=0,031 → **layak**              |
| MAE pada hipo  | +2,95 (SD 2,27) p=0,031 → layak                  | +4,56 (SD 5,21) p=0,063 → **jangan dinarasikan** |
| bias pada hipo | +3,22 (SD 2,31) p=0,031 → layak                  | +4,63 (SD 5,28) p=0,063 → **jangan dinarasikan** |

**LSTM unggul pada hipoglikemia dan itu sah dinarasikan pada kedua horizon
(sensitivitas).** Ini bersinggungan langsung dengan keputusan mempertahankan RF.

Menggantikan `hypo_safety_uncertainty.json` yang dihitung sebelum Tugas 5, hanya untuk RF,
dan hanya satu horizon.

### T4.2 — logbook manual

Tersambung dengan penjaga cadence: jendela yang jaraknya melampaui `max_gap_steps`
**ditolak, bukan diinterpolasi**. Kotak centang mati secara bawaan. 20 tes baru.

### T5.1 — distribusi token

`all-MiniLM-L6-v2` memotong pada **256 token** tanpa peringatan (bukan 512 — itu
`model_max_length` BERT-nya).

| chunk              | % chunk >256 | **% token hilang** | MRR (B4)  |
| ------------------ | ------------ | ------------------ | --------- |
| 300                | 0,00%        | 0,00%              | 0,306     |
| 500                | 0,11%        | 0,01%              | **0,541** |
| **900 (produksi)** | **35,04%**   | **8,23%**          | 0,425     |
| 1400               | 73,27%       | 29,66%             | 0,313     |
| 2000               | 81,21%       | 47,95%             | 0,302     |

Paling terdampak: KB-04 67,19%, **KB-03 48,84%** (dokumen sumber evaluasi RAGAS), KB-01
44,87%. Spearman(% hilang, MRR) = −0,400 **p=0,505 tidak signifikan**; hubungannya tidak
monoton karena chunk 300 nol pemotongan tetapi MRR 0,306.

### J7 — kontribusi fitur

| ukuran                   | mentah    | engineered | kelipatan  |
| ------------------------ | --------- | ---------- | ---------- |
| MDI                      | 2,81%     | 21,67%     | 7,71×      |
| **permutasi (hold-out)** | **0,53%** | **15,92%** | **30,04×** |

**Angka untuk laporan 15,92%, bukan 22,3%.** MDI melebih-lebihkan karena bias
kardinalitas: `iob` punya 165.950 nilai unik lawan `glucose` 361 — ~460× lebih banyak titik
pisah. `activity` menyumbang **0,18%** (engineered) dan 0,01% (mentah) — klaim aktivitas
sebagai modalitas **tidak didukung data**. MDI melebihkan fitur diurnal 24×.

### TEMUAN — jangkauan penelusuran

Sapuan glukosa 40–400 mg/dL langkah 5 (73 nilai, setiap angka yang dapat dihasilkan
prediktor):

|                           | potongan  | % korpus   |
| ------------------------- | --------- | ---------- |
| bentuk produksi           | **24**    | **1,16%**  |
| tujuh bentuk kueri        | 69        | 3,35%      |
| **tidak pernah terambil** | **2.037** | **98,84%** |

10 potongan teratas menyerap **75,07%** pengambilan. **Bukan `lambda_mult`**: 0,0 → 24
unik, 0,5 → 22 unik.

Mekanisme: `build_ablation_query` meruntuhkan seluruh rentang glukosa menjadi **3 frasa
kondisi** (frasa saja 14 potongan; ditambah angka 24 — angka menambah 10), lalu
`fetch_k`=12 mengunci kolam kandidat di **31** potongan (batas teoretis 3 × 12 = 36).
Tuasnya **keragaman kueri** dan **`fetch_k`**, bukan `lambda_mult` dan bukan `top_k`.

Menyatukan tiga temuan yang tadinya terpisah: pola biner `context_precision`, jurang
Hit@1 23,3% lawan Hit@5 91,7%, dan konteks tak relevan pada E01.

### T3.1 — susunan kueri

Tujuh varian, Bagian C penuh, kriteria MRR ditetapkan di muka, 90 kasus per himpunan.

| varian          | MRR        | Hit@1      | p vs produksi | **bobot kata kunci pelabel** |
| --------------- | ---------- | ---------- | ------------- | ---------------------------- |
| `kata_kunci`    | 0,7722     | 0,7111     | 9e-06         | **22**                       |
| `hanya_kondisi` | **0,7111** | **0,7111** | 0,00058       | 17                           |
| `kondisi_dulu`  | 0,6231     | 0,4333     | 0,73          | 17                           |
| **produksi**    | 0,6211     | 0,4222     | acuan         | 17                           |
| `ringkas`       | 0,5774     | 0,3778     | 0,24          | 10                           |
| `pertanyaan`    | 0,4744     | 0,1556     | 2,3e-05       | 6                            |
| `hanya_angka`   | 0,2276     | **0,0000** | ~0            | 0                            |

**Pemenang `kata_kunci` TERKONTAMINASI dan tidak boleh diadopsi.**
Spearman(bobot kata kunci `classify_chunk`, MRR) = **0,964, p=0,0005** — peringkat
antar-varian nyaris seluruhnya diramalkan oleh tumpang tindih kosakata dengan pelabelnya
sendiri. Peringkat antar-varian **tidak boleh dilaporkan** sampai T2.2 selesai.

**Perbandingan yang KEBAL:** produksi, `hanya_kondisi`, dan `kondisi_dulu` punya bobot kata
kunci **identik (17)**. Membuang angka glukosa dari teks kueri menaikkan **Hit@1 dari 42,2%
ke 71,1%** (+68% relatif, p=0,00058). Pendukung: `hanya_angka` → Hit@1 **0,0000**.

**Tidak membatalkan PC-RAG:** `hanya_kondisi` tetap prediction-conditioned — frasa dipilih
`classify_glucose(prediksi)`. Yang dibuang hanya numeral di teks kueri.

**Prapendaftaran `kondisi_dulu` TERKONFIRMASI:** MRR +0,0020, Hit@1 +0,0111 = 1 kasus dari
90, p=0,73. Posisi klausa tidak berpengaruh; yang berpengaruh isi.

Tambahkan ke keputusan menggantung tentang adopsi hanya_kondisi:

Sebelum diputuskan, ukur jangkauan varian hanya_kondisi, yaitu jumlah potongan
unik dan porsi ke sepuluh teratas, lalu bandingkan dengan bentuk produksi.

Alasannya, temuan konsentrasi menunjukkan tiga frasa kondisi saja menjangkau
14 potongan sedangkan penambahan angka menaikkannya ke 24. Varian
hanya_kondisi memperbaiki peringkat tetapi kemungkinan memperburuk jangkauan,
dan kedua hal itu belum pernah diukur bersama.

Periksa pula apakah ada bagian sistem lain yang bergantung pada nilai numerik
di dalam teks kueri, karena dengan hanya_kondisi glukosa 55 dan 68 akan
menghasilkan kueri identik.

### Retrieval crossfold (6 fold, `top_k` 5) — **KONFIGURASI PRODUKSI, lambda 0,0**

Sumber: `results/retrieval_realcases_kb12_final/crossfold.json`

| mode              | MRR divergen      | MRR natural   |
| ----------------- | ----------------- | ------------- |
| standard          | 0,382 ± 0,009     | 0,647 ± 0,068 |
| pc_rag            | **0,433 ± 0,023** | 0,648 ± 0,070 |
| pc_rag_classifier | **0,464 ± 0,036** | —             |
| oracle            | **0,829 ± 0,005** | 0,686 ± 0,064 |

**Temuan utamanya bukan angka absolutnya, melainkan pemisahannya:** PC-RAG unggul +0,051
pada kasus **divergen** dan +0,001 pada kasus **natural** — yang kedua jauh di bawah SD
antar-fold 0,068. Pengondisian prediksi memberi perbaikan nyata pada kasus divergen dan
tidak memberi apa-apa pada kasus non-divergen, dan 13,2% jendela nyata termasuk divergen.

> **KOREKSI 11 Agustus 2026.** Tabel ini semula memuat angka dari korpus **`_kb12_sym`**
> (standard 0,335 · pc_rag 0,383 · pc_rag_classifier 0,424 · oracle 0,794). Berkas
> bersufiks `_kb12_sym` adalah **jejak konfigurasi lambda 0,5 yang TIDAK diadopsi** (lihat
> journey part-5). Sesi baru yang membaca HANDOFF saja akan mengutip angka lambda 0,5
> sebagai hasil produksi Bab VI.
>
> **Berkas `_kb12_sym` TETAP DISIMPAN** sebagai pembanding lambda 0,5 dan **jangan
> dihapus** — tabel perbandingan dua lambda di journey part-5 bersandar padanya.
>
> Cacat yang menyertainya: `buat_ringkasan_bab6.py` menunjuk direktori **`_kb12`** (5
> Agustus, era pra-A1), dan meminta medan `rerata_lintas_fold` yang tidak ada — nama yang
> benar `ringkasan_lintas_fold`. Keduanya sudah diperbaiki.
>
> **Cacat provenans yang tersisa:** `crossfold.json` hanya merekam `top_k`, tidak
> `lambda_mult` maupun `chunk_size`. Bukti bahwa berkas itu dari lambda 0,0 hanyalah **nama
> direktorinya**. Memperbaikinya menuntut skrip crossfold menuliskan konfigurasi efektifnya
> lalu dijalankan ulang 2 jam 3 menit — masuk daftar pekerjaan menunggu.

---

## 4. Keterbatasan K1–K14

Rincian di `docs/DAFTAR_KETERBATASAN.md`.

| Kode    | Keterbatasan                                                        | Dampak                                                                                   |
| ------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **K1**  | Relevansi dari pelabel kata kunci `classify_chunk()`, bukan manusia | **Tinggi** — batas atas seluruh metrik penelusuran                                       |
| K2      | Korpus evaluasi RAGAS sempit                                        | Sedang — `context_precision` optimistis                                                  |
| **K3**  | Penanda `grounded` tidak mendeteksi konteks tak relevan             | **Tinggi** — pengaman KNF-03 bertumpu pada perilaku model, bukan mekanisme deterministik |
| K4      | Ekstraksi PDF merusak label dekorasi gambar                         | Rendah — tabel dosis selamat                                                             |
| **K5**  | Tidak ada validasi klinis                                           | **Tinggi** — batas klaim kelayakan                                                       |
| K6      | Disclaimer dibuang sebelum penilaian RAGAS                          | Rendah — perlu dinyatakan, bukan diperbaiki                                              |
| K7      | Kuota LLM membatasi rancangan evaluasi                              | Sedang — memaksa korpus evaluasi kecil                                                   |
| K8      | Positif palsu pada pemeriksa angka kasus negatif                    | Rendah                                                                                   |
| K9      | `top_k` produksi sempat berbeda dari evaluasi                       | Rendah — sudah diperbaiki                                                                |
| **K10** | `faithfulness` bukan ukuran mutu jawaban, labil per kasus           | **Tinggi** — hanya reratanya boleh dikutip                                               |
| K11     | Logbook tersambung tetapi catatan sering ditolak                    | Sedang — KF-01/KF-02 tidak ADA penuh                                                     |
| **K12** | Jangkauan penelusuran hanya 1,16% korpus                            | **Tinggi** — satu sebab bagi K2, jurang Hit@1/Hit@5, dan E01                             |
| K13     | Angka penelusuran diukur pada bentuk kueri yang tidak dipakai aplikasi | Sedang — terukur pada T3.3; bentuk aplikasi **tidak lebih buruk** |
| **K14** | Aturan pelaporan efek mencampur konsistensi dengan kebermaknaan | **Tinggi** — empat klaim keunggulan RAPUH; lima skala derau beredar |

**Lima paling menentukan batas klaim: K1, K3, K5, K10, K14.** K12 tetap tinggi dampaknya.
K14 satu-satunya yang dapat diselesaikan **tanpa data baru**, lewat keputusan #8.

Kestabilan metrik RAGAS: hanya `faithfulness` yang punya data run berulang (labil per
kasus, rerata stabil). `answer_relevancy`, `context_precision`, dan `context_recall`
**belum diuji dan tidak boleh diasumsikan stabil**.

---

## 5. Prapendaftaran yang berlaku

### `docs/PRAPENDAFTARAN_T3.1.md` — commit `4292960`

Dibuat **sebelum** T3.1 dijalankan; diverifikasi `results/eval_rag/susunan_kueri.json`
belum ada saat itu.

**Isi:** varian `kondisi_dulu` (isi identik dengan produksi, urutan klausa dibalik)
menyumbang **nol** potongan unik baru pada 180 pengambilan T2.2. Diperkirakan MRR, Hit@1,
dan nDCG@5-nya identik dengan produksi atau berbeda hanya sebesar galat pembulatan.
Alasannya mekanistik: `all-MiniLM-L6-v2` memakai _mean pooling_, dan rata-rata bersifat
komutatif.

**Aturan penafsiran yang mengikat:**

- Bila T3.1 melaporkan selisih berarti untuk `kondisi_dulu`, **yang dicurigai lebih dulu
  adalah pengukurannya, bukan temuannya.** Tiga pemeriksaan wajib sebelum angka itu
  dilaporkan: (a) apakah teks kedua varian benar-benar sama isinya, (b) apakah kasus,
  benih, dan pembagian pasien identik, (c) apakah selisihnya melampaui variasi antar-jalan
  retriever.
- Bila selisihnya nol, **itu bukan hasil kosong**: ia mengukur bahwa posisi klausa tidak
  berpengaruh pada model embedding ini, sehingga menyusun ulang urutan kalimat bukan tuas
  yang layak dikejar.

---

## 6. Aturan kerja yang mengikat

### Git — lokal saja

**Jangan pernah** `git push`, `git remote add`, atau perintah lain yang mengirim ke
repositori jarak jauh. Remote `origin` ada
(`https://github.com/jackund25/final-bachelor-project.git`) dan tetap **tidak boleh**
didorongi. Bila sesuatu tampak perlu didorong, **tanyakan dulu**.

Commit tetap dikerjakan seperti biasa, **satu percobaan satu commit**, karena riwayatnya
dipakai menyusun timeline di laporan.

Diabaikan git: `docs/journey/`, `docs/METHODOLOGY.md`, `docs/doc.md`, `docs/proposal_ta/`,
PDF korpus, berkas model `.pkl`.

### Protokol Bagian C — pemisahan set penyetelan dan pelaporan

Mengikat untuk **setiap** penyetelan parameter:

1. Set penyetelan dan set pelaporan **dipisah menurut pasien**, dibagi sebelum satu pun
   hasil dilihat, dan tidak pernah bertukar.
2. Kriteria **ditetapkan di muka**. Untuk penyetelan retrieval, kriterianya **MRR** —
   diimpor dari `scripts/tuning_protocol.py`, bukan ditulis ulang sebagai konstanta lokal.
   Percobaan sisi prediksi boleh memakai kriteria lain, tetapi wajib menimpanya lewat
   argumen `kriteria=` agar tercatat jujur.
3. **Seluruh** konfigurasi dicatat ke `results/tuning_log.json`, termasuk yang kalah.
4. Set evaluasi **tidak boleh diubah** setelah hasil terlihat.
5. Set pelaporan disentuh **sekali**, di akhir.
6. Jumlah konfigurasi yang diuji dilaporkan.

Bila skenario terlalu sedikit untuk dibagi dua, **laporkan bahwa penyetelan tidak
dilakukan** — lebih baik daripada melaporkan angka yang tidak sah.

### Pengukuran proyek mengalahkan dokumentasi vendor

Kuota Gemini yang benar (20/hari untuk sebagian model) sudah tercatat dari pengalaman nyata
8 Juli, lalu **saya timpa** dengan 15 RPM / 1.000 RPD hasil membaca dokumentasi, dan
**mengubah rencana** berdasarkan angka palsu itu. **Angka terukur dari proyek ini selalu
mengalahkan angka dari dokumentasi vendor.**

### Aturan pelaporan hasil

- **Selisih yang lebih kecil daripada simpangan baku antar-fold tidak boleh dinarasikan
  sebagai temuan.** Berlaku sejak penarikan klaim "27% lebih banyak galat zona D" pada
  T1.1.

  > **ATURAN INI AMBIGU, dan ambiguitasnya mengubah kesimpulan — lihat K14.** T4.1
  > menemukan aturan ini tidak menyebut **SD yang mana**, dan **lima skala derau beredar
  > di proyek ini**:
  >
  > | # | skala | dipakai di | status |
  > | - | ----- | ---------- | ------ |
  > | 1 | SD gabungan `std(concat([A,B]))` | T2.1 | tanpa disebut |
  > | 2 | SD maks per model `max(SD_A, SD_B)` | T4.1 RMSE | tanpa disebut |
  > | 3 | SD selisih berpasangan `std(A−B)` | T4.1, sejak temuan ini | paling dapat dipertahankan |
  > | 4 | SD satu model `SD_RF` | part-6, pencabutan zona D | tanpa disebut, **dan keliru** |
  > | 5 | SD bootstrap RMSE | T1.4 | **SAH** — ditetapkan di muka |
  >
  > Lebih pokok: aturan ini **mencampur dua pertanyaan**. Konsistensi antar-kelompok
  > dijawab SD selisih dan Wilcoxon; **kebermaknaan besaran tidak dijawab varians
  > antar-kelompok sama sekali**. Menyamakan keduanya adalah kekeliruan kategori.
  >
  > **Empat klaim keunggulan berstatus RAPUH** dan tidak boleh dikutip sebagai mantap:
  > T2.1 h12 sensitivitas, T4.1 h6 dan h12 RMSE RF vs GBM, dan T1.1 h6 RMSE RF vs LSTM.
  > Tabel audit lengkap di K14.
  >
  > **Sampai keputusan #8 diambil:** setiap klaim keunggulan wajib **menyebut definisi yang
  > dipakainya**, dan klaim yang rapuh dinyatakan rapuh.
  >
  > Akar yang sama dengan T1.1b: di sana **margin kesetaraan** tak pernah ditetapkan di
  > muka, di sini **ambang kebermaknaan** tak pernah ditetapkan.
- Proporsi dilaporkan dengan **selang kepercayaan Wilson**, terutama pada n kecil.
- Skor `faithfulness` **per kasus tidak boleh dikutip**; hanya rerata atas sepuluh kasus.
  Perbandingan antarmodel pada satu kasus tidak sah.
- Hasil negatif dan kekeliruan sendiri dilaporkan **penuh**, bukan dihaluskan.
- Berhenti melapor setelah setiap percobaan.

### Operasional

- Seluruh panggilan LLM lewat **Google Gemini**, tidak pernah Ollama — termasuk sebagai
  juri RAGAS.
- Sapuan panjang **wajib punya checkpoint** (pelajaran T1.1b: 10 konfigurasi hilang).
  Checkpoint di `.cache/`, bukan `results/`.
- **Uji jalur serialisasi JSON dengan data tiruan sebelum jalan panjang** (pelajaran T1.3:
  22 menit hilang karena `numpy.bool_`).
- Waktu latih adalah satu-satunya kolom yang peka beban CPU. Jangan jalankan pekerjaan lain
  saat percobaan yang melaporkan waktu sedang berjalan.
- Hindari karakter di luar cp1252 pada baris yang dicetak — konsol Windows gagal.
- **Bila operasi berkas diblokir sistem** (Controlled Folder Access aktif di folder
  Documents): berhenti setelah satu diagnosis dan minta pengguna melakukannya manual.
  Jangan iterasi mencoba jalur lain.

---

## 7. Keputusan yang menggantung, menunggu pengguna

| #     | Keputusan                                                                           | Konteks                                                                                                                                                                                | Konsekuensi bila ditunda                                                                                   |
| ----- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| ~~**1**~~ | ~~Isi `evaluation/verifikasi_relevansi.csv`~~ — **SELESAI** | κ = 0,2505 (**lemah**), kesepakatan mentah 52,5% lawan tebakan mayoritas 77,5%. Pelabel **terlalu ketat**: recall hiperglikemia 0,452; presisi `lain` 0,200; presisi `normal` 0,000. Rincian di K1 | **Tidak lagi menggantung.** Aturan κ<0,40 TERPICU: seluruh angka penelusuran wajib berkualifikasi eksplisit di setiap penyebutannya |
| **2** | **Ganti bundle produksi RF ke 50/12?**                                              | 297,15 → 11,09 MB dengan biaya 0,081 mg/dL RMSE. Nisbah terhadap LSTM turun 2.396× → 89×. T1.1b **tidak** membuktikan kesetaraan                                                       | Narasi keterterapan tetap memakai angka 297 MB yang bukan batas kemampuan RF                               |
| **3** | **Tinjau ulang `chunk_size` 900?**                                                  | Dipertahankan pada Tahap 0 karena selisih B4 tidak signifikan (p=0,078). Saat itu **belum diketahui** 900 membuang 8,23% token korpus dan KB-03 kehilangan 48,84% chunk                | Seluruh angka penelusuran tetap diukur pada korpus yang sebagian tidak terindeks                           |
| **4** | **Kerjakan K12?** (perbanyak frasa kondisi, naikkan `fetch_k`)                      | Pekerjaan kecil–sedang, keduanya parameter tanpa indeks ulang. Kandidat perbaikan berdampak tertinggi yang tersisa                                                                     | 98,84% korpus tetap tak terjangkau; K2 dan jurang Hit@1/Hit@5 tetap tanpa perbaikan                        |
| **5** | **Pemilihan model RF vs LSTM — final atau belum?**                                  | T2.1 menunjukkan **LSTM unggul signifikan pada sensitivitas hipoglikemia di kedua horizon** (21,7% vs 32,1%; 4,4% vs 10,6%). Pembenaran RF bertumpu pada keterjelasan kontribusi fitur | Bab VI menyatakan pilihan yang datanya sendiri menantang pada aspek paling kritis secara klinis            |
| **6** | **Adopsi `hanya_kondisi` ke produksi?**                                             | Membuang numeral dari teks kueri menaikkan Hit@1 42,2% → 71,1%, perbandingan kebal kontaminasi. Perlu ubah `build_ablation_query()` di `src/rag/ablation_query.py`                     | Jalur produksi tetap memakai susunan yang terbukti merugikan peringkat                                     |
| **7** | **Uji kestabilan tiga metrik RAGAS lain?**                                          | ~90 panggilan LLM. Saat ini hanya `faithfulness` yang punya data run berulang                                                                                                          | Tiga metrik tetap berstatus "belum diuji, jangan diasumsikan stabil"                                       |
| **8** | **Tetapkan skala derau dan ambang kebermaknaan?** (BARU, dari T4.1)                 | Lima skala derau beredar di repo; aturan pelaporan mencampur konsistensi dengan kebermaknaan. Empat klaim keunggulan berstatus RAPUH. Usulan: SD selisih berpasangan untuk konsistensi, ambang orde-besaran ISO 15197 untuk kebermaknaan. Rincian K14 | Empat klaim keunggulan tidak dapat dikutip sebagai mantap, dan setiap klaim wajib menyebut definisinya sendiri |

---

## Lampiran — di mana mencari apa

| Yang dicari                    | Berkas                                                                      |
| ------------------------------ | --------------------------------------------------------------------------- |
| Rekaman lengkap kronologis     | `docs/journey/README.md` → tujuh part                                       |
| **Keputusan menunggu pembimbing** | **`docs/RINGKASAN_KEPUTUSAN_PEMBIMBING.md`** (delapan, berikut angka pendukung) |
| Rincian tiap keterbatasan      | `docs/DAFTAR_KETERBATASAN.md`                                               |
| Status tiap kebutuhan KF/KNF   | `docs/AUDIT_KEBUTUHAN.md`                                                   |
| Angka final siap kutip         | `results/ringkasan_untuk_bab6.json` (T5.2)                                  |
| Seluruh konfigurasi penyetelan | `results/tuning_log.json`                                                   |
| Prapendaftaran T3.1            | `docs/PRAPENDAFTARAN_T3.1.md`                                               |
| Instruksi kerja asli           | `docs/PROMPT_PERBAIKAN_DAN_REKAYASA.md`, `docs/PROMPT_UJI_COBA_LANJUTAN.md` |

**Lingkungan:** conda `diabetes-ta` di `~/anaconda3/envs/diabetes-ta/python.exe`.
Ryzen 5 5600H, 6C/12T, 15,4 GB RAM, **tanpa GPU**. Jalankan dengan `PYTHONPATH=.`

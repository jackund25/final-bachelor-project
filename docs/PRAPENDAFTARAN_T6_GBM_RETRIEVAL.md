# Prapendaftaran T6 — penghitungan ulang retrieval dan SMBG dengan prediktor GBM

> **Ditulis 14 Agustus 2026, SEBELUM satu pun skrip evaluasi disentuh dan sebelum
> hasil apa pun dilihat.** Cabang `refaktor-tujuh-tugas`.
>
> Diverifikasi saat penulisan: `results/retrieval_realcases_gbm/` belum ada, dan
> `results/eval_prediksi/smbg_deployment.json` masih berisi angka era Random Forest.

---

## 1. Mengapa prapendaftaran ini ada

Prediktor produksi berpindah dari Random Forest ke Gradient Boosting pada 13 Agustus
2026 (alasan: ukuran dan kecepatan, lihat catatan `config.yaml`). Penghitungan ulang
yang menyusul **menyentuh klaim inti penelitian ini** — keunggulan PC-RAG atas RAG
standar pada kasus divergen.

Tanpa dugaan yang ditulis di muka, hasil yang membaik tidak dapat dibedakan dari
**pemilihan model sampai kontribusinya terlihat bagus**. Berkas ini menutup celah itu.

Kekeliruan yang hendak dihindari, ketiganya sudah pernah terjadi di proyek ini:
1. **T1.1b** — margin kesetaraan tidak pernah ditetapkan di muka.
2. **T1.4** — dua dugaan yang tidak dapat benar bersama.
3. **T-RAGAS** — dugaan didaftarkan tanpa memeriksa kode yang sudah menjawabnya.

Untuk menghindari yang ketiga, kode sudah dibaca lebih dulu. Temuan pembacaan itu
justru menjadi isi Bagian 3.

---

## 2. Angka acuan (era Random Forest)

Sumber: `results/retrieval_realcases_kb12_final/crossfold.json`, 6 fold, `top_k` 5,
60 kasus per himpunan per fold, `lambda_mult` 0,0.

| mode | MRR divergen | MRR natural |
|---|---:|---:|
| standard | 0,382 ± 0,009 | 0,647 ± 0,068 |
| pc_rag | 0,433 ± 0,023 | 0,648 ± 0,070 |
| pc_rag_classifier | 0,464 ± 0,036 | 0,641 ± 0,068 |
| oracle | 0,829 ± 0,005 | 0,686 ± 0,064 |

Selisih `pc_rag − standard` pada kasus divergen, per fold:
**[0,048 · 0,070 · 0,067 · 0,084 · 0,025 · 0,009]**, rerata **+0,0505**, positif 6/6.

---

## 3. DUA MODE TIDAK BERGANTUNG PADA PREDIKTOR — ini kontrolnya

Pembacaan kode `eval_retrieval_crossfold.py` menunjukkan:

- **`standard`** memakai `c["current"]`, yaitu glukosa terakhir jendela (`anchor`).
  Nilai terobservasi, bukan keluaran model.
- **`oracle`** memakai `c["actual_future"]`, yaitu glukosa yang benar-benar terjadi.
- **Pemilihan kasus** (`divergent`) dibentuk dari `cond_current` lawan `cond_actual`,
  keduanya dari nilai terobservasi. `pick()` menyampel dengan `random_state=SEED` tetap.

Ketiganya **tidak menyentuh model sama sekali**. Karena korpus, retriever, `lambda_mult`,
`top_k`, dan benih tidak berubah, ketiganya WAJIB tereproduksi persis.

### Dugaan D1 — pemeriksaan integritas (paling penting)

| besaran | nilai yang WAJIB muncul kembali |
|---|---|
| `oracle` divergen per fold | 0,828 · 0,831 · 0,819 · 0,831 · 0,831 · 0,831 |
| `oracle` natural per fold | 0,650 · 0,656 · 0,642 · 0,739 · 0,792 · 0,639 |
| `standard` divergen rerata | 0,382 |
| `standard` natural rerata | 0,647 |
| `n_divergen_tersedia` per fold | 3832 · 3106 · 3678 · 3200 · 3888 · 3416 |

**Aturan keputusan D1.** Bila salah satu meleset lebih dari pembulatan (0,001),
**berhenti dan jangan laporkan angka apa pun.** Artinya yang berubah bukan
prediktornya melainkan jalur pengukurannya, dan seluruh perbandingan menjadi tidak
sah. Periksa lebih dulu: korpus terindeks, `lambda_mult`, `fetch_k`, `chunk_size`,
dan benih.

---

## 4. Dugaan atas dua mode yang memang bergantung pada prediktor

### Mekanismenya, dinyatakan lebih dulu

Yang menentukan keunggulan PC-RAG pada kasus divergen **bukan** galat RMSE, melainkan
seberapa sering prediktor menebak **KONDISI** masa depan dengan benar pada kasus
divergen. Besaran itu sudah terukur pada `results/eval_prediksi/condition_classifier.json`
(pembagian dua pasien, h6):

| jalur | RF | GBM | selisih |
|---|---:|---:|---:|
| regresi lalu ambang | 36,5% | 36,8% | **+0,3 poin** |
| pengklasifikasi kondisi | 29,0% | 35,4% | **+6,4 poin** |

### Dugaan D2 — `pc_rag` nyaris tidak bergerak

Karena mekanismenya hanya bergeser **+0,3 poin persen**, MRR `pc_rag` pada kasus
divergen diperkirakan **0,433 ± 0,02**, dan selisih `pc_rag − standard` tetap
**+0,0505 ± 0,02** serta tetap positif pada **minimal 5 dari 6 fold**.

### Dugaan D3 — `pc_rag_classifier` naik

Karena akurasi kondisi pengklasifikasi pada kasus divergen naik **+6,4 poin persen**,
MRR `pc_rag_classifier` divergen diperkirakan **naik dari 0,464**, dengan kenaikan
**antara +0,01 dan +0,06**.

### Dugaan D4 — distribusi natural tetap tidak konsisten

Pada himpunan natural, selisih `pc_rag − standard` diperkirakan tetap **lebih kecil
daripada SD selisih berpasangannya** dan **tandanya tetap berganti-ganti antar-fold**.
Penggantian prediktor tidak diperkirakan mengubah kesimpulan bahwa PC-RAG tidak
memberi apa-apa pada kasus non-divergen.

---

## 5. Aturan penafsiran yang mengikat

1. **Bila D1 gagal:** berhenti. Tidak ada angka yang dilaporkan sampai penyebabnya
   ditemukan.
2. **Bila selisih `pc_rag − standard` melampaui +0,10 pada kasus divergen:**
   **yang dicurigai lebih dulu adalah pengukurannya, bukan temuannya.** Mekanisme
   yang terukur (+0,3 poin akurasi kondisi) tidak dapat menjelaskan kenaikan dua kali
   lipat. Tiga pemeriksaan wajib sebelum angka itu dilaporkan: (a) apakah `standard`
   dan `oracle` benar-benar tereproduksi, (b) apakah kasus yang terpilih identik,
   (c) apakah selisihnya melampaui variasi antar-jalan retriever.
3. **Bila selisihnya menyusut atau berbalik tanda:** dilaporkan **penuh**, dan
   klaim kontribusi dirumuskan ulang mengikuti angka baru. Penggantian prediktor
   tidak boleh dibatalkan hanya karena hasilnya kurang menguntungkan — itu akan
   menjadikan pemilihan model bergantung pada hasil hilirnya.
4. **Konsistensi diukur SD selisih berpasangan** `std(A−B)` sesuai keputusan #8.
   Ambang kebermaknaan ±15 mg/dL **tidak berlaku** bagi MRR; untuk metrik penelusuran
   hanya pertanyaan konsistensi yang dapat dijawab.
5. **Seluruh angka penelusuran tetap tunduk pada K1** (κ = 0,2505, pelabel lemah) dan
   **K12** (jangkauan 1,16% korpus). Penghitungan ulang ini tidak memperbaiki keduanya.

---

## 6. Skenario SMBG (paruh kedua Rumusan Masalah 1)

Acuan era RF: pada +30 menit model praktis setara *baseline persistence*
(RMSE 27,09 lawan 27,33 mg/dL); keunggulan baru muncul pada +60 menit
(42,86 lawan 47,87 mg/dL).

### Dugaan D5

Keterbatasan pada +30 menit bersifat **informasional, bukan representasional** —
jarak antar-pembacaan SMBG (median 124,8 menit) jauh melampaui horizon prediksinya,
dan penambahan fitur sadar-waktu sudah diuji serta **tidak membantu** (27,20 mg/dL).
Karena sebabnya ada pada data dan bukan pada keluarga model, GBM diperkirakan
**tetap tidak melampaui *baseline persistence* secara bermakna pada +30 menit**.

Kegagalan D5 (yaitu GBM justru melampaui persistence dengan selisih melebihi ambang
±15 mg/dL) akan menjadi temuan yang **membantah penjelasan informasional** yang
sekarang tertulis di laporan, dan wajib dilaporkan sebagai koreksi atas penjelasan
itu — bukan sekadar sebagai kabar baik.

---

## 7. Yang TIDAK diukur oleh penghitungan ulang ini

- Tidak menguji apakah PC-RAG bekerja pada retriever berjangkauan luas (K12 tetap).
- Tidak memperbaiki pelabel relevansi (K1 tetap).
- Tidak menyentuh RAGAS — `run_ragas.py` memakai glukosa tetap 58/120/230 dan tidak
  pernah memanggil prediktor.
- Tidak menyetel GBM. Hiperparameternya tetap bawaan, sama seperti T4.1.

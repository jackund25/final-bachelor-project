# Prapendaftaran T13 — penelusuran leksikal dan hibrida pada evaluasi crossfold

> **Ditulis 17 Agustus 2026, SEBELUM satu pun angka crossfold hibrida dilihat.**
> Cabang `refaktor-tujuh-tugas`.
>
> Diverifikasi saat penulisan: `results/retrieval_realcases_kb12_gbm_kalimat/`
> hanya memuat lengan berbasis vektor; belum ada lengan BM25 maupun hibrida.

---

## 1. Mengapa percobaan ini dijalankan

Bukan karena angka sekarang kurang menguntungkan, melainkan karena **bukti yang sudah
ada di repositori ini belum pernah ditindaklanjuti**.

`results/baseline_ablation_fullkb_kb12_sym/hybrid_bm25.json` (n=120 per himpunan,
korpus 2.061 potongan, chunking lama) mencatat:

| Himpunan | Vektor (MMR) | BM25 saja | Hibrida RRF |
|---|---:|---:|---:|
| natural, terkondisi | 0,506 | **0,913** | 0,887 |
| natural, standard | 0,495 | **0,904** | 0,876 |
| divergen, terkondisi | 0,393 | **0,444** | 0,418 |
| divergen, standard | 0,327 | 0,290 | 0,272 |

Dan yang paling menentukan, **selisih kontribusi** pada kasus divergen
(terkondisi − standard), yakni klaim inti penelitian ini:

| Retriever | Selisih |
|---|---:|
| Vektor | +0,066 |
| **BM25** | **+0,154** |
| Hibrida RRF | +0,146 |

---

## 2. Mekanisme yang didugakan, dinyatakan lebih dulu

Model embedding produksi `all-MiniLM-L6-v2` adalah model **berbahasa Inggris**,
sedangkan korpusnya **berbahasa Indonesia**. T9 sudah menunjukkan mengganti model ke
multibahasa merusak kemampuan membedakan kondisi, sehingga jalan itu tertutup.
Penelusuran leksikal menawarkan jalan lain: **BM25 tidak bergantung pada ruang
semantik sama sekali.** Istilah klinis pada kueri — "hipoglikemia", "70 mg/dL",
"ketoasidosis" — dicocokkan persis, tepat pada titik model Inggris paling lemah.

Dugaan mengapa selisih kontribusi MELEBAR, bukan sekadar naik: kueri terkondisi
memuat istilah kondisi yang **benar** (kondisi masa depan), sedangkan kueri standard
memuat istilah kondisi yang **salah** (kondisi sekarang, yang pada kasus divergen
memang berbeda). Pencocokan leksikal menghukum istilah yang salah lebih keras
daripada kemiripan semantik, yang cenderung mengaburkan perbedaannya.

Bila mekanisme ini benar, kenaikannya akan **terkonsentrasi pada kasus divergen** dan
**pada lengan terkondisi**, bukan merata.

---

## 3. Mengapa bukti yang ada belum memadai

Tiga alasan, semuanya harus ditutup sebelum angkanya boleh masuk naskah:

1. **Chunking lama.** Diukur pada 2.061 potongan dengan pemisah lama; produksi kini
   2.248 potongan dengan pemisah kalimat.
2. **Bukan crossfold.** Diukur pada set ablasi, bukan enam *fold* lintas-pasien yang
   menjadi dasar klaim inti.
3. **Metrik yang sama lemahnya.** `classify_chunk` tetap dipakai, dan ia sudah
   terbukti berbias panjang serta salah melabeli isi yang benar
   (`results/eval_rag/diagnosis_metrik_chunking.json`). **T13 tidak memperbaiki ini**,
   dan karena itu tidak boleh diklaim memperbaikinya.

---

## 4. Dugaan yang didaftarkan di muka

- **D1 — kontrol.** Lengan `standard` dan `oracle` berbasis vektor WAJIB tereproduksi
  terhadap `results/retrieval_realcases_kb12_gbm_kalimat/crossfold.json`
  (`standard` 0,166; `oracle` 0,776). Bila meleset, jalur pengukurannya berubah dan
  seluruh hasil T13 batal.
- **D2 — oracle tetap batas atas.** Pada setiap lengan retriever, `oracle` WAJIB tetap
  di atas `standard`. Pemeriksaan ini yang membongkar kegagalan T9; ia dipakai lagi
  di sini.
- **D3 — arah pada kasus divergen.** BM25 diduga **menaikkan** selisih
  terkondisi − standard di atas +0,083 yang berlaku sekarang. Ambang keputusan
  ditetapkan di muka: kenaikan selisih **> 0,02** dianggap nyata.
- **D4 — konsentrasi kenaikan.** Bila mekanisme Bagian 2 benar, kenaikan terbesar ada
  pada **divergen terkondisi**, dan lengan **divergen standard tidak ikut naik**
  (boleh turun). Bila justru kedua lengan naik merata, mekanisme yang didugakan
  **salah** dan tafsirnya harus ditulis ulang meskipun angkanya membaik.
- **D5 — konsistensi antar-fold.** Kenaikan wajib positif pada **minimal 5 dari 6**
  *fold*, sama dengan ambang yang dipakai T6. Kenaikan rerata yang besar tetapi
  hanya positif pada 3 fold TIDAK dianggap hasil.

---

## 5. Aturan penafsiran yang mengikat

1. **Bila D1 atau D2 gagal:** berhenti, tidak ada angka yang dilaporkan.
2. **Bila hasilnya membaik namun D4 gagal:** angkanya tetap dilaporkan, tetapi
   penjelasan mekanismenya **dicabut** dan diganti "sebab belum diketahui". Angka
   yang benar dengan sebab yang salah lebih berbahaya daripada angka yang jujur
   tanpa sebab.
3. **Peringatan yang tidak boleh dihilangkan:** seluruh angka T13 tetap tunduk pada
   K1 (κ = 0,2505) dan K12 (jangkauan 1,16% korpus), dan tetap diukur dengan
   `classify_chunk` yang lemah. T13 **tidak** menyelesaikan keduanya.
4. **Bila BM25 sendirian mengungguli hibrida**, yang diadopsi BM25 sendirian.
   Memilih hibrida hanya karena terdengar lebih canggih adalah memilih berdasarkan
   kesan, bukan bukti.

---

## 6. Lengan yang diuji

Seluruhnya pada korpus produksi sekarang (2.248 potongan, pemisah kalimat), enam
*fold* lintas-pasien, `top_k` 5:

| Kode | Retriever |
|---|---|
| **V** | vektor + MMR (kontrol, konfigurasi produksi) |
| **B** | BM25 saja |
| **H** | hibrida RRF (vektor + BM25), `k_rrf` 60 tidak disetel |

Setiap lengan diukur pada empat mode yang sudah ada: `standard`, `pc_rag`,
`pc_rag_classifier`, `oracle`.

`k_rrf` = 60 dipertahankan pada nilai bakunya dan **tidak disetel**, agar tidak
menambah parameter bebas yang menuntut protokol penyetelan tersendiri.

---

## 7. Keluaran

`results/retrieval_realcases_kb12_gbm_kalimat/hibrida_crossfold.json`, memuat
seluruh lengan termasuk yang kalah.

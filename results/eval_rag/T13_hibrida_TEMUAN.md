# T13 — hasil penelusuran leksikal dan hibrida, beserta keberatan atasnya

Sumber angka: `results/retrieval_realcases_kb12_gbm_kalimat/crossfold.json`,
medan `divergen_lengan` dan `natural_lengan`.
Prapendaftaran: `docs/PRAPENDAFTARAN_T13_HIBRIDA.md`.

---

## 1. Hasil, kasus divergen (rerata 6 fold)

| Lengan | standard | pc_rag | pc_rag_classifier | oracle |
|---|---:|---:|---:|---:|
| vektor (produksi) | 0,166 | 0,249 | 0,308 | 0,776 |
| bm25 | 0,275 | 0,369 | 0,449 | 1,000 |
| **hibrida RRF** | 0,164 | 0,273 | **0,354** | **1,000** |

Selisih kontribusi (terkondisi − standard), seluruhnya positif 6/6 fold:

| Lengan | pc_rag | pc_rag_classifier |
|---|---:|---:|
| vektor | +0,0832 | +0,1423 |
| bm25 | +0,0938 | +0,1742 |
| **hibrida** | **+0,1087** | **+0,1895** |

Kasus natural: BM25 dan hibrida menaikkan seluruh mode (0,692 → 0,912/0,902),
tetapi selisih kontribusinya tetap sekitar nol (+0,007 dan +0,009, positif 4/6 dan
3/6). Kesimpulan lama bertahan: **tidak ada manfaat pada kasus non-divergen.**

---

## 2. Pemeriksaan dugaan prapendaftaran

| Dugaan | BM25 | Hibrida |
|---|---|---|
| D1 lengan vektor tereproduksi | LOLOS | LOLOS |
| D2 oracle di atas standard | LOLOS | LOLOS |
| D3 kenaikan selisih > 0,02 | **GAGAL** pada `pc_rag` (+0,0107) | LOLOS (+0,0255) |
| D4 kenaikan terkonsentrasi | **GAGAL** (standard ikut naik +0,1087) | LOLOS (−0,0018) |
| D5 konsisten ≥ 5/6 fold | **GAGAL** pada `pc_rag` (3/6) | LOLOS (6/6) |

**Hibrida lolos kelimanya; BM25 sendirian gagal tiga.** Sesuai Aturan 4
prapendaftaran, yang diadopsi adalah lengan yang menang menurut bukti, yaitu
hibrida — bukan yang terdengar lebih sederhana.

**Mekanisme yang didugakan DICABUT untuk BM25.** Bagian 2 prapendaftaran menduga
lengan `standard` tidak akan ikut naik karena pencocokan leksikal menghukum istilah
kondisi yang salah. Pada BM25 ia justru naik +0,1087, sehingga dugaan itu salah dan
Aturan 2 berlaku: angkanya dilaporkan, penjelasannya dicabut. Pada hibrida dugaannya
bertahan (standard −0,0018).

---

## 3. KEBERATAN YANG MELEMAHKAN SELURUH ANGKA DI ATAS

**Retriever dan alat ukur memakai sinyal yang sama.**

`classify_chunk` melabeli potongan menurut kata kunci:
`hipoglikemi`, `gula darah rendah`, `15-15`, `hiperglikemi`, `ketoasidosis`, dan
seterusnya. Kueri yang dibangun `build_ablation_query()` memuat kata-kata itu secara
harfiah — kueri oracle hipoglikemia berbunyi *"Hipoglikemia, gula darah rendah di
bawah 70 mg/dL ... (aturan 15-15)"*.

BM25 bekerja dengan mencocokkan token persis. Jadi ia mengoptimalkan **sinyal yang
sama persis dengan yang diukur metriknya**.

**Akibatnya:**

1. `oracle` = 1,000 **bukan** bukti penelusuran sempurna. Sebagian besar adalah
   artefak keselarasan retriever dengan alat ukur.
2. Perbandingan **antar-lengan** (vektor lawan BM25 lawan hibrida) **berat sebelah**
   memihak lengan leksikal, dan tidak sah dipakai untuk menyatakan BM25 "lebih baik".
3. Selisih kontribusi **di dalam satu lengan** lebih tahan, karena kedua mode diukur
   dengan retriever yang sama. Tetapi ia pun ikut terangkat: kueri terkondisi memuat
   istilah yang benar dan kueri standard memuat istilah yang salah, dan pencocokan
   leksikal memperbesar beda itu secara mekanis.

**Kesimpulan yang sah:** hasil ini **menjanjikan tetapi belum dapat diklaim**. Ia
tidak boleh masuk naskah sebagai perbaikan yang terbukti sampai divalidasi dengan
alat ukur yang **tidak berbasis kata kunci**.

---

## 3b. KEBERATAN ITU SUDAH DIJAWAB — hasil T14

`results/ragas/lengan_penelusuran.json`, prapendaftaran
`docs/PRAPENDAFTARAN_T14_RAGAS_LENGAN.md`. Sepuluh kasus, korpus produksi,
metrik berbasis rujukan.

| Lengan | context_precision | context_recall |
|---|---:|---:|
| vektor | 0,025 | 0,100 |
| **hibrida** | **0,545** | **0,800** |
| selisih | **+0,520** | **+0,700** |

Sebaran per kasus, yang lebih meyakinkan daripada reratanya:

| | vektor | hibrida |
|---|---:|---:|
| kasus `context_recall` = 0 | **9 dari 10** | 2 dari 10 |
| kasus `context_recall` = 1 | 1 | **8** |
| menang | **0 kasus** | **7 kasus** (3 seri) |

**Lengan vektor tidak pernah menang sekali pun.**

**Mengapa ini menjawab keberatan Bagian 3.** RAGAS `context_recall` menilai apakah
klaim pada jawaban acuan **didukung** oleh konteks terambil, lewat penilaian LLM
yang bersifat semantik — bukan pencocokan kata kunci. Ia karena itu tidak berbagi
mekanisme dengan BM25 sebagaimana `classify_chunk` berbagi. Keunggulan hibrida
bertahan pada alat ukur yang **tidak** dioptimalkan BM25.

**Dugaan D2 GAGAL, dan itu dicatat.** Prapendaftaran menduga `context_precision`
tidak akan naik, mengikuti pola pertukaran yang dilaporkan Nicolas pada Parent-Child
RAG (recall naik, precision turun tipis). Nyatanya precision naik **+0,52**. Tidak
ada pertukaran; hibrida unggul pada kedua metrik. Dugaan mekanisme "memperluas
jangkauan menukar ketepatan" karena itu **tidak berlaku di sini**.

**Batas yang tetap melekat:**

- **n = 10.** Terlalu kecil untuk uji signifikansi; tidak ada nilai-p yang dihitung.
- Independensinya **substansial, bukan sempurna**: pertanyaan, jawaban acuan, dan
  potongan sama-sama memakai istilah klinis yang sama. Yang berbeda adalah penilaian
  dukungan semantik lawan pencocokan token.
- **E01 dan E02 gagal pada KEDUA lengan** — ada pertanyaan yang tidak terjawab oleh
  retriever mana pun, dan itu tidak diperbaiki hibrida.
- K1 (κ = 0,2505) dan K12 tidak tersentuh T14.

**Kesimpulan yang kini sah:** keunggulan hibrida **bukan artefak alat ukur**. Ia
bertahan pada metrik berbasis rujukan, konsisten pada 7 dari 10 kasus tanpa satu pun
kekalahan. Temuan Bagian 1–2 karena itu **dapat dilaporkan**, dengan batas n=10 dan
tanpa klaim signifikansi.

## 4. Langkah yang diperlukan sebelum angka ini boleh dipakai

Validasi dengan metrik yang tidak berbagi sinyal dengan retriever:

1. **RAGAS `context_precision` dan `context_recall`** — berbasis rujukan, bukan kata
   kunci. Berkas `results/ragas/caveat.md` proyek ini sendiri sudah menyatakan
   keduanya lebih objektif daripada faithfulness. Dua TA sepembimbing memakainya dan
   diterima.
2. **Penilaian relevansi oleh manusia** pada sampel — infrastrukturnya sudah ada
   (`docs/PANDUAN_PENILAI_v2.md`, skrip kappa antar-penilai).

Bila salah satu dari keduanya menunjukkan hibrida tetap unggul, temuan ini menjadi
kokoh dan layak menjadi kontribusi tambahan. Bila tidak, ia tetap dilaporkan sebagai
**hasil yang tampak menjanjikan pada satu alat ukur namun tidak bertahan pada alat
ukur lain** — dan itu pun temuan yang jujur dan bernilai.

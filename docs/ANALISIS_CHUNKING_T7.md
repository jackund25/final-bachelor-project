# T7 — Analisis dan perbaikan pemecahan dokumen (*chunking*)

Ditulis 16 Agustus 2026, cabang `refaktor-tujuh-tugas`.
Prapendaftaran: `docs/PRAPENDAFTARAN_T7_CHUNKING.md` (ditulis lebih dulu).

Dokumen ini menjawab tiga pertanyaan yang diajukan:

1. Mengapa kutipan di Streamlit terpotong bukan "titik ke titik"?
2. Apakah *chunking* di sisi vektor sama dengan di sisi model Gemini?
3. Apakah kalimat sebelum dan sesudah kutipan masih berkaitan, sehingga tidak ada
   konteks yang tertinggal?

---

## 1. Jawaban pertanyaan 2 lebih dahulu — vektor lawan Gemini

Pertanyaan ini didahulukan karena jawabannya mengubah cara membaca dua pertanyaan
lain.

**Pemecahannya SAMA. Yang berbeda adalah berapa banyak dari hasil pemecahan itu
yang benar-benar dipakai masing-masing pihak.**

Korpus dipecah **satu kali saja**, saat *indexing*, di
`MedicalKnowledgeBase.chunk_documents()`. Tidak ada pemecahan kedua di jalur
Gemini. Tetapi satu potongan yang sama dikonsumsi tiga pihak dengan panjang
efektif yang berbeda:

| Pihak | Berkas | Yang benar-benar dipakai | Terlihat? |
|---|---|---|---|
| **Indeks vektor** (MiniLM) | `knowledge_base.py:_build_embeddings` | **256 token pertama saja.** Sisanya dibuang senyap | **Tidak** |
| **Gemini** (LLM) | `prompts.py:41` — `row.get('text','')` | **Potongan UTUH**, tanpa pemotongan | — |
| **UI Streamlit** | `citations.py:build_source_list` | 700 karakter pertama | Ya |

Verifikasi yang dilakukan, bukan diasumsikan:

- `.env` menetapkan `EMBED_PROVIDER=sentence-transformers`, yang **mengungguli**
  `config.yaml` menurut presedensi `src/config.py` (ctor > env > yaml > default).
- Koleksi Chroma produksi berdimensi **384** (`chroma.sqlite3`, tabel `collections`),
  yaitu MiniLM. `models/embedding-001` milik Google berdimensi 768. Jadi jalur
  Google **tidak** dipakai produksi meskipun tersedia di konfigurasi.
- `max_tokens: 700` pada `advisor_chain.py` adalah `max_output_tokens` — membatasi
  **keluaran** Gemini, bukan masukannya. Konteks masuk tidak dipotong di mana pun.

**Akibat yang menentukan.** Karena pengambilan terjadi **sebelum** Gemini, potongan
yang jawabannya terletak setelah token ke-256 tidak akan pernah terambil, sehingga
Gemini tidak pernah mendapat kesempatan melihatnya. Keunggulan Gemini yang menerima
teks utuh karena itu **semu**: ia hanya berlaku bagi potongan yang berhasil lolos
saringan vektor yang cacat.

---

## 2. Dua cacat, bukan satu

| | **Cacat A — pemotongan senyap** | **Cacat B — potongan di tengah kalimat** |
|---|---|---|
| Letak | Sisi vektor | Sisi teks (pemecahan + tampilan) |
| Terlihat pengguna | Tidak | Ya — inilah yang Anda lihat |
| Sebab | `max_seq_length` = 256 token | Daftar pemisah tanpa tanda akhir kalimat |
| Besaran (V0) | **8,60%** token korpus terbuang | hanya **19,4%** potongan berakhir kalimat utuh |

Memperbaiki B **tidak** memperbaiki A. Ini didaftarkan sebagai dugaan D2 dan diuji,
bukan diandaikan.

---

## 3. Akar masalah "bukan titik ke titik" — dan koreksi atas diagnosis sebelumnya

Diagnosis pada catatan serah-terima sebelumnya menyebut: daftar pemisah
`["\n## ", "\n### ", "\n\n", "\n", " ", ""]` tidak memuat pemisah kalimat, dan
perbaikannya adalah **menyisipkan `". "` sebelum `" "`**.

**Diagnosisnya benar, tetapi perbaikan yang diusulkan hampir tidak berpengaruh.**

Sebabnya: menyisipkan `". "` sesudah `"\n"` berarti `"\n"` tetap dicoba lebih dulu
dan **selalu berhasil**, sehingga `". "` tidak pernah sempat dipertimbangkan. Pada
teks hasil ekstraksi PDF, `"\n"` tunggal adalah **pembungkusan baris** — artefak
tata letak halaman, bukan batas makna.

Diukur pada 531 halaman korpus, potongan 256 token:

| Urutan pemisah | Berakhir kalimat utuh | Bermula utuh |
|---|---|---|
| kalimat **sesudah** `"\n"` (usulan lama) | 37,4% | 62,9% |
| kalimat **sebelum** `"\n"` (dipakai) | **79,6%** | **86,7%** |

Karena itu urutan yang diadopsi mendahulukan batas paragraf (`"\n\n"`) dan judul,
lalu **batas kalimat**, baru `"\n"` tunggal:

```python
PEMISAH_KALIMAT = [
    "\n## ", "\n### ", "\n\n",
    ". ", ".\n", "! ", "!\n", "? ", "?\n",
    "\n", "; ", " ", "",
]
```

Varian `".\n"` diperlukan terpisah: kalimat yang berakhir tepat di ujung baris
menghasilkan `".\n"` tanpa spasi, sehingga `". "` saja tidak mengenainya.

### Jebakan kedua yang ditemukan saat implementasi

`RecursiveCharacterTextSplitter` ber-default `keep_separator=True`, yang menempelkan
pemisah ke **awal** potongan berikutnya. Dengan pemisah `". "` hasilnya adalah
potongan yang dibuka tanda titik dan potongan sebelumnya kehilangan titiknya —
memindahkan cacat, bukan memperbaikinya. Diverifikasi pada versi terpasang:

```
keep_separator=True   -> ['... per kgBB',  '. Titrasi tiap tiga hari', ...]
keep_separator="end"  -> ['... per kgBB.', 'Titrasi tiap tiga hari.',  ...]
```

Karena itu `keep_separator="end"` wajib, dan dikunci konstanta
`KEEP_SEPARATOR_AKHIR`.

---

## 4. Perubahan yang dikerjakan — 5W+1H

### 4.1 Pemotongan kutipan pada batas kalimat (menjawab pertanyaan 1)

- **What** — fungsi `potong_batas_kalimat()` beserta penentu batas kalimat
  ber-empat penjaga, menggantikan pemotongan batas kata untuk kutipan.
- **Who** — `src/rag/citations.py`; konsumennya daftar "Sumber Rujukan" di
  Streamlit, log keputusan dokter, dan ringkasan sumber `advisor_chain`.
- **Where** — `src/rag/citations.py`, dipanggil di `build_source_list()`. Fungsi
  lama `_truncate()` **dipertahankan** untuk judul dokumen, yang memang bukan
  kalimat.
- **When** — setiap kali satu baris rujukan dibentuk, yakni tiap kali dokter
  menjalankan satu konsultasi. Tidak menyentuh *indexing*, sehingga berlaku
  **tanpa** indeks ulang.
- **Why** — Gao dkk. (2023) §V.A.1 hal. 8 menyebut kelemahan pemecahan berukuran
  tetap sebagai *"truncation within sentences"*. Batas kata hanya menjamin kata
  terakhir utuh, bukan kalimatnya. Manning dkk. (2009) hal. 217 menegaskan satuan
  yang dikembalikan kepada pembaca pada *passage retrieval* adalah *passage*
  berbatas bermakna.
- **How** — kandidat batas dicari dengan regex lalu **disaring** empat penjaga:
  (1) singkatan yang berakhir titik (`hal.`, `dr.`, `dkk.`, `mg.`); (2) huruf
  tunggal (inisial nama); (3) angka pendek (penanda daftar bernomor `1.`);
  (4) karakter sesudahnya harus tampak seperti awal kalimat. Bilangan desimal
  (`7.5`) aman dengan sendirinya karena kandidat mensyaratkan spasi sesudah titik.
  Bila batas kalimat terdekat memangkas lebih dari 40% jatah tampilan, sistem
  **jatuh kembali** ke batas kata — kutipan terlalu pendek tidak berguna untuk
  mencocokkan ke PDF sumber (bukti KNF-08).

### 4.2 Pemisah kalimat pada pemecahan korpus (menjawab akar masalah)

- **What** — konstanta `PEMISAH_KALIMAT` dan argumen `pemisah_kalimat=` pada
  `chunk_documents()`.
- **Who** — `src/rag/knowledge_base.py`, satu-satunya jalur pemecahan korpus.
- **Where** — `chunk_documents()`; dipilih dari baris perintah lewat
  `scripts/reingest_kb.py --pemisah-kalimat`.
- **When** — hanya saat *indexing*. Karena itu perubahannya **menuntut indeks
  dibangun ulang**; indeks lama tidak ikut berubah.
- **Why** — sama dengan 4.1, ditambah Reimers & Gurevych (2019): SBERT menurunkan
  *sentence embeddings*, sehingga satuan yang model ini dilatih untuk mewakili
  adalah kalimat. Menyelaraskan batas potongan dengan batas kalimat berarti
  menyelaraskannya dengan satuan yang dipahami model, bukan sekadar merapikan
  tampilan.
- **How** — daftar pemisah berprioritas dengan urutan terkoreksi (Bagian 3) dan
  `keep_separator="end"`. Default **sengaja tetap perilaku lama** supaya angka
  retrieval yang sudah dilaporkan tidak berubah diam-diam; varian dipilih eksplisit.

### 4.3 Penakar panjang berbasis token (menjawab cacat A)

- **What** — `_penakar_token()` dan argumen `satuan_panjang="token"`.
- **Who** — `src/rag/knowledge_base.py`, dipakai `chunk_documents()` sebagai
  `length_function` splitter.
- **Where** — `scripts/reingest_kb.py --satuan-panjang token`.
- **When** — saat *indexing*, pada varian V3.
- **Why** — inilah alasan paling penting di dokumen ini. Menakar panjang dengan
  `len()` berarti batas potongan diukur dengan **satuan yang berbeda** dari satuan
  yang dipakai model embedding. Selama dua satuan itu berbeda, **tidak ada** nilai
  `chunk_size` karakter yang dapat menjamin potongan muat di jendela model:
  `chunk_size=900` menghasilkan potongan **9 sampai 443 token**. Menakar dengan
  tokenizer model itu sendiri mengubah jaminannya dari **statistik** menjadi
  **konstruktif**.
- **How** — `AutoTokenizer` model embedding yang **sama** dengan yang dipakai saat
  kueri, dengan `add_special_tokens=True` supaya `[CLS]` dan `[SEP]` ikut memakan
  jatah 256 (mengabaikannya membuat potongan meleset dua token).

### 4.4 Penanda konteks tepi (menjawab pertanyaan 3)

- **What** — `bermula_di_batas_kalimat()`, `berakhir_di_batas_kalimat()`, dan medan
  `mulai_kalimat_utuh` / `akhir_kalimat_utuh` / `cara_potong` pada tiap baris rujukan.
- **Who** — `src/rag/citations.py`, ditampilkan `app/streamlit_app.py`.
- **Where** — `build_source_list()`; keterangan di UI dipisah menjadi dua baris.
- **When** — setiap penyusunan baris rujukan.
- **Why** — pertanyaan "apakah ada konteks yang tertinggal" tidak dapat dijawab
  dengan ya/tidak untuk seluruh sistem; jawabannya **berbeda per potongan**.
  Menandainya membuat dokter tahu kalimat di tepi kutipan bersambung ke potongan
  tetangga, bukan kalimat yang rusak. Sejalan dengan gagasan *Small2Big* yang
  disurvei Gao dkk. §V.A.1: kalimat sebagai satuan, tetangganya sebagai konteks.
- **How** — kedua penanda menggambarkan **potongan hasil indexing**, bukan hasil
  pemotongan tampilan. Dua sebab itu sengaja dipisah di UI karena kerap tertukar:
  pemotongan tampilan dapat dibatalkan dengan mencentang "utuh", pemotongan
  *indexing* tidak dapat.

---

## 5. Sejauh mana pertanyaan 3 benar-benar terjawab

Jujur pada batasnya:

**Yang sudah dijamin.** Proporsi potongan yang bermula dan berakhir pada kalimat
utuh naik tajam (47,1% → 92,7% dan 19,4% → 82,7% pada V1). Setiap potongan yang
masih bersambung kini **ditandai** di UI, sehingga tidak ada penyambungan diam-diam.

**Yang belum dijamin.** `chunk_overlap` 120 karakter menyalin ekor potongan
sebelumnya ke potongan berikutnya, tetapi 120 karakter **tidak menjamin** satu
kalimat penuh ikut tersalin — kalimat pedoman kerap lebih panjang dari itu.
*Small2Big* sepenuhnya menuntut penyimpanan tetangga sebagai metadata dan
pengembangan konteks saat pengambilan; itu **belum** diimplementasikan dan
sengaja tidak diklaim.

**Kejujuran sitasi.** Rujukan Gao untuk *Small2Big* adalah [90], dan [88]–[90]
seluruhnya **pos blog, bukan makalah wasit**. Karena itu *Small2Big* disebut
sebagai praktik yang **disurvei** Gao dkk., bukan sebagai metode ber-wasit. Dasar
ber-wasit untuk memilih batas kalimat berdiri pada Reimers & Gurevych (2019) dan
Manning dkk. (2009).

---

## 6. Hasil percobaan

Sumber: `results/eval_rag/strategi_chunking.json`. Set penyetelan n=240 dan set
pelaporan n=240, tidak beririsan.

| Varian | n chunk | MRR (lapor) | Hit@1 | Token terbuang | Akhir kalimat utuh |
|---|---:|---:|---:|---:|---:|
| **V0** kontrol | 2.061 | 0,492 | 29,2% | **8,23%** | **19,7%** |
| **V1** 900 kar + kalimat | 2.248 | 0,468 | 29,2% | 6,14% | **83,5%** |
| **V2** 500 kar + kalimat | 4.063 | **0,507** | **34,2%** | 0,00% | 85,3% |
| **V3** 256 token + kalimat | 2.587 | 0,150 | 3,3% | **0,00%** | 80,4% |

**Dugaan prapendaftaran:**

- **D1 LOLOS, dan meyakinkan.** V0 menghasilkan **2.061 potongan — sama persis**
  dengan produksi, dengan MRR 0,492 yang juga sama persis dengan angka pada
  `config.yaml`. Kontrolnya tereproduksi.
- **D2 LOLOS.** V1 menaikkan potongan berakhir-utuh 19,7% → 83,5%, tetapi token
  terbuang **tetap 6,14%**. Terbukti: memperbaiki cacat B tidak memperbaiki cacat A.
- **D3 LOLOS.** V3 nol potongan melewati 256 token (maksimum 253).

## 7. V3 anjlok — dan mengapa itu TIDAK boleh dibaca sebagai V3 buruk

MRR V3 0,150 lawan V0 0,492 adalah penurunan yang terlalu besar untuk sekadar
pertukaran mutu. Penelusurannya menemukan sebabnya bukan pada V3, melainkan pada
**alat ukurnya**. Analisis penuh: `results/eval_rag/diagnosis_metrik_chunking.json`,
dapat diulang dengan `scripts/diagnosa_metrik_chunking.py`.

**Bukti kualitatif.** Untuk kueri kondisi normal, dua potongan teratas V3 adalah
*"Target Glukosa Darah Berdasarkan ISPAD dan IDF"* dan *"Target Glukosa Darah Untuk
Penyandang Diabetes Melitus Gestational"* — keduanya justru **tabel sasaran
glikemik**, yakni isi paling tepat untuk kueri itu. Keduanya dilabeli `lain`, karena
daftar kata kunci kelas `normal` menuntut frasa `target glikemik` / `kontrol
glikemik` / `hba1c`, sedangkan potongan itu menulis *"Target Glukosa Darah"*.
**Alat ukurnya menghukum potongan yang benar.**

**Bukti kuantitatif — bias panjang.** `classify_chunk()` memberi label dari teks
potongan itu sendiri dengan kata kunci berbobot. Makin banyak teks dalam satu
potongan, makin besar peluangnya menangkap salah satu frasa. Peluang mendapat label
topik (bukan `lain`), menurut panjang potongan:

| Varian | <300 | 300–600 | 600–900 | ≥900 |
|---|---:|---:|---:|---:|
| V0 | 30,0% | 43,1% | 53,2% | 77,8% |
| V1 | 19,3% | 41,0% | 51,6% | 100,0% |
| V2 | 25,1% | 38,2% | — | — |
| V3 | 26,4% | 45,3% | 47,9% | 48,7% |

**Naik monoton di keempat varian.** Karena percobaan ini justru mengubah panjang
dan batas potongan, label yang menjadi dasar MRR **ikut bergerak bersama
perlakuan**. Kelas pembandingnya bukan besaran tetap.

Akibatnya: Aturan 4 prapendaftaran mengandaikan alat ukur invarian lintas lengan.
Pengandaian itu **tidak terpenuhi**. Penolakan V3 karena itu **ditangguhkan, bukan
disimpulkan**.

**Yang TIDAK boleh disimpulkan dari ini.** Temuan di atas tidak membuktikan V3 lebih
baik — hanya membuktikan perbandingan ini tidak dapat memutuskan. Bias panjang juga
tidak menjelaskan seluruhnya: V2 berpotongan lebih pendek daripada V0 namun ber-MRR
lebih tinggi, sehingga masih ada faktor yang belum dipisahkan.

**Status ini POST-HOC.** Dijalankan sesudah hasil dilihat, ditulis pada skrip
terpisah supaya batas antara yang diprapendaftarkan dan yang tidak tetap terlihat.
Tidak ada nilai-p yang sah dihitung atasnya.

### Yang diadopsi dan yang ditangguhkan

- **Diadopsi — pemotongan kutipan pada batas kalimat** (`citations.py`). Tidak
  bergantung pada indeks maupun MRR, langsung menjawab keluhan yang terlihat.
- **Layak diadopsi — pemisah kalimat pada indexing** (V1). Aturan 5 prapendaftaran
  sudah menetapkannya tanpa menunggu MRR; dasarnya integritas kalimat yang terukur
  langsung (19,7% → 83,5%), dan MRR-nya tidak bergerak nyata (0,492 → 0,468 pada set
  pelaporan; 0,495 → 0,500 pada set penyetelan — berlawanan arah, jadi selisihnya
  derau).
- **Ditangguhkan — V3 sadar-token.** Menuntut label kebenaran yang tidak diturunkan
  dari teks potongan (label tingkat dokumen/halaman, atau penilaian relevansi oleh
  manusia).
- **`config.yaml` TIDAK diubah.** Keputusan indeks ulang produksi menunggu pembimbing.

## 8. T8 — apakah pemotongan senyap itu sendiri merugikan retrieval?

Sumber: `results/eval_rag/jendela_token.json`, skrip `scripts/eval_jendela_token.py`.

**Mengapa percobaan ini SAH sedangkan T7 tidak bisa memutuskan.** W256 dan W512
memakai `chunk_size`, `chunk_overlap`, dan daftar pemisah yang **persis sama**,
sehingga potongannya identik teks demi teks dan label `classify_chunk()` tidak
bergeser. Konfound yang membatalkan perbandingan V3 **tidak berlaku di sini**.
Satu-satunya yang berubah adalah jendela token. Inilah uji yang diminta kesimpulan
T5.1 dan belum pernah dijalankan.

**Dasar angka 256.** Berasal dari `sentence_bert_config.json` milik
`all-MiniLM-L6-v2` (`"max_seq_length": 256`) — **pilihan penulis model, bukan batas
arsitektur**. BERT di bawahnya ber-`max_position_embeddings` 512, sehingga posisi
257–512 bobotnya ada. Potongan terpanjang korpus 443 token, jadi 512 menghapus
pemotongan seluruhnya tanpa menyentuh satu pun batas potongan.

| Varian | Jendela | n chunk | MRR | Hit@1 | Token terbuang |
|---|---:|---:|---:|---:|---:|
| **W256** kontrol | 256 | 2.061 | **0,492** | 29,2% | **8,23%** |
| **W512** satu faktor | 512 | 2.061 | 0,469 | 29,2% | **0,00%** |
| W512K + pemisah kalimat | 512 | 2.248 | 0,438 | 29,2% | 0,00% |

**Validasi alat (wajib, karena setelan yang diabaikan diam-diam adalah persis
jenis cacat yang sedang diteliti):**

- **H1 LOLOS** — nol potongan melewati jendela pada 512.
- **H2 LOLOS** — vektor potongan yang tadinya terpotong berubah, kemiripan rerata
  **0,964**.
- **H3 LOLOS** — kontrol negatif: potongan ≤256 token bervektor **identik tepat
  1,0**. Jendela hanya menyentuh yang memang terpotong, sesuai harapan.

> **Koreksi kriteria yang dicatat terbuka.** Rumusan awal H2 memakai kemiripan
> **maks** < 0,9999, yaitu menuntut *setiap* potongan terpotong berubah. Itu salah
> rancang dan menghasilkan GAGAL palsu: potongan 257 token hanya kehilangan satu
> token sehingga vektornya wajar nyaris tidak bergerak (0,999966). Dinilai ulang
> dengan **rerata**. Koreksi ini menyangkut **validasi alat, bukan arah hasil**;
> kedua rumusan tetap tercatat di berkas keluaran.

### Temuan: menghapus pemotongan TIDAK menaikkan MRR

Selisih **W512 − W256 = −0,023**. Menghapus pemotongan justru **sedikit
menurunkan** MRR, bukan menaikkannya.

Penjelasan yang paling sesuai: `all-MiniLM-L6-v2` **disetel** (fine-tuned) pada
jendela 256. Posisi 257–512 memang ada pada BERT di bawahnya, tetapi tidak pernah
ikut dalam penyetelan *sentence embedding*. Memakainya menghasilkan representasi
yang kurang terkalibrasi, dan kerugian itu sedikit melebihi keuntungan dari
memulihkan 8,23% token.

Perhatikan pula **Hit@1 identik 29,2% pada ketiga varian** — pengaruhnya ada di
peringkat 2–5, bukan di peringkat teratas.

### Akibat bagi naskah — ini yang penting

Temuan ini **memisahkan dua hal yang selama ini tercampur**:

1. **Integritas korpus** — 8,23% korpus tidak tervektor. Ini cacat nyata dan
   berdiri sendiri: isi yang tidak pernah dapat terambil tetap tidak dapat
   terambil, betapa pun relevannya.
2. **Kinerja retrieval** — ternyata **bukan** disebabkan pemotongan itu. Menaikkan
   jendela pada model yang sama tidak memperbaikinya.

Artinya langit-langit MRR ~0,5 punya sebab lain, dan mengejar pemotongan tidak
akan menembusnya. Menaikkan `max_seq_length` pada model yang sama **tidak
diadopsi**. Bila integritas korpus hendak diperbaiki, jalannya adalah model dengan
jendela yang memang **dilatih** untuk teks panjang — bukan menarik paksa jendela
model yang disetel pada 256.

## 9. T9 — apakah ketidakcocokan BAHASA penyebabnya?

Sumber: `results/eval_rag/model_embedding.json`, skrip `scripts/eval_model_embedding.py`.

T8 menyingkirkan pemotongan sebagai penyebab, sehingga dugaan berikutnya diuji:
`all-MiniLM-L6-v2` adalah model **bahasa Inggris**, sedangkan korpusnya berbahasa
Indonesia. Pembanding `paraphrase-multilingual-MiniLM-L12-v2` sudah tersedia luring,
berdimensi sama (384), dan **berjendela 128 — lebih kecil** daripada 256 milik
kontrol. Rancangan ini **sengaja memberatkan pihak yang didugakan menang**: bila
model multibahasa tetap unggul meski terpotong jauh lebih parah, bahasa terbukti
lebih menentukan daripada jendela.

Potongan identik di semua varian (900/120, pemisah lama), sehingga label tidak
bergeser dan perbandingannya sah.

| Varian | Jendela | MRR | Hit@1 | Token terbuang | Dim |
|---|---:|---:|---:|---:|---:|
| **EN256** kontrol | 256 | 0,492 | 29,2% | 8,23% | 384 |
| **ML128** multibahasa | 128 | 0,463 | **30,0%** | **39,89%** | 384 |
| **ML256** multibahasa | 256 | **0,506** | **30,0%** | 2,30% | 384 |

H1 LOLOS (ML128 memang terbuang lebih banyak), H3 LOLOS (dimensi sama 384, jadi
perbedaan hasil tidak dapat dijelaskan oleh dimensi).

**Selisih ML256 − EN256 = +0,014 — di bawah ambang 0,02 yang ditetapkan di muka.
Dugaan ketidakcocokan bahasa TIDAK didukung.**

Model multibahasa memang sedikit lebih baik (dan Hit@1-nya konsisten lebih tinggi,
30,0% lawan 29,2%), tetapi tidak cukup untuk menyimpulkan bahasa sebagai penyebab.

### Bukti tambahan yang memperkuat T8

Perhatikan **ML128: membuang 39,89% token — hampir lima kali lipat kontrol — namun
MRR-nya hanya turun 0,029.** Ini menguatkan kesimpulan T8 dari arah yang berbeda:
sistem retrieval ini **sangat tidak peka terhadap pemotongan**. Kehilangan empat
dari sepuluh token korpus hampir tidak menggerakkan metriknya.

## 10. Kesimpulan gabungan T7–T9: penyebabnya bukan yang dikira

Tiga percobaan berpisah jalan menuju satu kesimpulan yang sama:

| Dugaan | Diuji di | Hasil |
|---|---|---|
| Potongan berhenti di tengah kalimat | T7 | **Nyata**, dan sudah diperbaiki (19,7% → 83,5%) |
| Pemotongan senyap 256 token | T8 | **Bukan** penyebab; menghapusnya justru −0,023 |
| Ketidakcocokan bahasa model | T9 | **Tidak didukung**; +0,014, di bawah ambang |

Langit-langit MRR ~0,5 karena itu **tidak berasal dari sisi representasi sama
sekali**. Kandidat yang tersisa, berurut menurut kekuatan bukti:

1. **Alat ukurnya sendiri.** Sudah terbukti berbias panjang dan melabeli isi yang
   benar sebagai `lain` (Bagian 7). Selama label kebenaran diturunkan dari teks
   potongan, seluruh angka MRR di proyek ini menaksir sesuatu yang bergerak.
2. **Susunan kueri.** Kueri disintesis dari angka glukosa lewat
   `build_ablation_query()`; keragamannya sempit dan mungkin tidak menuntut
   pembedaan yang halus.
3. **`lambda_mult` 0,0.** MMR pada nilai itu memaksimalkan KERAGAMAN dan nyaris
   mengabaikan relevansi dalam pengurutan. Ini pilihan yang sudah disetel, tetapi
   interaksinya dengan metrik berbasis label topik belum pernah diperiksa.

**Implikasi bagi Bab IV:** perbaikan chunking layak dinarasikan sebagai perbaikan
**integritas korpus dan keterbacaan sitasi** — keduanya terukur dan tercapai.
Ia **tidak boleh** dinarasikan sebagai perbaikan kinerja retrieval, karena tiga
percobaan menunjukkan kinerja tidak bergerak di sana.

## 11. T10 — embedding terkelola TIDAK LAYAK pada free tier

Sumber: `results/eval_rag/embedding_gemini_TIDAK_LAYAK.json`,
skrip `scripts/eval_embedding_gemini.py`.

Percobaan ini **tidak menghasilkan satu pun angka mutu retrieval**, dan itu
sendiri temuannya. Dua batas kuota ditemui berurutan:

| Batas | Nilai | Hasil |
|---|---:|---|
| Per menit | 100 | **Teratasi** dengan pembatas laju sisi klien (`_EmbeddingsBerlaju`) |
| **Per hari** | **1.000** | **Menghalangi** — tidak ada penyelesaian pada free tier |

**Aritmetika yang menentukan:** korpus 2.061 potongan menuntut 2.061 permintaan
untuk **sekali** indeks, sedangkan jatahnya 1.000 per hari per model. Sekali
indeks karena itu memerlukan **tiga hari**, ditambah 480 permintaan per sekali
evaluasi. Kuota bersifat per-model, sehingga berpindah ke `gemini-embedding-001`
memberi jatah 1.000/hari tersendiri — tetap di bawah 2.061, jadi **tidak
menyelesaikannya**. Batas menghitung **konten**, bukan permintaan batch, sehingga
`batchEmbedContents` mempercepat tetapi tidak menghemat kuota.

**Mengapa ini menghalangi penelitian, bukan sekadar merepotkan.** Setiap varian
chunking menuntut indeks ulang. Dengan tiga hari per lengan, protokol Bagian C
— yang mensyaratkan seluruh varian diukur termasuk yang kalah — tidak dapat
dijalankan sama sekali.

**Nilai bagi naskah.** Ini temuan **kelayakan penerapan**, dan sejalan dengan
konteks penelitian ini: sistem ditujukan bagi layanan kesehatan bersumber daya
terbatas. Embedding terkelola berjendela besar memang menghapus pemotongan, tetapi
menukarnya dengan ketergantungan kuota pihak ketiga yang membuat korpus **tidak
dapat diindeks ulang secara mandiri**. Layak masuk Bab keterbatasan sebagai
**alasan terukur memilih embedding lokal**, bukan sebagai kegagalan percobaan.

**Yang TIDAK boleh disimpulkan.** Mutu retrieval `gemini-embedding-2` **tetap
tidak diketahui**. Yang terukur hanya kelayakan penerapannya.

## 12. Ringkasan keputusan seluruh T7–T10

| Perubahan | Status | Dasar |
|---|---|---|
| Kutipan UI berhenti di batas kalimat | **Diadopsi** | Gao dkk. §V.A.1; tidak bergantung indeks maupun MRR |
| Pemisah kalimat saat indexing | **Layak diadopsi** | 19,7% → 83,5% berakhir utuh; MRR tak bergerak nyata |
| Penakar token (V3) | **Ditangguhkan** | Metrik terkonfound, tidak dapat memutuskan |
| Menaikkan `max_seq_length` ke 512 | **Tidak diadopsi** | T8: MRR −0,023; model disetel pada 256 |
| Pindah ke model multibahasa (ML256) | **Kandidat kuat** | T9: MRR 0,506 vs 0,492 dan token terbuang 8,23% → 2,30%, luring, tanpa kuota |
| Embedding terkelola Gemini | **Tidak layak** | T10: 1.000/hari lawan kebutuhan 2.061 |

Catatan atas ML256: dugaan bahasa **tidak didukung** menurut ambang yang
ditetapkan di muka (+0,014 < 0,02). Namun sebagai **pilihan produksi** ia
mengungguli EN256 pada integritas korpus tanpa merugikan MRR — bentuk penalaran
yang sama dengan Aturan 3 prapendaftaran T7, dan harus dinyatakan sebagai alasan
integritas, **bukan** sebagai kemenangan hipotesis bahasa.

## 13. Berkas yang disentuh

| Berkas | Perubahan |
|---|---|
| `src/rag/citations.py` | pemotong batas kalimat + penanda tepi |
| `src/rag/knowledge_base.py` | `PEMISAH_KALIMAT`, `_penakar_token()`, argumen baru |
| `scripts/reingest_kb.py` | `--pemisah-kalimat`, `--satuan-panjang` |
| `app/streamlit_app.py` | dua keterangan pemotongan yang dipisah |
| `tests/test_citations_snippet.py` | kontrak batas kalimat (7 uji) |
| `tests/test_chunking_kalimat.py` | **baru** — kontrak chunker (5 uji) |
| `scripts/eval_strategi_chunking.py` | **baru** — pembanding empat varian |
| `scripts/diagnosa_metrik_chunking.py` | **baru** — diagnosis post-hoc alat ukur |
| `docs/PRAPENDAFTARAN_T7_CHUNKING.md` | **baru** |

`config.yaml` **tidak disentuh**; keputusan adopsi menunggu hasil Bagian 6
prapendaftaran.

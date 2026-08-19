# Spesifikasi revisi gambar — Bab II dan Bab IV

Dokumen mandiri untuk pembuat gambar. **Tidak perlu membaca berkas lain.**

Tiga pekerjaan pada dokumen ini:

1. **Gambar II.3 dan II.4** pada Bab II — mengganti **satu label kotak** yang isinya
   bertentangan dengan keputusan produksi. Gambarnya tidak digambar ulang.
2. **Gambar IV.2** pada Bab IV, yaitu alur rinci sistem — **digambar ulang** karena
   pembimbing menilainya terlalu teknis dan kurang ilmiah.
3. Gambar pipeline yang dahulu menempati IV.2 **sudah dihapus dari naskah**, sebab
   isinya terbukti mengulang Gambar II.4 dan sisi daring alur rinci. Penomoran gambar
   Bab IV karena itu maju satu: alur rinci menjadi IV.2, diagram kelas IV.3, diagram
   sekuens IV.4, dan wireframe IV.5.

Gambar Bab IV lainnya, yakni arsitektur berlapis, diagram kelas, diagram sekuens, dan
wireframe, **tidak berubah isinya**.

---

## 0. Mengapa direvisi, dan apa yang sebenarnya salah

Penyebabnya sudah ditelusuri dengan membandingkan ke tiga laporan tugas akhir rekan
sepembimbing yang sudah lulus. Hasilnya penting, sebab **penyebabnya bukan jumlah kotak**.

Satu laporan rekan memuat diagram alur berisi **sekitar dua puluh kotak** dengan pemisahan
tahap luring dan tahap daring — susunan yang sama persis dengan Gambar IV.2 kita — dan
**diterima**. Diagram itu bahkan memakai label pada panah dan menyorot satu kotak sebagai
kontribusi, sama seperti punya kita.

Yang membedakan adalah **jenis labelnya**. Seluruh kotak pada diagram rekan itu diberi nama
**peran**, misalnya *Semantic Chunker*, *Metadata Enricher*, *Query Embedding*,
*Retrieval Evaluator*, dan *Response Generator*. Tidak ada satu pun nama berkas, nama fungsi,
nama kelas, angka waktu, maupun jumlah data di dalam gambarnya.

Laporan rekan kedua menempuh jalan yang lebih ringkas lagi, dengan diagram alur berisi hanya
empat sampai tujuh kotak berlabel peran seperti *AI Agent*, *Router*, dan *Synthesizer*.
Rincian teknisnya tidak dibuang, melainkan **dipindahkan ke tabel pendamping** berkolom
Lapisan, Komponen, dan Peran.

### Aturan yang berlaku pada kedua gambar ini

**DILARANG muncul di dalam gambar:**

| Yang dilarang | Contoh yang harus hilang |
|---|---|
| Nama berkas | `ohio_parser.py`, `conformal_h6.json`, `ohio_t1dm_merged.csv` |
| Nama fungsi | `predict`, `build_source_list`, `evaluate_divergence`, `engineer_features` |
| Nama kelas | `PatientState`, `MMRRetriever`, `ClinicalDecisionLog` |
| Nama pustaka atau model | `all-MiniLM-L6-v2`, `gemini-3.5-flash-lite`, ChromaDB, Okapi BM25 |
| Nama medan data | `risk_level`, `trend_direction`, `urgency`, `glucose_delta`, `iob`, `cob` |
| **Angka waktu** | 23,0 ms · 284,3 ms · 314,8 ms · 9,3 detik |
| **Angka jumlah** | 4.038 potongan · 494 halaman · 12 pasien · 500/67 |

Angka waktu dan angka jumlah sudah dimuat tabel di dalam naskah. Mengulanginya di gambar
menciptakan dua sumber kebenaran yang pasti berselisih pada revisi berikutnya.

**WAJIB dipertahankan**, sebab ketiganya justru ada pada diagram rekan yang diterima:

1. Pemisahan **luring** dan **daring** sebagai dua kelompok bernama (khusus Gambar IV.2).
2. **Label pada panah** yang menyebut **objek data** yang berpindah, bukan cara berpindahnya.
3. **Penandaan kotak kontribusi**, memakai satu penanda yang konsisten.

### Istilah yang dilarang karena sudah dicabut dari penelitian

| Dilarang | Gantinya |
|---|---|
| *digital twin*, *twin*, *DT* | kondisi klinis terstruktur |
| *what-if*, simulasi pengandaian | komponennya dicabut, jangan digambar |
| *Random Forest* sebagai model utama | prakiraan tidak perlu disebut nama modelnya |
| *MMR* atau pencarian vektor sebagai jalur produksi | penelusuran leksikal |
| *surrogate*, *tipe 2*, *T2DM* | penelitian ini pada diabetes tipe 1 |

---

## 1. `Gambar_II4_AlurRAGTerkondisiPrediksi.png` — perbaikan satu kotak

Tercetak **Gambar II.4**, pada Bab II Studi Literatur. **Gambarnya sudah baik dan tidak perlu
digambar ulang**; hanya **satu kotak** yang perlu diganti labelnya.

### 1.1 Mengapa gambar ini justru menjadi acuan

Gambar II.4 yang ada sekarang **sudah memenuhi seluruh aturan gaya** pada Bagian 0: labelnya
berupa peran (*Feature Engineering*, *Prediction Model*, *Condition Classification*,
*Query Construction*, *LLM Generation*), panahnya berlabel objek data (*feature vector*,
*predicted level and interval*, *structured condition*, *query*, *top-k*, *context*), dan satu
kotak kontribusi disorot. Tidak ada nama berkas, nama fungsi, angka waktu, maupun jumlah data.

Pembuat gambar dianjurkan **memakai gambar ini sebagai contoh gaya** ketika menggambar
Gambar IV.2.

### 1.2 Satu kotak yang harus diganti

| Kotak | Label sekarang | Label pengganti |
|---|---|---|
| Kotak keenam, sesudah *Query Construction* | **Semantic Search** | **Guideline Retrieval** |

**Alasannya menyangkut kebenaran isi, bukan gaya.** Jalur produksi penelitian ini memakai
**penelusuran leksikal**, bukan pencarian berbasis makna. Bab IV dan Bab VI justru
memaparkan panjang lebar mengapa penelusuran padat berbasis makna **dicabut** dari produksi.
Pembaca yang membaca berurutan akan menemukan Bab II menjanjikan *semantic search*, lalu Bab
IV mengatakan yang dipakai kebalikannya.

Label penggantinya sengaja **netral terhadap cara menelusur**, sebab Bab II memang belum
sampai pada keputusan itu; keputusannya baru diambil pada Bab IV atas dasar pengukuran Bab VI.

Kotak sumbernya, yang sekarang berlabel *Guideline Corpus*, **tetap** dan tidak berubah.

### 1.3 Periksa juga pasangannya

`Gambar_II3_AlurRAGKonvensional.png` (tercetak **Gambar II.3**) adalah pasangan konvensional
gambar ini. Bila ia juga memuat kotak *Semantic Search*, kotak itu diganti dengan label yang
sama, yaitu **Guideline Retrieval**, agar kedua gambar tetap sebanding.

---

## 2. `Gambar_IV2_AlurRinciSistem.png` — Alur Rinci Sistem

Tercetak **Gambar IV.2**. Gambar terbesar di Bab IV, dan yang paling banyak berubah.

**Keterangan gambar di naskah:** *"Alur rinci sistem, memisahkan tahap luring yang dijalankan
sekali di luar aplikasi dari tahap daring yang dijalankan pada tiap konsultasi."*

### 2.1 Kelompok LURING — dijalankan sekali, di luar aplikasi

| Kode | Label pada kotak | Menuju |
|---|---|---|
| A | Rekaman pemantauan glukosa | B |
| B | Penyelarasan waktu kejadian | C |
| C | Deret waktu terpadu | D |
| D | Rekayasa fitur fisiologis | E, E2, E3 |
| E | Pelatihan model prakiraan | — |
| E2 | Kalibrasi interval ketidakpastian | — |
| E3 ★ | **Pelatihan pengklasifikasi kondisi** | — |
| F | Korpus pedoman klinis | G |
| G | Pemecahan pada batas kalimat | H |
| H | Basis pengetahuan terindeks | — |

### 2.2 Kelompok DARING — dijalankan tiap konsultasi

| Kode | Label pada kotak | Menuju |
|---|---|---|
| I | Masukan logbook oleh dokter | J |
| J | Pembentukan jendela dan fitur | K, M |
| K | Prakiraan glukosa | L, N, O |
| L | Interval ketidakpastian terkalibrasi | N, U |
| M ★ | **Penentuan kondisi terprediksi** | O |
| N | Deteksi divergensi | U |
| O ★ | **Perakitan kondisi klinis terstruktur** | P |
| P ★ | **Transformasi kueri terkondisi-prediksi** | Q |
| Q | Penelusuran leksikal | R |
| R | Pembangkitan rekomendasi | S |
| S | Penjaminan penyangkalan | T |
| T ★ | **Resolusi sitasi dari metadata** | U |
| U | Layar konsultasi dokter | V |
| V | Pencatatan keputusan dokter | — |

### 2.3 Tentang kotak bertanda — penting, jangan sampai keliru

**Yang berjumlah empat adalah TAHAPnya, bukan kotaknya.** Empat tahap kontribusi itu:

1. Pengklasifikasi kondisi
2. Perakitan kondisi klinis terstruktur
3. Transformasi kueri terkondisi-prediksi
4. Resolusi sitasi dari metadata

Pengklasifikasi kondisi muncul **dua kali**, yaitu **E3** di sisi luring untuk pelatihannya
dan **M** di sisi daring untuk pemakaiannya. Keduanya satu komponen yang sama, dan
**keduanya ditandai**. Jadi gambar ini memuat **lima kotak bertanda untuk empat tahap**, dan
itu memang benar. Keterangan gambar di naskah sudah menjelaskannya.

### 2.4 Label pada panah — sebutkan objek data yang berpindah

Panah tidak dibiarkan telanjang. Tiap panah penting diberi label berisi **apa** yang
berpindah, bukan bagaimana ia berpindah.

| Panah | Label |
|---|---|
| C → D | deret waktu |
| D → E, E2, E3 | jendela berlabel |
| E → K | model terlatih |
| E2 → L | faktor kalibrasi |
| E3 → M | pengklasifikasi terlatih |
| H → Q | indeks potongan |
| J → K, J → M | vektor fitur |
| K → L, K → N | nilai terprediksi |
| M → O | kondisi terprediksi |
| O → P | kondisi klinis terstruktur |
| P → Q | kueri terkondisi |
| Q → R | lima potongan teratas |
| R → S | teks rekomendasi |
| T → U | sumber beserta nomor halaman |

Empat panah **E → K**, **E2 → L**, **E3 → M**, dan **H → Q** menyeberang dari kelompok luring
ke kelompok daring. Keempatnya menandai **artefak yang dipakai ulang**, bukan aliran data
dalam satu jalan, sehingga digambar berbeda dari panah biasa, misalnya putus-putus.

### 2.5 Tiga hal yang harus terbaca dari gambar

Ketiganya dapat dibaca **tanpa satu pun angka** di dalam gambar.

1. Kotak **O** memakai **nilai terprediksi**, bukan nilai yang sedang berlaku. Bila tahap ini
   memakai nilai terkini, seluruh sifat antisipatif sistem batal.
2. Panah **Q → R** membawa potongan **tanpa nomor halaman**; nomor halaman baru muncul pada
   panah **T → U**. Model bahasa karena itu tidak mungkin mengarangnya.
3. **Tidak ada** panah dari kotak **L** menuju kotak **P**. Interval ketidakpastian tidak
   pernah ikut membentuk kueri.

---

## 3. Daftar periksa sebelum menyerahkan gambar

- [ ] Nol nama berkas, nama fungsi, nama kelas, nama pustaka, dan nama medan data
- [ ] Nol angka waktu dan nol angka jumlah
- [ ] Nol istilah dari daftar yang dicabut pada Bagian 0
- [ ] Gambar IV.2 memisahkan luring dan daring sebagai dua kelompok bernama
- [ ] Tiap panah penting berlabel objek data yang berpindah
- [ ] Gambar II.3 dan II.4 tidak lagi memuat kotak *Semantic Search*
- [ ] Gambar IV.2 memuat lima kotak bertanda untuk empat tahap, sesuai Subbab 2.3
- [ ] Tidak ada panah dari kotak interval ketidakpastian menuju kotak pembentuk kueri, pada
      kedua gambar
- [ ] Nama berkas keluaran persis: `Gambar_II3_AlurRAGKonvensional.png`,
      `Gambar_II4_AlurRAGTerkondisiPrediksi.png`, dan `Gambar_IV2_AlurRinciSistem.png`

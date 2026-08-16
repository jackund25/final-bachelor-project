# Pelajaran dari tiga TA rekan sepembimbing yang sudah lulus

Disusun 17 Agustus 2026 dari pembacaan tiga laporan TA berpembimbing sama, ketiganya
berkonteks UKM dan berbasis RAG. Tujuannya dua: **mengalibrasi ekspektasi terhadap
besaran hasil**, dan **mengenali pola penulisan yang diterima pembimbing**.

Catatan etika: yang diambil di sini adalah **metode yang sudah baku dan dipublikasikan**
(Parent-Child RAG, RAGAS, CRAG, hybrid retrieval, LLM-as-Judge), bukan teks, rancangan,
atau data milik rekan. Menyebut bahwa suatu opsi diketahui dari karya rekan adalah
praktik yang benar dan justru dianjurkan.

---

## 1. Kalibrasi: seperti apa angka TA yang LULUS

### Nicolas — Parent-Child RAG lawan Regular RAG, n=30 kasus

| Metrik | Parent-Child | Regular | Selisih |
|---|---:|---:|---:|
| Faithfulness | 0,9917 | 0,9072 | +0,0845 |
| **Context Precision** | 0,5969 | 0,6141 | **−0,0172** |
| Context Recall | 0,6333 | 0,4222 | +0,2111 |
| Answer Relevancy | **0,4662** | 0,1808 | +0,2854 |
| Answer Correctness | 0,6172 | 0,4074 | +0,2098 |

Yang penting diperhatikan, dan semuanya **diterima pembimbing**:

- **n = 30 kasus saja**, satu kali jalan, **tanpa uji statistik apa pun** — tidak ada
  nilai-p, tidak ada selang kepercayaan, tidak ada variasi antar-*fold*.
- **Metodenya KALAH pada satu metrik** (context precision −2,80%), dan itu ditulis
  terus terang.
- **Answer relevancy 0,4662 — di bawah 0,5**, dan penulisnya menyebutnya sendiri:
  *"nilai absolut answer relevancy ... masih berada di bawah 0,5"*.
- Kesimpulannya ditulis **berhedge**: *"hasil penelitian tidak menunjukkan bahwa
  Parent-Child RAG selalu lebih baik pada setiap aspek retrieval"*.

### Gabriel — studi ablasi tiga varian, 270 kueri

Paling ketat dari ketiganya, dan paling dekat dengan cara kerja TA ini:

- Tiga varian dibandingkan: Naive RAG, SRAG-only, CRAG-only
- **270 kueri sintetis**, tiga domain, n=90 per sel
- **Hipotesis dinyatakan di muka** (IV.7.1 Hipotesis Penelitian)
- Uji statistik **beserta ukuran efek Cohen's d**: *"H1 terkonfirmasi ... p<0,001, d=0,4..."*
- Ada subbab tersendiri **VI.1.3 Prosedur Uji Statistik** dan
  **Lampiran A.3 Hasil Lengkap Uji Statistik**
- Metodologi **DSRM (Peffers dkk. 2007)** — sama dengan TA ini

### Reffy — hybrid retrieval

- Judulnya sendiri memuat *hybrid retrieval*
- Definisinya **berbeda** dari fusi BM25+vektor: yang dimaksud adalah **pemisahan jalur**
  antara data tabular (Supabase, untuk angka presisi) dan data naratif (Qdrant, untuk
  semantik)
- Pemilihan arsitektur dilakukan lewat **weighted scoring** atas tiga alternatif
  (nilai akhir tertinggi 3,65)

---

## 2. Bagaimana posisi TA ini dibandingkan ketiganya

| | Nicolas | Gabriel | Reffy | **TA ini** |
|---|---|---|---|---|
| Ukuran sampel | 30 kasus | 270 kueri | — | **240 kasus × 6 fold** |
| Validasi silang | tidak | tidak | tidak | **6 fold lintas-pasien** |
| Uji statistik | tidak ada | p + Cohen's d | tidak | **Wilcoxon p=0,031** |
| Hipotesis di muka | tidak | ya | tidak | **ya, prapendaftaran per percobaan** |
| Kontrol batas atas | tidak | tidak | tidak | **ya (`oracle`)** |
| Hasil negatif dilaporkan | sebagian | ya | — | **ya (T8, T9, T10)** |

**Kesimpulan kalibrasi: kekakuan metodologis TA ini sudah di atas ketiganya.** Yang
belum setara adalah **keragaman alat ukur** — ketiganya memakai RAGAS, TA ini masih
bertumpu pada `classify_chunk`.

---

## 3. Tiga teknik yang layak diadopsi

### 3.1 RAGAS context precision / context recall — PRIORITAS TERTINGGI

Menjawab langsung masalah κ = 0,2505, dan **sudah diterima pembimbing** pada dua TA.

Alasannya kuat dan sudah tertulis di berkas proyek ini sendiri
(`results/ragas/caveat.md`): *"context_recall/precision (berbasis reference) lebih
objektif daripada faithfulness/answer_relevancy"*. Keduanya **tidak bergantung sama
sekali** pada `classify_chunk`, sehingga memberi alat ukur kedua yang bebas dari
kelemahan yang sudah terbukti.

Keadaan sekarang: RAGAS proyek ini baru n=10 dan **hanya faithfulness** — justru metrik
yang paling tidak stabil (selisih median 0,18 antar-jalan identik). Nicolas memakai
lima metrik pada n=30.

### 3.2 Ground truth diturunkan dari basis pengetahuan (Gabriel)

Gabriel memakai *"kueri sintetis per domain beserta ground truth yang diturunkan dari
knowledge base"*. Artinya kebenaran diketahui **secara konstruksi**, bukan dari
pelabelan manusia yang lemah kesepakatannya.

Ini jalan kedua untuk melewati κ = 0,2505 tanpa menunggu penilai manusia.

### 3.3 Parent-Child retrieval (Nicolas)

Menjawab langsung ketegangan yang ditemui pada T7: potongan kecil bagus untuk
penelusuran, potongan besar bagus untuk konteks. Parent-Child mengambil *child* yang
kecil lalu mengembalikan *parent* yang luas.

Bukti Nicolas: context recall **+0,2111 (+50%)**, dengan biaya context precision
−0,0172. Pertukaran yang jelas dan dapat dipertahankan.

---

## 4. Pola penulisan yang diterima pembimbing

Diamati konsisten pada ketiganya:

1. **Sebut angka, lalu jelaskan mekanismenya, lalu sebut batasnya** — dalam satu alinea.
   Contoh Nicolas: melaporkan +0,2111, menjelaskan sebabnya (*parent context* lebih
   lengkap), lalu menyebut nilai absolutnya masih di bawah 0,5.
2. **Tabel "Selisih Absolut" dan "Perubahan Relatif" terpisah**, disertai peringatan
   bila persentase relatif menyesatkan: Nicolas menulis +157,85% lalu langsung
   mengingatkan *"perlu ditafsirkan secara hati-hati karena nilai awal ... relatif
   rendah"*.
3. **Kesimpulan berhedge, bukan absolut.** *"tidak menunjukkan bahwa X selalu lebih
   baik pada setiap aspek"*.
4. **Subbab "Keterbatasan Evaluasi" tersendiri**, bukan diselipkan.
5. **Pemilihan alternatif dijustifikasi lewat skor berbobot** (Nicolas: Weighted Sum
   Model; Reffy: weighted scoring) — pembimbing tampaknya menghargai keputusan desain
   yang dapat ditelusuri, bukan sekadar dinyatakan.
6. **Studi ablasi sebagai tulang punggung Bab VI** pada ketiganya.

---

## 5. Yang TIDAK perlu ditiru

- **n8n dan Qdrant.** Ketiganya memakainya, tetapi itu pilihan platform, bukan
  kontribusi. Chroma pada TA ini sudah memadai dan berjalan luring, yang justru sejalan
  dengan batasan desain tanpa GPU.
- **Arsitektur agentic.** Menambah lapisan agen tidak menjawab satu pun masalah yang
  terukur pada TA ini.

---

## 6. Urutan yang disarankan

1. **RAGAS context precision + context recall** pada perbandingan `standard` lawan
   terkondisi. Alat ukur kedua yang bebas dari `classify_chunk`, dan sudah terbukti
   diterima pembimbing.
2. **Cohen's d** di samping Wilcoxon, mengikuti Gabriel. Murah, dan menjawab
   "seberapa besar" bukan sekadar "apakah nyata".
3. **Parent-Child retrieval** bila masih ada waktu sesudah T13.

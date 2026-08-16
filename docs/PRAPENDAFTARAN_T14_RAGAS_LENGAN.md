# Prapendaftaran T14 — validasi lengan penelusuran dengan alat ukur bebas kata kunci

> **Ditulis 17 Agustus 2026, SEBELUM satu pun skor RAGAS per lengan dilihat.**
> Diverifikasi saat penulisan: `results/ragas/` hanya memuat hasil `faithfulness`
> n=10 atas koleksi terkontrol; belum ada satu pun skor per lengan penelusuran.

---

## 1. Mengapa percobaan ini ada

T13 menemukan hibrida RRF mengungguli vektor pada kasus divergen (selisih kontribusi
+0,1423 → +0,1895, positif 6/6 fold). **Temuan itu tidak dapat diklaim**, karena
keberatan yang ditemukan sesudahnya:

> `classify_chunk` melabeli potongan menurut **kata kunci**
> (`hipoglikemi`, `gula darah rendah`, `15-15`), dan `build_ablation_query()` menyusun
> kueri yang memuat kata-kata itu **harfiah**. BM25 mencocokkan token persis. Retriever
> dan alat ukur karena itu **mengoptimalkan sinyal yang sama**.

T14 menguji temuan itu dengan alat ukur yang **tidak berbasis kata kunci**.

Dasar pemilihan metrik ada pada berkas proyek ini sendiri
(`results/ragas/caveat.md`): *"context_recall/precision (berbasis reference) lebih
objektif daripada faithfulness/answer_relevancy"*. Dua TA sepembimbing memakai
keduanya dan diterima.

---

## 2. Rancangan

| | Nilai |
|---|---|
| Kasus | 10 kasus positif `evaluation/ragas_dataset.json`, `jawaban_acuan` terverifikasi manual |
| Korpus | **produksi** (2.248 potongan) — BUKAN koleksi terkontrol 17 potongan |
| Lengan | `vektor` (MMR, produksi) dan `hibrida` (RRF vektor+BM25) |
| Metrik | `context_precision`, `context_recall` |
| Penilai | model yang sama dengan `scripts/run_ragas.py` |

**Tahap pembangkitan jawaban DILEWATI.** Kedua metrik hanya memerlukan pertanyaan,
konteks terambil, dan jawaban acuan. Melewatinya menghemat kuota sekaligus
menghilangkan satu sumber variasi yang tidak relevan bagi pertanyaan penelusuran.

### PENYIMPANGAN DARI ARAHAN PEMBIMBING — dinyatakan terbuka

> **Koleksi RAGAS terkontrol (2–4 halaman, 17 potongan) adalah rancangan yang
> DISENGAJA dan diarahkan pembimbing**, dengan tujuan menjaga kuota Gemini tidak
> habis. `scripts/run_ragas.py` mencatat dua alasannya: *"ground truth dapat
> diverifikasi manual sampai ke kalimat sumbernya"* dan *"jumlah panggilan LLM
> penilai terkendali dan dapat diperkirakan di muka"*.
>
> **Rumusan awal prapendaftaran ini menyebut koleksi itu tidak bermakna. Itu keliru
> dan dicabut.** Koleksi kecil bukan kelemahan; ia instrumen yang tepat untuk
> pertanyaan yang menjadi tujuannya.

Kedua koleksi menjawab **pertanyaan yang berbeda**:

| Pertanyaan | Koleksi yang tepat | Alasan |
|---|---|---|
| Seberapa baik mutu RAG sistem ini? | **terkontrol, 17 potongan** | ground truth terverifikasi sampai kalimat; kuota terkendali |
| Retriever mana yang lebih baik? | **produksi, 2.248 potongan** | lihat di bawah |

T14 menjawab pertanyaan **kedua**, dan hanya untuk itulah korpus produksi dipakai.
Alasannya bersifat mengukur, bukan preferensi: metadata dataset itu sendiri mencatat
`porsi_koleksi_terambil_per_kueri_persen` = **29,4**. Dengan `top_k` 5 atas 17
potongan, setiap retriever mengambil hampir sepertiga koleksi yang sama, sehingga
perbandingan antar-retriever tidak dapat membedakan apa pun — bukan karena keduanya
setara, melainkan karena koleksinya terlalu kecil untuk memisahkannya.

**Tujuan arahan pembimbing tetap dipatuhi.** Kuota bukan hanya tidak dilanggar,
melainkan ditekan: 120 panggilan dari jatah 500 per hari, dan tahap pembangkitan
jawaban dilewati seluruhnya.

**Wajib disampaikan saat bimbingan berikutnya**: bahwa T14 memakai korpus produksi,
mengapa koleksi terkontrol tidak dapat menjawab pertanyaan ini, dan bahwa koleksi
terkontrol TETAP dipakai untuk seluruh evaluasi RAGAS lainnya.

---

## 3. Dugaan yang didaftarkan di muka

- **D1 — arah.** Hibrida diduga **menaikkan `context_recall`** terhadap vektor, karena
  pencocokan leksikal menjangkau istilah klinis persis yang tidak tertangkap model
  embedding berbahasa Inggris. Ambang ditetapkan di muka: kenaikan **> 0,05** dianggap
  nyata pada n=10.
- **D2 — pertukaran.** `context_precision` diduga **tidak naik**, dan boleh turun.
  Pola ini sejalan dengan temuan Nicolas pada Parent-Child RAG (context recall +0,2111,
  context precision −0,0172): memperluas jangkauan biasanya menukar ketepatan.
- **D3 — pembantah.** Bila hibrida **tidak** unggul pada kedua metrik, maka keunggulan
  yang terlihat di T13 **dinyatakan sebagai artefak alat ukur**, dan hibrida TIDAK
  diadopsi. Ini hasil yang sepenuhnya dapat diterima dan wajib dilaporkan apa adanya.

---

## 4. Aturan penafsiran yang mengikat

1. **n = 10 terlalu kecil untuk uji signifikansi.** Tidak ada nilai-p yang akan
   dihitung. Yang dilaporkan selisih dan arahnya, disertai pernyataan bahwa ukuran
   sampelnya kecil.
2. **Skor RAGAS bervariasi antar-jalan.** Proyek ini sudah mengukurnya sendiri:
   `faithfulness` berselisih median 0,18 antar dua jalan identik. Karena itu selisih
   di bawah 0,05 **tidak boleh** ditafsirkan sebagai perbedaan.
3. **T14 tidak menyelesaikan K1 (κ = 0,2505) maupun K12 (jangkauan 1,16%).** Ia hanya
   memberi alat ukur kedua yang tidak berbagi sinyal dengan retriever.
4. **Bila kuota habis di tengah jalan**, hasil parsial disimpan dengan penanda dan
   TIDAK dilaporkan sebagai hasil lengkap.

---

## 5. Keluaran

`results/ragas/lengan_penelusuran.json`, memuat ketiga lengan termasuk yang kalah.

---

## 6. AMANDEMEN — 17 Agustus 2026, ditulis SEBELUM jalan ulang dimulai

> Diverifikasi saat penulisan: `results/ragas/lengan_penelusuran.json` masih memuat
> hasil jalan pertama, dan **belum ada satu pun skor untuk lengan `bm25`**. Amandemen
> ini karena itu tidak dapat disusun mengikuti hasil yang hendak diaturnya.

### 6.1 Mengapa jalan pertama dibatalkan

Jalan pertama T14 memakai **implementasi ulang** fusi RRF di dalam skrip evaluasi,
bukan kode produksi. Implementasi itu **cacat**:

> Ia memanggil `r.retrieve(query, top_k=50)`, padahal `retrieve()` dibatasi
> `self.fetch_k = 12`, sehingga MMR hanya mengembalikan **12 dokumen, bukan 50**.

Yang dinilai jalan pertama karena itu adalah fusi **12 lawan 50** — daftar padat pendek
digabung dengan daftar leksikal panjang, sehingga berat sebelah ke BM25. Produksi
menggabungkan **50 lawan 50**. Cacat yang sama ada pada skrip crossfold T13 dan sudah
diperbaiki di sana; besarannya terukur: cara lama 12 dokumen, cara produksi 50.

**Akibatnya angka jalan pertama menggambarkan konfigurasi yang tidak pernah dijalankan
siapa pun.** Angka itu dicabut, bukan direvisi. Berkas hasilnya dipertahankan sebagai
arsip agar cacatnya dapat ditelusuri, dan **tidak boleh masuk naskah**.

Sejak amandemen ini, tiap lengan dibangun dengan `MMRRetriever(retrieval_mode=...)`
yang modenya **dipatok**, dan skrip **berhenti dengan galat** bila mode aktif tidak sama
dengan mode yang diminta. Tidak ada lagi jalur penelusuran yang ditulis dua kali.

### 6.2 Lengan `bm25` ditambahkan

Crossfold atas kode produksi (`f59667f`) menunjukkan BM25 sendirian meraih **selisih
kontribusi terbesar**: +0,1742 lawan hibrida +0,1583 dan vektor +0,1423. **Aturan 4
prapendaftaran T13** menyatakan bila BM25 sendirian mengungguli hibrida, maka BM25 yang
diadopsi.

Aturan itu **belum dijalankan**, dan sengaja belum. Alasannya sama dengan alasan T14 ada:
angka itu diukur dengan `classify_chunk`, yang melabeli menurut kata kunci dan karena itu
**berbagi sinyal dengan BM25**. Menjalankan Aturan 4 atas alat ukur yang berpihak kepada
salah satu lengan berarti membiarkan cacat alat ukur menentukan arsitektur produksi.

Aturan 4 karena itu **dipindahkan adjudikasinya ke sini**, atas `context_recall` yang
berbasis rujukan.

### 6.3 Dugaan tambahan, didaftarkan di muka

- **D4 — Aturan 4.** Bila `context_recall` lengan `bm25` **melampaui** hibrida, maka
  keunggulan BM25 pada crossfold **bukan artefak kata kunci**, dan Aturan 4 T13
  dijalankan: **BM25 sendirian diadopsi** sebagai penelusuran produksi, meskipun itu
  berarti mencabut hibrida yang baru saja dipasang dan menulis ulang subbab teorinya.
- **D5 — pembantah D4.** Bila `context_recall` hibrida **setara atau melampaui** BM25,
  maka keunggulan BM25 pada crossfold **dinyatakan sebagai artefak alat ukur**, dan
  hibrida dipertahankan. Selisih di bawah **0,05** dihitung sebagai setara, mengikuti
  Aturan 2 di atas — dan pada keadaan setara, hibrida dipertahankan karena ia tidak
  membuang jalur padat, sehingga tidak bergantung pada satu cara pencocokan saja.
- **D6 — kemungkinan ketiga.** Bila **vektor** yang tertinggi, maka baik T13 maupun T14
  jalan pertama tidak dapat dipertahankan, dan seluruh keputusan penelusuran diulang
  dari awal. Kemungkinan ini kecil, tetapi didaftarkan supaya tidak dijelaskan sesudah
  terlihat.

**Yang mengikat:** keputusan diambil dari `context_recall`, ditetapkan **sebelum** hasil
dilihat, dan diberlakukan apa pun arahnya — termasuk bila ia mencabut pekerjaan yang
sudah dipasang di produksi.

### 6.4 Biaya

180 panggilan penilai (3 lengan × 10 kasus × 6), dari jatah 500 per hari. Tahap
pembangkitan jawaban tetap dilewati seluruhnya.

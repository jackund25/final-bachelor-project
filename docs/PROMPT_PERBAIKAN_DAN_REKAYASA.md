# PROMPT PERBAIKAN DAN REKAYASA

> Disimpan ke repo pada 6 Agustus 2026. Sebelumnya prompt ini hanya dilampirkan di
> percakapan dan sempat hilang dari konteks setelah pemadatan, sehingga Bagian B tidak
> dapat dikerjakan sampai dilampirkan ulang. Berkas ini menjaga agar itu tidak terulang.

Urutan: A1, A2, A3, A4, lalu B1, B2, B3, B4, B5.

Aturan kerja: satu tugas satu commit. Tulis ringkasan ke `docs/journey.md` setiap kali
selesai. Jalankan pytest sebelum dan sesudah tiap tugas.

---

## BAGIAN A — PERBAIKAN CACAT

A1 harus lebih dulu karena mengubah kueri, sehingga seluruh angka retrieval berubah dan
pengujian pada Bagian B harus berjalan di atas kueri yang sudah benar.

### A1 — Kebocoran `current_glucose` ke kueri

1. Ubah `_enhance_query()` agar menerima parameter eksplisit yang menentukan sumber
   kondisi, bukan mengambil `current_glucose` secara implisit.
2. Pada mode prediction-conditioned, tag kondisi harus diturunkan dari
   `predicted_glucose`. Pada mode standard, dari `current_glucose`. Kedua mode harus
   simetris dalam struktur kueri, hanya berbeda pada sumber angkanya.
3. Tag "stress tinggi" tetap berlaku pada kedua mode.
4. Cetak contoh kueri final untuk kedua mode pada satu skenario divergen yang sama, dan
   tampilkan berdampingan agar perbedaannya terlihat. Simpan ke `docs/journey.md`.
5. JALANKAN ULANG seluruh evaluasi retrieval dan ablasi. Angka lama tidak berlaku.
   Laporkan tabel perbandingan angka sebelum dan sesudah.

### A2 — Peringatan divergensi asimetris

Peringatan divergensi harus berbasis kategori dan simetris; pesannya menyesuaikan arah
perpindahan dan menyebut kedua kategori. Pindahkan logikanya keluar dari `app/` ke `src/`
(KNF-07). Sertakan tes untuk keenam perpindahan kategori.

### A3 — Interval dua horizon

Tampilkan kedua horizon dengan interval konformalnya masing-masing. Divergensi dievaluasi
pada keduanya, menyebutkan horizon mana yang memicu. Pastikan kalibrasi konformal h12 ada;
jalankan bila belum.

### A4 — Waktu tanggap ujung-ke-ujung

Instrumentasi latensi ujung-ke-ujung yang dipecah per tahap (rekayasa fitur, prediksi,
kalibrasi, penyusunan kueri, retrieval, generasi). Simpan ke `results/`. Buat
`scripts/benchmark_latency.py` yang melaporkan median dan p95. Laporkan proporsi LLM
terhadap komputasi lokal.

Setelah Bagian A selesai, laporkan tabel angka sebelum dan sesudah untuk seluruh metrik
retrieval.

---

## BAGIAN B — PERBAIKAN BERBASIS LITERATUR

Untuk setiap butir, LAPORKAN HASIL PENGUKURAN LEBIH DULU. Jangan langsung mengadopsi.
Saya yang memutuskan. Setelah tiap butir, laporkan hasilnya dan BERHENTI.

### B1 — Model embedding tidak cocok dengan korpus dwibahasa

Yang paling dicurigai sebagai penyebab Hit@1 rendah.

Masalah: sistem memakai `all-MiniLM-L6-v2` yang dilatih pada korpus berbahasa Inggris,
sementara korpus memuat empat dokumen berbahasa Indonesia (KB-01..KB-04) dan delapan
berbahasa Inggris, sedangkan kueri disusun dalam Bahasa Indonesia.

Dasar literatur: Reimers dan Gurevych (2020), "Making Monolingual Sentence Embeddings
Multilingual using Knowledge Distillation", EMNLP 2020, hlm. 4512-4525. Model monolingual
memetakan kalimat berbahasa lain ke wilayah ruang vektor yang berbeda meskipun maknanya
sama. Rujukan ini SUDAH ADA di Subbab II.6.1.

1. Periksa apakah `scripts/eval_embedding_alternative.py` sudah menguji model
   multilingual. Kalau sudah, laporkan hasilnya. Kalau belum, tambahkan.
2. Bandingkan `all-MiniLM-L6-v2` terhadap `paraphrase-multilingual-MiniLM-L12-v2`.
   Keduanya berdimensi 384, sehingga penggantian tidak mengubah struktur koleksi.
3. Laporkan Hit@1, Hit@3, dan MRR untuk kedua model, DIPECAH menurut bahasa dokumen
   sasaran. Kalau hipotesis benar, perbaikan terbesar muncul pada dokumen Bahasa Indonesia.
4. Laporkan pula perubahan waktu penyematan dan ukuran koleksi (KNF-01).

### B2 — Pemeringkatan ulang

Masalah: tidak ada pemeringkatan ulang sama sekali. Hit@1 33,3% menunjukkan dokumen
relevan sering ditemukan tetapi tidak menempati peringkat pertama.

Dasar literatur: Gao dkk. (2023), "Retrieval-Augmented Generation for Large Language
Models: A Survey". Rujukan ini SUDAH ADA di Subbab II.6.2.

1. Tambahkan pemeringkatan ulang dengan cross-encoder yang berjalan lokal di CPU.
   Kandidat: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Kalau ada varian multilingual yang
   lebih sesuai, laporkan pilihannya.
2. Terapkan pada kolam `fetch_k`, lalu ambil `top_k` setelah pemeringkatan ulang.
3. Laporkan Hit@1, MRR, dan tambahan waktu tanggap.
4. PENTING: terapkan pada KEDUA mode ablasi. Menerapkannya hanya pada satu mode membuat
   perbandingan tidak sah.

### B3 — Penelusuran hibrida

Masalah: penelusuran sepenuhnya bertumpu pada kemiripan vektor. Istilah klinis seperti
nama obat, angka ambang, dan singkatan sering lebih baik ditangani pencocokan leksikal.

Dasar literatur: Gao dkk. (2023) membahas penelusuran hibrida jarang + padat.

1. Tambahkan BM25 sebagai penelusur kedua, gabungkan dengan reciprocal rank fusion.
2. Laporkan perubahan metrik. Kalau perbaikannya kecil, katakan apa adanya. Kompleksitas
   tambahan hanya sepadan bila perbaikannya nyata.
3. Terapkan pada kedua mode ablasi.

### B4 — Ukuran potongan dokumen

`chunk_size` 900 karakter dengan tumpang tindih 120 ditetapkan tanpa pengujian. Gao dkk.
(2023) menyebut ukuran lazim 100, 256, dan 512 token.

1. Uji beberapa ukuran potongan. Karena pemecahan kini per halaman, ukuran lebih kecil
   tidak lagi berisiko memutus lintas halaman.
2. Laporkan metrik penelusuran dan jumlah potongan untuk tiap ukuran.
3. IKUTI PROTOKOL BAGIAN C. Ini penyetelan parameter, bukan perbaikan cacat.

### B5 — Parameter MMR

`lambda_mult` saat ini 0,5. Pada korpus 12 dokumen dengan tumpang tindih topik tinggi,
keragaman mungkin kurang bernilai dibandingkan relevansi.

1. Uji beberapa nilai `lambda_mult`.
2. IKUTI PROTOKOL BAGIAN C.
3. Laporkan pengaruhnya terhadap keragaman sumber yang tampil di antarmuka; menampilkan
   empat potongan dari satu dokumen yang sama kurang berguna bagi dokter.

---

## BAGIAN C — PROTOKOL PENYETELAN (WAJIB)

Berlaku untuk B4 dan B5, dan untuk setiap penyetelan parameter lain.

Alasan protokol ini ada: menyetel parameter sampai angkanya bagus lalu melaporkan angka
itu sebagai hasil adalah overfitting pada set evaluasi. Hasilnya tidak dapat
dipertanggungjawabkan saat sidang.

1. PISAHKAN set penyetelan dari set pelaporan. Bagi skenario evaluasi menjadi dua bagian
   yang tidak beririsan. Penyetelan HANYA pada bagian pertama. Angka Bab VI diambil dari
   bagian kedua dengan parameter yang sudah dikunci.
2. CATAT SELURUH KONFIGURASI YANG DIUJI, bukan hanya yang terbaik. Simpan ke
   `results/tuning_log.json` dengan medan: parameter, nilai, metrik pada set penyetelan,
   dan waktu pengujian.
3. TETAPKAN KRITERIA SEBELUM MENGUJI. **MRR adalah kriteria utama**, karena Hit@k jenuh
   pada mode prediction-conditioned.
4. JANGAN MENGUBAH SET EVALUASI setelah melihat hasil. Skenario yang tampak tidak wajar
   dilaporkan, bukan dihapus.
5. LAPORKAN BERAPA KONFIGURASI YANG DIUJI untuk tiap parameter.

Kalau protokol ini membuat suatu pengujian tidak dapat dilakukan karena jumlah skenario
terlalu sedikit untuk dibagi dua, KATAKAN. Lebih baik melaporkan bahwa penyetelan tidak
dilakukan daripada melaporkan angka yang tidak sah.

---

Terakhir, perbarui `docs/AUDIT_KEBUTUHAN.md` dengan status terbaru tiap butir.

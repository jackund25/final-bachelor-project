# Prapendaftaran T3.3 — jangkauan varian kueri dan bentuk kueri aplikasi

> **Ditulis 11 Agustus 2026 pada commit `b5dfdcf`.**
> **T3.3 BELUM dijalankan pada saat catatan ini dibuat.** Berkas
> `results/eval_rag/jangkauan_varian.json` belum ada, `scripts/eval_jangkauan_varian.py`
> belum ditulis, dan tidak satu pun angka jangkauan maupun peringkat untuk lengan
> `aplikasi` sudah dilihat.
>
> Sama alasannya dengan `PRAPENDAFTARAN_T3.1.md`: catatan yang dibuat sebelum melihat
> hasil punya bobot yang berbeda dari catatan yang dibuat sesudahnya.

## Mengapa percobaan ini ada

Dua pertanyaan terpisah yang kebetulan menyentuh berkas yang sama.

**Pertanyaan 1 datang dari T3.1.** Membuang angka glukosa dari teks kueri menaikkan Hit@1
dari 42,2% ke 71,1%, dan perbandingannya kebal kontaminasi karena produksi dan
`hanya_kondisi` punya bobot kata kunci pelabel yang identik (17). Tetapi TEMUAN jangkauan
mengukur hal yang berlawanan arah: frasa kondisi saja menjangkau **14** potongan, dan
penambahan angka menaikkannya ke **24**. Varian `hanya_kondisi` karena itu diduga
memperbaiki peringkat sambil **memperburuk jangkauan**, dan kedua hal itu belum pernah
diukur bersama. Keputusan mengadopsinya ke produksi tidak boleh diambil sebelum keduanya
terlihat berdampingan.

**Pertanyaan 2 datang dari pemeriksaan kode.** Yang diukur T3.1 adalah
`build_ablation_query()`. Kueri retrieval yang benar-benar dipakai aplikasi dibangun
`PredictionConditionedQueryBuilder._primary_query()` di `src/rag/conditioned_query.py`, dan
keduanya **berbeda jauh**. Seluruh angka penelusuran di laporan karena itu mengukur bentuk
kueri yang tidak pernah dijalankan aplikasi. Itu perlu diukur, bukan sekadar dicatat.

## Yang diperkirakan, dinyatakan sebelum diukur

### D1 — jangkauan `hanya_kondisi` memburuk, ke sekitar 14 potongan

Bentuk produksi menjangkau 24 potongan unik (1,16% korpus) pada sapuan glukosa 40–400
langkah 5. `hanya_kondisi` membuang numeralnya, sehingga 73 nilai glukosa runtuh menjadi
**tepat 3 kueri berbeda**. `eval_konsentrasi_penelusuran.py` sudah mengukur bahwa ketiga
frasa kondisi itu sendirian menjangkau **14** potongan.

**Perkiraan: `hanya_kondisi` menghasilkan tepat 14 potongan unik, turun dari 24 (−41,7%),
dan porsi pengambilan ke sepuluh potongan teratas naik dari 75,07% mendekati 100%.**

Angka 14 bukan perkiraan longgar melainkan **prediksi titik**, karena ketiga kueri itu
sudah pernah dijalankan. Bila hasilnya bukan 14, yang dicurigai lebih dulu adalah
pengukurannya — kemungkinan besar identitas potongan (`chunk_id` lawan cuplikan teks)
dihitung dengan cara berbeda antara kedua skrip.

### D2 — bentuk kueri aplikasi berperingkat terburuk atau mendekati terburuk

Bentuk aplikasi menggabungkan **dua** hal yang masing-masing sudah terbukti merugikan
peringkat pada T3.1:

| faktor | bukti T3.1 |
| --- | --- |
| numeral di teks kueri | membuangnya menaikkan Hit@1 42,2% → 71,1%, p=0,00058 |
| bentuk kalimat tanya | varian `pertanyaan` MRR 0,4744 · Hit@1 0,1556, p=2,3e-05 |

Ditambah lagi ia memuat numeral **empat kali** (prediksi, kondisi sekarang, delta, dan
pengulangan pada kalimat penutup), bukan sekali seperti bentuk produksi.

**Perkiraan: MRR lengan `aplikasi` jatuh di bawah `pertanyaan` (0,4744) dan di atas
`hanya_angka` (0,2276), yaitu di rentang kira-kira 0,25–0,47.**

### D3 — bobot kata kunci pelabel lengan `aplikasi` adalah 6

Dihitung tangan dari `KW` pada `ablation_rag_fullkb.py` terhadap teks kueri yang akan
dibangun, sebelum skripnya ada:

| kelas | kata kunci yang cocok | bobot |
| --- | --- | ---: |
| hipoglikemia | `hipoglikemi`, lewat `risk_label` "BAHAYA - Hipoglikemia" | 3 |
| hiperglikemia | `hiperglikemi`, lewat `risk_label` "HATI-HATI - Hiperglikemia" | 3 |
| normal | tidak ada — `risk_label` kelas normal adalah "AMAN" | 0 |
| **total** | | **6** |

**Ini yang membuat D2 dapat diuji.** Bobot 6 **identik** dengan varian `pertanyaan` (6),
sehingga `aplikasi` lawan `pertanyaan` adalah perbandingan **kebal kontaminasi** dengan
cara yang sama seperti produksi lawan `hanya_kondisi` pada bobot 17. Spearman(bobot, MRR) =
0,964 pada T3.1 berarti hampir seluruh peringkat antar-varian diramalkan bobotnya; dua
varian berbobot sama adalah satu-satunya tempat selisih peringkat dapat dibaca sebagai
selisih bentuk kueri.

**Bila bobot terukur ternyata bukan 6, D2 batal dan tidak boleh dinarasikan sebagai
peringkat** — persis nasib pemenang `kata_kunci` pada T3.1.

## Aturan penafsiran, ditetapkan sekarang

1. **Set PELAPORAN T3.1 tidak disentuh.** Ia sudah dipakai sekali (Bagian C butir 5).
   Lengan `aplikasi` dijalankan pada **set penyetelan saja** (`ohio_584`, `ohio_588`) dan
   dilaporkan sebagai pengukuran susulan, **bukan** sebagai baris kedelapan tabel T3.1.
   Angkanya tidak boleh disandingkan dengan kolom set pelaporan T3.1 di tabel mana pun.

2. **Jangkauan dan peringkat adalah dua besaran, bukan satu.** Bila `hanya_kondisi`
   memperbaiki peringkat sambil memperburuk jangkauan, itu **pertukaran**, bukan
   kemenangan maupun kekalahan. Yang dilaporkan adalah pertukarannya, dan keputusan
   adopsinya bukan milik saya.

3. **D2 mengukur bentuk kueri, bukan mutu aplikasi.** Bila lengan `aplikasi` benar
   berperingkat terburuk, yang terbukti adalah bahwa bentuk kuerinya merugikan penelusuran
   menurut pelabel `classify_chunk()`. Ia **tidak** membuktikan jawaban yang dihasilkan
   aplikasi lebih buruk, karena mutu jawaban diukur RAGAS dan tidak diuji di sini.

4. **K1 berlaku penuh.** Seluruh angka peringkat memakai pelabel kata kunci, bukan
   penilaian manusia. Selama T2.2 belum diisi, tidak satu pun angka T3.3 boleh dinyatakan
   mantap.

5. **Batas ukur yang sudah diketahui.** Pada sapuan jangkauan, lengan `aplikasi` dibangun
   dengan `current_glucose = predicted_glucose` (delta 0, tren stabil, tanpa IOB/COB aktif
   dan tanpa interval konformal), supaya hanya angka prediksi yang berubah dan ia sebanding
   dengan tujuh lengan lain. Kueri aplikasi sungguhan lebih beragam, sehingga angka
   jangkauan lengan `aplikasi` adalah **batas bawah**, bukan nilai sebenarnya.

## Tambahan saat penulisan skrip — dicatat sebelum hasil terlihat

Ditambahkan setelah bagian di atas selesai ditulis tetapi **sebelum skrip dijalankan** dan
sebelum satu pun angka terlihat. Dicatat sebagai tambahan, bukan disisipkan diam-diam ke
atas, supaya urutannya tetap terbaca.

Lengan `aplikasi` dipecah **dua**, karena `nilai_varian()` pada `eval_susunan_kueri.py`
hanya meneruskan glukosa TERPREDIKSI kepada pembentuk kueri, sementara
`_primary_query()` sungguhan juga memakai glukosa SAAT INI:

| lengan | `current_glucose` | untuk apa |
| --- | --- | --- |
| `aplikasi_delta0` | disamakan dengan prediksi | sebanding dengan tujuh varian lain — hanya angka prediksi yang berubah |
| `aplikasi_penuh` | nilai `current` sebenarnya dari kasus | setia pada kueri yang benar-benar dijalankan aplikasi |

Selisih keduanya mengisolasi sumbangan klausa tren dan delta. **D2 berlaku bagi keduanya**;
bila hanya salah satu yang jatuh di rentang 0,25–0,47, itu dilaporkan sebagai perkiraan
yang terpenuhi sebagian, bukan dibulatkan menjadi terpenuhi.

Horizon yang dipakai **30 menit** (`default_horizon` 6 × 5 menit), bukan 60 menit yang
menjadi bawaan `build_conditioned_query()`, supaya sepadan dengan `HORIZON = 6` pada T3.1.

## Kaitan

- `docs/PRAPENDAFTARAN_T3.1.md` — pola dan aturan penafsiran yang diikuti berkas ini
- `results/eval_rag/konsentrasi_penelusuran.json` — sumber angka 14, 24, dan 75,07%
- `results/eval_rag/susunan_kueri.json` — sumber angka T3.1 dan uji kontaminasinya
- `results/eval_rag/tumpang_tindih_kata_kunci.json` — bobot tujuh varian, yang akan
  ditambah lengan kedelapan
- Keputusan menggantung #6 pada `docs/HANDOFF.md` — yang percobaan ini sediakan datanya

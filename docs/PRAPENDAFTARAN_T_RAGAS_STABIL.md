# Prapendaftaran — kestabilan tiga metrik RAGAS yang belum diuji

> **Ditulis 11 Agustus 2026 pada commit `d31394e`.**
> **Pengukuran BELUM dijalankan.** `results/ragas/kestabilan.json` belum ada, dan tidak satu
> pun skor jalan kedua sudah dilihat. Skor jalan **pertama** memang sudah ada dan sudah
> dilihat — ia tersimpan di `results/ragas/cache/` sejak T1.2 — dan itu dicatat di sini
> sebagai bagian dari rancangan, bukan disembunyikan.

## Aturan keputusan, ditetapkan sebelum melihat hasil

**Pengukuran ini DIAGNOSTIK.** Tidak satu pun angka RAGAS yang sudah ada diubah atau
diganti, apa pun hasilnya:

| tetap berlaku | nilai |
|---|---|
| faithfulness | 0,678 |
| answer_relevancy | 0,763 |
| context_precision | 0,450 |
| context_recall | 0,450 |

Yang dihasilkan adalah **pernyataan kestabilan yang menyertai angka-angka itu di Bab VI** —
bukan angka pengganti, bukan rerata dua jalan.

Bila hasilnya menunjukkan ketiga metrik sama labilnya dengan `faithfulness`, itu **temuan
penting** dan masuk keterbatasan baru: **angka RAGAS satu-jalan tidak dapat
dipertanggungjawabkan pada tingkat kasus, hanya pada rerata** — memperluas K10 dari satu
metrik menjadi keempatnya.

## Rancangan, dan mengapa ia sepadan dengan pengukuran faithfulness

`faithfulness` memperoleh status "labil" dari **dua jalan penuh** berkonfigurasi identik.
Rancangan di sini sama:

| | |
|---|---|
| jalan 1 | skor yang **sudah tersimpan** di `results/ragas/cache/`, dari T1.2 |
| jalan 2 | jalan baru dengan `--no-cache`, konfigurasi identik |
| kasus | sepuluh kasus yang sama, E01–E10 |
| juri | `gemini-3.5-flash-lite`, `top_k` 5 |
| yang dilaporkan | selisih absolut **median** dan **maksimum** per metrik, plus kestabilan **reratanya** — persis bentuk yang dipakai faithfulness (median 0,18 · maksimum 0,56 · rerata 0,003) |

**Jalan 1 disalin keluar lebih dulu** ke `results/ragas/kestabilan_jalan1.json` sebelum
apa pun dijalankan, karena jalan kedua menimpa cache.

**Anggaran kuota:** ~100 panggilan (10 pembangkitan + 30 `answer_relevancy` + 50
`context_precision` + 10 `context_recall`). Kuota `gemini-3.5-flash-lite` 500 RPD, jadi muat
dalam satu hari dengan margin. Bila terbentur 429, **berhenti dan lapor** — jangan turunkan
laju dan jangan pindah model.

### Satu asimetri yang wajib dicatat, karena ia mengubah tafsir

Ketiga metrik **tidak** bergantung pada hal yang sama:

| metrik | masukan | sumber variasi |
|---|---|---|
| `answer_relevancy` | pertanyaan + **jawaban** | pembangkitan **dan** juri |
| `context_precision` | pertanyaan + konteks + acuan | **juri saja** |
| `context_recall` | konteks + acuan | **juri saja** |

Jawaban dibangkitkan ulang pada jalan kedua (temperature 0,2), sehingga variasi
`answer_relevancy` bercampur antara pembangkitan dan penjurian, sedangkan kedua metrik
konteks mengukur **variasi juri murni** (penelusuran sudah terbukti deterministik, Part 5).

Ini **tidak** dikoreksi — mengunci jawaban akan membuat rancangannya tidak lagi sepadan
dengan faithfulness. Ia dinyatakan sebagai batas tafsir.

## Bukti yang sudah ada

Skor per kasus jalan pertama:

```
answer_relevancy   0,685 0,505 0,650 0,661 0,887 0,827 0,775 0,933 0,821 0,886
context_precision  0,0   NaN   0,0   1,0   0,0   1,0   0,0   1,0   0,0   1,0
context_recall     0,0   1,0   0,0   0,5   0,0   1,0   0,0   1,0   0,0   1,0
```

Dua hal langsung terlihat: `answer_relevancy` **kontinu**, sedangkan kedua metrik konteks
**nyaris biner**. E02 pada `context_precision` bernilai **NaN**.

## Yang diperkirakan, dinyatakan sebelum diukur

### D1 — `answer_relevancy` PALING STABIL dari ketiganya, dan lebih stabil daripada faithfulness

Mekanismenya: RAGAS menilai `answer_relevancy` dengan membangkitkan beberapa pertanyaan dari
jawaban lalu mengukur **kemiripan embedding** terhadap pertanyaan asli. Langkah LLM-nya masuk
ke perata-rataan embedding, dan perata-rataan meredam variasi. `faithfulness` sebaliknya
memecah jawaban menjadi pernyataan lalu menilai tiap pernyataan **secara diskret**, sehingga
satu keputusan yang berubah menggeser rasionya sekaligus.

**Perkiraan: selisih absolut median ≤ 0,10 dan maksimum ≤ 0,25** — keduanya lebih kecil
daripada faithfulness (0,18 dan 0,56).

### D2 — kedua metrik konteks labil secara BINER, bukan bertahap

Keduanya nyaris biner di tingkat kasus. Ketidakstabilan pada besaran biner tidak dapat
bertahap: ia **berbalik penuh** 0↔1 atau tidak bergerak sama sekali.

**Perkiraan: selisih absolut maksimum = 1,00 tepat** untuk sekurang-kurangnya salah satu
dari `context_precision` dan `context_recall`, dan **selisih median bernilai 0,00 atau 1,00**,
bukan nilai antara.

**Perkiraan jumlah kasus yang berbalik: 1 sampai 3 dari 10** untuk masing-masing.

### D3 — rerata tetap stabil meski kasus berpindah

Pola faithfulness (0,681 lawan 0,678, selisih 0,003) berasal dari pembatalan: kasus yang naik
mengimbangi kasus yang turun.

**Perkiraan: |selisih rerata| ≤ 0,10 untuk ketiga metrik.**

Bila D2 dan D3 sama-sama terpenuhi, kesimpulannya menguat menjadi: **keempat metrik RAGAS
hanya sah dikutip sebagai rerata**, dan aturan pelaporan K10 yang selama ini khusus
`faithfulness` berlaku untuk keempatnya.

### D4 — NaN pada E02 `context_precision` BERULANG, bukan acak

NaN muncul bila tidak ada konteks yang dinilai relevan sehingga penyebutnya nol — sebab
**deterministik**, bukan variasi juri.

**Perkiraan: E02 tetap NaN pada jalan kedua.** Bila ia justru menghasilkan angka, sebabnya
bukan determinisme melainkan variasi juri pada tingkat yang lebih dalam daripada dugaan, dan
itu memperburuk penilaian kestabilan `context_precision`, bukan memperbaikinya.

## Aturan penafsiran, ditetapkan sekarang

1. **Rerata dua jalan TIDAK dilaporkan sebagai angka baru.** Angka Bab VI tetap dari jalan
   pertama; yang ditambahkan hanya pernyataan kestabilannya.
2. **Selisih pada satu kasus tidak boleh dipakai membandingkan apa pun** — itu justru yang
   sedang diukur ketidakabsahannya.
3. **NaN tidak diperlakukan sebagai nol.** Kasus ber-NaN dikeluarkan dari perhitungan median
   dan maksimum, dan jumlahnya dilaporkan terpisah.
4. **n = 10 kasus terlalu kecil untuk menyatakan besaran ketidakstabilan secara presisi.**
   Yang sah disimpulkan adalah **ada atau tidaknya** ketidakstabilan tingkat kasus, bukan
   nilai pastinya.
5. Bila kuota terbentur di tengah, **hasil sebagian dilaporkan apa adanya** beserta metrik
   mana yang belum lengkap — bukan disembunyikan sampai lengkap.

## Kaitan

- `docs/DAFTAR_KETERBATASAN.md` K10 — aturan pelaporan yang mungkin diperluas hasil ini
- `results/ragas/summary.json` — medan `kestabilan_metrik` yang akan diisi
- `docs/journey/part-6-uji-lanjutan-1-t11-kuota-t12.md` — dua jalan faithfulness
- Keputusan menggantung #7 pada `docs/KEPUTUSAN_DIAMBIL.md`

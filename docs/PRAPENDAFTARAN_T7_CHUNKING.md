# Prapendaftaran T7 — strategi pemecahan dokumen sadar-kalimat dan sadar-token

> **Ditulis 16 Agustus 2026, SEBELUM satu pun angka retrieval varian baru dilihat.**
> Cabang `refaktor-tujuh-tugas`.
>
> Diverifikasi saat penulisan: `results/eval_rag/strategi_chunking.json` belum ada.
> Perubahan kode yang sudah dikerjakan (chunker dan kutipan UI) TIDAK menyentuh
> `config.yaml`; indeks produksi `models/chroma_db` masih 2.061 potongan hasil
> pemisah lama, dan default `chunk_documents()` masih perilaku lama.

---

## 1. Mengapa prapendaftaran ini ada

Keluhan pengguna sederhana — kutipan di layar terpotong di tengah kalimat. Tetapi
penelusurannya membuka cacat kedua yang jauh lebih serius dan tidak terlihat di
layar: **8,23% token korpus tidak pernah masuk ke vektor**. Memperbaikinya berarti
**mengindeks ulang korpus**, dan itu mengubah setiap angka retrieval yang sudah
dilaporkan (MRR, Hit@1, nDCG, dan seluruh hasil crossfold).

Bahaya spesifik yang hendak ditutup berkas ini: sesudah melihat hasilnya, sangat
mudah memilih varian yang kebetulan menaikkan MRR lalu menyusun alasan teknis
untuk membenarkannya. Dugaan di Bagian 4 ditulis supaya pilihan itu tidak bisa
dibuat belakangan.

Ada pula ketegangan yang sudah terdokumentasi dan **tidak boleh diabaikan**:
`config.yaml:135` menyatakan chunk_size 900 SENGAJA dipertahankan, karena
menggabungkan `lambda_mult` 0,0 dengan chunk 500 justru MENURUNKAN MRR ke 0,464
dan ditolak uji Wilcoxon terhadap chunk 500 sendirian (p=0,029). Karena itu
"kecilkan saja potongannya" bukan jawaban yang sudah diketahui benar.

---

## 2. Dua cacat yang WAJIB dipisahkan

Menyatukan keduanya adalah kesalahan analisis yang paling mungkin terjadi di sini.

| | **Cacat A — pemotongan senyap** | **Cacat B — potongan di tengah kalimat** |
|---|---|---|
| Di mana | Sisi vektor, saat *embedding* | Sisi teks, saat pemecahan + tampilan |
| Terlihat? | **Tidak.** Tanpa peringatan, tanpa galat | Ya, dilihat pengguna di layar |
| Sebab | `all-MiniLM-L6-v2` `max_seq_length` = 256 token; token ke-257 dst. dibuang | Daftar pemisah tidak memuat tanda akhir kalimat |
| Akibat | 8,23% korpus tak pernah dapat terambil | Makna kutipan kabur; sulit dicocokkan ke PDF sumber |
| Terukur di | `results/eval_rag/distribusi_token.json` (T5.1) | Diukur ulang di T7 (Bagian 5) |

Keduanya bertemu pada satu tempat — `chunk_documents()` — tetapi obatnya berbeda,
dan **memperbaiki B tidak memperbaiki A**. Ini diuji, bukan diasumsikan (D2).

---

## 3. Dasar rujukan (sudah dibaca, bukan dikutip dari ingatan)

1. **Gao dkk. (2023), §V.A.1 "Chunking Strategy", hal. 8** — dibaca langsung dari
   `docs/sumber referensi/15_Gao-2023_...pdf`. Menyatakan bahwa pemecahan berukuran
   tetap "leads to truncation within sentences", dan memerikan *Small2Big*: kalimat
   sebagai satuan pengambilan, dengan kalimat sebelum dan sesudahnya diberikan
   sebagai konteks. Dua hal itu persis dua keluhan pengguna.
2. **Reimers & Gurevych (2019)** — SBERT menurunkan *sentence embeddings*; satuan
   yang model ini dilatih untuk mewakili adalah KALIMAT. Menyelaraskan batas
   potongan dengan batas kalimat berarti menyelaraskannya dengan satuan yang
   dipahami model, bukan sekadar merapikan tampilan.
3. **Manning dkk. (2009), hal. 217** — *passage retrieval*: yang dikembalikan
   adalah *passage* dengan batas bermakna. Manning merujuk Hearst & Plaunt (1993)
   dan Hearst (1997) sebagai asalnya; **kedua paper itu belum dibaca**, jadi
   keduanya tidak dikutip langsung dan hanya disebut lewat Manning.

**Kejujuran sitasi — penting untuk sidang.** Rujukan Gao untuk Small2Big adalah
[90], dan [88]–[90] semuanya **pos blog, bukan makalah wasit**. Karena itu Small2Big
dikutip sebagai *praktik yang disurvei Gao dkk.*, BUKAN sebagai metode ber-wasit.
Dasar ber-wasit untuk memilih batas kalimat berdiri pada rujukan 2 dan 3.

---

## 4. Dugaan yang didaftarkan di muka

Ditulis sebelum hasil retrieval varian mana pun dilihat.

- **D1 — reproduksi.** Varian V0 (900 karakter, pemisah lama) WAJIB menghasilkan
  jumlah potongan dan MRR yang sama dengan angka produksi yang berlaku sekarang.
  Bila tidak, ada yang berubah di luar kendali dan **seluruh hasil T7 batal**.
- **D2 — dua cacat itu terpisah.** V1 (900 karakter + pemisah kalimat) akan
  menaikkan proporsi potongan berakhir-kalimat-utuh secara nyata, TETAPI proporsi
  token terbuang tetap di atas 5%. Yakni: memperbaiki B tidak memperbaiki A.
- **D3 — jaminan konstruktif.** V3 (256 token + pemisah kalimat) akan menghasilkan
  **nol** potongan melewati 256 token — bukan "sedikit", melainkan nol, karena
  panjangnya ditakar dengan tokenizer model itu sendiri.
- **D4 — arah MRR TIDAK didugakan.** Sengaja. Sapuan B4 menunjukkan hubungan
  chunk_size dengan MRR **tidak monoton** (Spearman p=0,50 atas lima ukuran), dan
  chunk 300 yang sama sekali tidak terpotong justru ber-MRR terburuk kedua.
  Mendugakan arahnya di sini berarti berpura-pura tahu. Yang didaftarkan adalah
  **kriteria keputusannya**, di Bagian 6.
- **D5 — biaya.** V3 menghasilkan potongan lebih banyak daripada V0, sehingga
  waktu indexing naik. Kenaikan di atas 3x dicatat sebagai biaya yang harus
  disebut di naskah, bukan disembunyikan.

---

## 5. Varian yang diuji — seluruhnya, termasuk yang kalah

| Kode | chunk_size | Satuan | Pemisah | Menjawab |
|---|---|---|---|---|
| **V0** | 900 | karakter | lama (kontrol) | — |
| **V1** | 900 | karakter | kalimat | Cacat B saja |
| **V2** | 500 | karakter | kalimat | B; A secara kebetulan |
| **V3** | 256 | **token** | kalimat | A **dan** B |

Tumpang tindih dijaga proporsional ~13% (mengikuti rasio 120/900 yang berlaku),
supaya yang berubah adalah strategi, bukan campuran strategi dan rasio tumpang
tindih.

V2 penting justru karena ia menyelesaikan A **secara kebetulan** — kecil cukup
sehingga hampir tidak ada potongan melewati 256 token, tetapi tanpa jaminan.
Membandingkan V2 dengan V3 memisahkan "kebetulan muat" dari "dijamin muat".

---

## 6. Kriteria keputusan — ditetapkan SEBELUM hasil dilihat

Protokol Bagian C berlaku: set penyetelan dan set pelaporan dibagi dengan benih
tetap sebelum hasil dilihat; seluruh varian dicatat termasuk yang kalah.

Urutan aturan, diterapkan dari atas:

1. **Bila D1 gagal** → seluruh hasil T7 batal, tidak ada varian yang diadopsi.
2. **Bila V3 mengungguli V0 pada MRR set pelaporan** → V3 diadopsi. Ia unggul di
   dua sumbu sekaligus (MRR dan integritas korpus).
3. **Bila V3 setara V0** (selisih MRR < 0,02) → **V3 tetap diadopsi**, dengan alasan
   integritas korpus: 8,23% korpus yang tak pernah terambil adalah cacat yang berdiri
   sendiri, tidak bergantung pada apakah MRR ikut naik. Alasan ini ditulis di sini,
   di muka, supaya tidak tampak sebagai pembenaran yang disusun belakangan.
4. **Bila V3 kalah nyata dari V0** (MRR turun ≥ 0,02) → **V3 TIDAK diadopsi diam-diam.**
   Yang dilaporkan adalah pertukarannya secara terbuka: korpus utuh dengan MRR lebih
   rendah, lawan korpus terpotong dengan MRR lebih tinggi. Keputusannya diserahkan
   kepada pembimbing, dan Bab IV memerikan keduanya.
5. **Pemisah kalimat (V1) diadopsi tanpa menunggu hasil MRR.** Ia menjawab keluhan
   pengguna secara langsung, dasarnya Gao dkk. §V.A.1, dan perbaikan tampilan pada
   `citations.py` sudah tidak bergantung pada indeks sama sekali.

---

## 7. Yang TIDAK diubah oleh T7

Ditulis eksplisit supaya cakupannya tidak melebar diam-diam:

- `config.yaml` tidak disentuh sampai ada keputusan pada Bagian 6.
- Model embedding tetap `all-MiniLM-L6-v2`. Mengganti model akan mengubah dua
  faktor sekaligus dan membuat hasilnya tak dapat ditafsirkan.
- `manual_kb` (350/40) tidak diubah: potongannya sudah jauh di bawah 256 token.
- Prediktor, conformal, dan seluruh angka Bab prediksi tidak tersentuh.
- Default `chunk_documents()` tetap perilaku lama; varian dipilih eksplisit lewat
  argumen baris perintah.

---

## 8. Keluaran

- `results/eval_rag/strategi_chunking.json` — seluruh varian, termasuk yang kalah.
- Pencatatan ke `results/tuning_log.json` lewat `tuning_protocol.catat()`.
- Skrip: `scripts/eval_strategi_chunking.py`.

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

**Korpus produksi dipakai, bukan koleksi terkontrol**, karena membandingkan retriever
pada koleksi 17 potongan tidak bermakna: dengan `top_k` 5, satu kueri sudah menyentuh
29% koleksinya.

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

`results/ragas/lengan_penelusuran.json`, memuat kedua lengan termasuk yang kalah.

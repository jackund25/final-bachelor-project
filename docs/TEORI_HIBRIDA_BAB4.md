# Bahan perancangan penelusuran hibrida untuk Bab IV

Pendamping `docs/TEORI_HIBRIDA_BAB2.md`. Bab II memuat **teorinya**; berkas ini memuat
**keputusan perancangannya** — apa yang dipilih pada sistem ini, dan atas dasar apa.

Seluruh angka di bawah dibaca langsung dari `config.yaml` dan indeks produksi pada
17 Agustus 2026, bukan dari ingatan.

---

## 1. Yang harus dikoreksi lebih dulu pada Bab IV yang ada

Subbab IV.3 sudah memuat alur pipeline berbutir enam. **Tiga butirnya kini basi**, dan
mengoreksinya lebih penting daripada menambah teks baru.

| Butir | Tertulis sekarang | Seharusnya |
|---|---|---|
| (2) | model *Random Forest* | **Gradient Boosting** |
| (4) | kueri di-*embed* lalu *retrieval* dari ChromaDB | **penelusuran hibrida** — lihat Bagian 2 |
| (5) | LLM `gemini-2.5-flash-lite` | **`gemini-3.5-flash-lite`** |

Selain itu, subbab IV.4 menyebut *"engine digital twin"* — istilah yang sudah dicabut.

---

## 2. Usulan penggantian butir (4)

Butir (4) yang berlaku sekarang hanya menyebut satu jalur penelusuran. Penggantinya:

> (4)~kueri yang telah dikondisikan ditelusurkan pada basis pengetahuan klinis
> memakai **penelusuran hibrida**: jalur padat memakai \texttt{all-MiniLM-L6-v2} atas
> indeks vektor ChromaDB dengan penataan ulang MMR, sedangkan jalur leksikal memakai
> BM25 atas potongan yang sama; kedua daftar peringkat digabung dengan
> \emph{Reciprocal Rank Fusion} sebagaimana diuraikan pada
> Subbab~\ref{subsec:penelusuran-hibrida}, lalu lima potongan teratas diambil;

---

## 3. Usulan subbab baru — keputusan perancangan

Ditempatkan sesudah IV.3, sebelum "Perancangan Perangkat Lunak".

```latex
\subsection{Perancangan Penelusuran Hibrida}
\label{subsec:penelusuran-hibrida}

Penelusuran pada sistem ini tidak bertumpu pada satu jalur. Kueri yang telah
dikondisikan prediksi ditelusurkan melalui \textbf{dua jalur yang berjalan atas
potongan dokumen yang sama}, lalu hasilnya digabung menurut peringkat.

\textbf{Jalur padat} memakai model \texttt{all-MiniLM-L6-v2} atas indeks vektor
ChromaDB, dengan penataan ulang \emph{Maximal Marginal Relevance}
\autocite{carbonell1998} agar konteks yang diberikan kepada model bahasa tidak
berulang isi. \textbf{Jalur leksikal} memakai pembobotan Okapi BM25
\autocite{manning2008} atas potongan yang sama, sehingga istilah klinis yang harus
cocok persis --- nama sediaan insulin, ambang seperti $70$~mg/dL, dan aturan
15--15 --- tetap terjangkau meskipun ruang maknanya tidak menangkapnya.

Kedua jalur menghasilkan skor yang \textbf{tidak sebanding}: skor kosinus berbatas,
sedangkan skor BM25 tidak berbatas dan bergantung pada statistik korpus.
Penggabungannya karena itu dilakukan pada \textbf{peringkat}, bukan skor, memakai
\emph{Reciprocal Rank Fusion} yang juga dipakai Xiong dkk. \autocite{xiong2024} pada
tolok ukur penelusuran medis. Tetapan peredamnya dipertahankan pada nilai baku dan
\textbf{tidak disetel}, agar tidak menambah parameter bebas yang menuntut protokol
penyetelan tersendiri.

\textbf{Dasar pemilihan.} Keputusan ini bukan penyetelan umum, melainkan jawaban atas
keadaan yang khas pada penelitian ini. Model penelusuran padat yang tersedia dan
memenuhi Batasan~5 (komputasi inti berjalan lokal pada perangkat kelas konsumen) adalah model
\textbf{berbahasa Inggris}, sedangkan korpus pedomannya \textbf{berbahasa Indonesia}.
Jalan yang paling langsung, yaitu mengganti ke model multibahasa, telah ditempuh dan
\textbf{ditolak berdasarkan pengukuran}: kemampuan sistem membedakan kelas kondisi
justru merosot dan kelas kondisi normal hampir hilang, sebagaimana dilaporkan pada
Bab~\ref{chap:evaluasi}. Pencocokan leksikal menempuh jalan yang berbeda sama sekali,
sebab ia tidak bergantung pada ruang makna, sehingga bekerja tepat pada titik model
berbahasa Inggris paling lemah.

\textbf{Perilaku saat gagal.} Bila indeks leksikal tidak dapat dibangun, sistem tetap
melayani penelusuran melalui jalur padat, tetapi \textbf{menyatakan penurunan itu
secara eksplisit} beserta sebabnya. Sistem sengaja tidak dibiarkan mengaku menelusur
secara hibrida sambil sesungguhnya menelusur dengan satu jalur, sejalan dengan
KNF-04 yang menuntut kegagalan komponen tidak terjadi secara diam-diam.

\textbf{Keterbatasan yang diakui.} Pemenggalan kata pada jalur leksikal belum
memenggal imbuhan bahasa Indonesia, sehingga \emph{pemberian}, \emph{diberikan}, dan
\emph{berikan} diperlakukan sebagai tiga istilah yang berbeda. Lee dkk.
\autocite{lee2024} menunjukkan pemenggal khusus bahasa berpengaruh pada pedoman
diabetes berbahasa Korea; padanan bagi bahasa Indonesia belum diuji pada penelitian
ini dan dicatat sebagai arah pengembangan pada Bab~\ref{chap:penutup}.
```

---

## 4. Tabel parameter penelusuran

Untuk Bab IV atau lampiran. Seluruh nilai dibaca dari `config.yaml`, 17 Agustus 2026.

| Parameter | Nilai | Keterangan |
|---|---|---|
| Model penelusuran padat | `all-MiniLM-L6-v2` | 384 dimensi, berjalan pada prosesor |
| Cara penelusuran | `hibrida` | padat + leksikal, digabung RRF |
| Potongan terindeks | 2.248 | pemecahan per halaman, batas kalimat |
| Ukuran potongan | 900 / 120 | karakter / tumpang tindih |
| Kolam tiap jalur sebelum digabung | 50 | |
| Tetapan peredam RRF | 60 | **tidak disetel**, nilai baku |
| Potongan yang diambil | 5 | |
| Kolam kandidat MMR | 12 | di dalam jalur padat |
| $\lambda$ MMR | 0,0 | keragaman penuh |
| Model bahasa | `gemini-3.5-flash-lite` | suhu 0,2; batas 700 token keluaran |

---

## 5. Keselarasan dengan Gambar IV.1

Perubahan ini **tidak menambah maupun menghapus tahapan** pada alur sistem:

- **Penyimpan pengetahuan tetap satu.** Indeks BM25 dibangun dari potongan yang sudah
  tersimpan pada indeks vektor, bukan dari korpus terpisah. Bila keduanya dibangun
  dari sumber berbeda, selisih hasil dapat berasal dari perbedaan korpus dan hal itu
  tidak akan terlihat dari angka mana pun.
- **Masukan dan keluaran tahap penelusuran tidak berubah**: kueri terkondisi masuk,
  lima potongan keluar.
- Yang berubah hanya **isi kotak penelusuran**, dari "MMR Retrieval" menjadi
  penelusuran hibrida.

Dua angka pada gambar perlu diperbarui: **2.061 → 2.248 potongan**, dan label
pemecahan perlu menyebut batas kalimat.

---

## 6. Yang BELUM boleh ditulis

Angka hasil belum lengkap saat berkas ini disusun:

- Crossfold pada konfigurasi hibrida produksi **sedang berjalan**. Sebelum selesai,
  Bab~VI belum boleh memuat angka retrieval yang baru.
- Angka T13 dan T14 yang sudah ada **boleh** dipakai, dengan syarat disertai
  keterbatasannya: T14 hanya sepuluh kasus, dan tidak ada uji signifikansi yang sah
  pada ukuran itu.

# Arsip artefak laporan TA

Berkas di folder ini **tidak lagi digunakan** dan **tidak dikompilasi**. Tidak
ada `\input` maupun `\includegraphics` dari `TA.tex` yang menunjuk ke sini —
diverifikasi lewat penelusuran transitif dari `TA.tex`, bukan dari pola nama.

Disimpan, bukan dihapus, karena memuat rumusan lama yang mungkin perlu dirujuk
saat menjelaskan perubahan arah penelitian kepada pembimbing atau penguji.

> **Peringatan.** Isi folder ini memuat istilah dan klaim yang **sudah dicabut**:
> *digital twin*, *what-if*, *surrogate*, `T2DM`, *Prediction-Conditioned*, FKTP.
> Jangan menyalin kalimat dari sini ke naskah aktif tanpa memeriksa ulang.

## Isi

### `bab/` — naskah bab sebelum revisi

| Berkas | Digantikan oleh |
|---|---|
| `Bab I - Pendahuluan_ARSIP_PRAREVISI.tex` | `Bab I - Pendahuluan.tex` |
| `Bab II - Studi_ARSIP.tex` | `Bab II - Studi.tex` |
| `Bab III - Analisis_ARSIP_PRAREVISI.tex` | `Bab III - Analisis.tex` |

### `tables/` — tabel era pra-revisi

| Berkas | Status |
|---|---|
| `tabel_II1_integrasi.tex` | tanpa pengganti — konsep *digital twin* dicabut |
| `tabel_II2_karakteristik.tex` | tanpa pengganti — konsep *digital twin* dicabut |
| `tabel_II3_terkait.tex` | digantikan `tables/tabel_II2_terkait.tex` |
| `tabel_III1_KF_ARSIP.tex` | versi lama dari `tables/tabel_III1_KF.tex` |
| `tabel_III2_KNF_ARSIP.tex` | versi lama dari `tables/tabel_III2_KNF.tex` |
| `tabel_III3_solusi.tex` | digantikan `tables/tabel_III3_pemetaan.tex` |
| `tabel_III4_matriks.tex` | digantikan `tables/tabel_III5_matriks.tex` |

Lima berkas berakhiran `_ARSIP` (`tabel_II1_integrasi_ARSIP`,
`tabel_II2_karakteristik_ARSIP`, `tabel_II3_terkait_ARSIP`,
`tabel_III3_solusi_ARSIP`, `tabel_III4_matriks_ARSIP`) **byte-identik** dengan
berkas bernama sama tanpa akhiran itu — salinan pengaman yang kini mubazir dan
aman dihapus.

### `images/` — gambar era pra-revisi

| Berkas | Digantikan oleh |
|---|---|
| `Gambar_II1_TingkatIntegrasiDT.png` | tanpa pengganti — *digital twin* dicabut |
| `Gambar_II2_TaksonomiDT.png` | tanpa pengganti — *digital twin* dicabut |
| `Gambar_III1_KondisiSaatIni.png` | `images/Gambar_III1_AlurPerawatanDMT1.png` |
| `Gambar_III2_UseCase.png` | `images/Gambar_III2_GambaranUmumSistem.png` |

## Yang sengaja TIDAK diarsipkan

Tidak semua berkas yang tak terpakai adalah berkas lama. Tiga golongan berikut
tetap di tempatnya:

**Belum dipakai karena babnya belum dimigrasi.**
`images/Gambar_IV1_AlurRinciSistem.png` — gambar baru, disiapkan untuk Bab IV.
`listings/conditioned-query.tex` dan `listings/engineer-features.tex` — cuplikan
kode milik penelitian ini.

**Struktur bawaan template yang sengaja dinonaktifkan.**
`7a Daftar Lampiran.tex` dikomentari di `TA.tex:369` sesuai petunjuk template.
`Lampiran-B.tex` adalah slot lampiran cadangan.

**Contoh bawaan template.** `tables/tabel1.tex`, `tables/longtable1.tex`,
`images/gambar1.png`, `algorithms/binary-search-alg.tex`,
`listings/binary-search-python.tex` — bagian dari TA-STI-template-1.0, bukan
karya penulis.

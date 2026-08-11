# Prapendaftaran T3.1 — varian `kondisi_dulu`

> **Ditulis 11 Agustus 2026, 09:42 WIB, pada commit `b29eef7`.**
> **T3.1 BELUM dijalankan pada saat catatan ini dibuat.** Berkas
> `results/eval_rag/susunan_kueri.json` belum ada, dan tidak satu pun angka MRR untuk
> varian mana pun sudah dilihat.
>
> Catatan yang dibuat sebelum melihat hasil punya bobot yang berbeda dari catatan yang
> dibuat sesudahnya. Itulah alasan berkas ini ada, terpisah dan bertanggal, alih-alih
> ditulis di dalam entri T3.1 nanti.

## Bukti yang sudah ada

Sewaktu menyiapkan T2.2 (commit `35d5866`), kerangka sampel diperlebar dengan menjalankan
beberapa bentuk kueri T3.1 atas grid glukosa 36 nilai, `top_k` 5, pada indeks produksi.
Cacah potongan **unik baru** yang disumbangkan tiap bentuk:

| bentuk kueri | potongan unik baru |
|---|---|
| produksi | 25 |
| `hanya_angka` | +6 |
| `hanya_kondisi` | +5 |
| **`kondisi_dulu`** | **+0** |
| `pertanyaan` | +11 |

`kondisi_dulu` berisi teks yang **identik** dengan bentuk produksi — angka dan frasa
kondisi yang sama persis — hanya urutannya dibalik:

```
produksi     : "Kadar glukosa darah 58 mg/dL. Hipoglikemia, gula darah rendah di bawah
                70 mg/dL. Penyebab, gejala, dan penanganan segera (aturan 15-15)."
kondisi_dulu : "Hipoglikemia, gula darah rendah di bawah 70 mg/dL. Penyebab, gejala, dan
                penanganan segera (aturan 15-15). Kadar glukosa darah 58 mg/dL."
```

Atas 36 kueri × `top_k` 5 = 180 pengambilan, `kondisi_dulu` **tidak menghasilkan satu pun
potongan yang belum terambil bentuk produksi**.

## Yang diperkirakan, dinyatakan sebelum diukur

**`kondisi_dulu` diperkirakan menghasilkan MRR, Hit@1, dan nDCG@5 yang identik — atau
berbeda hanya sebesar galat pembulatan — terhadap bentuk produksi.**

Alasannya mekanistik. `all-MiniLM-L6-v2` menghasilkan embedding kalimat lewat *mean
pooling* atas token. Rata-rata bersifat komutatif, sehingga menukar urutan dua klausa yang
isinya sama hampir tidak menggeser vektor hasilnya. Yang tersisa hanyalah efek attention
antar-token lintas batas klausa, dan besarnya tampaknya di bawah ambang yang dapat
mengubah peringkat lima besar.

## Aturan penafsiran, ditetapkan sekarang

**Bila T3.1 melaporkan selisih yang berarti untuk `kondisi_dulu`, yang harus dicurigai
lebih dulu adalah pengukurannya, bukan temuannya.** Tiga hal yang wajib diperiksa sebelum
selisih itu dilaporkan sebagai hasil:

1. Apakah `v_kondisi_dulu()` benar-benar menghasilkan teks yang isinya sama dengan
   `v_produksi()`, atau ada spasi/tanda baca yang tanpa sengaja berbeda.
2. Apakah kasus, benih acak, dan pembagian pasien pada kedua lengan benar-benar identik.
3. Apakah selisihnya melampaui variasi antar-jalan retriever itu sendiri.

Sebaliknya, **bila selisihnya nol atau mendekati nol, itu bukan hasil kosong.** Ia
mengukur satu hal yang berguna bagi Bab VI: **posisi klausa di dalam kueri tidak
berpengaruh pada model embedding ini**, sehingga upaya menyusun ulang urutan kalimat kueri
bukan tuas yang layak dikejar. Varian lain (`pertanyaan`, `hanya_kondisi`) mengubah *isi*,
dan hanya di situ perbedaan dapat diharapkan.

## Kaitan

Bukti pendukung berasal dari pengukuran konsentrasi penelusuran
(`results/eval_rag/konsentrasi_penelusuran.json`), yang menunjukkan jangkauan retrieval
sangat sempit. Bila jangkauan memang sesempit itu, wajar bila perubahan yang tidak
mengubah isi kueri sama sekali tidak menggeser hasil.

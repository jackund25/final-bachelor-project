# Checklist Perbaikan Formatting Laporan Tugas Akhir

**Dokumen acuan:** `TA.pdf` versi terbaru yang tersedia  
**Tujuan:** menjadi checklist finalisasi tata letak sebelum PDF TA dianggap siap cetak/sidang.

> **Catatan penting:** Dokumen ini berfokus pada **formatting, layout, keterbacaan gambar/tabel, caption, dan konsistensi visual**. Isi ilmiah tidak perlu dirombak hanya untuk memperbaiki temuan-temuan berikut.

---

# 1. Aturan Formatting yang Menjadi Acuan

## 1.1 Gambar dan tabel tidak boleh bertumpuk tanpa narasi

Dilarang menempatkan dua elemen visual secara berurutan tanpa paragraf penjelas substantif di antaranya.

Berlaku untuk semua kombinasi:

- Gambar → Gambar
- Tabel → Tabel
- Gambar → Tabel
- Tabel → Gambar

Pola yang diharapkan:

> Paragraf → Gambar → Paragraf → Gambar

atau

> Paragraf → Tabel → Paragraf → Tabel

Paragraf penghubung harus benar-benar menjelaskan isi, temuan, hubungan, atau alasan keberadaan elemen berikutnya. Paragraf filler tidak diperbolehkan.

---

## 1.2 Caption harus singkat

Caption/judul gambar atau tabel berfungsi sebagai **identitas dan deskripsi singkat**, bukan sebagai tempat untuk menuliskan:

- hasil numerik panjang;
- metodologi;
- interpretasi;
- alasan pemilihan;
- definisi variabel yang panjang;
- kesimpulan;
- penjelasan beberapa paragraf.

Target praktis:

- ideal: 1 baris;
- masih baik: 2 baris;
- lebih dari 3 baris: harus diperiksa dan umumnya perlu dipangkas.

Penjelasan rinci harus dipindahkan ke paragraf sebelum atau sesudah gambar/tabel.

---

## 1.3 Ukuran gambar ditentukan berdasarkan keterbacaan

Gambar harus cukup besar sehingga:

- teks di dalam diagram dapat dibaca pada ukuran halaman normal;
- label sumbu dapat dibaca;
- legenda dapat dibaca;
- elemen diagram tidak berubah menjadi terlalu kecil;
- screenshot UI tetap dapat menunjukkan fungsi yang ingin dibuktikan.

Jangan mengecilkan gambar hanya agar seluruhnya muat dalam satu halaman.

Jika diagram terlalu kompleks:

1. sederhanakan diagram; atau
2. gunakan halaman landscape/halaman khusus; atau
3. pindahkan detail teknis lengkap ke lampiran.

---

## 1.4 Tabel harus fit terhadap area teks

Tidak boleh ada:

- kolom keluar margin;
- teks bertabrakan;
- teks terpotong;
- kolom terlalu sempit hingga sulit dibaca;
- font dikecilkan secara berlebihan hanya untuk memaksa tabel muat.

Jika tabel terlalu lebar, prioritaskan:

1. menyederhanakan isi kolom;
2. mengatur ulang lebar kolom;
3. memecah tabel secara logis;
4. menggunakan `tabularx`/layout kolom fleksibel;
5. menggunakan landscape bila benar-benar diperlukan.

---

# 2. Prioritas P0 — Wajib Diperbaiki

## P0-01 — Gambar IV.2/IV.3: diagram arsitektur/alur terlalu kecil dan terlalu kompleks

### Lokasi
Bab IV, diagram alur rinci sistem.

Pada versi laporan yang terindeks, diagram tersebut dirujuk sebagai **Gambar IV.2** di beberapa bagian, sedangkan daftar/versi tertentu menunjukkan **Gambar IV.3**. Nomor akhir harus mengikuti nomor yang benar pada PDF final.

### Masalah

Diagram memuat:

- tahap luring;
- tahap daring;
- beberapa komponen;
- penanda kontribusi metodologis;
- pengklasifikasi yang muncul pada tahap berbeda.

Akibatnya, ketika dipaksa masuk ke halaman portrait, teks di dalam diagram menjadi terlalu kecil.

Caption saat ini juga terlalu panjang dan sudah berisi penjelasan metodologis.

### Perbaikan

**Perbesar diagram semaksimal mungkin.**

Jika masih tidak terbaca:

- gunakan halaman landscape; atau
- sederhanakan diagram utama dan pindahkan detail teknis ke tabel/lampiran.

### Caption yang disarankan

> **Gambar IV.x Alur rinci sistem luring dan daring.**

### Penjelasan yang dipindahkan keluar caption

Penjelasan mengenai:

- empat kontribusi metodologis;
- lima kotak bertanda;
- pengklasifikasi yang muncul dua kali;
- perbedaan tahap luring dan daring;

ditulis sebagai paragraf sebelum/sesudah gambar.

---

## P0-02 — Gambar IV.5: sequence diagram terlalu padat/kecil

### Lokasi
Bab IV, sequence/alur interaksi sistem.

### Masalah

Diagram sequence memuat terlalu banyak aktor/komponen untuk ukuran halaman yang tersedia sehingga teks internal sulit dibaca.

### Perbaikan

Prioritas utama: **sederhanakan diagram utama**.

Tampilkan alur tingkat tinggi:

> Dokter → API → Predictor → Query Builder → Retriever → LLM → Citation Resolver → Dokter

Detail teknis yang terlalu rinci dapat dipindahkan ke lampiran.

Alternatif jika detail harus dipertahankan:

- gunakan halaman landscape.

---

## P0-03 — Gambar VI.2 dan VI.3 terlalu bertumpuk

### Lokasi
Bab VI, sekitar halaman 106 pada versi yang diperiksa.

### Masalah

Gambar VI.2 dan Gambar VI.3 ditempatkan pada area yang berdekatan/bertumpuk sehingga:

- kedua gambar menjadi relatif kecil;
- Clarke Error Grid sulit dibaca;
- grafik perbandingan model kehilangan ruang;
- pembaca harus berpindah fokus antar dua visual.

### Perbaikan

Jangan menampilkan:

> Gambar VI.2 → caption → Gambar VI.3 → caption

secara langsung.

Gunakan:

> Paragraf → Gambar VI.2 → paragraf → Gambar VI.3

Jika perlu, beri masing-masing ruang halaman yang lebih besar.

### Caption yang disarankan

> **Gambar VI.2 Clarke Error Grid prediksi model produksi pada horizon +60 menit.**

> **Gambar VI.3 Perbandingan metrik ketiga model pada horizon +30 menit.**

Angka persentase dan interpretasi dipindahkan ke paragraf pembahasan.

---

## P0-04 — Tabel V.2 mengalami tabrakan/overlap teks

### Lokasi
Bab V, tabel pustaka utama dan versinya.

### Masalah

Nama package/library yang panjang, terutama package teknis, bertabrakan atau terlalu berdesakan dengan kolom di sebelahnya.

### Perbaikan

Jangan mengecilkan seluruh tabel.

Lakukan:

- lebarkan kolom **Pustaka (versi)**;
- sempitkan kolom yang deskripsinya dapat dipadatkan;
- izinkan line break pada nama package;
- gunakan kolom fixed-width yang sesuai;
- gunakan `\raggedright` bila diperlukan.

Pastikan tidak ada satu karakter pun yang bertabrakan setelah PDF di-render ulang.

---

## P0-05 — Tabel VI.12 terlalu lebar/berpotensi keluar margin

### Lokasi
Bab VI, tabel pengukuran kelayakan penerapan sistem.

### Masalah

Kolom paling kanan terlalu dekat dengan atau melewati batas area teks. Tabel juga terlalu padat karena kolom keterangan panjang.

### Perbaikan

Susun ulang menjadi kolom yang lebih ekonomis, misalnya:

| Tahap | Keterangan | Waktu |
|---|---|---:|
| Rekayasa fitur | Jendela logbook → vektor fitur | xx ms |
| Prediksi | Regresi, klasifikasi, interval | xx ms |
| Penelusuran | BM25 atas korpus | xx ms |

Gunakan kolom fleksibel untuk **Keterangan** dan kolom fixed-width kecil untuk **Waktu**.

### Caption yang disarankan

> **Tabel VI.12 Hasil pengukuran kelayakan penerapan sistem.**

Spesifikasi laptop dipindahkan ke paragraf sebelum tabel:

> Pengukuran dilakukan pada laptop dengan AMD Ryzen 12 core logis, RAM 15,4 GB, Windows, tanpa akselerator grafis.

---

## P0-06 — Source note pada Gambar II.4 dan II.5 harus konsisten posisinya

### Masalah

Pada versi yang diperiksa, keterangan sumber terlihat tidak selalu mengikuti pola visual yang konsisten terhadap caption.

### Format yang disarankan

> **[Gambar]**  
> **Gambar II.x Judul singkat gambar.**  
> *Sumber: diadaptasi/disusun penulis berdasarkan ...*

Jangan menempatkan source note sebelum caption apabila aturan template laporan mengharuskan caption langsung berada di bawah gambar.

Lakukan pengecekan yang sama untuk **semua gambar Bab II**, bukan hanya II.4 dan II.5.

---

# 3. Prioritas P1 — Sangat Disarankan

## P1-01 — Pangkas caption Gambar VI.1

### Saat ini

Caption memuat:

- nama grafik;
- horizon;
- persentase zona A+B;
- interpretasi klinis.

### Disarankan

> **Gambar VI.1 Clarke Error Grid prediksi model produksi pada horizon +30 menit.**

Kemudian di paragraf:

> Sebanyak 94,70% titik berada pada zona A dan B.

Interpretasi mengenai keamanan klinis tetap berada di pembahasan.

---

## P1-02 — Pangkas caption Tabel VI.9

### Disarankan

> **Tabel VI.9 Hasil evaluasi penelusuran pada kasus nyata.**

Jangan memasukkan seluruh definisi:

- ground truth;
- kondisi benar;
- t+30;
- hukuman prediction error;
- Wilcoxon;

ke dalam caption.

Semua itu sudah menjadi bagian metode/pembahasan.

---

## P1-03 — Pangkas caption Tabel VI.12

Gunakan:

> **Tabel VI.12 Hasil pengukuran kelayakan penerapan sistem.**

Detail perangkat keras berada di paragraf.

---

## P1-04 — Pangkas caption Tabel VI.13

### Disarankan

> **Tabel VI.13 Sintesis temuan evaluasi lintas lapisan.**

Kalimat seperti:

> “Tabel ini merangkum temuan seluruh subbab evaluasi, termasuk yang tidak menguntungkan, agar batas manfaat metode ini terbaca jelas.”

dipindahkan menjadi paragraf.

---

## P1-05 — Evaluasi caption seluruh Bab VI

Audit seluruh caption Bab VI dengan prinsip:

> **Caption = apa objeknya.**  
> **Paragraf = apa hasilnya dan apa artinya.**

Jangan mengulang hasil numerik panjang di caption apabila angka tersebut sudah dibahas di paragraf.

---

## P1-06 — Tabel II.6 terlalu padat

### Masalah

Kolom:

- Peneliti;
- Fokus/pendekatan;
- Jenis data;
- Temuan;
- Keterbatasan

menjadi terlalu sempit.

### Perbaikan yang disarankan

Pertimbangkan:

| Penelitian | Pendekatan | Temuan utama | Gap terhadap penelitian |
|---|---|---|---|

Kolom **Jenis data** dapat dihilangkan apabila informasinya tidak esensial terhadap argumentasi posisi penelitian.

Tujuan perubahan adalah meningkatkan keterbacaan, bukan menghilangkan informasi penting.

---

## P1-07 — Tabel IV.5 terlalu panjang

### Masalah

Tabel pemetaan tahap ke komponen rancangan terbentang beberapa halaman.

Penggunaan `(lanjutan)` secara teknis memungkinkan, tetapi tabel menjadi berat untuk dibaca.

### Solusi yang disarankan

Pertimbangkan pemisahan logis menjadi:

> **Tabel IV.5 Pemetaan tahap luring ke komponen rancangan.**

dan

> **Tabel IV.6 Pemetaan tahap daring ke komponen rancangan.**

Nomor tabel berikutnya harus disesuaikan otomatis.

Jika tetap satu tabel, pastikan:

- header berulang pada setiap halaman;
- `(lanjutan)` konsisten;
- tidak ada baris terpotong;
- tidak ada row split yang membuat satu entri sulit dipahami.

---

## P1-08 — Tabel III.5 terlalu padat

Tabel matriks penilaian alternatif memiliki banyak kolom dan nama alternatif yang panjang.

### Perbaikan

- gunakan line break terkontrol;
- pertahankan ukuran font yang masih terbaca;
- atur lebar kolom `Keputusan` dan `Alternatif`;
- hindari pemenggalan kata yang tidak natural;
- pastikan legenda tidak terlalu dekat dengan tabel.

Jangan mengurangi font secara ekstrem hanya untuk memaksa tabel masuk halaman portrait.

---

## P1-09 — Tabel III.6 perlu diaudit keterbacaan

Tabel alasan pemilihan dan kelemahan memiliki teks panjang pada dua kolom deskriptif.

### Perbaikan

- gunakan kolom `Alasan pemilihan` dan `Kelemahan yang diakui` sebagai kolom fleksibel;
- jangan gunakan lebar sama untuk semua kolom;
- biarkan teks membungkus secara natural;
- bila perlu, pecah tabel berdasarkan kelompok keputusan.

---

## P1-10 — Tabel V.3 perlu audit kepadatan

Pastikan tabel struktur modul:

- tidak memiliki kolom terlalu sempit;
- nama modul/kode tidak bertabrakan;
- deskripsi tidak menjadi terlalu kecil;
- line break berada pada titik semantik yang wajar.

---

# 4. Prioritas P2 — Polishing

## P2-01 — Perbesar gambar yang masih terlalu kecil walaupun halaman masih menyediakan ruang

Periksa terutama:

- Gambar II.4;
- Gambar II.5;
- diagram Bab III;
- diagram Bab IV;
- screenshot UI Bab V.

Prinsip:

> **Jika pembaca harus memperbesar PDF secara signifikan untuk membaca teks pada gambar, gambar terlalu kecil.**

---

## P2-02 — Hindari whitespace besar akibat float

Periksa halaman yang memiliki:

> paragraf → ruang kosong besar → tabel/gambar

Jika tidak ada alasan tipografis, rapikan posisi float agar aliran:

> paragraf → visual → paragraf

terlihat alami.

---

## P2-03 — Seragamkan ukuran screenshot UI Bab V

Screenshot UI tidak harus memiliki ukuran identik secara absolut, tetapi harus memiliki:

- lebar visual yang konsisten;
- margin konsisten;
- skala teks UI yang terbaca;
- crop yang konsisten.

Untuk screenshot PWA:

- desktop dan mobile boleh ditampilkan sebagai pasangan hanya jika digunakan untuk membuktikan responsivitas;
- jangan menampilkan desktop + mobile untuk setiap halaman;
- browser chrome sebaiknya di-crop dari screenshot mobile;
- gunakan screenshot final yang paling konsisten dengan implementasi aktual.

---

## P2-04 — Audit semua caption terhadap daftar gambar/tabel

Setelah caption diubah:

1. compile ulang;
2. regenerate Daftar Gambar;
3. regenerate Daftar Tabel;
4. pastikan tidak ada judul panjang yang membuat daftar menjadi tidak rapi;
5. pastikan nomor halaman otomatis berubah dengan benar.

---

# 5. Aturan Khusus Caption yang Harus Dipakai

## Contoh buruk

> Gambar IV.x Alur rinci sistem, memisahkan tahap luring yang dijalankan sekali di luar aplikasi dari tahap daring yang dijalankan pada tiap konsultasi. Kotak bertanda menunjukkan empat tahap yang menjadi kontribusi metodologis, yaitu pengklasifikasi kondisi, perakitan kondisi klinis terstruktur, transformasi kueri, dan resolusi sitasi...

Masalah:

- terlalu panjang;
- sudah menjadi pembahasan;
- memuat metodologi;
- memuat interpretasi.

## Contoh baik

> **Gambar IV.x Alur rinci sistem luring dan daring.**

Kemudian paragraf:

> Gambar IV.x memperlihatkan pemisahan tahap luring dan daring. Tahap yang menjadi kontribusi metodologis ditandai pada diagram dan dijelaskan lebih lanjut pada paragraf berikut.

---

# 6. Aturan Khusus Tabel

## Jangan

Memperkecil seluruh tabel hingga font menjadi sangat kecil.

## Lakukan

1. Pangkas kata yang redundan.
2. Pendekkan judul kolom.
3. Atur lebar kolom berdasarkan kebutuhan.
4. Gunakan wrapping.
5. Pecah tabel jika memang mempunyai dua kelompok logis.
6. Gunakan landscape jika tabel tetap tidak terbaca.
7. Pastikan header diulang pada halaman lanjutan.

---

# 7. Audit Konsistensi Nomor Gambar/Tabel

Versi laporan yang diperiksa menunjukkan beberapa perubahan penomoran antar versi dokumen. Contohnya diagram alur rinci sistem pernah muncul sebagai nomor yang berbeda pada indeks/versi tertentu.

### Wajib dilakukan setelah semua editing

- [ ] Semua `\label{}` unik.
- [ ] Semua `\ref{}` mengarah ke objek yang benar.
- [ ] Tidak ada caption yang tertinggal dari versi lama.
- [ ] Daftar Gambar otomatis sesuai isi.
- [ ] Daftar Tabel otomatis sesuai isi.
- [ ] Tidak ada nomor gambar/tabel yang lompat.
- [ ] Tidak ada nomor ganda.
- [ ] Semua referensi dalam paragraf menunjuk ke nomor terbaru.

**Jangan memperbaiki nomor secara manual di teks. Gunakan cross-reference otomatis.**

---

# 8. Audit Source Note Gambar

Untuk setiap gambar yang bukan sepenuhnya hasil penelitian sendiri:

- [ ] Sumber disebut.
- [ ] Jika diadaptasi, gunakan formulasi `Diadaptasi dari ...`.
- [ ] Jika dibuat sendiri berdasarkan beberapa sumber, gunakan formulasi yang sesuai.
- [ ] Source note konsisten posisinya.
- [ ] Source note tidak menggantikan caption.
- [ ] Source note tidak ditempatkan di tengah paragraf.
- [ ] Sumber tidak dijejalkan ke caption apabila sebenarnya merupakan informasi terpisah.

---

# 9. Audit Screenshot UI Bab V

## Wajib

- [ ] Screenshot berasal dari implementasi final.
- [ ] Tidak ada browser chrome yang tidak diperlukan.
- [ ] Tidak ada data pasien yang tidak perlu ditampilkan.
- [ ] Identitas pasien pada elemen UI konsisten.
- [ ] Tidak ada state UI yang saling bertentangan.
- [ ] Ukuran screenshot cukup besar untuk membaca fungsi utama.
- [ ] Caption singkat.
- [ ] Setiap screenshot dijelaskan dalam paragraf.
- [ ] Tidak ada dua screenshot berturut-turut tanpa paragraf penjelas.

## Untuk PWA/responsivitas

Gunakan maksimal satu pasangan representatif:

> Desktop → paragraf → Mobile

Tidak perlu desktop dan mobile untuk setiap halaman.

---

# 10. Audit Halaman per Halaman — Prioritas

## Bab II

### Gambar II.4
- [ ] Periksa ukuran.
- [ ] Rapikan posisi source note.
- [ ] Pangkas caption jika lebih dari dua baris.

### Gambar II.5
- [ ] Periksa ukuran.
- [ ] Rapikan posisi source note.
- [ ] Pangkas caption jika lebih dari dua baris.

### Tabel II.6
- [ ] Periksa lebar kolom.
- [ ] Kurangi kepadatan.
- [ ] Pertimbangkan pengurangan kolom.

---

## Bab III

### Tabel III.4
- [ ] Periksa caption.
- [ ] Pastikan judul tidak memuat penjelasan metodologis yang terlalu panjang.

### Tabel III.5
- [ ] Periksa lebar kolom.
- [ ] Periksa line break.
- [ ] Pastikan tabel lanjutan memiliki header yang sama.
- [ ] Pastikan legenda tidak terlalu kecil.

### Tabel III.6
- [ ] Periksa lebar kolom deskriptif.
- [ ] Pastikan teks tidak terlalu kecil.

---

## Bab IV

### Diagram alur sistem
- [ ] Perbesar.
- [ ] Jika perlu gunakan landscape.
- [ ] Pangkas caption.
- [ ] Pindahkan penjelasan kontribusi metodologis ke paragraf.

### Sequence diagram
- [ ] Sederhanakan.
- [ ] Pastikan teks internal dapat dibaca.
- [ ] Pertimbangkan landscape atau lampiran.

### Tabel pemetaan tahap
- [ ] Periksa panjang tabel.
- [ ] Pertimbangkan pemisahan tahap luring/daring.
- [ ] Pastikan header berulang.

---

## Bab V

### Screenshot UI
- [ ] Gunakan versi implementasi final.
- [ ] Seragamkan skala.
- [ ] Crop browser chrome pada mobile.
- [ ] Pastikan state/identitas pasien konsisten.
- [ ] Jangan menumpuk screenshot tanpa paragraf.

### Tabel V.2
- [ ] Perbaiki text collision.
- [ ] Lebarkan kolom nama pustaka/package.
- [ ] Jangan mengecilkan font secara berlebihan.

### Tabel V.3
- [ ] Audit kepadatan.
- [ ] Periksa wrapping.

---

## Bab VI

### Gambar VI.1
- [ ] Pangkas caption.
- [ ] Pindahkan angka dan interpretasi ke paragraf.

### Gambar VI.2
- [ ] Jangan ditumpuk terlalu dekat dengan VI.3.
- [ ] Perbesar agar Clarke Error Grid terbaca.
- [ ] Pangkas caption.

### Gambar VI.3
- [ ] Beri ruang lebih besar.
- [ ] Pastikan label model/sumbu terbaca.
- [ ] Pangkas caption.

### Tabel VI.9
- [ ] Pangkas caption.
- [ ] Pindahkan definisi ground truth/prosedur ke paragraf.

### Tabel VI.12
- [ ] Perbaiki overflow kanan.
- [ ] Atur ulang lebar kolom.
- [ ] Pangkas caption.
- [ ] Pindahkan spesifikasi perangkat keras ke paragraf.

### Tabel VI.13
- [ ] Pangkas caption.
- [ ] Pindahkan interpretasi sintesis ke paragraf.

### Semua tabel/gambar Bab VI
- [ ] Tidak ada visual berurutan tanpa paragraf.
- [ ] Caption tidak menjadi paragraf pembahasan.
- [ ] Font tabel/gambar tetap terbaca.

---

# 11. Checklist Final Sebelum PDF Diserahkan

## Gambar

- [ ] Semua gambar dirujuk dalam teks.
- [ ] Tidak ada gambar → gambar tanpa paragraf.
- [ ] Tidak ada gambar → tabel tanpa paragraf.
- [ ] Tidak ada tabel → gambar tanpa paragraf.
- [ ] Tidak ada tabel → tabel tanpa paragraf.
- [ ] Semua gambar cukup besar untuk dibaca.
- [ ] Diagram kompleks tidak dipaksakan menjadi kecil.
- [ ] Screenshot UI cukup besar.
- [ ] Caption maksimal sekitar 1–2 baris jika memungkinkan.
- [ ] Source note konsisten.
- [ ] Tidak ada gambar keluar margin.

## Tabel

- [ ] Semua tabel dirujuk dalam teks.
- [ ] Tidak ada tabel keluar margin.
- [ ] Tidak ada text collision.
- [ ] Tidak ada teks terpotong.
- [ ] Font masih terbaca.
- [ ] Header tabel lanjutan konsisten.
- [ ] Tabel yang sangat panjang sudah dipertimbangkan untuk dipecah.
- [ ] Caption singkat.
- [ ] Penjelasan metodologis berada di paragraf, bukan caption.

## Caption

- [ ] Caption mengidentifikasi objek.
- [ ] Caption tidak menjadi mini-paragraf.
- [ ] Caption tidak memuat interpretasi panjang.
- [ ] Caption tidak memuat terlalu banyak angka.
- [ ] Caption tidak memuat metode eksperimen lengkap.
- [ ] Caption konsisten gaya bahasanya.
- [ ] Daftar Gambar/Tabel sudah diperbarui otomatis.

## Layout

- [ ] Tidak ada whitespace yang tidak wajar.
- [ ] Tidak ada halaman yang terlihat kosong karena float.
- [ ] Tidak ada gambar/tabel yang terlalu kecil hanya demi menghemat halaman.
- [ ] Margin konsisten.
- [ ] Nomor halaman benar.
- [ ] Semua cross-reference benar.
- [ ] Tidak ada halaman yang memiliki objek keluar batas.
- [ ] Tidak ada font yang berubah ukuran secara tidak sengaja.

---

# 12. Urutan Pengerjaan yang Disarankan

Kerjakan dalam urutan berikut agar tidak bolak-balik:

### Tahap 1 — Fix error layout nyata

1. Tabel V.2 — text collision.
2. Tabel VI.12 — overflow kanan.
3. Gambar IV.x — diagram terlalu kecil.
4. Sequence diagram — terlalu padat.
5. Gambar VI.2 dan VI.3 — pisahkan/beri ruang.

### Tahap 2 — Caption cleanup

6. Gambar IV.x.
7. Gambar VI.1.
8. Gambar VI.2.
9. Gambar VI.3.
10. Tabel VI.9.
11. Tabel VI.12.
12. Tabel VI.13.
13. Caption lain yang >2 baris.

### Tahap 3 — Table cleanup

14. Tabel II.6.
15. Tabel III.5.
16. Tabel III.6.
17. Tabel IV.5.
18. Tabel V.3.
19. Tabel VI lainnya yang masih padat.

### Tahap 4 — Source/caption consistency

20. Gambar II.4.
21. Gambar II.5.
22. Seluruh source note.
23. Seluruh daftar gambar.
24. Seluruh daftar tabel.

### Tahap 5 — Screenshot UI

25. Gunakan screenshot final.
26. Seragamkan ukuran.
27. Crop browser chrome.
28. Pastikan state UI konsisten.
29. Gunakan desktop + mobile hanya sebagai bukti responsivitas.

### Tahap 6 — Final compile

30. Compile PDF.
31. Periksa seluruh halaman secara visual.
32. Periksa Daftar Gambar.
33. Periksa Daftar Tabel.
34. Cari objek yang melewati margin.
35. Cari caption >2 baris.
36. Cari dua visual berurutan.
37. Cari tabel dengan teks bertabrakan.
38. Periksa ulang halaman yang sebelumnya bermasalah.

---

# 13. Kriteria Lulus Formatting

PDF dapat dianggap **lulus formatting** apabila:

> **Tidak ada gambar atau tabel yang bertumpuk tanpa paragraf penjelas, tidak ada visual yang terlalu kecil untuk dibaca, tidak ada tabel yang keluar margin atau mengalami text collision, caption tidak berubah menjadi paragraf pembahasan, source note konsisten, dan seluruh gambar/tabel memiliki alur narasi yang jelas.**

Fokus akhir bukan membuat laporan menjadi penuh ruang kosong atau memaksa semua objek masuk ke satu halaman, melainkan menghasilkan **keterbacaan dan kesinambungan narasi ilmiah**.

---

## Status Checklist

| Kelompok | Status awal | Target |
|---|---|---|
| Gambar bertumpuk | 🔴 Ada | 🟢 Tidak ada |
| Ukuran diagram | 🔴 Beberapa terlalu kecil | 🟢 Terbaca |
| Screenshot UI | 🟡 Perlu uniformisasi | 🟢 Konsisten |
| Caption gambar | 🔴 Beberapa terlalu panjang | 🟢 Singkat |
| Caption tabel | 🔴 Beberapa terlalu panjang | 🟢 Singkat |
| Tabel V.2 | 🔴 Collision | 🟢 Bersih |
| Tabel VI.12 | 🔴 Overflow | 🟢 Fit |
| Tabel II.6 | 🟠 Padat | 🟢 Terbaca |
| Tabel III.5 | 🟠 Padat | 🟢 Terbaca |
| Tabel IV.5 | 🟠 Sangat panjang | 🟢 Terstruktur |
| Source note | 🟠 Perlu konsistensi | 🟢 Seragam |
| Whitespace | 🟡 Perlu audit | 🟢 Proporsional |
| Daftar Gambar/Tabel | 🟡 Harus regenerate | 🟢 Sinkron |

---

# 14. Catatan Konsistensi yang Perlu Dicek Setelah Formatting

Walaupun dokumen ini berfokus pada formatting, finalisasi visual harus dilakukan terhadap **versi isi final yang sama**.

Secara khusus, sebelum screenshot UI dimasukkan:

- pastikan implementasi yang dideskripsikan di Bab V sama dengan sistem yang benar-benar ditampilkan;
- pastikan screenshot tidak berasal dari versi lama;
- pastikan nomor gambar yang disebut di teks sama dengan nomor caption final;
- pastikan Daftar Gambar dan Daftar Tabel dihasilkan ulang setelah seluruh perubahan.

**Jangan memperbaiki formatting dengan mengorbankan konsistensi isi.**

---

# 15. Prinsip Final

> **Satu visual harus punya fungsi. Satu caption harus singkat. Satu tabel harus terbaca. Dan setiap visual harus terhubung dengan narasi melalui paragraf yang substantif.**

Dokumen ini menjadi checklist final sebelum laporan Tugas Akhir dicetak atau dikirim untuk pemeriksaan akhir.

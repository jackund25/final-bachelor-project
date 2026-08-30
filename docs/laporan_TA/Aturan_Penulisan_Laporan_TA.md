ATURAN PENULISAN LAPORAN TUGAS AKHIR
Master Writing Rules — Tugas Akhir

Versi: 1.0
Tanggal: 25 Agustus 2026

============================================================
A. ATURAN STRUKTUR ANTAR-BAB
============================================================

1. REFERENSI ANTAR-BAB

- Bab yang lebih awal TIDAK BOLEH merujuk ke bab yang sesudahnya.
- Bab yang lebih akhir BOLEH merujuk ke bab-bab sebelumnya.
- Pengecualian: bagian Metodologi di Bab I diperbolehkan memberikan gambaran mengenai tahapan penelitian yang akan dilakukan.
- Prinsipnya: informasi boleh dibawa ke depan, tetapi tidak boleh dibocorkan dari masa depan ke masa lalu.

Contoh DILARANG:
"Metode tersebut akan dibahas pada Bab IV."
"Implementasi sistem akan dijelaskan pada Bab V."
"Hasil pengujian akan dibahas pada Bab VI."

Contoh BOLEH:
Bab IV: "Perancangan mekanisme tersebut didasarkan pada teori yang telah dijelaskan pada Bab II."
Bab V: "Implementasi mengikuti arsitektur yang telah dirancang pada Bab IV."
Bab VI: "Pengujian dilakukan terhadap implementasi sistem sebagaimana dijelaskan pada Bab V."
Bab VII: "Berdasarkan hasil evaluasi pada Bab VI, ..."

2. HINDARI FORWARD REFERENCE DI DALAM BAB

- Hindari kalimat seperti "pada subbab berikutnya akan dibahas..."
- Jika informasi masih berada dalam ruang lingkup bab yang sama, jelaskan substansinya secara langsung.

============================================================
B. PEMISAHAN FUNGSI SETIAP BAB
============================================================

3. BAB I = MOTIVASI DAN KONTEKS MASALAH
   Bab I harus menjawab:
   "Mengapa penelitian ini perlu dilakukan?"

Bab I berisi:

- konteks permasalahan;
- kondisi ideal;
- kondisi saat ini;
- gap;
- urgensi;
- rumusan masalah;
- tujuan;
- batasan;
- metodologi;
- kontribusi penelitian jika diperlukan;
- sistematika penulisan.

DILARANG memasukkan keputusan desain sistem secara rinci.

4. BAB II = TEORI DAN STUDI LITERATUR
   Bab II harus menjawab:
   "Apa yang perlu diketahui secara ilmiah untuk memahami masalah dan ruang solusi?"

Bab II berisi:

- teori;
- konsep;
- algoritma/metode;
- karakteristik pendekatan;
- penelitian terdahulu;
- kelebihan dan keterbatasan berdasarkan literatur.

Bab II TIDAK BOLEH menjadi tempat menetapkan solusi penelitian.

5. BAB II TIDAK BOLEH MENYELIPKAN KEPUTUSAN SOLUSI
   Jika terdapat beberapa alternatif metode, Bab II boleh membahas semuanya berdasarkan literatur.
   Keputusan metode yang dipilih harus muncul melalui analisis yang sesuai, bukan sekadar karena "penelitian ini menggunakan X".

Prinsip:
BAB II = apa yang dikatakan literatur.
BAB III = apa yang dibutuhkan masalah.
BAB III/IV = apa yang dipilih dan dirancang.

============================================================
C. ATURAN SUMBER DAN REFERENSI
============================================================

6. SETIAP KLAIM ILMIAH HARUS PUNYA DASAR
   Setiap klaim faktual/ilmiah harus dapat ditelusuri ke sumber yang sesuai.

Jangan menggunakan satu referensi sebagai sumber untuk klaim-klaim yang sebenarnya tidak didukung oleh sumber tersebut.

7. SITASI HARUS SESUAI DENGAN KLAIM
   Jika satu paragraf memiliki beberapa klaim dari sumber berbeda, gunakan sumber yang sesuai untuk masing-masing klaim.
   Jangan memaksakan satu sitasi untuk seluruh paragraf.

8. GAMBAR DAN TABEL BAB II WAJIB DIAUDIT TERHADAP SUMBER ASLI
   Untuk setiap gambar/tabel Bab II:

- pastikan benar-benar terdapat pada sumber;
- pastikan bentuk dan isi sesuai jika disebut sebagai gambar/tabel asli;
- jika diadaptasi, pastikan adaptasinya tidak mengubah makna;
- periksa istilah;
- periksa angka;
- periksa caption;
- pastikan "Sumber:" atau "Diadaptasi dari:" digunakan dengan tepat;
- pastikan sumber yang dikutip memang sumber asal gambar/tabel;
- pastikan interpretasi setelah gambar/tabel didukung oleh sumber.

9. BEDAKAN GAMBAR ASLI, ADAPTASI, DAN SINTESIS
   A. Gambar asli sumber:
   Nyatakan sumber dengan benar.

B. Gambar adaptasi:
Nyatakan "Diadaptasi dari ..." dan pastikan perubahan tidak mengubah makna ilmiah.

C. Gambar hasil sintesis penulis:
Jangan memberikan kesan bahwa gambar tersebut berasal dari sumber tertentu.

Bab II boleh menggunakan ilustrasi yang dibuat ulang oleh penulis berdasarkan sumber referensi. Ilustrasi tersebut tidak harus identik secara visual dengan sumber, tetapi substansi ilmiah, komponen, hubungan, alur, istilah, dan informasi yang direpresentasikan wajib sesuai dengan sumber. Setiap ilustrasi wajib melalui audit terhadap referensi aslinya.

============================================================
D. BAB III — ANALISIS
============================================================

10. ANALISIS HARUS BERANGKAT DARI MASALAH
    Urutan argumentasi:
    As-Is → masalah → kebutuhan → alternatif → analisis → keputusan.

Jangan menggunakan pola:
"Saya ingin menggunakan X → kemudian mencari alasan agar X terlihat cocok."

11. SETIAP KEBUTUHAN HARUS MEMPUNYAI DASAR
    Untuk setiap requirement, harus dapat dijawab:
    "Kebutuhan ini muncul karena masalah apa?"

Idealnya terdapat hubungan:
Masalah → Kebutuhan → Requirement → Rancangan → Implementasi → Evaluasi.

12. ALTERNATIF SOLUSI HARUS DINILAI BERDASARKAN KRITERIA
    Kriteria pemilihan harus berasal dari kebutuhan/tujuan penelitian.
    Hindari klaim seperti "metode X dipilih karena paling bagus" tanpa kriteria dan bukti.

13. BAB III BOLEH MENETAPKAN SOLUSI, TETAPI TIDAK BOLEH MASUK KE DETAIL IMPLEMENTASI
    Boleh:
    "Pendekatan X dipilih karena memenuhi kebutuhan A, B, dan C."

Tidak boleh:
"Pendekatan X menggunakan library Y versi Z, fungsi A(), dan endpoint B."

============================================================
E. BAB IV — PERANCANGAN
============================================================

14. BAB IV MENJAWAB "BAGAIMANA SOLUSI DIRANCANG?"
    Bab IV berisi:

- arsitektur;
- komponen;
- alur;
- data;
- hubungan antar-komponen;
- aturan pemrosesan;
- struktur input/output;
- algoritma secara konseptual;
- rancangan antarmuka;
- rancangan basis data;
- rancangan workflow.

Bab IV BUKAN tempat menjelaskan bagaimana sistem diprogram.

15. BAB IV TIDAK BOLEH MENGGUNAKAN DETAIL SOURCE CODE
    DILARANG:

- nama fungsi;
- nama file;
- nama class;
- nama variable;
- nama folder/path;
- nama script;
- endpoint;
- potongan kode;
- detail implementasi spesifik.

BOLEH:

- retriever;
- generator;
- query transformation;
- embedding;
- basis pengetahuan;
- komponen prediksi;
- pipeline retrieval;
- modul preprocessing;
- data store;
- antarmuka.

16. BAB IV HARUS MENGGUNAKAN ISTILAH ILMIAH/KEKONSEPTUALAN
    Rancangan harus dapat dipahami tanpa membuka source code.

17. SETIAP RANCANGAN HARUS TERHUBUNG DENGAN REQUIREMENT
    Idealnya terdapat pemetaan:
    Requirement → Komponen/alur rancangan.

18. GAMBAR BAB IV HARUS MENGGAMBARKAN RANCANGAN
    Gambar Bab IV harus menunjukkan desain konseptual/arsitektural.
    Screenshot implementasi dan detail tool aktual ditempatkan pada bagian implementasi jika relevan.

19. BEDAKAN STATUS ANGKA/PARAMETER
    Setiap angka/parameter harus jelas apakah:

- berasal dari literatur;
- merupakan parameter rancangan;
- merupakan parameter implementasi aktual.

============================================================
F. BAB V — IMPLEMENTASI
============================================================

20. BAB V HARUS MERUPAKAN REALISASI BAB IV
    Komponen penting pada Bab V harus memiliki asal dari rancangan Bab IV.
    Jika terjadi perubahan dari rancangan ke implementasi, perubahan harus dapat dijelaskan.

21. SETIAP KLAIM IMPLEMENTASI HARUS DAPAT DIAUDIT KE SOURCE CODE/ARTEFAK
    Untuk klaim seperti:

- model yang digunakan;
- parameter;
- top-k;
- alur;
- library;
- konfigurasi;
- fitur;
- output;
  harus tersedia bukti pada source code atau artefak implementasi.

22. JANGAN MENYAMAKAN "KODE ADA" DENGAN "FITUR BERJALAN"
    Keberadaan kode tidak otomatis membuktikan fitur tersedia.
    Fitur harus diverifikasi melalui eksekusi/pengujian yang sesuai.

23. FITUR YANG DIRENCANAKAN TETAPI TIDAK DIIMPLEMENTASIKAN HARUS DIKOREKSI DALAM NARASI
    Jika fitur ada di proposal/rancangan tetapi akhirnya dicabut:

- jangan mengklaim fitur tersebut tersedia;
- nyatakan perubahan jika relevan;
- gunakan implementasi final sebagai sumber kebenaran.

24. BAB V BOLEH MENGGUNAKAN ISTILAH TEKNIS IMPLEMENTASI
    Pada Bab V baru diperbolehkan menyebut:

- bahasa pemrograman;
- library;
- framework;
- API;
- database;
- nama model aktual;
- struktur modul;
- nama file;
- nama fungsi;
- konfigurasi;
- versi software;
- screenshot workflow;
- source code.

============================================================
G. AUDIT SOURCE CODE
============================================================

25. BAB V WAJIB MELALUI SOURCE CODE AUDIT
    Untuk setiap bagian implementasi, harus dapat dijawab:
    "Tunjukkan buktinya di source code."

Audit minimal:

- input;
- preprocessing;
- model;
- parameter;
- retrieval;
- generation;
- database;
- interface;
- error handling;
- evaluasi;
- deployment.

26. VERIFIKASI FUNGSIONAL
    Jika laporan menyatakan suatu fitur tersedia, harus ada bukti bahwa fitur tersebut benar-benar dapat dijalankan sesuai kebutuhan.

============================================================
H. KONSISTENSI BAB IV → V → VI
============================================================

27. TERMINOLOGI HARUS KONSISTEN
    Satu konsep penting harus memiliki istilah resmi yang konsisten lintas bab.

28. NAMA KOMPONEN HARUS KONSISTEN
    Jika Bab IV menyebut suatu komponen dengan nama tertentu, jangan mengganti nama komponen tersebut di Bab V/VI tanpa alasan yang jelas.

29. PARAMETER HARUS KONSISTEN
    Jika parameter rancangan berubah pada implementasi:

- catat perubahan;
- jelaskan alasan bila relevan;
- pastikan hasil evaluasi menggunakan konfigurasi yang benar-benar diuji.

30. ARSITEKTUR HARUS SESUAI SISTEM AKTUAL
    Jika implementasi berbeda dari diagram rancangan, diagram/narasi harus diperbaiki atau perbedaannya dijelaskan.

============================================================
I. BAB VI — EVALUASI
============================================================

31. EVALUASI HARUS MENGUJI KLAIM/KEBUTUHAN YANG SUDAH DITETAPKAN
    Hubungan ideal:
    Research Question → Requirement → Design → Evaluation Metric.

32. EVALUASI HARUS MENGGUNAKAN IMPLEMENTASI/KONFIGURASI YANG JELAS
    Jangan menyajikan hasil dari konfigurasi yang berbeda seolah-olah merupakan hasil sistem final.

Eksperimen alternatif diperbolehkan jika jelas statusnya sebagai:

- baseline;
- comparison;
- ablation;
- eksperimen tambahan.

33. JANGAN MENGUBAH SEJARAH RANCANGAN BERDASARKAN HASIL
    Pisahkan:
    Rancangan → Eksperimen → Hasil → Keputusan berdasarkan hasil.

Jangan menulis seolah-olah hasil evaluasi sudah diketahui ketika rancangan dibuat.

============================================================
J. ATURAN KLAIM
============================================================

34. BEDAKAN "DIRANCANG", "DIIMPLEMENTASIKAN", "DIUJI", DAN "TERBUKTI"

- dirancang = ada dalam desain;
- diimplementasikan = benar-benar dibuat;
- diuji = dijalankan dalam pengujian;
- terbukti = didukung oleh hasil evaluasi.

Jangan menggunakan istilah tersebut sebagai sinonim.

35. KLAIM "LEBIH BAIK" HARUS MEMPUNYAI BASELINE
    Jangan menulis "X lebih baik" tanpa:

- baseline;
- metrik;
- nilai pembanding;
- konteks perbandingan.

36. JANGAN MENGUBAH KORELASI MENJADI KAUSALITAS
    Jika penelitian hanya menunjukkan hubungan/korelasi, jangan menyimpulkan hubungan sebab-akibat tanpa dasar.

============================================================
K. ATURAN TABEL DAN GAMBAR GLOBAL
============================================================

37. SETIAP TABEL/GAMBAR HARUS MEMILIKI FUNGSI NARATIF
    Sebelum memasukkan tabel/gambar, harus dapat dijawab:
    "Mengapa pembaca membutuhkan ini?"

Setelahnya harus dapat dijawab:
"Apa yang harus dipahami pembaca dari ini?"

38. SETIAP TABEL/GAMBAR HARUS DIRUJUK DALAM NARASI
    Jangan menampilkan tabel/gambar tanpa pernah membahasnya.

39. NARASI HARUS MEMBACA ISI TABEL/GAMBAR
    Jangan hanya menulis:
    "Tabel III.3 menunjukkan perbandingan alternatif."
    Jelaskan temuan penting dari tabel/gambar tersebut.

40. SEMUA DATA DALAM TABEL/GAMBAR HARUS DAPAT DIVERIFIKASI
    Terutama:

- angka;
- statistik;
- konfigurasi;
- arsitektur;
- hasil eksperimen;
- sumber data.

============================================================
L. ATURAN REFERENSI
============================================================

41. PRIORITASKAN SUMBER PRIMER
    Preferensi:
    paper asli → standar resmi → dokumentasi resmi → review → sumber sekunder.

42. REFERENSI HARUS BENAR-BENAR MENDUKUNG TEKS
    Jangan menggunakan referensi hanya karena topiknya mirip.
    Yang harus diperiksa adalah apakah klaim yang ditulis benar-benar didukung oleh sumber.

43. GAMBAR/TABEL WAJIB MASUK REFERENCE AUDIT
    Buat audit yang mencatat:

- ID;
- klaim/gambar/tabel;
- sumber;
- halaman;
- kesesuaian;
- status.

============================================================
M. SINGLE SOURCE OF TRUTH
============================================================

44. TETAPKAN SUMBER KEBENARAN UNTUK SETIAP JENIS INFORMASI

## Informasi Source of Truth

Teori Sumber referensi
Kondisi masalah Data/literatur
Requirement Bab III
Desain Bab IV
Implementasi Source code/artefak
Hasil Output eksperimen
Kesimpulan Hasil evaluasi

45. IMPLEMENTASI FINAL MENJADI SUMBER KEBENARAN KLAIM SISTEM
    Bab IV adalah sumber kebenaran rancangan.
    Source code/artefak implementasi adalah sumber kebenaran implementasi.
    Hasil eksperimen adalah sumber kebenaran hasil.

============================================================
N. PERUBAHAN DARI PROPOSAL
============================================================

46. PROPOSAL BUKAN KONTRAK TEKNIS FINAL
    Jika terjadi perubahan:
    Proposal → Implementasi aktual
    laporan final mengikuti penelitian aktual.

Perubahan signifikan harus dapat dijelaskan secara ilmiah jika relevan.

47. JANGAN MEMBAWA FITUR PROPOSAL YANG SUDAH DICABUT
    Jika fitur hanya pernah direncanakan tetapi tidak direalisasikan, jangan menulis seolah-olah fitur tersebut tersedia pada sistem final.

============================================================
O. BAB VII — PENUTUP
============================================================

48. KESIMPULAN HARUS MENJAWAB RUMUSAN MASALAH
    Idealnya:
    RQ1 → jawaban berdasarkan hasil.
    RQ2 → jawaban berdasarkan hasil.
    RQ3 → jawaban berdasarkan hasil.

49. JANGAN MEMASUKKAN INFORMASI BARU DI KESIMPULAN
    Tidak boleh ada:

- metode baru;
- data baru;
- hasil baru;
- argumen baru
  yang belum pernah dibahas sebelumnya.

50. SARAN HARUS BERASAL DARI KETERBATASAN NYATA
    Gunakan pola:
    Keterbatasan → konsekuensi → saran.

============================================================
P. TRACEABILITY CHAIN
============================================================

51. SETIAP KLAIM PENTING HARUS DAPAT DITELUSURI
    Alur utama:

SUMBER REFERENSI
↓
BAB II — TEORI
↓
BAB III — MASALAH & KEBUTUHAN
↓
BAB III — ALTERNATIF & KEPUTUSAN
↓
BAB IV — RANCANGAN
↓
SOURCE CODE / DATA
↓
BAB V — IMPLEMENTASI
↓
EXPERIMENT
↓
BAB VI — EVALUASI
↓
BAB VII — KESIMPULAN

Jika suatu klaim melompati tahap tanpa dasar yang jelas, klaim tersebut harus diaudit.

============================================================
Q. ATURAN ABSOLUT
============================================================

52. JANGAN MENULIS APA YANG TIDAK DAPAT DIBUKTIKAN.

Jika berasal dari literatur:
→ tunjukkan sumbernya.

Jika merupakan rancangan:
→ tunjukkan dasar kebutuhannya.

Jika merupakan implementasi:
→ tunjukkan buktinya di source code/artefak.

Jika merupakan hasil:
→ tunjukkan hasil eksperimennya.

Jika merupakan kesimpulan:
→ tunjukkan hasil yang mendasarinya.

============================================================
R. 12 GOLDEN RULES
============================================================

1. Tidak boleh forward reference antar-bab, kecuali Metodologi Bab I.
   Bab sesudahnya boleh merujuk bab sebelumnya.

2. Bab I hanya motivasi dan konteks masalah, bukan desain sistem.

3. Bab II hanya teori dan penelitian terdahulu, bukan keputusan solusi.

4. Semua gambar/tabel Bab II wajib diaudit terhadap sumber aslinya.

5. Bab III harus menurunkan kebutuhan dari masalah, bukan dari solusi yang sudah diinginkan.

6. Bab IV adalah rancangan ilmiah, bukan implementasi teknis.

7. Bab IV tidak boleh menyebut nama fungsi, file, script, endpoint,
   atau detail source code.

8. Setiap rancangan Bab IV harus memiliki hubungan dengan requirement Bab III.

9. Bab V harus diaudit langsung terhadap source code/artefak implementasi.

10. Setiap gambar/tabel Bab IV harus diaudit terhadap sistem aktual.

11. Bab VI hanya boleh mengevaluasi sistem/konfigurasi yang benar-benar diuji,
    dengan baseline dan protokol yang jelas.

12. Semua bab harus membentuk traceability chain:
    referensi → teori → masalah → requirement → rancangan →
    implementasi → evaluasi → kesimpulan.

13. Bahasa Indonesia yang baik dan benar. Jangan gunakan kata "yang" setelah tanda koma.
    TIDAK ADA istilah "which is" atau "yang mana" bahasa Indonesia, terus terakhir jangan gunakan
    em-dash, kalau bertemu seperti itu ubah pola kalimatnya.

============================================================
END OF MASTER WRITING RULES
============================================================

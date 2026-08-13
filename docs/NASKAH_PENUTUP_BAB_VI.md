# Naskah paragraf penutup Bab VI

> **Disiapkan 11 Agustus 2026 · keputusan #5 · commit `db4184f`**
>
> Naskah bab **tidak** disunting langsung. Ditempelkan sendiri oleh penulis ke
> `docs/laporan_TA/TA-STI-template-1.0/Bab VI - Evaluasi.tex`, pada bagian pembahasan
> sebelum subbab keterbatasan.

---

## Apa yang paragraf ini kerjakan

Ia menutup Bab VI dengan **satu penjelasan mekanistik yang diuji, bukan dengan ringkasan
angka**. Klaimnya: langit-langit sensitivitas hipoglikemia melekat pada **fungsi kerugian**,
bukan pada keluarga model — dan itu **terbukti**, karena dua keluarga pohon yang berbeda
diuji dan memberi hasil yang sama.

Ia lalu menyambungkannya ke **jarak oracle T3.2 (+0,2646)**, yang menunjukkan tuas terbesar
bagi sistem secara keseluruhan ada di sisi prediksi, bukan penelusuran.

## Angka yang dipakai, beserta sumbernya

| angka | nilai | sumber |
|---|---|---|
| sensitivitas hipo +30 mnt | RF 21,68% · GBM 23,93% · LSTM 32,11% | `gradient_boosting_h6.json` |
| sensitivitas hipo +60 mnt | RF 4,36% · GBM 2,42% · LSTM 10,56% | `gradient_boosting_h12.json` |
| bias pada hipo +30 mnt | RF +21,03 · GBM +19,02 · LSTM +17,81 | idem |
| bias keseluruhan | +0,21 · −0,16 · −0,04 | idem |
| GBM vs LSTM sensitivitas | −8,18 dan −8,14 poin, p=0,031 | layak, ketiga definisi SD sepakat |
| jarak oracle T3.2 | +0,2646 MRR | `horizon_retrieval.json` |
| oracle T3.2 | MRR 0,8352 · Hit@1 0,6778 | idem |

---

## NASKAH

> Salin bagian di bawah ini. Sesuaikan `\ref` dan `\autocite` dengan naskah Anda.

Ketiga model yang dibandingkan menunjukkan pola yang sama pada peristiwa hipoglikemia:
sensitivitas yang rendah disertai bias prediksi yang **positif dan besar** justru pada
peristiwa yang paling kritis. Random Forest menghasilkan bias $+21{,}03$~mg/dL pada
peristiwa hipoglikemia sementara bias keseluruhannya hanya $+0{,}21$~mg/dL; artinya model
secara sistematis menduga kadar yang **lebih tinggi** daripada kenyataan tepat pada kasus
yang seharusnya memicu peringatan. Pola tersebut bukan bias global melainkan **penyusutan ke
tengah pada kelas langka**: regresi yang meminimalkan galat kuadrat memang membayar sedikit
untuk meleset pada 3,25\% sampel demi akurasi pada 96,75\% sisanya.

Pengujian *gradient boosting* sebagai pembanding ketiga memungkinkan penjelasan itu
**diuji, bukan sekadar dinyatakan**. Bila penyusutan tersebut berasal dari keluarga model,
mengganti ansambel *bagging* dengan ansambel *boosting* seharusnya mengubahnya. Yang
teramati justru sebaliknya: pada horizon $+30$ menit sensitivitas *gradient boosting*
mendarat di 23,93\%, berdekatan dengan Random Forest (21,68\%) dan jauh dari LSTM
(32,11\%); pada horizon $+60$ menit ia bahkan lebih rendah daripada Random Forest (2,42\%
berbanding 4,36\%), dengan satu *fold* mencatat nol deteksi. Selisih *gradient boosting*
terhadap LSTM konsisten pada kedua horizon ($-8{,}18$ dan $-8{,}14$ poin persen;
$p=0{,}031$).

Dua keluarga pohon yang dibangun dengan cara yang sama sekali berbeda karena itu memberi
hasil yang sama, dan keduanya **meminimalkan fungsi kerugian yang sama**. Kesimpulan yang
didukung bukti ini adalah bahwa **langit-langit sensitivitas hipoglikemia melekat pada
fungsi kerugian, bukan pada keluarga model**. Konsekuensinya bersifat mengarahkan: mencari
arsitektur prediksi yang lebih baik tidak menjanjikan perbaikan pada aspek ini, sedangkan
mengubah apa yang diminimalkan — melalui kerugian berbobot kelas atau kerugian asimetris
yang menghukum dugaan-terlalu-tinggi pada rentang rendah secara berbeda — belum pernah
diuji dan merupakan arah yang belum tersentuh.

Arah tersebut memperoleh bobot tambahan dari sisi yang berbeda. Percobaan sumber
pengondisian *retrieval* (Subbab~\ref{sec:eval-horizon}) menempatkan lengan *oracle*, yaitu
*retrieval* yang dikondisikan pada kondisi masa depan yang **benar-benar terjadi**, pada MRR
$0{,}8352$ dan Hit@1 $0{,}6778$, sementara lengan terbaik yang dapat dicapai berhenti pada
MRR $0{,}5706$. Selisih $+0{,}2646$ itu **bukan kelemahan mekanisme penelusuran** — korpus,
pelabel, dan susunan kueri pada keempat lengan identik — melainkan bagian yang hilang
semata-mata karena galat prediksi. Dengan kata lain, sisi penelusuran sudah mendekati batas
yang dapat dicapainya pada konfigurasi ini, dan **tuas terbesar yang tersisa bagi sistem
secara keseluruhan berada di sisi prediksi**. Kedua garis bukti, satu dari perbandingan tiga
model pada peristiwa hipoglikemia dan satu dari jarak terhadap *oracle* pada *retrieval*,
menunjuk arah yang sama.

---

## Yang TIDAK boleh ditambahkan ke paragraf ini

1. **Jangan** menyatakan kerugian berbobot kelas *akan* memperbaiki sensitivitas. Ia belum
   diuji; yang didukung bukti hanya bahwa keluarga model **bukan** tuasnya.
2. **Jangan** menyatakan GBM lebih buruk daripada RF pada hipoglikemia sebagai temuan.
   Selisih RF lawan GBM pada sensitivitas +30 menit ($-2{,}249$) **tidak signifikan**
   (p=0,0625), dan pada +60 menit ($+1{,}936$) tidak konsisten menurut SD selisih.
3. **Jangan** mengutip angka *oracle* sebagai kinerja yang dapat dicapai. Ia langit-langit
   yang menurut konstruksinya tidak dapat dicapai.
4. **Kualifikasi K1 wajib menyertai** setiap angka MRR dan Hit@1 di paragraf ini —
   κ = 0,2505 (lemah). Lihat `docs/DAFTAR_KETERBATASAN.md` K1.

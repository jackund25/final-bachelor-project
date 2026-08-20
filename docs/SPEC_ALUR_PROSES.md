# Spesifikasi alur proses sistem: pengelompokan tahap ilmiah

Dokumen mandiri untuk menjawab arahan pembimbing: alur sistem harus **dikelompokkan menurut
tahap baku pengembangan**, serta **di dalam tiap kotak harus terlihat prosesnya**, bukan nama
berkasnya.

Isi dokumen ini dua hal. Bagian 0 menjawab tiga pertanyaan yang beliau ajukan langsung.
Bagian 1 sampai 12 memerikan **sebelas kelompok proses** beserta proses di dalamnya, dengan
penamaan yang mengikuti istilah baku pada literatur pembelajaran mesin dan temu kembali
informasi, disertai rujukannya.

---

## 0. Tiga pertanyaan pembimbing, dijawab lebih dulu

### 0.1 Apakah ada *fine tuning*? **Tidak ada.**

Tidak ada satu pun bobot model bahasa yang diperbarui pada penelitian ini. Model bahasa
dipakai **apa adanya** melalui antarmuka pemrograman, sedangkan pengetahuannya tidak pernah
dipindahkan ke dalam bobot.

Empat alasan, seluruhnya berpijak pada batasan yang sudah ditetapkan:

1. **Batasan komputasi.** Seluruh komputasi inti harus berjalan pada laptop kelas konsumen
   tanpa akselerator grafis. Penalaan halus model bahasa tidak mungkin dijalankan di situ.
2. **Ukuran korpus.** Dua belas dokumen pedoman berjumlah 494 halaman terindeks jauh di
   bawah kebutuhan data penalaan halus, sehingga hasilnya akan menghafal alih-alih
   menggeneralisasi.
3. **Keterlacakan.** Ini alasan yang paling menentukan. Pengetahuan yang ditanam ke dalam
   bobot **tidak dapat ditelusuri kembali ke halaman sumbernya**, sedangkan KNF-08 menuntut
   tiap rekomendasi dapat diverifikasi sampai ke nomor halaman dokumen. Penambatan lewat
   penelusuran memenuhi tuntutan itu, penalaan halus tidak.
4. **Pemutakhiran pedoman.** Bila PERKENI menerbitkan revisi, sistem ini cukup mengindeks
   ulang korpus. Sistem yang ditalakan halus harus dilatih ulang.

Argumen ketiga inilah yang menjadi alasan pokok pemilihan arsitektur *Retrieval-Augmented
Generation* sejak awal \autocite{lewis2020,gao2023}.

### 0.2 Apakah ada *prompt engineering*? **Ada, dengan bentuk yang spesifik.**

Ada, tetapi perlu ditegaskan bahwa **kebaruan penelitian ini bukan di situ**. Rekayasa
prompt di sini berperan sebagai **pagar pengaman keluaran**, bukan sebagai mekanisme yang
diteliti. Enam bentuknya:

| Bentuk | Wujudnya pada sistem |
|---|---|
| Penetapan peran | Model ditetapkan sebagai asisten klinis pendukung keputusan dokter, bukan pemberi keputusan |
| Pembatasan sumber | Jawaban wajib bersandar pada blok konteks yang diberikan, tanpa pengetahuan luar |
| Protokol penanda sitasi | Sumber dirujuk **hanya** dengan penanda `[S1]` sampai `[Sn]` sesuai nomor blok konteks |
| Larangan angka rujukan | Model **dilarang** menulis nomor halaman, nomor bab, nomor tabel, atau tautan |
| Instruksi penolakan terkendali | Bila konteks kosong atau tidak cukup, model wajib menyatakannya dan dilarang mengarang rujukan |
| Dekode konservatif | Suhu 0,2 dan batas 700 token, menekan pemunculan angka dosis yang tidak berdasar |

**Satu keputusan rancangan di sini layak disampaikan sendiri, sebab ia melampaui rekayasa
prompt biasa.** Aturan "jangan menulis nomor halaman" hanyalah jaminan lunak, sebab model
bahasa dapat melanggarnya. Jaminan kerasnya adalah **nomor halaman sama sekali tidak pernah
dimasukkan ke dalam blok konteks**. Model tidak dapat menyalin angka yang tidak pernah
dilihatnya. Nomor halaman baru dilekatkan sesudah pembangkitan selesai, diambil dari metadata
potongan. Inilah salah satu dari empat kontribusi metodologis penelitian ini.

### 0.3 Jadi sistemnya apa?

Sistem ini adalah **sistem pendukung keputusan klinis** yang menggabungkan dua kelompok
metode yang selama ini berkembang terpisah:

1. **Pembelajaran mesin terbimbing** atas deret waktu fisiologis, untuk memprakirakan kadar
   glukosa beserta ketidakpastiannya dan menetapkan kelas kondisi yang akan datang.
2. ***Retrieval-Augmented Generation*** atas korpus pedoman klinis, untuk menghasilkan
   rekomendasi berbahasa alami yang tertambat pada dokumen.

**Sambungan di antara keduanya itulah kebaruannya.** Pada RAG baku, kueri penelusuran
bersifat **eksogen**, yaitu datang dari luar sistem berupa pertanyaan pengguna atau kondisi
yang sedang berlaku. Pada sistem ini, kueri bersifat **endogen**: ia **dibangkitkan dari
keluaran model prakiraan**, yakni dari kondisi yang **diprediksi akan terjadi**.

Karena itu **pembentukan kueri berdiri sebagai kelompok tahapnya sendiri**, dengan letak
**sesudah pemodelan, sebelum penelusuran**. Jawaban ini menjawab pertanyaan beliau secara
langsung: pembentukan kueri **bukan** bagian dari seleksi data. Seleksi data bekerja atas
data masukan sebelum pelatihan, sedangkan pembentukan kueri bekerja atas **keluaran model**
pada saat inferensi. Menempatkannya di seleksi data akan menghapus justru sifat antisipatif
yang menjadi pokok penelitian.

---

## 1. Prinsip pengelompokan

Sebelas kelompok, disusun mengikuti tahap baku pengembangan sistem pembelajaran mesin
\autocite{peffers2007} yang dipadukan dengan tahap baku sistem temu kembali informasi
\autocite{manning2008}.

Kelompok A sampai F membentuk **jalur data fisiologis**. Kelompok G membentuk **jalur
korpus**. Keduanya berjalan **luring**, yaitu sekali di luar aplikasi. Kelompok H sampai L
berjalan **daring**, yaitu pada tiap konsultasi.

> **Catatan penggambaran.** Pemisahan luring dan daring **tetap dipertahankan** pada gambar,
> sebab ia menyatakan apa yang dibayar sekali dan apa yang dibayar tiap konsultasi. Kesebelas
> kelompok digambar sebagai **kotak besar bernama tahap**, sedangkan proses di dalamnya sebagai
> **kotak kecil di dalam kotak besar itu**.

Tiga istilah yang sering tertukar dibedakan tegas di sini, sebab pembimbing menyebut
ketiganya:

| Istilah | Yang dikerjakannya | Kelompok |
|---|---|---|
| **Feature collection** (pemerolehan fitur) | Mengambil variabel yang **sudah terukur** dari kanal data mentah | C |
| **Feature engineering** (rekayasa fitur) | **Menurunkan variabel baru** yang tidak terukur langsung, memakai pengetahuan domain | D |
| **Feature selection** (seleksi fitur) | **Membuang** variabel yang tidak informatif, dengan kriteria terukur | E |

---

## 2. Kelompok A: Pengumpulan Data (*Data Collection*)

Pemerolehan seluruh bahan mentah, sebelum satu keputusan pemodelan pun diambil.

| Kode | Proses | Uraian |
|---|---|---|
| A1 | Akuisisi rekaman fisiologis multikanal | Pengambilan rekaman dua belas penyandang DMT1 dari dataset publik OhioT1DM \autocite{marling2020}, memuat kanal pemantauan glukosa kontinu, bolus insulin, asupan karbohidrat, serta latihan fisik |
| A2 | Akuisisi kanal pemantauan mandiri | Pengambilan kanal *finger-stick* dari dataset yang sama, dipakai menguji ketahanan kinerja pada kondisi pemantauan periodik |
| A3 | Kurasi korpus pedoman klinis | Penghimpunan dua belas dokumen pedoman terbitan resmi berjumlah 543 halaman \autocite{perkeni2021,ada2023}, disertai penyusunan manifes bibliografis yang memuat identitas dokumen beserta penyeimbang nomor halaman |
| A4 | Akuisisi data prospektif | Perekaman catatan klinis oleh dokter melalui antarmuka, dipakai pada saat sistem berjalan |

**Keluaran kelompok:** rekaman peristiwa mentah, korpus dokumen mentah, manifes bibliografis.

---

## 3. Kelompok B: Seleksi dan Prapemrosesan Data (*Data Selection and Preprocessing*)

Penentuan **data mana yang layak dipakai**, beserta penyeragaman bentuknya. Inilah tahap
yang paling banyak membuang data, sedangkan tiap pembuangan punya kriteria tertulis.

| Kode | Proses | Uraian |
|---|---|---|
| B1 | Penyelarasan temporal multikanal | Penyelarasan peristiwa yang stempel waktunya tidak sinkron ke kisi seragam lima menit dengan toleransi ±2,5 menit |
| B2 | Integrasi kanal | Penggabungan kanal terselaraskan menjadi satu deret waktu terpadu per pasien |
| B3 | Penapisan kelayakan segmen | Pembuangan jendela yang memuat jeda antar-baris melebihi enam langkah, yaitu 30 menit. **Jeda tidak diinterpolasi melainkan dibuang**, sebab jeda terpanjang pada dataset mencapai 118 jam sehingga interpolasi setara mengarang data |
| B4 | Imputasi terbatas | Interpolasi hanya atas runtun kosong sepanjang paling banyak enam langkah, berlaku bagi catatan manual |
| B5 | Partisi lintas-pasien | Pemisahan himpunan latih, kalibrasi, serta uji **menurut pasien**, bukan menurut waktu, sehingga tidak ada pasien yang muncul di dua himpunan sekaligus |
| B6 | Seleksi unit dokumen | Penapisan halaman depan (sampul, kata pengantar, daftar isi) beserta halaman yang isinya di bawah ambang jumlah karakter. Halaman depan dibuang sebab padat kata kunci topik tanpa isi klinis, sehingga akan mencemari penelusuran. Dari 543 halaman tersisa **494** yang diindeks |

**Keluaran kelompok:** deret waktu terpadu yang lolos kelayakan, partisi lintas-pasien, serta
himpunan halaman dokumen yang layak diindeks.

> **Mengapa B5 penting disebut.** Partisi menurut pasien adalah alasan seluruh angka pada
> penelitian ini dapat diklaim menggeneralisasi ke pasien baru, bukan hanya ke waktu baru
> pada pasien yang sama.

---

## 4. Kelompok C: Pemerolehan Fitur (*Feature Collection*)

Pengambilan variabel yang **sudah terukur** pada data, tanpa penurunan apa pun.

| Kode | Proses | Uraian |
|---|---|---|
| C1 | Ekstraksi variabel terukur | Pengambilan empat variabel yang benar-benar terekam: kadar glukosa, asupan karbohidrat, dosis insulin, serta intensitas aktivitas fisik |
| C2 | Pembentukan representasi jendela geser | Penyusunan jendela dua belas langkah lima menitan, yaitu satu jam ke belakang, sebagai satuan pengamatan model. Kanal pemantauan mandiri memakai enam langkah |

**Keluaran kelompok:** matriks jendela berisi variabel terukur.

---

## 5. Kelompok D: Rekayasa Fitur (*Feature Engineering*)

Penurunan variabel baru yang **tidak terukur langsung**, memakai pengetahuan fisiologi. Inilah
tahap tempat pengetahuan domain masuk ke dalam model.

| Kode | Proses | Uraian dan dasar teorinya |
|---|---|---|
| D1 | Penurunan fitur tren | Perhitungan selisih kadar glukosa atas tiga langkah, yaitu 15 menit. Satu titik kadar tidak menyatakan arah; nilai yang sedang naik dan yang sedang turun menuntut tindakan berlawanan |
| D2 | Pemodelan insulin aktif | Akumulasi bolus dengan peluruhan eksponensial orde pertama, tetapan waktu 240 menit, mengikuti lama kerja insulin analog kerja cepat 4 sampai 6 jam \autocite{perkeni2021insulin}. Menangkap penurunan yang sedang berlangsung tetapi belum terlihat pada angka |
| D3 | Pemodelan karbohidrat aktif | Akumulasi asupan dengan peluruhan eksponensial, tetapan waktu 180 menit. Nilainya **parameter rancangan yang ditetapkan empiris**, sebab laju penyerapan bergantung pada komposisi makanan sehingga tidak ada rentang tunggal yang berlaku umum \autocite{annan2022} |
| D4 | Penyandian siklik waktu | Transformasi jam menjadi pasangan sinus dan kosinus, sehingga pukul 23.00 dan pukul 00.00 berdekatan sebagaimana mestinya. Menangkap pola diurnal, khususnya *dawn phenomenon* |
| D5 | Transformasi sasaran | Pengubahan sasaran pembelajaran menjadi **selisih** kadar terhadap nilai jangkar, lalu rekonstruksi kembali ke nilai absolut sesudah prediksi |
| D6 | Penskalaan fitur | Penyeragaman skala antar-fitur, dengan penskala yang **disimpan bersama model** sehingga bentuk masukan saat inferensi identik dengan saat pelatihan |

**Dasar teoretis bentuk peluruhannya.** Peluruhan eksponensial orde pertama pada D2 dan D3
adalah penyederhanaan *minimal model* glukosa-insulin \autocite{bergman1979} beserta
pengembangannya \autocite{dallaman2014}, yang bentuk penuhnya menuntut parameter fisiologis
per pasien yang tidak tersedia pada data catatan harian.

**Keluaran kelompok:** vektor tujuh fitur, yaitu kadar glukosa, tren glukosa, insulin aktif,
karbohidrat aktif, aktivitas, serta dua komponen waktu dalam hari.

---

## 6. Kelompok E: Seleksi Fitur (*Feature Selection*)

Pembuangan variabel yang tidak informatif, dengan kriteria terukur dan tertulis.

| Kode | Proses | Uraian |
|---|---|---|
| E1 | Penyaringan varians mendekati nol | Pembuangan variabel tingkat stres, sebab pada seluruh dataset hanya tercatat pada tujuh peristiwa sehingga variansnya praktis nol dan tidak ada yang dapat dipelajari darinya. Variabel itu **tetap direkam** sebagai rekam jejak klinis tetapi tidak menjadi fitur |
| E2 | Penilaian kepentingan fitur | Pengukuran kontribusi tiap fitur memakai dua cara berdampingan, yaitu **penurunan ketakmurnian rerata** (*mean decrease in impurity*) yang melekat pada model berbasis pohon \autocite{breiman2001}, serta **kepentingan permutasi** yang tidak bergantung pada struktur model |
| E3 | Penegakan kontrak fitur | Pemeriksaan terprogram bahwa himpunan fitur saat inferensi **identik** dengan saat pelatihan, baik nama maupun urutannya, sehingga pergeseran diam-diam tidak mungkin terjadi |

**Keluaran kelompok:** himpunan tujuh fitur final beserta kontrak yang menegakkannya.

> **Mengapa dua cara pada E2.** Penurunan ketakmurnian bias terhadap variabel berkardinalitas
> tinggi, sedangkan kepentingan permutasi tidak. Memakai keduanya membuat kesimpulan
> kontribusi fitur tidak bergantung pada kelemahan salah satu alat ukur.

---

## 7. Kelompok F: Pemodelan (*Modelling*)

Pelatihan seluruh model. **Tiga model terpisah dilatih**, sehingga hal itu perlu terlihat pada gambar
sebab ketiganya sering dikira satu.

| Kode | Proses | Uraian |
|---|---|---|
| F1 | Pelatihan regresor prakiraan | Pelatihan model *gradient boosting* berbasis histogram \autocite{friedman2001,ke2017} untuk memprakirakan selisih kadar glukosa. **Satu model per horizon**, yaitu +30 menit dan +60 menit |
| F2 | Pelatihan regresor kuantil | Pelatihan sepasang model kuantil pada aras 0,025 dan 0,975, dipakai menaksir sebaran galat |
| F3 | Kalibrasi ketidakpastian | Pengubahan lebar antar-kuantil menjadi simpangan baku, lalu penerapan **prediksi konformal terpisah** (*split conformal prediction*) \autocite{angelopoulos2021} atas lipatan kalibrasi yang **terpisah dari lipatan latih maupun uji**, menghasilkan faktor pengali yang menjamin cakupan |
| F4 | Pelatihan pengklasifikasi kondisi sadar-biaya | Pelatihan pengklasifikasi tiga kelas atas fitur dan jendela yang sama, dengan **pembobotan kelas berimbang** sehingga kelas hipoglikemia yang jarang tidak tenggelam oleh kelas mayoritas |
| F5 | Persistensi artefak | Penyimpanan model, penskala, spesifikasi fitur, serta faktor kalibrasi sebagai satu bundel inferensi, sehingga sisi daring memuat artefak jadi dan tidak melatih apa pun |

**Dua hal yang wajib dinyatakan pada tahap ini.**

**Pertama, tidak ada penyetelan hiperparameter.** Hiperparameter dibiarkan pada nilai bawaan
pustaka kecuali benih acak. Ini **disengaja**: model pembanding juga tidak disetel, sehingga
menyetel salah satu saja akan membuat perbandingannya berat sebelah. Konsekuensinya, hasil
yang dilaporkan adalah **batas bawah** kemampuan metode.

**Kedua, dasar F4 bersifat klinis, bukan statistik.** Metrik yang memperlakukan kedua arah
kesalahan setara tidak sah dipakai memilih model di sini, sebab akibatnya tidak setara: kadar
terlalu tinggi merusak perlahan selama bertahun-tahun, sedangkan kadar terlalu rendah dapat
mencederai dalam hitungan menit \autocite{nathan2014}.

**Keluaran kelompok:** bundel inferensi berisi regresor, pengklasifikasi, faktor kalibrasi.

---

## 8. Kelompok G: Pembangunan Basis Pengetahuan (*Knowledge Base Construction*)

Jalur korpus, berjalan luring sejajar dengan kelompok C sampai F.

| Kode | Proses | Uraian |
|---|---|---|
| G1 | Ekstraksi teks per halaman | Pengubahan dokumen menjadi teks dengan **satuan halaman dipertahankan**, sebab nomor halaman adalah satuan sitasi yang dituntut KNF-08 |
| G2 | Penautan metadata bibliografis | Pelekatan identitas dokumen, lembaga penerbit, tahun, nomor halaman berkas, serta nomor halaman cetak pada tiap satuan teks |
| G3 | Segmentasi sadar-batas kalimat | Pemecahan teks menjadi potongan 500 karakter dengan tumpang-tindih 67, **dipotong pada batas kalimat**. Rancangan ini menuntaskan dua cacat sekaligus, yaitu potongan berakhir utuh serta tidak ada token yang terbuang melampaui jendela model representasi |
| G4 | Representasi vektor | Pengubahan tiap potongan menjadi vektor 384 dimensi memakai model *sentence embedding* \autocite{reimers2019}, berjalan pada prosesor |
| G5 | Pengindeksan ganda | Pembangunan **indeks leksikal** berbasis pembobotan Okapi BM25 \autocite{manning2008} sebagai jalur produksi, berdampingan dengan **indeks vektor** yang dipertahankan sebagai pembanding. Korpus akhir berisi 4.038 potongan |

**Keluaran kelompok:** basis pengetahuan terindeks beserta metadata sitasinya.

---

## 9. Kelompok H: Pembentukan Kueri (*Query Construction*) ★

**Inilah kelompok yang memuat kebaruan penelitian.** Ia berdiri sendiri, sesudah pemodelan
dan sebelum penelusuran.

| Kode | Proses | Uraian |
|---|---|---|
| H1 | Inferensi prakiraan | Penerapan bundel inferensi atas jendela terkini, menghasilkan kadar terprediksi pada +30 menit dan +60 menit |
| H2 | Kuantifikasi ketidakpastian | Penerapan faktor konformal, menghasilkan interval prediksi 95% yang cakupannya terukur |
| H3 | Inferensi kondisi terprediksi ★ | Penetapan kelas kondisi masa depan memakai pengklasifikasi hasil F4, **bukan** dengan menerapkan ambang atas nilai regresi |
| H4 | Perakitan kondisi klinis terstruktur ★ | Perakitan tingkat risiko, arah tren, serta tingkat kegentingan menjadi satu objek terstruktur, **seluruhnya diturunkan dari nilai terprediksi**. Objek ini satu-satunya penghubung antara kelompok prakiraan dan kelompok penelusuran, sehingga sambungannya dapat diuji terpisah |
| H5 | Deteksi divergensi | Pembandingan **kategori** kondisi terkini terhadap kategori kondisi terprediksi. Yang diuji perpindahan kategori, bukan besar selisih angkanya |
| H6 | Transformasi kueri terkondisi-prediksi ★ | Penyusunan kueri penelusuran dari kondisi yang **akan datang**, bukan dari kondisi yang sedang berlaku |

**Keputusan rancangan yang perlu terlihat.** Batas interval prediksi **sengaja tidak
diteruskan** ke pembentuk kueri. Rancangan yang meneruskannya sudah diuji dan **ditolak
berdasarkan pengukuran**: ia mencapai cakupan kondisi tertinggi tetapi mutu penelusurannya
runtuh, sebab menambahkan kondisi kedua membagi bobot istilah ke dua kosakata sekaligus
sehingga menggeser keluar potongan benar yang sudah ditemukan. Interval tetap dipakai, tetapi
sebagai peringatan klinis kepada dokter.

**Keluaran kelompok:** kueri terkondisi-prediksi beserta peringatan divergensi.

---

## 10. Kelompok I: Penelusuran (*Retrieval*)

| Kode | Proses | Uraian |
|---|---|---|
| I1 | Tokenisasi kueri | Penguraian kueri menjadi token yang sebanding dengan token indeks |
| I2 | Pembobotan leksikal | Perhitungan skor Okapi BM25 kueri terhadap **seluruh** potongan pada indeks \autocite{manning2008} |
| I3 | Pemeringkatan dan pemotongan | Pengurutan menurut skor lalu pengambilan lima potongan teratas |
| I4 | Penurunan mode terkendali | Bila indeks leksikal gagal dibangun, sistem beralih ke jalur vektor tetapi **menyatakan penurunan itu secara eksplisit beserta sebabnya**, sehingga ia tidak pernah mengaku menelusur secara leksikal sambil sesungguhnya menelusur secara padat |

**Dua jalur pembanding tetap ada di dalam kode dan perlu disebut pada tahap evaluasi, bukan
pada tahap ini**, yaitu penelusuran padat dengan penataan ulang *Maximal Marginal Relevance*
\autocite{carbonell1998}, serta penggabungan hibrida pada peringkat memakai *Reciprocal Rank
Fusion* \autocite{xiong2024}. Keduanya **tidak** dipakai produksi, sebab alasannya ditetapkan
oleh pengukuran.

**Keluaran kelompok:** lima potongan teratas beserta metadata sitasinya.

---

## 11. Kelompok J: Pembangkitan (*Generation*)

| Kode | Proses | Uraian |
|---|---|---|
| J1 | Perakitan konteks bertanda | Penyusunan kelima potongan menjadi blok konteks bertanda `[S1]` sampai `[S5]` beserta identitas dokumennya, **tanpa nomor halaman sama sekali** |
| J2 | Penerapan pagar prompt | Penerapan keenam bentuk rekayasa prompt pada Subbab 0.2, mencakup penetapan peran, pembatasan sumber, protokol sitasi, larangan angka rujukan, serta instruksi penolakan terkendali |
| J3 | Pembangkitan terkendali | Pemanggilan model bahasa dengan dekode konservatif, yaitu suhu 0,2 dan batas 700 token |
| J4 | Penjaminan penyangkalan | Pemeriksaan pascapembangkitan yang memastikan tiap keluaran membawa pernyataan bahwa keputusan klinis akhir berada pada dokter |
| J5 | Resolusi sitasi dari metadata ★ | Pelekatan nomor halaman **dari metadata potongan**, sesudah pembangkitan selesai, sehingga nomor halaman tidak mungkin dikarang model bahasa |
| J6 | Pemotongan kutipan pada batas kalimat | Pemastian bahwa kutipan yang ditampilkan kepada dokter berhenti di akhir kalimat |
| J7 | Cadangan lokal | Penyusunan rekomendasi dari templat lokal ketika layanan awan tidak tersedia |

**Keluaran kelompok:** rekomendasi berbahasa Indonesia yang tertambat, beserta daftar sumber
bernomor halaman.

---

## 12. Kelompok K: Evaluasi (*Evaluation*)

Kelompok ini **tidak berada pada jalur konsultasi**. Ia digambar terpisah, mengukur keluaran
kelompok F, I, serta J.

| Kode | Proses | Uraian |
|---|---|---|
| K1 | Evaluasi galat prakiraan | Pengukuran RMSE, MAE, serta MAPE terhadap nilai sebenarnya |
| K2 | Evaluasi keamanan klinis prakiraan | Penilaian dengan *Clarke Error Grid* \autocite{clarke1987}, yang menilai **akibat klinis** galat alih-alih besarnya |
| K3 | Evaluasi kalibrasi ketidakpastian | Pembandingan cakupan empiris terhadap aras nominal. **Bukan makin tinggi makin baik**, melainkan makin mendekati nominal makin baik |
| K4 | Evaluasi pengklasifikasi kondisi | Pengukuran sensitivitas serta nilai prediktif positif **per kelas**, sebab akurasi keseluruhan menyembunyikan kegagalan pada kelas minoritas |
| K5 | Evaluasi mutu penelusuran | Pengukuran *Mean Reciprocal Rank*, Hit@k, serta nDCG \autocite{manning2008} |
| K6 | Validasi silang lintas-pasien | Pengulangan seluruh pelatihan dan pengukuran pada enam lipatan, dengan model **dilatih ulang dari nol** pada tiap lipatan dan diuji pada pasien yang tak pernah dilihat |
| K7 | Evaluasi bebas kata kunci | Pengukuran *context precision* serta *context recall* \autocite{es2023}, dipakai sebagai alat ukur kedua yang **tidak berbagi sinyal** dengan pembobotan leksikal |
| K8 | Kontrol dan ablasi | Penjalanan kendali acak, kendali berderau, serta **kontrol batas atas** yang menerima kondisi masa depan sebenarnya, sehingga langit-langit metode terukur alih-alih diandaikan |
| K9 | Uji signifikansi | Uji peringkat bertanda Wilcoxon berpasangan atas keenam lipatan, disertai ukuran efek Cohen's *d_z* untuk rancangan berpasangan |
| K10 | Evaluasi kelayakan penerapan | Pengukuran waktu tiap tahap serta jejak memori pada perangkat sasaran tanpa akselerator grafis |
| K11 | Evaluasi keamanan keluaran | Pemeriksaan bahwa besaran klinis pada keluaran tertelusur ke sumbernya, tanpa tindakan salah arah |

**Keluaran kelompok:** seluruh angka Bab VI.

---

## 13. Kelompok L: Penyajian dan Pencatatan Keputusan

| Kode | Proses | Uraian |
|---|---|---|
| L1 | Penyajian terpadu | Penampilan prakiraan, interval, peringatan divergensi, rekomendasi, serta potongan sumbernya pada satu layar |
| L2 | Pernyataan keandalan kondisi | Penampilan ketepatan pengklasifikasi berdampingan dengan label kondisi, beserta pernyataan bahwa seluruh potongan ditelusur untuk **kondisi terprediksi** |
| L3 | Pencatatan keputusan klinis | Perekaman keputusan dokter beserta **potongan sumber yang persis dilihatnya**, sehingga keputusan dapat diaudit sampai ke halaman dokumen |

---

## 14. Ringkasan satu halaman

| Kelompok | Nama tahap | Jalur | Jumlah proses |
|---|---|---|---|
| A | Pengumpulan Data | luring | 4 |
| B | Seleksi dan Prapemrosesan Data | luring | 6 |
| C | Pemerolehan Fitur | luring | 2 |
| D | Rekayasa Fitur | luring | 6 |
| E | Seleksi Fitur | luring | 3 |
| F | Pemodelan | luring | 5 |
| G | Pembangunan Basis Pengetahuan | luring | 5 |
| **H** | **Pembentukan Kueri ★** | daring | 6 |
| I | Penelusuran | daring | 4 |
| J | Pembangkitan | daring | 7 |
| K | Evaluasi | terpisah | 11 |
| L | Penyajian dan Pencatatan Keputusan | daring | 3 |

**Empat kontribusi metodologis**, ditandai ★ pada kelompok H dan J:

1. Inferensi kondisi terprediksi lewat pengklasifikasi sadar-biaya (H3)
2. Perakitan kondisi klinis terstruktur dari nilai terprediksi (H4)
3. Transformasi kueri terkondisi-prediksi (H6)
4. Resolusi sitasi dari metadata (J5)

---

## 15. Aturan menggambar

Tetap berlaku, meneruskan spesifikasi sebelumnya.

**Dilarang muncul di dalam gambar:** nama berkas, nama fungsi, nama kelas, nama pustaka, nama
model, nama medan data, angka waktu, serta angka jumlah. Seluruhnya sudah dimuat tabel di
dalam naskah; mengulanginya di gambar menciptakan dua sumber kebenaran.

**Wajib dipertahankan:**

1. **Kotak besar bernama kelompok tahap**, dengan kotak kecil proses di dalamnya. Inilah
   perubahan pokok yang diminta pembimbing.
2. Pemisahan **luring** dan **daring** sebagai dua wilayah bernama.
3. **Label pada panah** yang menyebut objek data yang berpindah, bukan cara berpindahnya.
4. **Penandaan kotak kontribusi** dengan satu penanda konsisten.
5. Kelompok **K (Evaluasi) digambar terpisah** dari jalur konsultasi, dengan panah masuk dari
   kelompok F, I, serta J.

**Istilah yang dilarang sebab sudah dicabut dari penelitian:** *digital twin*, *twin*, *DT*,
*what-if*, *surrogate*, *tipe 2*, *T2DM*.

**Label panah antar-kelompok:**

| Panah | Label |
|---|---|
| A → B | rekaman peristiwa mentah |
| B → C | deret waktu terpadu |
| C → D | matriks jendela |
| D → E | vektor fitur kandidat |
| E → F | himpunan fitur final |
| F → H | bundel inferensi |
| G → I | indeks potongan |
| A → G | dokumen pedoman |
| H → I | kueri terkondisi |
| I → J | lima potongan teratas |
| J → L | rekomendasi beserta sumber |
| F, I, J → K | keluaran terukur |

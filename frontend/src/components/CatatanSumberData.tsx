"use client";

import { useState } from "react";

// Keterangan asal data dan status pengujian.
//
// Dokter penilai meminta kejelasan apakah sistem ini pernah dicoba pada pasien
// sungguhan. Jawabannya belum, dan jawaban itu harus terbaca di dalam aplikasi —
// bukan hanya pada laporan — karena penilaian dokter terhadap kelayakan klinis
// bergantung pada apakah ia tahu angka yang dilihatnya berasal dari dataset
// penelitian atau dari pasien yang ia tangani.
export default function CatatanSumberData() {
  const [terbuka, setTerbuka] = useState(false);

  return (
    <section className="catatan-data">

      <div className="catatan-data-ringkas">
        <span className="catatan-data-tanda">i</span>

        <p>
          <strong>Purwarupa penelitian.</strong> Sistem ini belum pernah diuji
          pada pasien nyata di layanan klinis.
        </p>

        <button
          type="button"
          className="catatan-data-tombol"
          onClick={() => setTerbuka((sebelumnya) => !sebelumnya)}
        >
          {terbuka ? "Tutup keterangan" : "Selengkapnya"}
        </button>
      </div>

      {terbuka && (
        <div className="catatan-data-isi">

          <p>
            <strong>Asal data.</strong> Data pasien yang tersedia di aplikasi ini
            berasal dari dataset penelitian publik OhioT1DM: 12 orang dewasa
            penyandang diabetes melitus tipe 1 di Amerika Serikat yang seluruhnya
            menggunakan pompa insulin dan sensor CGM. Data tersebut tidak diambil
            dari pasien di Indonesia dan tidak berasal dari rekam medis mana pun.
          </p>

          <p>
            <strong>Status pengujian.</strong> Penelitian ini bersifat{" "}
            <em>proof-of-concept</em>. Belum dilakukan uji klinis terkontrol,
            belum ada pasien nyata yang menggunakan sistem ini, dan keluarannya
            belum pernah dipakai untuk mengambil keputusan terapi pada pasien
            sungguhan. Yang sedang Anda nilai adalah kelayakan rancangan dan mutu
            keluarannya, bukan hasil penerapan klinis.
          </p>

          <p>
            <strong>Pasien yang Anda buat sendiri.</strong> Kode pasien baru yang
            Anda daftarkan pada sesi ini adalah data uji coba untuk keperluan
            evaluasi. Jangan memasukkan data pasien sungguhan maupun identitas
            yang dapat mengenali orang.
          </p>

          <p>
            <strong>Peran sistem.</strong> Sistem berperan sebagai pendukung
            keputusan dengan alur <em>doctor-mediated</em>: seluruh keluaran wajib
            ditinjau dokter, dan sistem tidak dimaksudkan mengambil keputusan
            klinis secara mandiri.
          </p>

        </div>
      )}

    </section>
  );
}

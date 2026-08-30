"use client";

import { useState } from "react";

import type { GlucoseSource } from "../lib/api";
import { buatPasien } from "../lib/patients";

const SEMUA_SUMBER: GlucoseSource[] = ["CGM", "FINGER_STICK"];

const NAMA_SUMBER: Record<GlucoseSource, string> = {
  CGM: "CGM (sensor kontinu)",
  FINGER_STICK: "Finger-stick (glukometer)",
};

interface Props {
  pasienAktif: string;
  daftarPasien: string[];
  onPilih: (kode: string) => void;
  // Dipanggil setelah pasien baru tersimpan, supaya halaman pemanggil memuat ulang
  // daftar pasien dan kanalnya dari sumber data — bukan menebaknya dari state lokal.
  onPasienBaru: (kode: string) => Promise<void> | void;
}

// Panel ganti pasien.
//
// Sebelumnya pemilihan pasien hanya berupa <select> tanpa judul yang terselip di
// dalam bagian "Glucose source", dan pendaftaran pasien baru tidak ada sama sekali.
// Dokter penilai melaporkan keduanya: tidak dapat memasukkan pasien baru, dan
// tombol ganti pasien tidak terlihat. Karena itu panel ini berdiri sendiri, diberi
// judul eksplisit, dan memisahkan dua maksud yang berbeda — memilih pasien yang
// sudah terdaftar, versus mendaftarkan pasien baru.
export default function PatientSwitcher({
  pasienAktif,
  daftarPasien,
  onPilih,
  onPasienBaru,
}: Props) {
  const [terbuka, setTerbuka] = useState(false);
  const [mode, setMode] = useState<"lama" | "baru">("lama");

  const [pilihan, setPilihan] = useState(pasienAktif);
  const [kodeBaru, setKodeBaru] = useState("");
  const [sumberBaru, setSumberBaru] =
    useState<GlucoseSource[]>(SEMUA_SUMBER);

  const [menyimpan, setMenyimpan] = useState(false);
  const [galat, setGalat] = useState<string | null>(null);
  const [pesan, setPesan] = useState<string | null>(null);

  const bukaPanel = () => {
    setPilihan(pasienAktif);
    setGalat(null);
    setPesan(null);
    setTerbuka(true);
  };

  const gunakanPasienLama = () => {
    if (!pilihan || pilihan === pasienAktif) {
      setTerbuka(false);
      return;
    }
    onPilih(pilihan);
    setTerbuka(false);
  };

  const simpanPasienBaru = async () => {
    const kode = kodeBaru.trim().toUpperCase();

    if (!kode) {
      setGalat("Kode pasien belum diisi.");
      return;
    }
    if (sumberBaru.length === 0) {
      setGalat(
        "Pilih minimal satu sumber pengukuran glukosa untuk pasien ini.",
      );
      return;
    }
    if (daftarPasien.includes(kode)) {
      setGalat(
        "Kode " + kode + " sudah dipakai. Pilih lewat mode Pasien lama, atau gunakan kode lain.",
      );
      return;
    }

    setMenyimpan(true);
    setGalat(null);
    setPesan(null);

    try {
      await buatPasien(kode, sumberBaru);
      await onPasienBaru(kode);
      setKodeBaru("");
      setPesan(
        "Pasien " + kode + " berhasil dibuat. Catat data glukosa lewat menu Logbook sebelum sistem dapat memprediksi.",
      );
      setTerbuka(false);
    } catch (err) {
      setGalat(
        err instanceof Error ? err.message : "Pasien baru gagal dibuat.",
      );
    } finally {
      setMenyimpan(false);
    }
  };

  const ubahSumber = (sumber: GlucoseSource) => {
    setSumberBaru((sebelumnya) =>
      sebelumnya.includes(sumber)
        ? sebelumnya.filter((item) => item !== sumber)
        : [...sebelumnya, sumber],
    );
  };

  return (
    <section className="kartu-pasien">

      <div className="kartu-pasien-ringkas">

        <div>
          <p className="eyebrow">Pasien</p>

          <p className="kartu-pasien-kode">
            {pasienAktif ? pasienAktif : "Belum ada pasien dipilih"}
          </p>

          <p className="muted">
            Seluruh prediksi dan rekomendasi di halaman ini merujuk pasien tersebut.
          </p>
        </div>

        <button
          type="button"
          className="primary-button"
          onClick={() => (terbuka ? setTerbuka(false) : bukaPanel())}
        >
          {terbuka ? "Tutup" : "Ganti / tambah pasien"}
        </button>

      </div>

      {pesan && !terbuka && (
        <p className="kartu-pasien-pesan">{pesan}</p>
      )}

      {terbuka && (
        <div className="kartu-pasien-panel">

          <div className="kartu-pasien-mode">
            <label>
              <input
                type="radio"
                name="mode-pasien"
                checked={mode === "lama"}
                onChange={() => {
                  setMode("lama");
                  setGalat(null);
                }}
              />
              Pasien lama (sudah terdaftar)
            </label>

            <label>
              <input
                type="radio"
                name="mode-pasien"
                checked={mode === "baru"}
                onChange={() => {
                  setMode("baru");
                  setGalat(null);
                }}
              />
              Pasien baru (daftarkan sekarang)
            </label>
          </div>

          {mode === "lama" ? (
            <div className="kartu-pasien-baris">
              <label htmlFor="pemilih-pasien">Pilih pasien</label>

              <select
                id="pemilih-pasien"
                value={pilihan}
                onChange={(event) => setPilihan(event.target.value)}
              >
                {daftarPasien.length === 0 && (
                  <option value="">Belum ada pasien terdaftar</option>
                )}
                {daftarPasien.map((kode) => (
                  <option key={kode} value={kode}>
                    {kode}
                  </option>
                ))}
              </select>

              <button
                type="button"
                className="primary-button"
                onClick={gunakanPasienLama}
                disabled={daftarPasien.length === 0}
              >
                Gunakan pasien ini
              </button>
            </div>
          ) : (
            <div className="kartu-pasien-form">

              <div className="kartu-pasien-baris">
                <label htmlFor="kode-pasien-baru">Kode pasien</label>

                <input
                  id="kode-pasien-baru"
                  className="pertanyaan-input"
                  value={kodeBaru}
                  placeholder="Contoh: P013"
                  maxLength={32}
                  onChange={(event) => setKodeBaru(event.target.value)}
                />
              </div>

              <p className="muted">
                Gunakan kode, bukan nama pasien. Aplikasi ini tidak menyimpan
                identitas yang dapat mengenali orang.
              </p>

              <fieldset className="kartu-pasien-sumber">
                <legend>
                  Sumber pengukuran glukosa yang akan dicatat
                </legend>

                {SEMUA_SUMBER.map((sumber) => (
                  <label key={sumber}>
                    <input
                      type="checkbox"
                      checked={sumberBaru.includes(sumber)}
                      onChange={() => ubahSumber(sumber)}
                    />
                    {NAMA_SUMBER[sumber]}
                  </label>
                ))}
              </fieldset>

              <p className="muted">
                Setelah pasien dibuat, prediksi belum langsung tersedia. Sistem
                memerlukan 12 pencatatan CGM atau 8 pencatatan finger-stick
                terlebih dahulu, yang diisi lewat menu Logbook.
              </p>

              <button
                type="button"
                className="primary-button"
                onClick={simpanPasienBaru}
                disabled={menyimpan}
              >
                {menyimpan ? "Menyimpan..." : "Buat pasien dan gunakan"}
              </button>

            </div>
          )}

          {galat && <p className="kartu-pasien-galat">{galat}</p>}

        </div>
      )}

    </section>
  );
}

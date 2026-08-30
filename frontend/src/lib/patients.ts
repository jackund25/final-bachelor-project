import { API_BASE_URL, headerApi, type GlucoseSource } from "./api";

export interface HasilPasienBaru {
  id: number | string;
  patient_code: string;
  glucose_sources: GlucoseSource[];
}

// Pendaftaran pasien MELALUI BACKEND, bukan lewat klien Supabase di peramban.
// Kunci publik yang dipakai peramban tunduk pada RLS dan tidak berhak menulis ke
// tabel `patients`; backend memegang kunci layanan. Selain itu kanal glukosa harus
// ikut dibuat pada transaksi yang sama, karena `save_logbook` menolak entri yang
// kanalnya belum ada.
export async function buatPasien(
  patientCode: string,
  glucoseSources: GlucoseSource[],
): Promise<HasilPasienBaru> {
  const response = await fetch(`${API_BASE_URL}/api/patients`, {
    method: "POST",
    headers: headerApi({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      patient_code: patientCode,
      glucose_sources: glucoseSources,
    }),
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(
      result.detail ?? "Pasien baru gagal dibuat.",
    );
  }

  return result.data as HasilPasienBaru;
}

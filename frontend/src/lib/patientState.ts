// Pasien aktif dipakai tiga tempat sekaligus: Dashboard, Logbook, dan lencana di
// Navbar. Sebelumnya masing-masing membaca localStorage sendiri-sendiri dan Navbar
// bahkan menampilkan "Patient P001" secara hardcoded, sehingga lencana bisa
// menyebut pasien yang berbeda dari pasien yang sedang dinilai. Modul ini
// menjadikan localStorage satu-satunya sumber kebenaran dan menyiarkan
// perubahannya, supaya ketiga tempat itu tidak pernah berselisih.

export const KUNCI_PASIEN_AKTIF = "last_patient_id";
export const EVENT_PASIEN_AKTIF = "pasien-aktif-berubah";

export function bacaPasienAktif(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(KUNCI_PASIEN_AKTIF) ?? "";
}

export function simpanPasienAktif(kode: string) {
  if (typeof window === "undefined" || !kode) return;

  window.localStorage.setItem(KUNCI_PASIEN_AKTIF, kode);
  window.dispatchEvent(
    new CustomEvent<string>(EVENT_PASIEN_AKTIF, { detail: kode }),
  );
}

// Mengembalikan fungsi pembatalan langganan, supaya komponen pemanggil dapat
// membersihkan diri pada useEffect.
export function langganPasienAktif(
  saatBerubah: (kode: string) => void,
): () => void {
  if (typeof window === "undefined") return () => {};

  const dariHalamanIni = (event: Event) => {
    saatBerubah((event as CustomEvent<string>).detail ?? "");
  };

  // `storage` hanya menyala pada TAB LAIN, tidak pada tab yang menulis. Karena itu
  // dua pendengar diperlukan: CustomEvent untuk tab ini, `storage` untuk tab lain
  // yang dibuka dokter secara berdampingan.
  const dariTabLain = (event: StorageEvent) => {
    if (event.key === KUNCI_PASIEN_AKTIF) {
      saatBerubah(event.newValue ?? "");
    }
  };

  window.addEventListener(EVENT_PASIEN_AKTIF, dariHalamanIni);
  window.addEventListener("storage", dariTabLain);

  return () => {
    window.removeEventListener(EVENT_PASIEN_AKTIF, dariHalamanIni);
    window.removeEventListener("storage", dariTabLain);
  };
}

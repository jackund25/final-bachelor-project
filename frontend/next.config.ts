import type { NextConfig } from "next";

// Asal backend yang dituju proksi. Dibaca di SISI SERVER saja, sehingga nilainya
// tidak ikut dipanggang ke bundel peramban dan boleh berubah tanpa build ulang.
const BACKEND_ORIGIN =
  process.env.BACKEND_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Proksi /api/* ke backend.
  //
  // SEBAB (24 Agustus 2026). Evaluasi dokter dijalankan dari penerapan LOKAL yang
  // diekspos lewat Cloudflare Tunnel, dan URL terowongan gratis berubah setiap kali
  // dijalankan. Tanpa proksi ini, peramban dokter harus memanggil backend langsung
  // lewat NEXT_PUBLIC_API_URL — nilai yang DIPANGGANG saat build, sehingga tiap sesi
  // menuntut build ulang frontend hanya karena URL terowongannya berganti.
  //
  // Dengan proksi, peramban hanya mengenal satu asal: asal frontend itu sendiri.
  // Cukup SATU terowongan, tidak ada build ulang, dan CORS tidak lagi terlibat
  // karena permintaan ke backend berjalan server-ke-server di dalam satu mesin.
  //
  // Penerapan Vercel TIDAK terpengaruh: di sana NEXT_PUBLIC_API_URL tetap disetel
  // ke URL backend publik, dan src/lib/api.ts memakainya lebih dulu sehingga
  // permintaan tidak pernah melewati jalur proksi ini.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_ORIGIN}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import {
  bacaPasienAktif,
  langganPasienAktif,
} from "../lib/patientState";

export default function Navbar() {
  const pathname = usePathname();

  // Lencana ini dulu bertuliskan "Patient P001" secara hardcoded, sehingga ia
  // menyebut pasien yang salah setiap kali dokter berganti pasien — persis
  // keluhan yang muncul pada sesi evaluasi. Nilainya sekarang mengikuti pasien
  // aktif yang sesungguhnya.
  //
  // Pembacaan ditunda ke useEffect, bukan nilai awal useState, karena
  // localStorage tidak ada saat perenderan di server dan perbedaan nilai antara
  // server dan peramban akan memicu galat hidrasi.
  const [pasienAktif, setPasienAktif] = useState("");

  useEffect(() => {
    setPasienAktif(bacaPasienAktif());
    return langganPasienAktif(setPasienAktif);
  }, []);

  return (
    <header className="site-navbar">
      <div className="navbar-inner">

        <Link href="/" className="navbar-brand">
          <div className="navbar-logo">
            +
          </div>

          <div className="navbar-brand-text">
            <span className="navbar-title">
              Glucose Monitor
            </span>

            <span className="navbar-subtitle">
              Clinical Decision Support
            </span>
          </div>
        </Link>

        <nav className="navbar-nav">

          <Link
            href="/"
            className={`navbar-link ${
              pathname === "/" ? "active" : ""
            }`}
          >
            Dashboard
          </Link>

          <Link
            href="/logbook"
            className={`navbar-link ${
              pathname === "/logbook" ? "active" : ""
            }`}
          >
            Logbook
          </Link>

          <Link
            href="/history"
            className={`navbar-link ${
              pathname === "/history" ? "active" : ""
            }`}
          >
            History
          </Link>

        </nav>

        <div className="navbar-patient">
          <span className="patient-status-dot" />
          <span>
            {pasienAktif ? `Pasien ${pasienAktif}` : "Pasien belum dipilih"}
          </span>
        </div>

      </div>
    </header>
  );
}
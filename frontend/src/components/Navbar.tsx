"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

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
          <span>Patient P001</span>
        </div>

      </div>
    </header>
  );
}
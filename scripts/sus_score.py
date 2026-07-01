#!/usr/bin/env python3
"""T8 (opsional) — Hitung skor SUS dari CSV respons.

Format CSV (satu baris per responden): kolom q1..q10 wajib (nilai 1-5).
Kolom lain (mis. responden, peran) diabaikan untuk perhitungan.

Contoh:
    python scripts/sus_score.py --input data/sus_responses.csv
    python scripts/sus_score.py --make-template data/sus_responses_template.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

POSITIVE = [1, 3, 5, 7, 9]   # item positif: nilai - 1
NEGATIVE = [2, 4, 6, 8, 10]  # item negatif: 5 - nilai
QCOLS = [f"q{i}" for i in range(1, 11)]


def grade_and_adjective(score: float) -> tuple[str, str]:
    if score >= 80.3:
        return "A", "Excellent"
    if score >= 68.0:
        return "B-C", "Good"
    if score >= 51.0:
        return "D", "OK/Marginal"
    return "F", "Poor"


def sus_row(row: pd.Series) -> float:
    total = sum(row[f"q{i}"] - 1 for i in POSITIVE) + sum(5 - row[f"q{i}"] for i in NEGATIVE)
    return total * 2.5


def make_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    demo = pd.DataFrame([
        {"responden": "R01", "peran": "dokter", **{f"q{i}": v for i, v in
         zip(range(1, 11), [4, 2, 4, 2, 4, 2, 5, 1, 4, 2])}},
        {"responden": "R02", "peran": "mahasiswa_kedokteran", **{f"q{i}": v for i, v in
         zip(range(1, 11), [5, 1, 4, 1, 5, 2, 4, 2, 4, 1])}},
    ])
    demo.to_csv(path, index=False)
    print(f"Template contoh ditulis: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Hitung skor SUS dari CSV respons")
    ap.add_argument("--input", help="CSV respons (kolom q1..q10)")
    ap.add_argument("--output-dir", default="results/sus")
    ap.add_argument("--make-template", metavar="PATH", help="Tulis CSV template contoh lalu keluar")
    args = ap.parse_args()

    if args.make_template:
        make_template(Path(args.make_template))
        return 0
    if not args.input:
        ap.error("--input wajib (atau gunакan --make-template)")

    df = pd.read_csv(args.input)
    missing = [c for c in QCOLS if c not in df.columns]
    if missing:
        print(f"ERROR: kolom hilang: {missing}. Wajib q1..q10.")
        return 1

    # Validasi nilai 1-5
    bad = ((df[QCOLS] < 1) | (df[QCOLS] > 5)).any(axis=1)
    if bad.any():
        print(f"ERROR: {int(bad.sum())} baris punya nilai di luar 1-5. Perbaiki dulu.")
        return 1

    df = df.copy()
    df["sus_score"] = df.apply(sus_row, axis=1)
    df["grade"] = df["sus_score"].map(lambda s: grade_and_adjective(s)[0])
    df["adjektif"] = df["sus_score"].map(lambda s: grade_and_adjective(s)[1])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    id_col = "responden" if "responden" in df.columns else df.columns[0]
    keep = [c for c in [id_col, "peran"] if c in df.columns] + ["sus_score", "grade", "adjektif"]
    df[keep].to_csv(out_dir / "sus_per_responden.csv", index=False)

    mean = df["sus_score"].mean()
    g, adj = grade_and_adjective(mean)
    summary = pd.DataFrame([{
        "n": len(df), "mean_sus": round(mean, 1), "std": round(df["sus_score"].std(ddof=1), 1) if len(df) > 1 else 0.0,
        "min": round(df["sus_score"].min(), 1), "max": round(df["sus_score"].max(), 1),
        "grade": g, "adjektif": adj,
    }])
    summary.to_csv(out_dir / "sus_summary.csv", index=False)

    print("=== SKOR SUS PER RESPONDEN ===")
    print(df[keep].to_string(index=False))
    print("\n=== RINGKASAN ===")
    print(summary.to_string(index=False))
    print(f"\nRata-rata SUS = {mean:.1f} ({g}, {adj}). Baseline rata-rata industri = 68.")
    print(f"Output -> {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

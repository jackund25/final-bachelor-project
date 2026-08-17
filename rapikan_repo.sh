#!/usr/bin/env bash
# Merapikan repositori: ganti nama gambar, arsipkan yang tidak dipakai, pisahkan
# dokumen kerja internal dari repositori.
#
# Dijalankan sekali dari akar repositori:
#     bash rapikan_repo.sh
#
# Seluruh operasi bersifat MEMINDAHKAN, bukan menghapus. Tidak ada berkas yang hilang;
# semuanya berakhir di arsip/ dan tetap dapat dibuka dari sana.
set -euo pipefail

AKAR="$(cd "$(dirname "$0")" && pwd)"
cd "$AKAR"
TA="docs/laporan_TA/TA-STI-template-1.0"

echo "== 1/5  Menyiapkan direktori arsip"
mkdir -p arsip/gambar arsip/tabel arsip/dokumen_kerja arsip/protokol

echo "== 2/5  Menyeragamkan nama gambar Clarke"
mv -f "$TA/images/clarke_grid_gbm_h6.png"   "$TA/images/Gambar_VI1_ClarkeGBM_h6.png"
mv -f "$TA/images/clarke_grid_gbm_h12.png"  "$TA/images/Gambar_VI2_ClarkeGBM_h12.png"
mv -f "$TA/images/comparison_bar.png"       "$TA/images/Gambar_VI3_ComparisonBar_h6.png"

echo "== 3/5  Mengarsipkan gambar dan tabel yang tidak lagi dirujuk"
for g in Gambar_VI1_ClarkeRF_h6.png Gambar_VI2_ClarkeRF_h12.png \
         Gambar_V5_UI_WhatIf.png gambar1.png; do
  [ -f "$TA/images/$g" ] && mv -f "$TA/images/$g" arsip/gambar/ || true
done
for t in longtable1.tex tabel1.tex; do
  [ -f "$TA/tables/$t" ] && mv -f "$TA/tables/$t" arsip/tabel/ || true
done
# Gambar_IV1_AlurRinciSistem.png SENGAJA TIDAK diarsipkan: sumber draw.io-nya baru
# dibuat dan gambar ini akan diganti hasil ekspornya.

echo "== 4/5  Memindahkan protokol percobaan"
for p in docs/PRAPENDAFTARAN_*.md; do
  [ -f "$p" ] && mv -f "$p" arsip/protokol/ || true
done

echo "== 5/5  Memindahkan dokumen kerja dan pembelajaran"
# docs/METHODOLOGY.md SENGAJA TIDAK dipindahkan: ia dokumen metodologi publik repositori ini.
for d in ALUR_SISTEM_FINAL.md ANALISIS_CHUNKING_T7.md ARGUMEN_GBM.md AUDIT_KEBUTUHAN.md \
         DAFTAR_KETERBATASAN.md HANDOFF.md KEPUTUSAN_DIAMBIL.md KONTEKS_SESI_2026-08-16.md \
         NASKAH_PENUTUP_BAB_VI.md PANDUAN_PENILAI_v2.md PELAJARAN_TA_REKAN.md \
         PROMPT_PERBAIKAN_DAN_REKAYASA.md PROMPT_UJI_COBA_LANJUTAN.md REVISI_BAB_II_4_1.md \
         RINGKASAN_KEPUTUSAN_PEMBIMBING.md SPEC_BAB4.md TEORI_HIBRIDA_BAB2.md \
         TEORI_HIBRIDA_BAB4.md SUS_kuesioner.md doc.md; do
  [ -f "docs/$d" ] && mv -f "docs/$d" arsip/dokumen_kerja/ || true
done
[ -d docs/journey ] && mv -f docs/journey arsip/dokumen_kerja/ || true

echo
echo "== Selesai. Ringkasan:"
echo "   arsip/gambar        : $(ls -1 arsip/gambar 2>/dev/null | wc -l) berkas"
echo "   arsip/tabel         : $(ls -1 arsip/tabel 2>/dev/null | wc -l) berkas"
echo "   arsip/protokol      : $(ls -1 arsip/protokol 2>/dev/null | wc -l) berkas"
echo "   arsip/dokumen_kerja : $(ls -1 arsip/dokumen_kerja 2>/dev/null | wc -l) berkas"
echo "   sisa di docs/       : $(ls -1 docs/*.md 2>/dev/null | wc -l) berkas .md"
echo
echo "LANGKAH BERIKUTNYA yang TIDAK dijalankan skrip ini, karena mengubah pelacakan git:"
echo "   git rm -r --cached arsip/ docs/"
echo "   git add docs/laporan_TA"
echo "Perintah itu mengeluarkan berkas kerja dari commit BERIKUTNYA. Berkas yang sudah"
echo "pernah di-commit TETAP ADA pada riwayat git dan tidak hilang karena .gitignore."

# Merapikan repositori: ganti nama gambar, arsipkan yang tidak dipakai, pisahkan
# dokumen kerja internal dari repositori.
#
# Dijalankan sekali dari akar repositori:
#     powershell -ExecutionPolicy Bypass -File .\rapikan_repo.ps1
#
# Seluruh operasi bersifat MEMINDAHKAN, bukan menghapus. Tidak ada berkas yang hilang;
# semuanya berakhir di arsip\ dan tetap dapat dibuka dari sana.
# Skrip aman dijalankan ulang: berkas yang sudah pindah akan dilewati.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$TA = "docs\laporan_TA\TA-STI-template-1.0"

function Pindahkan($dari, $ke) {
    if (Test-Path -LiteralPath $dari) {
        Move-Item -LiteralPath $dari -Destination $ke -Force
        Write-Host "   pindah : $dari"
    }
}

Write-Host "== 1/5  Menyiapkan direktori arsip"
foreach ($d in @("arsip\gambar", "arsip\tabel", "arsip\dokumen_kerja", "arsip\protokol")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Write-Host "== 2/5  Menyeragamkan nama gambar Clarke"
Pindahkan "$TA\images\clarke_grid_gbm_h6.png"  "$TA\images\Gambar_VI1_ClarkeGBM_h6.png"
Pindahkan "$TA\images\clarke_grid_gbm_h12.png" "$TA\images\Gambar_VI2_ClarkeGBM_h12.png"
Pindahkan "$TA\images\comparison_bar.png"      "$TA\images\Gambar_VI3_ComparisonBar_h6.png"

Write-Host "== 3/5  Mengarsipkan gambar dan tabel yang tidak lagi dirujuk"
foreach ($g in @("Gambar_VI1_ClarkeRF_h6.png", "Gambar_VI2_ClarkeRF_h12.png",
                 "Gambar_V5_UI_WhatIf.png", "gambar1.png")) {
    Pindahkan "$TA\images\$g" "arsip\gambar\"
}
foreach ($t in @("longtable1.tex", "tabel1.tex")) {
    Pindahkan "$TA\tables\$t" "arsip\tabel\"
}
# Gambar_IV1_AlurRinciSistem.png SENGAJA TIDAK diarsipkan: sumber draw.io-nya baru dibuat
# dan gambar ini akan diganti hasil ekspornya.

Write-Host "== 4/5  Memindahkan protokol percobaan"
Get-ChildItem -Path "docs" -Filter "PRAPENDAFTARAN_*.md" -File -ErrorAction SilentlyContinue |
    ForEach-Object { Pindahkan $_.FullName "arsip\protokol\" }

Write-Host "== 5/5  Memindahkan dokumen kerja dan pembelajaran"
# docs\METHODOLOGY.md SENGAJA TIDAK dipindahkan: ia dokumen metodologi publik repositori ini.
$kerja = @(
    "ALUR_SISTEM_FINAL.md", "ANALISIS_CHUNKING_T7.md", "ARGUMEN_GBM.md", "AUDIT_KEBUTUHAN.md",
    "DAFTAR_KETERBATASAN.md", "HANDOFF.md", "KEPUTUSAN_DIAMBIL.md", "KONTEKS_SESI_2026-08-16.md",
    "NASKAH_PENUTUP_BAB_VI.md", "PANDUAN_PENILAI_v2.md", "PELAJARAN_TA_REKAN.md",
    "PROMPT_PERBAIKAN_DAN_REKAYASA.md", "PROMPT_UJI_COBA_LANJUTAN.md", "REVISI_BAB_II_4_1.md",
    "RINGKASAN_KEPUTUSAN_PEMBIMBING.md", "SPEC_BAB4.md", "TEORI_HIBRIDA_BAB2.md",
    "TEORI_HIBRIDA_BAB4.md", "SUS_kuesioner.md", "doc.md"
)
foreach ($d in $kerja) { Pindahkan "docs\$d" "arsip\dokumen_kerja\" }
Pindahkan "docs\journey" "arsip\dokumen_kerja\"

function Cacah($path) {
    if (Test-Path -LiteralPath $path) {
        return (Get-ChildItem -LiteralPath $path -Force -ErrorAction SilentlyContinue | Measure-Object).Count
    }
    return 0
}

Write-Host ""
Write-Host "== Selesai. Ringkasan:"
Write-Host ("   arsip\gambar        : {0} berkas" -f (Cacah "arsip\gambar"))
Write-Host ("   arsip\tabel         : {0} berkas" -f (Cacah "arsip\tabel"))
Write-Host ("   arsip\protokol      : {0} berkas" -f (Cacah "arsip\protokol"))
Write-Host ("   arsip\dokumen_kerja : {0} berkas" -f (Cacah "arsip\dokumen_kerja"))
$sisa = (Get-ChildItem -Path "docs" -Filter "*.md" -File -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host ("   sisa di docs\       : {0} berkas .md  (seharusnya 1: METHODOLOGY.md)" -f $sisa)

Write-Host ""
Write-Host "LANGKAH BERIKUTNYA yang TIDAK dijalankan skrip ini, karena mengubah pelacakan git:"
Write-Host "   git rm -r --cached docs/"
Write-Host "   git add docs/laporan_TA docs/METHODOLOGY.md"
Write-Host ""
Write-Host "Perintah itu mengeluarkan berkas kerja dari commit BERIKUTNYA. Berkas yang sudah"
Write-Host "pernah di-commit TETAP ADA pada riwayat git dan tidak hilang karena .gitignore."

# buat_ai_snapshot_v2.ps1
# Membuat snapshot repo TA untuk di-upload ke ChatGPT.
# AMAN: TIDAK mengubah, memindahkan, atau menghapus file repo asli.
#
# Jalankan dari ROOT repo:
#   powershell -ExecutionPolicy Bypass -File .\buat_ai_snapshot_v2.ps1

$ErrorActionPreference = "Stop"

$Root = (Get-Location).Path
$Snapshot = Join-Path $Root "TA_AI_SNAPSHOT"
$ZipPath = Join-Path $Root "TA_AI_SNAPSHOT.zip"

$ExcludeDirs = @(
    ".git", ".cache", ".pytest_cache", "__pycache__",
    ".venv", "venv", "node_modules",
    "chroma_db", "chroma_db_eval", "chroma_db_multilingual",
    "logs", "TA_AI_SNAPSHOT"
)

$ExcludeExtensions = @(
    ".pyc", ".pyo", ".pkl", ".npz", ".npy", ".bin",
    ".sqlite", ".sqlite3",
    ".aux", ".bbl", ".bcf", ".blg", ".lof", ".log",
    ".lol", ".lot", ".out", ".run.xml", ".synctex", ".toc"
)

$ExcludeNames = @(".env", "TA_AI_SNAPSHOT.zip")

$IncludeRoots = @(
    "src", "app", "scripts", "notebooks", "evaluation", "results"
)

$IncludeRootFiles = @(
    "README.md", "LEARN.md", "requirements.txt", "config.yaml",
    "pytest.ini", ".gitignore", ".env.example", "run_app.py"
)

$IncludeDocs = @(
    "docs\METHODOLOGY.md",
    "docs\SPEC_ALUR_PROSES.md",
    "docs\SPEC_GAMBAR.md",
    "docs\SPEC_REVISI_GAMBAR.md"
)

$IncludeWorkingDocs = @(
    "docs\dokumen-kerja\ALUR_SISTEM_FINAL.md",
    "docs\dokumen-kerja\ARGUMEN_GBM.md",
    "docs\dokumen-kerja\AUDIT_KEBUTUHAN.md",
    "docs\dokumen-kerja\DAFTAR_KETERBATASAN.md",
    "docs\dokumen-kerja\HANDOFF.md",
    "docs\dokumen-kerja\KEPUTUSAN_DIAMBIL.md",
    "docs\dokumen-kerja\KONTEKS_SESI_2026-08-16.md",
    "docs\dokumen-kerja\PANDUAN_PENILAI_v2.md",
    "docs\dokumen-kerja\PROMPT_PERBAIKAN_DAN_REKAYASA.md",
    "docs\dokumen-kerja\RINGKASAN_KEPUTUSAN_PEMBIMBING.md",
    "docs\dokumen-kerja\SPEC_BAB4.md",
    "docs\dokumen-kerja\TEORI_HIBRIDA_BAB2.md",
    "docs\dokumen-kerja\TEORI_HIBRIDA_BAB4.md"
)

$IncludeLatex = @(
    "docs\laporan_TA\TA-STI-template-1.0\TA.tex",
    "docs\laporan_TA\TA-STI-template-1.0\Bab I - Pendahuluan.tex",
    "docs\laporan_TA\TA-STI-template-1.0\Bab II - Studi.tex",
    "docs\laporan_TA\TA-STI-template-1.0\Bab III - Analisis.tex",
    "docs\laporan_TA\TA-STI-template-1.0\Bab IV - Perancangan.tex",
    "docs\laporan_TA\TA-STI-template-1.0\Bab V - Implementasi.tex",
    "docs\laporan_TA\TA-STI-template-1.0\Bab VI - Evaluasi.tex",
    "docs\laporan_TA\TA-STI-template-1.0\Bab VII - Penutup.tex",
    "docs\laporan_TA\TA-STI-template-1.0\daftar-pustaka.bib"
)

function Should-Exclude([System.IO.FileInfo]$File) {
    if ($ExcludeNames -contains $File.Name) { return $true }
    if ($ExcludeExtensions -contains $File.Extension.ToLower()) { return $true }

    foreach ($dir in $ExcludeDirs) {
        if ($File.FullName -match [regex]::Escape("\$dir\")) {
            return $true
        }
    }
    return $false
}

function Copy-FileSafe([string]$RelativePath) {
    $Source = Join-Path $Root $RelativePath

    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        Write-Host "  [SKIP] Tidak ditemukan: $RelativePath" -ForegroundColor Yellow
        return
    }

    $File = Get-Item -LiteralPath $Source
    if (Should-Exclude $File) {
        Write-Host "  [SKIP] Dikecualikan: $RelativePath" -ForegroundColor DarkYellow
        return
    }

    $Destination = Join-Path $Snapshot $RelativePath
    $Parent = Split-Path -Parent $Destination

    # PENTING: buat folder tujuan SEBELUM Copy-Item.
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Copy-TreeSafe([string]$RelativeRoot) {
    $SourceRoot = Join-Path $Root $RelativeRoot

    if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
        Write-Host "  [SKIP] Folder tidak ditemukan: $RelativeRoot" -ForegroundColor Yellow
        return
    }

    $Files = Get-ChildItem -LiteralPath $SourceRoot -File -Recurse -Force |
        Where-Object { -not (Should-Exclude $_) }

    foreach ($File in $Files) {
        $Relative = $File.FullName.Substring($Root.Length).TrimStart("\")
        $Destination = Join-Path $Snapshot $Relative
        $Parent = Split-Path -Parent $Destination

        # BUG versi sebelumnya ada di sini:
        # folder parent belum tentu ada.
        New-Item -ItemType Directory -Force -Path $Parent | Out-Null

        Copy-Item -LiteralPath $File.FullName -Destination $Destination -Force
    }
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "       TA AI SNAPSHOT v2 - READ ONLY" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Root     : $Root"
Write-Host "Snapshot : $Snapshot"
Write-Host ""

# Jangan pernah menyentuh repo asli.
# Hanya hapus snapshot hasil script sebelumnya.
if (Test-Path -LiteralPath $Snapshot) {
    Write-Host "Membersihkan snapshot lama..." -ForegroundColor Yellow
    Remove-Item -LiteralPath $Snapshot -Recurse -Force
}

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

New-Item -ItemType Directory -Force -Path $Snapshot | Out-Null

try {
    Write-Host "[1/5] Source code..." -ForegroundColor Green
    foreach ($dir in $IncludeRoots) {
        Write-Host "  -> $dir"
        Copy-TreeSafe $dir
    }

    Write-Host "[2/5] File konfigurasi/root..." -ForegroundColor Green
    foreach ($file in $IncludeRootFiles) {
        Copy-FileSafe $file
    }

    Write-Host "[3/5] Dokumentasi desain/metodologi..." -ForegroundColor Green
    foreach ($file in $IncludeDocs) {
        Copy-FileSafe $file
    }
    foreach ($file in $IncludeWorkingDocs) {
        Copy-FileSafe $file
    }

    Write-Host "[4/5] Source laporan TA..." -ForegroundColor Green
    foreach ($file in $IncludeLatex) {
        Copy-FileSafe $file
    }

    $Manifest = @"
TA AI SNAPSHOT v2
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

Snapshot ini adalah SALINAN READ-ONLY untuk analisis/coding assistant.
Repo asli tidak dimodifikasi.

Sengaja tidak disertakan:
- .git/history
- virtual environment dan cache
- __pycache__
- ChromaDB
- raw dataset besar
- model binary (*.pkl)
- binary/cache (*.npz, *.npy, *.bin)
- PDF knowledge base
- artefak compile LaTeX
- .env/credential

Tujuan:
- memahami arsitektur source code
- debugging/refactoring
- memahami pipeline ML/RAG
- membaca konfigurasi dan dokumentasi
- meninjau hasil evaluasi JSON/CSV/MD
"@

    Set-Content -LiteralPath (Join-Path $Snapshot "SNAPSHOT_INFO.txt") `
        -Value $Manifest -Encoding UTF8

    Write-Host "[5/5] Menghitung ukuran dan membuat ZIP..." -ForegroundColor Green

    $Files = Get-ChildItem -LiteralPath $Snapshot -File -Recurse
    $TotalBytes = ($Files | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $TotalBytes) { $TotalBytes = 0 }

    $TotalMB = [math]::Round($TotalBytes / 1MB, 2)

    Compress-Archive `
        -Path (Join-Path $Snapshot "*") `
        -DestinationPath $ZipPath `
        -CompressionLevel Optimal `
        -Force

    $ZipSize = (Get-Item -LiteralPath $ZipPath).Length
    $ZipMB = [math]::Round($ZipSize / 1MB, 2)

    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "              SELESAI" -ForegroundColor Green
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "File dalam snapshot : $($Files.Count)"
    Write-Host "Ukuran folder       : $TotalMB MB"
    Write-Host "Ukuran ZIP          : $ZipMB MB"
    Write-Host ""
    Write-Host "Folder : $Snapshot"
    Write-Host "ZIP    : $ZipPath"
    Write-Host ""
    Write-Host "Repo asli TIDAK diubah." -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "GAGAL membuat snapshot." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Repo asli tidak diubah. Snapshot parsial boleh dihapus jika perlu:"
    Write-Host "  Remove-Item -LiteralPath `"$Snapshot`" -Recurse -Force"
    exit 1
}
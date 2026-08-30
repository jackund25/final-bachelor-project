#!/usr/bin/env python3
"""
collect_ta_source.py

Collect source code and configuration files from a TA/project repository into
a ZIP bundle for source-code -> thesis evidence auditing.

Design goals:
- Collect source/config broadly rather than guessing what is "used".
- Exclude generated environments, caches, package/vendor directories, binaries,
  large datasets, model weights, and common build artifacts.
- Redact secrets in .env files instead of copying their values.
- Produce:
    1) SOURCE_CODE_BUNDLE.zip
    2) MANIFEST.csv
    3) MANIFEST.md
    4) SUMMARY.json

Usage:
    python collect_ta_source.py
    python collect_ta_source.py /path/to/project
    python collect_ta_source.py /path/to/project --out audit_bundle
    python collect_ta_source.py /path/to/project --max-file-mb 5

The output directory itself is automatically excluded to avoid recursive
collection if it lives inside the project root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "virtualenv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".npm",
    ".yarn",
    ".pnpm-store",
    ".tox",
    ".nox",
    ".coverage",
    "htmlcov",
    ".terraform",
    "dist",
    "build",
    "out",
    "coverage",
    "site-packages",
    "vendor",
    "vendors",
    ".idea",
    ".vscode",   # editor settings are usually not needed for research audit
}

EXCLUDED_FILENAMES = {
    ".DS_Store",
    "Thumbs.db",
}

# Source/config/document types to prefer collecting.
INCLUDED_EXTENSIONS = {
    # Python / notebooks
    ".py", ".pyw", ".ipynb",
    # JS/TS/frontend/backend
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".html", ".css", ".scss", ".sass", ".less",
    # Other code / scripts
    ".java", ".kt", ".kts", ".go", ".rs", ".cpp", ".cc", ".cxx", ".c",
    ".h", ".hpp", ".cs", ".php", ".rb", ".r", ".R",
    ".sql", ".sh", ".bash", ".zsh", ".fish",
    ".bat", ".cmd", ".ps1",
    # Config / structured data
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties",
    ".xml",
    # Documentation / experiment notes
    ".md", ".markdown", ".txt", ".rst",
    ".tex", ".bib",
    # API / schema
    ".graphql", ".gql",
}

# Explicit files that are useful even if extensionless.
INCLUDED_FILENAMES = {
    "Dockerfile",
    "Containerfile",
    "Makefile",
    "Procfile",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "constraints.txt",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    ".dockerignore",
    ".gitignore",
    ".gitattributes",
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.example",
    ".env.template",
}

# Large/generated/data/model artifacts are excluded by extension.
EXCLUDED_EXTENSIONS = {
    # Python/build/package artifacts
    ".pyc", ".pyo", ".whl", ".egg",
    # Native/binary
    ".so", ".dll", ".dylib", ".exe", ".bin", ".dat",
    # Model weights/checkpoints
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".tflite",
    ".h5", ".hdf5", ".pb", ".joblib",
    # Large data stores/datasets
    ".csv", ".tsv", ".parquet", ".feather", ".arrow",
    ".sqlite", ".sqlite3", ".db", ".mdb",
    ".pkl", ".pickle",
    ".mat",
    # Archives/media/binaries
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff",
    ".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".ogg",
    ".pdf",
}

# These are often package-manager artifacts. We keep package manifests such
# as package.json and requirements.txt, but exclude lockfiles if desired by
# default only when they are huge; the script still allows them if small.
LOCKFILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
}

SECRET_PATTERNS = [
    # Common API keys/tokens. This is intentionally broad; only values are
    # redacted inside env-like files, not in source code.
    re.compile(r"(?i)^(\s*(?:export\s+)?[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|API[_-]?KEY|AUTH[_-]?TOKEN)\s*=\s*)(.*)$"),
]

TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "latin-1")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileRecord:
    path: str
    category: str
    extension: str
    size_bytes: int
    sha256: str
    action: str
    reason: str
    redacted: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_probably_text(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:8192]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    return True


def redact_env_text(text: str) -> tuple[str, bool]:
    changed = False
    output_lines = []

    for line in text.splitlines():
        new_line = line
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            output_lines.append(line)
            continue

        for pattern in SECRET_PATTERNS:
            match = pattern.match(line)
            if match:
                new_line = f"{match.group(1)}<REDACTED>"
                changed = True
                break

        output_lines.append(new_line)

    # Always remove a few common quoted-secret forms that may evade the
    # pattern above.
    joined = "\n".join(output_lines)
    replacements = [
        (re.compile(r'(?i)(AKIA[0-9A-Z]{16})'), "<REDACTED_AWS_KEY>"),
        (re.compile(r'(?i)(gh[pousr]_[A-Za-z0-9_]{20,})'), "<REDACTED_GITHUB_TOKEN>"),
        (re.compile(r'(?i)(sk-[A-Za-z0-9_-]{20,})'), "<REDACTED_API_KEY>"),
    ]
    for pattern, repl in replacements:
        joined, count = pattern.subn(repl, joined)
        if count:
            changed = True

    return joined + ("\n" if text.endswith("\n") else ""), changed


def read_text_best_effort(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, "unsupported text encoding")


def category_for(path: Path) -> str:
    name = path.name
    ext = path.suffix.lower()

    if name.startswith(".env"):
        return "secrets-config"
    if name in {"Dockerfile", "Containerfile"} or name.startswith("docker-compose") or name in {"compose.yml", "compose.yaml", ".dockerignore"}:
        return "deployment-config"
    if name in {
        "requirements.txt", "requirements-dev.txt", "requirements-test.txt",
        "constraints.txt", "Pipfile", "Pipfile.lock", "poetry.lock",
        "pyproject.toml", "setup.py", "setup.cfg",
        "package.json", "package-lock.json", "npm-shrinkwrap.json",
        "yarn.lock", "pnpm-lock.yaml", "bun.lockb",
    }:
        return "dependency-config"
    if ext in {".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties", ".json", ".xml"}:
        return "configuration"
    if ext in {".md", ".markdown", ".txt", ".rst", ".tex", ".bib"}:
        return "documentation"
    if ext == ".ipynb":
        return "notebook"
    if ext in {".sh", ".bash", ".zsh", ".fish", ".bat", ".cmd", ".ps1"}:
        return "script"
    if ext in {".html", ".css", ".scss", ".sass", ".less", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        return "frontend/backend-web"
    return "source-code"


def should_include(
    path: Path,
    max_file_bytes: int,
    include_docs: bool,
) -> tuple[bool, str]:
    name = path.name
    ext = path.suffix.lower()

    if name in EXCLUDED_FILENAMES:
        return False, "excluded OS/editor artifact"

    if ext in EXCLUDED_EXTENSIONS:
        return False, f"excluded extension {ext}"

    if name in INCLUDED_FILENAMES:
        pass
    elif ext in INCLUDED_EXTENSIONS:
        pass
    else:
        return False, "not a recognized source/config/document file"

    if not include_docs and ext in {".md", ".markdown", ".txt", ".rst", ".tex", ".bib"}:
        return False, "documentation collection disabled"

    try:
        size = path.stat().st_size
    except OSError:
        return False, "cannot stat file"

    if size > max_file_bytes:
        return False, f"exceeds max-file size ({max_file_bytes} bytes)"

    return True, "included"


def iter_files(root: Path, output_dir: Path) -> Iterable[Path]:
    """
    Walk the project without following symlinked directories.
    """
    for current_root, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_root)

        pruned = []
        for dirname in dirs:
            if dirname in EXCLUDED_DIRS:
                continue

            dpath = current / dirname

            # Avoid collecting output bundle if it is inside the project root.
            try:
                if dpath.resolve() == output_dir.resolve():
                    continue
            except OSError:
                pass

            # Avoid symlinked directories to external locations.
            if dpath.is_symlink():
                continue

            pruned.append(dirname)

        dirs[:] = pruned

        for filename in files:
            path = current / filename
            if path.is_symlink():
                continue
            yield path


def write_redacted_copy(src: Path, dst: Path) -> bool:
    """
    Copy text file; redact secrets only for .env-like files.
    Returns True if redaction changed content.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.name.startswith(".env"):
        dst.write_bytes(src.read_bytes())
        return False

    text = read_text_best_effort(src)
    redacted, changed = redact_env_text(text)
    dst.write_text(redacted, encoding="utf-8")
    return changed


def make_manifest_md(records: list[FileRecord], root: Path, out_dir: Path) -> str:
    included = [r for r in records if r.action == "included"]
    excluded = [r for r in records if r.action == "excluded"]

    by_category: dict[str, int] = {}
    for record in included:
        by_category[record.category] = by_category.get(record.category, 0) + 1

    lines = [
        "# TA Source Collection Manifest",
        "",
        f"- Project root: `{root}`",
        f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Included files: **{len(included)}**",
        f"- Excluded files: **{len(excluded)}**",
        "",
        "## Included by category",
        "",
    ]

    for category, count in sorted(by_category.items()):
        lines.append(f"- `{category}`: {count}")

    lines += [
        "",
        "## Included files",
        "",
        "| Path | Category | Size (bytes) | SHA-256 | Redacted |",
        "|---|---|---:|---|:---:|",
    ]

    for record in included:
        lines.append(
            f"| `{record.path}` | {record.category} | {record.size_bytes} | "
            f"`{record.sha256}` | {'yes' if record.redacted else 'no'} |"
        )

    lines += [
        "",
        "## Excluded files",
        "",
        "| Path | Reason | Size (bytes) |",
        "|---|---|---:|",
    ]

    for record in excluded:
        lines.append(
            f"| `{record.path}` | {record.reason} | {record.size_bytes} |"
        )

    lines += [
        "",
        "## Interpretation guidance",
        "",
        "- Excluded package/vendor/cache/model/data files are not necessarily irrelevant; they are excluded to keep the upload manageable.",
        "- Source/config files are collected broadly on purpose, including potentially unused artifacts such as Docker files.",
        "- After upload, determine whether each collected artifact is actually used by tracing imports, entry points, configs, scripts, and experiment outputs.",
        "- `.env`-like files are included only after secret redaction.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main collection
# ---------------------------------------------------------------------------

def collect(
    root: Path,
    out_dir: Path,
    max_file_mb: float,
    include_docs: bool,
) -> int:
    root = root.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    max_file_bytes = int(max_file_mb * 1024 * 1024)

    staging_dir = out_dir / "collected_files"
    staging_dir.mkdir(parents=True, exist_ok=True)

    records: list[FileRecord] = []

    print(f"[INFO] Project root : {root}")
    print(f"[INFO] Output dir   : {out_dir}")
    print(f"[INFO] Max file size: {max_file_mb:.2f} MB")
    print()

    for path in iter_files(root, out_dir):
        relative = path.relative_to(root)
        relative_str = relative.as_posix()

        try:
            size = path.stat().st_size
        except OSError as exc:
            records.append(
                FileRecord(
                    path=relative_str,
                    category="unknown",
                    extension=path.suffix.lower(),
                    size_bytes=0,
                    sha256="",
                    action="excluded",
                    reason=f"stat failed: {exc}",
                )
            )
            continue

        include, reason = should_include(
            path,
            max_file_bytes=max_file_bytes,
            include_docs=include_docs,
        )

        if not include:
            records.append(
                FileRecord(
                    path=relative_str,
                    category=category_for(path),
                    extension=path.suffix.lower(),
                    size_bytes=size,
                    sha256="",
                    action="excluded",
                    reason=reason,
                )
            )
            continue

        # If it is a text/source file, make sure it is actually readable.
        if not is_probably_text(path):
            records.append(
                FileRecord(
                    path=relative_str,
                    category=category_for(path),
                    extension=path.suffix.lower(),
                    size_bytes=size,
                    sha256="",
                    action="excluded",
                    reason="binary/non-text content",
                )
            )
            continue

        target = staging_dir / relative
        try:
            redacted = write_redacted_copy(path, target)
            digest = sha256_file(path)
        except Exception as exc:
            records.append(
                FileRecord(
                    path=relative_str,
                    category=category_for(path),
                    extension=path.suffix.lower(),
                    size_bytes=size,
                    sha256="",
                    action="excluded",
                    reason=f"copy failed: {exc}",
                )
            )
            continue

        records.append(
            FileRecord(
                path=relative_str,
                category=category_for(path),
                extension=path.suffix.lower(),
                size_bytes=size,
                sha256=digest,
                action="included",
                reason="included",
                redacted=redacted,
            )
        )

    # Write machine-readable manifest.
    manifest_csv = out_dir / "MANIFEST.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else [
            "path", "category", "extension", "size_bytes", "sha256",
            "action", "reason", "redacted"
        ])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    # Write markdown manifest.
    manifest_md = out_dir / "MANIFEST.md"
    manifest_md.write_text(
        make_manifest_md(records, root, out_dir),
        encoding="utf-8",
    )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "output_dir": str(out_dir),
        "max_file_mb": max_file_mb,
        "include_docs": include_docs,
        "included_files": sum(r.action == "included" for r in records),
        "excluded_files": sum(r.action == "excluded" for r in records),
        "included_bytes": sum(r.size_bytes for r in records if r.action == "included"),
        "redacted_files": sum(r.redacted for r in records),
        "notes": [
            "Source/config files are collected broadly, including potentially unused files.",
            "Generated environments, package/vendor directories, binaries, model weights, datasets, and large files are filtered.",
            "Env-like secrets are redacted before bundling.",
            "Use the manifest to understand what was included/excluded.",
        ],
    }
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Create ZIP.
    zip_path = out_dir / "SOURCE_CODE_BUNDLE.zip"
    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        # Bundle collected source/config.
        for file_path in staging_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(staging_dir))

        # Bundle manifests at root of ZIP.
        zf.write(manifest_csv, arcname="MANIFEST.csv")
        zf.write(manifest_md, arcname="MANIFEST.md")
        zf.write(out_dir / "SUMMARY.json", arcname="SUMMARY.json")

    # Remove staging directory; ZIP is the upload artifact.
    for file_path in sorted(staging_dir.rglob("*"), reverse=True):
        if file_path.is_file() or file_path.is_symlink():
            file_path.unlink(missing_ok=True)
        elif file_path.is_dir():
            try:
                file_path.rmdir()
            except OSError:
                pass
    try:
        staging_dir.rmdir()
    except OSError:
        pass

    print("Collection complete.")
    print(f"  ZIP     : {zip_path}")
    print(f"  Manifest: {manifest_md}")
    print(f"  CSV     : {manifest_csv}")
    print(f"  Summary : {out_dir / 'SUMMARY.json'}")
    print()
    print(f"  Included: {summary['included_files']} files")
    print(f"  Excluded: {summary['excluded_files']} files")
    print(f"  Redacted: {summary['redacted_files']} env-like files")
    print()
    print("[NEXT] Upload SOURCE_CODE_BUNDLE.zip to the chat.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect TA source code/configuration for evidence auditing."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Project root directory (default: current directory).",
    )
    parser.add_argument(
        "--out",
        default="TA_SOURCE_AUDIT",
        help="Output directory (default: TA_SOURCE_AUDIT).",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=5.0,
        help="Maximum individual file size to include (default: 5 MB).",
    )
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Exclude markdown/text/LaTeX documentation files.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out_dir = Path(args.out).expanduser()

    if not root.exists():
        print(f"[ERROR] Project root does not exist: {root}", file=sys.stderr)
        return 2

    if not root.is_dir():
        print(f"[ERROR] Project root is not a directory: {root}", file=sys.stderr)
        return 2

    if args.max_file_mb <= 0:
        print("[ERROR] --max-file-mb must be > 0", file=sys.stderr)
        return 2

    try:
        return collect(
            root=root,
            out_dir=out_dir,
            max_file_mb=args.max_file_mb,
            include_docs=not args.no_docs,
        )
    except KeyboardInterrupt:
        print("\n[ERROR] Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

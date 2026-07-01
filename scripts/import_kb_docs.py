#!/usr/bin/env python3
"""Import additional documents into manual_kb.json and optionally re-ingest to ChromaDB.

Supported formats:
- .txt
- .md
- .pdf (requires PyPDF2)
- .docx (requires python-docx)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _topic_from_name(path: Path) -> str:
    name = path.stem.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name.title() if name else "General"


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_md(path: Path) -> str:
    # Keep markdown content mostly intact; remove fenced code blocks to reduce noise.
    text = path.read_text(encoding="utf-8", errors="ignore")
    return re.sub(r"```[\s\S]*?```", "", text)


def _read_pdf(path: Path) -> str:
    try:
        import PyPDF2
    except Exception as exc:
        raise RuntimeError("PyPDF2 is required for PDF import") from exc

    pages: List[str] = []
    with path.open("rb") as handle:
        reader = PyPDF2.PdfReader(handle)
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    try:
        import docx
    except Exception as exc:
        raise RuntimeError("python-docx is required for DOCX import") from exc

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _read_txt(path)
    if suffix == ".md":
        return _read_md(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    raise ValueError(f"Unsupported extension: {suffix}")


def _load_json_array(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("KB JSON must be a list")
    return payload


def _doc_id_for(path: Path, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem).strip("_").lower() or "doc"
    return f"ext_{stem}_{digest}"


def _iter_files(input_dir: Path, recursive: bool, extensions: Sequence[str]) -> List[Path]:
    normalized = {f".{ext.lower().lstrip('.')}" for ext in extensions}
    pattern = "**/*" if recursive else "*"
    files = [p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in normalized]
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import extra docs into data/knowledge_base/manual_kb.json")
    parser.add_argument("--input-dir", default="data/knowledge_base/additional_docs", help="Folder containing new docs")
    parser.add_argument("--kb-file", default="data/knowledge_base/manual_kb.json", help="Target KB JSON file")
    parser.add_argument("--recursive", action="store_true", help="Scan input folder recursively")
    parser.add_argument("--extensions", default="txt,md,pdf,docx", help="Comma-separated extensions")
    parser.add_argument("--min-chars", type=int, default=120, help="Skip docs shorter than this length")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported without writing")
    parser.add_argument("--reingest", action="store_true", help="Run RAG ingest after importing")
    parser.add_argument("--reset", action="store_true", help="Reset Chroma collection if reingest is enabled")
    args = parser.parse_args()

    root = _repo_root()
    input_dir = (root / args.input_dir).resolve() if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    kb_file = (root / args.kb_file).resolve() if not Path(args.kb_file).is_absolute() else Path(args.kb_file)

    input_dir.mkdir(parents=True, exist_ok=True)
    kb_file.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_json_array(kb_file)
    existing_ids = {str(item.get("doc_id", "")) for item in existing}

    files = _iter_files(input_dir, recursive=args.recursive, extensions=[e.strip() for e in args.extensions.split(",") if e.strip()])

    added: List[Dict] = []
    skipped: List[str] = []

    for path in files:
        try:
            raw = _read_document(path)
            text = _normalize_text(raw)
            if len(text) < args.min_chars:
                skipped.append(f"{path.name} (too short)")
                continue

            doc_id = _doc_id_for(path, text)
            if doc_id in existing_ids:
                skipped.append(f"{path.name} (duplicate)")
                continue

            entry = {
                "text": text,
                "source": path.name,
                "topic": _topic_from_name(path),
                "doc_id": doc_id,
            }
            existing.append(entry)
            existing_ids.add(doc_id)
            added.append(entry)
        except Exception as exc:
            skipped.append(f"{path.name} ({exc})")

    print(f"Found files: {len(files)}")
    print(f"Imported: {len(added)}")
    print(f"Skipped: {len(skipped)}")

    if skipped:
        print("\\nSkip details:")
        for item in skipped:
            print(f"- {item}")

    if args.dry_run:
        print("\\nDry-run mode: no file changes written.")
        return 0

    with kb_file.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, ensure_ascii=False, indent=2)

    print(f"\\nUpdated KB file: {kb_file}")

    if args.reingest:
        sys.path.insert(0, str(root))
        from src.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline.ingest(reset_collection=args.reset)
        print(f"Re-ingest result: {result}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

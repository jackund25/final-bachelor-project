#!/usr/bin/env python3
"""CLI helper to ingest manual KB into ChromaDB.

Quick start (recommended — sentence-transformers, no server needed):
    python scripts/ingest_kb.py --reset

With Google embeddings:
    python scripts/ingest_kb.py --embed google --reset

With Ollama (legacy):
    python scripts/ingest_kb.py --embed ollama --reset
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest diabetes KB into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing Chroma collection before ingest (required when switching embed provider)",
    )
    parser.add_argument(
        "--embed",
        default=os.getenv("EMBED_PROVIDER", "sentence-transformers"),
        choices=["sentence-transformers", "google", "ollama"],
        help="Embedding provider (default: env EMBED_PROVIDER or sentence-transformers)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", "gemini"),
        choices=["gemini", "ollama", "template"],
        help="LLM provider — only affects advisory generation, not ingestion",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("Starting KB ingestion — embed=%s, llm=%s, reset=%s", args.embed, args.provider, args.reset)

    pipeline = RAGPipeline(llm_provider=args.provider, embed_provider=args.embed)
    result = pipeline.ingest(reset_collection=args.reset)

    logging.info("Ingest completed: %s", result)
    print("\n✓ Knowledge base ingested successfully.")
    print(f"  Documents : {result['documents']}")
    print(f"  Chunks    : {result['chunks']}")
    print(f"  Chroma dir: {result['persist_dir']}")
    print(f"  Embed     : {result['embed_provider']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DATASETS
from src.index import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for a conversational RAG dataset")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data_root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--embed_model",
        default=str(ROOT / "models" / "all-MiniLM-L6-v2"),
        help="Local SentenceTransformer directory (default: models/all-MiniLM-L6-v2)",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--max_passages",
        type=int,
        default=None,
        help="Optional cap on raw passages before chunking (recommended for qrecc/topiocqa)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = build_index(
        dataset=args.dataset,
        data_root=args.data_root,
        embed_model=args.embed_model,
        device=args.device,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        max_passages=args.max_passages,
    )
    print(f"index saved to {out}")


if __name__ == "__main__":
    main()

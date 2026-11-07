#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DATASETS, iter_examples
from src.generate import Generator
from src.index import FaissStore
from src.matching import hit_is_gold, source_id
from src.query import QUERY_MODES, build_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conversational RAG baseline")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data_root", type=Path, default=ROOT / "data")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--query_mode", default="question", choices=QUERY_MODES)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--max_history_turns", type=int, default=6)
    parser.add_argument("--llm_model", type=Path, default=ROOT / "models" / "Qwen2.5-3B")
    parser.add_argument(
        "--embed_model",
        default=None,
        help="Local SentenceTransformer directory. Default: models/all-MiniLM-L6-v2",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL path under results/",
    )
    parser.add_argument(
        "--retrieve_only",
        action="store_true",
        help="Skip LLM generation; only print retrieval hits",
    )
    args = parser.parse_args()

    store = FaissStore(
        dataset=args.dataset,
        data_root=args.data_root,
        embed_model=args.embed_model,
        device=args.device,
    )
    generator = None if args.retrieve_only else Generator(str(args.llm_model), device=args.device)

    out_path = args.output
    if out_path is None:
        out_path = (
            ROOT
            / "results"
            / f"{args.dataset}_{args.split}_{args.query_mode}_k{args.top_k}.jsonl"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for example in iter_examples(
        args.dataset,
        args.data_root,
        args.split,
        max_examples=args.max_examples,
    ):
        query = build_query(
            example,
            mode=args.query_mode,
            max_history_turns=args.max_history_turns,
        )
        hits = store.search(query, top_k=args.top_k)
        passages = [p for p, _ in hits]
        prediction = ""
        if generator is not None:
            prediction = generator.generate(
                question=example.question,
                history=example.history_text(max_turns=args.max_history_turns),
                passages=passages,
                max_new_tokens=args.max_new_tokens,
            )

        rows.append(
            {
                "example_id": example.example_id,
                "question": example.question,
                "query": query,
                "gold_answer": example.answer,
                "prediction": prediction,
                "retrieved": [
                    {
                        "pid": p.pid,
                        "source_id": source_id(p),
                        "score": score,
                        "is_gold": hit_is_gold(
                            p,
                            example,
                            store.chunk_size,
                            store.chunk_overlap,
                        ),
                        "title": p.title,
                        "text": p.text[:300],
                    }
                    for p, score in hits
                ],
                "gold_passage_ids": example.gold_passage_ids,
            }
        )
        n = len(rows)
        print("=" * 60)
        print(f"[{n}] {example.example_id}")
        print("Q:", example.question)
        print("query_mode=", args.query_mode, "->", query[:180].replace("\n", " | "))
        print("gold:", example.answer[:180])
        if prediction:
            print("pred:", prediction[:180])
        print("top1:", hits[0][0].pid if hits else None, hits[0][1] if hits else None)

    out_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} examples to {out_path}")


if __name__ == "__main__":
    main()

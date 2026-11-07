from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from src.schema import Example, Passage


def iter_topiocqa_passages(data_root: Path) -> Iterator[Passage]:
    data_dir = data_root / "topiocqa-collection" / "data"
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    for path in files:
        table = pq.read_table(path, columns=["id", "contents"])
        ids = table.column("id").to_pylist()
        texts = table.column("contents").to_pylist()
        for pid, text in zip(ids, texts):
            text = (text or "").strip()
            if not text:
                continue
            yield Passage(pid=str(pid), text=text)


def _split_path(data_root: Path, split: str) -> Path:
    mapping = {
        "train": "topiocqa_train.jsonl",
        "validation": "topiocqa_valid.jsonl",
        "valid": "topiocqa_valid.jsonl",
        "dev": "topiocqa_valid.jsonl",
        "test": "topiocqa_valid.jsonl",  # official release has train/valid
    }
    if split not in mapping:
        raise ValueError(f"topiocqa split must be one of {list(mapping)}; got {split}")
    path = data_root / "topiocqa" / mapping[split]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _normalize_history(context) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    if not context:
        return history
    for i, utt in enumerate(context):
        role = "user" if i % 2 == 0 else "agent"
        history.append((role, str(utt).strip()))
    return history


def iter_topiocqa_examples(data_root: Path, split: str) -> Iterator[Example]:
    path = _split_path(data_root, split)
    with path.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            gold = row.get("Gold_passage") or {}
            gold_ids = [str(gold["id"])] if gold.get("id") else []
            yield Example(
                example_id=f"{row['Conversation_no']}:{row['Turn_no']}:{i}",
                question=str(row["Question"]).strip(),
                answer=str(row.get("Answer") or "").strip(),
                history=_normalize_history(row.get("Context")),
                gold_passage_ids=gold_ids,
                meta={
                    "conversation_no": row["Conversation_no"],
                    "turn_no": row["Turn_no"],
                    "topic": row.get("Topic"),
                    "topic_section": row.get("Topic_section"),
                    "gold_text": str(gold.get("text") or ""),
                    "gold_title": str(gold.get("title") or ""),
                },
            )

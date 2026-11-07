from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from src.schema import Example, Passage


def iter_qrecc_passages(data_root: Path) -> Iterator[Passage]:
    data_dir = data_root / "qrecc-passages" / "data"
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


def _normalize_history(context) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    if not context:
        return history
    # QReCC Context is usually a list alternating question/answer strings.
    for i, utt in enumerate(context):
        role = "user" if i % 2 == 0 else "agent"
        history.append((role, str(utt).strip()))
    return history


def _split_path(data_root: Path, split: str) -> Path:
    split = "validation" if split in {"valid", "dev"} else split
    if split == "validation":
        # Official release ships train/test; treat validation as test.
        split = "test"
    path = data_root / "qrecc" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Expected JSONL exported under data/qrecc/{{train,test}}.jsonl"
        )
    return path


def iter_qrecc_examples(data_root: Path, split: str) -> Iterator[Example]:
    path = _split_path(data_root, split)
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            yield Example(
                example_id=f"{row['Conversation_no']}:{row['Turn_no']}:{i}",
                question=str(row["Question"]).strip(),
                answer=str(row.get("Truth_answer") or "").strip(),
                history=_normalize_history(row.get("Context")),
                gold_passage_ids=[str(x) for x in (row.get("Truth_passages") or [])],
                rewrite=(str(row["Truth_rewrite"]).strip() if row.get("Truth_rewrite") else None),
                meta={"conversation_no": row["Conversation_no"], "turn_no": row["Turn_no"]},
            )

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from src.chunking import overlapping_chunk_ids
from src.schema import Example, Passage


def _root(data_root: Path) -> Path:
    return data_root / "multidoc2dial" / "multidoc2dial"


def iter_multidoc2dial_passages(data_root: Path) -> Iterator[Passage]:
    docs = json.loads((_root(data_root) / "multidoc2dial_doc.json").read_text())["doc_data"]
    for domain, domain_docs in docs.items():
        for _, doc in domain_docs.items():
            text = (doc.get("doc_text") or "").strip()
            if not text:
                continue
            yield Passage(
                pid=str(doc["doc_id"]),
                text=text,
                title=str(doc.get("title") or ""),
                meta={"domain": domain},
            )


def _split_filename(split: str) -> str:
    mapping = {
        "train": "multidoc2dial_dial_train.json",
        "validation": "multidoc2dial_dial_validation.json",
        "valid": "multidoc2dial_dial_validation.json",
        "dev": "multidoc2dial_dial_validation.json",
        "test": "multidoc2dial_dial_test.json",
    }
    if split not in mapping:
        raise ValueError(f"multidoc2dial split must be one of {list(mapping)}; got {split}")
    return mapping[split]


def _chunk_params(data_root: Path) -> tuple[int, int]:
    meta_path = data_root / "indices" / "multidoc2dial" / "meta.json"
    if not meta_path.exists():
        return 512, 64
    meta = json.loads(meta_path.read_text())
    return int(meta.get("chunk_size") or 512), int(meta.get("chunk_overlap") or 64)


def _docs_by_id(data_root: Path) -> dict[str, dict]:
    docs = json.loads((_root(data_root) / "multidoc2dial_doc.json").read_text())["doc_data"]
    out: dict[str, dict] = {}
    for domain_docs in docs.values():
        for doc in domain_docs.values():
            spans = {}
            for sp in (doc.get("spans") or {}).values():
                if sp.get("start_sp") is None or sp.get("end_sp") is None:
                    continue
                spans[str(sp.get("id_sp"))] = (int(sp["start_sp"]), int(sp["end_sp"]))
            out[str(doc["doc_id"])] = {
                "length": len(doc.get("doc_text") or ""),
                "spans": spans,
            }
    return out


def _gold_ids_for_refs(refs: list[dict], docs: dict[str, dict], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunk pids (`doc_id#k`) that overlap a referenced span. Fall back to the doc id."""
    resolved: list[str] = []
    fallback: list[str] = []
    for ref in refs:
        doc_id = str(ref.get("doc_id") or "")
        if not doc_id:
            continue
        info = docs.get(doc_id)
        span = info["spans"].get(str(ref.get("id_sp"))) if info else None
        if not info or span is None:
            fallback.append(doc_id)
            continue
        chunk_ids = overlapping_chunk_ids(info["length"], span[0], span[1], chunk_size, chunk_overlap)
        if not chunk_ids:
            fallback.append(doc_id)
            continue
        resolved.extend(f"{doc_id}#{cid}" for cid in chunk_ids)
    return list(dict.fromkeys(resolved or fallback))


def iter_multidoc2dial_examples(data_root: Path, split: str) -> Iterator[Example]:
    dials = json.loads((_root(data_root) / _split_filename(split)).read_text())["dial_data"]
    docs = _docs_by_id(data_root)
    chunk_size, chunk_overlap = _chunk_params(data_root)
    for domain, domain_dials in dials.items():
        for dial in domain_dials:
            dial_id = dial["dial_id"]
            history: list[tuple[str, str]] = []
            turns = dial["turns"]
            for i, turn in enumerate(turns):
                role = "user" if turn["role"] == "user" else "agent"
                utt = turn["utterance"].strip()
                if role == "user":
                    # Gold answer = next agent turn if present.
                    answer = ""
                    refs: list[dict] = []
                    if i + 1 < len(turns) and turns[i + 1]["role"] == "agent":
                        answer = turns[i + 1]["utterance"].strip()
                        refs.extend(turns[i + 1].get("references") or [])
                    refs.extend(turn.get("references") or [])
                    gold_ids = _gold_ids_for_refs(refs, docs, chunk_size, chunk_overlap)
                    yield Example(
                        example_id=f"{dial_id}:{turn['turn_id']}",
                        question=utt,
                        answer=answer,
                        history=list(history),
                        gold_passage_ids=gold_ids,
                        meta={"domain": domain, "dial_id": dial_id},
                    )
                history.append((role, utt))

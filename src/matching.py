"""Decide whether a retrieved chunk is the annotated passage.

MultiDoc2Dial gold ids are chunk pids (`doc_id#k`) when the referenced span
can be placed in the same windows used to build the index. QReCC gold ids are
the passage id before `#chunk`. TopiOCQA's wiki page id is not the collection
id, so a hit counts when its text is contained in Gold_passage.text.
"""

from __future__ import annotations

from src.chunking import overlapping_chunk_ids
from src.schema import Example, Passage


def source_id(passage: Passage) -> str:
    if passage.meta.get("source_id"):
        return str(passage.meta["source_id"])
    if "#" in passage.pid:
        return passage.pid.rsplit("#", 1)[0]
    return passage.pid


def _chunk_id(passage: Passage) -> int:
    if passage.meta.get("chunk_id") is not None:
        return int(passage.meta["chunk_id"])
    if "#" in passage.pid:
        tail = passage.pid.rsplit("#", 1)[1]
        if tail.isdigit():
            return int(tail)
    return 0


def _norm(text: str) -> str:
    return " ".join(text.replace('""', '"').split()).casefold()


def _text_in_gold(passage_text: str, gold_text: str) -> bool:
    snippet = _norm(passage_text)
    gold = _norm(gold_text)
    if len(snippet) < 80 or not gold:
        return False
    return snippet in gold


def hit_is_gold(
    passage: Passage,
    example: Example,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> bool:
    gold = set(example.gold_passage_ids)
    if passage.pid in gold:
        return True

    source = source_id(passage)
    spans = example.meta.get("gold_spans") or []
    if spans:
        cid = _chunk_id(passage)
        for sp in spans:
            if str(sp.get("doc_id")) != source:
                continue
            ids = overlapping_chunk_ids(
                int(sp["doc_len"]),
                int(sp["start"]),
                int(sp["end"]),
                chunk_size,
                chunk_overlap,
            )
            if cid in ids:
                return True
        return False

    if source in gold:
        return True

    gold_text = str(example.meta.get("gold_text") or "")
    if gold_text:
        return _text_in_gold(passage.text, gold_text)
    return False

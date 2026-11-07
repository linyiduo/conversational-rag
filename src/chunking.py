from __future__ import annotations

from collections.abc import Iterable, Iterator

from src.schema import Passage


def overlapping_chunk_ids(
    text_len: int,
    span_start: int,
    span_end: int,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[int]:
    """Chunk ids whose window overlaps [span_start, span_end). Same windows as chunk_passages."""
    if text_len <= 0 or span_end <= span_start:
        return []
    if text_len <= chunk_size:
        return [0] if span_start < text_len and span_end > 0 else []

    ids: list[int] = []
    start = 0
    chunk_id = 0
    while start < text_len:
        end = min(text_len, start + chunk_size)
        if span_start < end and span_end > start:
            ids.append(chunk_id)
        if end >= text_len:
            break
        start = end - chunk_overlap
        chunk_id += 1
    return ids


def chunk_passages(
    passages: Iterable[Passage],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Iterator[Passage]:
    """Simple character-window chunking; keeps short docs as one chunk."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    for passage in passages:
        text = passage.text.strip()
        if not text:
            continue
        if len(text) <= chunk_size:
            yield Passage(
                pid=f"{passage.pid}#0",
                text=text,
                title=passage.title,
                meta={**passage.meta, "source_id": passage.pid, "chunk_id": 0},
            )
            continue

        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                yield Passage(
                    pid=f"{passage.pid}#{chunk_id}",
                    text=chunk,
                    title=passage.title,
                    meta={**passage.meta, "source_id": passage.pid, "chunk_id": chunk_id},
                )
                chunk_id += 1
            if end >= len(text):
                break
            start = max(0, end - chunk_overlap)

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from src.data.multidoc2dial import iter_multidoc2dial_examples, iter_multidoc2dial_passages
from src.data.qrecc import iter_qrecc_examples, iter_qrecc_passages
from src.data.topiocqa import iter_topiocqa_examples, iter_topiocqa_passages
from src.schema import Example, Passage

DATASETS = ("multidoc2dial", "qrecc", "topiocqa")


def iter_passages(
    dataset: str,
    data_root: Path,
    max_passages: int | None = None,
) -> Iterator[Passage]:
    dataset = dataset.lower()
    if dataset == "multidoc2dial":
        it = iter_multidoc2dial_passages(data_root)
    elif dataset == "qrecc":
        it = iter_qrecc_passages(data_root)
    elif dataset == "topiocqa":
        it = iter_topiocqa_passages(data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {DATASETS}")

    for i, passage in enumerate(it):
        if max_passages is not None and i >= max_passages:
            break
        yield passage


def iter_examples(
    dataset: str,
    data_root: Path,
    split: str,
    max_examples: int | None = None,
) -> Iterator[Example]:
    dataset = dataset.lower()
    if dataset == "multidoc2dial":
        it = iter_multidoc2dial_examples(data_root, split)
    elif dataset == "qrecc":
        it = iter_qrecc_examples(data_root, split)
    elif dataset == "topiocqa":
        it = iter_topiocqa_examples(data_root, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {DATASETS}")

    for i, example in enumerate(it):
        if max_examples is not None and i >= max_examples:
            break
        yield example

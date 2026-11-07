from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.chunking import chunk_passages
from src.data import iter_passages
from src.schema import Passage

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBED_MODEL = REPO_ROOT / "models" / "all-MiniLM-L6-v2"
_EMBED_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_HUB_EMBED_IDS = {
    _EMBED_REPO,
    "all-MiniLM-L6-v2",
    "models/all-MiniLM-L6-v2",
    str(DEFAULT_EMBED_MODEL),
}


def _embed_ready(path: Path) -> bool:
    weight = path / "model.safetensors"
    return (path / "modules.json").is_file() and weight.is_file() and not weight.is_symlink()


def _download_embed_model(path: Path) -> None:
    """Download the embedding model into models/ as real files."""
    from huggingface_hub import snapshot_download

    path.mkdir(parents=True, exist_ok=True)
    cache = REPO_ROOT / "models" / ".hf_cache"
    print(f"downloading {_EMBED_REPO} to {path}", flush=True)
    snapshot_download(
        repo_id=_EMBED_REPO,
        local_dir=str(path),
        cache_dir=str(cache),
        allow_patterns=[
            "config.json",
            "config_sentence_transformers.json",
            "modules.json",
            "sentence_bert_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "1_Pooling/*",
            "README.md",
        ],
    )
    if not _embed_ready(path):
        raise RuntimeError(f"Embedding model download incomplete: {path}")
    shutil.rmtree(cache, ignore_errors=True)
    shutil.rmtree(path / ".cache", ignore_errors=True)


def resolve_embed_model(embed_model: str | None) -> Path:
    """Directory under models/. Downloads the default model there on first use."""
    if embed_model is None or embed_model in _HUB_EMBED_IDS:
        path = DEFAULT_EMBED_MODEL
        if not _embed_ready(path):
            _download_embed_model(path)
        return path.resolve()

    path = Path(embed_model)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Embedding model not found at {path}")
    return path


def embed_model_label(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def index_dir(data_root: Path, dataset: str) -> Path:
    return data_root / "indices" / dataset


def build_index(
    dataset: str,
    data_root: Path,
    embed_model: str,
    device: str = "cuda",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    batch_size: int = 64,
    max_passages: int | None = None,
) -> Path:
    out = index_dir(data_root, dataset)
    out.mkdir(parents=True, exist_ok=True)

    model_path = resolve_embed_model(embed_model)
    model = SentenceTransformer(str(model_path), device=device)
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    passages: list[Passage] = []
    texts: list[str] = []

    raw = iter_passages(dataset, data_root, max_passages=max_passages)
    chunks = chunk_passages(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    emb_batches: list[np.ndarray] = []
    buffer_texts: list[str] = []

    def flush() -> None:
        nonlocal buffer_texts
        if not buffer_texts:
            return
        emb = model.encode(
            buffer_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        emb_batches.append(emb)
        buffer_texts = []

    for chunk in tqdm(chunks, desc=f"index:{dataset}"):
        passages.append(chunk)
        display = chunk.text if not chunk.title else f"{chunk.title}\n{chunk.text}"
        texts.append(display)
        buffer_texts.append(display)
        if len(buffer_texts) >= batch_size * 4:
            flush()
    flush()

    if not emb_batches:
        raise RuntimeError(f"No passages indexed for dataset={dataset}")

    vectors = np.vstack(emb_batches)
    index.add(vectors)

    faiss.write_index(index, str(out / "index.faiss"))
    (out / "passages.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "pid": p.pid,
                    "text": p.text,
                    "title": p.title,
                    "meta": p.meta,
                },
                ensure_ascii=False,
            )
            for p in passages
        )
        + "\n",
        encoding="utf-8",
    )
    meta = {
        "dataset": dataset,
        "embed_model": embed_model_label(model_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(passages),
        "max_passages": max_passages,
        "dim": dim,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


class FaissStore:
    def __init__(self, dataset: str, data_root: Path, embed_model: str | None = None, device: str = "cuda"):
        self.dir = index_dir(data_root, dataset)
        if not (self.dir / "index.faiss").exists():
            raise FileNotFoundError(
                f"Missing index at {self.dir}. Run scripts/build_index.py --dataset {dataset}"
            )
        self.index = faiss.read_index(str(self.dir / "index.faiss"))
        self.passages: list[Passage] = []
        with (self.dir / "passages.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.passages.append(
                    Passage(
                        pid=row["pid"],
                        text=row["text"],
                        title=row.get("title") or "",
                        meta=row.get("meta") or {},
                    )
                )
        meta = json.loads((self.dir / "meta.json").read_text())
        self.chunk_size = int(meta.get("chunk_size") or 512)
        self.chunk_overlap = int(meta.get("chunk_overlap") or 64)
        model_path = resolve_embed_model(embed_model or meta.get("embed_model"))
        self.model = SentenceTransformer(str(model_path), device=device)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Passage, float]]:
        q = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        scores, idxs = self.index.search(q, top_k)
        hits: list[tuple[Passage, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            hits.append((self.passages[int(idx)], float(score)))
        return hits

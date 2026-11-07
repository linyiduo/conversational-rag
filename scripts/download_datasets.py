"""Download conversational RAG datasets into data/.

Includes dialogue splits and retrieval corpora:
- MultiDoc2Dial (dialogue + documents)
- TopiOCQA (dialogue + Wikipedia collection)
- QReCC (topics/QA + passages)

Uses HF_ENDPOINT when set (recommended: https://hf-mirror.com).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / ".hf_cache"

# Prefer mirror if caller did not set one.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Avoid Xet CAS 401 failures that sometimes appear behind mirrors.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE / "datasets"))

HF_ENDPOINT = os.environ["HF_ENDPOINT"].rstrip("/")

MULTIDOC2DIAL_ZIP = (
    "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"
)
TOPIOCQA_FILES = {
    "train": f"{HF_ENDPOINT}/datasets/McGill-NLP/TopiOCQA/resolve/main/data/topiocqa_train.jsonl",
    "valid": f"{HF_ENDPOINT}/datasets/McGill-NLP/TopiOCQA/resolve/main/data/topiocqa_valid.jsonl",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name}")
        return
    print(f"[downloading] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"[ok] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _hf_download_dataset(repo_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    # If directory already has substantial content, skip.
    existing = list(local_dir.rglob("*"))
    files = [p for p in existing if p.is_file()]
    if files and sum(p.stat().st_size for p in files) > 1_000_000:
        print(f"[skip] {local_dir} already has {len(files)} files")
        return
    cmd = [
        "hf",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(local_dir),
        "--max-workers",
        "8",
    ]
    print(f"[hf download] {' '.join(cmd)}")
    subprocess.check_call(cmd, env=os.environ.copy())
    print(f"[ok] {repo_id} -> {local_dir}")


def download_multidoc2dial() -> None:
    out_dir = DATA / "multidoc2dial"
    marker = out_dir / "multidoc2dial"
    if marker.exists() or (out_dir / "data").exists():
        print(f"[skip] multidoc2dial already extracted under {out_dir}")
        return
    zip_path = DATA / "multidoc2dial.zip"
    _download(MULTIDOC2DIAL_ZIP, zip_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink(missing_ok=True)
    junk = out_dir / "__MACOSX"
    if junk.exists():
        shutil.rmtree(junk)
    print(f"[ok] extracted multidoc2dial -> {out_dir}")


def download_topiocqa_dialogue() -> None:
    out_dir = DATA / "topiocqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, url in TOPIOCQA_FILES.items():
        _download(url, out_dir / f"topiocqa_{split}.jsonl")


def download_topiocqa_collection() -> None:
    _hf_download_dataset(
        "slupart/topiocqa-collection",
        DATA / "topiocqa-collection",
    )


def download_qrecc_dialogue() -> None:
    """Save QReCC dialogue as JSONL to avoid datasets version lock-in."""
    out = DATA / "qrecc"
    train_jsonl = out / "train.jsonl"
    test_jsonl = out / "test.jsonl"
    if train_jsonl.exists() and test_jsonl.exists():
        print(f"[skip] {out} already exists")
        return
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("slupart/qrecc")
    for split, path in (("train", train_jsonl), ("test", test_jsonl)):
        with path.open("w", encoding="utf-8") as f:
            for row in ds[split]:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        print(f"[ok] saved {split} -> {path}")


def download_qrecc_passages() -> None:
    _hf_download_dataset(
        "slupart/qrecc-passages",
        DATA / "qrecc-passages",
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    print("=== MultiDoc2Dial ===")
    download_multidoc2dial()
    print("=== TopiOCQA dialogue ===")
    download_topiocqa_dialogue()
    print("=== TopiOCQA collection ===")
    download_topiocqa_collection()
    print("=== QReCC dialogue ===")
    download_qrecc_dialogue()
    print("=== QReCC passages ===")
    download_qrecc_passages()
    print("done")


if __name__ == "__main__":
    main()

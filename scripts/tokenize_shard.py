from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import sentencepiece as spm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wiki_lm.data.shards import stable_hash_to_split  # noqa: E402


def _flush_split(
    *,
    split_dir: Path,
    shard_idx: int,
    tokens: List[int],
    doc_offsets: List[int],
) -> int:
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_path = split_dir / f"shard{shard_idx:05d}.npz"
    arr = np.asarray(tokens, dtype=np.uint16 if max(tokens, default=0) < 65536 else np.int32)
    off = np.asarray(doc_offsets, dtype=np.int64)
    np.savez(shard_path, tokens=arr, doc_offsets=off)
    return shard_idx + 1


def tokenize_and_shard(
    *,
    in_jsonl: str,
    tokenizer_model: str,
    out_dir: str,
    seed: int,
    shard_size_tokens: int,
    split_seed: int,
    split_ratios: Tuple[float, float, float],
    doc_token: str,
) -> None:
    out_dir_p = Path(out_dir)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)

    doc_id = int(sp.piece_to_id(doc_token))
    pad_id = int(sp.pad_id())
    bos_id = int(sp.bos_id())
    eos_id = int(sp.eos_id())
    vocab_size = int(sp.get_piece_size())

    # Per-split shard buffers
    buf: Dict[str, Dict[str, object]] = {}
    shard_idx: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        buf[split] = {"tokens": [], "doc_offsets": [0]}

    def maybe_flush(split: str) -> None:
        tokens = buf[split]["tokens"]  # type: ignore[assignment]
        doc_offsets = buf[split]["doc_offsets"]  # type: ignore[assignment]
        assert isinstance(tokens, list)
        assert isinstance(doc_offsets, list)
        if len(tokens) >= shard_size_tokens:
            split_dir = out_dir_p / split
            shard_idx[split] = _flush_split(
                split_dir=split_dir,
                shard_idx=shard_idx[split],
                tokens=tokens,
                doc_offsets=doc_offsets,
            )
            buf[split] = {"tokens": [], "doc_offsets": [0]}

    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            did = str(obj.get("id", ""))
            text = str(obj.get("text", ""))
            split = stable_hash_to_split(did, seed=split_seed, ratios=split_ratios)

            tokens: List[int] = buf[split]["tokens"]  # type: ignore[assignment]
            doc_offsets: List[int] = buf[split]["doc_offsets"]  # type: ignore[assignment]

            # Document boundary markers + eos.
            tokens.append(doc_id)
            tokens.extend(sp.encode(text, out_type=int))
            tokens.append(eos_id)
            doc_offsets.append(len(tokens))

            maybe_flush(split)

    # Flush remaining buffers
    for split in ("train", "val", "test"):
        tokens = buf[split]["tokens"]  # type: ignore[assignment]
        doc_offsets = buf[split]["doc_offsets"]  # type: ignore[assignment]
        assert isinstance(tokens, list)
        assert isinstance(doc_offsets, list)
        if tokens:
            split_dir = out_dir_p / split
            shard_idx[split] = _flush_split(
                split_dir=split_dir,
                shard_idx=shard_idx[split],
                tokens=tokens,
                doc_offsets=doc_offsets,
            )

        meta = {
            "vocab_size": vocab_size,
            "pad_id": pad_id,
            "bos_id": bos_id,
            "eos_id": eos_id,
            "doc_id": doc_id,
            "doc_token": doc_token,
        }
        (out_dir_p / split).mkdir(parents=True, exist_ok=True)
        (out_dir_p / split / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--tokenizer_model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard_size_tokens", type=int, default=50_000_000)
    ap.add_argument("--split_seed", type=int, default=1234)
    ap.add_argument("--doc_token", type=str, default="<DOC>")
    ap.add_argument("--train_ratio", type=float, default=0.98)
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--test_ratio", type=float, default=0.01)
    args = ap.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    tokenize_and_shard(
        in_jsonl=args.in_jsonl,
        tokenizer_model=args.tokenizer_model,
        out_dir=args.out_dir,
        seed=args.seed,
        shard_size_tokens=args.shard_size_tokens,
        split_seed=args.split_seed,
        split_ratios=ratios,
        doc_token=args.doc_token,
    )


if __name__ == "__main__":
    main()


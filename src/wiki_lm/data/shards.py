"""Shard splitting and loading utilities."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


def stable_hash_to_split(
    doc_id: str,
    seed: int = 0,
    ratios: Tuple[float, float, float] = (0.90, 0.05, 0.05),
) -> str:
    """Deterministically assign a document to a split based on its ID hash.

    Args:
        doc_id: Document identifier string.
        seed: Seed for hash mixing.
        ratios: (train, val, test) fractions that must sum to ~1.0.

    Returns:
        One of "train", "val", "test".
    """
    h = hashlib.md5(f"{seed}:{doc_id}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    train_r, val_r, _test_r = ratios
    if bucket < train_r:
        return "train"
    elif bucket < train_r + val_r:
        return "val"
    else:
        return "test"


@dataclass(frozen=True)
class ShardMeta:
    """Metadata loaded from meta.json in a shard directory."""
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    doc_id: int
    doc_token: str


@dataclass
class TokenShard:
    """A loaded shard: a flat token array plus document boundary offsets."""
    tokens: np.ndarray       # (N,) uint16 or int64 token ids
    doc_offsets: np.ndarray   # (D+1,) int64 -- start of each document; last entry = len(tokens)
    meta: ShardMeta


def load_split(shards_dir: str, split: str) -> TokenShard:
    """Load all shard npz files for a given split and concatenate them.

    Expects ``shards_dir/<split>/meta.json`` and one or more ``shard*.npz``
    files each containing ``tokens`` (1-D) and ``doc_offsets`` (1-D).

    Returns:
        A single :class:`TokenShard` with concatenated tokens and merged
        document offsets.
    """
    split_dir = Path(shards_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Load meta
    meta_path = split_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {split_dir}")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = ShardMeta(
        vocab_size=int(raw["vocab_size"]),
        pad_id=int(raw["pad_id"]),
        bos_id=int(raw["bos_id"]),
        eos_id=int(raw["eos_id"]),
        doc_id=int(raw["doc_id"]),
        doc_token=str(raw.get("doc_token", "<DOC>")),
    )

    # Collect and sort shard files
    shard_files = sorted(split_dir.glob("shard*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"No shard*.npz files in {split_dir}")

    all_tokens = []
    all_offsets = []
    running_offset = 0

    for sf in shard_files:
        data = np.load(sf)
        tokens = data["tokens"]
        doc_offsets = data["doc_offsets"]

        all_tokens.append(tokens)
        # Shift offsets by the running token count
        all_offsets.append(doc_offsets + running_offset)
        running_offset += len(tokens)

    tokens_cat = np.concatenate(all_tokens, axis=0)

    # Merge doc_offsets: each shard's offsets mark document starts.
    # The last offset should be the total token count.
    # Remove duplicate boundary points that arise at shard joins.
    merged = np.concatenate(all_offsets, axis=0)
    # Deduplicate and sort
    merged = np.unique(merged)
    # Ensure final boundary is present
    if len(merged) == 0 or merged[-1] != running_offset:
        merged = np.append(merged, running_offset)

    return TokenShard(tokens=tokens_cat, doc_offsets=merged.astype(np.int64), meta=meta)


def doc_span(shard: TokenShard, doc_idx: int) -> Tuple[int, int]:
    """Return (start, end) token indices for document ``doc_idx``."""
    return int(shard.doc_offsets[doc_idx]), int(shard.doc_offsets[doc_idx + 1])

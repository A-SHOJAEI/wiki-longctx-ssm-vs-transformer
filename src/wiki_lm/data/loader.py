"""Data loaders for language model training with packed and non-packed sequences."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch

from .shards import TokenShard


@dataclass
class Batch:
    """A single training batch."""
    input_ids: torch.Tensor    # (B, T) int64 token ids
    labels: torch.Tensor       # (B, T) int64 next-token targets
    loss_mask: torch.Tensor    # (B, T) float32 mask (1 = compute loss, 0 = ignore)
    segment_ids: Optional[torch.Tensor]  # (B, T) int64 segment ids for attention masking, or None


class PackedStreamLoader:
    """Packs multiple documents into fixed-length sequences without padding.

    Documents are concatenated end-to-end in a shuffled order.  Each sequence
    of length ``seq_len`` is carved from the stream.  When ``reset_masks`` is
    True, ``segment_ids`` are provided so the model can avoid cross-document
    attention.  When ``boundary_markers`` is True the ``doc_token_id`` tokens
    present in the stream are kept; otherwise they are replaced with ``eos_id``.
    """

    def __init__(
        self,
        shard: TokenShard,
        *,
        seq_len: int,
        batch_size: int,
        seed: int,
        boundary_markers: bool = True,
        reset_masks: bool = True,
        doc_token_id: int,
        eos_id: int,
    ) -> None:
        self.shard = shard
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.boundary_markers = boundary_markers
        self.reset_masks = reset_masks
        self.doc_token_id = doc_token_id
        self.eos_id = eos_id

    def __iter__(self) -> Iterator[Batch]:
        rng = np.random.RandomState(self.seed)
        tokens = self.shard.tokens.copy().astype(np.int64)

        # Optionally replace doc boundary markers
        if not self.boundary_markers:
            tokens[tokens == self.doc_token_id] = self.eos_id

        # Build document order (shuffle documents, then concatenate)
        n_docs = len(self.shard.doc_offsets) - 1
        doc_order = np.arange(n_docs)
        rng.shuffle(doc_order)

        # Build a flat stream from shuffled documents
        stream_parts = []
        for di in doc_order:
            start = int(self.shard.doc_offsets[di])
            end = int(self.shard.doc_offsets[di + 1])
            stream_parts.append(tokens[start:end])
        stream = np.concatenate(stream_parts, axis=0)

        # Also build a segment-id stream: each document gets a unique segment id
        if self.reset_masks:
            seg_parts = []
            for seg_id, di in enumerate(doc_order):
                start = int(self.shard.doc_offsets[di])
                end = int(self.shard.doc_offsets[di + 1])
                seg_parts.append(np.full(end - start, seg_id, dtype=np.int64))
            seg_stream = np.concatenate(seg_parts, axis=0)
        else:
            seg_stream = None

        # Yield batches from the stream (infinite: loop when exhausted)
        total_len = len(stream)
        # We need seq_len + 1 tokens per sequence (input + label for last position)
        needed_per_batch = self.batch_size * (self.seq_len + 1)

        pos = 0
        while True:
            # Collect enough tokens for a batch
            batch_tokens = np.empty((self.batch_size, self.seq_len + 1), dtype=np.int64)
            batch_segs = np.empty((self.batch_size, self.seq_len + 1), dtype=np.int64) if self.reset_masks else None

            for bi in range(self.batch_size):
                # Handle wrap-around
                if pos + self.seq_len + 1 > total_len:
                    # Reshuffle and restart
                    rng.shuffle(doc_order)
                    stream_parts = []
                    for di in doc_order:
                        start = int(self.shard.doc_offsets[di])
                        end = int(self.shard.doc_offsets[di + 1])
                        stream_parts.append(tokens[start:end])
                    stream = np.concatenate(stream_parts, axis=0)
                    total_len = len(stream)

                    if self.reset_masks:
                        seg_parts = []
                        for seg_id, di in enumerate(doc_order):
                            start = int(self.shard.doc_offsets[di])
                            end = int(self.shard.doc_offsets[di + 1])
                            seg_parts.append(np.full(end - start, seg_id, dtype=np.int64))
                        seg_stream = np.concatenate(seg_parts, axis=0)

                    pos = 0

                batch_tokens[bi] = stream[pos : pos + self.seq_len + 1]
                if batch_segs is not None:
                    batch_segs[bi] = seg_stream[pos : pos + self.seq_len + 1]
                pos += self.seq_len

            input_ids = torch.from_numpy(batch_tokens[:, :-1])   # (B, T)
            labels = torch.from_numpy(batch_tokens[:, 1:])       # (B, T)

            # Loss mask: 1 everywhere except where label crosses a document boundary
            # when reset_masks is True. For simplicity, mask positions where the
            # input and label are from different segments.
            loss_mask = torch.ones_like(labels, dtype=torch.float32)
            if batch_segs is not None:
                input_segs = torch.from_numpy(batch_segs[:, :-1])
                label_segs = torch.from_numpy(batch_segs[:, 1:])
                # Mask cross-document transitions
                loss_mask[input_segs != label_segs] = 0.0
                segment_ids = input_segs
            else:
                segment_ids = None

            yield Batch(
                input_ids=input_ids,
                labels=labels,
                loss_mask=loss_mask,
                segment_ids=segment_ids,
            )


class NoPackingChunkLoader:
    """Yields one document per sequence, padded or truncated to ``seq_len``.

    Each document is placed in its own sequence slot.  Documents shorter than
    ``seq_len`` are padded (loss_mask = 0 on padding).  Documents longer than
    ``seq_len + 1`` are truncated.
    """

    def __init__(
        self,
        shard: TokenShard,
        *,
        seq_len: int,
        batch_size: int,
        seed: int,
        boundary_markers: bool = True,
        reset_masks: bool = True,
        doc_token_id: int,
        eos_id: int,
    ) -> None:
        self.shard = shard
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.boundary_markers = boundary_markers
        self.reset_masks = reset_masks
        self.doc_token_id = doc_token_id
        self.eos_id = eos_id

    def __iter__(self) -> Iterator[Batch]:
        rng = np.random.RandomState(self.seed)
        tokens = self.shard.tokens.copy().astype(np.int64)

        if not self.boundary_markers:
            tokens[tokens == self.doc_token_id] = self.eos_id

        n_docs = len(self.shard.doc_offsets) - 1
        doc_order = np.arange(n_docs)
        rng.shuffle(doc_order)

        di_idx = 0
        while True:
            batch_input = np.zeros((self.batch_size, self.seq_len), dtype=np.int64)
            batch_labels = np.zeros((self.batch_size, self.seq_len), dtype=np.int64)
            batch_mask = np.zeros((self.batch_size, self.seq_len), dtype=np.float32)
            batch_segs = np.zeros((self.batch_size, self.seq_len), dtype=np.int64)

            for bi in range(self.batch_size):
                if di_idx >= n_docs:
                    rng.shuffle(doc_order)
                    di_idx = 0

                doc_i = doc_order[di_idx]
                di_idx += 1

                start = int(self.shard.doc_offsets[doc_i])
                end = int(self.shard.doc_offsets[doc_i + 1])
                doc_tokens = tokens[start:end]

                # Need at least 2 tokens for input/label shift
                if len(doc_tokens) < 2:
                    # Fill with padding (mask=0)
                    continue

                # Truncate if needed
                usable = min(len(doc_tokens), self.seq_len + 1)
                chunk = doc_tokens[:usable]

                inp = chunk[:-1]
                lbl = chunk[1:]
                length = len(inp)

                batch_input[bi, :length] = inp
                batch_labels[bi, :length] = lbl
                batch_mask[bi, :length] = 1.0
                batch_segs[bi, :] = bi  # Each doc gets unique segment

            input_ids = torch.from_numpy(batch_input)
            labels = torch.from_numpy(batch_labels)
            loss_mask = torch.from_numpy(batch_mask)
            segment_ids = torch.from_numpy(batch_segs) if self.reset_masks else None

            yield Batch(
                input_ids=input_ids,
                labels=labels,
                loss_mask=loss_mask,
                segment_ids=segment_ids,
            )

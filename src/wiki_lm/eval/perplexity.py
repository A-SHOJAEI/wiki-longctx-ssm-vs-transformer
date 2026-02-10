from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..data.loader import PackedStreamLoader
from ..data.shards import TokenShard, load_split


@dataclass(frozen=True)
class PerplexityResult:
    nll: float
    ppl: float
    batches: int


@torch.no_grad()
def eval_perplexity(
    model: torch.nn.Module,
    *,
    shards_dir: str,
    seq_len: int,
    batch_size: int,
    max_batches: int,
    seed: int,
    device: torch.device,
) -> PerplexityResult:
    shard = load_split(shards_dir, "val")
    loader = PackedStreamLoader(
        shard,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
        boundary_markers=True,
        reset_masks=True,
        doc_token_id=shard.meta.doc_id,
        eos_id=shard.meta.eos_id,
    )
    it = iter(loader)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for bi in range(max_batches):
        batch = next(it)
        input_ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)
        loss_mask = batch.loss_mask.to(device)
        segment_ids = batch.segment_ids.to(device) if batch.segment_ids is not None else None

        logits = model(input_ids, segment_ids=segment_ids)
        v = logits.shape[-1]
        loss_flat = F.cross_entropy(logits.view(-1, v), labels.reshape(-1), reduction="none")
        loss = (loss_flat.view(labels.shape) * loss_mask).sum()
        tokens = int(loss_mask.sum().item())
        total_loss += float(loss.item())
        total_tokens += tokens

    nll = total_loss / max(total_tokens, 1)
    ppl = float(math.exp(min(nll, 50.0)))
    return PerplexityResult(nll=nll, ppl=ppl, batches=max_batches)


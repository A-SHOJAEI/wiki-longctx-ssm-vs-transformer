from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..data.shards import TokenShard, doc_span, load_split


@dataclass(frozen=True)
class SuffixIdBucketResult:
    bucket_ctx_len: int
    examples: int
    accuracy: float


@torch.no_grad()
def _score_suffix_nll(
    model: torch.nn.Module,
    *,
    context: torch.Tensor,  # (L,)
    suffix: torch.Tensor,   # (S,)
    device: torch.device,
) -> float:
    # Teacher-forcing NLL of suffix conditioned on context.
    # Build a standard LM pair: input_ids = seq[:-1], labels = seq[1:].
    seq = torch.cat([context, suffix], dim=0).to(device)  # (L+S,)
    input_ids = seq[:-1].unsqueeze(0)  # (1, L+S-1)
    labels = seq[1:].unsqueeze(0)      # (1, L+S-1)

    logits = model(input_ids, segment_ids=None)  # (1, L+S-1, V)
    v = logits.shape[-1]

    # Suffix tokens occupy positions [L-1, L+S-2] in labels.
    L = int(context.shape[0])
    S = int(suffix.shape[0])
    a = L - 1
    b = a + S
    loss_flat = F.cross_entropy(
        logits[:, a:b, :].contiguous().view(-1, v),
        labels[:, a:b].contiguous().view(-1),
        reduction="mean",
    )
    return float(loss_flat.item())


def _collect_eligible_docs(shard: TokenShard, *, min_len: int) -> List[int]:
    out: List[int] = []
    for di in range(len(shard.doc_offsets) - 1):
        a, b = doc_span(shard, di)
        if (b - a) >= min_len:
            out.append(di)
    return out


@torch.no_grad()
def eval_suffix_identification(
    model: torch.nn.Module,
    *,
    shards_dir: str,
    buckets: List[int],
    suffix_len: int,
    num_examples: int,
    num_distractors: int,
    seed: int,
    device: torch.device,
) -> Dict[int, SuffixIdBucketResult]:
    """
    Contrastive suffix identification:
    - Sample a context window of length L from a held-out doc.
    - The "true" suffix is the next `suffix_len` tokens.
    - Distractors are suffix_len-token spans sampled from other docs.
    - Model picks the candidate with the lowest NLL.

    This is a practical long-context stress test: as L grows, models that cannot
    effectively use long contexts often degrade.
    """

    shard = load_split(shards_dir, "test")
    rng = np.random.RandomState(seed)

    results: Dict[int, SuffixIdBucketResult] = {}
    model.eval()

    for L in buckets:
        min_doc_len = L + suffix_len + 2
        eligible = _collect_eligible_docs(shard, min_len=min_doc_len)
        if not eligible:
            results[L] = SuffixIdBucketResult(bucket_ctx_len=L, examples=0, accuracy=0.0)
            continue

        correct = 0
        used = 0
        for _ in range(num_examples):
            di = int(rng.choice(eligible))
            a, b = doc_span(shard, di)
            doc = shard.tokens[a:b].astype(np.int64)
            # Avoid using the initial <DOC> token as context.
            start_min = a + 1 + L
            start_max = b - suffix_len - 1
            if start_max <= start_min:
                continue
            pos = int(rng.randint(start_min, start_max))
            ctx = torch.from_numpy(shard.tokens[pos - L : pos].astype(np.int64))
            true_suffix = torch.from_numpy(shard.tokens[pos : pos + suffix_len].astype(np.int64))

            candidates = [true_suffix]
            for _d in range(num_distractors):
                odi = int(rng.choice(eligible))
                oa, ob = doc_span(shard, odi)
                opos_min = oa + 1
                opos_max = ob - suffix_len - 1
                if opos_max <= opos_min:
                    # fallback: resample
                    continue
                opos = int(rng.randint(opos_min, opos_max))
                cand = torch.from_numpy(shard.tokens[opos : opos + suffix_len].astype(np.int64))
                candidates.append(cand)

            # Score all candidates; pick lowest NLL.
            scores = [
                _score_suffix_nll(model, context=ctx, suffix=c, device=device)
                for c in candidates
            ]
            pred = int(np.argmin(scores))
            if pred == 0:
                correct += 1
            used += 1

        acc = float(correct / used) if used else 0.0
        results[L] = SuffixIdBucketResult(bucket_ctx_len=L, examples=used, accuracy=acc)

    return results

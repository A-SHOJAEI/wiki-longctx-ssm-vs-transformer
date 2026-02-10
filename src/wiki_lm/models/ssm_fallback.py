from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SSMFallbackConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    dropout: float


class SSMFallbackLM(nn.Module):
    """
    Fallback model used when mamba-ssm is not available.
    This is not a full Mamba implementation; it's a lightweight recurrent LM
    that keeps the project runnable on CPU-only environments.
    """

    def __init__(self, cfg: SSMFallbackConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rnn = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.n_layers,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, *, segment_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # segment_ids is accepted for API compatibility; this fallback does not implement resets.
        x = self.emb(input_ids)
        y, _ = self.rnn(x)
        y = self.drop(y)
        return self.lm_head(y)


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .ssm_fallback import SSMFallbackConfig, SSMFallbackLM


@dataclass(frozen=True)
class MambaConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    dropout: float
    mamba_d_state: int
    mamba_d_conv: int
    mamba_expand: int


class _MambaBlock(nn.Module):
    def __init__(self, d_model: int, *, d_state: int, d_conv: int, expand: int, dropout: float) -> None:
        super().__init__()
        try:
            from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "mamba-ssm is not installed or failed to import. "
                "Install it for GPU runs, or use the fallback model."
            ) from e

        self.ln = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mamba(self.ln(x))
        return x + self.drop(y)


class MambaLM(nn.Module):
    """
    Mamba/SSM decoder-only LM using mamba-ssm when available.
    If mamba-ssm is unavailable, call `build_mamba_or_fallback` to get a runnable model.

    Reset masks / doc-boundary enforcement:
    - If `segment_ids` is provided, this model enforces "no cross-document leakage" by
      running each contiguous segment independently (slower, but correct).
    """

    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                _MambaBlock(
                    cfg.d_model,
                    d_state=cfg.mamba_d_state,
                    d_conv=cfg.mamba_d_conv,
                    expand=cfg.mamba_expand,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def _forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def forward(self, input_ids: torch.Tensor, *, segment_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if segment_ids is None:
            return self._forward_full(input_ids)

        # Enforce segment resets by running each segment independently.
        b, t = input_ids.shape
        out = torch.empty((b, t, self.cfg.vocab_size), device=input_ids.device, dtype=torch.float32)
        for bi in range(b):
            seg = segment_ids[bi].to(torch.int64)
            ids = input_ids[bi]
            # Find segment boundaries (where seg changes).
            boundaries = torch.nonzero(seg[1:] != seg[:-1], as_tuple=False).flatten().tolist()
            cuts = [0] + [i + 1 for i in boundaries] + [t]
            for a, c in zip(cuts[:-1], cuts[1:]):
                logits = self._forward_full(ids[a:c].unsqueeze(0)).squeeze(0)
                out[bi, a:c] = logits.to(out.dtype)
        return out


def build_mamba_or_fallback(cfg: MambaConfig) -> nn.Module:
    try:
        return MambaLM(cfg)
    except Exception:
        # Keep smoke / CPU runs working.
        fb = SSMFallbackLM(
            SSMFallbackConfig(
                vocab_size=cfg.vocab_size,
                d_model=cfg.d_model,
                n_layers=max(1, cfg.n_layers // 2),
                dropout=cfg.dropout,
            )
        )
        return fb


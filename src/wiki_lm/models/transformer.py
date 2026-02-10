from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    use_flash_attn: bool
    max_seq_len: int


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, segment_ids: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.cfg.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(b, t, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.cfg.n_heads, self.head_dim).transpose(1, 2)

        # Fast path: PyTorch SDPA (FlashAttention kernel when available) for pure causal attention.
        if self.cfg.use_flash_attn and segment_ids is None and hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)  # (B, H, T, D)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
            causal = torch.tril(torch.ones((t, t), device=x.device, dtype=torch.bool))
            allowed = causal[None, None, :, :]  # (1,1,T,T)
            if segment_ids is not None:
                # Block attention across segments to prevent cross-document leakage.
                seg = segment_ids.to(torch.int64)  # (B, T)
                same = (seg[:, None, :, None] == seg[:, None, None, :])  # (B,1,T,T)
                allowed = allowed & same
            att = att.masked_fill(~allowed, torch.finfo(att.dtype).min)
            y = att.softmax(dim=-1) @ v  # (B, H, T, D)

        y = y.transpose(1, 2).contiguous().view(b, t, c)  # (B, T, C)
        y = self.proj(y)
        y = self.drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, segment_ids: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), segment_ids=segment_ids)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, *, segment_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: (B, T)
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}")

        pos = torch.arange(0, t, device=input_ids.device, dtype=torch.long)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, segment_ids=segment_ids)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        return logits


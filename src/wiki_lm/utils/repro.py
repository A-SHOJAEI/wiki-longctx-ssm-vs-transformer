from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReproConfig:
    seed: int
    deterministic: bool


def seed_everything(cfg: ReproConfig) -> None:
    # Python / hashing
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
    random.seed(cfg.seed)

    # Numpy
    np.random.seed(cfg.seed)

    # Torch
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Determinism (best-effort). Some kernels are inherently nondeterministic; we log cfg in manifest.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = cfg.deterministic

    if cfg.deterministic:
        # This can throw if unsupported ops are used; callers can disable in configs if needed.
        torch.use_deterministic_algorithms(True, warn_only=False)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def choose_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype}")

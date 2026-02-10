from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .data.loader import Batch, NoPackingChunkLoader, PackedStreamLoader
from .data.shards import load_split
from .models.mamba_lm import MambaConfig, build_mamba_or_fallback
from .models.transformer import TransformerConfig, TransformerLM
from .utils.manifest import write_manifest
from .utils.repro import ReproConfig, choose_device, choose_dtype, seed_everything


@dataclass(frozen=True)
class TrainConfig:
    device: str
    dtype: str
    deterministic: bool
    seq_len: int
    batch_size: int
    grad_accum_steps: int
    max_steps: int
    lr: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    log_every: int
    ckpt_every: int


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _dist_info() -> Dict[str, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"rank": rank, "world_size": world, "local_rank": local_rank}


def _maybe_init_dist(device: torch.device) -> Dict[str, int]:
    info = _dist_info()
    if info["world_size"] <= 1:
        return info
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    if device.type == "cuda":
        torch.cuda.set_device(info["local_rank"])
    return info


def _lr_at(step: int, *, base_lr: float, warmup: int) -> float:
    if warmup <= 0:
        return base_lr
    if step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    return base_lr


def _save_ckpt(path: Path, *, model: nn.Module, opt: torch.optim.Optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "optimizer": opt.state_dict(),
    }
    torch.save(state, path)


def _load_ckpt(path: Path, *, model: nn.Module, opt: torch.optim.Optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["model"]
    (model.module if hasattr(model, "module") else model).load_state_dict(sd, strict=True)
    opt.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


def _rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def build_model(model_cfg: Dict[str, Any], *, max_seq_len: int) -> nn.Module:
    mtype = model_cfg["type"]
    if mtype == "transformer":
        cfg = TransformerConfig(
            vocab_size=int(model_cfg["vocab_size"]),
            d_model=int(model_cfg["d_model"]),
            n_layers=int(model_cfg["n_layers"]),
            n_heads=int(model_cfg["n_heads"]),
            d_ff=int(model_cfg["d_ff"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
            use_flash_attn=bool(model_cfg.get("use_flash_attn", False)),
            max_seq_len=max_seq_len,
        )
        return TransformerLM(cfg)
    if mtype == "mamba":
        cfg = MambaConfig(
            vocab_size=int(model_cfg["vocab_size"]),
            d_model=int(model_cfg["d_model"]),
            n_layers=int(model_cfg["n_layers"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
            mamba_d_state=int(model_cfg.get("mamba_d_state", 64)),
            mamba_d_conv=int(model_cfg.get("mamba_d_conv", 4)),
            mamba_expand=int(model_cfg.get("mamba_expand", 2)),
        )
        return build_mamba_or_fallback(cfg)
    raise ValueError(f"Unknown model type: {mtype}")


def build_loader(
    *,
    split: str,
    shards_dir: str,
    seq_len: int,
    batch_size: int,
    seed: int,
    loader_cfg: Dict[str, Any],
    doc_token: str,
) -> Any:
    shard = load_split(shards_dir, split)
    # Prefer shard meta for ids; doc_token string is kept for configs/docs.
    doc_token_id = shard.meta.doc_id
    eos_id = shard.meta.eos_id
    packing = loader_cfg["packing"]
    boundary_markers = bool(loader_cfg.get("boundary_markers", True))
    reset_masks = bool(loader_cfg.get("reset_masks", False))
    if packing == "packed":
        return PackedStreamLoader(
            shard,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
            boundary_markers=boundary_markers,
            reset_masks=reset_masks,
            doc_token_id=doc_token_id,
            eos_id=eos_id,
        )
    if packing == "no_packing":
        return NoPackingChunkLoader(
            shard,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
            boundary_markers=boundary_markers,
            reset_masks=reset_masks,
            doc_token_id=doc_token_id,
            eos_id=eos_id,
        )
    raise ValueError(f"Unknown packing: {packing}")


def train_experiment(
    *,
    root_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    seed = int(root_cfg["seed"])
    data_cfg = root_cfg["data"]
    train_cfg_raw = root_cfg["train"]
    train_cfg = TrainConfig(
        device=str(train_cfg_raw["device"]),
        dtype=str(train_cfg_raw["dtype"]),
        deterministic=bool(train_cfg_raw.get("deterministic", False)),
        seq_len=int(train_cfg_raw["seq_len"]),
        batch_size=int(train_cfg_raw["batch_size"]),
        grad_accum_steps=int(train_cfg_raw["grad_accum_steps"]),
        max_steps=int(train_cfg_raw["max_steps"]),
        lr=float(train_cfg_raw["lr"]),
        weight_decay=float(train_cfg_raw["weight_decay"]),
        warmup_steps=int(train_cfg_raw.get("warmup_steps", 0)),
        max_grad_norm=float(train_cfg_raw.get("max_grad_norm", 1.0)),
        log_every=int(train_cfg_raw.get("log_every", 50)),
        ckpt_every=int(train_cfg_raw.get("ckpt_every", 1000)),
    )

    device = choose_device(train_cfg.device)
    dtype = choose_dtype(train_cfg.dtype)
    dist_info = _maybe_init_dist(device)

    # Use distinct seeds per-rank so packed sampling isn't identical across ranks.
    rank_seed = seed + 1000 * dist_info["rank"]
    seed_everything(ReproConfig(seed=rank_seed, deterministic=train_cfg.deterministic))

    run_dir = Path(exp_cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    if _rank0():
        write_manifest(
            run_dir,
            config={"root": root_cfg, "experiment": exp_cfg},
            seed=seed,
            extra={"dist": dist_info},
        )

    model = build_model(exp_cfg["model"], max_seq_len=train_cfg.seq_len).to(device)
    if dtype in (torch.float16, torch.bfloat16) and device.type == "cuda":
        # weights stay fp32; autocast controls matmul/attention; this keeps stability in tiny runs too.
        pass

    if dist_info["world_size"] > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_info["local_rank"]] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    ckpt_last = run_dir / "checkpoint_last.pt"
    start_step = 0
    if ckpt_last.exists():
        start_step = _load_ckpt(ckpt_last, model=model, opt=opt) + 1

    loader = build_loader(
        split="train",
        shards_dir=str(data_cfg["shards_dir"]),
        seq_len=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        seed=rank_seed,
        loader_cfg=exp_cfg["loader"],
        doc_token=str(data_cfg.get("doc_token", "<DOC>")),
    )
    it = iter(loader)

    model.train()
    t0 = time.time()
    last_log = t0
    tokens_seen = 0

    metrics_path = run_dir / "train_metrics.jsonl"
    if _rank0() and start_step == 0 and metrics_path.exists():
        metrics_path.unlink()

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and device.type == "cuda"))

    for step in range(start_step, train_cfg.max_steps):
        lr = _lr_at(step, base_lr=train_cfg.lr, warmup=train_cfg.warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(train_cfg.grad_accum_steps):
            batch: Batch = next(it)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)
            loss_mask = batch.loss_mask.to(device, non_blocking=True)
            segment_ids = batch.segment_ids.to(device, non_blocking=True) if batch.segment_ids is not None else None

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda" and dtype != torch.float32)):
                logits = model(input_ids, segment_ids=segment_ids)  # (B,T,V)
                v = logits.shape[-1]
                loss_flat = F.cross_entropy(logits.view(-1, v), labels.reshape(-1), reduction="none")
                loss = (loss_flat.view(labels.shape) * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

            loss = loss / float(train_cfg.grad_accum_steps)
            total_loss += float(loss.detach().cpu())

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tokens_seen += int(labels.numel())

        if scaler.is_enabled():
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        if dist_info["world_size"] > 1:
            # Reduce loss to rank0 for logging.
            loss_t = torch.tensor([total_loss], device=device, dtype=torch.float32)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            total_loss = float(loss_t.item() / dist_info["world_size"])

        now = time.time()
        if _rank0() and (step % train_cfg.log_every == 0 or step == train_cfg.max_steps - 1):
            dt = now - last_log
            tok_per_s = (train_cfg.batch_size * train_cfg.seq_len * train_cfg.grad_accum_steps) / max(dt, 1e-9)
            rec = {
                "step": step,
                "loss": total_loss,
                "lr": lr,
                "tokens_per_s": tok_per_s,
                "elapsed_s": now - t0,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            last_log = now

        if _rank0() and (step % train_cfg.ckpt_every == 0 or step == train_cfg.max_steps - 1):
            _save_ckpt(run_dir / f"checkpoint_step{step}.pt", model=model, opt=opt, step=step)
            _save_ckpt(ckpt_last, model=model, opt=opt, step=step)

        if dist_info["world_size"] > 1:
            dist.barrier()

    summary = {
        "run_dir": str(run_dir),
        "final_step": train_cfg.max_steps - 1,
        "train_seconds": time.time() - t0,
        "tokens_seen_local": tokens_seen,
    }
    if _rank0():
        (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

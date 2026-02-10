from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _git_rev() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def write_manifest(
    out_dir: str | Path,
    *,
    config: Dict[str, Any],
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "seed": seed,
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "git_commit": _git_rev(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
    }
    if extra:
        manifest["extra"] = extra

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


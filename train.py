from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow `src/` imports without installing a package.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wiki_lm.train_loop import train_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/smoke.yaml)")
    ap.add_argument("--stage", default="train", choices=["train"], help="Stage to run")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exps = cfg.get("experiments", [])
    if not exps:
        raise SystemExit("No experiments defined in config.")

    for exp in exps:
        train_experiment(root_cfg=cfg, exp_cfg=exp)


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wiki_lm.eval.perplexity import eval_perplexity  # noqa: E402
from wiki_lm.eval.suffix_id import eval_suffix_identification  # noqa: E402
from wiki_lm.train_loop import build_model  # noqa: E402
from wiki_lm.utils.repro import choose_device, choose_dtype  # noqa: E402


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing manifest.json in {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_ckpt(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "checkpoint_last.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing checkpoint_last.pt in {run_dir}")
    return torch.load(p, map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(str(cfg["train"]["device"]))
    dtype = choose_dtype(str(cfg["train"]["dtype"]))

    data_cfg = cfg["data"]
    shards_dir = str(data_cfg["shards_dir"])
    seq_len = int(cfg["train"]["seq_len"])

    eval_cfg = cfg.get("eval", {})
    ppl_cfg = eval_cfg.get("perplexity", {})
    suf_cfg = eval_cfg.get("suffix_id", {})

    runs = []
    for exp in cfg.get("experiments", []):
        run_dir = Path(exp["run_dir"])
        manifest = _load_manifest(run_dir)
        ckpt = _load_ckpt(run_dir)

        # Rebuild model from experiment config (not from checkpoint) to keep it explicit/reproducible.
        model = build_model(exp["model"], max_seq_len=seq_len)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)

        # Eval
        ppl = eval_perplexity(
            model,
            shards_dir=shards_dir,
            seq_len=seq_len,
            batch_size=4,
            max_batches=int(ppl_cfg.get("max_batches", 50)),
            seed=int(cfg["seed"]) + 999,
            device=device,
        )
        suffix = eval_suffix_identification(
            model,
            shards_dir=shards_dir,
            buckets=list(map(int, suf_cfg.get("buckets", [128, 512, 1024]))),
            suffix_len=int(suf_cfg.get("suffix_len", 16)),
            num_examples=int(suf_cfg.get("num_examples", 200)),
            num_distractors=int(suf_cfg.get("num_distractors", 3)),
            seed=int(cfg["seed"]) + 2024,
            device=device,
        )

        run_rec = {
            "name": exp["name"],
            "run_dir": str(run_dir),
            "model_type": exp["model"]["type"],
            "train_summary_path": str(run_dir / "train_summary.json"),
            "perplexity": {"nll": ppl.nll, "ppl": ppl.ppl, "batches": ppl.batches},
            "suffix_id": {
                str(k): {"ctx_len": v.bucket_ctx_len, "examples": v.examples, "accuracy": v.accuracy}
                for k, v in suffix.items()
            },
        }
        (run_dir / "eval.json").write_text(json.dumps(run_rec, indent=2, sort_keys=True), encoding="utf-8")
        runs.append(run_rec)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config_path": args.config,
        "device": str(device),
        "dtype": str(dtype),
        "runs": runs,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()


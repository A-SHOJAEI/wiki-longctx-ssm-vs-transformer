from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import sentencepiece as spm


def train_sentencepiece(
    *,
    in_jsonl: str,
    out_dir: str,
    vocab_size: int,
    seed: int,
    sample_bytes: int,
    doc_token: str,
) -> None:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    prefix = out_dir_p / "sp"

    # SentencePiece can read JSONL if we provide an "input" file containing plain text lines.
    # We generate it deterministically here.
    tmp_txt = out_dir_p / "sp_input.txt"
    if not tmp_txt.exists():
        import json

        written = 0
        with open(in_jsonl, "r", encoding="utf-8") as f_in, tmp_txt.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = str(obj.get("text", "")).replace("\r", " ").replace("\n", " ").strip()
                if text:
                    f_out.write(text + "\n")
                    written += len(text) + 1
                    if sample_bytes and written >= sample_bytes:
                        break

    args = [
        f"--input={tmp_txt}",
        f"--model_prefix={prefix}",
        "--model_type=bpe",
        f"--vocab_size={vocab_size}",
        f"--character_coverage=1.0",
        # Keep the smoke pipeline reproducible across SentencePiece versions.
        # sentencepiece==0.2.0 (Python 3.12 wheels) does not support the legacy
        # `random_seed` TrainerSpec field, so avoid any randomized sampling/shuffling.
        "--input_sentence_size=0",
        "--shuffle_input_sentence=false",
        # Explicit special token ids for stable downstream code.
        "--unk_id=0",
        "--bos_id=1",
        "--eos_id=2",
        "--pad_id=3",
        f"--user_defined_symbols={doc_token}",
        "--hard_vocab_limit=false",
    ]
    spm.SentencePieceTrainer.Train(" ".join(args))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--sample_bytes", type=int, default=2_000_000_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--doc_token", type=str, default="<DOC>")
    args = ap.parse_args()

    train_sentencepiece(
        in_jsonl=args.in_jsonl,
        out_dir=args.out_dir,
        vocab_size=args.vocab_size,
        seed=args.seed,
        sample_bytes=args.sample_bytes,
        doc_token=args.doc_token,
    )


if __name__ == "__main__":
    main()

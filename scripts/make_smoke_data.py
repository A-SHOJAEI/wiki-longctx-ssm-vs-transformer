from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from wiki_lm.utils.jsonl import write_jsonl  # noqa: E402


def _synthetic_docs(seed: int, n: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    topics = [
        "Astronomy",
        "Biology",
        "Computer_Science",
        "History",
        "Mathematics",
        "Music",
        "Physics",
        "Geography",
        "Linguistics",
        "Economics",
    ]
    patterns = [
        "This article discusses {topic}. It defines key terms, gives examples, and notes common misconceptions.",
        "A short overview of {topic}. The goal is clarity, not completeness. Readers should consult primary sources for details.",
        "{topic} is presented here as a structured note: definition, context, and a small list of related ideas.",
    ]
    docs = []
    for i in range(n):
        topic = rng.choice(topics)
        base = rng.choice(patterns).format(topic=topic.replace("_", " "))
        # Create longer structure to exercise packing and boundaries.
        paras = []
        for p in range(rng.randint(3, 7)):
            paras.append(
                f"Section {p+1}. {base} "
                f"Example {p+1}: consider a simple scenario where {topic.lower()} interacts with constraints. "
                f"Conclusion {p+1}: the main idea is preserved under reformulation."
            )
        text = "\n\n".join(paras)
        docs.append({"id": f"smoke_{i:05d}", "title": topic, "text": text})
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_docs", type=int, default=80)
    ap.add_argument("--vocab_size", type=int, default=8000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    clean_dir = out_dir / "clean"
    tok_dir = out_dir / "tokenizer"
    shards_dir = out_dir / "shards"
    clean_dir.mkdir(parents=True, exist_ok=True)

    docs_path = clean_dir / "docs.jsonl"
    if not docs_path.exists():
        docs = _synthetic_docs(args.seed, args.num_docs)
        write_jsonl(docs_path, docs)

    # Train tokenizer
    sp_model = tok_dir / "sp.model"
    if not sp_model.exists():
        from scripts.train_tokenizer import train_sentencepiece  # local import

        train_sentencepiece(
            in_jsonl=str(docs_path),
            out_dir=str(tok_dir),
            vocab_size=args.vocab_size,
            seed=args.seed,
            sample_bytes=2_000_000,
            doc_token="<DOC>",
        )

    # Tokenize + shard
    meta_path = shards_dir / "train" / "meta.json"
    if not meta_path.exists():
        from scripts.tokenize_shard import tokenize_and_shard  # local import

        tokenize_and_shard(
            in_jsonl=str(docs_path),
            tokenizer_model=str(sp_model),
            out_dir=str(shards_dir),
            seed=args.seed,
            shard_size_tokens=200_000,
            split_seed=args.seed,
            split_ratios=(0.90, 0.05, 0.05),
            doc_token="<DOC>",
        )


if __name__ == "__main__":
    main()

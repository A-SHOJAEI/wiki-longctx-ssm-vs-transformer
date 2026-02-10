from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import mwxml
import mwparserfromhell
from tqdm import tqdm


def _clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    # Collapse runs of whitespace but keep paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_wikitext(wikitext: str) -> str:
    code = mwparserfromhell.parse(wikitext)
    return _clean_text(code.strip_code(normalize=True, collapse=True))


def _simhash64(tokens: List[str]) -> int:
    # Deterministic simhash over tokens using sha1.
    v = [0] * 64
    for t in tokens:
        h = hashlib.sha1(t.encode("utf-8")).digest()
        x = int.from_bytes(h[:8], "big", signed=False)
        for i in range(64):
            bit = (x >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i, s in enumerate(v):
        if s > 0:
            out |= 1 << i
    return out


def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class SimhashDeduper:
    def __init__(self, *, max_hamming: int = 3) -> None:
        self.max_hamming = max_hamming
        self.buckets: List[Dict[int, List[int]]] = [dict() for _ in range(4)]
        self.seen: List[int] = []

    def is_duplicate(self, h: int) -> bool:
        # 4 bands of 16 bits each.
        cands: List[int] = []
        for band in range(4):
            key = (h >> (band * 16)) & 0xFFFF
            cands.extend(self.buckets[band].get(key, []))
        for idx in cands:
            if _hamming64(h, self.seen[idx]) <= self.max_hamming:
                return True
        idx = len(self.seen)
        self.seen.append(h)
        for band in range(4):
            key = (h >> (band * 16)) & 0xFFFF
            self.buckets[band].setdefault(key, []).append(idx)
        return False


def iter_wiki_docs(
    xml_bz2_path: Path,
    *,
    min_chars: int,
    max_docs: int,
    dedup: bool,
    max_hamming: int,
) -> Iterator[Dict[str, str]]:
    deduper = SimhashDeduper(max_hamming=max_hamming) if dedup else None
    with bz2.open(xml_bz2_path, "rb") as f:
        dump = mwxml.Dump.from_file(f)
        count = 0
        for page in dump.pages:
            if page.namespace != 0:
                continue
            if page.redirect:
                continue
            rev = None
            for r in page:
                rev = r
            if rev is None or rev.text is None:
                continue
            text = _strip_wikitext(rev.text)
            if len(text) < min_chars:
                continue

            if deduper is not None:
                toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
                h = _simhash64(toks[:5000])  # cap tokens for speed
                if deduper.is_duplicate(h):
                    continue

            yield {"id": str(page.id), "title": page.title, "text": text}
            count += 1
            if max_docs and count >= max_docs:
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to enwiki-...xml.bz2")
    ap.add_argument("--out", required=True, help="Output directory (writes docs.jsonl)")
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_docs", type=int, default=0, help="0 means no limit")
    ap.add_argument("--dedup", action="store_true", help="Enable near-duplicate filtering (SimHash)")
    ap.add_argument("--dedup_max_hamming", type=int, default=3)
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "docs.jsonl"

    max_docs = int(args.max_docs)
    docs = iter_wiki_docs(
        in_path,
        min_chars=int(args.min_chars),
        max_docs=max_docs,
        dedup=bool(args.dedup),
        max_hamming=int(args.dedup_max_hamming),
    )

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for doc in tqdm(docs, desc="extract", unit="doc"):
            f.write(json.dumps(doc, ensure_ascii=True) + "\n")
            written += 1
    meta = {
        "input": str(in_path),
        "output": str(out_path),
        "written_docs": written,
        "min_chars": int(args.min_chars),
        "max_docs": max_docs,
        "dedup": bool(args.dedup),
        "dedup_max_hamming": int(args.dedup_max_hamming),
    }
    (out_dir / "extract_manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()


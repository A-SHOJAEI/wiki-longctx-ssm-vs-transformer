from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from tqdm import tqdm


def _sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _try_fetch_sha256(url: str, timeout_s: int = 30) -> Optional[str]:
    """
    Wikimedia dumps often publish a sidecar `*.sha256` file for the dump file.
    We attempt `url + ".sha256"` and parse the first hex token.
    """
    sha_url = url + ".sha256"
    try:
        r = requests.get(sha_url, timeout=timeout_s)
        if r.status_code != 200:
            return None
        txt = r.text.strip().splitlines()
        if not txt:
            return None
        # Common format: "<hash> <filename>"
        first = txt[0].strip().split()
        if not first:
            return None
        h = first[0].strip().lower()
        if len(h) == 64 and all(c in "0123456789abcdef" for c in h):
            return h
        return None
    except Exception:
        return None


def _download(url: str, out_path: Path, *, timeout_s: int = 60) -> Dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0")) if r.headers.get("Content-Length") else None
        h = hashlib.sha256()
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                h.update(chunk)
                pbar.update(len(chunk))

    tmp.replace(out_path)
    return {"bytes": out_path.stat().st_size, "sha256": h.hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory (e.g., data/raw)")
    ap.add_argument("--dump", default="enwiki-latest", help="Dump prefix, default enwiki-latest")
    ap.add_argument(
        "--base_url",
        default="https://dumps.wikimedia.org/enwiki/latest/",
        help="Base URL for dumps (default points at enwiki/latest)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f"{args.dump}-pages-articles-multistream.xml.bz2",
        f"{args.dump}-pages-articles-multistream-index.txt.bz2",
    ]
    urls = [args.base_url.rstrip("/") + "/" + f for f in files]

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "dump": args.dump,
        "base_url": args.base_url,
        "files": [],
    }

    for url, fname in zip(urls, files):
        out_path = out_dir / fname
        expected = _try_fetch_sha256(url)
        rec: Dict[str, object] = {"url": url, "path": str(out_path), "expected_sha256": expected}

        if out_path.exists() and not args.overwrite:
            local = _sha256_file(out_path)
            rec["downloaded"] = False
            rec["sha256"] = local
        else:
            info = _download(url, out_path)
            rec["downloaded"] = True
            rec.update(info)

        if expected is not None:
            rec["verified"] = (str(rec["sha256"]) == expected)
            if not rec["verified"]:
                raise SystemExit(f"Checksum mismatch for {fname}: expected {expected}, got {rec['sha256']}")
        else:
            rec["verified"] = False

        manifest["files"].append(rec)

    (out_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()


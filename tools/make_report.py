from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _md_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to artifacts/results.json")
    ap.add_argument("--out_md", required=True, help="Path to write artifacts/report.md")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    runs: List[Dict[str, Any]] = results.get("runs", [])

    lines = []
    lines.append("# Report")
    lines.append("")
    lines.append(f"Generated: `{results.get('generated_at_utc', '')}`")
    lines.append(f"Device: `{results.get('device', '')}`")
    lines.append("")

    # Perplexity table
    rows = [["run", "model", "val_nll", "val_ppl"]]
    for r in runs:
        p = r.get("perplexity", {})
        rows.append(
            [
                str(r.get("name", "")),
                str(r.get("model_type", "")),
                f"{float(p.get('nll', 0.0)):.4f}",
                f"{float(p.get('ppl', 0.0)):.2f}",
            ]
        )
    lines.append("## Validation Perplexity")
    lines.append("")
    lines.append(_md_table(rows))
    lines.append("")

    # Suffix-id table (best-effort)
    lines.append("## Long-Range Suffix Identification")
    lines.append("")
    # Collect all bucket keys
    buckets = set()
    for r in runs:
        buckets |= set((r.get("suffix_id") or {}).keys())
    buckets_sorted = sorted(buckets, key=lambda s: int(s))
    header = ["run"] + [f"acc@{b}" for b in buckets_sorted]
    srows = [header]
    for r in runs:
        sid = r.get("suffix_id") or {}
        row = [str(r.get("name", ""))]
        for b in buckets_sorted:
            acc = float((sid.get(b) or {}).get("accuracy", 0.0))
            row.append(f"{acc:.3f}")
        srows.append(row)
    lines.append(_md_table(srows))
    lines.append("")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


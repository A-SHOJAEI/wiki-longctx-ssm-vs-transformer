# wiki-longctx-ssm-vs-transformer

Compare long-context language modeling behavior between a decoder-only Transformer and an SSM-style model on Wikipedia-like text, with explicit ablations around packing and document-boundary leakage.

## Problem Statement

Given the same tokenized corpus and training loop, how do:

- a causal Transformer LM (`src/wiki_lm/models/transformer.py`)
- a Mamba/SSM LM when available, otherwise an SSM fallback (`src/wiki_lm/models/mamba_lm.py`, `src/wiki_lm/models/ssm_fallback.py`)

trade off perplexity and long-range usage of context? This repo measures:

- validation NLL/perplexity on held-out tokens
- a contrastive long-range task: suffix identification accuracy as context length increases

## Dataset Provenance

This repo supports two data sources.

Smoke dataset (checked-in pipeline, no external downloads):
- `scripts/make_smoke_data.py` generates 80 synthetic "Wikipedia-like" documents (topic paragraphs), trains a SentencePiece BPE tokenizer (default vocab 8000), tokenizes, and writes shards under `data/smoke/shards/`.
- Deterministic split uses `stable_hash_to_split()` in `src/wiki_lm/data/shards.py` with ratios `(0.90, 0.05, 0.05)` and seed `1234` (called from `scripts/tokenize_shard.py`).

Full Wikipedia dataset (downloaded by you):
- `scripts/download_wiki.py` downloads the English Wikipedia multistream dump files from Wikimedia Dumps and verifies `*.sha256` when present.
- `scripts/extract_wiki.py` parses the `*.xml.bz2` dump (namespace 0 only), strips wikitext with `mwparserfromhell`, optionally near-dedups with a simple SimHash filter, and writes `docs.jsonl` (one JSON per page: `{id,title,text}`).
- `scripts/train_tokenizer.py` trains SentencePiece BPE with a user-defined `<DOC>` token.
- `scripts/tokenize_shard.py` inserts `<DOC>` at document starts and `<eos>` at ends, then writes per-split `shard*.npz` plus `meta.json` under `data/shards/{train,val,test}/`.

Wikipedia dumps are provided by Wikimedia under CC BY-SA 4.0 with attribution requirements (and some historical content under GFDL as noted by Wikimedia). This repo ships code only; data is downloaded separately.

## Methodology (What The Code Actually Does)

**Tokenization + shards**

- SentencePiece BPE with fixed special ids: `unk=0, bos=1, eos=2, pad=3` and one user-defined symbol `<DOC>` (`scripts/train_tokenizer.py`).
- Shard format: each `shard*.npz` contains a flat `tokens` array and `doc_offsets` so docs can be reconstructed; `meta.json` stores `vocab_size` and special token ids (`src/wiki_lm/data/shards.py`).

**Training objective**

- Standard next-token cross-entropy over `(input_ids, labels)` created from contiguous token windows (`src/wiki_lm/data/loader.py`).
- "Packed" training samples random windows from the global token stream (documents concatenated but marked by `<DOC>`).
- "No packing" samples only full fixed-length chunks wholly inside a single document, dropping remainders (`NoPackingChunkLoader`).

**Document boundary leakage controls**

- `boundary_markers`: if false, `<DOC>` is mapped to `<eos>` at runtime to remove the boundary signal (`src/wiki_lm/data/loader.py`).
- `reset_masks`: if true, segment ids are derived from `<DOC>` positions and used to block attention across documents for the Transformer; the Mamba path enforces this by running each segment independently (correct but slower) (`src/wiki_lm/models/transformer.py`, `src/wiki_lm/models/mamba_lm.py`).

**Evaluation**

- Perplexity (`src/wiki_lm/eval/perplexity.py`): always uses the packed stream loader on the `val` split with `boundary_markers=True` and `reset_masks=True`, regardless of the training packing ablation.
- Long-range suffix identification (`src/wiki_lm/eval/suffix_id.py`): for each context length bucket `L`, sample a window of `L` tokens from a held-out `test` document, take the true next `suffix_len` tokens as the correct suffix, sample `num_distractors` suffixes from other documents, and predict the candidate with lowest teacher-forced suffix NLL.

## Baselines And Ablations (As Run)

The included results were produced with `configs/smoke.yaml` (seq_len 128, max_steps 40):

- `smoke_transformer_packed`: Transformer baseline, packed loader, `reset_masks: true`
- `smoke_mamba_packed`: "mamba" config, packed loader, `reset_masks: true`
- Note: `mamba-ssm` is not installed in the pinned environment (`requirements.txt`), so `smoke_mamba_packed` uses the GRU-based `SSMFallbackLM` from `src/wiki_lm/models/ssm_fallback.py`.
- `smoke_transformer_no_packing`: packing ablation for the Transformer (`no_packing` vs `packed`)

Reference (not executed by default): `configs/transformer_130m.yaml`, `configs/mamba_130m.yaml` (4096 ctx, bf16, 10k steps).

## Results (Smoke Run)

Exact artifacts:

- machine-readable: `artifacts/results.json` (generated `2026-02-20 06:26:06 UTC`, device `cuda`, dtype `torch.float32`)
- human report: `artifacts/report.md`

Validation perplexity (see `artifacts/report.md` section `## Validation Perplexity`):

| run | model | val_nll | val_ppl |
| --- | --- | --- | --- |
| smoke_transformer_packed | transformer | 5.7704 | 320.65 |
| smoke_mamba_packed | mamba | 7.8316 | 2518.92 |
| smoke_transformer_no_packing | transformer | 5.8402 | 343.84 |

Long-range suffix identification accuracy (see `artifacts/report.md` section `## Long-Range Suffix Identification`):

| run | acc@32 | acc@64 | acc@112 |
| --- | --- | --- | --- |
| smoke_transformer_packed | 0.400 | 0.367 | 0.633 |
| smoke_mamba_packed | 0.350 | 0.500 | 0.583 |
| smoke_transformer_no_packing | 0.350 | 0.383 | 0.633 |

## Repro Instructions

Smoke (reproduces the pipeline end-to-end; clears prior checkpoints to avoid "resume and do nothing"):

```bash
make clean
make all
```

What runs:

- `make data`: `python scripts/make_smoke_data.py --out_dir data/smoke`
- `make train`: `python train.py --config configs/smoke.yaml`
- `make eval`: `python evaluate.py --config configs/smoke.yaml --out_dir artifacts`
- `make report`: `python tools/make_report.py --results artifacts/results.json --out_md artifacts/report.md`

Full Wikipedia (large; you manage disk/compute):

```bash
make setup
.venv/bin/python scripts/download_wiki.py --out data/raw --dump enwiki-latest
.venv/bin/python scripts/extract_wiki.py --in data/raw/enwiki-latest-pages-articles-multistream.xml.bz2 --out data/clean --max_docs 0
.venv/bin/python scripts/train_tokenizer.py --in_jsonl data/clean/docs.jsonl --out_dir data/tokenizer --vocab_size 32000 --seed 1234
.venv/bin/python scripts/tokenize_shard.py --in_jsonl data/clean/docs.jsonl --tokenizer_model data/tokenizer/sp.model --out_dir data/shards --seed 1234
.venv/bin/python train.py --config configs/transformer_130m.yaml
.venv/bin/python evaluate.py --config configs/transformer_130m.yaml --out_dir artifacts
```

Notes:

- CUDA vs CPU is controlled by `train.device` in YAML via `auto|cpu|cuda` (`src/wiki_lm/utils/repro.py`).
- Optional performance/faithfulness upgrades are not pinned in `requirements.txt`:
  - install `mamba-ssm` to run real Mamba blocks (`src/wiki_lm/models/mamba_lm.py`)
  - set `use_flash_attn: true` to use PyTorch SDPA fast path when available (`src/wiki_lm/models/transformer.py`)

## Limitations (Current Artifacts)

- The checked-in results are a smoke study on synthetic documents with tiny models (d_model=128, 2 layers) and only 40 optimizer steps; treat numbers as sanity checks, not model comparisons.
- The "mamba" smoke result is a GRU fallback (`SSMFallbackLM`), not `mamba-ssm`, because `mamba_ssm` is absent in the pinned environment.
- Perplexity evaluation always uses packed windows with boundary markers and reset masks enabled (`src/wiki_lm/eval/perplexity.py`), so "no packing" only changes training, not the val evaluation data loader.
- Suffix identification is teacher-forced NLL scoring, not free-form generation; it probes context conditioning but is not a full retrieval benchmark.
- No explicit "equal compute" accounting is implemented (no FLOPs/token budget matching, throughput reporting, or token-count-normalized training curves).

## Next Research Steps

- Run on real Wikipedia shards (`data/shards`) with matched parameter counts and a fixed token budget across architectures; log tokens processed and wall-clock throughput per run.
- Replace the smoke SSM fallback with `mamba-ssm` and compare across longer contexts (e.g., 1k/2k/4k) and multiple random seeds.
- Add the existing ablation toggles not exercised in `configs/smoke.yaml`: `reset_masks: false` (controlled doc leakage).
- Add the existing ablation toggles not exercised in `configs/smoke.yaml`: `boundary_markers: false` (remove boundary signal by mapping `<DOC>` to `<eos>`).
- Extend evaluation with perplexity vs context length (same tokens, different truncation).
- Extend evaluation with additional long-range probes (needle-in-haystack variants, copy tasks, entity recurrence).

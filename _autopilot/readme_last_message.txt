`README.md` has been fully rewritten to be specific to this repo’s implemented pipeline and the checked-in artifacts, including:

- Problem statement, dataset provenance (smoke + full Wikipedia), and methodology grounded in the actual code paths.
- Baselines/ablations exactly as in `configs/smoke.yaml`.
- Exact smoke results reproduced from `artifacts/results.json` / `artifacts/report.md` with explicit references to the report sections.
- Repro commands (`make clean && make all`), plus the full Wikipedia pipeline commands.
- Limitations that explicitly call out that `smoke_mamba_packed` used the GRU fallback (`mamba_ssm` not installed) and that perplexity eval always uses packed windows.
- Concrete next research steps aligned to what’s already supported by the code/configs.

File: `README.md`
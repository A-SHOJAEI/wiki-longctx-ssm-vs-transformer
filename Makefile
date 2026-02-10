PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := /usr/bin/env bash
.DEFAULT_GOAL := all

PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup data train eval report all clean

setup: .venv/.ready
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi

.venv/.ready: scripts/bootstrap_venv.sh requirements.txt
	@bash scripts/bootstrap_venv.sh
	@mkdir -p .venv
	@touch .venv/.ready

data: setup
	@$(PY) scripts/make_smoke_data.py --out_dir data/smoke

train: setup data
	@$(PY) train.py --config configs/smoke.yaml --stage train

eval: setup data
	@$(PY) evaluate.py --config configs/smoke.yaml --out_dir artifacts

report: setup eval
	@$(PY) tools/make_report.py --results artifacts/results.json --out_md artifacts/report.md

all: setup data train eval report

clean:
	@rm -rf artifacts/* runs/* data/smoke

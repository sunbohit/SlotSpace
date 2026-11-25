# GPU machine handoff

Code, schemas, templates and portable configuration live in this repository.
Raw STaRK files, model weights, environments and machine-specific run logs are
not Git payloads. Download the pinned public inputs below; do not treat the
unreleased source candidate as a completed M7/M8 experiment.

## Linux research environment

Run from the cloned repository. Use a CUDA-capable machine (the design target
is one 24 GiB GPU, 64 GiB RAM and 200 GiB free disk). Do not install or replace
system GPU drivers with this setup.

```bash
conda env create -f environments/research.yaml
conda activate slotspace
# CUDA 12.8 example: verify compatibility with the machine's driver first.
python -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c environments/research.constraints.txt -e ".[test]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The official index has a Python 3.11 Linux x86_64 CUDA 12.8 wheel for Torch
2.10.0. This is a prepared configuration, **not a GPU-tested environment**.
Select a different official build if the actual driver/GPU requires it.

## Separate Linux generation environment

```bash
conda env create -f environments/generation.yaml
conda activate slotspace-gen
python -m pip install -r environments/generation-linux.lock.txt
python -m pip install --no-deps -e .
```

This pinned generator stack uses vLLM 0.34.0 and its own Torch/CUDA dependencies.
It must not be mixed into the research environment. Its NVIDIA/runtime license
and driver requirements need review on the actual machine.

## Download inputs, not Git-hosted weights

Activate the research environment. All revisions and checksums are recorded in
[the original resource manifest](../reproducibility/huggingface.csv).

```bash
hf download Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --local-dir ../Resource/models/Qwen/Qwen3-Embedding-0.6B
hf download answerdotai/ModernBERT-base --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 --include "*.json" "README.md" "model.safetensors" --local-dir ../Resource/models/answerdotai/ModernBERT-base
hf download BAAI/bge-reranker-v2-m3 --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --local-dir ../Resource/models/BAAI/bge-reranker-v2-m3
hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir ../Resource/models/Qwen/Qwen3-4B-Instruct-2507
python -c "from pathlib import Path; from slotspace.prime_cache import prepare_prime; print(prepare_prime(Path('../Resource/datasets/stark'), download=True))"
```

`prepare_prime` downloads the pinned official release, checks its SHA-256 and
extracts only the expected six processed members. It preserves an existing
cache and reports incompatible files instead of overwriting them.

## Start with checks, then generate missing artifacts

```bash
python -m pytest -q -m "not vespa"
python -m slotspace.reproduce --resources ../Resource --dry-run
python -m slotspace.reproduce --resources ../Resource --gen-python /absolute/path/to/slotspace-gen/bin/python
```

The default work directory is `runs/reproduce-v0`. It contains generated
configuration and artifacts; those paths need not match the older readiness
check's default examples. Resume only with unchanged settings and explicit
review/approval files as described in [the README](../README.md).

H1-H4 need human reviews. M7 production acceptance must pass before a frozen
configuration is produced, and H5 must authorize its exact protocol before
formal test/human experiments. The committed pilot summaries are preliminary:
four validation queries, zero-shot T=1, high ECE and no demonstrated CPU
efficiency gain. They do not authorize full M8.

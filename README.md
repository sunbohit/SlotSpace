# SlotSpace

SlotSpace retrieves facts, not just documents that mention the same entity. A value, its role and its context must match **in the same fact**. Several plausible readings of a query share one candidate retrieval pass. See the [design](docs/DESIGN.md) and [milestone guide](docs/DEVELOPMENT.md).

## M0: prepare STaRK-Prime (CPU only)

Run from this repository in the prepared Python 3.11 environment. For the Windows workspace:

```powershell
$conda = "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
$prefix = (Resolve-Path ..\.envs\slotspace).Path
& $conda run --prefix $prefix python -m pip install --no-deps --no-build-isolation -e .
$env:SLOTSPACE_STARK_ROOT = (Resolve-Path ..\Resource\datasets\stark).Path
& $conda run --prefix $prefix python -m slotspace.stark_io --root $env:SLOTSPACE_STARK_ROOT --limit 100
& $conda run --prefix $prefix python -m slotspace.stark_io --root $env:SLOTSPACE_STARK_ROOT
$env:SLOTSPACE_PRIME_EXPORT = (Resolve-Path data\prime).Path
& $conda run --prefix $prefix python -m pytest -q
```

- The full export writes `docs.parquet`, `edges.parquet` and `queries_{train,val,test,test-0.1,human}.jsonl` under `data/prime/`. It is an export, **not** a retrieval run on the test sets. The reference and parity tests use synthetic data.
- A limited export has its own directory (`data/prime-sample-100/`). It includes only edges with both ends in the subset and queries with all answers in the subset. Neither mode overwrites an existing output. To repeat a run, give it a fresh `--out` and, for H0, point `SLOTSPACE_PRIME_EXPORT` there.
- No local cache yet? With the dependencies installed, `python -m slotspace.stark_io --download` populates `data/stark_cache/prime` from the pinned public release. Without `--download`, no data is fetched. The cache root is the **parent** of `prime/`. Per-run configuration, source revision, Git status, timestamps, machine and installed versions go to `runs/<id>/meta.json`.
- `python -m pytest -q -m "not slow"` runs without a dataset. For full H0 acceptance, set **both** environment variables as above; the slow tests also compare 100 rendered documents byte-for-byte with the official API.

The reference scorer, its tests, fast scorer, simulation and statistics helpers are the unmodified appendices from the guide.

## M1: retrieval baselines

The baseline runner supports BM25, exact dense search and min-max hybrid fusion. It only accepts `train` and `val`; test/human evaluation remains locked until H5. Every completed run contains the six files in the development guide, including per-query metrics, plans, counters and reproducibility metadata.

```powershell
& $conda run --no-capture-output --prefix $prefix python -m slotspace.experiments --mode bm25 --limit 20
& $conda run --no-capture-output --prefix $prefix python -m slotspace.experiments --mode bm25
```

For a GPU host, the dense/hybrid commands below prepare both 1024- and 256-dimensional topic caches in **one** encoding pass. The local model directory is optional; the default is the documented Hub model. Four lexical weights can be compared on validation MRR without touching test queries.

```powershell
& $conda run --no-capture-output --prefix $prefix python -m slotspace.experiments --mode dense --model ..\Resource\models\Qwen\Qwen3-Embedding-0.6B
& $conda run --no-capture-output --prefix $prefix python -m slotspace.experiments --mode hybrid --model ..\Resource\models\Qwen\Qwen3-Embedding-0.6B --lambda-L 0 0.25 0.5 1
```

Caches are immutable and fingerprint the input corpus and encoder/index settings; mismatched settings fail explicitly. Use a separate `--data` directory for a small CPU smoke run. This implementation uses a full BM25 score vector for fusion, not a top-1000 approximation. Dense/lexical scores always map through the exported document ID column.

### CPU validation — 2026-11-24

BM25 on all **2,241 validation queries**: Hit@1 **17.80**, Hit@5 **35.48**, Recall@20 **41.04**, Recall@100 **52.59**, MRR **25.97**, R-precision **17.05** (percentage points). The largest discrepancy from the official evaluator was below `1e-6` points. Query timing is the median of three warm repeats; quality values are not test results.

Full-corpus dense/hybrid GPU measurements and fusion-weight selection have **not** been run, as requested. No GPU throughput or final retrieval-quality claim is implied by CPU/unit checks.

### H0 acceptance — full CPU run, 2026-11-24

| Check | Observed | Result |
|---|---:|---|
| Candidate documents | 129,375 | Matches M0 |
| Stored edges / distinct relation names | 8,100,498 / 18 | Exact names from design §5.3 |
| Synthesized train / val / test | 6,162 / 2,241 / 2,801 | Matches official splits |
| `test-0.1` / human | 280 / 98 | Subset of test / separate table |
| Dangling answer IDs / missing reverse triples | 0 / 0 | All answers and all edges checked |
| Documents rendered identically to upstream | 100 / 100 | Seeded sample, byte-for-byte |

| Document type | Count | Document type | Count |
|---|---:|---|---:|
| disease | 17,080 | pathway | 2,516 |
| gene_protein | 27,671 | anatomy | 14,035 |
| molecular_function | 11,169 | effect_phenotype | 15,311 |
| drug | 7,957 | biological_process | 28,642 |
| cellular_component | 4,176 | exposure | 818 |

## Data and licenses

Code: [Apache-2.0](LICENSE). No benchmark records or weights are committed. STaRK data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): Shirley Wu et al., [*STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases*](https://arxiv.org/abs/2404.13207), 2024; [upstream data and attribution](https://huggingface.co/datasets/snap-stanford/stark). The export normalizes document types and detail fields and preserves original IDs, relations, answers and split membership.
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

The reference scorer, its tests, fast scorer, simulation and statistics helpers are the unmodified appendices from the guide. This milestone does not train models or implement the retrieval baselines.

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
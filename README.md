# SlotSpace

SlotSpace retrieves facts, not just documents that mention the same entity. A value, its role and its context must match **in the same fact**. Several plausible readings of a query share one candidate retrieval pass. See the [design](docs/DESIGN.md) and [milestone guide](docs/DEVELOPMENT.md).

**Current checkpoint: H1 human review.** M0, the M1 CPU checks and the M2 preprocessing checks have run; GPU/full dense experiments are explicitly deferred. The full CPU suite passes 41 tests.

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

For a GPU host, the dense/hybrid commands below prepare both 1024- and 256-dimensional topic caches in **one** encoding pass. Run these with a separate CUDA-enabled Python environment; the prepared local Conda prefix is CPU-only. The local model directory is optional; the default is the documented Hub model. Four lexical weights can be compared on validation MRR without touching test queries.

```powershell
python -m slotspace.experiments --mode dense --model ..\Resource\models\Qwen\Qwen3-Embedding-0.6B
python -m slotspace.experiments --mode hybrid --model ..\Resource\models\Qwen\Qwen3-Embedding-0.6B --lambda-L 0 0.25 0.5 1
```

Caches are immutable and fingerprint the input corpus and encoder/index settings; mismatched settings fail explicitly. Use a separate `--data` directory for a small CPU smoke run. This implementation uses a full BM25 score vector for fusion, not a top-1000 approximation. Dense/lexical scores always map through the exported document ID column.

### CPU validation — 2026-11-24

BM25 on all **2,241 validation queries**: Hit@1 **17.80**, Hit@5 **35.48**, Recall@20 **41.04**, Recall@100 **52.59**, MRR **25.97**, R-precision **17.05** (percentage points), median warm latency **2.06 ms** on this shared CPU host. The largest discrepancy from the official evaluator was below `1e-6` points. Query timing is the median of three warm repeats; quality values are not test results.

Full-corpus dense/hybrid GPU measurements and fusion-weight selection have **not** been run, as requested. No GPU throughput or final retrieval-quality claim is implied by CPU/unit checks.

## M0 / H0 acceptance — full CPU run, 2026-11-24

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

## M2: shared schema, entity linking and document facts

[The Prime schema](schemas/prime.yaml) contains the **56 observed typed slots**, English descriptions, query/document cues and ten `none` outcomes. Its inventory and per-slot counts are checked against the real graph, including reciprocity under each exact relation name.

```powershell
& $conda run --no-capture-output --prefix $prefix python -m slotspace.schema
& $conda run --no-capture-output --prefix $prefix python -m slotspace.facts --out data\prime\m2-sample-100 --runs runs\m2-smoke --limit-docs 100 --limit-queries 20
& $conda run --no-capture-output --prefix $prefix python -m slotspace.facts
```

The full run writes `native_keys.parquet`, `text_facts.parquet`, input fingerprints and statistics under `data/prime/facts_prime-0.1.0/`. It does not duplicate relation listings as text facts. Same-type ambiguous mentions are excluded as values but **remain masked**, and a cross-type span has one Parquet record per type sharing the same masked scorer input. Later scorers must group those records by span before computing a joint distribution.

Review locations are recorded in `runs/m2_stats.json`: each M2 run produces `query_mentions.jsonl` and `document_mentions.jsonl`, up to 100 seeded examples each. Their `correct` fields start as `null`; linked-query coverage is not a measurement of linking precision.

The full CPU preprocessing run produced **8,100,498 native keys** (no duplicate triples) and **287,492 text facts**. Of 2,241 validation queries, **99.87%** have at least one linked mention. All 287,492 fact rows passed offset/type/row-alignment checks, and native-key sets for 100 seeded documents matched their raw graph edges exactly. No document reached the 256-fact cap; 472 same-type-ambiguous mentions were skipped.

**H1 is a real human-review gate:** check those two samples and the schema descriptions, mark each example `correct: true/false`, and confirm linking precision of at least 85% before M3. No human labels are generated or assumed by the code.

The context-embedding path is implemented and tested with small fixtures; full GPU encoding is deferred. On the GPU host:

```powershell
python -m slotspace.embed --kind context --docs data\prime\facts_prime-0.1.0\text_facts.parquet --cache data\prime\ctx_prime-0.1.0_qwen3e06 --model ..\Resource\models\Qwen\Qwen3-Embedding-0.6B
```

This writes row-aligned, unit-normalized float16 128-dimensional fact vectors, without a query prompt or a target marker. It does not assign slot probabilities; that is M3/M5 work.

## Data and licenses

Code: [Apache-2.0](LICENSE). No benchmark records or weights are committed. STaRK data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): Shirley Wu et al., [*STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases*](https://arxiv.org/abs/2404.13207), 2024; [upstream data and attribution](https://huggingface.co/datasets/snap-stanford/stark). The export normalizes document types and detail fields and preserves original IDs, relations, answers and split membership.
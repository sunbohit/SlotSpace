# SlotSpace

SlotSpace retrieves facts, not just documents that mention the same entity. A value, its role and its context must match **in the same fact**. Several plausible readings of a query share one candidate retrieval pass. See the [design](docs/DESIGN.md) and [milestone guide](docs/DEVELOPMENT.md).

**Current implementation:** M0-M6 and the M7 search core are implemented; the full CPU suite passes **69 tests**. Human reviews remain pending, but do not prevent continued code work at the user's request. GPU jobs, full M7 tuning and final test experiments have not been run.

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

**H1 remains unapproved:** check those two samples and the schema descriptions and mark each example `correct: true/false`. The requested precision target is at least 85%. Later implementation and provisional data preparation have continued without falsely recording human approval.

The context-embedding path is implemented and tested with small fixtures; full GPU encoding is deferred. On the GPU host:

```powershell
python -m slotspace.embed --kind context --docs data\prime\facts_prime-0.1.0\text_facts.parquet --cache data\prime\ctx_prime-0.1.0_qwen3e06 --model ..\Resource\models\Qwen\Qwen3-Embedding-0.6B
```

This writes row-aligned, unit-normalized float16 128-dimensional fact vectors, without a query prompt or a target marker. It does not assign slot probabilities; that is M3/M5 work.

## M3-M4: corpus evidence and evaluation data

- The complete [native lexicon](data/prime/lexicon_native/) has **293,477 value/slot pairs**, with hard and soft totals of 8,100,498. It is explicitly native-only; it is not substituted for the full text-aware lexicon in the search pipeline.
- The synthetic binding experiment reproduces the guide's default R-precision values: bound 0.835, unbound 0.483 and gate-only 0.436. The 1,000-document delete/reinsert test and float/binary scorer parity tests pass.
- [Silver labels](data/prime/silver_prime-0.1.0/) cover 6,162 train queries and 2,241 validation queries. Even train qids give **3,102 calibration queries**; odd qids give **3,060 spare queries**. No silver test labels are built. The share of segments with a nonempty native label is 18.20% on train and 18.53% on validation, **not** a human agreement rate.
- [Role-swap data](data/prime/role_swap_prime-0.1.0/) has **737 dev / 667 test queries**, with value-disjoint, same-split partner groups. The prescribed fallback reduced minimum support from 3 to 2. The 1,000-test-query target was **not reached**; MDE is 4.85 points under the stated discordance assumption of 0.2.
- Review paths and counts are in [silver statistics](runs/m4_silver_stats.json) and [role-swap statistics](runs/m4_role_swap_stats.json). H2 and H3 samples are unlabelled; they have not been declared passed.

On a fresh workspace, reproduce the CPU data preparation in order:

```powershell
python -m slotspace.lexicon_build --native-only --out data\prime\lexicon_native
python -m slotspace.silver
python -m slotspace.role_swap
```

Existing artifacts are immutable. Repeated/smoke runs require new `--out` and `--runs` directories. Native support is aggregated per document before corpus summation. The combined lexicon implements PACC, including mentions with zero stored role probability, while retaining raw hard counts for lossless pruning. Missing/unstable TPR/FPR rates produce explicit warnings and recorded raw-count fallbacks.

## M5: offline generation, training and calibration

[Training templates](templates/train/prime_slots.yaml) are independent of test templates: 56 slots plus 10 `none` classes, twelve frames per style, corpus-derived minimal-pair roles, surrogate values and secondary mentions. The CPU smoke prepared **1,584 balanced masked examples** and verified template-disjoint train/dev splits. It ran **no paraphraser and no model training**.

The following are commands for a CUDA-capable research environment, not the prepared CPU-only Conda environment. Replace angle-bracket arguments with local paths. The vLLM command must run in its **separate Linux environment**, with this package installed using `pip install --no-deps -e .`.

```powershell
python -m slotspace.gen.fill_templates --out data\prime\gen_main_filled
```

In the Linux generation environment, use the matching filled file and local Qwen3-4B snapshot:

```text
python -m slotspace.gen.paraphrase_vllm --input <filled.jsonl> --out <new-paraphrase-directory> --model <Qwen3-4B-directory>
```

Back in the CUDA research environment:

```powershell
python -m slotspace.gen.validate --input <paraphrases.jsonl> --out data\prime\gen_main_validated
python -m slotspace.train_slot_scorer --data data\prime\gen_main_validated\validated.jsonl --out data\prime\models\slot_scorer_ft --model <ModernBERT-directory>
python -m slotspace.calibrate_slot_scorer --model data\prime\models\slot_scorer_ft\final --out data\prime\models\slot_scorer_ft\temperature.json
```

`--variant a14` excludes the held-out slot from both positive **and negative** training pairs. For A12, fill a separate source with `--a12-docs data\prime\docs.parquet`, validate it separately and train with `--variant a12`. Main/A14 and A12 sources cannot be interchanged. Query temperature uses only even-qid train silver; document temperature uses KB-connected text mentions. Validation is never used to fit temperatures.

Document scoring and combined lexicon construction are implemented for both `zeroshot` and `ft`. For FT, pass the saved document temperature:

```powershell
$t = Get-Content data\prime\models\slot_scorer_ft\temperature.json -Raw | ConvertFrom-Json
python -m slotspace.slot_scorer --kind ft --model data\prime\models\slot_scorer_ft\final --temperature $t.T_doc --out data\prime\scores_ft
python -m slotspace.lexicon_build --text-keys data\prime\scores_ft\text_fact_keys_ft.parquet --out data\prime\lexicon_ft --runs runs\m3-ft
```

For the bootstrap zero-shot path use `--kind zeroshot` and the BGE reranker, with slot text first. Full document scoring, combined-corpus PACC measurements, vLLM paraphrasing, main/A12/A14 training and empirical calibration targets remain **unexecuted**.

## M6 and M7 core: planning and single-pass search

The planner preserves one joint cross-type distribution, masks every linked mention, removes unsupported readings, applies conformal sets and scales each segment's weights by its non-`none` mass. Its worked posterior matches 0.6254 / 0.2014 / 0.1732. Plans serialize to JSON and preserve index/lexicon versions.

After the GPU scoring/embedding prerequisites above:

```powershell
python -m slotspace.planner --lexicon data\prime\lexicon_ft --model data\prime\models\slot_scorer_ft\final --temperatures data\prime\models\slot_scorer_ft\temperature.json --out data\prime\conformal_ft.json
python -m slotspace.embed --model <Qwen3-Embedding-directory>
python -m slotspace.embed --kind context --model <Qwen3-Embedding-directory>
python -m slotspace.pipeline --text-keys data\prime\scores_ft\text_fact_keys_ft.parquet --context-cache data\prime\ctx_prime-0.1.0_qwen3e06 --lexicon data\prime\lexicon_ft --model data\prime\models\slot_scorer_ft\final --embedding-model <Qwen3-Embedding-directory> --temperatures data\prime\models\slot_scorer_ft\temperature.json --conformal data\prime\conformal_ft.json --limit 20
```

The search core runs BM25 and dense search once, unions their candidates with capped value postings, uses the fast fact scorer, supports `p`, `p_over_H` and IA selection, returns evidence, and records stage/cost counters. Its CPU tests cover the complete link-plan-retrieve-bind chain, version failures, IA output ordering and exact hybrid behavior with binding disabled. Artifact-loading tests check the actual Parquet/manifest interchange, not only mocked search results.

**Remaining:** the full M7 reusable tuning-feature cache/grid search, configuration freezing and M8 baseline/ablation/report orchestration. No tuned configuration, formal test results, empirical conformal-coverage claim or M5 quality-target pass is fabricated. Both runners still reject test/human splits until the later H5-controlled experiment path is implemented.

## Data and licenses

Code: [Apache-2.0](LICENSE). No benchmark records or weights are committed. STaRK data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): Shirley Wu et al., [*STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases*](https://arxiv.org/abs/2404.13207), 2024; [upstream data and attribution](https://huggingface.co/datasets/snap-stanford/stark). The export normalizes document types and detail fields and preserves original IDs, relations, answers and split membership.
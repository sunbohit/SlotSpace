# SlotSpace

For a new GPU machine, start with [the clone-to-GPU setup guide](docs/GPU_SETUP.md).
Portable environment files, input revisions/checksums and preliminary data
summaries are included. Raw benchmark records and multi-GB weights are fetched
from their original publishers, not stored in Git.

Queries can mention the same value while asking for different roles: drugs that treat a disease, drugs contraindicated for it, or drugs that cause a similarly named effect. Matching the value or the overall topic alone does not establish which fact answers the question.

SlotSpace shares one schema-defined slot space across fact extraction, query interpretation, a corpus-derived ambiguity lexicon and retrieval. Value, role and context must match **in the same fact**. A small masked-context model supplies query evidence, corpus counts supply priors, and several plausible readings share one candidate retrieval pass.

The initial research path is Python/NumPy on STaRK-Prime, with a stock-Vespa adapter for later engine measurements. The claim is the combination, not novelty of every component; experiments must establish its value. See the [design](docs/DESIGN.md), [development guide](docs/DEVELOPMENT.md) and [citation](CITATION.cff).

**Current status:** 0.1.0 is an **unreleased code candidate** covering M0-M10 implementation and release preparation. It has not been uploaded to a registry. GPU jobs, human semantic reviews, empirical M7 acceptance, formal test experiments and real-engine M9 acceptance remain pending. Code review is not an H5 authorization receipt.

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

The standalone baseline/search runners remain restricted to train/validation. Only the H5-controlled M8 runner below can evaluate formal test sets. Empty hard-AND results have zero retrieval quality, an explicitly logged extension because the official evaluator cannot represent empty rankings correctly.

## M7: reusable features, staged tuning and freezing

Edit [the artifact bundle](configs/experiment_artifacts.example.yaml) to match the completed GPU artifacts. Use the same encoder identifier or local snapshot for topic, fact-context and baseline-representation caches. The following capture commands require those artifacts and a suitable research environment; they have **not** been run on the full corpus here.

```powershell
python -m slotspace.tuning cache --split calibration --artifacts configs\experiment_artifacts.example.yaml --out data\prime\tuning_calibration
python -m slotspace.tuning cache --split val --artifacts configs\experiment_artifacts.example.yaml --out data\prime\tuning_val
python -m slotspace.tuning fit --calibration data\prime\tuning_calibration --validation data\prime\tuning_val --out runs\m7_tuning
```

- Feature capture uses even-qid train silver for calibration and validation silver for selection, never test labels. `--limit` is for smoke checks; partial caches cannot pass production freezing.
- Candidate snapshots retain the **uncapped** union of all supported readings' postings. Exact float64 match matrices cover the configured `(a, b)` grid, so changing conformal sets can reconstruct a different top-`K_V` without rerunning neural inference.
- Every beta gets its own calibration bank, including matched raw-support and uniform-query conditions. Tuning follows beta/alpha by regret, a/b by MRR, fusion by MRR, then weight objective by MRR/Recall@100; B2's lexical weight is selected separately.
- Replay logs `replay_ms`, not online latency: `latency_ms` remains null. Checksums, geometry fingerprints and immutable matrix directories prevent incompatible cache reuse.

Additional scoring modes or binary matrices can be prepared without neural inference; repeat for each feature set and required grid:

```powershell
python -m slotspace.tuning augment --cache data\prime\tuning_val --mode unbound --a 0.5 1 --b 0.5 1 2
python -m slotspace.tuning augment --cache data\prime\tuning_val --precision binary --a 0.5 1 --b 0.5 1 2
```

Only complete, accepted production evidence can produce a runnable frozen configuration. Acceptance checks the Recall@100 floor against B2 and conformal miss rate; production freezing rechecks the captured data/model/schema fingerprints. After real acceptance:

```powershell
python -m slotspace.tuning freeze --selection runs\m7_tuning\selection.json --out configs\prime_v0_frozen.yaml
```

The accompanying `.yaml.freeze.json` receipt records **H5 pending**, not human approval. No production frozen configuration has been generated in this workspace.

### Baseline and ablation artifact preparation

B3 sentence MaxP and B4 fields use **256-dimensional unmasked topic embeddings**, not the 128-dimensional fact-context vectors. B4 has name, details and one relation-listing field per relation; weights use complete-validation coordinate ascent with the selected B2 lexical weight.

```powershell
python -m slotspace.representations --kind passages --model ..\Resource\models\Qwen\Qwen3-Embedding-0.6B --out data\prime\maxp_qwen3e06_256
python -m slotspace.representations --kind fields --model ..\Resource\models\Qwen\Qwen3-Embedding-0.6B --out data\prime\fields_qwen3e06_256
python -m slotspace.tuning fields --config configs\prime_v0_frozen.yaml --validation data\prime\tuning_val --representations data\prime\fields_qwen3e06_256 --out data\prime\field_weights.json
```

A7 requires a separate complete fact/context/score/lexicon/conformal artifact chain. Its windows use actual tokenizer offsets, 128 tokens and stride 128, retaining the normal masking and fact caps:

```powershell
python -m slotspace.facts --granularity chunk128 --tokenizer ..\Resource\models\Qwen\Qwen3-Embedding-0.6B --out data\prime\facts_chunk128-0.1.0 --runs runs\m2-chunk128
```

A11 can pack the existing float cache on CPU:

```powershell
python -m slotspace.embed --kind binary --source-cache data\prime\ctx_prime-0.1.0_qwen3e06 --cache data\prime\ctx_binary_prime-0.1.0_qwen3e06
```

Packed contexts are signed int8, big-endian, with exact zero mapped to bit zero. A11 storage reports actual serving-index input bytes (native/text facts, text keys, packed contexts, lexicon, manifests and schema) per document, not a theoretical vector-only estimate or a RAM measurement; shared topic caches and model weights are excluded.

For A12, add `--values-visible` to both document scoring and conformal fitting (`slotspace.slot_scorer` and `slotspace.planner`). Temperature calibration detects the A12 training manifest. Only the target is revealed to the slot scorer; query segment and document context embeddings remain masked. The bundle's visibility flags, model, temperature and document-key metadata must agree.

## M7/M8: progressive pre-experiment pilots

The production M7/M8 implementation is already present. The
[pilot schedule](configs/pilot_stages.yaml) adds a separate, incremental route
for exercising it before expensive/full experiments:

| Stage | Candidate documents | Even-train calibration queries | Validation queries | Fact cap/doc | Topic token cap | Measured repeats |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 64 | 4 | 4 | 1 | 96 | 1 |
| small | 256 | 16 | 16 | 4 | 256 | 3 |
| medium | 1,024 | 32 | 32 | 8 | 512 | 3 |

```powershell
python -m slotspace.experiments pilot prepare --stage smoke --out runs\pilot_smoke_01
python -m slotspace.experiments pilot run --bundle runs\pilot_smoke_01 --device cpu --threads 4
```

These pilots use real cached Qwen embeddings and a BGE zero-shot slot scorer,
not fabricated vectors or a supposedly fine-tuned main model. They retain the
**full KB dictionary**, all selected queries' answer sets and all native keys
for the selected documents. Text-fact sampling is explicitly capped. The
answer-closed candidate sampling is a diagnostic convenience, not a
representative full-benchmark protocol.

M7 captures model observations once, stores exact match matrices, recalibrates
reading sets using even-train silver only and runs the existing staged tuning.
M8 measures BM25/dense/hybrid/bound retrieval, validates live/replay parity,
logs warm-up/repeats, per-query metrics, evidence and actual latency/counters.
`--methods bm25 dense hybrid bound unbound fanout` enables additional comparisons.
The smoke's single measured repeat is a smoke timing, not robust performance evidence.

Existing input/model/feature caches are verified with `--resume`; completed
pilots are never implicitly rerun. A report regenerates from its saved files
without loading either model:

```powershell
python -m slotspace.experiments pilot report --run runs\pilot_smoke_01\runs\<pilot-run> --out runs\pilot_report_copy.md
python -m slotspace.experiments pilot prepare --stage small --previous runs\pilot_smoke_01 --out runs\pilot_small_01
```

Expansion requires the immediately preceding stage's completed engineering
checks and unchanged artifact/source/model identities. Cohorts and candidates
are nested; the corpus/fact/token budgets may grow. The small/medium stages
are not automatically launched and need resources appropriate to their size.

Every result is tagged `pilot`, with full-corpus completeness false and
production acceptance false. Pilot selection cannot freeze a production YAML
or issue H5 receipts. At 64 candidates Recall@100 is necessarily a ceiling.
Validation is reused for selection and measurement, so paired statistics are
descriptive, not confirmatory. Temperatures are fixed at one, PACC is disabled,
and silver labels remain human-unapproved. Final test/human queries are not
parsed or evaluated by this route.

### First measured CPU pilot - 2026-11-25

The successful run is under `runs/pilot_smoke_02/`, measured from clean code
commit `cf43e84`. It contains 64 candidates, 56,573 native keys and **52 of
450** available text facts in those documents, with four even-train queries
and four validation queries. The following percentages are diagnostic
validation results, selected/tuned on the same tiny cohort:

| Method | Hit@1 | MRR | Median client latency |
|---|---:|---:|---:|
| BM25 | 50.00 | 64.77 | 0.24 ms |
| Dense | 25.00 | 41.37 | 260.57 ms |
| Hybrid | 50.00 | 75.00 | 251.37 ms |
| Bound zero-shot prototype | 100.00 | 100.00 | 3,151.54 ms |

All four live rankings/scores exactly matched their feature replays, and the
report regenerated byte-for-byte from saved artifacts. There are 15 staged
tuning trials; the selected pilot uses beta 0.5, alpha 0.2, a=b=1,
lambda_B=lambda_D=1, lambda_L=0 and `p` weighting.

**Negative/limited evidence:** bound was about **12.5 times slower** than
hybrid on this shared CPU smoke; no efficiency improvement is demonstrated.
Uncalibrated query ECE is **0.5824**, not the 0.05 acceptance target.
Observed regret/miss rate are zero on only four validation queries/five
labelled segments. Recall@100 is 100% for all methods because the 64-candidate
corpus contains every answer. These results do not establish M7 acceptance
or a formal M8 claim.

The initial loader failure is retained under `pilot_smoke_01`; it was corrected
with a prepared-artifact regression test, not a fallback. Final tests pass
**120 CPU cases**, with one real-Vespa test skipped.

The next stage is prepared at `runs/pilot_small_pending/`: 256 candidates,
16 even-train queries, 16 validation queries and 626 sampled text facts.
It has **not** run model inference or experiments:

```powershell
python -m slotspace.experiments pilot run --bundle runs\pilot_small_pending --device cpu --threads 4
```

Expect more CPU work, or use a suitably prepared CUDA research environment.
The 1,024-candidate stage remains unprepared/unexecuted.

## M8: experiment protocol, execution and report

The catalog includes B0-B7, a 1024-dimensional B1 condition, and A0-A14 with expanded objective/alpha conditions. B7's oracle is validation-only. B6 runs one hybrid retrieval per kept reading and sums actual work counters before weighted RRF. Identical configurations alias prior jobs rather than silently rerunning a final evaluation.

The following **draft-only** workflow runs without GPU artifacts or formal evaluation. It records missing prerequisites and produces NOT RUN tables and placeholder figures, never synthetic benchmark scores:

```powershell
python -m slotspace.experiments plan --draft --config configs\prime_v0.yaml --variants configs\experiment_artifacts.example.yaml --auxiliary configs\experiment_artifacts.example.yaml --out runs\m8_protocol_draft.json
python -m slotspace.experiments report --plan runs\m8_protocol_draft.json --out runs\report_v0.md
```

For real experiments, first complete artifact preparation and acceptance, then commit the code, frozen YAML and freeze receipt. Keep the final plan and externally supplied approval under ignored `runs/` so creating them does not dirty the committed implementation:

```powershell
python -m slotspace.experiments plan --config configs\prime_v0_frozen.yaml --variants configs\experiment_artifacts.example.yaml --auxiliary configs\experiment_artifacts.example.yaml --out runs\m8_protocol.json
```

Only after human H5 approval, an external approval JSON must bind both exact hashes:

```json
{"approved": true, "frozen_sha256": "<frozen YAML SHA-256>", "plan_sha256": "<plan JSON SHA-256>"}
```

```powershell
python -m slotspace.experiments run --plan runs\m8_protocol.json --approval runs\h5_approval.json --out runs\m8_execution
python -m slotspace.experiments report --plan runs\m8_protocol.json --execution runs\m8_execution --out runs\report_v0_final.md
```

Draft plans are never executable. Authorization checks approval before parsing formal queries, verifies the clean committed implementation and rechecks artifact/dataset hashes. Each execution root is immutable: failures are recorded as failed, not silently retried. Quality uses the first measured ranking after warm-up; latency is the median of three measured repeats, with ranking instability recorded.

Reports regenerate the four main tables, separate human table, paired tests, table-family Holm correction, bootstrap intervals, observed discordance/MDE, pass lines and SVG figures from saved artifacts alone. PACC diagnostics fit rates on even document IDs and assess count error on odd IDs; query/document reliability points are copied into the execution bundle. Missing jobs stay NOT RUN, query-ID mismatches fail explicitly, and fixture runs are excluded by default. Zero-effect aliases do not become positive findings.

**Still pending:** actual GPU preparation/training, empirical calibration/tuning, H1-H5 approval and formal test execution. The 667-query role-swap test split still falls below the original 1,000-query target. Implemented code and CPU fixtures do not establish v0 experimental acceptance.

## M9: optional stock-Vespa engine

The [engine adapter](src/slotspace/vespa_app/) packages the same bound/max score in stock Vespa: native keys, fact-level sparse role probabilities and **raw big-endian signed int8** contexts. Its generated application has six `(a,b)` profiles for `a in {0.5,1}`, `b in {0.5,1,2}` plus `hybrid`. Query links are raised to `a` exactly once; the document gate raises role probabilities separately. Disjoint native-only and text-only readings are retained using `N + T - min(N,T)`.

**M9 handoff:** The CPU suite passed **110 tests**, with **one real-engine test explicitly skipped**. This includes wire-format/reference parity, actual Parquet-to-feed export without loading models, HTTP/error handling, fan-out and measurement artifacts. The distribution wheel includes the XML and schema resources. No Docker/Podman or Vespa server is available on this Windows host, so **application deployment/schema compilation, real-engine parity, feed throughput and engine latency/RAM are NOT RUN**, not passed. Full neural artifacts are also absent under the GPU-testing waiver.

On a Docker-capable host, use a **new, dedicated, disposable** instance, with at least 4 GiB Docker memory for smoke testing (16 GiB for Prime). Pin the image tag/digest you intend to use instead of treating `latest` as reproducible:

```powershell
docker run --detach --name slotspace-vespa --hostname vespa-container --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 vespaengine/vespa:<pinned-tag>
python -m slotspace.vespa_app smoke --deploy --allow-replace-application
```

The smoke deploys the application, feeds the four documented toy documents with seeded 128-dimensional contexts, and requires **every generated exponent profile** to match `reference.binding_score(..., precision="binary")` within absolute `1e-3`, with no non-tie order inversions. It also checks the no-readings hybrid path and removes only its four newly fed IDs. It refuses to feed over existing documents. An equivalent opt-in pytest is available:

```powershell
$env:SLOTSPACE_VESPA_TEST_ENDPOINT = "http://localhost:8080"
$env:SLOTSPACE_VESPA_DEPLOY_ENDPOINT = "http://localhost:19071"
$env:SLOTSPACE_VESPA_ALLOW_REPLACE = "1"
python -m pytest -q -m vespa tests\test_vespa_smoke.py
```

Deployment **replaces the application's configuration**; never point these commands at a production/shared application. No command starts Docker, installs a system runtime, deletes an entire corpus or disables TLS certificate checks. After use, stop only the dedicated container you created.

Once full research artifacts and a production configuration with its `artifacts` mapping exist (for example the M7 frozen YAML), start with that empty dedicated instance:

```powershell
python -m slotspace.vespa_app export --config configs\prime_v0_frozen.yaml --out data\prime\vespa_binary
python -m slotspace.vespa_app feed --bundle data\prime\vespa_binary --deploy --allow-replace-application
python -m slotspace.vespa_app measure --config configs\prime_v0_frozen.yaml --bundle data\prime\vespa_binary --feed-receipt runs\<m9-feed-run>\feed.json --container slotspace-vespa
```

- Export does not load neural models. It verifies the complete combined index/lexicon, topic/context encoders and schema, then writes an immutable, checksummed `feed.jsonl` and `app/`. Feeding uses the Document API with bounded concurrency, records server version, HTTP acknowledgement time and total time until all documents are searchable, and requires an empty app. A failed partial feed is not silently resumed or erased.
- The measurement runner uses **validation only**, sorted by qid: the first 200 queries for all-returned-hit parity, the first 1,000 for quality and timing (smaller explicit counts are smoke measurements). Engine candidates are expected to differ from the numpy path because HNSW/weakAnd/first-phase select the pool.
- It compares engine SlotSpace, engine B2, engine B6 fan-out and numpy binary SlotSpace. Fusion uses each returned pool's exact match features; B2/B6 share B2's selected lexical weight. B6 counts every rewrite and saves both sum/max server and RPC time. One warm-up precedes three repeats; quality uses the first measured ranking, timing uses medians, and instability is recorded.
- Output contains run/plan/per-query/counter/metric files and `parity.json`, with p50/p95 client and server latency. `docker stats` is labelled **whole-container memory**, never isolated index RAM; omitting `--container` records `NOT MEASURED`. HTTP errors, HTTP-200 query errors, degraded coverage, missing features, stale artifacts, parity errors and operational deployment actions fail explicitly and leave failed run metadata.
- This path supports the documented bound/max binary score and 256/128 dimensions, not IA, hard-AND or the v0 ablation matrix. It never authorizes H5 or evaluates final test/human splits.

## M10: reproduction and local distribution candidate

Use Python 3.11 and the compatible environment described in the development
guide. In a fresh clone with an appropriate PyTorch build already chosen:

```powershell
python -m pip install -e ".[test,release]"
python -m slotspace.reproduce --resources ..\Resource --dry-run
```

The cached `Resource` layout is the one documented during preparation: pinned
STaRK input files and four v0 model snapshots. The dry run prints **real existing
CLI commands for M0-M8**, not benchmark scores. A complete run needs a CUDA
research environment and a **separate Linux vLLM interpreter**:

```powershell
make reproduce-v0 PYTHON=python REPRO_ARGS="--gen-python <absolute-generator-python>"
```

On this CPU-only Windows workspace, GNU Make is isolated in the release-tools
prefix; the tested CPU-only subset is:

```powershell
$make = (Resolve-Path ..\.envs\release-tools\Library\bin\make.exe).Path
& $make reproduce-v0 "PYTHON=$prefix\python.exe" "THROUGH=M0"
```

An equivalent command needs no Make: `python -m slotspace.reproduce --through M0`.
The actual fresh M0 run and a verified Make-driven resume produced seven files
**byte-identical** to the original full Prime export. Full GPU M0-M8 execution
has **not** been run here.

- Use a new `--work` directory (default `runs/reproduce-v0`). Cached downloads
  are read-only; output configurations/data/runs are isolated there.
- `--resume` verifies the same source, interpreter/environment and settings,
  plus exact hashes of completed outputs/model inputs. It never adopts an
  unrelated existing cache or silently retries a partial final experiment.
- H1 requires labelled query/document samples and schema approval; H2 checks
  silver agreement, H3 naturalness, H4 measured post-scaling ECE and labelled
  scorer examples. The runner stops at these gates. Fill the generated review
  samples and supply `--reviews` with the gate's printed evidence hash:
  `{"H1": {"approved": true, "evidence_sha256": "<printed hash>"}}`.
  Later approvals can be added to the same JSON.
- Continue with unchanged settings, including the same `--gen-python`, using
  `--resume --reviews <review-json>`. The M8 protocol is produced before H5;
  only an externally supplied `--approval <h5-json>` bound to its exact
  frozen/config-plan hashes permits final evaluation. No approval is generated.
- Main/A14 training remains value-masked, A12 is separate, variant conformal
  calibration uses the selected configuration, and B3/B4/A7/A11 artifacts
  are prepared for the existing full experiment catalog.

### Packaging without publishing

```powershell
python -m build --no-isolation
python -m twine check --strict dist\*
python tools\check_dist.py --dist dist
```

`make dist` performs the same build/metadata/payload checks. The source archive
contains configs, schemas, templates, tests, normative docs and release records.
The wheel contains Python/Vespa code, SPDX/NOTICE files and research assets
under `share/slotspace/`. Neither contains benchmark records/answers, weights
or run outputs. These commands **do not publish** to PyPI or GitHub.

**M10 validation:** 116 CPU tests passed and one real-Vespa test was explicitly
skipped. Citation metadata passed the official CFF 1.2.0 schema. Both actual
archives passed strict Twine and payload checks. A wheel rebuilt from the
source archive was installed outside the checkout: 22 installed-code
reference/CLI tests passed, and its complete M0 export matched all seven
original files byte-for-byte using the prepared dependency environment.
This is not a full GPU reproduction or empirical release acceptance.

The [model card](docs/MODEL_CARD.md) explicitly records an absent fine-tuned
checkpoint and unmeasured M5 metrics. [Third-party findings](THIRD_PARTY_NOTICES.md)
include the exact local/generator/model inventories and GPL/proprietary review
items; a metadata check is not a license-compatibility approval.

### Measured results versus pending acceptance

| System | Dataset | Hit@1 | Recall@20 | MRR | Evidence |
|---|---|---:|---:|---:|---|
| B0 BM25 | Prime val, 2,241 queries | 17.80 | 41.04 | 25.97 | Measured CPU, percentage points |
| Full B1/B2 neural baselines | Prime val | - | - | - | NOT RUN |
| SlotSpace / formal ablations | Prime test/human/role-swap | - | - | - | NOT RUN; no H5 execution |
| Vespa path | Prime val | - | - | - | NOT RUN; no engine |

The prototype requires dictionary-linkable values, does not cover temporal
ambiguity, and combines context/corpus evidence with a fixed functional form.
Conformal coverage is marginal, not a per-query promise. Synthesized query
style, noisy silver labels, missing KB edges and linking mistakes can bias
results. Strong baselines may already capture roles; negative findings must
remain visible. The role-swap test has 667 queries, below the 1,000 target,
and the 98 human queries supply directional evidence only.

## Data and licenses

Code: [Apache-2.0](LICENSE). No benchmark records or weights are committed. STaRK data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): Shirley Wu et al., [*STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases*](https://arxiv.org/abs/2404.13207), 2024; [upstream data and attribution](https://huggingface.co/datasets/snap-stanford/stark). The export normalizes document types and detail fields and preserves original IDs, relations, answers and split membership.
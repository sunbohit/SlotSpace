# Implementation decisions

## 2026-11-24 — M0

- Pin the Prime inputs to STaRK Hub revision `88269e23e90587f99476c5dd74e235a0877e69be`. Verify the six processed files before the official loader can deserialize them. Existing input caches are read-only; fetching is explicit. The tested Windows CPU environment is recorded in `requirements.lock`; it is not a Linux CUDA lock.
- Use `load_skb("prime", ...)` and the upstream `get_doc_info(add_rel=True, compact=True)` formatter. Index the sorted, de-duplicated neighbours of each relation by source-node offset on a local view of the SKB. Selecting every row from an unindexed sparse COO tensor repeatedly scans the graph; indexing changes the lookup cost, not the order, text or exported edge multiplicity.
- The 1.1.0 release of `load_qa` lists the Hub even with a local root, and constructs its remote prefix with the OS path separator. Read its exact CSVs and split index files directly instead. Index files contain **query IDs**; they are not CSV row positions. Parse integer answer arrays as JSON, ignore any other upstream columns, preserve split-file order and emit the human table in ID order. No train/test data is mixed.
- Unlike the illustrative `--limit` skeleton, sample exports have no dangling edges or partial answer sets. Publish the complete directory only after all seven output files are written. A full export keeps every stored edge, even duplicate triples. H0 checks reverse edges for the entire graph, not just a sample.
- Keep the provided mathematical appendices verbatim. No empirical claims about GPU performance or retrieval quality follow from passing these CPU checks.

## 2026-11-24 — M1

- Resume automatically after the mechanical H0 count check. Only genuine human review/approval gates (starting at H1) require a handoff; ending M0 is not completion of the overall development request.
- Use `bm25s.BM25.get_scores_from_ids` for exact corpus-wide lexical scores. The CSC index pointer differences give document frequencies; posting work counts distinct query terms even if repeated terms contribute repeatedly to the score.
- Break retrieval-score ties by corpus position (the exported IDs are sorted). Pass ordinal scores encoding this returned order to the official evaluator so torchmetrics versions cannot reorder ties differently for different metrics. Saved run scores remain the actual retrieval scores. Unscored candidates retain the official evaluator's below-minimum behavior.
- Keep float32 matrix products, deterministic top-k, corpus/encoder cache fingerprints and three warm timing repetitions including query encoding. Candidate scores for hybrid fusion are computed for the entire lexical/dense union.
- Use validation MRR to compare hybrid lexical weights before readings or a plan-regret oracle exist. Full dense/hybrid GPU runs and their weight selection are deferred under the user's GPU-testing waiver, not reported as passed experiments. Existing local CPU model smoke checks and focused tests still validate the interfaces and scoring paths.

## 2026-11-24 — M2

- The graph has 56 `(source type, exact relation, destination type)` combinations. Hand-author the descriptions and cues in `schemas/prime.yaml`; validate every edge count and reciprocal relation from the data. Schema version mismatches, missing descriptions and duplicate IDs are hard failures.
- Follow the development guide's 2-6-word query-only fuzzy fallback. The design text also mentions one-word fuzzy spans; keep one-word matching exact-only for the more conservative v0 implementation. Common-word all-caps exceptions apply to gene/protein symbols, not every all-caps entity.
- Exclude same-type-ambiguous mentions from the fact's keys but retain them for masking, so excluded real names do not leak into another mention's scorer input. Self mentions remain self-masked. Pure numerical names include punctuation-separated numbers and are excluded.
- The specified Parquet mention record has one `value_type`; a cross-type mention therefore has multiple same-span records with one shared `scorer_input`. Group by `(fact_id, start, end)` at scoring time and union the candidate types; do not score each type with a separate softmax.
- Keep M2 artifacts in the versioned `data/prime/facts_prime-0.1.0/` subdirectory rather than mixing incompatible fact-cache generations into the M0 root. An M2 run refuses to overwrite either its artifact directory or `m2_stats.json`; use separate `--out` and `--runs` paths for smoke/repeated runs.
- The H1 review files contain sampled real mentions with unfilled `correct` labels. Full fact vectors and all full-corpus dense/GPU measurements remain unexecuted under the testing waiver. H1 precision and schema approval cannot be replaced with automatic coverage statistics.

## 2026-11-24 — continued development, M3

- On the user's request to continue, proceed with implementation and CPU/data checks without pretending that H1 has been approved. Keep human-review statuses pending in downstream artifacts; final experiment execution still requires H5 approval.
- Aggregate the full corpus columnarly using document-level maxima, equivalent to the reference `Lexicon`. A native-only build is explicitly tagged `native` and must not be used to claim complete text-aware support. It is the authoritative input for KB-grounded role-swap construction.
- PACC includes zero-scored text mentions in its denominator, uses only KB-connected document/value pairs for TPR/FPR estimation, preserves raw hard counts, and warns on missing or unstable rates. Native-only output reports PACC as unmeasured rather than inventing rates.
- Publish a lexicon manifest with matching index/lexicon state fingerprints and an output checksum. Partial smoke caches cannot drive pruning. Same-span cross-type mention records are always scored in one joint softmax, including the applicable `none` outcomes.
- The documented synthetic binding experiment reproduced its reference values on CPU (default R-precision: bound 0.835, unbound 0.483, gate-only 0.436). No zero-shot corpus-wide GPU scoring has been executed.

## 2026-11-24 — M4

- Silver labels use the mathematical answer **set**. The official files contain repeated answer IDs in 23 train and 7 validation queries; deduplicate before the majority threshold, log the condition and leave the M0 files untouched. An initial strict-list uniqueness check exposed this during the full run and was corrected with a regression test.
- Query-only fuzzy matching memoizes normalized spans without changing candidate order, thresholds or results. Train/validation silver data are separate; calibration uses even train qids, not CSV positions. No test silver labels are generated.
- Cross-type normalized-name groups take precedence over individual value hashes for every selected value sharing that name. This keeps same-value queries and their cross-type partners in one split even when the two hash parities disagree.
- The prescribed support fallback (3 to 2) yields 667 role-swap test queries, below the 1,000 target but above the fallback minimum of 600. Report MDE 4.8513 points with assumed discordance 0.2; do not lower eligibility further or claim the original size target passed.
- H2 and H3 material has been generated, not human-approved. The 130 independently checked silver queries include all repeated-answer cases; this verifies construction, not semantic label agreement.

## 2026-11-24 — M5 code and CPU data checks

- Training-only frames are generated from independently authored role predicates, never from test templates. Collision metadata is mined into the training-template sidecar fields rather than mutating the already verified schema and invalidating its input caches.
- Follow the development guide's per-slot/per-style interpretation of `samples_per_slot`: the default produces 2 * 66 * 2,000 balanced examples. This is larger than the design's approximate total-size paragraph; use 1,000 per style when that smaller budget is intended.
- Separate frame selection and surrogate RNG streams, preserve shared bindings in hard pairs, and give secondary mentions a distinct relation with the same document type. A12 is explicitly marked and loads real targets only through its dedicated input option.
- Cue validation treats shared family phrases as non-conflicting and lets an explicit primary cue dominate an overlapping affirmative fragment (for example, `not expressed in`). Planned other-role cues must bind to the secondary mention's clause, with a conservative distance check when both mentions share a clause. Otherwise the required multi-mention examples would be rejected merely for containing the other mention's cue.
- Deduplicate before masking; keep template IDs across paraphrases and resampling. Hold out at least 10% of available template IDs per slot/style, rounded up. Ensure even short training runs evaluate/save at least once per epoch so `load_best_model_at_end` is meaningful.
- Move the lightweight Git/run helpers into `artifacts.py` so the separate vLLM entry point does not import STaRK, pandas or PyArrow. No vLLM execution, neural optimization or real calibration-quality acceptance has been performed.

## 2026-11-24 — M6 and M7 core

- Use the exact reference posterior/conformal helpers. The worked-example test uses its four-slot toy schema; a real Prime query includes all compatible typed slots in the smoothing denominator, including slots with very small query-side probability.
- Conformal fitting is restricted to labelled even-qid train silver segments and the complete text-aware lexicon. Model, temperature, schema and corpus fingerprints must match. A native-only bootstrap cache is never silently accepted as a full planning index.
- Load aligned half-precision contexts as normalized float32 for indexing. Keep the provided fast scorer unchanged. The pipeline materializes per-reading matches for fusion/IA and rejects stale index/lexicon pairs or undercounted hard support.
- Binding weight zero follows the exact hybrid baseline path, skipping planning and value postings. IA emits ordinal result scores to preserve its reordered ranking for consumers that sort scores; the original fused scores remain in the logged plan.
- Evidence is recomputed only for the top ten documents to preserve the verbatim scorer API. This is not claimed to be free: the extra text-fact work and evidence-stage time are explicitly counted.
- The M7 core is tested end to end on CPU fixtures. The subsequent cache/tuning and M8 implementation is recorded below; no empirical GPU/full-corpus M7 acceptance is asserted.

## 2026-11-24 — M7 feature replay and tuning

- Share candidate selection, fact-work counting, hard AND and IA ranking between live search and replay. Capture all supported readings before conformal pruning and retain their uncapped posting union; capping that union early loses documents that become top-`K_V` after a competing reading is removed.
- Persist float64 match matrices and int64 work matrices in immutable, checksummed directories. Reuse verified hashes only while file stats are unchanged and bound the feature-set cache. Replay time is not end-to-end search latency: save it separately and leave online latency null.
- Refit calibration thresholds for every beta, with matched raw-support and uniform-query banks. Use only even-qid train labels for fitting and complete validation labels for staged selection. Select B2's lexical weight separately and verify final selected quality against the official evaluator.
- Freeze only accepted complete evidence, rechecking production data/schema/model inputs. The receipt binds the frozen YAML, selection and conformal bank but explicitly leaves H5 pending. Fixture freezes cannot authorize production experiments.
- B4 coordinate ascent uses complete validation features and records the representation hash, selected B2 weight and validation fingerprint. Sentence MaxP and field representations use unmasked 256-dimensional topic embeddings.

## 2026-11-24 — M8 protocol and reproducible reporting

- Describe the full baseline/ablation matrix before execution; alias identical configurations to the same saved result. B7 cannot use test/human queries. Formal execution requires an external H5 approval bound to both the frozen YAML and final-plan hashes, unchanged data/artifacts, and the same clean Git commit.
- Keep plans/approvals under ignored artifact directories. Check authorization before parsing formal queries; create one immutable execution ledger and record initialization/search failures as failed. Use the first measured ranking for quality and the median of three warm measurements for latency, recording instability instead of selecting the best repeat.
- Fan-out runs one B2 pass per kept reading, then weighted RRF with denominator `60 + rank`; count every rewrite and planning pass. Role-Confusion@10 excludes the query's own correct answers from the partner's answer set.
- A7 uses pinned fast-tokenizer offsets for non-overlapping 128-token facts and a separate downstream artifact chain. A11 packs signed int8 vectors with big-endian order and zero mapped to zero; account for actual serving-index inputs, including lexicon, manifests and schema, not vectors alone. The CPU reference may unpack vectors internally, so on-disk bytes are not a RAM claim.
- A12 reveals only the target to the slot scorer at training, document scoring, calibration and query inference; context/segment embeddings stay masked. Enforce visibility agreement across runtime artifacts and ignore role cues inside visible target names during generation validation.
- Extend official quality evaluation explicitly for empty rankings: all retrieval metrics are zero and the extension is counted. A fake zero-score candidate would incorrectly earn quality under the upstream evaluator.
- Copy all diagnostic inputs into the execution bundle so reports need no original corpus or models. Fit PACC diagnostic rates on even document IDs and evaluate odd IDs rather than measuring the algebraically guaranteed fit-set correction.
- Reports require exact paired query IDs and matching regret masks. Use planned table-family Holm correction, bootstrap intervals and measured discordance/MDE; identical zero-effect aliases cannot pass a positive-effect criterion. Missing jobs remain NOT RUN; fixtures are excluded unless explicitly requested and then visibly watermarked.
- The complete CPU suite has 94 passing tests. GPU training, empirical tuning, full neural retrieval experiments, human reviews and H5 approval remain unexecuted/unapproved; no production freeze or formal test result is claimed.

## 2026-11-25 — M9 stock-engine adapter

- Continue from the existing M1-M8 commits rather than reimplementing already completed milestones. Human-review and production-experiment approvals remain independent of continued code work.
- Use a generated single-node application with `hosts.xml`, the standard `redundancy` element, bound/max rank functions and six exponent profiles. Native/text sparse results are combined by the specified sum-minus-intersection formula. The query carries `link ** a` without six-decimal rounding; document text gates independently raise `pi` to `a`.
- Keep feed/query labels in the safe engine alphabet and slot indices in schema-file order. Drop sub-threshold text keys intentionally, pack contexts only with `reference.binarize`, preserve zero-to-bit-zero and reject already packed/invalid-dimensional contexts instead of double-packing them.
- Read user-defined match features under Vespa's documented `rankingExpression(name)` keys, with the direct-name form accepted for compatible responses. Missing features, partial/degraded coverage, nonfinite values, operational deployment actions and HTTP-200 query errors are failures, not zero scores.
- Feeding requires an empty dedicated deployment; replacing its application requires explicit acknowledgement. Smoke cleanup deletes only its four newly fed document IDs. Server version, application/feed/schema/corpus identities and matched topic bytes bind later measurements to the feed receipt.
- Benchmark only fixed validation IDs. Reference parity checks every candidate of each parity query, including results outside the final top 100, with absolute tolerance `1e-3` and an explicitly reported float32 tie tolerance of `1e-6`. Client-side min-max fusion uses the actual returned pool; different engine/research candidate sets are not called scoring parity failures.
- Record full-client latency and both sum/max server/RPC time for fan-out. Save HTTP feed acknowledgements separately from searchable-visibility time. Docker memory is a whole-container observation; unavailable memory is `NOT MEASURED`. Injected CPU fixture pipelines are watermarked `fixture` and cannot become real-engine measurement evidence.
- The host has no Docker/Podman executable, no installed WSL distribution and no responding local Vespa service. Real application compilation/deployment, engine parity, 200-query validation parity and 1,000-query timing/RAM remain **NOT RUN**; CPU serialization/algebra tests do not substitute for those checks. Full topic/context/role-score/combined-lexicon artifacts also remain absent under the GPU-testing waiver.
- Verification: full CPU suite `110 passed, 1 skipped`; the extended real-Parquet bridge and M9-only rerun `17 passed, 1 skipped`. The built wheel contains all application resources and `pip check` passes. The sole skipped check is the explicitly opt-in real Vespa deployment, not a GPU experiment.

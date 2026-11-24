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
- The M7 core is tested end to end on CPU fixtures. Full reusable tuning features, grid search, frozen-config approval and M8 experiment/report orchestration remain future work; no empirical GPU/full-corpus M7 acceptance is asserted.

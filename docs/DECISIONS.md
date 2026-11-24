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

# Implementation decisions

## 2026-11-24 — M0

- Pin the Prime inputs to STaRK Hub revision `88269e23e90587f99476c5dd74e235a0877e69be`. Verify the six processed files before the official loader can deserialize them. Existing input caches are read-only; fetching is explicit. The tested Windows CPU environment is recorded in `requirements.lock`; it is not a Linux CUDA lock.
- Use `load_skb("prime", ...)` and the upstream `get_doc_info(add_rel=True, compact=True)` formatter. Index the sorted, de-duplicated neighbours of each relation by source-node offset on a local view of the SKB. Selecting every row from an unindexed sparse COO tensor repeatedly scans the graph; indexing changes the lookup cost, not the order, text or exported edge multiplicity.
- The 1.1.0 release of `load_qa` lists the Hub even with a local root, and constructs its remote prefix with the OS path separator. Read its exact CSVs and split index files directly instead. Index files contain **query IDs**; they are not CSV row positions. Parse integer answer arrays as JSON, ignore any other upstream columns, preserve split-file order and emit the human table in ID order. No train/test data is mixed.
- Unlike the illustrative `--limit` skeleton, sample exports have no dangling edges or partial answer sets. Publish the complete directory only after all seven output files are written. A full export keeps every stored edge, even duplicate triples. H0 checks reverse edges for the entire graph, not just a sample.
- Keep the provided mathematical appendices verbatim. No empirical claims about GPU performance or retrieval quality follow from passing these CPU checks.

"""M2 native keys, document-side text facts and seeded H1 review material."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .artifacts import logged_run, new_directory, sha256, write_json
from .linker import Linker
from .schema import Schema, derive_slots, verify_inventory
from .stark_io import norm_name
from .text import mask, split_sentences

NATIVE_SCHEMA = pa.schema([("doc_id", pa.int64()), ("key", pa.string())])
MENTION = pa.struct([
    ("value_ids", pa.list_(pa.string())), ("value_type", pa.string()),
    ("start", pa.int64()), ("end", pa.int64()), ("scorer_input", pa.string()),
])
TEXT_SCHEMA = pa.schema([
    ("row", pa.int64()), ("fact_id", pa.string()), ("doc_id", pa.int64()),
    ("field", pa.string()), ("sentence", pa.string()), ("ctx_text", pa.string()),
    ("mentions", pa.list_(MENTION)),
])


def write_native_keys(docs: pd.DataFrame, edges: pd.DataFrame, schema: Schema, path: Path) -> int:
    types = docs.set_index("doc_id")["doc_type"]
    edges = edges.drop_duplicates(["src", "rel", "dst"])
    valid = {s.id for s in schema.slots}
    with pq.ParquetWriter(path, NATIVE_SCHEMA, compression="zstd") as writer:
        for start in range(0, len(edges), 131072):
            part = edges.iloc[start:start + 131072]
            st, dt = part.src.map(types), part.dst.map(types)
            if st.isna().any() or dt.isna().any():
                raise ValueError("Native edge references an unknown document")
            slots = st + "." + part.rel.astype(str).map(norm_name) + "." + dt
            if not set(slots.unique()) <= valid:
                raise ValueError("Native edge references an unknown schema slot")
            keys = slots + "|n" + part.dst.astype(str)
            writer.write_table(pa.Table.from_arrays([part.src.to_numpy(), keys.to_numpy()], schema=NATIVE_SCHEMA))
    return len(edges)


def document_facts(doc: dict, linker: Linker, max_words: int = 64, cap: int = 256):
    if cap < 1:
        raise ValueError("Text fact cap must be positive")
    facts, seen, ambiguous = [], 0, 0
    own = f"n{doc['doc_id']}"
    fields = json.loads(doc["details_json"])
    if not isinstance(fields, dict) or not all(isinstance(s, str) for s in fields.values()):
        raise ValueError(f"Document {doc['doc_id']} has non-string detail fields")
    for field, text in fields.items():
        for sentence in split_sentences(text, max_words):
            mentions = linker.link(sentence)  # never apply fuzzy matching to documents
            targets = []
            for i, mention in enumerate(mentions):
                mention["is_self"] = own in mention["values"]
                if mention["is_self"]:
                    continue
                seen += 1
                counts = Counter(linker.vtype[v] for v in mention["values"])
                if max(counts.values()) > 1:
                    ambiguous += 1
                else:
                    targets.append(i)
            if not targets:
                continue
            stored = []
            for i in targets:
                m = mentions[i]
                scorer_input = mask(sentence, mentions, i, doc["doc_type"], None)
                # The Parquet contract has one value_type per record. Same-span
                # records of different types remain one joint scorer input.
                for value_type in sorted(m["types"]):
                    stored.append({"value_ids": sorted(v for v in m["values"] if linker.vtype[v] == value_type),
                                   "value_type": value_type, "start": m["start"], "end": m["end"],
                                   "scorer_input": scorer_input})
            facts.append({
                "fact_id": f"{doc['doc_id']}.t{len(facts)}", "doc_id": int(doc["doc_id"]),
                "field": field, "sentence": sentence,
                "ctx_text": mask(sentence, mentions, None, doc["doc_type"], field), "mentions": stored,
            })
            if len(facts) == cap:
                return facts, seen, ambiguous
    return facts, seen, ambiguous


def _sample(sample: list, item: dict, seen: int, rng, size: int = 100) -> int:
    seen += 1
    if len(sample) < size:
        sample.append(item)
    else:
        pos = int(rng.integers(seen))
        if pos < size:
            sample[pos] = item
    return seen


def _review(mention: dict, linker: Linker) -> dict:
    return {
        "start": mention["start"], "end": mention["end"], "surface": mention["surface"],
        "fuzzy": mention.get("fuzzy", False),
        "values": [{"id": vid, "name": linker.name_by_id[vid], "type": linker.vtype[vid], "probability": p}
                   for vid, p in sorted(mention["values"].items())],
        "correct": None,
    }


def _distribution(values: list[int]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0}
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": int(max(values))}


def write_text_facts(docs: pd.DataFrame, linker: Linker, path: Path, *, max_words: int = 64, cap: int = 256):
    counts, caps = defaultdict(list), Counter()
    per_fact, review, buffer = [], [], []
    seen = ambiguous = n_rows = n_review = 0
    rng = np.random.default_rng(13)
    with pq.ParquetWriter(path, TEXT_SCHEMA, compression="zstd") as writer:
        for n, doc in enumerate(docs.to_dict(orient="records")):
            facts, total, skipped = document_facts(doc, linker, max_words, cap)
            counts[doc["doc_type"]].append(len(facts))
            seen += total
            ambiguous += skipped
            if len(facts) == cap:
                caps[doc["doc_type"]] += 1
            for fact in facts:
                fact["row"] = n_rows
                n_rows += 1
                groups = defaultdict(list)
                for mention in fact["mentions"]:
                    groups[(mention["start"], mention["end"])].append(mention)
                per_fact.append(len(groups))
                for (start, end), mentions in groups.items():
                    values = {v: 1.0 for m in mentions for v in m["value_ids"]}
                    item = {**_review({"start": start, "end": end, "surface": fact["sentence"][start:end],
                                      "values": values}, linker),
                            "doc_id": doc["doc_id"], "doc_name": doc["name"], "doc_type": doc["doc_type"],
                            "field": fact["field"], "sentence": fact["sentence"],
                            "scorer_input": mentions[0]["scorer_input"]}
                    n_review = _sample(review, item, n_review, rng)
                buffer.append(fact)
                if len(buffer) == 1024:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=TEXT_SCHEMA))
                    buffer.clear()
            if (n + 1) % 5000 == 0:
                print(f"M2 text: {n + 1}/{len(docs)} documents, {n_rows} facts", flush=True)
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=TEXT_SCHEMA))
    stats = {
        "text_facts": n_rows,
        "text_facts_per_doc": {kind: {"n_docs": len(vals), **_distribution(vals)} for kind, vals in sorted(counts.items())},
        "documents_hitting_cap": dict(sorted(caps.items())),
        "mentions_per_fact": _distribution(per_fact), "nonself_mentions_seen": seen,
        "same_type_ambiguous_skipped": ambiguous, "share_same_type_ambiguous": ambiguous / seen if seen else 0.0,
    }
    return stats, review


def query_link_stats(queries: list[dict], linker: Linker) -> tuple[dict, list[dict]]:
    review, counts, seen = [], [], 0
    rng = np.random.default_rng(13)
    for n, query in enumerate(queries):
        mentions = linker.link_query(query["query"])
        counts.append(len(mentions))
        for mention in mentions:
            item = {**_review(mention, linker), "qid": query["qid"], "query": query["query"]}
            seen = _sample(review, item, seen, rng)
        if (n + 1) % 200 == 0:
            print(f"M2 query links: {n + 1}/{len(queries)}", flush=True)
    share = float(np.mean(np.asarray(counts) > 0)) if counts else 0.0
    return {"n_queries": len(queries), "segments": sum(counts), "linked_query_share": share,
            "target_at_least_0.8": share >= 0.8, "segments_per_query": _distribution(counts)}, review


def prepare_facts(config_path: Path, *, out: Path | None = None, runs: Path | None = None,
                  limit_docs: int | None = None, limit_queries: int | None = None) -> Path:
    if any(n is not None and n < 1 for n in (limit_docs, limit_queries)):
        raise ValueError("Document/query limits must be positive")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["dataset"] != "prime" or config["seed"] != 13:
        raise ValueError("M2 requires Prime with seed 13")
    schema_path = Path(config["schema"])
    schema = Schema.load(schema_path, expected_version="0.1.0")
    data = Path(config["paths"]["data"])
    out = out or data / f"facts_prime-{schema.version}"
    runs = runs or Path(config["paths"]["runs"])
    stats_path = runs / "m2_stats.json"
    if out.exists() or stats_path.exists():
        raise FileExistsError("M2 artifacts already exist; select a new --out and --runs")
    docs_path, edges_path, queries_path = data / "docs.parquet", data / "edges.parquet", data / "queries_val.jsonl"
    meta = {"settings": config, "schema_sha256": sha256(schema_path), "schema_version": schema.version,
            "docs_sha256": sha256(docs_path), "edges_sha256": sha256(edges_path),
            "queries_sha256": sha256(queries_path), "limit_docs": limit_docs, "limit_queries": limit_queries}
    with logged_run(runs, "m2", meta) as (run, metadata):
        docs = pd.read_parquet(docs_path, columns=["doc_id", "name", "doc_type", "details_json"])
        edges = pd.read_parquet(edges_path)
        derived = derive_slots(docs, edges)
        verify_inventory(schema, derived)
        linking = config["linking"]
        linker = Linker(docs, max_ngram=linking["max_ngram"], min_chars=linking["min_name_chars"],
                        common_words=linking["common_words"], fuzzy_threshold=linking["fuzzy_threshold"],
                        fuzzy_mass=linking["fuzzy_mass"])
        with new_directory(out) as stage:
            native_count = write_native_keys(docs, edges, schema, stage / "native_keys.parquet")
            edge_count = len(edges)
            del edges
            print(f"M2 native: {native_count} keys, {len(derived)} slots", flush=True)
            stats, doc_review = write_text_facts(docs.iloc[:limit_docs], linker, stage / "text_facts.parquet",
                                                max_words=config["facts"]["max_sentence_words"],
                                                cap=config["facts"]["max_text_facts_per_doc"])
            with queries_path.open(encoding="utf-8") as src:
                queries = [json.loads(line) for line in src]
            query_stats, query_review = query_link_stats(queries[:limit_queries], linker)
            stats.update(slots=len(derived), native_keys=native_count, duplicate_native_edges=edge_count - native_count,
                         validation=query_stats, review_run=str(run), human_review="pending",
                         context_embeddings="not run; use slotspace.embed --kind context on the GPU host")
            write_json(stage / "manifest.json", meta)
            write_json(stage / "stats.json", stats)
        for name, rows in (("query_mentions", query_review), ("document_mentions", doc_review)):
            (run / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                              encoding="utf-8")
        write_json(stats_path, stats)
        metadata["counts"] = stats
        print(json.dumps({"out": str(out), "stats": str(stats_path), "review": str(run),
                          "text_facts": stats["text_facts"], "linked_query_share": query_stats["linked_query_share"]}),
              flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/prime_v0.yaml"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--limit-queries", type=int)
    args = parser.parse_args()
    prepare_facts(args.config, out=args.out, runs=args.runs,
                  limit_docs=args.limit_docs, limit_queries=args.limit_queries)


if __name__ == "__main__":
    main()

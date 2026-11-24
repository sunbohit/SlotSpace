"""Deterministic answer-closed diagnostic corpora, never a substitute for Prime test."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import new_directory, read_jsonl, sha256, write_json
from .feature_cache import digest
from .facts import TEXT_SCHEMA
from .schema import Schema
from .slot_scorer import mention_groups


def code_identity():
    root = Path(__file__).parent
    return digest({p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
                   for p in sorted(root.rglob("*.py"))})


def source_identity(source: Path, schema: Path):
    files = {"docs": source / "docs.parquet", "native": source / "facts_prime-0.1.0" / "native_keys.parquet",
             "facts": source / "facts_prime-0.1.0" / "text_facts.parquet", "schema": schema,
             "train": source / "queries_train.jsonl", "val": source / "queries_val.jsonl",
             "silver": source / "silver_prime-0.1.0" / "manifest.json",
             "calibration": source / "silver_prime-0.1.0" / "silver_calibration.jsonl",
             "silver_val": source / "silver_prime-0.1.0" / "silver_val.jsonl"}
    return {key: sha256(path) for key, path in files.items()}


def select_queries(queries, silver, count, split, *, max_mentions=2, max_answers=4, seed=13):
    rows = {row["qid"]: row for row in queries}
    eligible = []
    for row in silver:
        if row["qid"] not in rows:
            raise ValueError("Silver labels reference a query outside their split")
        query = rows[row["qid"]]
        if split == "calibration" and row["qid"] % 2:
            raise ValueError("Pilot calibration may use even train qids only")
        if (1 <= len(row["segments"]) <= max_mentions and any(s["true_readings"] for s in row["segments"])
                and 1 <= len(set(query["answers"])) <= max_answers):
            eligible.append(row)
    eligible.sort(key=lambda q: digest([seed, split, q["qid"]]))
    if count < 1 or len(eligible) < count or len({r["qid"] for r in eligible}) != len(eligible):
        raise ValueError(f"Cannot select {count} distinct eligible {split} queries from {len(eligible)}")
    chosen = eligible[:count]
    return [rows[r["qid"]] for r in chosen], chosen, len(eligible)


def verify_previous(path: Path):
    meta = json.loads((path / "completed.json").read_text(encoding="utf-8"))
    if meta["purpose"] != "pilot" or meta["status"] != "ok" or not all(meta["engineering_checks"].values()):
        raise ValueError("The previous pilot stage has not passed its engineering checks")
    for name, expected in meta["outputs"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(path / name) != expected:
            raise ValueError("The previous pilot result changed after completion")
    return meta


def prepare(source: Path, schema_path: Path, schedule: dict, stage_name: str, out: Path, previous: Path | None = None):
    if schedule["seed"] != 13 or stage_name not in schedule["stages"]:
        raise ValueError("Unknown pilot stage or seed")
    stages = list(schedule["stages"])
    stage = schedule["stages"][stage_name]
    if any(type(stage[k]) is not int or stage[k] < 1 for k in stage):
        raise ValueError("Pilot budgets/repeats must be positive integers")
    identity = source_identity(source, schema_path)
    prior_ids = set()
    if stages.index(stage_name):
        if previous is None:
            raise ValueError("Expansion requires --previous from the preceding completed stage")
        verify_previous(previous)
        parent = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
        if (parent["stage"] != stages[stages.index(stage_name) - 1] or parent["source_identity"] != identity
                or parent["code_identity"] != code_identity() or parent["schedule_sha256"] != digest(schedule)):
            raise ValueError("Pilot promotion source/code/schedule differs from the previous stage")
        prior_ids.update(pq.read_table(previous / "docs.parquet", columns=["doc_id"])["doc_id"].to_pylist())
    elif previous is not None:
        raise ValueError("The first pilot stage must not inherit another result")
    source_meta = json.loads((source / "facts_prime-0.1.0" / "manifest.json").read_text(encoding="utf-8"))
    silver_meta = json.loads((source / "silver_prime-0.1.0" / "manifest.json").read_text(encoding="utf-8"))
    if (source_meta["docs_sha256"] != identity["docs"] or source_meta["schema_sha256"] != identity["schema"]
            or silver_meta["docs_sha256"] != identity["docs"] or silver_meta["schema_sha256"] != identity["schema"]
            or silver_meta["native_sha256"] != identity["native"]
            or silver_meta["query_hashes"] != {"train": identity["train"], "val": identity["val"]}
            or not silver_meta["complete"] or source_meta["limit_docs"] is not None):
        raise ValueError("Pilot source facts/silver do not match the exported corpus/schema")
    schema = Schema.load(schema_path)
    selected, gold, eligible = {}, {}, {}
    for split, filename in (("calibration", "queries_train.jsonl"), ("val", "queries_val.jsonl")):
        selected[split], gold[split], eligible[split] = select_queries(
            list(read_jsonl(source / filename)), list(read_jsonl(source / "silver_prime-0.1.0" / f"silver_{split}.jsonl")),
            stage["calibration" if split == "calibration" else "validation"], split,
            max_mentions=schedule["max_query_mentions"], max_answers=schedule["max_answers"])
    if {q["qid"] for q in selected["calibration"]} & {q["qid"] for q in selected["val"]}:
        raise ValueError("Pilot train-calibration and validation qids overlap")
    if previous is not None and any(
        not set(parent["selected_qids"][split]) <= {q["qid"] for q in selected[split]} for split in selected
    ):
        raise ValueError("Expanded cohorts must retain preceding calibration/validation queries")
    required = prior_ids | {a for rows in selected.values() for q in rows for a in q["answers"]}
    if len(required) > stage["documents"]:
        raise ValueError("Document budget cannot contain the complete answers/preceding corpus; no answer trimming")
    dictionary = pd.read_parquet(source / "docs.parquet", columns=["doc_id", "doc_type", "name"])
    vtypes = {f"n{i}": t for i, t in dictionary[["doc_id", "doc_type"]].itertuples(index=False)}
    values = {v for rows in gold.values() for q in rows for seg in q["segments"] for v in seg["links"]}
    keys = [f"{s.id}|{v}" for v in sorted(values) for s in schema.slots if s.value_type == vtypes[v]]
    native_path = source / "facts_prime-0.1.0" / "native_keys.parquet"
    connected = pq.read_table(native_path, columns=["doc_id"], filters=[("key", "in", keys)])["doc_id"].to_numpy()
    rng = np.random.default_rng(13)
    for ids in (np.unique(connected), dictionary.doc_id.to_numpy()):
        for doc in rng.permutation(ids):
            if len(required) == stage["documents"]:
                break
            required.add(int(doc))
    if len(required) != stage["documents"]:
        raise ValueError("Source corpus is smaller than the requested pilot")
    ids = sorted(required)
    docs = pq.read_table(source / "docs.parquet", filters=[("doc_id", "in", ids)]).to_pandas().sort_values("doc_id")
    native = pq.read_table(native_path, filters=[("doc_id", "in", ids)])
    raw_facts = pq.read_table(source / "facts_prime-0.1.0" / "text_facts.parquet",
                              filters=[("doc_id", "in", ids)]).to_pylist()
    counts, facts, source_rows = {}, [], []
    for fact in raw_facts:
        doc = fact["doc_id"]
        if 1 <= len(mention_groups(fact)) <= schedule["max_fact_mentions"] and counts.get(doc, 0) < stage["facts_per_doc"]:
            source_rows.append(fact["row"])
            facts.append({**fact, "row": len(facts)})
            counts[doc] = counts.get(doc, 0) + 1
    if not facts:
        raise ValueError("Pilot has no eligible text facts; increase the explicit sampling budget")
    with new_directory(out) as dest:
        pq.write_table(pa.Table.from_pandas(docs, preserve_index=False), dest / "docs.parquet", compression="zstd")
        pq.write_table(native, dest / "native_keys.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(facts, schema=TEXT_SCHEMA), dest / "text_facts.parquet", compression="zstd")
        for split in selected:
            for stem, rows in (("queries", selected[split]), ("silver", gold[split])):
                (dest / f"{stem}_{split}.jsonl").write_text(
                    "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows), encoding="utf-8")
        manifest = {"purpose": "pilot", "production": False, "stage": stage_name, "budget": stage,
                    "source": str(source.resolve()), "schema": str(schema_path.resolve()), "source_identity": identity,
                    "code_identity": code_identity(), "schedule_sha256": digest(schedule), "schedule": schedule,
                    "settings": deepcopy(source_meta["settings"]), "dictionary": "full source KB, not pilot candidates",
                    "eligible_queries": eligible, "selected_qids": {s: [q["qid"] for q in rows] for s, rows in selected.items()},
                    "documents": len(ids), "source_fact_rows": source_rows, "text_facts": len(facts),
                    "source_text_facts_in_selected_documents": len(raw_facts), "selection": "answer-closed diagnostic sampling",
                    "parent_result_sha256": sha256(previous / "completed.json") if previous else None,
                    "previous": str(previous.resolve()) if previous else None,
                    "human_reviews": "pending", "files": {p.name: sha256(p) for p in dest.iterdir()}}
        write_json(dest / "manifest.json", manifest)
    return out

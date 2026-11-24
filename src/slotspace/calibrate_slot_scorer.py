"""Fit separate query/document temperatures without using validation or test labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import softmax

from .artifacts import logged_run, read_jsonl, sha256, write_json
from .facts import _sample
from .lexicon_build import native_by_doc
from .schema import Schema
from .slot_scorer import SlotScorer, ece, fit_temperature, mention_groups
from .text import mask, visible_target


def query_records(silver_rows, train_queries: dict[int, dict], doc_types: dict[int, str], schema: Schema,
                  values_visible=False):
    records = []
    for row in silver_rows:
        qid = row["qid"]
        if qid % 2 or qid not in train_queries:
            raise ValueError("Query temperature may use only even-qid train calibration records")
        query = train_queries[qid]["query"]
        mentions = [{"start": s["start"], "end": s["end"],
                     "types": {doc_types[int(v[1:])] for v in s["links"]}} for s in row["segments"]]
        for i, segment in enumerate(row["segments"]):
            candidates = schema.candidates(None, mentions[i]["types"])
            true = sorted({key.split("|", 1)[0] for key in segment["true_readings"]})
            true = true or [k for k in candidates if k.startswith("none.")]
            if not set(true) <= set(candidates):
                raise ValueError("Silver true slots are incompatible with the mention's candidate types")
            context = mask(query, mentions, i, "", None)
            if values_visible:
                context = visible_target(context, query[segment["start"]:segment["end"]])
            records.append({"style": "query", "qid": qid, "context": context,
                            "candidates": candidates, "true_slots": true, "links": segment["links"],
                            "true_readings": segment["true_readings"]})
    return records


def document_records(facts, native: dict[int, set[str]], doc_types: dict[int, str],
                     schema: Schema, limit: int = 10000, values_visible=False):
    if limit < 1:
        raise ValueError("Document calibration sample size must be positive")
    sample, seen = [], 0
    rng = np.random.default_rng(13)
    for fact in facts:
        for group in mention_groups(fact):
            candidates = schema.candidates(doc_types[fact["doc_id"]], group["types"])
            true = [k for k in candidates if any(f"{k}|{v}" in native.get(fact["doc_id"], ())
                                                 for v in group["values"])]
            if true:
                context = (visible_target(group["context"], fact["sentence"][group["start"]:group["end"]])
                           if values_visible else group["context"])
                record = {"style": "doc", "fact_id": fact["fact_id"], "doc_id": fact["doc_id"],
                          "context": context, "candidates": candidates, "true_slots": true}
                seen = _sample(sample, record, seen, rng, size=limit)
    return sample


def fit_style(records: list[dict], scorer: SlotScorer, points_sink: list[dict] | None = None):
    if not records:
        raise ValueError("Cannot calibrate an empty style")
    logits = scorer.logits([r["context"] for r in records], [r["candidates"] for r in records])
    labels = []
    for z, row in zip(logits, records):
        indices = [row["candidates"].index(k) for k in row["true_slots"]]
        labels.append(max(indices, key=lambda i: z[i]))
    temperature = fit_temperature(logits, labels)
    correct = [row["candidates"][int(z.argmax())] in row["true_slots"] for row, z in zip(records, logits)]
    before = [float(softmax(z).max()) for z in logits]
    after = [float(softmax(z / temperature).max()) for z in logits]
    if points_sink is not None:
        points_sink.extend({"style": row["style"], "confidence_before": a, "confidence_after": b,
                            "correct": bool(c)} for row, a, b, c in zip(records, before, after, correct))
    return temperature, {"n": len(records), "accuracy": float(np.mean(correct)),
                         "ece_before": ece(before, correct), "ece_after": ece(after, correct)}


def calibrate(model: str, silver: Path, train_queries: Path, docs: Path, facts: Path, schema_path: Path,
              out: Path, *, device: str = "cuda", kind: str = "ft", doc_limit: int = 10000):
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    points_path = out.with_suffix(".points.parquet")
    if points_path.exists():
        raise FileExistsError(f"Refusing to overwrite {points_path}")
    if kind not in {"ft", "zeroshot"}:
        raise ValueError("Unknown calibration scorer kind")
    schema = Schema.load(schema_path)
    values_visible = json.loads((Path(model).parent / "manifest.json").read_text(encoding="utf-8")).get(
        "variant") == "a12" if (Path(model).parent / "manifest.json").exists() else False
    meta = json.loads((silver / "manifest.json").read_text(encoding="utf-8"))
    if (not meta["complete"] or meta["schema_sha256"] != sha256(schema_path)
            or meta["query_hashes"]["train"] != sha256(train_queries)
            or meta["docs_sha256"] != sha256(docs)
            or meta["native_sha256"] != sha256(facts / "native_keys.parquet")):
        raise ValueError("Calibration inputs do not match complete train silver provenance")
    frame = pd.read_parquet(docs, columns=["doc_id", "doc_type"])
    doc_types = frame.set_index("doc_id").doc_type.to_dict()
    queries = {row["qid"]: row for row in read_jsonl(train_queries)}
    query = query_records(read_jsonl(silver / "silver_calibration.jsonl"), queries, doc_types, schema, values_visible)
    native = native_by_doc(facts / "native_keys.parquet")

    def fact_rows():
        for batch in pq.ParquetFile(facts / "text_facts.parquet").iter_batches(batch_size=1024):
            yield from batch.to_pylist()

    document = document_records(fact_rows(), native, doc_types, schema, doc_limit, values_visible)
    del native
    scorer = SlotScorer(model, schema, pair_order="slot_first" if kind == "zeroshot" else "context_first", device=device)
    scorer.values_visible = values_visible
    config = {"scorer": scorer.metadata(), "schema_sha256": sha256(schema_path),
              "schema_version": schema.version, "silver_sha256": sha256(silver / "silver_calibration.jsonl"),
              "text_facts_sha256": sha256(facts / "text_facts.parquet"), "seed": 13,
              "human_reviews": "pending", "query_source": "train/even qid", "doc_limit": doc_limit}
    with logged_run(Path("runs"), "m5-calibrate", config) as (run, metadata):
        points = []
        tq, q_metrics = fit_style(query, scorer, points)
        td, d_metrics = fit_style(document, scorer, points)
        result = {**config, "T_query": tq, "T_doc": td, "query": q_metrics, "doc": d_metrics}
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as dest:
            json.dump(result, dest, indent=2, allow_nan=False)
        with points_path.open("xb") as dest:
            pq.write_table(pa.Table.from_pylist(points), dest)
        probabilities = scorer.score([r["context"] for r in document[:30]],
                                     [r["candidates"] for r in document[:30]], td)
        review = [{**row, "top3": sorted(p.items(), key=lambda x: (-x[1], x[0]))[:3], "correct": None}
                  for row, p in zip(document[:30], probabilities)]
        (run / "scorer_review.jsonl").write_text("".join(json.dumps(r) + "\n" for r in review), encoding="utf-8")
        metadata["metrics"] = {"query": q_metrics, "doc": d_metrics}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--silver", type=Path, default=Path("data/prime/silver_prime-0.1.0"))
    parser.add_argument("--train-queries", type=Path, default=Path("data/prime/queries_train.jsonl"))
    parser.add_argument("--docs", type=Path, default=Path("data/prime/docs.parquet"))
    parser.add_argument("--facts", type=Path, default=Path("data/prime/facts_prime-0.1.0"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kind", choices=("ft", "zeroshot"), default="ft")
    parser.add_argument("--doc-limit", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.model, args.silver, args.train_queries, args.docs, args.facts,
                               args.schema, args.out, device=args.device, kind=args.kind, doc_limit=args.doc_limit)))


if __name__ == "__main__":
    main()

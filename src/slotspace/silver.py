"""KB-derived silver reading labels for train/validation only."""

import argparse
from collections import Counter
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .artifacts import logged_run, new_directory, sha256, write_json
from .facts import _sample
from .lexicon_build import native_by_doc
from .linker import Linker
from .schema import Schema

LOG = logging.getLogger(__name__)


def silver_segments(query: dict, linker: Linker, schema: Schema,
                    native: dict[int, set[str]], doc_types: dict[int, str]) -> list[dict]:
    answers = sorted(set(query["answers"]))
    if not answers:
        raise ValueError("Silver labels require a nonempty answer set")
    if len(answers) != len(query["answers"]):
        LOG.warning("Query %s has repeated answer IDs; majority labels count the answer set once",
                    query.get("qid", "<unknown>"))
    need = max(1, math.ceil(0.5 * len(answers)))
    result = []
    for mention in linker.link_query(query["query"]):
        true = set()
        for value in mention["values"]:
            counts = Counter()
            for answer in answers:
                for slot in schema.candidates(doc_types[answer], {linker.vtype[value]}):
                    if f"{slot}|{value}" in native.get(answer, ()):
                        counts[slot] += 1
            true.update(f"{slot}|{value}" for slot, n in counts.items() if n >= need)
        result.append({"start": mention["start"], "end": mention["end"],
                       "links": mention["values"], "true_readings": sorted(true)})
    return result


def build_silver(config_path: Path, facts: Path, out: Path, *, runs: Path = Path("runs"),
                 limit: int | None = None) -> Path:
    if limit is not None and limit < 1:
        raise ValueError("Query limit must be positive")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["dataset"] != "prime" or config["seed"] != 13:
        raise ValueError("Silver preparation requires Prime with seed 13")
    data, schema_path = Path(config["paths"]["data"]), Path(config["schema"])
    schema = Schema.load(schema_path)
    docs_path, native_path = data / "docs.parquet", facts / "native_keys.parquet"
    source = json.loads((facts / "manifest.json").read_text(encoding="utf-8"))
    if source["schema_sha256"] != sha256(schema_path) or source["docs_sha256"] != sha256(docs_path):
        raise ValueError("Silver labels require matching schema/document provenance")
    queries = {}
    hashes = {}
    for split in ("train", "val"):
        path = data / f"queries_{split}.jsonl"
        hashes[split] = sha256(path)
        with path.open(encoding="utf-8") as src:
            queries[split] = [json.loads(line) for line in src][:limit]
        if not queries[split] or len({q["qid"] for q in queries[split]}) != len(queries[split]):
            raise ValueError(f"Empty or duplicate {split} query IDs")
    if {q["qid"] for q in queries["train"]} & {q["qid"] for q in queries["val"]}:
        raise ValueError("Train and validation query IDs overlap")
    meta = {"schema_version": schema.version, "schema_sha256": sha256(schema_path),
            "docs_sha256": sha256(docs_path), "native_sha256": sha256(native_path),
            "query_hashes": hashes, "limit": limit, "complete": limit is None, "seed": 13,
            "human_reviews": {"H1": "pending", "H2": "pending"}}
    if (runs / "m4_silver_stats.json").exists():
        raise FileExistsError("Silver statistics already exist; select a fresh --runs root")
    with logged_run(runs, "m4-silver", meta) as (run, metadata):
        docs = pd.read_parquet(docs_path, columns=["doc_id", "name", "doc_type"])
        doc_types = docs.set_index("doc_id").doc_type.to_dict()
        linking = config["linking"]
        linker = Linker(docs, linking["max_ngram"], linking["min_name_chars"], linking["common_words"],
                        linking["fuzzy_threshold"], linking["fuzzy_mass"])
        answer_docs = {a for rows in queries.values() for q in rows for a in q["answers"]}
        native = native_by_doc(native_path, answer_docs)
        stats, review, seen = {}, [], 0
        rng = np.random.default_rng(13)
        with new_directory(out) as stage:
            calibration = (stage / "silver_calibration.jsonl").open("x", encoding="utf-8", newline="\n")
            spare = (stage / "silver_spare.jsonl").open("x", encoding="utf-8", newline="\n")
            with calibration, spare:
                for split, rows in queries.items():
                    total = labelled = multiple = 0
                    slots = Counter()
                    with (stage / f"silver_{split}.jsonl").open("x", encoding="utf-8", newline="\n") as dest:
                        for n, query in enumerate(rows):
                            segments = silver_segments(query, linker, schema, native, doc_types)
                            line = json.dumps({"qid": query["qid"], "segments": segments}, ensure_ascii=False) + "\n"
                            dest.write(line)
                            if split == "train":
                                (calibration if query["qid"] % 2 == 0 else spare).write(line)
                            for segment in segments:
                                total += 1
                                labelled += bool(segment["true_readings"])
                                multiple += len(segment["true_readings"]) > 1
                                slots.update(key.split("|", 1)[0] for key in segment["true_readings"])
                                if split == "train":
                                    item = {"qid": query["qid"], "query": query["query"], "segment": segment,
                                            "surface": query["query"][segment["start"]:segment["end"]],
                                            "values": {v: {"name": linker.name_by_id[v], "type": linker.vtype[v]}
                                                       for v in segment["links"]},
                                            "correct": None}
                                    seen = _sample(review, item, seen, rng)
                            if (n + 1) % 200 == 0:
                                print(f"Silver {split}: {n + 1}/{len(rows)}", flush=True)
                    stats[split] = {"queries": len(rows), "segments": total, "labelled_segments": labelled,
                                    "labelled_share": labelled / total if total else 0,
                                    "multiple_true_readings": multiple, "slots": dict(sorted(slots.items()))}
            write_json(stage / "manifest.json", meta)
        (run / "silver_review.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review),
                                                encoding="utf-8")
        stats["review_run"] = str(run)
        write_json(runs / "m4_silver_stats.json", stats)
        metadata["counts"] = stats
    print(json.dumps({"out": str(out), "review": str(run)}), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/prime_v0.yaml"))
    parser.add_argument("--facts", type=Path, default=Path("data/prime/facts_prime-0.1.0"))
    parser.add_argument("--out", type=Path, default=Path("data/prime/silver_prime-0.1.0"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build_silver(args.config, args.facts, args.out, runs=args.runs, limit=args.limit)


if __name__ == "__main__":
    main()

"""Columnar document-max lexicon aggregation and PACC over corpus evidence."""

import argparse
from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import logged_run, new_directory, sha256, write_json
from .reference import TAU_MIN, pacc_binary
from .schema import Schema
from .slot_scorer import mention_groups

LOG = logging.getLogger(__name__)
LEXICON_SCHEMA = pa.schema([
    ("value_id", pa.string()), ("slot", pa.string()), ("soft", pa.float64()),
    ("hard", pa.int64()), ("soft_pacc", pa.float64()), ("evidence", pa.list_(pa.string())),
])


def native_by_doc(path: Path, doc_ids: set[int] | None = None) -> dict[int, set[str]]:
    result = defaultdict(set)
    source = pq.ParquetFile(path)
    for batch in source.iter_batches(batch_size=131072, columns=["doc_id", "key"]):
        frame = batch.to_pandas()
        if doc_ids is not None:
            frame = frame[frame.doc_id.isin(doc_ids)]
        for doc, key in frame.itertuples(index=False):
            result[int(doc)].add(key)
    return dict(result)


def _mention_pairs(facts: Path, doc_types: dict[int, str]) -> pd.DataFrame:
    seen = set()
    for batch in pq.ParquetFile(facts).iter_batches(batch_size=1024, columns=["fact_id", "doc_id", "mentions"]):
        for fact in batch.to_pylist():
            for group in mention_groups(fact):
                for value, kind in group["values"].items():
                    seen.add((fact["doc_id"], value, doc_types[fact["doc_id"]], kind))
    return pd.DataFrame(sorted(seen), columns=["doc_id", "value_id", "doc_type", "value_type"])


def _pacc(table: pd.DataFrame, native: pd.DataFrame, text: pd.DataFrame, pairs: pd.DataFrame,
          schema: Schema):
    pair_ids = pairs[["doc_id", "value_id"]].drop_duplicates()
    truth = native.assign(value_id=native.key.str.rsplit("|", n=1).str[1]).merge(pair_ids)
    truth["slot"] = truth.key.str.split("|", n=1).str[0]
    true_slots = truth.groupby(["doc_id", "value_id"], observed=True).slot.agg(set).to_dict()
    del truth
    probabilities = dict(zip(zip(text.doc_id, text.key), text.pi))
    native_counts = native.groupby("key", sort=False).size()
    rates, ds = {}, []
    for slot in schema.slots:
        selected = pairs[(pairs.doc_type == slot.doc_type) & (pairs.value_type == slot.value_type)]
        if selected.empty:
            rates[slot.id] = {"tpr": None, "fpr": None, "positives": 0, "negatives": 0, "status": "no text mentions"}
            continue
        rows = [(int(d), v, probabilities.get((d, f"{slot.id}|{v}"), 0.0),
                 slot.id in true_slots.get((d, v), ()), bool(true_slots.get((d, v))))
                for d, v in selected[["doc_id", "value_id"]].itertuples(index=False)]
        frame = pd.DataFrame(rows, columns=["doc_id", "value_id", "pi", "native", "supervised"])
        positive = frame[frame.native]
        negative = frame[frame.supervised & ~frame.native]
        tpr = float(positive.pi.mean()) if len(positive) else None
        fpr = float(negative.pi.mean()) if len(negative) else None
        stable = tpr is not None and fpr is not None and tpr - fpr >= 0.05
        status = "corrected" if stable else "raw fallback"
        rates[slot.id] = {"tpr": tpr, "fpr": fpr, "positives": len(positive), "negatives": len(negative),
                          "status": status}
        if not stable:
            LOG.warning("PACC %s: insufficient/unstable TPR-FPR; retaining raw text counts", slot.id)
        ds.append(frame.loc[frame.supervised, ["doc_id", "value_id", "pi", "native"]].assign(slot=slot.id))
        pool = frame[~frame.native].groupby("value_id").pi.agg(["sum", "count"])
        for idx in table.index[table.slot == slot.id]:
            value = table.at[idx, "value_id"]
            n = int(pool.at[value, "count"]) if value in pool.index else 0
            q_sum = float(pool.at[value, "sum"]) if n else 0.0
            mean = q_sum / n if n else 0.0
            adjusted = pacc_binary(mean, tpr, fpr) if stable else mean
            table.at[idx, "soft_pacc"] = int(native_counts.get(f"{slot.id}|{value}", 0)) + n * adjusted
    columns = ["doc_id", "value_id", "pi", "native", "slot"]
    return rates, pd.concat(ds, ignore_index=True) if ds else pd.DataFrame(columns=columns)


def aggregate_lexicon(native: pd.DataFrame, schema: Schema, text: pd.DataFrame | None = None,
                      pairs: pd.DataFrame | None = None):
    native = native[["doc_id", "key"]].drop_duplicates()
    if text is None:
        text = pd.DataFrame(columns=["doc_id", "key", "pi", "fact_id"])
    else:
        if (not np.isfinite(text.pi).all() or ((text.pi < 0) | (text.pi > 1)).any()):
            raise ValueError("Invalid text role probabilities")
        text = text[text.pi >= TAU_MIN].sort_values("pi", ascending=False, kind="stable")
        text = text.drop_duplicates(["doc_id", "key"])
    extra = text.merge(native.assign(is_native=True), on=["doc_id", "key"], how="left")
    extra = extra[extra.is_native.isna()].drop(columns="is_native")
    contributions = native.assign(pi=1.0, fact_id="")
    if len(extra):
        contributions = pd.concat([contributions, extra], ignore_index=True)
    totals = contributions.groupby("key", sort=True).pi.agg(soft="sum", hard="size")
    evidence = contributions.groupby("key", sort=False).head(5).copy()
    missing = evidence.fact_id.eq("")
    evidence.loc[missing, "fact_id"] = (evidence.loc[missing, "doc_id"].astype(str)
                                      + ".n." + evidence.loc[missing, "key"])
    totals["evidence"] = evidence.groupby("key", sort=False).fact_id.agg(list)
    result = totals.reset_index()
    result[["slot", "value_id"]] = result.key.str.split("|", n=1, expand=True)
    if not set(result.slot) <= {s.id for s in schema.slots}:
        raise ValueError("Lexicon evidence contains a slot outside the schema")
    result["soft_pacc"] = result.soft.astype(float)
    rates, ds = {}, pd.DataFrame()
    if pairs is not None:
        rates, ds = _pacc(result, native, text, pairs, schema)
    result = result[["value_id", "slot", "soft", "hard", "soft_pacc", "evidence"]]
    return result.sort_values(["value_id", "slot"]).reset_index(drop=True), rates, ds


class CorpusLexicon:
    def __init__(self, frame: pd.DataFrame, version: str, mode: str):
        self.version, self.mode = version, mode
        self._soft, self._hard = defaultdict(dict), defaultdict(dict)
        for value, slot, soft, hard in frame[["value_id", "slot", "soft_pacc", "hard"]].itertuples(index=False):
            if slot in self._hard[value] or hard < 1 or not np.isfinite(soft) or soft < 0:
                raise ValueError("Invalid or duplicate lexicon count")
            self._soft[value][slot], self._hard[value][slot] = float(soft), int(hard)

    @classmethod
    def load(cls, folder: Path, expected_index_version: str | None = None):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        version = manifest["index_version"]
        if not manifest["complete"]:
            raise ValueError("Partial lexicons cannot drive corpus pruning")
        if manifest["lexicon_version"] != version or (expected_index_version is not None and expected_index_version != version):
            raise ValueError("Index/lexicon version mismatch")
        path = folder / manifest["file"]
        if sha256(path) != manifest["lexicon_sha256"]:
            raise ValueError("Lexicon checksum mismatch")
        return cls(pd.read_parquet(path), version, manifest["mode"])

    def support(self, value: str) -> dict[str, float]:
        return dict(self._soft.get(value, {}))

    def hard_support(self, value: str, slot: str) -> int:
        return self._hard.get(value, {}).get(slot, 0)


def build_lexicon(facts: Path, docs: Path, schema_path: Path, out: Path, *,
                  text_keys: Path | None = None, runs: Path = Path("runs"),
                  limit_docs: int | None = None) -> Path:
    if limit_docs is not None and limit_docs < 1:
        raise ValueError("Document limit must be positive")
    schema = Schema.load(schema_path)
    source_meta = json.loads((facts / "manifest.json").read_text(encoding="utf-8"))
    if source_meta["schema_sha256"] != sha256(schema_path) or source_meta["docs_sha256"] != sha256(docs):
        raise ValueError("M2 schema/document provenance mismatch")
    native_path, text_path = facts / "native_keys.parquet", facts / "text_facts.parquet"
    mode = "native" if text_keys is None else "combined"
    config = {"schema_version": schema.version, "schema_sha256": sha256(schema_path),
              "docs_sha256": sha256(docs), "native_sha256": sha256(native_path),
              "text_facts_sha256": sha256(text_path), "text_keys_sha256": sha256(text_keys) if text_keys else None,
              "mode": mode, "human_reviews": "pending", "seed": 13,
              "complete": limit_docs is None, "limit_docs": limit_docs}
    key_meta = None
    if text_keys is not None:
        key_meta = json.loads((text_keys.parent / "manifest.json").read_text(encoding="utf-8"))
        if (not key_meta["complete"] or key_meta["schema_sha256"] != config["schema_sha256"]
                or key_meta["docs_sha256"] != config["docs_sha256"]
                or key_meta["text_facts_sha256"] != config["text_facts_sha256"]
                or key_meta["kind"] not in {"ft", "zeroshot"}):
            raise ValueError("Cannot build a lexicon from partial or incompatible text scores")
        config["scorer"] = key_meta["scorer"]
    state = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (runs / "m3_lexicon_stats.json").exists():
        raise FileExistsError("M3 statistics already exist; select a fresh --runs root")
    with logged_run(runs, f"m3-{mode}", config):
        native = pd.read_parquet(native_path)
        doc_frame = pd.read_parquet(docs, columns=["doc_id", "name", "doc_type"])
        if limit_docs is not None:
            native = native[native.doc_id.isin(doc_frame.doc_id.iloc[:limit_docs])]
        text, pairs = None, None
        if text_keys is not None:
            frame = pq.read_table(text_path, columns=["row", "fact_id", "doc_id"]).to_pandas()
            text = pd.read_parquet(text_keys).merge(frame, on="row", validate="many_to_one", how="left")
            if text.doc_id.isna().any():
                raise ValueError("Text keys reference unknown fact rows")
            text = text[text.doc_id.isin(doc_frame.doc_id)]
            if limit_docs is not None:
                raise ValueError("Combined smoke lexicons require a complete, explicitly scoped M2 export")
            pairs = _mention_pairs(text_path, doc_frame.set_index("doc_id").doc_type.to_dict())
        table, rates, ds = aggregate_lexicon(native, schema, text, pairs)
        filename = f"lexicon_{'native' if text_keys is None else key_meta['kind']}.parquet"
        with new_directory(out) as stage:
            pq.write_table(pa.Table.from_pandas(table, schema=LEXICON_SCHEMA, preserve_index=False),
                           stage / filename, compression="zstd")
            write_json(stage / "pacc_rates.json", {"mode": mode, "rates": rates})
            if pairs is not None:
                ds.to_parquet(stage / "distant_supervision.parquet", index=False)
            write_json(stage / "manifest.json", {**config, "index_version": state, "lexicon_version": state,
                                                "file": filename, "lexicon_sha256": sha256(stage / filename)})
        collision = table[table.hard >= 3].groupby("value_id").slot.agg(list)
        top = sorted(collision.items(), key=lambda x: (-len(x[1]), x[0]))[:20]
        names = {f"n{i}": name for i, name in doc_frame[["doc_id", "name"]].itertuples(index=False)}
        stats = {"mode": mode, "pairs": len(table), "soft_sum": float(table.soft.sum()),
                 "hard_sum": int(table.hard.sum()), "pacc_measured": pairs is not None,
                 "top_collisions": [{"value_id": v, "name": names[v], "slots": slots} for v, slots in top]}
        write_json(runs / "m3_lexicon_stats.json", stats)
        print(json.dumps({k: v for k, v in stats.items() if k != "top_collisions"}), flush=True)
    return out / filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, default=Path("data/prime/facts_prime-0.1.0"))
    parser.add_argument("--docs", type=Path, default=Path("data/prime/docs.parquet"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
    parser.add_argument("--text-keys", type=Path)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--limit-docs", type=int)
    args = parser.parse_args()
    if args.native_only == (args.text_keys is not None):
        parser.error("Choose exactly one of --native-only or --text-keys")
    print(build_lexicon(args.facts, args.docs, args.schema, args.out, text_keys=args.text_keys,
                        runs=args.runs, limit_docs=args.limit_docs))


if __name__ == "__main__":
    main()

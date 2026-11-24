"""M8 MaxP and field-weighted baseline representations (never fact-context vectors)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import logged_run, new_directory, sha256, write_json
from .embed import Embedder
from .evaluate import per_query_metrics
from .retrieve import BM25Index, minmax, topk
from .text import split_sentences


def representation_texts(docs: pd.DataFrame, kind: str, edges: pd.DataFrame | None = None):
    if kind not in {"passages", "fields"} or docs.doc_id.duplicated().any():
        raise ValueError("Invalid representation kind or duplicate document IDs")
    rows = []
    for doc in docs.to_dict(orient="records"):
        if kind == "passages":
            texts = split_sentences(doc["text"])
            if not texts:
                raise ValueError(f"Document {doc['doc_id']} has no MaxP passages")
            rows.extend({"doc_id": doc["doc_id"], "field": "passage", "text": text} for text in texts)
        else:
            rows.append({"doc_id": doc["doc_id"], "field": "name", "text": doc["name"]})
            details = json.loads(doc["details_json"])
            if details:
                rows.append({"doc_id": doc["doc_id"], "field": "details",
                             "text": ". ".join(f"{key}: {value}" for key, value in details.items())})
    if kind == "fields":
        if edges is None:
            raise ValueError("Field baseline requires the original graph relation listings")
        names = docs.set_index("doc_id").name.to_dict()
        if not set(edges.src) <= names.keys() or not set(edges.dst) <= names.keys():
            raise ValueError("Field relation lists contain unknown document IDs")
        for (src, rel), group in edges.drop_duplicates(["src", "rel", "dst"]).groupby(["src", "rel"], sort=True):
            rows.append({"doc_id": int(src), "field": f"rel:{rel}",
                         "text": f"{rel}: " + ", ".join(names[int(dst)] for dst in sorted(group.dst))})
    return pd.DataFrame(rows, columns=["doc_id", "field", "text"])


def build_representations(docs: Path, out: Path, encoder: Embedder, kind: str, *,
                          edges: Path | None = None, batch_size: int = 32, schema_version="0.1.0"):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    frame = pd.read_parquet(docs)
    if not len(frame) or not np.all(np.diff(frame.doc_id.to_numpy()) > 0):
        raise ValueError("Representation rows require a sorted nonempty document corpus")
    texts = representation_texts(frame, kind, pd.read_parquet(edges) if edges else None)
    meta = {"kind": kind, "docs_sha256": sha256(docs), "edges_sha256": sha256(edges) if edges else None,
            "encoder": encoder.metadata(), "dimension": 256, "schema_version": schema_version,
            "n_docs": len(frame), "rows": len(texts), "seed": 13}
    with new_directory(out) as stage:
        np.save(stage / "doc_ids.npy", frame.doc_id.to_numpy(dtype=np.int64))
        texts.to_parquet(stage / "texts.parquet", index=False)
        values = np.lib.format.open_memmap(stage / "vectors.npy", mode="w+", dtype=np.float16,
                                           shape=(len(texts), 256))
        for start in range(0, len(texts), batch_size):
            values[start:start + batch_size] = encoder.documents(texts.text.iloc[start:start + batch_size].tolist(),
                                                                 256, batch_size)
            if start % 1024 == 0:
                print(f"{kind}: {start}/{len(texts)} vectors", flush=True)
        values.flush()
        del values
        if kind == "fields":
            fields = sorted(texts.field.unique())
            for i, field in enumerate(fields):
                folder = stage / f"field-{i}"
                folder.mkdir()
                selected = texts[texts.field == field].set_index("doc_id").text
                corpus = selected.reindex(frame.doc_id, fill_value="").tolist()
                pq.write_table(pa.table({"text": corpus}), folder / "docs.parquet")
                BM25Index.cached(folder / "docs.parquet", folder / "bm25", schema_version)
            meta["fields"] = fields
        meta["files"] = {name: sha256(stage / name) for name in ("doc_ids.npy", "texts.parquet", "vectors.npy")}
        write_json(stage / "manifest.json", meta)
    return out


class RepresentationIndex:
    def __init__(self, path: Path, expected_docs_sha: str, encoder_meta: dict):
        self.path = path
        self.meta = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if (self.meta["docs_sha256"] != expected_docs_sha or self.meta["encoder"] != encoder_meta
                or self.meta["dimension"] != 256
                or any(sha256(path / name) != checksum for name, checksum in self.meta["files"].items())):
            raise ValueError("Baseline representation provenance/checksum mismatch")
        self.doc_ids = np.load(path / "doc_ids.npy", allow_pickle=False)
        self.texts = pd.read_parquet(path / "texts.parquet")
        self.vectors = np.load(path / "vectors.npy", mmap_mode="r", allow_pickle=False)
        if (self.vectors.shape != (len(self.texts), 256) or len(self.doc_ids) != self.meta["n_docs"]
                or not self.texts.doc_id.isin(self.doc_ids).all()
                or not np.isfinite(self.vectors).all()
                or not np.allclose(np.linalg.norm(self.vectors.astype(np.float32), axis=1), 1, atol=1e-3)):
            raise ValueError("Invalid baseline vector rows")
        self.positions = np.searchsorted(self.doc_ids, self.texts.doc_id.to_numpy())
        self.fields = {}
        if self.meta["kind"] == "fields":
            for i, field in enumerate(self.meta["fields"]):
                self.fields[field] = BM25Index.cached(path / f"field-{i}" / "docs.parquet",
                                                     path / f"field-{i}" / "bm25", self.meta["schema_version"])

    def passage_scores(self, topic: np.ndarray) -> np.ndarray:
        if self.meta["kind"] != "passages" or topic.shape != (256,):
            raise ValueError("MaxP requires 256-dimensional topic embeddings, not fact contexts")
        values = np.asarray(self.vectors, dtype=np.float32) @ np.asarray(topic, np.float32)
        result = np.full(len(self.doc_ids), -np.inf, dtype=np.float32)
        np.maximum.at(result, self.positions, values)
        if not np.isfinite(result).all():
            raise ValueError("MaxP cache is missing a document")
        return result

    def field_features(self, text: str, topic: np.ndarray, candidates: np.ndarray):
        if self.meta["kind"] != "fields" or topic.shape != (256,):
            raise ValueError("Field scoring requires the field cache and 256-dimensional topic vectors")
        if candidates.dtype.kind not in "iu" or not len(candidates):
            raise ValueError("Field candidates must be nonempty corpus row positions")
        output, work = [], 0
        for field in self.meta["fields"]:
            selected = self.texts.field.to_numpy() == field
            dense = np.zeros(len(self.doc_ids), np.float32)
            dense[self.positions[selected]] = np.asarray(self.vectors[selected], np.float32) @ topic
            lexical, postings = self.fields[field].scores(text)
            work += postings
            output.append(np.column_stack((minmax(dense[candidates]), minmax(lexical[candidates]))))
        return np.stack(output, axis=1), work


def field_scores(features: np.ndarray, weights: dict[str, float], names: list[str], lambda_l=0.5):
    if (features.ndim != 3 or features.shape[1:] != (len(names), 2) or set(weights) != set(names)
            or not np.isfinite(features).all() or not np.isfinite(lambda_l) or lambda_l < 0
            or any(not np.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0):
        raise ValueError("Field weights/features must cover the exact field inventory")
    combined = features[:, :, 0] + lambda_l * features[:, :, 1]
    return combined @ np.array([weights[name] for name in names])


def tune_field_weights(records: list[dict], names: list[str], lambda_l=0.5):
    if not records or not names:
        raise ValueError("Field tuning requires validation examples and fields")
    weights = {name: 1.0 for name in names}

    def metric(candidate):
        return float(np.mean([per_query_metrics(
            r["doc_ids"][topk(field_scores(r["features"], candidate, names, lambda_l), min(100, len(r["doc_ids"])))].tolist(),
            set(r["answers"]))["mrr"] for r in records]))

    trials = []
    # Coordinate ascent terminates because each accepted change strictly
    # improves the finite-grid objective; equal scores keep the current value.
    while True:
        changed = False
        for name in names:
            best, score = weights[name], metric(weights)
            for value in (0, 0.25, 0.5, 1, 2):
                candidate = {**weights, name: value}
                if sum(candidate.values()) == 0:
                    continue
                trial = metric(candidate)
                trials.append({"field": name, "value": value, "mrr": trial})
                if trial > score + 1e-12:
                    best, score = value, trial
            if best != weights[name]:
                weights[name], changed = best, True
        if not changed:
            break
    return weights, trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("passages", "fields"), required=True)
    parser.add_argument("--docs", type=Path, default=Path("data/prime/docs.parquet"))
    parser.add_argument("--edges", type=Path, default=Path("data/prime/edges.parquet"))
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    with logged_run(Path("runs"), f"m8-{args.kind}-prepare", {
        "docs": str(args.docs), "model": args.model, "device": args.device, "batch_size": args.batch_size,
        "kind": args.kind, "seed": 13,
    }):
        print(build_representations(args.docs, args.out, Embedder(args.model, device=args.device), args.kind,
                                    edges=args.edges if args.kind == "fields" else None, batch_size=args.batch_size))


if __name__ == "__main__":
    main()

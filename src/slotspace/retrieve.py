"""Exact CPU retrieval and lexical/dense fallback fusion."""

from importlib.metadata import version
from pathlib import Path
import json

import bm25s
import numpy as np
import pyarrow.parquet as pq
import Stemmer

from .artifacts import new_directory, sha256, write_json


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or k < 1:
        raise ValueError("topk requires finite one-dimensional scores and positive k")
    k = min(k, len(scores))
    if not k:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    threshold = scores[part].min()
    higher = np.flatnonzero(scores > threshold)
    ties = np.flatnonzero(scores == threshold)[:k - len(higher)]
    chosen = np.concatenate((higher, ties))
    return chosen[np.lexsort((chosen, -scores[chosen]))]


def minmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("minmax requires a nonempty finite vector")
    span = float(values.max() - values.min())
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


class BM25Index:
    def __init__(self, texts: list[str]):
        if not texts:
            raise ValueError("Cannot index an empty corpus")
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = bm25s.BM25(backend="numpy")
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False)
        self.retriever.index(tokens, show_progress=False)

    @classmethod
    def cached(cls, docs: Path, cache: Path, schema_version: str = "0.1.0"):
        meta = {"docs_sha256": sha256(docs), "schema_version": schema_version,
                "bm25s": version("bm25s"), "PyStemmer": version("PyStemmer"),
                "stopwords": "en", "stemmer": "english", "method": "lucene", "k1": 1.5, "b": 0.75}
        if cache.exists():
            saved = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
            if saved != meta:
                raise ValueError(f"BM25 cache settings differ: {cache}; choose a new cache directory")
            obj = cls.__new__(cls)
            obj.stemmer = Stemmer.Stemmer("english")
            obj.retriever = bm25s.BM25.load(str(cache), mmap=True, show_progress=False,
                                          verbosity=0, override_params={"backend": "numpy"})
            return obj
        obj = cls(pq.read_table(docs, columns=["text"])["text"].to_pylist())
        with new_directory(cache) as stage:
            obj.retriever.save(str(stage), show_progress=False)
            write_json(stage / "manifest.json", meta)
        return obj

    def scores(self, query: str) -> tuple[np.ndarray, int]:
        tokens = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer,
                               return_ids=False, show_progress=False)[0]
        ids = self.retriever.get_tokens_ids(tokens)
        # A query containing only stopwords/OOV tokens has exactly zero lexical evidence.
        if not ids:
            return np.zeros(self.retriever.scores["num_docs"], dtype=np.float32), 0
        ptr = self.retriever.scores["indptr"]
        postings = sum(int(ptr[i + 1] - ptr[i]) for i in set(ids))
        return self.retriever.get_scores_from_ids(ids), postings

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        rows = [self.scores(query)[0] for query in queries]
        if not rows:
            shape = (0, min(k, self.retriever.scores["num_docs"]))
            return np.empty(shape, np.int64), np.empty(shape, np.float32)
        positions = np.stack([topk(row, k) for row in rows])
        return positions, np.stack([row[pos] for row, pos in zip(rows, positions)])


class DenseIndex:
    def __init__(self, documents: np.ndarray):
        self.documents = np.asarray(documents, dtype=np.float32)
        if (self.documents.ndim != 2 or not all(self.documents.shape)
                or not np.isfinite(self.documents).all()
                or not np.allclose(np.linalg.norm(self.documents, axis=1), 1, atol=1e-3)):
            raise ValueError("Dense documents must be a nonempty matrix of finite unit rows")

    def scores(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        if (query.shape != (self.documents.shape[1],) or not np.isfinite(query).all()
                or not np.isclose(np.linalg.norm(query), 1, atol=1e-3)):
            raise ValueError("Dense query must be a finite unit vector matching the index dimension")
        return self.documents @ query

    def search(self, queries: np.ndarray, k: int, block_size: int = 64):
        queries = np.asarray(queries, dtype=np.float32)
        if (queries.ndim != 2 or queries.shape[1] != self.documents.shape[1] or block_size < 1
                or not np.isfinite(queries).all()
                or not np.allclose(np.linalg.norm(queries, axis=1), 1, atol=1e-3)):
            raise ValueError("Dense queries must be finite unit rows matching the index")
        positions, scores = [], []
        for start in range(0, len(queries), block_size):
            for row in queries[start:start + block_size] @ self.documents.T:
                pos = topk(row, k)
                positions.append(pos)
                scores.append(row[pos])
        width = min(k, len(self.documents))
        return (np.asarray(positions, dtype=np.int64).reshape(-1, width),
                np.asarray(scores, dtype=np.float32).reshape(-1, width))


class BaselineRetriever:
    def __init__(self, bm25: BM25Index | None, dense: DenseIndex | None,
                 k_l: int = 200, k_d: int = 200, final_k: int = 100):
        if min(k_l, k_d, final_k) < 1:
            raise ValueError("Candidate and result limits must be positive")
        if bm25 is not None and dense is not None:
            if bm25.retriever.scores["num_docs"] != len(dense.documents):
                raise ValueError("Lexical and dense corpora have different row counts")
        self.bm25, self.dense = bm25, dense
        self.k_l, self.k_d, self.final_k = k_l, k_d, final_k

    def search(self, text: str, mode: str, query: np.ndarray | None = None,
               lambda_l: float = 0.5):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Unknown baseline: {mode}")
        if not np.isfinite(lambda_l) or lambda_l < 0:
            raise ValueError("Lexical fusion weight must be nonnegative and finite")
        lexical, dense, postings = None, None, 0
        if mode != "dense":
            if self.bm25 is None:
                raise ValueError("This baseline requires a BM25 index")
            lexical, postings = self.bm25.scores(text)
        if mode != "bm25":
            if self.dense is None or query is None:
                raise ValueError("This baseline requires a dense index and query vector")
            dense = self.dense.scores(query)
        if mode == "bm25":
            candidates = topk(lexical, self.k_l)
            values = lexical[candidates]
        elif mode == "dense":
            candidates = topk(dense, self.k_d)
            values = dense[candidates]
        else:
            candidates = np.union1d(topk(lexical, self.k_l), topk(dense, self.k_d))
            values = minmax(dense[candidates]) + lambda_l * minmax(lexical[candidates])
        order = np.lexsort((candidates, -values))[:self.final_k]
        counters = {"bm25_postings": postings, "dense_searches": int(mode != "bm25"),
                    "value_postings": 0, "candidates": len(candidates), "facts_scored": 0,
                    "encoder_texts": int(mode != "bm25")}
        return candidates[order], values[order], counters

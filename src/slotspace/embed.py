"""Qwen3 embedding wrapper and immutable topic-vector caches."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .artifacts import new_directory, sha256, write_json


class Embedder:
    def __init__(self, name: str, max_seq_length: int = 512, device: str = "cuda"):
        if max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device, local_files_only=Path(name).is_dir())
        self.model.max_seq_length = max_seq_length
        self.name = name
        self.max_seq_length = max_seq_length

    def metadata(self) -> dict:
        path = Path(self.name)
        if path.is_dir():
            weights = sorted(path.glob("*.safetensors"))
            if not weights:
                raise ValueError(f"No safetensors weights in {path}")
            model = {p.name: sha256(p) for p in [path / "config.json", *weights]}
        else:
            revision = self.model[0].auto_model.config._commit_hash
            if revision is None:
                raise ValueError("The encoder must have an identifiable model revision")
            model = {"id": self.name, "revision": revision}
        return {"model": model, "max_seq_length": self.max_seq_length,
                "sentence_transformers": version("sentence-transformers"),
                "transformers": version("transformers")}

    @staticmethod
    def _cut(e: np.ndarray, dim: int) -> np.ndarray:
        e = np.asarray(e, dtype=np.float32)
        if e.ndim != 2 or not 1 <= dim <= e.shape[1] or not np.isfinite(e).all():
            raise ValueError("Embedding truncation needs finite rows and a valid target dimension")
        e = e[:, :dim].copy()
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Cannot normalize a zero embedding")
        return e / norms

    def documents(self, texts, dim: int, batch_size: int = 32) -> np.ndarray:
        e = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False, prompt="")
        return self._cut(e, dim).astype(np.float16)

    def queries(self, texts, dim: int, prompt: str | None = None, batch_size: int = 32) -> np.ndarray:
        if prompt is None and "query" not in self.model.prompts:
            raise ValueError("The encoder does not define the required query prompt")
        kw = {"prompt": prompt} if prompt is not None else {"prompt_name": "query"}
        e = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False, **kw)
        return self._cut(e, dim)


def _vectors(docs: Path, cache: Path, encoder: Embedder, batch_size: int,
             schema_version: str, dimensions: tuple[int, ...], names: tuple[str, ...],
             text_column: str):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(docs)
    meta = {"docs_sha256": sha256(docs), "encoder": encoder.metadata(), "schema_version": schema_version,
            "rows": parquet.metadata.num_rows, "dimensions": list(dimensions), "dtype": "float16", "seed": 13}
    if cache.exists():
        if json.loads((cache / "manifest.json").read_text(encoding="utf-8")) != meta:
            raise ValueError(f"Topic cache settings differ: {cache}; choose a new directory")
        for name, dim in zip(names, dimensions):
            array = np.load(cache / name, mmap_mode="r", allow_pickle=False)
            if array.shape != (meta["rows"], dim) or array.dtype != np.float16:
                raise ValueError(f"Invalid topic cache shape or dtype: {cache / name}")
        return tuple(cache / name for name in names)
    if not meta["rows"]:
        raise ValueError("Cannot embed an empty document corpus")
    with new_directory(cache) as stage:
        arrays = [np.lib.format.open_memmap(stage / name, mode="w+", dtype=np.float16,
                                            shape=(meta["rows"], dim))
                  for name, dim in zip(names, dimensions)]
        offset = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=[text_column]):
            vectors = encoder.documents(batch.column(0).to_pylist(), 1024, batch_size)
            end = offset + len(vectors)
            for array, dim in zip(arrays, dimensions):
                array[offset:end] = vectors if dim == 1024 else encoder._cut(vectors, dim).astype(np.float16)
            offset = end
            if offset == batch_size or offset % 1024 == 0 or offset == meta["rows"]:
                print(f"Embedding {text_column}: {offset}/{meta['rows']} rows", flush=True)
        for array in arrays:
            array.flush()
        del array, arrays  # release mapped files before renaming their directory on Windows
        write_json(stage / "manifest.json", meta)
    return tuple(cache / name for name in names)


def topic_vectors(docs: Path, cache: Path, encoder: Embedder, batch_size: int = 32,
                  schema_version: str = "0.1.0") -> tuple[Path, Path]:
    return _vectors(docs, cache, encoder, batch_size, schema_version, (1024, 256),
                    ("topic_qwen3e06_1024.npy", "topic_qwen3e06_256.npy"), "text")


def context_vectors(facts: Path, cache: Path, encoder: Embedder, batch_size: int = 32,
                    schema_version: str = "0.1.0") -> Path:
    rows = pq.read_table(facts, columns=["row"])["row"].to_numpy()
    if not np.array_equal(rows, np.arange(len(rows))):
        raise ValueError("Text-fact row IDs must align with the context matrix")
    return _vectors(facts, cache, encoder, batch_size, schema_version, (128,),
                    ("text_fact_ctx_qwen3e06_128.npy",), "ctx_text")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("topic", "context"), default="topic")
    parser.add_argument("--docs", type=Path,
                        help="Document Parquet, or text_facts.parquet for context mode")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    import torch

    torch.manual_seed(13)
    encoder = Embedder(args.model, device=args.device)
    base = Path("data") / "prime"
    if args.kind == "topic":
        source = args.docs or base / "docs.parquet"
        cache = args.cache or base / "topic_prime-0.1.0_qwen3e06"
        print(*(str(p) for p in topic_vectors(source, cache, encoder, args.batch_size)), sep="\n")
    else:
        source = args.docs or base / "facts_prime-0.1.0" / "text_facts.parquet"
        cache = args.cache or base / "ctx_prime-0.1.0_qwen3e06"
        print(context_vectors(source, cache, encoder, args.batch_size))


if __name__ == "__main__":
    main()

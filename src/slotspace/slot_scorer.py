"""Joint cross-encoder slot distributions and document-fact role scoring."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .artifacts import logged_run, new_directory, sha256, write_json
from .reference import TAU_MIN
from .schema import Schema

KEY_SCHEMA = pa.schema([("row", pa.int64()), ("key", pa.string()), ("pi", pa.float32())])


def mention_groups(fact: dict) -> list[dict]:
    grouped = {}
    for mention in fact["mentions"]:
        loc = (mention["start"], mention["end"])
        group = grouped.setdefault(loc, {"context": mention["scorer_input"], "types": set(), "values": {}})
        if group["context"] != mention["scorer_input"]:
            raise ValueError(f"Inconsistent joint scorer input for fact {fact['fact_id']}, span {loc}")
        group["types"].add(mention["value_type"])
        for value in mention["value_ids"]:
            previous = group["values"].setdefault(value, mention["value_type"])
            if previous != mention["value_type"]:
                raise ValueError(f"Value {value} has conflicting types")
    return [{"start": loc[0], "end": loc[1], **group} for loc, group in grouped.items()]


class SlotScorer:
    def __init__(self, path: str, schema: Schema, temperature: float = 1.0,
                 pair_order: str = "context_first", device: str = "cuda", batch_size: int = 256):
        if pair_order not in {"context_first", "slot_first"} or batch_size < 1:
            raise ValueError("Invalid pair order or batch size")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive and finite")
        from sentence_transformers.cross_encoder import CrossEncoder

        self.model = CrossEncoder(path, num_labels=1, max_length=128, device=device,
                                  local_files_only=Path(path).is_dir())
        self.schema, self.T, self.order, self.batch_size = schema, temperature, pair_order, batch_size
        self.path = path

    def metadata(self) -> dict:
        path = Path(self.path)
        if path.is_dir():
            weights = sorted(path.glob("*.safetensors"))
            if not weights:
                raise ValueError(f"No safetensors model weights at {path}")
            model = {p.name: sha256(p) for p in (path / "config.json", *weights)}
        else:
            revision = self.model.model.config._commit_hash
            if not revision:
                raise ValueError("Slot scorer must have an identifiable model revision")
            model = {"id": self.path, "revision": revision}
        return {"model": model, "pair_order": self.order, "max_length": 128, "temperature": self.T}

    def logits(self, contexts: list[str], candidates: list[list[str]]) -> list[np.ndarray]:
        if len(contexts) != len(candidates):
            raise ValueError("Context/candidate batch lengths differ")
        if not contexts:
            return []
        pairs = []
        for context, slots in zip(contexts, candidates):
            if not slots or len(slots) != len(set(slots)):
                raise ValueError("Each context needs distinct nonempty slot candidates")
            for slot in slots:
                pair = (context, self.schema.slot_text(slot))
                pairs.append(pair if self.order == "context_first" else pair[::-1])
        import torch

        raw = self.model.predict(pairs, batch_size=self.batch_size, activation_fn=torch.nn.Identity(),
                                 convert_to_numpy=True, show_progress_bar=False)
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        if len(raw) != len(pairs) or not np.isfinite(raw).all():
            raise ValueError("Cross-encoder returned invalid scalar logits")
        offsets = np.cumsum([0, *map(len, candidates)])
        return [raw[a:b] for a, b in zip(offsets[:-1], offsets[1:])]

    def score(self, contexts, candidates, temperature: float | None = None) -> list[dict[str, float]]:
        t = self.T if temperature is None else temperature
        if not np.isfinite(t) or t <= 0:
            raise ValueError("Temperature must be positive and finite")
        result = []
        for logits, slots in zip(self.logits(contexts, candidates), candidates):
            scaled = (logits - logits.max()) / t
            p = np.exp(scaled)
            p /= p.sum()
            result.append(dict(zip(slots, p.tolist())))
        return result


def fit_temperature(logit_list, true_index) -> float:
    if not logit_list or len(logit_list) != len(true_index):
        raise ValueError("Temperature fitting requires nonempty aligned labels/logits")
    logits = [np.asarray(z, dtype=np.float64) for z in logit_list]
    for z, y in zip(logits, true_index):
        if z.ndim != 1 or not np.isfinite(z).all() or not 0 <= y < len(z):
            raise ValueError("Invalid calibration logits or target index")

    def nll(log_t):
        t = np.exp(log_t)
        return float(np.mean([logsumexp(z / t) - z[y] / t for z, y in zip(logits, true_index)]))

    fitted = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20)), method="bounded")
    if not fitted.success:
        raise RuntimeError(f"Temperature optimization failed: {fitted.message}")
    return float(np.exp(fitted.x))


def ece(conf, correct, bins: int = 15) -> float:
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if (conf.ndim != 1 or conf.shape != correct.shape or not len(conf) or bins < 1
            or not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1))
            or not np.isin(correct, [0, 1]).all()):
        raise ValueError("ECE requires aligned finite confidences and binary correctness labels")
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        selected = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if selected.any():
            total += selected.mean() * abs(correct[selected].mean() - conf[selected].mean())
    return float(total)


def score_fact_batch(facts: list[dict], doc_types: dict[int, str], scorer: SlotScorer) -> list[dict]:
    contexts, candidates, owners = [], [], []
    for fact in facts:
        for group in mention_groups(fact):
            contexts.append(group["context"])
            candidates.append(scorer.schema.candidates(doc_types[fact["doc_id"]], group["types"]))
            owners.append((fact["row"], group["values"]))
    probabilities = scorer.score(contexts, candidates)
    if len(probabilities) != len(owners):
        raise ValueError("Scorer did not return one joint distribution per mention")
    cells = defaultdict(dict)
    for (row, values), distribution in zip(owners, probabilities):
        for slot, p in distribution.items():
            if slot.startswith("none.") or p < TAU_MIN:
                continue
            value_type = scorer.schema.slot(slot).value_type
            for value, kind in values.items():
                if kind == value_type:
                    key = f"{slot}|{value}"
                    cells[row][key] = max(cells[row].get(key, 0), p)
    return [{"row": row, "key": key, "pi": p}
            for row, keys in sorted(cells.items()) for key, p in sorted(keys.items())]


def score_corpus(facts: Path, docs: Path, schema_path: Path, out: Path, model: str, *,
                 kind: str = "zeroshot", temperature: float = 1, device: str = "cuda",
                 limit: int | None = None, runs: Path = Path("runs")) -> Path:
    if kind not in {"zeroshot", "ft"} or (limit is not None and limit < 1):
        raise ValueError("Invalid scorer kind or fact limit")
    schema = Schema.load(schema_path)
    scorer = SlotScorer(model, schema, temperature, "slot_first" if kind == "zeroshot" else "context_first", device)
    table = pq.read_table(docs, columns=["doc_id", "doc_type"])
    doc_types = dict(zip(table["doc_id"].to_pylist(), table["doc_type"].to_pylist()))
    source = pq.ParquetFile(facts)
    target_rows = source.metadata.num_rows if limit is None else min(limit, source.metadata.num_rows)
    config = {"schema_version": schema.version, "schema_sha256": sha256(schema_path),
              "text_facts_sha256": sha256(facts), "docs_sha256": sha256(docs), "scorer": scorer.metadata(),
              "kind": kind, "tau_min": TAU_MIN, "n_facts": target_rows,
              "complete": target_rows == source.metadata.num_rows, "human_review": "pending", "seed": 13}
    with logged_run(runs, f"score-{kind}", config):
        with new_directory(out) as stage:
            seen = 0
            with pq.ParquetWriter(stage / f"text_fact_keys_{kind}.parquet", KEY_SCHEMA, compression="zstd") as writer:
                for batch in source.iter_batches(batch_size=64):
                    rows = batch.to_pylist()[:target_rows - seen]
                    if not rows:
                        break
                    if [r["row"] for r in rows] != list(range(seen, seen + len(rows))):
                        raise ValueError("Fact rows must be contiguous and aligned")
                    scored = score_fact_batch(rows, doc_types, scorer)
                    if scored:
                        writer.write_table(pa.Table.from_pylist(scored, schema=KEY_SCHEMA))
                    seen += len(rows)
                    print(f"{kind}: {seen}/{target_rows} facts scored", flush=True)
            write_json(stage / "manifest.json", config)
    return out / f"text_fact_keys_{kind}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, default=Path("data/prime/facts_prime-0.1.0/text_facts.parquet"))
    parser.add_argument("--docs", type=Path, default=Path("data/prime/docs.parquet"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--kind", choices=("zeroshot", "ft"), default="zeroshot")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(score_corpus(args.facts, args.docs, args.schema, args.out, args.model, kind=args.kind,
                       temperature=args.temperature, device=args.device, limit=args.limit))


if __name__ == "__main__":
    main()

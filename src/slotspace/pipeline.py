"""Single-pass retrieval, fact binding, evidence, and validation-only execution."""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .artifacts import logged_run, read_jsonl, sha256, write_json
from .embed import Embedder
from .evaluate import Evaluation
from .fast_scorer import FactStore
from .lexicon_build import CorpusLexicon, native_by_doc
from .linker import Linker
from .planner import PlannerContext, PlannerSettings, conformal_value, plan_query, plan_readings
from .ranking import fact_work, rank_matches, select_candidates
from .reference import Doc, Fact, ScoreConfig, TAU_MIN, binarize, fact_gate, psi, sim
from .retrieve import BM25Index, DenseIndex
from .schema import Schema


class CorpusIndex:
    def __init__(self, documents, dim: int, version, precision: str = "float", fingerprint: str | None = None):
        if dim < 1 or precision not in {"float", "binary"} or (precision == "binary" and dim % 8):
            raise ValueError("Binary fact contexts require a positive dimension divisible by eight")
        self.store = FactStore(dim, precision)
        self.version = version
        self.texts, self.native_ids, postings, ids = {}, {}, defaultdict(list), set()
        digest = hashlib.sha256()

        def tracked():
            for doc in documents:
                number = int(doc.doc_id)
                if number < 0 or number in ids:
                    raise ValueError("Corpus index needs distinct nonnegative numeric document IDs")
                ids.add(number)
                self.texts[doc.doc_id] = [f for f in doc.facts if not f.native]
                keys = set()
                for fact in doc.facts:
                    if fingerprint is None:
                        digest.update(json.dumps([doc.doc_id, fact.fact_id, sorted(fact.keys.items())]).encode())
                        if fact.ctx is not None:
                            digest.update(np.asarray(fact.ctx, dtype=np.float64).tobytes())
                    if fact.native:
                        if any(p != 1 for p in fact.keys.values()):
                            raise ValueError("Native facts must have exact role probability one")
                        for key in fact.keys:
                            canonical = f"{doc.doc_id}.n.{key}"
                            if fact.fact_id != canonical:
                                self.native_ids[(doc.doc_id, key)] = fact.fact_id
                    elif (np.asarray(fact.ctx).shape != (dim,) or not np.isfinite(fact.ctx).all()
                          or not np.isclose(np.linalg.norm(fact.ctx), 1, atol=1e-3)):
                        raise ValueError("Text contexts must be finite unit vectors aligned to the index dimension")
                    for key, probability in fact.keys.items():
                        if key.count("|") != 1 or not np.isfinite(probability) or not TAU_MIN <= probability <= 1:
                            raise ValueError("Invalid fact key/probability")
                        if probability >= TAU_MIN:
                            keys.add(key)
                for key in keys:
                    postings[key].append(number)
                yield doc

        self.store.add_docs(tracked())
        self.doc_ids = np.array(sorted(ids), dtype=np.int64)
        if not len(self.doc_ids):
            raise ValueError("Cannot search an empty fact index")
        self.postings = {key: np.array(sorted(rows), dtype=np.int64) for key, rows in postings.items()}
        self.fingerprint = fingerprint or digest.hexdigest()

    def evidence(self, doc: str, reading, cfg: ScoreConfig) -> dict:
        native = self.store.native.get(doc, set())
        values = {k.split("|", 1)[1] for k in reading.keys}
        matching = [k for k in native if (k.split("|", 1)[1] in values if cfg.mode == "no_role" else k in reading.keys)]
        native_key = max(sorted(matching), key=lambda k: 1 if cfg.mode == "no_role" else reading.keys[k]) if matching else None
        native_score = ((1 if cfg.mode == "no_role" else reading.keys[native_key] ** cfg.a)
                        if native_key is not None else 0)
        texts = self.texts.get(doc, [])
        text_score, text_ids = 0.0, []
        gates = [(fact_gate(f, reading), f) for f in texts]

        def context(fact):
            return psi(sim(reading.g, binarize(fact.ctx) if cfg.precision == "binary" else fact.ctx, cfg.precision))

        if cfg.mode == "bound":
            for gate, fact in gates:
                score = gate ** cfg.a * context(fact) ** cfg.b if gate > 0 else 0
                if score > text_score:
                    text_score, text_ids = score, [fact.fact_id]
        elif cfg.mode in {"unbound", "gate_only"} and gates:
            gate, fact = max(gates, key=lambda x: x[0])
            if gate > 0:
                text_score, text_ids = gate ** cfg.a, [fact.fact_id]
                if cfg.mode == "unbound":
                    other = max(texts, key=context)
                    text_score *= context(other) ** cfg.b
                    text_ids = list(dict.fromkeys([fact.fact_id, other.fact_id]))
        elif cfg.mode == "no_role":
            for fact in texts:
                if any(k.split("|", 1)[1] in values for k in fact.keys):
                    score = context(fact) ** cfg.b
                    if score > text_score:
                        text_score, text_ids = score, [fact.fact_id]
        parts = []
        if native_score > 0 and (cfg.combine == "sum" or native_score >= text_score):
            parts.append({"kind": "native", "fact_ids": [self.native_ids.get(
                (doc, native_key), f"{doc}.n.{native_key}")], "key": native_key})
        if text_score > 0 and (cfg.combine == "sum" or text_score > native_score):
            parts.append({"kind": "text", "fact_ids": text_ids})
        return {"reading": reading.rid, "match": native_score + text_score if cfg.combine == "sum"
                else max(native_score, text_score), "parts": parts}


class Pipeline:
    def __init__(self, ctx: PlannerContext, index: CorpusIndex, bm25: BM25Index, dense: DenseIndex,
                 scoring: ScoreConfig = ScoreConfig(), lambda_b: float = 1, lambda_d: float = 1,
                 lambda_l: float = 0.5, final_k: int = 100, hard_and: bool = False):
        if (ctx.lexicon.version != index.version or scoring.precision != index.store.precision
                or len(index.doc_ids) != len(dense.documents)
                or ctx.cfg.ctx_dim != index.store.dim
                or bm25.retriever.scores["num_docs"] != len(index.doc_ids)):
            raise ValueError("Index, lexicon, vector and corpus versions/dimensions differ")
        if (scoring.mode not in {"bound", "unbound", "gate_only", "no_role"}
                or scoring.combine not in {"max", "sum"} or final_k < 1
                or any(not np.isfinite(x) or x < 0 for x in (lambda_b, lambda_d, lambda_l))
                or any(not np.isfinite(x) or x <= 0 for x in (scoring.a, scoring.b))):
            raise ValueError("Invalid scoring settings")
        if ctx.cfg.weight_objective == "ia" and scoring.combine != "max":
            raise ValueError("IA coverage probabilities require max-combined reading matches")
        for key, postings in index.postings.items():
            slot, value = key.split("|", 1)
            if ctx.lexicon.hard_support(value, slot) != len(postings):
                raise ValueError(f"Index/lexicon hard-support mismatch: {key}")
        self.ctx, self.index, self.bm25, self.dense = ctx, index, bm25, dense
        self.scoring, self.weights, self.final_k = scoring, (lambda_b, lambda_d, lambda_l), final_k
        self.hard_and = hard_and

    def search(self, query: str, qid=None):
        start = perf_counter()
        if self.ctx.lexicon.version != self.index.version:
            raise ValueError("Index/lexicon version changed before query planning")
        if self.weights[0] == 0:
            cfg = self.ctx.cfg
            plan = {"query_id": qid, "schema_version": self.ctx.schema.version,
                    "lexicon_version": self.ctx.lexicon.version, "segments": [], "path": "vector_first",
                    "budgets": {"K_L": cfg.K_L, "K_D": cfg.K_D, "K_V": cfg.K_V},
                    "weight_objective": "p", "estimated_value_postings": 0, "planner_encoder_texts": 0,
                    "binding_disabled": True}
        else:
            plan = plan_query(query, qid, self.ctx)
        planned = perf_counter()
        result = self.execute_plan(query, plan)
        result["counters"]["planner_ms"] = 1000 * (planned - start)
        result["counters"]["latency_ms"] = 1000 * (perf_counter() - start)
        return result

    def execute_plan(self, query: str, plan: dict):
        start = perf_counter()
        if plan["lexicon_version"] != self.index.version or self.ctx.lexicon.version != self.index.version:
            raise ValueError("Query plan and index/lexicon snapshots differ")
        topic = self.ctx.embedder.queries([query], dim=self.dense.documents.shape[1])[0]
        encoded = perf_counter()
        dense = self.dense.scores(topic)
        lexical, bm25_postings = self.bm25.scores(query)
        readings = plan_readings(plan) if self.weights[0] != 0 else []
        keys = {key for reading in readings for key in reading.keys}
        postings = {}
        for key in sorted(keys):
            slot, value = key.split("|", 1)
            rows = self.index.postings.get(key, np.empty(0, dtype=np.int64))
            if len(rows) != self.ctx.lexicon.hard_support(value, slot):
                raise ValueError(f"Kept posting is inconsistent with lexicon: {key}")
            postings[key] = np.searchsorted(self.index.doc_ids, rows)
        candidates, posting_work = select_candidates(lexical, dense, postings, keys, plan["budgets"])
        retrieved = perf_counter()
        doc_ids = [str(d) for d in self.index.doc_ids[candidates]]
        matrix = (np.column_stack([self.index.store.scores(doc_ids, [replace(r, weight=1)], self.scoring)
                                   for r in readings]) if readings else np.empty((len(candidates), 0)))
        order, score_values, fused = rank_matches(self.index.doc_ids[candidates], lexical[candidates],
                                                  dense[candidates], matrix, readings, plan, self.weights,
                                                  self.final_k, self.hard_and)
        ranking = [doc_ids[i] for i in order]
        bound_at = perf_counter()
        evidence = {d: [self.index.evidence(d, r, self.scoring) for r in readings] for d in ranking[:10]}
        scoring_work = int(fact_work(self.index.store, doc_ids, readings, self.scoring.mode).sum())
        evidence_work = sum(len(self.index.texts.get(d, ())) * len(readings) for d in ranking[:10])
        return {
            "doc_ids": list(map(int, ranking)), "scores": score_values.astype(float).tolist(),
            "fusion_scores": fused[order].astype(float).tolist(), "plan": plan, "evidence": evidence,
            "counters": {"bm25_postings": bm25_postings, "dense_searches": 1, "value_postings": posting_work,
                         "candidates": len(candidates), "facts_scored": scoring_work + evidence_work,
                         "evidence_facts_scored": evidence_work,
                         "encoder_texts": 1 + (plan["planner_encoder_texts"] if self.weights[0] != 0 else 0),
                         "encoding_ms": 1000 * (encoded - start), "retrieval_ms": 1000 * (retrieved - encoded),
                         "binding_ms": 1000 * (bound_at - retrieved),
                         "evidence_ms": 1000 * (perf_counter() - bound_at),
                         "latency_ms": 1000 * (perf_counter() - start)},
        }


def load_index(docs: Path, facts: Path, text_keys: Path, context_cache: Path,
               lexicon: Path, schema: Path, precision="float"):
    lex = CorpusLexicon.load(lexicon)
    meta = json.loads((lexicon / "manifest.json").read_text(encoding="utf-8"))
    context = json.loads((context_cache / "manifest.json").read_text(encoding="utf-8"))
    if (lex.mode != "combined" or meta["docs_sha256"] != sha256(docs)
            or meta["schema_sha256"] != sha256(schema)
            or meta["native_sha256"] != sha256(facts / "native_keys.parquet")
            or meta["text_keys_sha256"] != sha256(text_keys)
            or meta["text_facts_sha256"] != sha256(facts / "text_facts.parquet")
            or context["docs_sha256"] != meta["text_facts_sha256"]
            or context["schema_version"] != meta["schema_version"]):
        raise ValueError("Search index, context vectors and lexicon are incompatible")
    if context.get("precision") == "binary":
        if precision != "binary":
            raise ValueError("Packed contexts cannot be used for floating-point scoring")
        vector_path = context_cache / "text_fact_ctx_qwen3e06_128.binary.npy"
        if sha256(vector_path) != context["vector_sha256"]:
            raise ValueError("Packed context checksum mismatch")
        packed = np.load(vector_path, allow_pickle=False)
        if packed.dtype != np.int8 or packed.shape != (context["rows"], 16):
            raise ValueError("Invalid packed context shape or dtype")
        vectors = (np.unpackbits(packed.view(np.uint8), axis=1, bitorder="big").astype(np.float32) * 2 - 1) / np.sqrt(128)
        vector_sha = context["source_sha256"]
    else:
        vector_path = context_cache / "text_fact_ctx_qwen3e06_128.npy"
        vectors = np.load(vector_path, allow_pickle=False)
        vector_sha = sha256(vector_path)
    info = pd.read_parquet(facts / "text_facts.parquet", columns=["row", "fact_id", "doc_id"])
    if not np.array_equal(info.row, np.arange(len(info))) or vectors.shape != (len(info), 128):
        raise ValueError("Fact context vectors are not aligned to the fact rows")
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0) or not np.allclose(norms, 1, atol=1e-3):
        raise ValueError("Invalid fact context vector norms")
    vectors /= norms
    keyed = pd.read_parquet(text_keys)
    if (not keyed.row.isin(info.row).all() or keyed.duplicated(["row", "key"]).any()
            or not np.isfinite(keyed.pi).all() or not keyed.pi.between(TAU_MIN, 1).all()):
        raise ValueError("Invalid or unaligned document-side role keys")
    by_row = {int(row): dict(zip(group.key, group.pi)) for row, group in keyed.groupby("row", sort=False)}
    text_by_doc = defaultdict(list)
    for row, fact_id, doc in info.itertuples(index=False):
        text_by_doc[int(doc)].append(Fact(fact_id, by_row.get(row, {}), vectors[row]))
    native = native_by_doc(facts / "native_keys.parquet")
    ids = pq.read_table(docs, columns=["doc_id"])["doc_id"].to_pylist()

    def documents():
        for doc in ids:
            yield Doc(str(doc), [Fact(f"{doc}.n.{k}", {k: 1.0}, None, True) for k in sorted(native.get(doc, ()))]
                      + text_by_doc.get(doc, []))

    fingerprint = hashlib.sha256((str(lex.version) + vector_sha).encode()).hexdigest()
    index = CorpusIndex(documents(), 128, lex.version, precision, fingerprint)
    if not np.array_equal(index.doc_ids, ids):
        raise ValueError("Topic-vector rows require sorted document IDs")
    return index, lex


@dataclass(frozen=True)
class RuntimePaths:
    text_keys: Path
    context_cache: Path
    lexicon: Path
    model: str
    temperatures: Path
    conformal: Path
    facts: Path = Path("data/prime/facts_prime-0.1.0")
    topic_cache: Path = Path("data/prime/topic_prime-0.1.0_qwen3e06")
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    kind: str = "ft"
    values_visible: bool = False

    @classmethod
    def from_mapping(cls, values: dict):
        names = {f.name for f in fields(cls)}
        if values.keys() - names:
            raise ValueError(f"Unknown runtime artifact fields: {sorted(values.keys() - names)}")
        mapped = {k: v if k in {"model", "embedding_model", "kind", "values_visible"} else Path(v) for k, v in values.items()}
        result = cls(**mapped)
        if result.kind not in {"ft", "zeroshot"}:
            raise ValueError("Invalid runtime scorer kind")
        if type(result.values_visible) is not bool:
            raise ValueError("values_visible must be a boolean")
        return result

    def to_dict(self):
        result = {}
        for key, value in asdict(self).items():
            result[key] = (str(value.resolve()) if isinstance(value, Path) else
                           str(Path(value).resolve()) if key in {"model", "embedding_model"} and Path(value).is_dir()
                           else value)
        return result


def load_pipeline(config: dict, paths: RuntimePaths, device: str = "cuda") -> Pipeline:
    if config["dataset"] != "prime" or config["seed"] != 13:
        raise ValueError("The v0 runtime requires Prime and seed 13")
    schema_path, data = Path(config["schema"]), Path(config["paths"]["data"])
    schema, settings = Schema.load(schema_path), PlannerSettings.from_config(config)
    temperatures = json.loads(paths.temperatures.read_text(encoding="utf-8"))
    conformal = json.loads(paths.conformal.read_text(encoding="utf-8"))
    index, lex = load_index(data / "docs.parquet", paths.facts, paths.text_keys, paths.context_cache,
                            paths.lexicon, schema_path, config["scoring"]["precision"])
    if (conformal["lexicon_version"] != lex.version or conformal["schema_sha256"] != sha256(schema_path)
            or conformal["temperatures_sha256"] != sha256(paths.temperatures)):
        raise ValueError("Conformal settings/model/index provenance mismatch")
    threshold = conformal_value(conformal, settings, config["planner"]["alpha"])
    encoder = Embedder(paths.embedding_model, config["embedding"]["max_seq_length"], device)
    topic_meta = json.loads((paths.topic_cache / "manifest.json").read_text(encoding="utf-8"))
    ctx_meta = json.loads((paths.context_cache / "manifest.json").read_text(encoding="utf-8"))
    if (topic_meta["docs_sha256"] != sha256(data / "docs.parquet") or topic_meta["encoder"] != encoder.metadata()
            or ctx_meta["encoder"] != topic_meta["encoder"]):
        raise ValueError("Query, document-topic and fact encoders differ")
    scorer = SlotScorer(paths.model, schema, temperatures["T_query"],
                        "slot_first" if paths.kind == "zeroshot" else "context_first", device)
    scorer.values_visible = paths.values_visible
    if scorer.metadata() != conformal["scorer"]:
        raise ValueError("Conformal and query slot scorers differ")
    lex_meta = json.loads((paths.lexicon / "manifest.json").read_text(encoding="utf-8"))
    if (lex_meta["scorer"]["model"] != scorer.metadata()["model"]
            or lex_meta["scorer"]["temperature"] != temperatures["T_doc"]
            or lex_meta["scorer"]["pair_order"] != scorer.order
            or lex_meta["scorer"].get("values_visible", False) != scorer.values_visible):
        raise ValueError("Document scores and the current slot scorer/temperature differ")
    linking = config["linking"]
    linker = Linker(pd.read_parquet(data / "docs.parquet", columns=["doc_id", "name", "doc_type"]),
                    linking["max_ngram"], linking["min_name_chars"], linking["common_words"],
                    linking["fuzzy_threshold"], linking["fuzzy_mass"])
    ctx = PlannerContext(schema, linker, encoder, scorer, lex, settings, threshold)
    dim = config["embedding"]["topic_dim"]
    dense = DenseIndex(np.load(paths.topic_cache / f"topic_qwen3e06_{dim}.npy", allow_pickle=False))
    bm25 = BM25Index.cached(data / "docs.parquet", data / "bm25_prime-0.1.0", schema.version)
    score = config["scoring"]
    return Pipeline(ctx, index, bm25, dense,
                        ScoreConfig(score["mode"], score["combine"], score["a"], score["b"], score["precision"]),
                        score["lambda_B"], score["lambda_D"], score["lambda_L"], config["retrieval"]["final_k"],
                        score.get("hard_and", False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/prime_v0.yaml"))
    for key in ("facts", "text-keys", "context-cache", "topic-cache", "lexicon", "temperatures", "conformal"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--kind", choices=("ft", "zeroshot"))
    parser.add_argument("--values-visible", action="store_true", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Query limit must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    supplied = {f.name: getattr(args, f.name) for f in fields(RuntimePaths) if getattr(args, f.name) is not None}
    paths = RuntimePaths.from_mapping({**config.get("artifacts", {}), **supplied})
    pipeline = load_pipeline(config, paths, args.device)
    data = Path(config["paths"]["data"])
    index, lex, scorer = pipeline.index, pipeline.ctx.lexicon, pipeline.ctx.scorer
    queries_path = data / f"queries_{args.split}.jsonl"
    queries = list(read_jsonl(queries_path))[:args.limit]
    with logged_run(Path(config["paths"]["runs"]), f"m7-{args.split}", {
        "settings": config, "index_version": lex.version, "queries_sha256": sha256(queries_path),
        "split": args.split, "limit": args.limit, "human_reviews": "pending", "scorer": scorer.metadata(),
    }) as (run, metadata):
        evaluation, saved, costs, plans, explanations = Evaluation(index.doc_ids.tolist()), {}, [], [], []
        for n, query in enumerate(queries):
            measured = []
            for repeat in range(4):
                result = pipeline.search(query["query"], query["qid"])
                if repeat:
                    measured.append(result["counters"])
            cost = {k: float(np.median([r[k] for r in measured])) if k.endswith("_ms") else measured[-1][k]
                    for k in measured[-1]}
            evaluation.add(query["qid"], result["doc_ids"], query["answers"])
            saved[query["qid"]] = dict(zip(result["doc_ids"], result["scores"]))
            costs.append({"qid": query["qid"], **cost})
            plans.append({**result["plan"], "fusion_scores": dict(zip(result["doc_ids"], result["fusion_scores"]))})
            explanations.append({"qid": query["qid"], "evidence": result["evidence"]})
            if (n + 1) % 100 == 0:
                print(f"SlotSpace: {n + 1}/{len(queries)}", flush=True)
        metrics = evaluation.summary()
        write_json(run / "run.json", saved)
        write_json(run / "metrics.json", metrics)
        pq.write_table(pa.Table.from_pylist(evaluation.rows), run / "per_query.parquet")
        pq.write_table(pa.Table.from_pylist(costs), run / "counters.parquet")
        for name, rows in (("plans", plans), ("evidence", explanations)):
            (run / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        metadata["metrics"] = metrics
        print(json.dumps({"run": str(run), **metrics}), flush=True)


if __name__ == "__main__":
    main()

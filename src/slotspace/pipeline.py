"""Single-pass retrieval, fact binding, evidence, and validation-only execution."""

import argparse
from collections import defaultdict
from dataclasses import replace
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
from .planner import PlannerContext, PlannerSettings, plan_query, plan_readings
from .reference import Doc, Fact, ScoreConfig, TAU_MIN, binarize, fact_gate, ia_select, psi, sim
from .retrieve import BM25Index, DenseIndex, minmax, topk
from .schema import Schema


class CorpusIndex:
    def __init__(self, documents, dim: int, version, precision: str = "float"):
        self.store = FactStore(dim, precision)
        self.version = version
        self.texts, self.native_ids, postings, ids = {}, {}, defaultdict(list), set()

        def tracked():
            for doc in documents:
                number = int(doc.doc_id)
                if number < 0 or number in ids:
                    raise ValueError("Corpus index needs distinct nonnegative numeric document IDs")
                ids.add(number)
                self.texts[doc.doc_id] = [f for f in doc.facts if not f.native]
                keys = set()
                for fact in doc.facts:
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
                 lambda_l: float = 0.5, final_k: int = 100):
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
        lpos = topk(lexical, plan["budgets"]["K_L"])
        dpos = topk(dense, plan["budgets"]["K_D"])
        readings = plan_readings(plan) if self.weights[0] != 0 else []
        keys = {key for reading in readings for key in reading.keys}
        posting_ids, posting_work = [], 0
        for key in sorted(keys):
            slot, value = key.split("|", 1)
            rows = self.index.postings.get(key, np.empty(0, dtype=np.int64))
            if len(rows) != self.ctx.lexicon.hard_support(value, slot):
                raise ValueError(f"Kept posting is inconsistent with lexicon: {key}")
            posting_work += len(rows)
            posting_ids.append(rows)
        vpos = (np.searchsorted(self.index.doc_ids, np.unique(np.concatenate(posting_ids)))
                if posting_ids else np.empty(0, np.int64))
        if len(vpos) > plan["budgets"]["K_V"]:
            vpos = vpos[topk(dense[vpos], plan["budgets"]["K_V"])]
        candidates = np.union1d(np.union1d(lpos, dpos), vpos)
        retrieved = perf_counter()
        doc_ids = [str(d) for d in self.index.doc_ids[candidates]]
        matrix = (np.column_stack([self.index.store.scores(doc_ids, [replace(r, weight=1)], self.scoring)
                                   for r in readings]) if readings else np.empty((len(candidates), 0)))
        binding = matrix @ np.array([r.weight for r in readings])
        m = sum(bool(segment["readings"]) for segment in plan["segments"])
        lb, ld, ll = self.weights
        fused = lb * binding / max(m, 1) + ld * minmax(dense[candidates]) + ll * minmax(lexical[candidates])
        order = np.lexsort((candidates, -fused))[:self.final_k]
        ranking = [doc_ids[i] for i in order]
        if plan["weight_objective"] == "ia" and readings:
            probabilities = {f"s{i}:{r['key']}": r["p"] for i, segment in enumerate(plan["segments"])
                             for r in segment["readings"]}
            if np.any(matrix < -1e-5) or np.any(matrix > 1 + 1e-5):
                raise ValueError("IA received non-probabilistic reading matches")
            bounded = np.clip(matrix, 0, 1)
            coverage = {r.rid: {doc_ids[i]: float(bounded[i, j]) for i in order} for j, r in enumerate(readings)}
            ranking = ia_select(ranking, probabilities, coverage, len(ranking))
            row_by_doc = {doc: i for i, doc in enumerate(doc_ids)}
            order = np.array([row_by_doc[d] for d in ranking])
        bound_at = perf_counter()
        evidence = {d: [self.index.evidence(d, r, self.scoring) for r in readings] for d in ranking[:10]}
        scoring_work = 0
        for doc in doc_ids:
            for r in readings:
                if self.scoring.mode == "no_role":
                    row_groups = [self.index.store.value_rows.get(doc, {}).get(k.split("|", 1)[1], np.empty(0, int))
                                  for k in r.keys]
                    scoring_work += len(np.unique(np.concatenate(row_groups))) if row_groups else 0
                else:
                    present = [self.index.store.key_index.get(doc, {})[k][0]
                               for k in r.keys if k in self.index.store.key_index.get(doc, {})]
                    if self.scoring.mode == "unbound":
                        scoring_work += len(self.index.store.text_rows.get(doc, ())) if present else 0
                    else:
                        scoring_work += sum(map(len, present))
        evidence_work = sum(len(self.index.texts.get(d, ())) * len(readings) for d in ranking[:10])
        score_values = (list(map(float, range(len(ranking), 0, -1))) if plan["weight_objective"] == "ia" and readings
                        else fused[order].astype(float).tolist())
        return {
            "doc_ids": list(map(int, ranking)), "scores": score_values,
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
    vectors = np.load(context_cache / "text_fact_ctx_qwen3e06_128.npy", allow_pickle=False)
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

    index = CorpusIndex(documents(), 128, lex.version, precision)
    if not np.array_equal(index.doc_ids, ids):
        raise ValueError("Topic-vector rows require sorted document IDs")
    return index, lex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/prime_v0.yaml"))
    parser.add_argument("--facts", type=Path, default=Path("data/prime/facts_prime-0.1.0"))
    parser.add_argument("--text-keys", type=Path, required=True)
    parser.add_argument("--context-cache", type=Path, required=True)
    parser.add_argument("--topic-cache", type=Path, default=Path("data/prime/topic_prime-0.1.0_qwen3e06"))
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Query slot scorer checkpoint")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--temperatures", type=Path, required=True)
    parser.add_argument("--conformal", type=Path, required=True)
    parser.add_argument("--kind", choices=("ft", "zeroshot"), default="ft")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Query limit must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    schema_path, data = Path(config["schema"]), Path(config["paths"]["data"])
    schema, settings = Schema.load(schema_path), PlannerSettings.from_config(config)
    temperatures = json.loads(args.temperatures.read_text(encoding="utf-8"))
    conformal = json.loads(args.conformal.read_text(encoding="utf-8"))
    index, lex = load_index(data / "docs.parquet", args.facts, args.text_keys, args.context_cache,
                            args.lexicon, schema_path, config["scoring"]["precision"])
    if (conformal["lexicon_version"] != lex.version or conformal["schema_sha256"] != sha256(schema_path)
            or conformal["temperatures_sha256"] != sha256(args.temperatures)
            or conformal["beta"] != settings.beta or conformal["eps"] != settings.eps):
        raise ValueError("Conformal settings/model/index provenance mismatch")
    encoder = Embedder(args.embedding_model, config["embedding"]["max_seq_length"], args.device)
    topic_meta = json.loads((args.topic_cache / "manifest.json").read_text(encoding="utf-8"))
    ctx_meta = json.loads((args.context_cache / "manifest.json").read_text(encoding="utf-8"))
    if (topic_meta["docs_sha256"] != sha256(data / "docs.parquet") or topic_meta["encoder"] != encoder.metadata()
            or ctx_meta["encoder"] != topic_meta["encoder"]):
        raise ValueError("Query, document-topic and fact encoders differ")
    scorer = SlotScorer(args.model, schema, temperatures["T_query"],
                        "slot_first" if args.kind == "zeroshot" else "context_first", args.device)
    if scorer.metadata() != conformal["scorer"]:
        raise ValueError("Conformal and query slot scorers differ")
    lex_meta = json.loads((args.lexicon / "manifest.json").read_text(encoding="utf-8"))
    if (lex_meta["scorer"]["model"] != scorer.metadata()["model"]
            or lex_meta["scorer"]["temperature"] != temperatures["T_doc"]
            or lex_meta["scorer"]["pair_order"] != scorer.order):
        raise ValueError("Document scores and the current slot scorer/temperature differ")
    linking = config["linking"]
    linker = Linker(pd.read_parquet(data / "docs.parquet", columns=["doc_id", "name", "doc_type"]),
                    linking["max_ngram"], linking["min_name_chars"], linking["common_words"],
                    linking["fuzzy_threshold"], linking["fuzzy_mass"])
    ctx = PlannerContext(schema, linker,
                         encoder, scorer, lex, settings, conformal["thresholds"][str(config["planner"]["alpha"])])
    dim = config["embedding"]["topic_dim"]
    dense = DenseIndex(np.load(args.topic_cache / f"topic_qwen3e06_{dim}.npy", allow_pickle=False))
    bm25 = BM25Index.cached(data / "docs.parquet", data / "bm25_prime-0.1.0", schema.version)
    score = config["scoring"]
    pipeline = Pipeline(ctx, index, bm25, dense,
                        ScoreConfig(score["mode"], score["combine"], score["a"], score["b"], score["precision"]),
                        score["lambda_B"], score["lambda_D"], score["lambda_L"], config["retrieval"]["final_k"])
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

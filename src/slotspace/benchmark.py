"""M8 online baseline adapters and oracle/role-confusion diagnostics."""

from copy import deepcopy
from dataclasses import replace
import gc
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd

from .artifacts import sha256
from .pipeline import RuntimePaths, load_pipeline
from .planner import plan_query
from .reference import ia_select
from .representations import RepresentationIndex, field_scores
from .retrieve import BaselineRetriever, minmax, topk


def oracle_plan(plan: dict, segments: list[dict], lexicon) -> dict:
    if [s["span"] for s in plan["segments"]] != [[s["start"], s["end"]] for s in segments]:
        raise ValueError("Oracle silver segments do not match the query linker")
    result = deepcopy(plan)
    for target, gold in zip(result["segments"], segments):
        readings = []
        for key in sorted(gold["true_readings"]):
            slot, value = key.split("|", 1)
            if value not in target["links"] or lexicon.hard_support(value, slot) == 0:
                raise ValueError("Oracle label is unsupported or incompatible with the query")
            readings.append({"key": key, "w": 1 / len(gold["true_readings"]),
                             "p": 1 / len(gold["true_readings"]), "support_hard": lexicon.hard_support(value, slot)})
        target["readings"] = readings
    result["path"] = "both" if any(s["readings"] for s in result["segments"]) else "vector_first"
    result["weight_objective"] = "p"
    return result


def role_confusion(ranked: list[int], answers: list[int], partner_answers: list[int]) -> float:
    wrong_role = set(partner_answers) - set(answers)
    return len(set(ranked[:10]) & wrong_role) / 10


def rewrite(query: str, slot, value_name: str) -> str:
    phrase = "the " + slot.value_type.replace("_", " ")
    pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
    description = pattern.sub(lambda _: value_name, slot.description)
    if description == slot.description:
        description += f" Value: {value_name}."
    return f"{query} ({description})"


def reciprocal_fusion(runs: list[tuple[list[int], float]], k: int = 60):
    if k < 0:
        raise ValueError("RRF constant cannot be negative")
    scores = {}
    for ranked, weight in runs:
        if not np.isfinite(weight) or weight < 0 or len(ranked) != len(set(ranked)):
            raise ValueError("Invalid weighted reciprocal-rank input")
        for rank, doc in enumerate(ranked, 1):
            scores[doc] = scores.get(doc, 0) + weight / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d)), scores


class OnlineBackend:
    source = "runtime"

    def __init__(self, job: dict, device="cuda"):
        self.job = job
        self.config = job["config"]
        self.paths = RuntimePaths.from_mapping(job["artifacts"])
        self.pipeline = load_pipeline(self.config, self.paths, device)
        self.doc_ids = self.pipeline.index.doc_ids.tolist()
        self.schema, self.lexicon = self.pipeline.ctx.schema, self.pipeline.ctx.lexicon
        self.names = pd.read_parquet(Path(self.config["paths"]["data"]) / "docs.parquet",
                                    columns=["doc_id", "name"]).set_index("doc_id").name.to_dict()
        self.ll = self.config.get("baselines", {}).get("B2", {}).get("lambda_L", self.config["scoring"]["lambda_L"])
        self.rep, self.field_weights = None, None
        strategy = job["experiment"]["strategy"]
        if strategy in {"maxp", "fields"}:
            path = Path(job["auxiliary"]["passages" if strategy == "maxp" else "fields"])
            self.rep = RepresentationIndex(path, sha256(Path(self.config["paths"]["data"]) / "docs.parquet"),
                                            self.pipeline.ctx.embedder.metadata())
            if not np.array_equal(self.rep.doc_ids, self.pipeline.index.doc_ids):
                raise ValueError("Baseline representation row IDs differ from the frozen corpus")
            if strategy == "fields":
                weights = json.loads(Path(job["auxiliary"]["field_weights"]).read_text(encoding="utf-8"))
                if (weights["split"] != "val" or weights["complete"] is not True
                        or weights["representation_sha256"] != sha256(path / "manifest.json")
                        or weights["lambda_L"] != self.ll
                        or weights["validation_fingerprint"] != self.config["tuning_evidence"]["validation"]):
                    raise ValueError("Field baseline weights do not match frozen validation provenance")
                self.field_weights = weights["weights"]
        if job["experiment"]["id"] == "A11":
            lex_meta = json.loads((self.paths.lexicon / "manifest.json").read_text(encoding="utf-8"))
            files = [self.paths.facts / "native_keys.parquet", self.paths.facts / "text_facts.parquet",
                     self.paths.text_keys, self.paths.context_cache / "text_fact_ctx_qwen3e06_128.binary.npy",
                     self.paths.context_cache / "manifest.json", self.paths.lexicon / lex_meta["file"],
                     self.paths.lexicon / "manifest.json", Path(self.config["schema"])]
            self.storage_per_doc = sum(p.stat().st_size for p in files) / len(self.doc_ids)
        else:
            self.storage_per_doc = None

    def baseline(self, text: str, mode: str):
        start = perf_counter()
        topic = (self.pipeline.ctx.embedder.queries([text], dim=self.pipeline.dense.documents.shape[1])[0]
                 if mode != "bm25" else None)
        if mode in {"bm25", "dense", "hybrid"}:
            retriever = BaselineRetriever(self.pipeline.bm25, self.pipeline.dense,
                                         self.pipeline.ctx.cfg.K_L, self.pipeline.ctx.cfg.K_D, self.pipeline.final_k)
            positions, scores, costs = retriever.search(text, mode, topic, self.ll)
        else:
            lexical, postings = self.pipeline.bm25.scores(text)
            if mode == "maxp":
                q = self.pipeline.ctx.embedder._cut(topic[None], 256)[0]
                dense = self.rep.passage_scores(q)
            else:
                dense = self.pipeline.dense.scores(topic)
            pool = np.union1d(topk(lexical, self.pipeline.ctx.cfg.K_L), topk(dense, self.pipeline.ctx.cfg.K_D))
            if mode == "maxp":
                combined = minmax(dense[pool]) + self.ll * minmax(lexical[pool])
                dense_passes = 1
            elif mode == "fields":
                features, field_work = self.rep.field_features(
                    text, self.pipeline.ctx.embedder._cut(topic[None], 256)[0], pool)
                combined = field_scores(features, self.field_weights, self.rep.meta["fields"], self.ll)
                postings += field_work
                dense_passes = 1 + len(self.rep.meta["fields"])
            else:
                raise ValueError(f"Unknown online baseline strategy: {mode}")
            order = np.lexsort((pool, -combined))[:self.pipeline.final_k]
            positions, scores = pool[order], combined[order]
            costs = {"bm25_postings": postings, "dense_searches": dense_passes, "value_postings": 0,
                     "candidates": len(pool), "facts_scored": 0, "encoder_texts": 1}
        costs["latency_ms"] = 1000 * (perf_counter() - start)
        return {"doc_ids": self.pipeline.index.doc_ids[positions].tolist(), "scores": scores.tolist(),
                "plan": {"query_id": None, "path": "vector_first", "segments": [], "baseline": mode},
                "counters": costs, "evidence": {}}

    def search(self, text: str, qid, gold=None):
        strategy = self.job["experiment"]["strategy"]
        if strategy in {"bm25", "dense", "hybrid", "maxp", "fields"}:
            result = self.baseline(text, strategy)
            result["plan"]["query_id"] = qid
            return result
        if strategy != "fanout":
            if self.job["experiment"]["oracle"]:
                if self.job["dataset"] != "val" or gold is None:
                    raise ValueError("Oracle readings are available on validation only")
                start = perf_counter()
                plan = oracle_plan(plan_query(text, qid, self.pipeline.ctx), gold, self.lexicon)
                result = self.pipeline.execute_plan(text, plan)
                result["counters"]["latency_ms"] = 1000 * (perf_counter() - start)
                return result
            return self.pipeline.search(text, qid)
        start = perf_counter()
        plan = plan_query(text, qid, self.pipeline.ctx)
        rewritten, results = [], []
        for segment in plan["segments"]:
            for reading in segment["readings"]:
                slot, value = reading["key"].split("|", 1)
                query = rewrite(text, self.schema.slot(slot), self.names[int(value[1:])])
                result = self.baseline(query, "hybrid")
                rewritten.append((result["doc_ids"], reading["p"]))
                results.append(result)
        if not results:
            result = self.baseline(text, "hybrid")
            result["plan"] = plan
            result["counters"]["encoder_texts"] += plan["planner_encoder_texts"]
            result["counters"]["latency_ms"] = 1000 * (perf_counter() - start)
            return result
        ranked, scores = reciprocal_fusion(rewritten)
        keys = set().union(*(r["counters"].keys() for r in results))
        costs = {k: sum(r["counters"].get(k, 0) for r in results) for k in keys if not k.endswith("_ms")}
        costs["encoder_texts"] += plan["planner_encoder_texts"]
        costs["latency_ms"] = 1000 * (perf_counter() - start)
        ranked = ranked[:self.pipeline.final_k]
        return {"doc_ids": ranked, "scores": [scores[d] for d in ranked], "plan": plan,
                "counters": costs, "evidence": {}}

    def close(self):
        self.pipeline = None
        self.rep = None
        gc.collect()

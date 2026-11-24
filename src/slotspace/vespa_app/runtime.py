"""Use the existing planner and artifact checks, then send all readings in one request."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..artifacts import sha256
from ..benchmark import reciprocal_fusion, rewrite
from ..pipeline import RuntimePaths, load_index
from ..planner import plan_query
from ..retrieve import DenseIndex
from ..schema import Schema
from .application import profile_name
from .codec import unit_vector
from .feed import export_feed, verify_bundle
from .query import fuse_hits, vespa_body


def export_runtime(config: dict, out: Path):
    paths = RuntimePaths.from_mapping(config["artifacts"])
    data, schema_path = Path(config["paths"]["data"]), Path(config["schema"])
    if config["embedding"]["topic_dim"] != 256 or config["embedding"]["ctx_dim"] != 128:
        raise ValueError("Vespa v1 requires 256-d topic and 128-d context embeddings")
    index, _ = load_index(data / "docs.parquet", paths.facts, paths.text_keys, paths.context_cache,
                          paths.lexicon, schema_path, "binary")
    topic = json.loads((paths.topic_cache / "manifest.json").read_text(encoding="utf-8"))
    context = json.loads((paths.context_cache / "manifest.json").read_text(encoding="utf-8"))
    if (topic["docs_sha256"] != sha256(data / "docs.parquet") or topic["encoder"] != context["encoder"]
            or topic["schema_version"] != context["schema_version"]):
        raise ValueError("Engine feed topic/context sources are incompatible")
    topics = DenseIndex(np.load(paths.topic_cache / "topic_qwen3e06_256.npy", allow_pickle=False)).documents
    return export_feed(data / "docs.parquet", Schema.load(schema_path), index, topics, out, schema_path)


class EngineBackend:
    def __init__(self, pipeline, client, bundle: Path, docs: Path, schema_path: Path, *, hybrid_l=None):
        score = pipeline.scoring
        if (score.precision != "binary" or score.mode != "bound" or score.combine != "max"
                or pipeline.hard_and or pipeline.ctx.cfg.weight_objective not in {"p", "p_over_H"}
                or pipeline.dense.documents.shape[1] != 256 or pipeline.index.store.dim != 128):
            raise ValueError("Engine path requires binary, bound/max, 256/128 dimensions, without IA/hard-AND")
        profile_name(score.a, score.b)
        manifest = verify_bundle(bundle)
        expected_slots = {s.id: i for i, s in enumerate(pipeline.ctx.schema.slots)}
        if (manifest["index_version"] != pipeline.index.version
                or manifest["index_fingerprint"] != pipeline.index.fingerprint
                or manifest["schema_version"] != pipeline.ctx.schema.version
                or manifest["schema_sha256"] != sha256(schema_path)
                or manifest["slot_index"] != expected_slots
                or manifest["docs_sha256"] != sha256(docs)
                or manifest["documents"] != len(pipeline.index.doc_ids)
                or manifest["topic_sha256"] != hashlib.sha256(pipeline.dense.documents.tobytes()).hexdigest()):
            raise ValueError("Engine feed, planner and reference corpus identities differ")
        self.pipeline, self.client, self.manifest = pipeline, client, manifest
        self.slots = expected_slots
        self.names = pd.read_parquet(docs, columns=["doc_id", "name"]).set_index("doc_id").name.to_dict()
        self.hybrid_l = pipeline.weights[2] if hybrid_l is None else hybrid_l
        if not np.isfinite(self.hybrid_l) or self.hybrid_l < 0:
            raise ValueError("B2 lexical weight must be finite and nonnegative")

    def empty_plan(self, qid=None):
        cfg = self.pipeline.ctx.cfg
        return {"query_id": qid, "segments": [], "weight_objective": "p",
                "schema_version": self.pipeline.ctx.schema.version, "lexicon_version": self.pipeline.index.version,
                "budgets": {"K_L": cfg.K_L, "K_D": cfg.K_D, "K_V": cfg.K_V},
                "planner_encoder_texts": 0}

    def _execute(self, text, plan, weights):
        start = perf_counter()
        pipe = self.pipeline
        if (plan["schema_version"] != self.manifest["schema_version"]
                or plan["lexicon_version"] != self.manifest["index_version"]
                or pipe.ctx.lexicon.version != self.manifest["index_version"]):
            raise ValueError("Engine plan and index/lexicon snapshots differ")
        topic = unit_vector(pipe.ctx.embedder.queries([text], dim=256)[0], 256, "query topic")
        response = self.client.search(vespa_body(plan, text, topic, self.slots, pipe.scoring.a, pipe.scoring.b))
        result = fuse_hits(response["hits"], plan, weights, pipe.final_k)
        result["plan"] = plan
        result["counters"] = {
            "candidates": len(response["hits"]), "http_requests": 1, "encoder_texts": 1,
            "server_sum_ms": response["server_ms"], "server_max_ms": response["server_ms"],
            "rpc_sum_ms": response["rpc_ms"], "rpc_max_ms": response["rpc_ms"],
            "latency_ms": 1000 * (perf_counter() - start),
        }
        return result

    def search(self, text, qid, strategy="slotspace"):
        start = perf_counter()
        pipe = self.pipeline
        if pipe.ctx.lexicon.version != pipe.index.version:
            raise ValueError("Index/lexicon version changed before engine query planning")
        if strategy == "hybrid":
            return self._execute(text, self.empty_plan(qid), (0, 1, self.hybrid_l))
        if strategy not in {"slotspace", "fanout"}:
            raise ValueError("Unknown engine strategy")
        plan = plan_query(text, qid, pipe.ctx)
        planned = perf_counter()
        if strategy == "slotspace":
            result = self._execute(text, plan, pipe.weights)
        else:
            runs, partial = [], []
            for segment in plan["segments"]:
                for reading in segment["readings"]:
                    slot, value = reading["key"].split("|")
                    query = rewrite(text, pipe.ctx.schema.slot(slot), self.names[int(value[1:])])
                    response = self._execute(query, self.empty_plan(qid), (0, 1, self.hybrid_l))
                    partial.append(response)
                    runs.append((response["doc_ids"], reading["p"]))
            if not partial:
                result = self._execute(text, self.empty_plan(qid), (0, 1, self.hybrid_l))
            else:
                ranked, scores = reciprocal_fusion(runs)
                ranked = ranked[:pipe.final_k]
                sums = ("candidates", "http_requests", "encoder_texts", "server_sum_ms", "rpc_sum_ms")
                costs = {k: sum(p["counters"][k] for p in partial) for k in sums}
                for k in ("server_max_ms", "rpc_max_ms"):
                    costs[k] = max(p["counters"][k] for p in partial)
                result = {"doc_ids": ranked, "scores": [scores[d] for d in ranked], "counters": costs,
                          "candidates": []}
            result["plan"] = plan
        result["counters"]["encoder_texts"] += plan["planner_encoder_texts"]
        result["counters"]["planner_ms"] = 1000 * (planned - start)
        result["counters"]["latency_ms"] = 1000 * (perf_counter() - start)
        return result

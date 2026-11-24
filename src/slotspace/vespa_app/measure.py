"""Validation-only engine parity, quality and warm latency measurements."""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..artifacts import logged_run, read_jsonl, sha256, write_json
from ..evaluate import Evaluation
from ..pipeline import RuntimePaths, load_pipeline
from .parity import binary_document, compare_binding
from .runtime import EngineBackend


def container_stats(container: str | None):
    if container is None:
        return {"status": "NOT MEASURED", "reason": "No dedicated Docker container was supplied"}
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", container):
        raise ValueError("Invalid container ID/name")
    result = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}", container],
                            capture_output=True, text=True, timeout=30, check=True)
    stats = json.loads(result.stdout)
    if not stats.get("MemUsage"):
        raise ValueError("docker stats omitted memory usage")
    return {"status": "measured", "source": "docker stats", "scope": "whole container, not isolated index RAM",
            "values": stats}


def _percentiles(values):
    p50, p95 = np.percentile(values, [50, 95])
    return {"p50": float(p50), "p95": float(p95)}


def run_measurements(config: dict, bundle: Path, receipt: Path, client, runs: Path, *,
                     queries=1000, parity_queries=200, device="cuda", container=None, pipeline=None) -> Path:
    if type(queries) is not int or type(parity_queries) is not int or not 1 <= parity_queries <= queries:
        raise ValueError("Require 1 <= parity_queries <= validation query count")
    config = deepcopy(config)
    score = config["scoring"]
    score["precision"] = "binary"
    if score.get("mode") != "bound" or score.get("combine") != "max":
        raise ValueError("M9 compares the bound/max binary path, not v0 ablations")
    data = Path(config["paths"]["data"])
    query_path = data / "queries_val.jsonl"
    selected = sorted(read_jsonl(query_path), key=lambda q: q["qid"])
    if len(selected) < queries or len({q["qid"] for q in selected}) != len(selected):
        raise ValueError("Validation input has too few queries or duplicate query IDs")
    selected = selected[:queries]
    source = "real Vespa HTTP" if pipeline is None else "fixture"
    with logged_run(runs, "m9-val", {"settings": config, "queries_sha256": sha256(query_path),
                                    "query_ids": [q["qid"] for q in selected], "parity_queries": parity_queries,
                                    "split": "val", "source": source, "receipt_sha256": sha256(receipt),
                                    "H5": "not applicable; validation only", "seed": 13}) as (out, metadata):
        pipe = pipeline or load_pipeline(config, RuntimePaths.from_mapping(config["artifacts"]), device)
        backend = EngineBackend(pipe, client, bundle, data / "docs.parquet", Path(config["schema"]),
                                hybrid_l=config.get("baselines", {}).get("B2", {}).get("lambda_L", pipe.weights[2]))
        fed = json.loads(receipt.read_text(encoding="utf-8"))
        if (fed.get("status") != "ok" or fed["endpoint"] != client.endpoint
                or fed["feed_sha256"] != backend.manifest["feed_sha256"]
                or fed["application"] != backend.manifest["application"]
                or fed["documents"] != backend.manifest["documents"]):
            raise ValueError("Feed receipt does not identify this engine/corpus/application")
        metadata["engine_version"] = client.version()
        if fed["engine_version"] != metadata["engine_version"]:
            raise ValueError("Vespa service version changed since the feed receipt")
        client.wait_count(backend.manifest["documents"])
        metadata["feed"] = fed
        metadata["memory_before"] = container_stats(container)
        results, counters, per_query, plans, metrics, parity = {}, [], [], [], {}, []
        reference = lru_cache(maxsize=1024)(lambda number: binary_document(pipe.index, number))
        for mode in ("slotspace", "hybrid", "fanout", "research_binary"):
            evaluation = Evaluation(pipe.index.doc_ids.tolist())
            results[mode] = {}
            costs = []
            for i, query in enumerate(selected):
                fn = pipe.search if mode == "research_binary" else (
                    lambda text, qid: backend.search(text, qid, mode))
                fn(query["query"], query["qid"])  # one unmeasured warmup
                measured = [fn(query["query"], query["qid"]) for _ in range(3)]
                result = measured[0]
                evaluation.add(query["qid"], result["doc_ids"], query["answers"])
                results[mode][query["qid"]] = dict(zip(result["doc_ids"], result["scores"]))
                cost = {k: float(np.median([r["counters"][k] for r in measured])) if k.endswith("_ms")
                        else result["counters"][k] for k in result["counters"]}
                cost["ranking_stable"] = all(r["doc_ids"] == result["doc_ids"] for r in measured)
                costs.append(cost)
                counters.append({"mode": mode, "qid": query["qid"], **cost})
                plans.append({"mode": mode, "qid": query["qid"], "plan": result["plan"]})
                if mode == "slotspace" and i < parity_queries:
                    check = compare_binding(result["candidates"], result["plan"], reference,
                                            score["a"], score["b"])
                    parity.append({"qid": query["qid"], **check})
                if (i + 1) % 100 == 0:
                    print(f"M9 {mode}: {i + 1}/{queries} validation queries", flush=True)
            metrics[mode] = {**evaluation.summary(),
                             "client_ms": _percentiles([c["latency_ms"] for c in costs]),
                             "unstable_rankings": sum(not c["ranking_stable"] for c in costs)}
            if mode != "research_binary":
                for key in ("server_sum_ms", "server_max_ms", "rpc_sum_ms", "rpc_max_ms"):
                    metrics[mode][key] = _percentiles([c[key] for c in costs])
            per_query.extend({"mode": mode, **row} for row in evaluation.rows)
        write_json(out / "run.json", results)
        write_json(out / "metrics.json", metrics)
        write_json(out / "parity.json", {"queries": parity, "n_queries": len(parity),
                                       "max_abs_difference": max(p["max_abs_difference"] for p in parity),
                                       "status": "ok", "source": source,
                                       "meets_query_counts": len(parity) >= 200 and queries >= 1000})
        pq.write_table(pa.Table.from_pylist(per_query), out / "per_query.parquet")
        columns = sorted(set().union(*(row.keys() for row in counters)))
        pq.write_table(pa.table({key: [row.get(key) for row in counters] for key in columns}),
                       out / "counters.parquet")
        (out / "plans.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plans), encoding="utf-8")
        metadata["memory_after"] = container_stats(container)
    return out

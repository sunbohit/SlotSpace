"""M1 baseline runner. Only training/validation are enabled before H5 approval."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .artifacts import logged_run, sha256, write_json
from .embed import Embedder, topic_vectors
from .evaluate import Evaluation
from .retrieve import BM25Index, BaselineRetriever, DenseIndex


def run_baseline(config_path: Path, mode: str, *, split: str = "val", limit: int | None = None,
                 model: str | None = None, device: str = "cuda", lambda_l: float = 0.5,
                 data: Path | None = None, runs: Path | None = None) -> Path:
    if split not in {"train", "val"}:
        raise ValueError("Test/human evaluation is locked until H5 approval")
    if limit is not None and limit < 1:
        raise ValueError("Query limit must be positive")
    if mode not in {"bm25", "dense", "hybrid"}:
        raise ValueError(f"Unknown baseline: {mode}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["dataset"] != "prime" or config["seed"] != 13:
        raise ValueError("v0 requires Prime with seed 13")
    folder = data or Path(config["paths"]["data"])
    docs_path, queries_path = folder / "docs.parquet", folder / f"queries_{split}.jsonl"
    ids = pq.read_table(docs_path, columns=["doc_id"])["doc_id"].to_numpy()
    if not len(ids) or not np.all(ids[:-1] < ids[1:]):
        raise ValueError("Document IDs must be sorted and unique")
    with queries_path.open(encoding="utf-8") as src:
        queries = [json.loads(line) for line in src]
    if len({q["qid"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    queries = queries[:limit]
    if not queries:
        raise ValueError("Cannot run on an empty query set")
    np.random.seed(13)
    metadata = {"settings": config, "mode": mode, "split": split, "limit": limit,
                "lambda_D": 1, "lambda_L": lambda_l, "device": device if mode != "bm25" else "cpu",
                "docs_sha256": sha256(docs_path), "queries_sha256": sha256(queries_path),
                "model": model, "timing": "median of three warm repeats; encoding included"}
    root = runs or Path(config["paths"]["runs"])
    with logged_run(root, f"m1-{mode}-{split}", metadata) as (out, meta):
        bm25 = BM25Index.cached(docs_path, folder / "bm25_prime-0.1.0") if mode != "dense" else None
        encoder, dense = None, None
        embedding = config["embedding"]
        if mode != "bm25":
            import torch

            torch.manual_seed(13)
            encoder = Embedder(model or embedding["model"], embedding["max_seq_length"], device)
            caches = topic_vectors(docs_path, folder / "topic_prime-0.1.0_qwen3e06", encoder,
                                   embedding["batch_size"])
            dim = embedding["topic_dim"]
            if dim not in (256, 1024):
                raise ValueError("v0 topic vectors support dimensions 256 or 1024")
            dense = DenseIndex(np.load(caches[0 if dim == 1024 else 1], allow_pickle=False))
            meta["encoder"] = encoder.metadata()
        limits = config["retrieval"]
        retriever = BaselineRetriever(bm25, dense, limits["K_L"], limits["K_D"], limits["final_k"])
        ev = Evaluation([int(i) for i in ids])
        counters, plans, saved = [], [], {}
        for n, query in enumerate(queries):
            durations, encoding_times = [], []
            for repeat in range(4):  # warm once, then measure three full query passes
                start = perf_counter()
                vector = encoder.queries([query["query"]], embedding["topic_dim"])[0] if encoder else None
                encoded = perf_counter()
                positions, scores, cost = retriever.search(query["query"], mode, vector, lambda_l)
                if repeat:
                    durations.append(1000 * (perf_counter() - start))
                    encoding_times.append(1000 * (encoded - start))
            ranked = [int(i) for i in ids[positions]]
            qid = int(query["qid"])
            ev.add(qid, ranked, query["answers"])
            saved[qid] = {doc: float(score) for doc, score in zip(ranked, scores)}
            counters.append({"qid": qid, **cost, "latency_ms": float(np.median(durations)),
                             "encoding_ms": float(np.median(encoding_times))})
            plans.append({"qid": qid, "path": "vector_first", "baseline": mode, "readings": []})
            if (n + 1) % 100 == 0:
                print(f"{mode}: {n + 1}/{len(queries)} queries", flush=True)
        metrics = ev.summary()
        metrics["median_latency_ms"] = float(np.median([r["latency_ms"] for r in counters]))
        write_json(out / "run.json", saved)
        write_json(out / "metrics.json", metrics)
        pq.write_table(pa.Table.from_pylist(ev.rows), out / "per_query.parquet")
        pq.write_table(pa.Table.from_pylist(counters), out / "counters.parquet")
        (out / "plans.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plans), encoding="utf-8")
        meta["n_queries"] = len(queries)
        print(json.dumps({"run": str(out), **metrics}, sort_keys=True), flush=True)
    return out


def baseline_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog="M8 commands: plan, run, report. Use COMMAND --help.")
    parser.add_argument("--config", type=Path, default=Path("configs/prime_v0.yaml"))
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", help="Local model directory or Hub ID")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambda-L", dest="weights", type=float, nargs="+", default=[0.5],
                        choices=(0, 0.25, 0.5, 1))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--runs", type=Path)
    args = parser.parse_args(argv)
    if len(args.weights) > 1 and (args.mode != "hybrid" or args.split != "val"):
        parser.error("The fusion-weight grid is validation-only and requires hybrid mode")
    results = []
    for weight in args.weights:
        out = run_baseline(args.config, args.mode, split=args.split, limit=args.limit,
                           model=args.model, device=args.device, lambda_l=weight,
                           data=args.data, runs=args.runs)
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        results.append((metrics["official"]["mrr"], -weight, out))
    if len(results) > 1:
        best = max(results, key=lambda x: (x[0], x[1]))
        print(json.dumps({"selected_lambda_L": -best[1], "criterion": "val MRR", "run": str(best[2])}))


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "pilot":
        from .pilot import main as pilot_main

        return pilot_main(argv[1:])
    if not argv or argv[0] not in {"plan", "run", "report"}:
        return baseline_main(argv[1:] if argv and argv[0] == "baseline" else argv)
    parser = argparse.ArgumentParser(description="Frozen M8 experiment planning, authorization and reports")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", type=Path)
    plan.add_argument("--draft", action="store_true")
    plan.add_argument("--variants", type=Path)
    plan.add_argument("--auxiliary", type=Path)
    plan.add_argument("--only", nargs="+")
    plan.add_argument("--out", type=Path, required=True, help="Use an ignored artifact directory such as runs/")
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--approval", type=Path, required=True, help="Externally written H5 JSON; keep it under runs/")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    report = commands.add_parser("report")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--execution", type=Path)
    report.add_argument("--out", type=Path, default=Path("runs/report_v0.md"))
    args = parser.parse_args(argv)
    if args.command == "plan":
        from .protocol import build_plan

        config = args.config or Path("configs/prime_v0.yaml" if args.draft else "configs/prime_v0_frozen.yaml")
        variants = yaml.safe_load(args.variants.read_text(encoding="utf-8")) if args.variants else {}
        auxiliary = yaml.safe_load(args.auxiliary.read_text(encoding="utf-8")) if args.auxiliary else {}
        variants, auxiliary = variants.get("variants", variants), auxiliary.get("auxiliary", auxiliary)
        result = build_plan(config, args.out, draft=args.draft, variants=variants, auxiliary=auxiliary, only=args.only)
        print(json.dumps({"jobs": len(result["jobs"]), "draft": result["draft"],
                          "missing_prerequisites": sum(bool(j["missing"]) for j in result["jobs"])}))
    elif args.command == "run":
        from .experiment_run import execute

        result = execute(args.plan, args.approval, args.out, device=args.device)
        print(json.dumps({"completed_job_records": len(result["jobs"])}))
    else:
        from .reporting import generate_report

        result = generate_report(args.plan, args.execution, args.out)
        print(json.dumps({"completed_rows": result["completed_rows"], "draft": result["draft"]}))


if __name__ == "__main__":
    main()

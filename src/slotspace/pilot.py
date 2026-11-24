"""Incremental real-model CPU/GPU pilots; no final-test execution or production freezing."""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .artifacts import logged_run, new_directory, read_jsonl, sha256, write_json
from .benchmark import reciprocal_fusion, rewrite
from .embed import Embedder, context_vectors, topic_vectors
from .evaluate import Evaluation
from .feature_cache import digest, replay
from .lexicon_build import native_by_doc
from .linker import Linker
from .pilot_data import code_identity, prepare, source_identity, verify_previous
from .pipeline import CorpusIndex, Pipeline
from .planner import PlannerContext, PlannerSettings, conformal_value, plan_query
from .protocol import artifact_fingerprint
from .reference import Doc, Fact, Lexicon, ScoreConfig
from .retrieve import BM25Index, BaselineRetriever, DenseIndex
from .schema import Schema
from .slot_scorer import KEY_SCHEMA, SlotScorer, score_fact_batch
from .tuning import FeatureSet, build_feature_set, tune


def manifest(bundle: Path):
    info = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if info["purpose"] != "pilot" or info["production"] or info["code_identity"] != code_identity():
        raise ValueError("Pilot source code/purpose changed after preparation")
    if source_identity(Path(info["source"]), Path(info["schema"])) != info["source_identity"]:
        raise ValueError("Full source dictionary, facts or labels changed after pilot preparation")
    for name, expected in info["files"].items():
        if sha256(bundle / name) != expected:
            raise ValueError(f"Prepared pilot artifact changed: {name}")
    return info


def document_scores(bundle, scorer):
    out = bundle / "cache" / "scores"
    expected = {"purpose": "pilot", "scorer": scorer.metadata(), "docs_sha256": sha256(bundle / "docs.parquet"),
                "facts_sha256": sha256(bundle / "text_facts.parquet")}
    if out.exists():
        meta = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if meta["settings"] != expected or meta["keys_sha256"] != sha256(out / "keys.parquet"):
            raise ValueError("Pilot role-score cache is stale or corrupt")
    else:
        frame = pd.read_parquet(bundle / "docs.parquet", columns=["doc_id", "doc_type"])
        kinds = frame.set_index("doc_id").doc_type.to_dict()
        with new_directory(out) as stage:
            with pq.ParquetWriter(stage / "keys.parquet", KEY_SCHEMA, compression="zstd") as writer:
                for batch in pq.ParquetFile(bundle / "text_facts.parquet").iter_batches(batch_size=8):
                    cells = score_fact_batch(batch.to_pylist(), kinds, scorer)
                    if cells:
                        writer.write_table(pa.Table.from_pylist(cells, schema=KEY_SCHEMA))
                    print(f"Pilot document scoring: {batch['row'][-1].as_py() + 1} facts", flush=True)
            write_json(stage / "manifest.json", {"settings": expected, "keys_sha256": sha256(stage / "keys.parquet")})
    return pd.read_parquet(out / "keys.parquet")


def load_pilot(bundle: Path, info: dict, encoder, scorer):
    schema = scorer.schema
    config = deepcopy(info["settings"])
    config["dataset"] = "prime_pilot"
    config["schema"] = info["schema"]
    config["paths"] = {"data": str(bundle), "runs": str(bundle / "runs")}
    config["embedding"].update(max_seq_length=info["budget"]["max_seq_length"], batch_size=4)
    config["planner"].update(use_pacc=False)
    config["retrieval"].update(K_L=min(32, info["documents"]), K_D=min(32, info["documents"]),
                               K_V=min(500, info["documents"]), final_k=100)
    config["tuning_grid"] = deepcopy(info["schedule"]["tuning_grid"])
    docs = pd.read_parquet(bundle / "docs.parquet")
    dictionary = pd.read_parquet(Path(info["source"]) / "docs.parquet", columns=["doc_id", "name", "doc_type"])
    linking = config["linking"]
    linker = Linker(dictionary, linking["max_ngram"], linking["min_name_chars"], linking["common_words"],
                    linking["fuzzy_threshold"], linking["fuzzy_mass"])
    top = topic_vectors(bundle / "docs.parquet", bundle / "cache" / "topic", encoder, 4, schema.version)
    ctx_path = context_vectors(bundle / "text_facts.parquet", bundle / "cache" / "context", encoder, 4, schema.version)
    ctx = np.load(ctx_path, allow_pickle=False).astype(np.float32)
    norms = np.linalg.norm(ctx, axis=1, keepdims=True)
    if not np.isfinite(ctx).all() or not np.allclose(norms, 1, atol=1e-3):
        raise ValueError("Pilot context cache is not aligned/unit-normalized")
    ctx /= norms
    keys = document_scores(bundle, scorer)
    facts = pd.read_parquet(bundle / "text_facts.parquet", columns=["row", "fact_id", "doc_id"])
    if ctx.shape != (len(facts), 128) or not np.array_equal(facts.row, np.arange(len(facts))):
        raise ValueError("Pilot context rows differ from the fact table")
    by_row = {int(i): dict(zip(g.key, g.pi)) for i, g in keys.groupby("row", sort=False)}
    native = native_by_doc(bundle / "native_keys.parquet")
    text = {}
    for row, fid, doc in facts.itertuples(index=False):
        text.setdefault(int(doc), []).append(Fact(fid, by_row.get(row, {}), ctx[row]))
    corpus = [Doc(str(i), [Fact(f"{i}.n.{k}", {k: 1.0}, None, True) for k in sorted(native.get(i, ()))]
              + text.get(int(i), [])) for i in docs.doc_id]
    lex = Lexicon()
    for doc in corpus:
        lex.upsert(doc)
    index = CorpusIndex(corpus, 128, lex.version)
    settings = PlannerSettings.from_config(config)
    context = PlannerContext(schema, linker, encoder, scorer, lex, settings, 0.0)
    pipeline = Pipeline(context, index, BM25Index(docs.text.tolist()),
                        DenseIndex(np.load(top[1], allow_pickle=False)))
    identity = {"purpose": "pilot", "source_identity": info["source_identity"],
                "schema_sha256": sha256(Path(info["schema"])), "docs_sha256": sha256(bundle / "docs.parquet"),
                "index_fingerprint": index.fingerprint, "encoder": encoder.metadata(), "scorer": scorer.metadata(),
                "artifacts": {}, "conformal_identity": {"schema_version": schema.version,
                    "schema_sha256": sha256(Path(info["schema"])), "lexicon_version": lex.version,
                    "scorer": scorer.metadata(), "temperatures_sha256": "pilot-fixed-temperature-1"}}
    return pipeline, config, identity


def features(bundle, pipe, config, identity):
    caches = {}
    scorings = [replace(pipe.scoring, a=a, b=b) for a in config["tuning_grid"]["a"] for b in config["tuning_grid"]["b"]]
    for split in ("calibration", "val"):
        path = bundle / "cache" / f"features_{split}"
        if not path.exists():
            queries = list(read_jsonl(bundle / f"queries_{split}.jsonl"))
            labels = {q["qid"]: q["segments"] for q in read_jsonl(bundle / f"silver_{split}.jsonl")}
            build_feature_set(pipe, queries, labels, path, split, identity, scorings, complete=False, purpose="pilot")
        cache = FeatureSet(path)
        if cache.meta["identity"] != identity or cache.meta["purpose"] != "pilot":
            raise ValueError("Pilot feature cache identity differs")
        caches[split] = cache
    out = bundle / "cache" / "tuning"
    if not out.exists():
        selection = tune(caches["calibration"], caches["val"], pipe.ctx.schema, config, out)
    else:
        selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
        if (selection["purpose"] != "pilot" or selection["calibration_fingerprint"] != caches["calibration"].meta["fingerprint"]
                or selection["validation_fingerprint"] != caches["val"].meta["fingerprint"]
                or json.loads((out / "meta.json").read_text(encoding="utf-8"))["config"]["settings"] != config):
            raise ValueError("Pilot tuning cache is stale")
    return caches, selection


def baseline(pipe, text, qid, mode, lexical_weight):
    start = perf_counter()
    vector = pipe.ctx.embedder.queries([text], dim=256)[0] if mode != "bm25" else None
    positions, scores, counters = BaselineRetriever(pipe.bm25, pipe.dense, pipe.ctx.cfg.K_L,
        pipe.ctx.cfg.K_D, pipe.final_k).search(text, mode, vector, lexical_weight)
    counters.update(latency_ms=1000 * (perf_counter() - start), encoder_texts=int(mode != "bm25"))
    return {"doc_ids": pipe.index.doc_ids[positions].tolist(), "scores": scores.tolist(), "counters": counters,
            "plan": {"query_id": qid, "segments": [], "baseline": mode}, "evidence": {}}


def fanout(pipe, text, qid, lexical_weight):
    start = perf_counter()
    plan = plan_query(text, qid, pipe.ctx)
    partial, weighted = [], []
    for segment in plan["segments"]:
        for reading in segment["readings"]:
            slot, value = reading["key"].split("|")
            rewritten = rewrite(text, pipe.ctx.schema.slot(slot), pipe.ctx.linker.name_by_id[value])
            result = baseline(pipe, rewritten, qid, "hybrid", lexical_weight)
            partial.append(result)
            weighted.append((result["doc_ids"], reading["p"]))
    if not partial:
        result = baseline(pipe, text, qid, "hybrid", lexical_weight)
    else:
        ranked, scores = reciprocal_fusion(weighted)
        ranked = ranked[:pipe.final_k]
        keys = set().union(*(r["counters"].keys() for r in partial))
        costs = {k: sum(r["counters"].get(k, 0) for r in partial) for k in keys if not k.endswith("_ms")}
        result = {"doc_ids": ranked, "scores": [scores[d] for d in ranked], "counters": costs, "evidence": {}}
    result["plan"] = plan
    result["counters"]["encoder_texts"] += plan["planner_encoder_texts"]
    result["counters"]["latency_ms"] = 1000 * (perf_counter() - start)
    return result


def measure(pipe, queries, caches, selected, bank, methods, repeats):
    settings = PlannerSettings.from_config(selected)
    threshold = conformal_value(bank, settings, selected["planner"]["alpha"])
    score = selected["scoring"]
    pipe = Pipeline(replace(pipe.ctx, cfg=settings, t_alpha=threshold), pipe.index, pipe.bm25, pipe.dense,
                    ScoreConfig(score["mode"], score["combine"], score["a"], score["b"], score["precision"]),
                    score["lambda_B"], score["lambda_D"], score["lambda_L"])
    ll = selected["baselines"]["B2"]["lambda_L"]
    lookup = {feature.payload["qid"]: feature for feature in caches["val"]}
    results, metrics, costs, rows, plans, evidence = {}, {}, [], [], [], []
    parity = []
    for method in methods:
        evaluation = Evaluation(pipe.index.doc_ids.tolist())
        results[method] = {}
        for query in queries:
            def search():
                if method in {"bm25", "dense", "hybrid"}:
                    return baseline(pipe, query["query"], query["qid"], method, ll)
                if method == "fanout":
                    return fanout(pipe, query["query"], query["qid"], ll)
                active = pipe if method == "bound" else Pipeline(
                    pipe.ctx, pipe.index, pipe.bm25, pipe.dense, replace(pipe.scoring, mode="unbound"),
                    *pipe.weights)
                return active.search(query["query"], query["qid"])
            search()
            measured = [search() for _ in range(repeats)]
            result = measured[0]
            evaluation.add(query["qid"], result["doc_ids"], query["answers"])
            results[method][query["qid"]] = dict(zip(result["doc_ids"], result["scores"]))
            costs.append({"method": method, "qid": query["qid"],
                          **{k: float(np.median([r["counters"][k] for r in measured])) if k.endswith("_ms")
                             else result["counters"][k] for k in result["counters"]},
                          "ranking_stable": all(r["doc_ids"] == result["doc_ids"] for r in measured)})
            plans.append({"method": method, "qid": query["qid"], "plan": result["plan"]})
            evidence.append({"method": method, "qid": query["qid"], "evidence": result.get("evidence", {})})
            if method == "bound":
                replayed = replay(lookup[query["qid"]], pipe.ctx.schema, settings, threshold, pipe.scoring,
                                  pipe.weights, pipe.final_k)
                if result["doc_ids"] != replayed["doc_ids"] or not np.allclose(result["scores"], replayed["scores"], atol=1e-5):
                    raise ValueError("Pilot live/replay ranking or scores disagree")
                parity.append({"qid": query["qid"], "max_abs": float(np.max(np.abs(
                    np.array(result["scores"]) - replayed["scores"]))), "ranking_equal": True})
        metrics[method] = evaluation.summary()
        rows.extend({"method": method, **row} for row in evaluation.rows)
        print(f"Pilot online {method}: {len(queries)} validation queries, {repeats} measured repeat(s)", flush=True)
    return results, metrics, costs, rows, plans, evidence, parity


def run(bundle, encoder_path, scorer_path, device="cpu", threads=4, resume=False, methods=None):
    if (bundle / "completed.json").exists():
        raise FileExistsError("Pilot stage already completed; regenerate its report or explicitly prepare the next stage")
    if (bundle / "cache").exists() and not resume:
        raise FileExistsError("Pilot cache exists; use --resume to validate it, not silently rebuild it")
    info = manifest(bundle)
    methods = methods or info["schedule"]["methods"]
    if not methods or len(methods) != len(set(methods)) or set(methods) - {"bm25", "dense", "hybrid", "bound", "unbound", "fanout"}:
        raise ValueError("Unknown/duplicate pilot methods")
    if "bound" not in methods or "hybrid" not in methods:
        raise ValueError("Pilot engineering checks need bound and hybrid")
    import torch

    if threads < 1:
        raise ValueError("CPU thread budget must be positive")
    torch.set_num_threads(threads)
    torch.manual_seed(13)
    np.random.seed(13)
    model_assets = {"encoder": artifact_fingerprint(str(encoder_path)), "scorer": artifact_fingerprint(str(scorer_path))}
    if any(v is None for v in model_assets.values()):
        raise ValueError("Pilot inference needs complete local model snapshots")
    previous = info.get("parent_result_sha256")
    if previous is not None:
        parent_path = Path(info["previous"])
        if sha256(parent_path / "completed.json") != previous:
            raise ValueError("Previous-stage authorization changed after preparation")
        parent = verify_previous(parent_path)
        if parent["model_assets"] != model_assets:
            raise ValueError("Stage promotion must retain the same encoder/scorer snapshots")
    identity_path = bundle / "cache" / "models.json"
    cache_settings = {"assets": model_assets, "device": device, "threads": threads, "torch": torch.__version__,
                      "max_seq_length": info["budget"]["max_seq_length"]}
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != cache_settings:
            raise ValueError("Pilot cached inference/model/runtime settings changed")
    else:
        if identity_path.parent.exists() and any(identity_path.parent.iterdir()):
            raise ValueError("Existing pilot caches have no owned model/runtime identity")
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, cache_settings)
    with logged_run(bundle / "runs", "pilot", {"purpose": "pilot", "stage": info["stage"], "budget": info["budget"],
        "device": device, "threads": threads, "encoder_path": str(encoder_path), "scorer_path": str(scorer_path),
        "methods": methods, "interpretation": "answer-closed diagnostic subset; not a full benchmark",
        "scorer": "BGE zero-shot, T=1; no main-model fine-tuning or PACC calibration"}) as (out, metadata):
        schema = Schema.load(info["schema"])
        architectures = json.loads((Path(scorer_path) / "config.json").read_text(encoding="utf-8"))["architectures"]
        if not any(a.endswith("ForSequenceClassification") for a in architectures):
            raise ValueError("Pilot needs an existing reranker head, not an untrained masked-language-model base")
        encoder = Embedder(str(encoder_path), info["budget"]["max_seq_length"], device)
        scorer = SlotScorer(str(scorer_path), schema, 1.0, "slot_first", device, batch_size=8)
        pipe, config, identity = load_pilot(bundle, info, encoder, scorer)
        caches, selection = features(bundle, pipe, config, identity)
        bank = json.loads((bundle / "cache" / "tuning" / "conformal_bank.json").read_text(encoding="utf-8"))
        queries = list(read_jsonl(bundle / "queries_val.jsonl"))
        data = measure(pipe, queries, caches, selection["config"], bank, methods, info["budget"]["repeats"])
        result, metrics, costs, per_query, plans, explanations, parity = data
        write_json(out / "run.json", result)
        write_json(out / "metrics.json", metrics)
        write_json(out / "parity.json", parity)
        write_json(out / "pilot.json", {"purpose": "pilot", "stage": info["stage"], "budget": info["budget"],
            "source_identity": info["source_identity"], "model_identity": identity, "selection": selection,
            "dictionary": info["dictionary"], "sampling": info["selection"], "formal_acceptance": False,
            "data_summary": {"calibration_qids": info["selected_qids"]["calibration"],
                             "validation_qids": info["selected_qids"]["val"], "text_facts": info["text_facts"],
                             "source_text_facts_in_selected_documents": info["source_text_facts_in_selected_documents"]}})
        pq.write_table(pa.Table.from_pylist(per_query), out / "per_query.parquet")
        columns = sorted(set().union(*(row.keys() for row in costs)))
        pq.write_table(pa.table({k: [r.get(k) for r in costs] for k in columns}), out / "counters.parquet")
        for name, values in (("plans", plans), ("evidence", explanations)):
            (out / f"{name}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")
        from .reporting import generate_pilot_report

        metadata["engineering_checks"] = {"live_replay_parity": len(parity) == len(queries),
            "all_queries_measured": all(v["n_queries"] == len(queries) for v in metrics.values()),
            "finite_latencies": all(np.isfinite(c["latency_ms"]) and c["latency_ms"] >= 0 for c in costs)}
    if not all(metadata["engineering_checks"].values()):
        raise ValueError("Pilot engineering checks did not pass")
    generate_pilot_report(out, out / "report.md")
    files = {p.relative_to(bundle).as_posix(): sha256(p) for p in bundle.rglob("*") if p.is_file()}
    write_json(bundle / "completed.json", {"purpose": "pilot", "status": "ok", "results": str(out.resolve()),
        "outputs": files, "engineering_checks": metadata["engineering_checks"], "model_identity": identity,
        "model_assets": model_assets, "production_accepted": False})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, default=Path("data/prime"))
    prep.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
    prep.add_argument("--schedule", type=Path, default=Path("configs/pilot_stages.yaml"))
    prep.add_argument("--stage", default="smoke")
    prep.add_argument("--previous", type=Path)
    prep.add_argument("--out", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--bundle", type=Path, required=True)
    execute.add_argument("--encoder", type=Path, default=Path("../Resource/models/Qwen/Qwen3-Embedding-0.6B"))
    execute.add_argument("--scorer", type=Path, default=Path("../Resource/models/BAAI/bge-reranker-v2-m3"))
    execute.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    execute.add_argument("--threads", type=int, default=4)
    execute.add_argument("--resume", action="store_true")
    execute.add_argument("--methods", nargs="+")
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        print(prepare(args.source.resolve(), args.schema.resolve(),
            yaml.safe_load(args.schedule.read_text(encoding="utf-8")), args.stage, args.out.resolve(), args.previous))
    elif args.action == "run":
        print(run(args.bundle.resolve(), args.encoder.resolve(), args.scorer.resolve(),
                  args.device, args.threads, args.resume, args.methods))
    else:
        from .reporting import generate_pilot_report

        generate_pilot_report(args.run, args.out)


if __name__ == "__main__":
    main()

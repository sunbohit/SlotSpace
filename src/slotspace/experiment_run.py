"""H5-controlled one-pass experiment execution; reportable outputs are never fabricated."""

import json
from contextlib import contextmanager
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import logged_run, read_jsonl, sha256, write_json
from .benchmark import OnlineBackend, oracle_plan, role_confusion
from .evaluate import Evaluation, per_query_metrics
from .protocol import authorize
from .stats import bootstrap_ci


def _silver(job):
    data = Path(job["config"]["paths"]["data"])
    folder = Path(job["config"]["paths"].get("silver", str(data / "silver_prime-0.1.0")))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"] or manifest["query_hashes"]["val"] != sha256(data / "queries_val.jsonl"):
        raise ValueError("Validation diagnostics require the matching complete silver labels")
    return {row["qid"]: row["segments"] for row in read_jsonl(folder / "silver_val.jsonl")}


def _planner_row(result, query, labels, backend):
    if labels is None or not result["plan"].get("segments"):
        return None
    plan = result["plan"]
    gold = labels[query["qid"]]
    reference = oracle_plan(plan, gold, backend.lexicon)
    if any(s["true_readings"] for s in gold):
        oracle = backend.pipeline.execute_plan(query["query"], reference)
        regret = (per_query_metrics(oracle["doc_ids"], set(query["answers"]))["mrr"]
                  - per_query_metrics(result["doc_ids"], set(query["answers"]))["mrr"])
    else:
        regret = None
    misses, confidence, before, correct, heldout = [], [], [], [], []
    chosen_slot = backend.config["slot_scorer"]["heldout_slot"]
    for segment, truth in zip(plan["segments"], gold):
        if truth["true_readings"]:
            misses.append(not bool(set(truth["true_readings"]) & {r["key"] for r in segment["readings"]}))
        distribution = segment.get("distribution")
        if distribution is not None:
            target = {key.split("|", 1)[0] for key in truth["true_readings"]}
            target = target or {k for k in distribution if k.startswith("none.")}
            winner = max(distribution, key=distribution.get)
            confidence.append(distribution[winner])
            t = backend.pipeline.ctx.scorer.T
            if min(distribution.values()) > 0 and not backend.pipeline.ctx.cfg.uniform_slots:
                from scipy.special import softmax

                before.append(float(softmax(np.log(list(distribution.values())) * t).max()))
            else:
                before.append(None)
            correct.append(winner in target)
            if chosen_slot in target:
                heldout.append(winner in target)
    return {"qid": query["qid"], "plan_regret": regret, "misses": misses, "confidence": confidence,
            "confidence_before": before,
            "correct": correct, "heldout_correct": heldout}


@contextmanager
def _job_status(root, state, job_id):
    status = "failed"
    try:
        yield
        status = "ok"
    finally:
        if job_id in state["jobs"]:
            state["jobs"][job_id]["status"] = status
            write_json(root / "execution.json", state)


def execute(plan_path: Path, approval_path: Path, out: Path, *, device="cuda", backend_factory=None):
    plan = authorize(plan_path, approval_path)
    if out.exists():
        raise FileExistsError("This frozen experiment execution already exists; no implicit final-test reruns")
    out.mkdir(parents=True)
    factory = backend_factory or (lambda job: OnlineBackend(job, device))
    state = {"plan_sha256": sha256(plan_path), "frozen_sha256": plan["frozen_sha256"],
             "approval_sha256": sha256(approval_path), "jobs": {}, "references": {}}
    if plan.get("references"):
        (out / "references").mkdir()
        for key, value in plan["references"].items():
            source = Path(value)
            target = out / "references" / (key + source.suffix)
            shutil.copyfile(source, target)
            state["references"][key] = {"path": str(target.resolve()), "sha256": sha256(target)}
    write_json(out / "execution.json", state)
    for job in plan["jobs"]:
        if "alias_of" in job:
            if job["alias_of"] not in state["jobs"] or state["jobs"][job["alias_of"]]["status"] != "ok":
                raise ValueError("Experiment aliases must point to an earlier completed job")
            state["jobs"][job["id"]] = {**state["jobs"][job["alias_of"]], "alias_of": job["alias_of"]}
            write_json(out / "execution.json", state)
            continue
        dataset = plan["datasets"][job["dataset"]]
        if sha256(Path(dataset["path"])) != dataset["sha256"]:
            raise ValueError("Benchmark input changed before execution")
        # Authorization precedes parsing any final evaluation queries.
        queries = list(read_jsonl(Path(dataset["path"])))
        if not queries or len({q["qid"] for q in queries}) != len(queries):
            raise ValueError("Empty or duplicate experiment query IDs")
        if job["experiment"]["oracle"] and job["dataset"] != "val":
            raise ValueError("Oracle evaluation is validation-only")
        config = {"protocol_job": job["id"], "dataset": job["dataset"],
                  "experiment": job["experiment"], "frozen_sha256": plan["frozen_sha256"],
                  "plan_sha256": sha256(plan_path), "queries_sha256": dataset["sha256"],
                  "settings": job["config"], "H5": "explicit external approval", "seed": 13}
        with _job_status(out, state, job["id"]), logged_run(out, job["id"], config) as (run, metadata):
            state["jobs"][job["id"]] = {"path": str(run.resolve()), "status": "running"}
            write_json(out / "execution.json", state)
            backend = factory(job)
            try:
                metadata["measurement_source"] = backend.source
                evaluation, saved, costs, plans, evidence, diagnostics = Evaluation(backend.doc_ids), {}, [], [], [], []
                labels = _silver(job) if job["dataset"] == "val" and job["experiment"]["strategy"] == "slotspace" else None
                for n, query in enumerate(queries):
                    measured = []
                    for repeat in range(4):
                        result = backend.search(query["query"], query["qid"],
                                                labels[query["qid"]] if job["experiment"]["oracle"] else None)
                        if repeat:
                            measured.append(result)
                    result = measured[0]
                    if len(result["scores"]) != len(result["doc_ids"]) or not np.isfinite(result["scores"]).all():
                        raise ValueError("Experiment backend returned invalid scores")
                    evaluation.add(query["qid"], result["doc_ids"], query["answers"])
                    row = evaluation.rows[-1]
                    row["plan_regret"] = None
                    if job["dataset"] == "role_swap_test":
                        row["role_confusion@10"] = role_confusion(result["doc_ids"], query["answers"], query["partner_answers"])
                    cost = {key: float(np.median([r["counters"][key] for r in measured]))
                            if key.endswith("_ms") else measured[0]["counters"][key] for key in result["counters"]}
                    cost["ranking_stable"] = all(r["doc_ids"] == result["doc_ids"] for r in measured)
                    saved[query["qid"]] = dict(zip(result["doc_ids"], result["scores"]))
                    costs.append({"qid": query["qid"], **cost})
                    plans.append(result["plan"])
                    evidence.append({"qid": query["qid"], "evidence": result.get("evidence", {})})
                    diag = _planner_row(result, query, labels, backend) if labels is not None else None
                    if diag is not None:
                        diagnostics.append(diag)
                        row["plan_regret"] = diag["plan_regret"]
                    if (n + 1) % 100 == 0:
                        print(f"{job['id']}: {n + 1}/{len(queries)}", flush=True)
                metrics = evaluation.summary()
                if job["dataset"] == "role_swap_test":
                    metrics["role_confusion@10"] = 100 * float(np.mean([r["role_confusion@10"] for r in evaluation.rows]))
                metrics["median_latency_ms"] = float(np.median([c["latency_ms"] for c in costs]))
                metrics["additional_index_bytes_per_doc"] = getattr(backend, "storage_per_doc", None)
                if diagnostics:
                    from .slot_scorer import ece

                    regrets = [d["plan_regret"] for d in diagnostics if d["plan_regret"] is not None]
                    misses = [m for d in diagnostics for m in d["misses"]]
                    conf = [v for d in diagnostics for v in d["confidence"]]
                    before = [v for d in diagnostics for v in d["confidence_before"]]
                    correct = [v for d in diagnostics for v in d["correct"]]
                    heldout = [v for d in diagnostics for v in d["heldout_correct"]]
                    metrics["planner"] = {"regret": float(np.mean(regrets)) if regrets else None,
                                          "regret_ci95": list(bootstrap_ci(regrets, np.zeros(len(regrets)))) if regrets else None,
                                          "miss_rate": float(np.mean(misses)) if misses else None,
                                          "ece_after": ece(conf, correct) if conf else None,
                                          "ece_before": ece(before, correct) if before and all(v is not None for v in before) else None,
                                          "slot_accuracy": float(np.mean(correct)) if correct else None,
                                          "heldout_slot_n": len(heldout),
                                          "heldout_slot_accuracy": float(np.mean(heldout)) if heldout else None}
                write_json(run / "run.json", saved)
                write_json(run / "metrics.json", metrics)
                pq.write_table(pa.Table.from_pylist(evaluation.rows), run / "per_query.parquet")
                pq.write_table(pa.Table.from_pylist(costs), run / "counters.parquet")
                for name, rows in (("plans", plans), ("evidence", evidence), ("planner_diagnostics", diagnostics)):
                    (run / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            finally:
                backend.close()
        state["jobs"][job["id"]]["status"] = "ok"
        write_json(out / "execution.json", state)
    return state

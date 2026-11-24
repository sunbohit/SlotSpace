"""M7 exact feature capture, staged validation tuning and unapproved configuration freezing."""

import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import softmax
import yaml

from .artifacts import _git, logged_run, new_directory, read_jsonl, record_run, sha256, write_json
from .evaluate import Evaluation, METRICS, per_query_metrics
from .feature_cache import QueryFeatures, cache_matches, capture_query, digest, replay
from .pipeline import RuntimePaths, load_index, load_pipeline
from .planner import PlannerSettings, calibrate_conformal, condition_key, plan_readings
from .reference import ScoreConfig
from .representations import RepresentationIndex, tune_field_weights
from .ranking import select_candidates
from .schema import Schema
from .slot_scorer import ece
from .stats import bootstrap_ci


class FeatureSet:
    def __init__(self, path: Path):
        self.path = path
        self.meta = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        base = {k: v for k, v in self.meta.items() if k != "fingerprint"}
        if digest(base) != self.meta["fingerprint"] or len({q["qid"] for q in self.meta["queries"]}) != len(self.meta["queries"]):
            raise ValueError("Invalid feature-set manifest")
        if sha256(path / "candidate_ids.npy") != self.meta["candidate_ids_sha256"]:
            raise ValueError("Feature-set corpus identity changed")
        self.doc_ids = np.load(path / "candidate_ids.npy", allow_pickle=False)
        self.loaded, self.verified = OrderedDict(), defaultdict(dict)

    def __iter__(self):
        for row in self.meta["queries"]:
            key = row["folder"]
            if key not in self.loaded:
                feature = QueryFeatures.load(self.path / key)
                if feature.payload["qid"] != row["qid"] or feature.fingerprint != row["fingerprint"]:
                    raise ValueError("Query features do not match their set manifest")
                feature.verified_matrices = self.verified[key]
                self.loaded[key] = feature
                if len(self.loaded) > 16:
                    self.loaded.popitem(last=False)
            self.loaded.move_to_end(key)
            yield self.loaded[key]


def build_feature_set(pipeline, queries: list[dict], silver: dict[int, list[dict]], out: Path,
                      split: str, identity: dict, scorings: list[ScoreConfig], *, complete: bool,
                      purpose: str = "production"):
    if purpose not in {"production", "fixture", "pilot"}:
        raise ValueError("Unknown feature-set purpose")
    if split not in {"calibration", "val"} or not queries:
        raise ValueError("Feature capture is restricted to nonempty calibration/validation sets")
    ids = [q["qid"] for q in queries]
    if len(set(ids)) != len(ids) or set(ids) - silver.keys():
        raise ValueError("Duplicate query IDs or missing silver records")
    if split == "calibration" and any(type(q) is not int or q % 2 for q in ids):
        raise ValueError("Calibration cache requires even-qid train records")
    with new_directory(out) as stage:
        np.save(stage / "candidate_ids.npy", pipeline.index.doc_ids)
        rows = []
        for n, query in enumerate(queries):
            feature = capture_query(pipeline, query, silver[query["qid"]])
            name = "q-" + digest(query["qid"])[:16]
            feature.save(stage / name)
            for scoring in scorings:
                cache_matches(feature, pipeline.index, scoring)
            rows.append({"qid": query["qid"], "folder": name, "fingerprint": feature.fingerprint})
            if (n + 1) % 20 == 0:
                print(f"{split} features: {n + 1}/{len(queries)}", flush=True)
        meta = {"format": 1, "identity": identity, "split": split, "complete": complete,
                "purpose": purpose, "queries": rows,
                "candidate_ids_sha256": sha256(stage / "candidate_ids.npy")}
        write_json(stage / "manifest.json", {**meta, "fingerprint": digest(meta)})
    return out


def conformal_bank(calibration: FeatureSet, schema, base: PlannerSettings, betas, alphas):
    if calibration.meta["split"] != "calibration":
        raise ValueError("Conformal thresholds cannot be fitted on validation features")
    conditions = {}
    for beta in sorted(set([0.0, *betas])):
        for use_pacc, uniform in ((True, False), (False, False), (True, True)):
            settings = replace(base, beta=beta, use_pacc=use_pacc, uniform_slots=uniform)
            posteriors, truth = [], []
            for feature in calibration:
                plan = feature.plan(schema, settings, 0)
                posteriors.extend(s["posterior"] for s in plan["segments"])
                truth.extend(feature.payload["true_readings"])
            conditions[condition_key(settings)] = {
                "settings": {"beta": beta, "eps": settings.eps, "use_pacc": use_pacc, "uniform_slots": uniform},
                **calibrate_conformal(posteriors, truth, tuple(alphas))}
    return conditions


def _threshold(conditions, settings, alpha):
    if alpha == "all":
        return 0.0
    return conditions[condition_key(settings)]["thresholds"][str(alpha)]


def _scoring(config):
    values = config["scoring"]
    return ScoreConfig(values["mode"], values["combine"], values["a"], values["b"], values["precision"])


def evaluate_features(features: FeatureSet, schema, config: dict, conditions: dict, *, keep_rows=False,
                      baseline=False, official=False):
    settings, scoring = PlannerSettings.from_config(config), _scoring(config)
    score = config["scoring"]
    weights = (0 if baseline else score["lambda_B"], score["lambda_D"], score["lambda_L"])
    threshold = 0.0 if baseline else _threshold(conditions, settings, config["planner"]["alpha"])
    metrics, costs, regrets, misses, correct, before, after = [], [], [], [], [], [], []
    saved, plans = {}, []
    evaluator = Evaluation(features.doc_ids.tolist()) if official else None
    for feature in features:
        result = replay(feature, schema, settings, threshold, scoring, weights,
                        config["retrieval"]["final_k"], hard_and=score.get("hard_and", False))
        qid, answers = feature.payload["qid"], set(feature.payload["answers"])
        row = {"qid": qid, **per_query_metrics(result["doc_ids"], answers), "plan_regret": None}
        if evaluator is not None:
            evaluator.add(qid, result["doc_ids"], feature.payload["answers"])
        if not baseline:
            if any(feature.payload["true_readings"]):
                oracle = replay(feature, schema, settings, 0, scoring, weights,
                                config["retrieval"]["final_k"], oracle=True)
                regret = per_query_metrics(oracle["doc_ids"], answers)["mrr"] - row["mrr"]
                regrets.append(regret)
                row["plan_regret"] = regret
            for segment, truth in zip(result["plan"]["segments"], feature.payload["true_readings"]):
                if truth:
                    misses.append(not bool(set(truth) & {r["key"] for r in segment["readings"]}))
                target = {key.split("|", 1)[0] for key in truth}
                p = segment["distribution"]
                if not target:
                    target = {key for key in p if key.startswith("none.")}
                best = max(p, key=p.get)
                correct.append(best in target)
                after.append(p[best])
                t = feature.payload["query_temperature"]
                if t is not None and t > 0 and min(p.values()) > 0 and not settings.uniform_slots:
                    raw = softmax(np.log(list(p.values())) * t)
                    before.append(float(raw.max()))
                else:
                    before.append(None)
        metrics.append(row)
        costs.append({"qid": qid, **result["counters"]})
        if keep_rows:
            saved[qid] = dict(zip(result["doc_ids"], result["scores"]))
            plans.append(result["plan"])
    means = {m: 100 * float(np.mean([r[m] for r in metrics])) for m in METRICS}
    summary = {"n_queries": len(metrics), "top100": means,
               "plan_regret": float(np.mean(regrets)) if regrets else None,
               "conformal_miss_rate": float(np.mean(misses)) if misses else None,
               "labelled_segments": len(misses), "ece_after": ece(after, correct) if correct else None,
               "ece_before": ece(before, correct) if before and all(p is not None for p in before) else None,
               "timing_kind": "feature replay; no measured end-to-end latency",
               "mean_work": {k: float(np.mean([c[k] for c in costs]))
                             for k in ("bm25_postings", "value_postings", "facts_scored", "candidates")}}
    if evaluator is not None:
        summary["official_check"] = evaluator.summary()
    if keep_rows:
        return summary, metrics, costs, saved, plans, regrets
    return summary


def tune(calibration: FeatureSet, validation: FeatureSet, schema, config: dict, out: Path):
    if (validation.meta["split"] != "val" or calibration.meta["split"] != "calibration"
            or calibration.meta["identity"] != validation.meta["identity"]
            or calibration.meta["purpose"] != validation.meta["purpose"]):
        raise ValueError("Tuning feature sets have incompatible provenance or roles")
    if {r["qid"] for r in calibration.meta["queries"]} & {r["qid"] for r in validation.meta["queries"]}:
        raise ValueError("Calibration and validation query IDs overlap")
    if validation.meta["purpose"] == "production" and validation.meta["identity"]["schema_sha256"] != sha256(Path(config["schema"])):
        raise ValueError("Schema changed after query feature extraction")
    if validation.meta["purpose"] == "production":
        score = config["scoring"]
        if (score["mode"] != "bound" or score["combine"] != "max" or score["precision"] != "float"
                or score["lambda_D"] != 1 or any(w <= 0 for w in config["tuning_grid"]["lambda_B"])):
            raise ValueError("Production A0 tuning requires bound/max/float scoring with lambda_D=1 and positive binding weights")
    selected, grid = deepcopy(config), config["tuning_grid"]
    settings = PlannerSettings.from_config(config)
    conditions = conformal_bank(calibration, schema, settings, grid["beta"], grid["alpha"])
    tables = []

    def trial(stage, candidate):
        measured = evaluate_features(validation, schema, candidate, conditions)
        tables.append({"stage": stage, "planner": deepcopy(candidate["planner"]),
                       "scoring": deepcopy(candidate["scoring"]), **measured})
        return measured

    def choose(stage, candidates, key):
        ranked = []
        for candidate in candidates:
            result = trial(stage, candidate)
            ranked.append((key(result), digest(candidate), candidate))
        return min(ranked, key=lambda x: (x[0], x[1]))[2]

    with new_directory(out) as stage, record_run(stage, {
        "settings": config, "calibration_features": calibration.meta["fingerprint"],
        "validation_features": validation.meta["fingerprint"], "timing_kind": "feature replay"
    }):
        baseline_results = []
        for weight in (0, 0.25, 0.5, 1):
            cfg = deepcopy(config)
            cfg["scoring"]["lambda_L"] = weight
            result = evaluate_features(validation, schema, cfg, conditions, baseline=True)
            baseline_results.append((result["top100"]["mrr"], -weight, result))
        _, neg_weight, baseline = max(baseline_results, key=lambda x: (x[0], x[1]))
        candidates = []
        for beta, alpha in product(grid["beta"], grid["alpha"]):
            cfg = deepcopy(selected)
            cfg["planner"].update(beta=beta, alpha=alpha)
            candidates.append(cfg)
        if not candidates:
            raise ValueError("Empty planner tuning grid")
        def regret_key(result):
            if result["plan_regret"] is None:
                raise ValueError("No labelled validation queries for plan-regret tuning")
            return result["plan_regret"]
        selected = choose("beta-alpha", candidates, regret_key)
        candidates = []
        for a, b in product(grid["a"], grid["b"]):
            cfg = deepcopy(selected)
            cfg["scoring"].update(a=a, b=b)
            candidates.append(cfg)
        selected = choose("a-b", candidates, lambda r: -r["top100"]["mrr"])
        candidates = []
        for lb, ll in product(grid["lambda_B"], grid["lambda_L"]):
            cfg = deepcopy(selected)
            cfg["scoring"].update(lambda_B=lb, lambda_L=ll)
            candidates.append(cfg)
        selected = choose("fusion", candidates, lambda r: -r["top100"]["mrr"])
        candidates = []
        for objective in ("p", "p_over_H", "ia"):
            cfg = deepcopy(selected)
            cfg["planner"]["weight_objective"] = objective
            candidates.append(cfg)
        selected = choose("weight-objective", candidates,
                          lambda r: (-r["top100"]["mrr"], -r["top100"]["recall@100"]))
        final, rows, costs, saved, plans, regrets = evaluate_features(
            validation, schema, selected, conditions, keep_rows=True, official=True)
        alpha = selected["planner"]["alpha"]
        checks = {"complete_calibration": calibration.meta["complete"], "complete_validation": validation.meta["complete"],
                  "recall_floor": final["top100"]["recall@100"] >= baseline["top100"]["recall@100"] - 1,
                  "conformal_miss_rate": final["conformal_miss_rate"] is not None
                  and final["conformal_miss_rate"] <= alpha + 0.02}
        selected.setdefault("baselines", {})["B2"] = {"lambda_L": -neg_weight}
        identity = validation.meta["identity"]
        bank = {**identity["conformal_identity"], "conditions": conditions,
                "calibration_features": calibration.meta["fingerprint"], "human_reviews": "pending"}
        write_json(stage / "conformal_bank.json", bank)
        selected["artifacts"] = {**identity["artifacts"], "conformal": str((out / "conformal_bank.json").resolve())}
        selected["tuning_evidence"] = {"calibration": calibration.meta["fingerprint"],
                                       "validation": validation.meta["fingerprint"],
                                       "timing_kind": "feature-replay"}
        write_json(stage / "trials.json", tables)
        write_json(stage / "metrics.json", final)
        write_json(stage / "run.json", saved)
        pq.write_table(pa.Table.from_pylist(rows), stage / "per_query.parquet")
        pq.write_table(pa.Table.from_pylist(costs), stage / "counters.parquet")
        (stage / "plans.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plans), encoding="utf-8")
        interval = bootstrap_ci(regrets, np.zeros(len(regrets))) if regrets else None
        selection = {"config": selected, "checks": checks, "accepted": all(checks.values()),
                     "purpose": validation.meta["purpose"], "metrics": final, "hybrid": baseline,
                     "regret_ci95": list(interval) if interval else None,
                     "calibration_fingerprint": calibration.meta["fingerprint"],
                     "validation_fingerprint": validation.meta["fingerprint"],
                     "input_fingerprints": identity.get("input_fingerprints", {}),
                     "bank_sha256": sha256(stage / "conformal_bank.json")}
        write_json(stage / "selection.json", selection)
    return selection


def freeze(selection_path: Path, out: Path):
    evidence = json.loads(selection_path.read_text(encoding="utf-8"))
    if evidence["purpose"] == "pilot":
        raise ValueError("Pilot validation cannot create a production frozen configuration or H5 receipt")
    record = out.with_suffix(out.suffix + ".freeze.json")
    if out.exists() or record.exists():
        raise FileExistsError("Frozen configuration or its receipt already exists")
    required = {"complete_calibration", "complete_validation", "recall_floor", "conformal_miss_rate"}
    if (evidence["accepted"] is not True or set(evidence["checks"]) != required
            or any(value is not True for value in evidence["checks"].values())):
        raise ValueError("M7 validation acceptance has not passed; configuration cannot be frozen")
    if evidence["purpose"] == "production":
        from .protocol import artifact_fingerprint

        pinned = evidence.get("input_fingerprints", {})
        if not pinned:
            raise ValueError("Production freezing requires the captured artifact fingerprints")
        for path, expected in pinned.items():
            if expected is None or artifact_fingerprint(path, remote_ok="remote_id" in expected) != expected:
                raise ValueError(f"Tuning input changed after feature extraction: {path}")
    bank = selection_path.parent / "conformal_bank.json"
    if sha256(bank) != evidence["bank_sha256"]:
        raise ValueError("Selected conformal bank changed after validation")
    content = yaml.safe_dump(evidence["config"], sort_keys=False, allow_unicode=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as dest:
        dest.write(content)
    write_json(record, {"frozen_sha256": sha256(out), "selection_sha256": sha256(selection_path),
                        "bank_sha256": sha256(bank), "purpose": evidence["purpose"],
                        "accepted": True, "H5": "pending", "seed": 13})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("cache")
    capture.add_argument("--config", type=Path, default=Path("configs/prime_v0.yaml"))
    capture.add_argument("--artifacts", type=Path, help="YAML mapping of RuntimePaths fields")
    capture.add_argument("--split", choices=("calibration", "val"), required=True)
    capture.add_argument("--silver", type=Path, default=Path("data/prime/silver_prime-0.1.0"))
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--limit", type=int)
    capture.add_argument("--device", default="cuda")
    fit = commands.add_parser("fit")
    fit.add_argument("--config", type=Path, default=Path("configs/prime_v0.yaml"))
    fit.add_argument("--calibration", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    extra = commands.add_parser("augment")
    extra.add_argument("--config", type=Path, default=Path("configs/prime_v0.yaml"))
    extra.add_argument("--cache", type=Path, required=True)
    extra.add_argument("--mode", choices=("bound", "unbound", "gate_only", "no_role"), default="bound")
    extra.add_argument("--precision", choices=("float", "binary"), default="float")
    extra.add_argument("--a", type=float, nargs="+", default=[1])
    extra.add_argument("--b", type=float, nargs="+", default=[1])
    field = commands.add_parser("fields")
    field.add_argument("--config", type=Path, default=Path("configs/prime_v0.yaml"))
    field.add_argument("--validation", type=Path, required=True)
    field.add_argument("--representations", type=Path, required=True)
    field.add_argument("--out", type=Path, required=True)
    lock = commands.add_parser("freeze")
    lock.add_argument("--selection", type=Path, required=True)
    lock.add_argument("--out", type=Path, default=Path("configs/prime_v0_frozen.yaml"))
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.selection, args.out))
        return
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    schema = Schema.load(config["schema"])
    if args.command == "fields":
        validation = FeatureSet(args.validation)
        if validation.meta["split"] != "val" or not validation.meta["complete"]:
            raise ValueError("Field weights require the complete validation feature set")
        identity = validation.meta["identity"]
        rep = RepresentationIndex(args.representations, identity["docs_sha256"], identity["encoder"])
        if not np.array_equal(rep.doc_ids, validation.doc_ids):
            raise ValueError("Field representations and validation corpus IDs differ")
        records = []
        for feature in validation:
            a = feature.arrays
            budgets = {**feature.payload["limits"], "K_V": 1}
            positions, _ = select_candidates(a["lexical"], a["dense"], {}, set(), budgets)
            ids = a["doc_ids"][positions]
            values, _ = rep.field_features(feature.payload["query"], a["topic"], np.searchsorted(rep.doc_ids, ids))
            records.append({"doc_ids": ids, "features": values, "answers": feature.payload["answers"]})
        ll = config.get("baselines", {}).get("B2", {}).get("lambda_L", config["scoring"]["lambda_L"])
        weights, trials = tune_field_weights(records, rep.meta["fields"], ll)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as dest:
            json.dump({"weights": weights, "trials": trials, "lambda_L": ll, "split": "val",
                       "validation_fingerprint": validation.meta["fingerprint"], "complete": True,
                       "representation_sha256": sha256(args.representations / "manifest.json")}, dest, indent=2)
        return
    if args.command == "augment":
        source = FeatureSet(args.cache)
        paths = RuntimePaths.from_mapping(source.meta["identity"]["artifacts"])
        index, _ = load_index(Path(config["paths"]["data"]) / "docs.parquet", paths.facts, paths.text_keys,
                              paths.context_cache, paths.lexicon, Path(config["schema"]), args.precision)
        for feature in source:
            for a, b in product(args.a, args.b):
                cache_matches(feature, index, ScoreConfig(mode=args.mode, precision=args.precision, a=a, b=b))
        return
    if args.command == "fit":
        print(json.dumps(tune(FeatureSet(args.calibration), FeatureSet(args.validation), schema, config, args.out)["checks"]))
        return
    if args.limit is not None and args.limit < 1:
        parser.error("Query limit must be positive")
    values = yaml.safe_load(args.artifacts.read_text(encoding="utf-8")) if args.artifacts else config.get("artifacts", {})
    values = values.get("artifacts", values)
    paths = RuntimePaths.from_mapping(values)
    pipeline = load_pipeline(config, paths, args.device)
    source = json.loads((args.silver / "manifest.json").read_text(encoding="utf-8"))
    data = Path(config["paths"]["data"])
    query_path = data / ("queries_train.jsonl" if args.split == "calibration" else "queries_val.jsonl")
    if (source["schema_sha256"] != sha256(Path(config["schema"]))
            or source["query_hashes"]["train" if args.split == "calibration" else "val"] != sha256(query_path)):
        raise ValueError("Feature-capture silver provenance mismatch")
    original = {q["qid"]: q for q in read_jsonl(query_path)}
    label_path = args.silver / ("silver_calibration.jsonl" if args.split == "calibration" else "silver_val.jsonl")
    labelled = list(read_jsonl(label_path))
    if any(q["qid"] not in original for q in labelled):
        raise ValueError("Silver calibration/validation source IDs differ")
    queries = [original[q["qid"]] for q in labelled][:args.limit]
    conf = json.loads(paths.conformal.read_text(encoding="utf-8"))
    identity = {"schema_sha256": sha256(Path(config["schema"])), "docs_sha256": sha256(data / "docs.parquet"),
                "implementation_commit": _git("rev-parse", "HEAD"),
                "index_fingerprint": pipeline.index.fingerprint,
                "artifacts": paths.to_dict(), "encoder": pipeline.ctx.embedder.metadata(),
                "scorer": pipeline.ctx.scorer.metadata(), "linking": config["linking"],
                "conformal_identity": {k: conf[k] for k in ("schema_version", "schema_sha256", "lexicon_version",
                                                          "scorer", "temperatures_sha256")}}
    from .protocol import artifact_fingerprint

    dependencies = [*[(str(value), key in {"model", "embedding_model"})
                       for key, value in paths.to_dict().items() if key not in {"kind", "values_visible"}],
                    (str(Path(config["schema"]).resolve()), False), (str((data / "docs.parquet").resolve()), False),
                    (str(args.silver.resolve()), False), (str((data / "queries_train.jsonl").resolve()), False),
                    (str((data / "queries_val.jsonl").resolve()), False)]
    identity["input_fingerprints"] = {path: artifact_fingerprint(path, remote_ok=remote)
                                      for path, remote in dependencies}
    if any(value is None for value in identity["input_fingerprints"].values()):
        raise ValueError("A feature-capture dependency is missing or not an identifiable artifact")
    scorings = [replace(pipeline.scoring, a=a, b=b) for a, b in product(config["tuning_grid"]["a"], config["tuning_grid"]["b"])]
    with logged_run(Path(config["paths"]["runs"]), f"m7-cache-{args.split}", {"identity": identity, "limit": args.limit}):
        build_feature_set(pipeline, queries, {q["qid"]: q["segments"] for q in labelled}, args.out,
                          args.split, identity, scorings, complete=args.limit is None and source["complete"])


if __name__ == "__main__":
    main()

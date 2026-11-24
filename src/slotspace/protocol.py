"""Frozen M8 experiment specifications and explicit H5 authorization."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import yaml

from .artifacts import _git, sha256, write_json
from .feature_cache import digest


@dataclass(frozen=True)
class Experiment:
    id: str
    strategy: str = "slotspace"
    changes: dict = field(default_factory=dict)
    variant: str = "main"
    oracle: bool = False


def catalog() -> list[Experiment]:
    return [
        Experiment("A0"), Experiment("B0", "bm25"), Experiment("B1", "dense"),
        Experiment("B1_1024", "dense", {"embedding.topic_dim": 1024}), Experiment("B2", "hybrid"),
        Experiment("B3", "maxp"), Experiment("B4", "fields"),
        Experiment("B5", changes={"scoring.mode": "unbound"}), Experiment("B6", "fanout"),
        Experiment("B7", oracle=True),
        Experiment("A1", changes={"scoring.mode": "no_role"}),
        Experiment("A2", changes={"scoring.mode": "gate_only"}),
        Experiment("A3", changes={"scoring.mode": "unbound"}),
        Experiment("A4", changes={"planner.beta": 0.0}),
        Experiment("A5", changes={"planner.uniform_slots": True}),
        Experiment("A6", changes={"scoring.hard_and": True}),
        Experiment("A7", changes={"facts.granularity": "chunk128"}, variant="chunk128"),
        Experiment("A8", changes={"planner.use_pacc": False}),
        *[Experiment(f"A9_{objective}", changes={"planner.weight_objective": objective})
          for objective in ("p", "p_over_H", "ia")],
        *[Experiment(f"A10_{alpha}", changes={"planner.alpha": alpha}) for alpha in (0.05, 0.1, 0.2, "all")],
        Experiment("A11", changes={"scoring.precision": "binary"}),
        Experiment("A12", variant="a12"), Experiment("A13", variant="zeroshot"), Experiment("A14", variant="a14"),
    ]


def changed_config(base: dict, experiment: Experiment):
    config = deepcopy(base)
    for path, value in experiment.changes.items():
        section, name = path.split(".")
        if section not in {"planner", "scoring", "embedding", "facts"}:
            raise ValueError(f"Unsupported experimental intervention: {path}")
        config[section][name] = value
    return config


def artifact_fingerprint(value: str, *, remote_ok=False):
    path = Path(value)
    if path.is_file():
        return {"sha256": sha256(path)}
    if not path.is_dir():
        return {"remote_id": value} if remote_ok else None
    if (path / "manifest.json").is_file():
        files = [p for p in path.rglob("*") if p.is_file()]
    elif (path / "config.json").is_file():
        files = [p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".txt", ".model"}]
    else:
        return None
    return {"files": {p.relative_to(path).as_posix(): sha256(p) for p in sorted(files)}}


def load_frozen(path: Path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    receipt = json.loads(path.with_suffix(path.suffix + ".freeze.json").read_text(encoding="utf-8"))
    if receipt["accepted"] is not True or receipt["purpose"] != "production" or receipt["frozen_sha256"] != sha256(path):
        raise ValueError("A complete production validation freeze is required before M8")
    bank = Path(config["artifacts"]["conformal"])
    selection = bank.parent / "selection.json"
    if (sha256(bank) != receipt["bank_sha256"] or sha256(selection) != receipt["selection_sha256"]
            or digest(json.loads(selection.read_text(encoding="utf-8"))["config"]) != digest(config)):
        raise ValueError("Frozen configuration/selection/conformal receipt mismatch")
    return config, receipt


def build_plan(config_path: Path, out: Path, *, draft=False, variants: dict | None = None,
               auxiliary: dict | None = None, only: list[str] | None = None):
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not draft:
        base, _ = load_frozen(config_path)
        if not _git("rev-parse", "HEAD") or _git("status", "--porcelain"):
            raise ValueError("Commit implementation and frozen configuration before planning final runs")
    base = deepcopy(base)
    base["schema"] = str(Path(base["schema"]).resolve())
    base["paths"] = {k: str(Path(v).resolve()) for k, v in base["paths"].items()}
    variants, auxiliary = variants or {}, auxiliary or {}
    specs = catalog()
    if only is not None:
        known = {s.id for s in specs}
        if set(only) - known:
            raise ValueError(f"Unknown experiment IDs: {sorted(set(only) - known)}")
        specs = [s for s in specs if s.id in only]
    if not specs:
        raise ValueError("An experiment plan cannot be empty")
    data = Path(base["paths"]["data"])
    role_data = Path(base["paths"].get("role_swap", str(data / "role_swap_prime-0.1.0")))
    datasets = {"prime_test": data / "queries_test.jsonl", "human": data / "queries_human.jsonl",
                "role_swap_test": role_data / "role_swap_test.jsonl", "val": data / "queries_val.jsonl"}
    fingerprints, files, jobs, aliases = {}, {}, [], {}

    def pin(value, remote=False):
        value = str(value)
        key = value if remote and not Path(value).exists() else str(Path(value).resolve())
        if key not in fingerprints:
            fingerprints[key] = artifact_fingerprint(value, remote_ok=remote)
        return key

    schema_pin = pin(base["schema"])
    for name, path in datasets.items():
        count = 0
        if path.exists():
            with path.open(encoding="utf-8") as source:
                count = sum(1 for _ in source)
        files[name] = {"path": str(path.resolve()), "sha256": sha256(path) if path.exists() else None,
                       "count": count, "expected_count": {"prime_test": 2801, "human": 98, "val": 2241}.get(name)}
    for spec in specs:
        config = changed_config(base, spec)
        resources = dict(base.get("artifacts", {}))
        missing = ["schema"] if fingerprints[schema_pin] is None else []
        if spec.variant != "main":
            if spec.variant not in variants:
                missing.append(f"variant:{spec.variant}")
            else:
                resources.update(variants[spec.variant])
        resources["values_visible"] = spec.variant == "a12"
        if spec.variant == "zeroshot":
            resources["kind"] = "zeroshot"
        required = {"facts", "text_keys", "context_cache", "topic_cache", "lexicon", "model",
                    "embedding_model", "temperatures", "conformal"}
        missing += [f"artifact:{k}" for k in sorted(required - resources.keys())]
        pins = {}
        for key in sorted(required & resources.keys()):
            pins[key] = pin(resources[key], key in {"model", "embedding_model"})
            resources[key] = pins[key]
            if fingerprints[pins[key]] is None:
                missing.append(f"artifact:{key}")
        aux = {}
        needed = (["passages"] if spec.strategy == "maxp" else
                  ["fields", "field_weights"] if spec.strategy == "fields" else [])
        if spec.id == "A11":
            needed += ["binary_context_cache"]
        for key in needed:
            if key not in auxiliary:
                missing.append(f"auxiliary:{key}")
            else:
                aux[key] = str(Path(auxiliary[key]).resolve())
                location = pin(aux[key])
                if fingerprints[location] is None:
                    missing.append(f"auxiliary:{key}")
        if spec.id == "A11" and "binary_context_cache" in aux:
            resources["context_cache"] = aux["binary_context_cache"]
        if "model" in resources and resources.get("kind", "ft") == "ft":
            model_meta = Path(resources["model"]).parent / "manifest.json"
            expected = spec.variant if spec.variant in {"a12", "a14"} else "main"
            if not model_meta.exists() or json.loads(model_meta.read_text(encoding="utf-8")).get("variant") != expected:
                missing.append(f"trained-model:{expected}")
            else:
                pin(model_meta)
        if spec.id == "A7" and "facts" in resources:
            fact_meta = Path(resources["facts"]) / "manifest.json"
            if not fact_meta.exists() or json.loads(fact_meta.read_text(encoding="utf-8")).get("granularity") != "chunk128":
                missing.append("chunk128 fact generation")
        sets = ("val",) if spec.oracle else ("prime_test", "human", "role_swap_test")
        if spec.id in {"A0", "A4", "A5", "A12", "A13", "A14"} or spec.id.startswith("A10_"):
            sets = (*sets, "val")
        for dataset in sets:
            job_id = f"{spec.id}--{dataset}"
            signature = digest({"dataset": dataset, "strategy": spec.strategy, "config": config,
                                "resources": resources, "auxiliary": aux, "oracle": spec.oracle})
            job = {"id": job_id, "experiment": asdict(spec), "dataset": dataset,
                   "config": config, "artifacts": resources, "auxiliary": aux,
                   "missing": sorted(set(missing + ([] if files[dataset]["sha256"] else [f"dataset:{dataset}"])))}
            if files[dataset]["expected_count"] is not None and files[dataset]["count"] != files[dataset]["expected_count"]:
                job["missing"].append(f"dataset-count:{dataset}")
            if signature in aliases:
                job["alias_of"] = aliases[signature]
            else:
                aliases[signature] = job_id
            jobs.append(job)
    references = {}
    for key, path in {
        "pacc": Path(base.get("artifacts", {}).get("lexicon", "data/prime/lexicon_ft")) / "pacc_diagnostic.json",
        "calibration_points": Path(base.get("artifacts", {}).get("temperatures", "data/prime/models/slot_scorer_ft/temperature.json")).with_suffix(".points.parquet"),
    }.items():
        if path.exists():
            references[key] = str(path.resolve())
            pin(path)
    plan = {"format": 1, "draft": draft, "frozen": str(config_path.resolve()),
            "frozen_sha256": sha256(config_path), "git_commit": _git("rev-parse", "HEAD"),
            "datasets": files, "artifacts": fingerprints, "references": references, "jobs": jobs,
            "selection_scope": "explicit subset" if only is not None else "complete catalog"}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, plan)
    return plan


def authorize(plan_path: Path, approval_path: Path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["draft"]:
        raise ValueError("Draft protocols cannot run final experiments")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (approval.get("approved") is not True or approval.get("frozen_sha256") != plan["frozen_sha256"]
            or approval.get("plan_sha256") != sha256(plan_path)):
        raise ValueError("H5 approval must explicitly bind this exact frozen configuration and experiment plan")
    load_frozen(Path(plan["frozen"]))
    if sha256(Path(plan["frozen"])) != plan["frozen_sha256"]:
        raise ValueError("Frozen configuration changed after plan creation")
    if _git("rev-parse", "HEAD") != plan["git_commit"] or _git("status", "--porcelain"):
        raise ValueError("Implementation changed after H5 planning")
    if any(job["missing"] for job in plan["jobs"]):
        raise ValueError("Experiment plan still has missing or incompatible prerequisites")
    for path, expected in plan["artifacts"].items():
        if expected is None or artifact_fingerprint(path, remote_ok="remote_id" in expected) != expected:
            raise ValueError(f"Experiment artifact changed: {path}")
    for dataset in plan["datasets"].values():
        if dataset["sha256"] is not None and sha256(Path(dataset["path"])) != dataset["sha256"]:
            raise ValueError("Benchmark query data changed after approval")
    return plan

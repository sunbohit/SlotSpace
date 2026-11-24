"""M0-M8 from cached public inputs, with immutable receipts and real human/H5 gates."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import distributions
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

from .artifacts import _git, read_jsonl, sha256, write_json


@dataclass(frozen=True)
class Step:
    id: str
    stage: str
    module: str = ""
    args: tuple = ()
    outputs: tuple = ()
    external: tuple = ()
    generation: bool = False
    gate: str = ""


def fingerprint(path: Path):
    if path.is_file():
        return {".": sha256(path)}
    if not path.is_dir():
        raise FileNotFoundError(f"Required reproduction input/output is absent: {path}")
    items = {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not items:
        raise ValueError(f"Required artifact directory is empty: {path}")
    return items


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_steps(project: Path, resources: Path, work: Path):
    """Return generated config texts and the actual existing-CLI dependency order."""
    data, runs = work / "data" / "prime", work / "runs"
    cfg, bundle, frozen = work / "config.yaml", work / "artifacts.yaml", work / "frozen.yaml"
    schema, templates = project / "schemas" / "prime.yaml", project / "templates"
    models = resources / "models"
    encoder = models / "Qwen" / "Qwen3-Embedding-0.6B"
    base = models / "answerdotai" / "ModernBERT-base"
    generator = models / "Qwen" / "Qwen3-4B-Instruct-2507"
    zero = models / "BAAI" / "bge-reranker-v2-m3"
    facts, silver, native = (data / name for name in ("facts_prime-0.1.0", "silver_prime-0.1.0", "lexicon_native"))
    roles, chunk = data / "role_swap_prime-0.1.0", data / "facts_chunk128-0.1.0"
    paths = {}
    for kind in ("ft", "a12", "a14", "zeroshot", "chunk128"):
        paths[kind] = {
            "facts": str(chunk if kind == "chunk128" else facts),
            "text_keys": str(data / f"scores_{kind}" / f"text_fact_keys_{'zeroshot' if kind == 'zeroshot' else 'ft'}.parquet"),
            "context_cache": str(data / f"ctx_{'chunk128' if kind == 'chunk128' else 'prime'}-0.1.0_qwen3e06"),
            "topic_cache": str(data / "topic_prime-0.1.0_qwen3e06"),
            "lexicon": str(data / f"lexicon_{kind}"),
            "model": str(zero if kind == "zeroshot" else data / "models" / f"slot_scorer_{'ft' if kind == 'chunk128' else kind}" / "final"),
            "embedding_model": str(encoder),
            "temperatures": str(data / f"temperature_{'ft' if kind == 'chunk128' else kind}.json"),
            "conformal": str(data / f"conformal_{kind}.json"),
            "kind": "zeroshot" if kind == "zeroshot" else "ft", "values_visible": kind == "a12",
        }
    config = yaml.safe_load((project / "configs" / "prime_v0.yaml").read_text(encoding="utf-8"))
    config.update(schema=str(schema), paths={"data": str(data), "runs": str(runs), "role_swap": str(roles)})
    config["embedding"]["model"] = str(encoder)
    config["slot_scorer"].update(model=str(base), zero_shot_model=str(zero))
    config["artifacts"] = paths["ft"]
    auxiliary = {"passages": str(data / "passages"), "fields": str(data / "fields"),
                 "field_weights": str(data / "field_weights.json"), "binary_context_cache": str(data / "ctx_binary")}
    generated = {
        cfg: yaml.safe_dump(config, sort_keys=False),
        bundle: yaml.safe_dump({"artifacts": paths["ft"], "variants": {k: paths[k] for k in paths if k != "ft"},
                                "auxiliary": auxiliary}, sort_keys=False),
    }
    steps = []

    def add(id, stage, module, args=(), outputs=(), external=(), generation=False):
        steps.append(Step(id, stage, module, tuple(args), tuple(map(str, outputs)), tuple(map(str, external)), generation))

    def gate(name, stage):
        steps.append(Step(name, stage, gate=name))

    m0_files = ("docs.parquet", "edges.parquet",
                *(f"queries_{s}.jsonl" for s in ("train", "val", "test", "test-0.1", "human")))
    add("export", "M0", "stark_io", ["--root", resources / "datasets" / "stark", "--out", data, "--runs", runs],
        [data / name for name in m0_files])
    for mode in ("bm25", "dense", "hybrid"):
        # Snapshot only M0's seven files, not its parent: later stages add caches there.
        outputs = [runs / f"m1_{mode}"]
        if mode == "bm25":
            outputs += [data / "bm25_prime-0.1.0"]
        elif mode == "dense":
            outputs += [data / "topic_prime-0.1.0_qwen3e06"]
        args = ["--config", cfg, "--mode", mode, "--runs", outputs[0], "--device", "cuda"]
        if mode == "hybrid":
            args += ["--lambda-L", "0", "0.25", "0.5", "1"]
        add(f"baseline-{mode}", "M1", "experiments", args, outputs, [encoder] if mode != "bm25" else [])
    add("schema", "M2", "schema", ["--docs", data / "docs.parquet", "--edges", data / "edges.parquet", "--schema", schema])
    add("facts", "M2", "facts", ["--config", cfg, "--out", facts], [facts, runs / "m2_stats.json"])
    gate("H1", "M2")
    add("native-lexicon", "M3", "lexicon_build", ["--native-only", "--facts", facts, "--docs", data / "docs.parquet",
        "--schema", schema, "--out", native, "--runs", runs / "native"], [native])
    add("silver", "M4", "silver", ["--config", cfg, "--facts", facts, "--out", silver], [silver, runs / "m4_silver_stats.json"])
    add("role-swap", "M4", "role_swap", ["--docs", data / "docs.parquet", "--native", facts / "native_keys.parquet",
        "--lexicon", native, "--project", project, "--out", roles],
        [roles, runs / "m4_role_swap_stats.json"])
    gate("H2", "M4")
    gate("H3", "M4")
    for kind in ("ft", "a12"):
        filled, paraphrased, masked = (data / f"{n}_{kind}" for n in ("filled", "paraphrased", "masked"))
        args = ["--schema", schema, "--templates", templates / "train" / "prime_slots.yaml", "--out", filled]
        if kind == "a12":
            args += ["--a12-docs", data / "docs.parquet"]
        add(f"fill-{kind}", "M5", "gen.fill_templates", args, [filled])
        add(f"paraphrase-{kind}", "M5", "gen.paraphrase_vllm",
            ["--input", filled / "filled.jsonl", "--out", paraphrased, "--model", generator],
            [paraphrased], [generator], generation=True)
        add(f"mask-{kind}", "M5", "gen.validate", ["--input", paraphrased / "paraphrases.jsonl",
            "--schema", schema, "--out", masked], [masked])
    for kind, variant in (("ft", "main"), ("a12", "a12"), ("a14", "a14")):
        folder = Path(paths[kind]["model"]).parent
        add(f"train-{kind}", "M5", "train_slot_scorer", ["--data", data / f"masked_{'a12' if kind == 'a12' else 'ft'}" / "validated.jsonl",
            "--schema", schema, "--out", folder, "--model", base, "--variant", variant], [folder], [base])
    for kind in ("ft", "a12", "a14", "zeroshot"):
        p = paths[kind]
        t = Path(p["temperatures"])
        add(f"temperature-{kind}", "M5", "calibrate_slot_scorer", ["--model", p["model"], "--silver", silver,
            "--train-queries", data / "queries_train.jsonl", "--docs", data / "docs.parquet", "--facts", facts,
            "--schema", schema, "--out", t, "--kind", p["kind"]], [t, t.with_suffix(".points.parquet")],
            [zero] if kind == "zeroshot" else [])
    gate("H4", "M5")
    add("chunk-facts", "M5", "facts", ["--config", cfg, "--granularity", "chunk128", "--tokenizer", encoder,
        "--out", chunk, "--runs", runs / "chunk"], [chunk], [encoder])
    for kind, folder in (("prime", facts), ("chunk128", chunk)):
        add(f"contexts-{kind}", "M5", "embed", ["--kind", "context", "--docs", folder / "text_facts.parquet",
            "--cache", data / f"ctx_{kind}-0.1.0_qwen3e06", "--model", encoder],
            [data / f"ctx_{kind}-0.1.0_qwen3e06"], [encoder])
    for kind, p in paths.items():
        out = Path(p["text_keys"]).parent
        args = ["--facts", Path(p["facts"]) / "text_facts.parquet", "--docs", data / "docs.parquet",
                "--schema", schema, "--out", out, "--model", p["model"], "--kind", p["kind"],
                "--temperature", {"json": p["temperatures"], "key": "T_doc"}]
        if kind == "a12":
            args += ["--values-visible"]
        add(f"score-{kind}", "M5", "slot_scorer", args, [out], [zero] if kind == "zeroshot" else [])
        add(f"lexicon-{kind}", "M5", "lexicon_build", ["--facts", p["facts"], "--docs", data / "docs.parquet",
            "--schema", schema, "--text-keys", p["text_keys"], "--out", p["lexicon"], "--runs", runs / f"lexicon_{kind}"],
            [p["lexicon"]])

    def conformal(kind, stage, selected):
        p = paths[kind]
        args = ["--config", selected, "--silver", silver, "--lexicon", p["lexicon"], "--model", p["model"],
                "--temperatures", p["temperatures"], "--kind", p["kind"], "--out", p["conformal"]]
        if kind == "a12":
            args += ["--values-visible"]
        add(f"conformal-{kind}", stage, "planner", args, [p["conformal"]], [encoder, zero] if kind == "zeroshot" else [encoder])

    conformal("ft", "M6", cfg)
    for split in ("calibration", "val"):
        add(f"features-{split}", "M7", "tuning", ["cache", "--config", cfg, "--artifacts", bundle,
            "--split", split, "--silver", silver, "--out", data / f"tuning_{split}"], [data / f"tuning_{split}"], [encoder])
    add("tune", "M7", "tuning", ["fit", "--config", cfg, "--calibration", data / "tuning_calibration",
        "--validation", data / "tuning_val", "--out", work / "tuning"], [work / "tuning"])
    add("freeze", "M7", "tuning", ["freeze", "--selection", work / "tuning" / "selection.json", "--out", frozen],
        [frozen, frozen.with_suffix(".yaml.freeze.json")])
    for kind in paths:
        if kind != "ft":
            conformal(kind, "M7", frozen)
    for kind in ("passages", "fields"):
        add(f"representations-{kind}", "M7", "representations", ["--kind", kind, "--docs", data / "docs.parquet",
            "--edges", data / "edges.parquet", "--out", data / kind, "--model", encoder], [data / kind], [encoder])
    add("field-weights", "M7", "tuning", ["fields", "--config", frozen, "--validation", data / "tuning_val",
        "--representations", data / "fields", "--out", data / "field_weights.json"], [data / "field_weights.json"])
    add("binary", "M7", "embed", ["--kind", "binary", "--source-cache", Path(paths["ft"]["context_cache"]),
        "--cache", data / "ctx_binary"], [data / "ctx_binary"])
    add("protocol", "M8", "experiments", ["plan", "--config", frozen, "--variants", bundle, "--auxiliary", bundle,
        "--out", work / "protocol.json"], [work / "protocol.json"])
    gate("H5", "M8")
    add("execute", "M8", "experiments", ["run", "--plan", work / "protocol.json", "--approval", "@H5",
        "--out", work / "execution"], [work / "execution"])
    add("report", "M8", "experiments", ["report", "--plan", work / "protocol.json", "--execution", work / "execution",
        "--out", work / "report_v0.md"], [work / n for n in ("report_v0.md", "report_v0.json", "report_v0_figures")])
    return generated, steps


class ReviewRequired(RuntimeError):
    pass


def generator_identity(executable):
    if executable is None:
        raise ReviewRequired("Resume requires the same separate --gen-python environment")
    code = ("import json,sys; from importlib.metadata import distributions; "
            "print(json.dumps({'python':sys.version,'packages':"
            "sorted((d.metadata['Name'],d.version) for d in distributions())}))")
    return json.loads(subprocess.check_output([str(executable), "-I", "-c", code], text=True))


def _labels(path: Path, field: str, minimum: int, threshold: float):
    rows = list(read_jsonl(path))
    if len(rows) < minimum or any(type(r.get(field)) is not bool for r in rows):
        raise ReviewRequired(f"Label at least {minimum} examples using boolean {field!r} in {path}")
    rate = sum(r[field] for r in rows) / len(rows)
    if rate < threshold:
        raise ReviewRequired(f"{path}: reviewed rate {rate:.3f} is below {threshold}")
    return rate


def check_review(gate: str, work: Path, schema: Path, approvals: Path | None, h5: Path | None):
    if gate == "H5":
        if h5 is None:
            raise ReviewRequired(f"Supply --approval with external H5 authorization for {work / 'protocol.json'}")
        from .protocol import authorize

        authorize(work / "protocol.json", h5)
        return {"H5_sha256": sha256(h5)}
    mapping = {"H1": ("m2_stats.json", ("query_mentions", "document_mentions"), "correct", 100, 0.85),
               "H2": ("m4_silver_stats.json", ("silver_review",), "correct", 100, 0.8),
               "H3": ("m4_role_swap_stats.json", ("role_swap_review",), "natural", 50, 0.9)}
    evidence = {str(schema): sha256(schema)}
    if gate in mapping:
        stat, names, field, minimum, threshold = mapping[gate]
        info = json.loads((work / "runs" / stat).read_text(encoding="utf-8"))
        review = work / info["review_run"]
        for name in names:
            path = review / f"{name}.jsonl"
            _labels(path, field, minimum, threshold)
            evidence[str(path)] = sha256(path)
    elif gate == "H4":
        temp = work / "data" / "prime" / "temperature_ft.json"
        result = json.loads(temp.read_text(encoding="utf-8"))
        if any(not math.isfinite(result[s]["ece_after"]) or not 0 <= result[s]["ece_after"] <= 0.05 for s in ("query", "doc")):
            raise ReviewRequired("H4 requires both measured post-scaling ECE values <= 0.05")
        reviews = [p for p in (work / "runs").glob("m5-calibrate-*/scorer_review.jsonl")
                   if json.loads((p.parent / "meta.json").read_text(encoding="utf-8"))["config"]["scorer"] == result["scorer"]]
        if len(reviews) != 1:
            raise ReviewRequired("H4 needs one provenance-matched main-model scorer review")
        _labels(reviews[0], "correct", 30, 0)
        evidence.update({str(temp): sha256(temp), str(reviews[0]): sha256(reviews[0])})
    else:
        raise ValueError(f"Unknown human gate: {gate}")
    expected = _digest(evidence)
    supplied = json.loads(approvals.read_text(encoding="utf-8")) if approvals and approvals.exists() else {}
    if supplied.get(gate) != {"approved": True, "evidence_sha256": expected}:
        raise ReviewRequired(f"After human review, add {gate}: "
                             f'{json.dumps({"approved": True, "evidence_sha256": expected})} to --reviews JSON')
    return {"evidence_sha256": expected}


def execute_steps(steps, work, signature, *, resume=False, gen_python=None, reviews=None, approval=None,
                  schema=None, runner=None):
    state_path = work / "reproduction.json"
    if state_path.exists():
        if not resume:
            raise FileExistsError("Reproduction workspace exists; use --resume to verify completed receipts")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("Source, environment or reproduction settings changed; use a new workspace")
    else:
        state = {"signature": signature, "source": "fixture" if runner else "subprocess", "completed": {}}
    execute = runner or subprocess.run
    for step in steps:
        previous = state["completed"].get(step.id)
        if previous is not None:
            current = (check_review(step.gate, work, schema, reviews, approval) if step.gate else
                       {p: fingerprint(Path(p)) for p in (*step.outputs, *step.external)})
            if step.generation:
                current["generator_environment"] = generator_identity(gen_python)
            if current != previous:
                raise ValueError(f"Reproduction artifact/review changed after {step.id}")
            continue
        state.update(status="running", current=step.id)
        write_json(state_path, state)
        try:
            if step.gate:
                result = check_review(step.gate, work, schema, reviews, approval)
            else:
                if any(Path(p).exists() for p in step.outputs):
                    raise FileExistsError(f"Unowned/partial outputs at {step.id}; never silently overwrite or retry final runs")
                if step.generation and (gen_python is None or Path(gen_python).resolve() == Path(sys.executable).resolve()):
                    raise ReviewRequired("Offline generation needs --gen-python from a separate Linux vLLM environment")
                external = {p: fingerprint(Path(p)) for p in step.external}
                if step.generation:
                    external["generator_environment"] = generator_identity(gen_python)
                args = []
                for arg in step.args:
                    if isinstance(arg, dict):
                        value = json.loads(Path(arg["json"]).read_text(encoding="utf-8"))[arg["key"]]
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                            raise ValueError("Invalid measured document temperature")
                        args.append(str(value))
                    elif arg == "@H5":
                        if approval is None:
                            raise ReviewRequired("No H5 approval supplied")
                        args.append(str(approval))
                    else:
                        args.append(str(arg))
                executable = str(gen_python) if step.generation else sys.executable
                env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_MODE": "disabled", "PYTHONNOUSERSITE": "1"}
                parent = Path(executable).resolve().parent
                env["PATH"] = os.pathsep.join(map(str, (parent, parent / "Scripts", parent / "Library" / "bin"))) + os.pathsep + env["PATH"]
                execute([executable, "-m", f"slotspace.{step.module}", *args], cwd=work, env=env, check=True)
                result = {**external, **{p: fingerprint(Path(p)) for p in step.outputs}}
            state["completed"][step.id] = result
            state["status"] = "ok"
        except ReviewRequired:
            state["status"] = "blocked"
            raise
        finally:
            if step.id not in state["completed"] and state["status"] == "running":
                state["status"] = "blocked" if step.gate else "failed"
            write_json(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--resources", type=Path, default=Path("../Resource"))
    parser.add_argument("--work", type=Path, default=Path("runs/reproduce-v0"))
    parser.add_argument("--through", choices=tuple(f"M{i}" for i in range(9)), default="M8")
    parser.add_argument("--gen-python", type=Path)
    parser.add_argument("--reviews", type=Path, help="External hash-bound H1-H4 approvals, never auto-generated")
    parser.add_argument("--approval", type=Path, help="External existing-protocol H5 approval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project, resources, work = args.project.resolve(), args.resources.resolve(), args.work.resolve()
    if (work == project or project.is_relative_to(work) or resources.is_relative_to(work)
            or work.is_relative_to(resources)
            or any(work.is_relative_to(project / name) for name in ("src", "schemas", "templates", "configs"))):
        parser.error("--work must not contain the source or cached Resource directory")
    generated, steps = build_steps(project, resources, work)
    steps = [s for s in steps if int(s.stage[1:]) <= int(args.through[1:])]
    if args.dry_run:
        print(json.dumps([asdict(s) for s in steps], default=str, indent=2))
        return
    source = {str(p.relative_to(project)): sha256(p) for folder in ("src", "schemas", "templates/train", "configs")
              for p in sorted((project / folder).rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts and not any(s.endswith(".egg-info") for s in p.parts)}
    from .role_swap import template_source

    signature = _digest({"source": source, "test_template_sha256": template_source(project)[1],
                         "git_commit": _git("rev-parse", "HEAD"),
                         "generated": {str(p): text for p, text in generated.items()},
                         "generation_python": str(args.gen_python.resolve()) if args.gen_python else None,
                         "python": sys.version, "packages": sorted((d.metadata["Name"], d.version) for d in distributions())})
    if not (work / "reproduction.json").exists() and work.exists():
        parser.error("Use a new empty --work path, not an existing data/artifact directory")
    work.mkdir(parents=True, exist_ok=True)
    for path, text in generated.items():
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Generated reproduction configuration changed: {path}")
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    try:
        state = execute_steps(steps, work, signature, resume=args.resume,
                              gen_python=args.gen_python.resolve() if args.gen_python else None,
                              reviews=args.reviews.resolve() if args.reviews else None,
                              approval=args.approval.resolve() if args.approval else None,
                              schema=project / "schemas" / "prime.yaml")
    except ReviewRequired as exc:
        parser.exit(2, f"{exc}\nResume with unchanged source/settings using --resume.\n")
    print(json.dumps({"through": args.through, "completed": len(state["completed"]), "workspace": str(work)}))


if __name__ == "__main__":
    main()

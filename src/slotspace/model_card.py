"""Generate an evidence-backed card only after an actual fine-tuning/calibration run."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open

from .artifacts import sha256
from .schema import Schema
from .slot_scorer import ece, local_model_identity


def generate(model: Path, calibration: Path, schema: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError("Refusing to replace an existing model card")
    train = json.loads((model.parent / "manifest.json").read_text(encoding="utf-8"))
    scores = json.loads(calibration.read_text(encoding="utf-8"))
    inventory = Schema.load(schema)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    weights = sorted(model.glob("*.safetensors"))
    if (not weights or train["variant"] not in {"main", "a12", "a14"}
            or train["schema_sha256"] != sha256(schema) or scores["schema_sha256"] != sha256(schema)
            or train["schema_version"] != inventory.version or scores["schema_version"] != inventory.version):
        raise ValueError("Training/calibration/schema evidence is absent or incompatible")
    if (config["model_type"] != "modernbert"
            or not any(a.endswith("ForSequenceClassification") for a in config["architectures"])
            or train["model"] != "answerdotai/ModernBERT-base" and Path(train["model"]).name != "ModernBERT-base"):
        raise ValueError("This v0 card requires the documented ModernBERT-base sequence classifier")
    if (any(type(train[k]) is not int or train[k] < 1 for k in ("epochs", "batch_size", "train_pairs", "dev_pairs"))
            or not train["train_templates"] or not train["dev_templates"]
            or set(train["train_templates"]) & set(train["dev_templates"])):
        raise ValueError("Training requires positive sample counts and disjoint template splits")
    identity = local_model_identity(model)
    keys = set()
    for weight in weights:
        with safe_open(weight, framework="np") as checkpoint:
            keys.update(checkpoint.keys())
    if "classifier.weight" not in keys:
        raise ValueError("Checkpoint is missing the trained classifier")
    if scores["scorer"]["model"] != identity:
        raise ValueError("Calibrated scorer identity differs from the checkpoint")
    if (scores["query_source"] != "train/even qid"
            or scores["scorer"]["values_visible"] is not (train["variant"] == "a12")
            or scores["scorer"]["pair_order"] != "context_first" or scores["scorer"]["max_length"] != 128):
        raise ValueError("Calibration split, pair order or target-visibility evidence differs")
    if any(type(scores[k]) not in {int, float} or not math.isfinite(scores[k]) or scores[k] <= 0
           for k in ("T_query", "T_doc")):
        raise ValueError("Measured temperatures must be finite and positive")
    points_path = calibration.with_suffix(".points.parquet")
    points = pq.read_table(points_path).to_pylist()
    if any(r["style"] not in {"query", "doc"} or type(r["correct"]) is not bool for r in points):
        raise ValueError("Calibration points have invalid styles or correctness labels")
    for style in ("query", "doc"):
        values = scores[style]
        rows = [r for r in points if r["style"] == style]
        if (type(values["n"]) is not int or values["n"] < 1 or values["n"] != len(rows)
                or any(type(values[k]) not in {int, float} or not 0 <= values[k] <= 1
                                  for k in ("accuracy", "ece_before", "ece_after"))):
            raise ValueError("Model card cannot report invalid or missing measured metrics")
        correct = [r["correct"] for r in rows]
        measured = {"accuracy": float(np.mean(correct)),
                    "ece_before": ece([r["confidence_before"] for r in rows], correct),
                    "ece_after": ece([r["confidence_after"] for r in rows], correct)}
        if any(not math.isclose(values[k], v, rel_tol=0, abs_tol=1e-12) for k, v in measured.items()):
            raise ValueError("Calibration metrics differ from the per-example measurements")
    lines = ["---", "license: apache-2.0", "base_model: answerdotai/ModernBERT-base",
             "library_name: sentence-transformers", "pipeline_tag: text-classification", "---", "",
             "# SlotSpace measured checkpoint", "",
             f"Variant: {train['variant']}; schema version: {train['schema_version']}; seed: {train['seed']}.",
             "The card reports checkpoint/calibration evidence, not final retrieval acceptance.", "",
             "| Style | Calibration n | Slot accuracy | ECE before | ECE after |",
             "|---|---:|---:|---:|---:|"]
    for style in ("query", "doc"):
        v = scores[style]
        lines.append(f"| {style} | {v['n']} | {v['accuracy']:.6f} | {v['ece_before']:.6f} | {v['ece_after']:.6f} |")
    lines += ["", "## Training and intended use", "",
              f"Training/dev pairs: {train['train_pairs']} / {train['dev_pairs']}; "
              f"epochs: {train['epochs']}; batch size: {train['batch_size']}.",
              f"Temperatures: query {scores['T_query']:.6f}; document {scores['T_doc']:.6f}.",
              "Post-scaling ECE <= 0.05 target: "
              + ("PASS" if all(scores[s]["ece_after"] <= 0.05 for s in ("query", "doc")) else "FAIL")
              + " (not an H4 approval).",
              "Masked schema-generated contexts, template-disjoint train/dev and offline paraphrases. "
              "A12 deliberately exposes targets; main/A14 do not.",
              "This is retrieval research, not a medical diagnostic or treatment system. "
              "Synthetic style bias, noisy silver labels, KB incompleteness and linking errors remain limitations.",
              "Calibration uses even-qid train silver and KB-linked document mentions, not test labels.", "",
              "## Evidence", "", f"- Training manifest SHA-256: {sha256(model.parent / 'manifest.json')}",
              f"- Calibration SHA-256: {sha256(calibration)}", f"- Schema SHA-256: {sha256(schema)}",
              f"- Calibration measurements SHA-256: {sha256(points_path)}",
              f"- Training input SHA-256: {train['data_sha256']}",
              f"- Human review status: training {train['human_reviews']}; calibration {scores['human_reviews']}",
              "- Checkpoint identity: `" + json.dumps(identity, sort_keys=True) + "`",
              "", "Final M7/M8 acceptance and human approvals are separate; this card does not invent them.", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as target:
        target.write("\n".join(lines))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.model, args.calibration, args.schema, args.out))


if __name__ == "__main__":
    main()

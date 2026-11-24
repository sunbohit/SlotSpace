"""M5 masked-context cross-encoder training; GPU execution is opt-in via this CLI."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from .artifacts import logged_run, new_directory, read_jsonl, sha256, write_json
from .schema import Schema


def split_templates(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    families = defaultdict(set)
    for row in samples:
        families[(row["slot"], row["style"])].add(row["template_id"])
    heldout = set()
    for group, ids in families.items():
        if len(ids) < 2:
            raise ValueError(f"Need at least two template IDs for an honest dev split: {group}")
        ordered = sorted(ids, key=lambda x: hashlib.sha256(f"13:{x}".encode()).hexdigest())
        heldout.update(ordered[:max(1, math.ceil(0.1 * len(ordered)))])
    return ([r for r in samples if r["template_id"] not in heldout],
            [r for r in samples if r["template_id"] in heldout])


def training_pairs(samples: list[dict], schema: Schema, *, heldout_slot: str | None = None):
    pairs = {"context": [], "slot_text": [], "label": []}
    for sample in samples:
        if sample["slot"] == heldout_slot:
            continue
        kinds = {sample["value_type"]}
        if sample["untyped"] and sample["value_type"] in {"disease", "effect_phenotype"}:
            kinds = {"disease", "effect_phenotype"}
        candidates = schema.candidates(sample["doc_type"] if sample["style"] == "doc" else None, kinds)
        candidates = [s for s in candidates if s != heldout_slot]
        if sample["slot"] not in candidates:
            raise ValueError(f"True training slot is outside its candidate set: {sample['sample_id']}")
        for slot in candidates:
            pairs["context"].append(sample["context"])
            pairs["slot_text"].append(schema.slot_text(slot))
            pairs["label"].append(float(slot == sample["slot"]))
    positives = sum(pairs["label"])
    if not positives or positives == len(pairs["label"]):
        raise ValueError("Training requires both positive and negative slot pairs")
    return pairs, (len(pairs["label"]) - positives) / positives


def train(data: Path, schema_path: Path, out: Path, model: str, *, variant: str = "main",
          heldout_slot: str = "drug.off_label_use.disease", device: str = "cuda",
          epochs: int = 2, batch_size: int = 32) -> Path:
    if variant not in {"main", "a12", "a14"} or epochs < 1 or batch_size < 1:
        raise ValueError("Invalid training variant, epoch count or batch size")
    schema = Schema.load(schema_path)
    provenance = json.loads((data.parent / "manifest.json").read_text(encoding="utf-8"))
    if provenance["schema_sha256"] != sha256(schema_path):
        raise ValueError("Training data/schema version mismatch")
    samples = list(read_jsonl(data))
    expected_variant = "a12" if variant == "a12" else "masked"
    if not samples or any(r["variant"] != expected_variant for r in samples):
        raise ValueError("Masked main/A14 and visible A12 training data cannot be interchanged")
    train_samples, dev_samples = split_templates(samples)
    excluded = heldout_slot if variant == "a14" else None
    train_rows, pos_weight = training_pairs(train_samples, schema, heldout_slot=excluded)
    dev_rows, _ = training_pairs(dev_samples, schema)
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no automatic CPU training fallback")
    torch.manual_seed(13)
    config = {"variant": variant, "data_sha256": sha256(data), "schema_sha256": sha256(schema_path),
              "schema_version": schema.version, "model": model, "epochs": epochs, "batch_size": batch_size,
              "lr": 2e-5, "warmup_ratio": 0.1, "max_length": 128, "seed": 13,
              "heldout_slot": excluded, "positive_weight": pos_weight, "human_reviews": "pending"}
    cadence = min(1000, math.ceil(len(train_rows["label"]) / batch_size))
    with logged_run(Path("runs"), f"m5-train-{variant}", config):
        with new_directory(out) as stage:
            encoder = CrossEncoder(model, num_labels=1, max_length=128, device=device,
                                   local_files_only=Path(model).is_dir())
            loss = BinaryCrossEntropyLoss(encoder, pos_weight=torch.tensor(pos_weight))
            args = CrossEncoderTrainingArguments(
                output_dir=str(stage / "checkpoints"), num_train_epochs=epochs,
                per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                learning_rate=2e-5, warmup_ratio=0.1, bf16=device == "cuda", use_cpu=device == "cpu",
                eval_strategy="steps", save_strategy="steps", eval_steps=cadence, save_steps=cadence,
                load_best_model_at_end=True, metric_for_best_model="eval_loss", seed=13, data_seed=13,
                report_to=[], save_safetensors=True,
            )
            trainer = CrossEncoderTrainer(model=encoder, args=args, train_dataset=Dataset.from_dict(train_rows),
                                           eval_dataset=Dataset.from_dict(dev_rows), loss=loss)
            trainer.train()
            encoder.save_pretrained(str(stage / "final"))
            write_json(stage / "manifest.json", {**config, "train_pairs": len(train_rows["label"]),
                                                "dev_pairs": len(dev_rows["label"]),
                                                "train_templates": sorted({r["template_id"] for r in train_samples
                                                                           if r["slot"] != excluded}),
                                                "dev_templates": sorted({r["template_id"] for r in dev_samples})})
    return out / "final"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--variant", choices=("main", "a12", "a14"), default="main")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(train(args.data, args.schema, args.out, args.model, variant=args.variant, device=args.device,
                epochs=args.epochs, batch_size=args.batch_size))


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from safetensors.numpy import save_file

from slotspace.artifacts import sha256, write_json
from slotspace.model_card import generate
from slotspace.schema import Schema
from slotspace.slot_scorer import ece, local_model_identity

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "prime.yaml"


def _inputs(root):
    model = root / "synthetic-checkpoint" / "final"
    model.mkdir(parents=True)
    write_json(model / "config.json", {"model_type": "modernbert",
                                      "architectures": ["ModernBertForSequenceClassification"]})
    save_file({"classifier.weight": np.ones((1, 2), dtype=np.float32)}, model / "model.safetensors")
    write_json(model.parent / "manifest.json", {
        "variant": "main", "schema_sha256": sha256(SCHEMA), "schema_version": Schema.load(SCHEMA).version,
        "model": "answerdotai/ModernBERT-base", "seed": 13, "epochs": 1, "batch_size": 2,
        "train_pairs": 4, "dev_pairs": 2, "train_templates": ["synthetic-train"],
        "dev_templates": ["synthetic-dev"], "data_sha256": "1" * 64, "human_reviews": "pending"})
    points = [{"style": style, "confidence_before": 0.9,
               "confidence_after": 0.6 if style == "query" else 0.5, "correct": i < 2}
              for style in ("query", "doc") for i in range(4)]
    calibration = root / "temperature.json"
    pq.write_table(pa.Table.from_pylist(points), calibration.with_suffix(".points.parquet"))
    result = {"scorer": {"model": local_model_identity(model), "pair_order": "context_first",
                         "max_length": 128, "values_visible": False},
              "query_source": "train/even qid", "schema_sha256": sha256(SCHEMA),
              "schema_version": Schema.load(SCHEMA).version, "T_query": 2.0, "T_doc": 1.5,
              "human_reviews": "pending"}
    for style in ("query", "doc"):
        rows = [r for r in points if r["style"] == style]
        correct = [r["correct"] for r in rows]
        result[style] = {"n": 4, "accuracy": 0.5,
                         **{f"ece_{s}": ece([r[f"confidence_{s}"] for r in rows], correct)
                            for s in ("before", "after")}}
    write_json(calibration, result)
    return model, calibration


def test_card_uses_actual_calibration_points_and_keeps_negative_results_and_pending_review(tmp_path):
    model, calibration = _inputs(tmp_path)
    out = tmp_path / "card.md"
    generate(model, calibration, SCHEMA, out)
    text = out.read_text(encoding="utf-8")
    assert "| query | 4 | 0.500000 | 0.400000 | 0.100000 |" in text
    assert "target: FAIL (not an H4 approval)" in text and "calibration pending" in text
    assert sha256(calibration.with_suffix(".points.parquet")) in text
    with pytest.raises(FileExistsError):
        generate(model, calibration, SCHEMA, out)


@pytest.mark.parametrize("invalid", ["weights", "summary", "split", "temperature", "templates"])
def test_card_rejects_mismatched_or_ineligible_evidence_before_writing(tmp_path, invalid):
    model, calibration = _inputs(tmp_path)
    if invalid == "weights":
        save_file({"classifier.weight": np.zeros((1, 2), dtype=np.float32)}, model / "model.safetensors")
    elif invalid == "templates":
        manifest = model.parent / "manifest.json"
        value = json.loads(manifest.read_text())
        value["dev_templates"] = value["train_templates"]
        write_json(manifest, value)
    else:
        value = json.loads(calibration.read_text())
        if invalid == "summary":
            value["query"]["accuracy"] = 0.75
        elif invalid == "split":
            value["query_source"] = "val"
        else:
            value["T_query"] = 0
        write_json(calibration, value)
    out = tmp_path / "card.md"
    with pytest.raises(ValueError):
        generate(model, calibration, SCHEMA, out)
    assert not out.exists()

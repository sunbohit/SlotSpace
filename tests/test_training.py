import json

import pytest

from slotspace.artifacts import sha256, write_json
from slotspace.calibrate_slot_scorer import document_records, query_records
from slotspace.schema import Schema
from slotspace.train_slot_scorer import split_templates, train, training_pairs
from test_schema import SCHEMA

IND, OFF = "drug.indication.disease", "drug.off_label_use.disease"


def row(slot, template=0):
    return {"sample_id": f"{slot}:{template}", "slot": slot, "value_type": "disease", "style": "doc",
            "doc_type": "drug", "template_id": f"{slot}:doc:{template}", "variant": "masked",
            "context": "[SELF:drug] treats [TARGET].", "untyped": True}


def test_a14_excludes_the_heldout_description_as_a_negative_and_splits_by_template():
    schema = Schema.load(SCHEMA)
    rows = [row(s, t) for s in (IND, OFF) for t in range(12) for _ in range(2)]
    train_rows, dev_rows = split_templates(rows)
    assert {r["template_id"] for r in train_rows}.isdisjoint(r["template_id"] for r in dev_rows)
    pairs, weight = training_pairs(train_rows, schema, heldout_slot=OFF)
    assert list(pairs) == ["context", "slot_text", "label"] and weight > 0
    assert all(not text.startswith(OFF + ":") for text in pairs["slot_text"])
    assert any(text.startswith("none.effect_phenotype:") for text in pairs["slot_text"])
    assert sum(pairs["label"]) == sum(r["slot"] != OFF for r in train_rows)


def test_main_training_rejects_a12_before_loading_any_model(tmp_path):
    data = tmp_path / "validated.jsonl"
    sample = row(IND)
    sample["variant"] = "a12"
    data.write_text(json.dumps(sample) + "\n")
    write_json(tmp_path / "manifest.json", {"schema_sha256": sha256(SCHEMA)})
    with pytest.raises(ValueError, match="cannot be interchanged"):
        train(data, SCHEMA, tmp_path / "out", "never-load-this-model", variant="main")


def test_query_calibration_uses_only_train_even_ids_and_doc_calibration_uses_native_truth():
    schema = Schema.load(SCHEMA)
    silver = [{"qid": 2, "segments": [{"start": 0, "end": 6, "links": {"n4": 1.0}, "true_readings": [f"{IND}|n4"]}]}]
    records = query_records(silver, {2: {"query": "asthma is mentioned"}}, {4: "disease"}, schema)
    assert records[0]["context"] == "[TARGET:disease] is mentioned" and records[0]["true_slots"] == [IND]
    silver[0]["qid"] = 3
    with pytest.raises(ValueError, match="even-qid"):
        query_records(silver, {3: {"query": "asthma"}}, {4: "disease"}, schema)
    fact = {"fact_id": "1.t0", "doc_id": 1, "mentions": [
        {"start": 0, "end": 6, "value_ids": ["n4"], "value_type": "disease",
         "scorer_input": "[SELF:drug] treats [TARGET:disease]."}]}
    docs = document_records([fact], {1: {f"{IND}|n4"}}, {1: "drug"}, schema, limit=1)
    assert len(docs) == 1 and docs[0]["true_slots"] == [IND]

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from slotspace.artifacts import sha256, write_json
from slotspace.linker import Linker
from slotspace.schema import Schema
from slotspace.silver import build_silver, silver_segments
from test_schema import SCHEMA

IND, CON = "drug.indication.disease", "drug.contraindication.disease"


def test_majority_labels_bind_the_slot_to_the_correct_ambiguous_value():
    docs = pd.DataFrame({"doc_id": [4, 5], "name": ["hypertension", "hypertension"],
                         "doc_type": ["disease", "disease"]})
    linker = Linker(docs, common_words=0)
    native = {10: {f"{IND}|n4", f"{CON}|n5"}, 20: {f"{IND}|n4"}, 30: {f"{CON}|n4"}}
    query = {"query": "hypertension", "answers": [10, 20, 30]}
    rows = silver_segments(query, linker, Schema.load(SCHEMA), native, {10: "drug", 20: "drug", 30: "drug"})
    assert rows[0]["links"] == {"n4": 0.5, "n5": 0.5}
    assert rows[0]["true_readings"] == [f"{IND}|n4"]
    query["answers"] = [10, 20, 20, 20]
    rows = silver_segments(query, linker, Schema.load(SCHEMA), native, {10: "drug", 20: "drug"})
    assert rows[0]["true_readings"] == sorted([f"{IND}|n4", f"{CON}|n5"])


def test_train_calibration_split_uses_qid_parity_not_csv_position(tmp_path):
    data, facts, runs = (tmp_path / name for name in ("data", "facts", "runs"))
    data.mkdir()
    facts.mkdir()
    docs = data / "docs.parquet"
    pq.write_table(pa.table({"doc_id": [4, 10, 20], "name": ["hypertension", "alpha drug", "beta drug"],
                            "doc_type": ["disease", "drug", "drug"]}), docs)
    pq.write_table(pa.table({"doc_id": [10, 20], "key": [f"{IND}|n4", f"{CON}|n4"]}), facts / "native_keys.parquet")
    write_json(facts / "manifest.json", {"schema_sha256": sha256(SCHEMA), "docs_sha256": sha256(docs)})
    for split, ids in (("train", [3, 2]), ("val", [4])):
        (data / f"queries_{split}.jsonl").write_text("".join(
            json.dumps({"qid": i, "query": "hypertension", "answers": [10]}) + "\n" for i in ids))
    config = {"dataset": "prime", "seed": 13, "schema": str(SCHEMA), "paths": {"data": str(data)},
              "linking": {"max_ngram": 8, "min_name_chars": 3, "common_words": 0,
                          "fuzzy_threshold": 92, "fuzzy_mass": 0.5}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    out = build_silver(path, facts, tmp_path / "silver", runs=runs)
    assert json.loads((out / "silver_calibration.jsonl").read_text())["qid"] == 2
    assert json.loads((out / "silver_spare.jsonl").read_text())["qid"] == 3
    assert json.loads((out / "silver_val.jsonl").read_text())["qid"] == 4
    assert not list(out.glob("*test*"))

import json

import pandas as pd
import pyarrow.parquet as pq

from slotspace.facts import TEXT_SCHEMA, document_facts, write_native_keys, write_text_facts
from slotspace.linker import Linker
from slotspace.schema import Schema
from test_schema import SCHEMA


def test_native_keys_preserve_both_ends_and_drop_only_duplicate_edges(tmp_path):
    docs = pd.DataFrame({"doc_id": [17, 91], "doc_type": ["drug", "disease"]})
    edges = pd.DataFrame({"src": [17, 91, 17], "dst": [91, 17, 91], "rel": ["indication"] * 3})
    path = tmp_path / "native.parquet"
    assert write_native_keys(docs, edges, Schema.load(SCHEMA), path) == 2
    assert pq.read_table(path).to_pylist() == [
        {"doc_id": 17, "key": "drug.indication.disease|n91"},
        {"doc_id": 91, "key": "disease.indication.drug|n17"},
    ]


def test_ambiguous_values_stay_masked_but_do_not_create_fact_keys(tmp_path):
    docs = pd.DataFrame({"doc_id": [1, 2, 3, 4, 5], "name": [
        "Aspirin", "asthma", "asthma", "hypertension", "hypertension"],
        "doc_type": ["drug", "disease", "disease", "disease", "effect_phenotype"]})
    linker = Linker(docs, common_words=0)
    doc = {"doc_id": 1, "name": "Aspirin", "doc_type": "drug", "details_json": json.dumps({
        "contraindications": "Do not use Aspirin in asthma or hypertension."})}
    facts, seen, ambiguous = document_facts(doc, linker)
    assert seen == 2 and ambiguous == 1 and len(facts) == 1
    fact = facts[0]
    assert fact["fact_id"] == "1.t0" and "[SELF:drug]" in fact["ctx_text"]
    assert "asthma" not in fact["ctx_text"] and "[disease]" in fact["ctx_text"]
    assert {m["value_type"] for m in fact["mentions"]} == {"disease", "effect_phenotype"}
    assert {v for m in fact["mentions"] for v in m["value_ids"]} == {"n4", "n5"}
    assert all("[TARGET]" in m["scorer_input"] and "contraindications" not in m["scorer_input"]
               for m in fact["mentions"])
    path = tmp_path / "text.parquet"
    stats, review = write_text_facts(pd.DataFrame([doc]), linker, path)
    assert stats["text_facts"] == 1 and stats["mentions_per_fact"]["mean"] == 1
    assert len(review) == 1 and pq.read_schema(path) == TEXT_SCHEMA
    assert pq.read_table(path)["row"].to_pylist() == [0]
    doc["details_json"] = json.dumps({"description": "Hypertension occurs. Hypertension remains. Hypertension recurs."})
    capped, _, _ = document_facts(doc, linker, cap=2)
    assert len(capped) == 2 and [f["fact_id"] for f in capped] == ["1.t0", "1.t1"]

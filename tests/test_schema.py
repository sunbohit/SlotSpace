from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from slotspace.schema import Schema, derive_slots, verify_inventory

SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "prime.yaml"


def test_joint_candidates_include_none_per_type_and_respect_document_type():
    schema = Schema.load(SCHEMA, expected_version="0.1.0")
    candidates = schema.candidates("drug", {"disease", "effect_phenotype"})
    assert set(candidates) == {
        "drug.indication.disease", "drug.contraindication.disease", "drug.off_label_use.disease",
        "drug.side_effect.effect_phenotype", "none.disease", "none.effect_phenotype",
    }
    assert set(candidates) < set(schema.candidates(None, {"disease", "effect_phenotype"}))
    assert schema.slot_text("none.disease").startswith("none.disease: ")
    with pytest.raises(ValueError, match="version mismatch"):
        Schema.load(SCHEMA, expected_version="9.0.0")


@pytest.mark.parametrize("change", ["duplicate", "pipe", "description"])
def test_invalid_schema_is_rejected(tmp_path, change):
    data = yaml.safe_load(SCHEMA.read_text())
    if change == "duplicate":
        data["slots"].append(deepcopy(data["slots"][0]))
    elif change == "pipe":
        data["slots"][0]["id"] += "|bad"
    else:
        data["slots"][0]["description"] = ""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        Schema.load(path)


def test_slot_derivation_uses_real_ids_and_verifies_reciprocity():
    docs = pd.DataFrame({"doc_id": [17, 91], "doc_type": ["drug", "disease"]})
    edges = pd.DataFrame({"src": [17, 91], "dst": [91, 17], "rel": ["off-label use"] * 2})
    slots = derive_slots(docs, edges)
    assert {s["id"] for s in slots} == {"drug.off_label_use.disease", "disease.off_label_use.drug"}
    assert all(s["direction"] == "out" and s["n_edges"] == 1 for s in slots)
    with pytest.raises(ValueError, match="symmetrically"):
        derive_slots(docs, edges.iloc[:1])
    with pytest.raises(ValueError, match="slot sets differ"):
        verify_inventory(Schema.load(SCHEMA), slots)

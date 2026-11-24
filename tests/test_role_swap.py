from pathlib import Path

import pandas as pd

from slotspace.role_swap import DRUG_SLOTS, SIDE, h, role_swap_rows
from slotspace.text import normalize


def test_cross_type_name_group_overrides_value_split_and_keeps_all_partners_together():
    name = "hypertension"
    disease = next(i for i in range(1, 100) if h(f"n{i}") % 2 != h(name) % 2)
    phenotype = 200
    docs = pd.DataFrame({"doc_id": [disease, phenotype], "name": ["Hypertension.", name],
                         "doc_type": ["disease", "effect_phenotype"]})
    postings = {f"{slot}|n{disease}": (1, 2, 3) for slot in DRUG_SLOTS}
    postings[f"{SIDE}|n{phenotype}"] = (2, 3, 4)
    templates = {slot: [f"{slot} {{V}}", f"another {slot} {{V}}"] for slot in (*DRUG_SLOTS, SIDE)}
    dev, test = role_swap_rows(docs, postings, templates)
    rows = dev + test
    assert len(rows) == 4 and bool(test) == bool(h(name) % 2)
    assert len({q["qid"] for q in rows}) == 4
    for row in rows:
        assert row["answers"] == list(postings[f"{row['slot']}|{row['value_id']}"])
        partners = [r for r in rows if r["slot"] != row["slot"]]
        assert row["partners"] == sorted(r["qid"] for r in partners)
        assert row["partner_answers"] == sorted({a for r in partners for a in r["answers"]})
        assert row["template_id"].endswith(str(h(f"{row['value_id']}|{row['slot']}") % 2))
    assert role_swap_rows(docs, postings, templates) == (dev, test)


def test_support_fallback_changes_eligibility_without_trimming_answer_sets():
    docs = pd.DataFrame({"doc_id": [1], "name": ["asthma"], "doc_type": ["disease"]})
    postings = {f"{slot}|n1": (8, 11) for slot in DRUG_SLOTS[:2]}
    templates = {slot: ["A separate role question about {V}"] for slot in (*DRUG_SLOTS, SIDE)}
    assert role_swap_rows(docs, postings, templates, lo=3) == ([], [])
    dev, test = role_swap_rows(docs, postings, templates, lo=2)
    assert len(dev + test) == 2 and all(r["answers"] == [8, 11] for r in dev + test)


def test_test_only_template_path_is_not_read_by_training_or_other_modules():
    source = Path(__file__).resolve().parents[1] / "src" / "slotspace"
    readers = [p.name for p in source.rglob("*.py") if "role_swap.yaml" in p.read_text(encoding="utf-8")]
    assert readers == ["role_swap.py"]

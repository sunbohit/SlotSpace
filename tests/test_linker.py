import pandas as pd

from slotspace.linker import Linker


def test_longest_match_cross_type_probabilities_and_case_sensitive_common_symbols(monkeypatch):
    monkeypatch.setattr("slotspace.linker.top_n_list", lambda *args: ["the", "cat", "and"])
    docs = pd.DataFrame({
        "doc_id": [1, 2, 3, 4, 5, 6], "name": [
            "hypertension", "hypertension", "pulmonary hypertension", "CAT", "123-456", "ab"],
        "doc_type": ["disease", "effect_phenotype", "disease", "gene_protein", "drug", "drug"],
    })
    linker = Linker(docs)
    mentions = linker.link("Pulmonary hypertension and hypertension")
    assert [m["values"] for m in mentions] == [{"n3": 1.0}, {"n1": 1.0, "n2": 1.0}]
    assert mentions[1]["types"] == {"disease", "effect_phenotype"}
    assert not linker.link("the cat sat 123-456 ab")
    assert linker.link("CAT activity")[0]["values"] == {"n4": 1.0}


def test_fuzzy_fallback_is_query_only_half_mass_and_does_not_overlap_exact_matches():
    docs = pd.DataFrame({"doc_id": [1, 2, 3], "name": [
        "pulmonary hypertension", "pulmonary hypertension", "Aspirin"],
        "doc_type": ["disease", "disease", "drug"]})
    linker = Linker(docs, common_words=0)
    text = "pulmonry hypertension and Aspirin"
    assert [m["surface"] for m in linker.link(text)] == ["Aspirin"]
    mentions = linker.link_query(text)
    assert mentions[0]["fuzzy"] and mentions[0]["values"] == {"n1": 0.25, "n2": 0.25}
    assert mentions[0]["end"] <= mentions[1]["start"] and not mentions[1]["fuzzy"]

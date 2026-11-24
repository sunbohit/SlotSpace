import json

import numpy as np
import pandas as pd
import pytest

from slotspace.artifacts import sha256, write_json
from slotspace.lexicon_build import CorpusLexicon, aggregate_lexicon
from slotspace.reference import Doc, Fact, Lexicon
from slotspace.schema import Schema
from test_fast_scorer import random_corpus
from test_schema import SCHEMA

IND, CON = "drug.indication.disease", "drug.contraindication.disease"


def test_columnar_doc_max_matches_reference_and_native_overrides_text():
    native = pd.DataFrame({"doc_id": [1, 1, 2], "key": [f"{IND}|n4", f"{IND}|n4", f"{CON}|n4"]})
    text = pd.DataFrame([
        (1, f"{IND}|n4", 0.2, "1.t0"), (1, f"{CON}|n4", 0.3, "1.t0"),
        (1, f"{CON}|n4", 0.7, "1.t1"), (2, f"{IND}|n4", 0.01, "2.t0"),
    ], columns=["doc_id", "key", "pi", "fact_id"])
    result, _, _ = aggregate_lexicon(native, Schema.load(SCHEMA), text)
    lex = Lexicon()
    lex.upsert(Doc("1", [Fact("n", {f"{IND}|n4": 1}, None, True),
                         Fact("1.t0", {f"{IND}|n4": 0.2, f"{CON}|n4": 0.3}, None),
                         Fact("1.t1", {f"{CON}|n4": 0.7}, None)]))
    lex.upsert(Doc("2", [Fact("n", {f"{CON}|n4": 1}, None, True), Fact("2.t0", {f"{IND}|n4": 0.01}, None)]))
    actual = {(r.value_id, r.slot): (r.soft, r.hard) for r in result.itertuples()}
    for key, (soft, hard) in lex.snapshot().items():
        assert actual[key][1] == hard and np.isclose(actual[key][0], soft)
    assert all(len(e) <= 5 for e in result.evidence)


def test_pacc_denominator_includes_mentions_with_no_surviving_role_key(caplog):
    native = pd.DataFrame({"doc_id": [1, 2], "key": [f"{IND}|n4", f"{CON}|n4"]})
    text = pd.DataFrame([
        (1, f"{IND}|n4", 0.8, "1.t0"), (2, f"{IND}|n4", 0.2, "2.t0"),
        (3, f"{IND}|n4", 0.4, "3.t0"), (1, f"{CON}|n4", 0.1, "1.t0"),
        (2, f"{CON}|n4", 0.9, "2.t0"),
    ], columns=["doc_id", "key", "pi", "fact_id"])
    pairs = pd.DataFrame({"doc_id": [1, 2, 3, 5], "value_id": ["n4"] * 4,
                          "doc_type": ["drug"] * 4, "value_type": ["disease"] * 4})
    result, rates, ds = aggregate_lexicon(native, Schema.load(SCHEMA), text, pairs)
    ind = result.set_index("slot").loc[IND]
    assert np.isclose(ind.soft, 1.6) and ind.hard == 3 and np.isclose(ind.soft_pacc, 1)
    assert rates[IND]["tpr"] == 0.8 and rates[IND]["fpr"] == 0.2
    assert len(ds) == 6 and "raw text counts" in caplog.text


def test_incremental_delete_1000_and_reinsert_agrees_with_a_rebuild():
    rng = np.random.default_rng(13)
    docs = random_corpus(rng, n_docs=1200)
    order = rng.permutation(len(docs))
    lex = Lexicon()
    for half in np.array_split(order, 2):
        for i in half:
            lex.upsert(docs[i])
    for i in order[:1000]:
        lex.delete(docs[i].doc_id)
    for i in order[:1000]:
        lex.upsert(docs[i])
    rebuilt = Lexicon()
    for doc in docs:
        rebuilt.upsert(doc)
    a, b = lex.snapshot(), rebuilt.snapshot()
    assert a.keys() == b.keys()
    assert all(a[k][1] == b[k][1] and np.isclose(a[k][0], b[k][0], rtol=0, atol=1e-9) for k in a)


def test_loaded_lexicon_rejects_stale_versions_partial_caches_and_corruption(tmp_path):
    frame = pd.DataFrame({"value_id": ["n4"], "slot": [IND], "soft": [1.0], "soft_pacc": [1.0],
                          "hard": [1], "evidence": [["1.n"]]})
    path = tmp_path / "lexicon.parquet"
    frame.to_parquet(path, index=False)
    meta = {"file": path.name, "lexicon_sha256": sha256(path), "mode": "native", "complete": True,
            "index_version": "v1", "lexicon_version": "v1"}
    write_json(tmp_path / "manifest.json", meta)
    assert CorpusLexicon.load(tmp_path, "v1").hard_support("n4", IND) == 1
    with pytest.raises(ValueError, match="version"):
        CorpusLexicon.load(tmp_path, "v2")
    meta["complete"] = False
    write_json(tmp_path / "manifest.json", meta)
    with pytest.raises(ValueError, match="Partial"):
        CorpusLexicon.load(tmp_path)
    meta["complete"] = True
    write_json(tmp_path / "manifest.json", meta)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        CorpusLexicon.load(tmp_path)

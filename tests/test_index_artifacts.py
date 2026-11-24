import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from slotspace.artifacts import sha256, write_json
from slotspace.facts import NATIVE_SCHEMA, TEXT_SCHEMA
from slotspace.embed import pack_contexts
from slotspace.lexicon_build import build_lexicon
from slotspace.pipeline import load_index
from slotspace.slot_scorer import KEY_SCHEMA
from slotspace.reference import Reading
from test_schema import SCHEMA

IND, CON = "drug.indication.disease", "drug.contraindication.disease"


def test_combined_lexicon_and_fact_index_share_the_exact_snapshot(tmp_path):
    docs, facts, scores, vectors = (tmp_path / name for name in ("docs.parquet", "facts", "scores", "context"))
    for folder in (facts, scores, vectors):
        folder.mkdir()
    pq.write_table(pa.table({"doc_id": [4, 17, 91], "name": ["asthma", "alpha", "beta"],
                            "doc_type": ["disease", "drug", "drug"]}), docs)
    pq.write_table(pa.Table.from_pylist([{"doc_id": 17, "key": f"{IND}|n4"},
                                       {"doc_id": 91, "key": f"{CON}|n4"}], schema=NATIVE_SCHEMA),
                   facts / "native_keys.parquet")
    pq.write_table(pa.Table.from_pylist([{
        "row": 0, "doc_id": 91, "fact_id": "91.t0", "field": "details", "sentence": "asthma",
        "ctx_text": "field: details | [disease]", "mentions": [{
            "value_ids": ["n4"], "value_type": "disease", "start": 0, "end": 6,
            "scorer_input": "[TARGET:disease]"}]}], schema=TEXT_SCHEMA), facts / "text_facts.parquet")
    write_json(facts / "manifest.json", {"schema_sha256": sha256(SCHEMA), "docs_sha256": sha256(docs)})
    keys = scores / "text_fact_keys_zeroshot.parquet"
    pq.write_table(pa.Table.from_pylist([{"row": 0, "key": f"{IND}|n4", "pi": 0.3}], schema=KEY_SCHEMA), keys)
    write_json(scores / "manifest.json", {
        "complete": True, "kind": "zeroshot", "schema_sha256": sha256(SCHEMA), "docs_sha256": sha256(docs),
        "text_facts_sha256": sha256(facts / "text_facts.parquet"),
        "scorer": {"model": "fixture", "temperature": 1, "pair_order": "slot_first"}})
    vector = np.zeros((1, 128), dtype=np.float16)
    vector[0, 0] = 1
    np.save(vectors / "text_fact_ctx_qwen3e06_128.npy", vector)
    write_json(vectors / "manifest.json", {"docs_sha256": sha256(facts / "text_facts.parquet"),
                                          "schema_version": "0.1.0", "rows": 1, "dimensions": [128]})
    lexicon = tmp_path / "lexicon"
    build_lexicon(facts, docs, SCHEMA, lexicon, text_keys=keys, runs=tmp_path / "runs")
    index, lex = load_index(docs, facts, keys, vectors, lexicon, SCHEMA)
    assert index.version == lex.version and index.doc_ids.tolist() == [4, 17, 91]
    assert index.postings[f"{IND}|n4"].tolist() == [17, 91] and lex.hard_support("n4", IND) == 2
    assert index.store.ctx.shape == (1, 128)
    packed_path = pack_contexts(vectors, tmp_path / "packed")
    assert np.load(packed_path).dtype == np.int8 and np.load(packed_path)[0, 0] == -128
    binary, _ = load_index(docs, facts, keys, vectors, lexicon, SCHEMA, "binary")
    packed, _ = load_index(docs, facts, keys, tmp_path / "packed", lexicon, SCHEMA, "binary")
    reading = Reading("r", 0, {f"{IND}|n4": 1}, 1, np.eye(128)[0])
    assert binary.fingerprint == packed.fingerprint
    from slotspace.reference import ScoreConfig
    assert np.array_equal(binary.store.scores(["17", "91"], [reading], ScoreConfig(precision="binary")),
                          packed.store.scores(["17", "91"], [reading], ScoreConfig(precision="binary")))
    with pytest.raises(ValueError, match="floating"):
        load_index(docs, facts, keys, tmp_path / "packed", lexicon, SCHEMA, "float")
    np.save(vectors / "text_fact_ctx_qwen3e06_128.npy", np.zeros_like(vector))
    with pytest.raises(ValueError, match="norms"):
        load_index(docs, facts, keys, vectors, lexicon, SCHEMA)

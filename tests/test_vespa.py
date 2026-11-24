import json
import re
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from slotspace.pipeline import CorpusIndex
from slotspace.planner import plan_readings
from slotspace.reference import Doc, Fact, ScoreConfig, binarize, binding_score, unit, unpack_pm1
from slotspace.schema import Schema
from slotspace.vespa_app.application import EXPONENTS, profile_name
from slotspace.vespa_app.codec import engine_key
from slotspace.vespa_app.feed import export_feed, verify_bundle, vespa_doc
from slotspace.vespa_app.parity import compare_binding
from slotspace.vespa_app.query import fuse_hits, vespa_body
from slotspace.vespa_app.smoke import IND, smoke_fixture
from test_schema import SCHEMA


def _wire_binding(operation, body, a, b):
    """Evaluate the serialized sparse joins, not the original Fact/Reading objects."""
    data = operation["fields"]
    rk = [(rid, key, float(value)) for rid, key, value in
          re.findall(r"\{r:(r\d+),key:([a-z0-9_]+)\}:([^,}]+)", body["input.query(rk)"])]
    nk = {c["address"]["key"] for c in data["nk"]["cells"]}
    tk = {(c["address"]["fact"], c["address"]["key"]): c["value"] for c in data["tk"]["cells"]}
    native, text = {}, {}
    for rid, key, link in rk:
        if key in nk:
            native[rid] = max(native.get(rid, 0), link)
        for label, packed in data["tc"]["blocks"].items():
            if (label, key) in tk:
                context = unpack_pm1(packed)
                similarity = max(0, np.dot(body["input.query(g)"][rid], context) / np.sqrt(128))
                score = link * tk[label, key] ** a * similarity ** b
                text[rid] = max(text.get(rid, 0), score)
    weights = body["input.query(w)"]
    return (sum(weights[r] * s for r, s in native.items()) + sum(weights[r] * s for r, s in text.items())
            - sum(weights[r] * min(native[r], text[r]) for r in native.keys() & text.keys()))


@pytest.mark.parametrize("a,b", EXPONENTS)
def test_serialized_sparse_tensors_preserve_binary_binding_for_each_profile(a, b):
    operations, reference, plan, topic, slots = smoke_fixture()
    body = vespa_body(plan, "hypertension", topic, slots, a, b)
    assert body["ranking"] == profile_name(a, b)
    readings = plan_readings(plan)
    for op in operations:
        expected = binding_score(reference[op["fields"]["doc_id"]], readings,
                                 ScoreConfig(a=a, b=b, precision="binary"))
        assert _wire_binding(op, body, a, b) == pytest.approx(expected, abs=1e-6)
    assert "dotProduct" in body["yql"]
    assert json.loads(json.dumps(body)) == body


def test_packing_thresholds_and_empty_facts_do_not_create_false_postings():
    topic, ctx = np.zeros(256), np.zeros(128)
    topic[0], ctx[0] = 1, 1
    key = f"{IND}|n4"
    op = vespa_doc(17, "drug", "", topic, [], [("t0", {key: 0.049}, ctx)], {IND: 5})
    assert op["fields"]["tc"]["blocks"]["t0"] == [-128] + [0] * 15
    assert op["fields"]["vkeys"] == {} and op["fields"]["tk"]["cells"] == []
    assert vespa_doc(17, "drug", "", topic, [], [], {IND: 5})["fields"]["tc"] == {"blocks": {}}
    for bad in (f"{IND}|n4|other", f"{IND}|n4 injection", "unlisted|n4"):
        with pytest.raises(ValueError):
            engine_key(bad, {IND: 5})
    with pytest.raises(ValueError, match="distinct"):
        vespa_doc(17, "drug", "", topic, [], [("t0", {}, ctx), ("t0", {}, ctx)], {IND: 5})
    with pytest.raises(ValueError, match="probabilities"):
        vespa_doc(17, "drug", "", topic, [], [("t0", {key: np.nan}, ctx)], {IND: 5})


def test_query_fallback_duplicate_readings_and_true_pool_normalization():
    _, _, plan, topic, slots = smoke_fixture()
    fallback = vespa_body({**plan, "segments": []}, "", topic, slots)
    assert fallback["ranking"] == "hybrid" and "input.query(rk)" not in fallback
    assert "dotProduct" not in fallback["yql"]
    bad = deepcopy(plan)
    bad["segments"][0]["readings"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        vespa_body(bad, "", topic, slots)
    with pytest.raises(ValueError, match="unit vector"):
        vespa_body(plan, "", np.zeros(256), slots)
    with pytest.raises(ValueError, match="IA"):
        vespa_body({**plan, "weight_objective": "ia"}, "", topic, slots)
    hits = [{"doc_id": 91, "bind": 0.7, "dense": 0.3, "bm25": 4.0},
            {"doc_id": 4, "bind": 0.6, "dense": 0.1, "bm25": 4.0},
            {"doc_id": 17, "bind": 0.5, "dense": 0.5, "bm25": 4.0}]
    result = fuse_hits(hits, plan, (1, 1, 100))
    assert result["doc_ids"] == [17, 91, 4]
    assert result["scores"] == pytest.approx([1.5, 1.2, 0.6])
    assert fuse_hits([], plan)["doc_ids"] == []
    # No readings means binding contributes nothing, even with unexpected nonzero features.
    assert fuse_hits(hits, {**plan, "segments": []})["doc_ids"] == [17, 91, 4]


def test_parity_checks_tolerance_and_order_not_only_final_fused_topk():
    _, reference, plan, _, _ = smoke_fixture()
    readings = plan_readings(plan)
    hits = [{"doc_id": i, "bind": binding_score(doc, readings, ScoreConfig(precision="binary"))}
            for i, doc in reference.items()]
    assert compare_binding(hits, plan, reference.__getitem__)["max_abs_difference"] == 0
    bad = deepcopy(hits)
    bad[-1]["bind"] += 0.002
    with pytest.raises(ValueError, match="parity failed"):
        compare_binding(bad, plan, reference.__getitem__)
    with pytest.raises(ValueError, match="returned candidates"):
        compare_binding([], plan, reference.__getitem__)
    ctx = unit(np.ones(128))
    p = {"segments": [{"g": ctx.tolist(), "links": {"n4": 1.0},
                       "readings": [{"key": f"{IND}|n4", "w": 1.0}]}]}
    docs = {i: Doc(str(i), [Fact("f", {f"{IND}|n4": probability}, binarize(ctx))])
            for i, probability in ((4, 0.4), (17, 0.4005))}
    with pytest.raises(ValueError, match="order inversions"):
        compare_binding([{"doc_id": 4, "bind": 0.4005}, {"doc_id": 17, "bind": 0.4}],
                        p, docs.__getitem__)


def test_feed_bundle_is_immutable_and_binds_the_schema_and_corpus_rows(tmp_path):
    operations, reference, _, topic, _ = smoke_fixture()
    # Feed contexts are floating point; the reference smoke documents contain packed bytes.
    docs = [Doc(str(i), [f if f.native else replace(f, ctx=unit(unpack_pm1(f.ctx))) for f in doc.facts])
            for i, doc in reference.items()]
    index = CorpusIndex(docs, 128, "fixture", "binary")
    path = tmp_path / "docs.parquet"
    pq.write_table(pa.Table.from_pylist([{"doc_id": op["fields"]["doc_id"], "doc_type": "drug", "text": "test"}
                                       for op in operations]), path)
    out = tmp_path / "bundle"
    schema = Schema.load(SCHEMA)
    export_feed(path, schema, index, np.tile(topic, (4, 1)), out, SCHEMA)
    manifest = verify_bundle(out)
    assert manifest["documents"] == 4 and len(manifest["profiles"]) == 7
    assert manifest["slot_index"] == {s.id: i for i, s in enumerate(schema.slots)}
    with pytest.raises(FileExistsError):
        export_feed(path, schema, index, np.tile(topic, (4, 1)), out, SCHEMA)
    with pytest.raises(ValueError, match="topic"):
        export_feed(path, schema, index, np.zeros((4, 256)), tmp_path / "bad", SCHEMA)
    assert not (tmp_path / "bad").exists()
    (out / "app" / "schemas" / "doc.sd").write_text("changed")
    with pytest.raises(ValueError, match="application changed"):
        verify_bundle(out)

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from slotspace.artifacts import read_jsonl, write_json
from slotspace.pipeline import CorpusIndex, Pipeline
from slotspace.planner import PlannerSettings
from slotspace.reference import Doc, Lexicon, ScoreConfig, unit, unpack_pm1
from slotspace.retrieve import BM25Index, DenseIndex
from slotspace.schema import Schema
from slotspace.vespa_app.application import EXPONENTS, profile_name
from slotspace.vespa_app.feed import export_feed, verify_bundle
from slotspace.vespa_app.measure import run_measurements
from slotspace.vespa_app.runtime import EngineBackend
from slotspace.vespa_app.smoke import smoke, smoke_fixture
from test_schema import SCHEMA
from test_vespa import _wire_binding


@pytest.fixture
def engine_fixture(tmp_path, monkeypatch):
    _, packed, plan, _, _ = smoke_fixture()
    schema = Schema.load(SCHEMA)
    docs = [Doc(str(i), [f if f.native else type(f)(f.fact_id, f.keys, unit(unpack_pm1(f.ctx)))
                        for f in doc.facts]) for i, doc in packed.items()]
    docs += [Doc(str(i), []) for i in (101, 103, 104, 202)]
    lex = Lexicon()
    for doc in docs:
        lex.upsert(doc)
    index = CorpusIndex(docs, 128, lex.version, "binary")
    topics = np.tile(unit(np.ones(256)), (len(docs), 1)).astype(np.float32)

    class Encoder:
        def queries(self, texts, dim, prompt=None):
            return np.tile(unit(np.ones(dim)), (len(texts), 1)).astype(np.float32)

    ctx = SimpleNamespace(lexicon=lex, schema=schema, embedder=Encoder(),
                          cfg=PlannerSettings(K_L=8, K_D=8, K_V=8))
    pipeline = Pipeline(ctx, index, BM25Index(["hypertension"] * len(docs)), DenseIndex(topics),
                        ScoreConfig(precision="binary"))
    plan.update(schema_version=schema.version, lexicon_version=lex.version, planner_encoder_texts=4,
                budgets={"K_L": 8, "K_D": 8, "K_V": 8})

    def planning(text, qid, context):
        return {**deepcopy(plan), "query_id": qid}

    monkeypatch.setattr("slotspace.vespa_app.runtime.plan_query", planning)
    monkeypatch.setattr("slotspace.pipeline.plan_query", planning)
    data = tmp_path / "data"
    data.mkdir()
    doc_path = data / "docs.parquet"
    pq.write_table(pa.table({"doc_id": index.doc_ids.tolist(), "doc_type": ["drug"] * len(docs),
                            "text": ["hypertension"] * len(docs), "name": ["hypertension"] * len(docs)}), doc_path)
    rows = [{"qid": 7, "query": "hypertension", "answers": [1]}, {"qid": 3, "query": "avoid hypertension", "answers": [2]}]
    (data / "queries_val.jsonl").write_text("".join(json.dumps(q) + "\n" for q in rows))
    (data / "queries_test.jsonl").write_text("MUST NOT BE READ")
    (data / "queries_human.jsonl").write_text("MUST NOT BE READ")
    bundle = tmp_path / "bundle"
    export_feed(doc_path, schema, index, pipeline.dense.documents, bundle, SCHEMA)
    operations = list(read_jsonl(bundle / "feed.jsonl"))
    manifest = verify_bundle(bundle)

    class Client:
        endpoint = "http://127.0.0.1:0"
        calls = []

        def version(self):
            return "fixture"

        def wait_count(self, count):
            assert count == len(docs)

        def search(self, body):
            self.calls.append(body)
            a, b = next((a, b) for a, b in EXPONENTS if body["ranking"] in {"hybrid", profile_name(a, b)})
            hits = [{"doc_id": op["fields"]["doc_id"],
                     "bind": 0.0 if body["ranking"] == "hybrid" else _wire_binding(op, body, a, b),
                     "dense": float(i) / len(docs), "bm25": float(i % 3)}
                    for i, op in enumerate(operations)]
            return {"hits": hits, "server_ms": 1.0, "rpc_ms": 2.0}

    client = Client()
    receipt = tmp_path / "feed.json"
    write_json(receipt, {"status": "ok", "endpoint": client.endpoint, "documents": len(docs),
                         "engine_version": "fixture",
                         "feed_sha256": manifest["feed_sha256"], "application": manifest["application"],
                         "feed_ms": 12.0})
    config = {"scoring": {"mode": "bound", "combine": "max", "a": 1, "b": 1},
              "paths": {"data": str(data)}, "schema": str(SCHEMA),
              "baselines": {"B2": {"lambda_L": 0.25}}}
    return pipeline, client, bundle, doc_path, receipt, config


def test_backend_uses_one_request_for_all_readings_and_fanout_counts_every_rewrite(engine_fixture):
    pipe, client, bundle, docs, _, _ = engine_fixture
    backend = EngineBackend(pipe, client, bundle, docs, SCHEMA, hybrid_l=0.25)
    one = backend.search("hypertension", 3)
    assert one["counters"]["http_requests"] == 1 and one["counters"]["encoder_texts"] == 5
    assert len(client.calls[-1]["input.query(w)"]) == 3
    fanout = backend.search("hypertension", 3, "fanout")
    assert fanout["counters"]["http_requests"] == 3 and fanout["counters"]["encoder_texts"] == 7
    assert fanout["counters"]["server_sum_ms"] == 3 and fanout["counters"]["server_max_ms"] == 1
    assert fanout["counters"]["rpc_sum_ms"] == 6 and fanout["counters"]["rpc_max_ms"] == 2
    assert fanout["doc_ids"] and all(b["ranking"] == "hybrid" for b in client.calls[-3:])
    pipe.ctx.lexicon.version += 1
    before = len(client.calls)
    with pytest.raises(ValueError, match="version"):
        backend.search("hypertension", 3)
    assert len(client.calls) == before


def test_validation_measurements_write_artifacts_without_pretending_fixture_or_missing_ram_is_real(engine_fixture, tmp_path):
    pipe, client, bundle, _, receipt, config = engine_fixture
    out = run_measurements(config, bundle, receipt, client, tmp_path / "runs",
                           queries=2, parity_queries=1, pipeline=pipe)
    meta = json.loads((out / "meta.json").read_text())
    check = json.loads((out / "parity.json").read_text())
    assert meta["status"] == "ok" and meta["config"]["source"] == "fixture"
    assert meta["config"]["query_ids"] == [3, 7] and meta["memory_after"]["status"] == "NOT MEASURED"
    assert check["source"] == "fixture" and check["n_queries"] == 1 and not check["meets_query_counts"]
    assert check["max_abs_difference"] < 1e-6
    stats = json.loads((out / "metrics.json").read_text())
    assert set(stats) == {"slotspace", "hybrid", "fanout", "research_binary"}
    assert stats["fanout"]["server_sum_ms"]["p50"] == 3
    assert all(v["n_queries"] == 2 for v in stats.values())
    costs = pq.read_table(out / "counters.parquet").to_pandas()
    assert len(costs) == 8 and "facts_scored" in costs and "http_requests" in costs
    assert costs.loc[costs["mode"] == "research_binary", "http_requests"].isna().all()
    broken = deepcopy(json.loads(receipt.read_text()))
    broken["feed_sha256"] = "changed"
    write_json(receipt, broken)
    with pytest.raises(ValueError, match="receipt"):
        run_measurements(config, bundle, receipt, client, tmp_path / "failed",
                         queries=2, parity_queries=1, pipeline=pipe)
    failure = next((tmp_path / "failed").glob("*/meta.json"))
    assert json.loads(failure.read_text())["status"] == "failed"


def test_smoke_refuses_existing_data_and_cleans_only_its_own_ids_on_failure():
    class Existing:
        def wait_ready(self):
            pass

        def document_count(self):
            return 10

    with pytest.raises(ValueError, match="EMPTY"):
        smoke(Existing())

    class Broken(Existing):
        removed = []

        def document_count(self):
            return 0

        def feed(self, operations):
            return {"documents": len(operations), "feed_ms": 1}

        def wait_count(self, count):
            pass

        def search(self, body):
            return {"hits": []}

        def delete_one(self, number):
            self.removed.append(number)

    client = Broken()
    with pytest.raises(ValueError, match="four"):
        smoke(client)
    assert client.removed == [1, 2, 3, 4]

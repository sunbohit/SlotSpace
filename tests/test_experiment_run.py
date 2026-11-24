import json
from pathlib import Path

import pandas as pd
import pytest

from slotspace.artifacts import sha256, write_json
from slotspace.benchmark import reciprocal_fusion, role_confusion, rewrite
from slotspace.experiment_run import execute
from slotspace.schema import Schema
from test_schema import SCHEMA


def authorized_fixture(tmp_path, monkeypatch):
    query = tmp_path / "fixture.jsonl"
    query.write_text("".join(json.dumps({"qid": i, "query": text, "answers": [17]}) + "\n"
                              for i, text in enumerate(("normal", "empty"))))
    job = {"id": "B0--prime_test", "dataset": "prime_test", "config": {},
           "experiment": {"id": "B0", "strategy": "bm25", "oracle": False}, "missing": []}
    plan = {"frozen_sha256": "fixture-only", "references": {}, "draft": False,
            "datasets": {"prime_test": {"path": str(query), "sha256": sha256(query)}},
            "jobs": [job, {**job, "id": "alias", "alias_of": job["id"]}]}
    path, approval = tmp_path / "plan.json", tmp_path / "approval.json"
    write_json(path, plan)
    write_json(approval, {"fixture": True})
    monkeypatch.setattr("slotspace.experiment_run.authorize", lambda *args: plan)
    return path, approval


def test_execution_writes_all_contract_files_handles_empty_results_and_does_not_rerun_aliases(tmp_path, monkeypatch):
    path, approval = authorized_fixture(tmp_path, monkeypatch)
    calls = []

    class Backend:
        source = "fixture"
        doc_ids = [17, 91]
        storage_per_doc = None

        def search(self, text, qid, gold):
            calls.append(qid)
            return {"doc_ids": [17] if text == "normal" else [], "scores": [1.0] if text == "normal" else [],
                    "plan": {"query_id": qid, "segments": []}, "evidence": {},
                    "counters": {"bm25_postings": 1, "dense_searches": 0, "value_postings": 0,
                                 "candidates": 1, "facts_scored": 0, "encoder_texts": 0, "latency_ms": 1.0}}

        def close(self):
            pass

    state = execute(path, approval, tmp_path / "runs", backend_factory=lambda job: Backend())
    assert calls == [0] * 4 + [1] * 4
    first = state["jobs"]["B0--prime_test"]
    assert first["status"] == "ok" and state["jobs"]["alias"]["path"] == first["path"]
    run = Path(first["path"])
    assert {p.name for p in run.iterdir()} >= {
        "meta.json", "run.json", "per_query.parquet", "counters.parquet", "plans.jsonl", "metrics.json"}
    assert json.loads((run / "metrics.json").read_text())["top100"]["hit@1"] == 50
    assert pd.read_parquet(run / "counters.parquet").latency_ms.tolist() == [1, 1]
    with pytest.raises(FileExistsError, match="reruns"):
        execute(path, approval, tmp_path / "runs", backend_factory=lambda job: Backend())


def test_backend_failure_is_recorded_as_failed_not_running(tmp_path, monkeypatch):
    path, approval = authorized_fixture(tmp_path, monkeypatch)

    def failed(job):
        raise RuntimeError("model failure")

    with pytest.raises(RuntimeError, match="model failure"):
        execute(path, approval, tmp_path / "runs", backend_factory=failed)
    state = json.loads((tmp_path / "runs" / "execution.json").read_text())
    assert state["jobs"]["B0--prime_test"]["status"] == "failed"


def test_fanout_fusion_and_role_confusion_follow_the_documented_definitions():
    ranked, scores = reciprocal_fusion([([17, 91], 0.8), ([91, 303], 0.2)])
    assert ranked[0] == 91 and scores[91] == pytest.approx(0.8 / 62 + 0.2 / 61)
    assert role_confusion([17, 91, 303], [17], [17, 91]) == 0.1
    slot = Schema.load(SCHEMA).slot("drug.indication.disease")
    assert "asthma" in rewrite("original", slot, "asthma") and "original" in rewrite("original", slot, "asthma")


def test_fanout_runs_one_hybrid_pass_per_reading_and_sums_real_counters(tmp_path, monkeypatch):
    from slotspace.benchmark import OnlineBackend
    from test_feature_cache import fixture_pipeline

    pipeline = fixture_pipeline()
    pd.DataFrame({"doc_id": [101, 202], "name": ["hypertension", "hypertension phenotype"]}).to_parquet(
        tmp_path / "docs.parquet", index=False)
    monkeypatch.setattr("slotspace.benchmark.load_pipeline", lambda *args: pipeline)
    resources = {"text_keys": str(tmp_path / "keys"), "context_cache": str(tmp_path / "ctx"),
                 "lexicon": str(tmp_path / "lex"), "model": "fixture",
                 "temperatures": str(tmp_path / "temperature"), "conformal": str(tmp_path / "conformal")}
    job = {"experiment": {"strategy": "fanout", "id": "B6", "oracle": False},
           "artifacts": resources, "config": {"paths": {"data": str(tmp_path)}, "scoring": {"lambda_L": 0.5}}}
    backend = OnlineBackend(job)
    result = backend.search("hypertension", 1)
    count = sum(len(s["readings"]) for s in result["plan"]["segments"])
    assert result["counters"]["dense_searches"] == count and pipeline.ctx.scorer.calls == 1
    assert result["counters"]["encoder_texts"] == count + result["plan"]["planner_encoder_texts"]
    fallback = backend.search("unknown", 2)
    assert fallback["counters"]["dense_searches"] == 1 and fallback["plan"]["path"] == "vector_first"


def test_binary_storage_includes_lexicon_metadata_and_schema_and_rejects_missing_inputs(tmp_path, monkeypatch):
    from slotspace.benchmark import OnlineBackend
    from test_feature_cache import fixture_pipeline

    pipeline = fixture_pipeline()
    pd.DataFrame({"doc_id": pipeline.index.doc_ids, "name": ["fixture"] * len(pipeline.index.doc_ids)}).to_parquet(
        tmp_path / "docs.parquet", index=False)
    monkeypatch.setattr("slotspace.benchmark.load_pipeline", lambda *args: pipeline)
    for folder in ("facts", "ctx", "lex"):
        (tmp_path / folder).mkdir()
    payloads = {
        tmp_path / "facts" / "native_keys.parquet": b"n" * 11,
        tmp_path / "facts" / "text_facts.parquet": b"t" * 13,
        tmp_path / "keys.parquet": b"k" * 17,
        tmp_path / "ctx" / "text_fact_ctx_qwen3e06_128.binary.npy": b"b" * 19,
        tmp_path / "ctx" / "manifest.json": b"{}",
        tmp_path / "lex" / "lexicon.parquet": b"l" * 23,
        tmp_path / "lex" / "manifest.json": b'{"file": "lexicon.parquet"}',
        tmp_path / "schema.yaml": b"version: fixture\n",
    }
    for path, content in payloads.items():
        path.write_bytes(content)
    resources = {"facts": str(tmp_path / "facts"), "text_keys": str(tmp_path / "keys.parquet"),
                 "context_cache": str(tmp_path / "ctx"), "lexicon": str(tmp_path / "lex"),
                 "model": "fixture", "temperatures": str(tmp_path / "temperature"),
                 "conformal": str(tmp_path / "conformal")}
    job = {"experiment": {"strategy": "slotspace", "id": "A11", "oracle": False},
           "artifacts": resources, "config": {"paths": {"data": str(tmp_path)}, "scoring": {"lambda_L": 0.5},
                                             "schema": str(tmp_path / "schema.yaml")}}
    backend = OnlineBackend(job)
    assert backend.storage_per_doc == pytest.approx(sum(map(len, payloads.values())) / len(pipeline.index.doc_ids))
    (tmp_path / "lex" / "lexicon.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        OnlineBackend(job)

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from slotspace.feature_cache import QueryFeatures, cache_matches, capture_query, matrix_folder, replay
from slotspace.pipeline import CorpusIndex, Pipeline
from slotspace.planner import PlannerContext, PlannerSettings
from slotspace.reference import Doc, Lexicon, ScoreConfig, unit
from slotspace.retrieve import BM25Index, DenseIndex
from slotspace.schema import Schema
from test_planner import Linker
from test_reference import CON, G, HTN, IND, OFF, SIDE, toy_docs
from test_schema import SCHEMA


def fixture_pipeline():
    docs = [Doc(str(i), d.facts) for i, d in zip([17, 91, 303, 900], toy_docs())]
    docs[0].facts[1].keys.pop(f"{CON}|{HTN}")  # A must not belong to the surviving contraindication posting.
    lex = Lexicon()
    for doc in docs:
        lex.upsert(doc)
    schema = Schema.load(SCHEMA)
    schema = SimpleNamespace(version=schema.version, slot=schema.slot,
                             candidates=lambda doc_type, kinds, s=schema: s.candidates("drug", kinds))

    class Encoder:
        def queries(self, texts, dim, prompt=None):
            return np.tile(G if prompt else np.eye(4)[0], (len(texts), 1)).astype(np.float32)

    class Scorer:
        T = 1.0
        calls = 0

        def score(self, texts, candidates):
            self.calls += 1
            p = {IND: 0.1, CON: 0.84, OFF: 0.03, SIDE: 0.02, "none.disease": 0.01}
            return [{k: p.get(k, 0) for k in cands} for cands in candidates]

    ctx = PlannerContext(schema, Linker(), Encoder(), Scorer(), lex,
                         PlannerSettings(ctx_dim=4, K_L=1, K_D=1, K_V=1), 0.5)
    index = CorpusIndex(docs, 4, lex.version)
    topics = np.array([unit([1, 0, 0, 0]), unit([0.8, 0.6, 0, 0]),
                       unit([0.6, 0.8, 0, 0]), unit([-1, 0, 0, 0])], dtype=np.float32)
    return Pipeline(ctx, index, BM25Index(["hypertension"] * 4), DenseIndex(topics),
                    lambda_d=0, lambda_l=0)


def capture(tmp_path, pipeline, text="hypertension", qid=2):
    q = {"qid": qid, "query": text, "answers": [91]}
    truth = [{"start": 0, "end": len("hypertension"), "true_readings": [f"{CON}|{HTN}"]}] if text == "hypertension" else []
    feature = capture_query(pipeline, q, truth)
    feature.save(tmp_path / f"q{qid}")
    return QueryFeatures.load(feature.folder)


def test_uncapped_feature_universe_replays_a_different_pruned_posting_topk_exactly(tmp_path):
    pipeline = fixture_pipeline()
    feature = capture(tmp_path, pipeline)
    assert 91 in feature.arrays["doc_ids"]  # not the top dense document in the unpruned union
    cache_matches(feature, pipeline.index, pipeline.scoring)
    expected = pipeline.search("hypertension", 2)
    calls = pipeline.ctx.scorer.calls
    actual = replay(feature, pipeline.ctx.schema, pipeline.ctx.cfg, 0.5, pipeline.scoring,
                    pipeline.weights, pipeline.final_k)
    assert actual["doc_ids"] == expected["doc_ids"] and actual["doc_ids"][0] == 91
    assert np.array_equal(actual["scores"], expected["scores"])
    assert all(actual["counters"][k] == expected["counters"][k] for k in
               ("candidates", "bm25_postings", "dense_searches", "value_postings", "facts_scored"))
    assert pipeline.ctx.scorer.calls == calls and actual["counters"]["latency_ms"] is None


def test_cache_supports_new_weights_modes_oracle_and_no_mentions_without_inference(tmp_path):
    pipeline = fixture_pipeline()
    feature = capture(tmp_path, pipeline)
    for mode in ("bound", "unbound", "gate_only", "no_role"):
        cfg = replace(pipeline.scoring, mode=mode, a=0.5, b=2)
        cache_matches(feature, pipeline.index, cfg)
        pipeline.scoring = cfg
        expected = pipeline.search("hypertension", 2)
        actual = replay(feature, pipeline.ctx.schema, pipeline.ctx.cfg, 0.5, cfg, pipeline.weights)
        assert actual["doc_ids"] == expected["doc_ids"]
        assert np.array_equal(actual["scores"], expected["scores"])
    oracle = replay(feature, pipeline.ctx.schema, pipeline.ctx.cfg, 0.5, pipeline.scoring,
                    pipeline.weights, oracle=True)
    assert oracle["plan"]["segments"][0]["readings"][0]["w"] == 1
    empty = capture(tmp_path, pipeline, "unknown", 4)
    cache_matches(empty, pipeline.index, pipeline.scoring)
    assert replay(empty, pipeline.ctx.schema, pipeline.ctx.cfg, 0.5, pipeline.scoring)["plan"]["path"] == "vector_first"
    disabled = replay(feature, pipeline.ctx.schema, pipeline.ctx.cfg, 0.5, pipeline.scoring, (0, 1, 0.5))
    assert disabled["plan"]["binding_disabled"] and disabled["counters"]["facts_scored"] == 0


def test_cache_rejects_stale_geometry_budget_and_corrupted_matrices(tmp_path):
    pipeline = fixture_pipeline()
    feature = capture(tmp_path, pipeline)
    cache_matches(feature, pipeline.index, pipeline.scoring)
    with pytest.raises(ValueError, match="maxima"):
        replay(feature, pipeline.ctx.schema, replace(pipeline.ctx.cfg, K_L=2), 0.5, pipeline.scoring)
    pipeline.index.fingerprint = "different"
    with pytest.raises(ValueError, match="geometry"):
        cache_matches(feature, pipeline.index, pipeline.scoring)
    (matrix_folder(feature, pipeline.scoring) / "matches.npy").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        replay(feature, pipeline.ctx.schema, pipeline.ctx.cfg, 0.5, pipeline.scoring)

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from slotspace.linker import Linker
from slotspace.pipeline import CorpusIndex, Pipeline
from slotspace.planner import PlannerContext, PlannerSettings
from slotspace.reference import Doc, Fact, Lexicon, ScoreConfig, binding_score, unit
from slotspace.retrieve import BM25Index, BaselineRetriever, DenseIndex
from slotspace.schema import Schema
from test_reference import CON, G, IND, OFF, toy_docs, toy_readings
from test_schema import SCHEMA


def setup():
    docs = [Doc(str(i), d.facts) for i, d in zip([17, 91, 303, 900], toy_docs())]
    lex = Lexicon()
    for doc in docs:
        lex.upsert(doc)
    index = CorpusIndex(docs, 4, lex.version)
    topics = np.array([unit([1, 0, 0, 0]), unit([0, 1, 0, 0]), unit([0, 0, 1, 0]), unit([0, 0, 0, 1])],
                      dtype=np.float32)

    class Encoder:
        def queries(self, texts, dim):
            return np.tile(topics[0], (len(texts), 1))

    ctx = SimpleNamespace(lexicon=lex, cfg=PlannerSettings(ctx_dim=4, K_L=4, K_D=4, K_V=1),
                          embedder=Encoder(), schema=SimpleNamespace(version="0.1.0"))
    bm25, dense = BM25Index(["hypertension treatment", "hypertension avoid", "hypertension report", "other data"]), DenseIndex(topics)
    readings = toy_readings()
    plan = {"query_id": 1, "lexicon_version": lex.version, "weight_objective": "p",
            "budgets": {"K_L": 4, "K_D": 4, "K_V": 1}, "planner_encoder_texts": 5,
            "segments": [{"g": G.tolist(), "links": {k.split("|")[1]: 1.0 for r in readings for k in r.keys},
                          "readings": [{"key": next(iter(r.keys)), "p": r.weight, "w": r.weight} for r in readings]}]}
    return docs, ctx, index, bm25, dense, plan


def test_full_candidate_binding_matches_reference_and_returns_the_winning_fact():
    docs, ctx, index, bm25, dense, plan = setup()
    pipeline = Pipeline(ctx, index, bm25, dense, lambda_l=0, lambda_d=0)
    result = pipeline.execute_plan("hypertension", plan)
    expected = sorted(docs, key=lambda d: -binding_score(d, toy_readings()))
    assert result["doc_ids"] == [int(d.doc_id) for d in expected]
    assert np.allclose(result["scores"], [binding_score(d, toy_readings()) for d in expected], atol=1e-5)
    best = result["evidence"]["91"]
    assert any("B.t1" in p["fact_ids"] for e in best for p in e["parts"])
    assert result["counters"]["dense_searches"] == 1 and result["counters"]["candidates"] == 4
    assert result["counters"]["value_postings"] == sum(len(index.postings[k]) for k in
                                                      {r["key"] for r in plan["segments"][0]["readings"]})
    for doc, explanations in result["evidence"].items():
        assert np.allclose([e["match"] for e in explanations],
                           [binding_score(next(d for d in docs if d.doc_id == doc), [r]) / r.weight for r in toy_readings()])


def test_disabled_binding_is_exactly_the_hybrid_baseline_without_query_planning():
    _, ctx, index, bm25, dense, _ = setup()
    pipeline = Pipeline(ctx, index, bm25, dense, lambda_b=0)
    result = pipeline.search("hypertension")
    positions, values, _ = BaselineRetriever(bm25, dense, 4, 4).search(
        "hypertension", "hybrid", dense.documents[0])
    assert result["doc_ids"] == index.doc_ids[positions].tolist()
    assert np.allclose(result["scores"], values) and result["counters"]["encoder_texts"] == 1
    assert result["counters"]["facts_scored"] == result["counters"]["value_postings"] == 0


def test_hard_and_can_return_no_results_without_changing_the_pre_filter_pool():
    _, ctx, index, bm25, dense, plan = setup()
    plan["segments"].append({"g": G.tolist(), "links": {"n104": 1.0},
                             "readings": [{"key": f"{CON}|n104", "p": 1.0, "w": 1.0}]})
    result = Pipeline(ctx, index, bm25, dense, hard_and=True).execute_plan("hypertension", plan)
    assert result["doc_ids"] == result["scores"] == []
    assert result["counters"]["candidates"] == 4 and result["counters"]["facts_scored"] > 0
    with pytest.raises(ValueError, match="divisible"):
        CorpusIndex([], 4, "invalid-binary", "binary")


def test_version_mismatch_is_not_silently_served_and_ia_scores_preserve_the_new_order():
    _, ctx, index, bm25, dense, plan = setup()
    pipeline = Pipeline(ctx, index, bm25, dense)
    broken = deepcopy(plan)
    broken["lexicon_version"] += 1
    with pytest.raises(ValueError, match="snapshots"):
        pipeline.execute_plan("hypertension", broken)
    plan["weight_objective"] = "ia"
    result = pipeline.execute_plan("hypertension", plan)
    assert result["scores"] == [4.0, 3.0, 2.0, 1.0]
    assert len(result["fusion_scores"]) == len(result["doc_ids"])
    ctx.lexicon.version += 1
    with pytest.raises(ValueError, match="version"):
        pipeline.search("hypertension")


def test_search_runs_the_actual_link_plan_retrieve_bind_chain_and_keeps_fallback():
    docs = [Doc("4", []), Doc("10", [Fact(f"10.n.{IND}|n4", {f"{IND}|n4": 1.0}, None, True)]),
            Doc("20", [Fact("20.t0", {f"{CON}|n4": 0.9, f"{IND}|n4": 0.1}, np.eye(4)[1])])]
    lex = Lexicon()
    for doc in docs:
        lex.upsert(doc)
    real = Schema.load(SCHEMA)
    schema = SimpleNamespace(version=real.version, slot=real.slot,
                             candidates=lambda doc_type, kinds: real.candidates("drug", kinds))
    linker = Linker(pd.DataFrame({"doc_id": [4, 10, 20], "doc_type": ["disease", "drug", "drug"],
                                   "name": ["hypertension", "drugalpha", "drugbeta"]}), common_words=0)

    class Encoder:
        def queries(self, texts, dim, prompt=None):
            return np.tile(np.eye(dim, dtype=np.float32)[1], (len(texts), 1))

    class Scorer:
        calls = 0

        def score(self, texts, candidates):
            self.calls += 1
            p = {IND: 0.1, CON: 0.8, OFF: 0.05, "none.disease": 0.05}
            return [{k: p[k] for k in cands} for cands in candidates]

    scorer = Scorer()
    ctx = PlannerContext(schema, linker, Encoder(), scorer, lex,
                         PlannerSettings(ctx_dim=4, K_L=3, K_D=3, K_V=2), 0)
    index = CorpusIndex(docs, 4, lex.version)
    pipeline = Pipeline(ctx, index, BM25Index(["hypertension", "drug treats hypertension", "avoid hypertension"]),
                        DenseIndex(np.eye(4, dtype=np.float32)[:3]), lambda_l=0, lambda_d=0)
    result = pipeline.search("avoid hypertension")
    assert result["doc_ids"][0] == 20 and result["plan"]["path"] == "both"
    assert result["counters"]["candidates"] == 3 and result["counters"]["dense_searches"] == 1
    assert result["counters"]["encoder_texts"] == 6 and scorer.calls == 1
    fallback = pipeline.search("unlinked nonsense")
    assert fallback["plan"]["path"] == "vector_first" and scorer.calls == 1
    assert fallback["counters"]["value_postings"] == 0 and fallback["counters"]["candidates"] == 3

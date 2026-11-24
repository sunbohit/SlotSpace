from types import SimpleNamespace

import numpy as np

from slotspace.planner import plan_query
from slotspace.slot_scorer import score_fact_batch
from slotspace.schema import Schema
from test_planner import context
from test_schema import SCHEMA


def test_visible_targets_affect_only_the_slot_scorer_not_segment_embeddings():
    ctx = context()
    original = ctx.scorer.score
    seen = []

    def capture(texts, candidates):
        seen.extend(texts)
        assert all("hypertension" in text and "[TARGET" not in text for text in texts)
        return original(texts, candidates)

    ctx.scorer.values_visible = True
    ctx.scorer.score = capture
    plan = plan_query("Avoid hypertension", 1, ctx)
    assert seen and plan["segments"][0]["g"] == np.eye(128, dtype=np.float32)[0].tolist()
    schema = Schema.load(SCHEMA)

    class Scorer:
        values_visible = True

        def __init__(self):
            self.schema = schema

        def score(self, texts, candidates):
            assert texts == ["avoid asthma"]
            return [{"none.disease": 1.0}]

    fact = {"row": 0, "doc_id": 1, "fact_id": "1.t0", "sentence": "asthma",
            "mentions": [{"start": 0, "end": 6, "value_type": "disease", "value_ids": ["n4"],
                          "scorer_input": "avoid [TARGET:disease]"}]}
    assert score_fact_batch([fact], {1: "drug"}, Scorer()) == []

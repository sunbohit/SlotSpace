import numpy as np
import pytest
from scipy.special import logsumexp

from slotspace.schema import Schema
from slotspace.slot_scorer import SlotScorer, ece, fit_temperature, score_fact_batch
from test_schema import SCHEMA


def test_joint_softmax_and_reranker_pair_order_use_raw_logits():
    class Model:
        def predict(self, pairs, **kwargs):
            self.pairs = pairs
            assert kwargs["activation_fn"].__class__.__name__ == "Identity"
            return np.array([3, 1, 0, 2, 0], dtype=float)

    scorer = SlotScorer.__new__(SlotScorer)
    scorer.schema, scorer.model, scorer.T, scorer.order, scorer.batch_size = Schema.load(SCHEMA), Model(), 2, "slot_first", 4
    cands = [["drug.indication.disease", "none.disease", "none.effect_phenotype"],
             ["drug.side_effect.effect_phenotype", "none.effect_phenotype"]]
    result = scorer.score(["a", "b"], cands)
    expected = np.exp(np.array([3, 1, 0]) / 2 - logsumexp(np.array([3, 1, 0]) / 2))
    assert np.allclose(list(result[0].values()), expected)
    assert all(np.isclose(sum(row.values()), 1) for row in result)
    assert scorer.model.pairs[0] == (scorer.schema.slot_text(cands[0][0]), "a")
    with pytest.raises(ValueError, match="lengths"):
        scorer.score(["a"], [])
    with pytest.raises(ValueError, match="Temperature"):
        scorer.score(["a"], [cands[0]], temperature=0)


def test_cross_type_fact_is_scored_once_and_none_never_opens_a_key():
    schema = Schema.load(SCHEMA)

    class Scorer:
        def __init__(self):
            self.schema = schema

        def score(self, contexts, candidates):
            assert contexts == ["avoid [TARGET]"]
            assert {"none.disease", "none.effect_phenotype"} <= set(candidates[0])
            return [{"drug.indication.disease": 0.4, "drug.side_effect.effect_phenotype": 0.03,
                     "none.disease": 0.27, "none.effect_phenotype": 0.3}]

    fact = {"row": 8, "fact_id": "1.t0", "doc_id": 1, "mentions": [
        {"start": 2, "end": 7, "scorer_input": "avoid [TARGET]", "value_type": t, "value_ids": [v]}
        for t, v in (("disease", "n3"), ("effect_phenotype", "n4"))]}
    assert score_fact_batch([fact], {1: "drug"}, Scorer()) == [
        {"row": 8, "key": "drug.indication.disease|n3", "pi": 0.4}]


def test_temperature_reduces_overconfident_nll_and_ece_includes_zero_confidence():
    logits = [np.array([8.0, -8.0])] * 4
    y = [0, 0, 1, 1]
    t = fit_temperature(logits, y)
    before = np.mean([logsumexp(z) - z[i] for z, i in zip(logits, y)])
    after = np.mean([logsumexp(z / t) - z[i] / t for z, i in zip(logits, y)])
    assert t > 1 and after < before
    assert ece([0], [1]) == 1
    with pytest.raises(ValueError):
        fit_temperature([], [])

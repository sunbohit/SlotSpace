import json
from types import SimpleNamespace

import numpy as np
import pytest

from slotspace.planner import PlannerContext, PlannerSettings, calibrate_conformal, plan_query, plan_readings
from slotspace.schema import Schema
from test_reference import CON, HTN, HTNP, IND, OFF, SIDE, corpus_lexicon
from test_schema import SCHEMA


class Linker:
    vtype = {HTN: "disease", HTNP: "effect_phenotype"}

    def link_query(self, query):
        if "hypertension" not in query:
            return []
        start = query.index("hypertension")
        return [{"start": start, "end": start + len("hypertension"), "surface": "hypertension",
                 "types": set(self.vtype.values()), "values": {HTN: 1.0, HTNP: 1.0}}]


class Encoder:
    def queries(self, texts, dim, prompt):
        assert prompt.endswith("Query:") and all("[TARGET]" in t for t in texts)
        return np.tile(np.eye(dim, dtype=np.float32)[0], (len(texts), 1))


class Scorer:
    def __init__(self, none=0):
        self.none = none

    def score(self, texts, candidates):
        probability = {IND: 0.135, CON: 0.72, OFF: 0.045, SIDE: 0.1}
        return [{k: self.none if k == "none.disease" else (1 - self.none) * probability.get(k, 0)
                 for k in cands} for cands in candidates]


def context(none=0, objective="p", threshold=0):
    real = Schema.load(SCHEMA)
    schema = SimpleNamespace(version=real.version, slot=real.slot,
                             candidates=lambda doc_type, kinds: real.candidates("drug", kinds))
    return PlannerContext(schema, Linker(), Encoder(), Scorer(none), corpus_lexicon(),
                          PlannerSettings(weight_objective=objective), threshold)


def test_joint_worked_posterior_pruning_none_mass_and_json_serialization():
    plan = plan_query("Avoid hypertension", 1, context())
    segment = plan["segments"][0]
    assert segment["posterior"][f"{CON}|{HTN}"] == pytest.approx(0.6254, abs=1e-4)
    assert segment["posterior"][f"{IND}|{HTN}"] == pytest.approx(0.2014, abs=1e-4)
    assert segment["posterior"][f"{SIDE}|{HTNP}"] == pytest.approx(0.1732, abs=1e-4)
    assert f"{OFF}|{HTN}" in segment["pruned_zero_support"]
    assert plan["estimated_value_postings"] == 475 and plan["path"] == "both"
    assert json.loads(json.dumps(plan))["query_id"] == 1
    assert all(list(r.keys.values()) == [1.0] for r in plan_readings(plan))
    plan = plan_query("Avoid hypertension", 1, context(none=0.2))
    assert sum(r.weight for r in plan_readings(plan)) == pytest.approx(0.8)


def test_weight_objective_changes_only_weights_and_no_mentions_falls_back():
    expected = plan_query("Avoid hypertension", 1, context())
    adjusted = plan_query("Avoid hypertension", 1, context(objective="p_over_H"))
    assert expected["segments"][0]["posterior"] == adjusted["segments"][0]["posterior"]
    assert expected["segments"][0]["readings"][0]["w"] != adjusted["segments"][0]["readings"][0]["w"]
    kept = plan_query("Avoid hypertension", 1, context(threshold=0.35))
    assert [r["key"] for r in kept["segments"][0]["readings"]] == [f"{CON}|{HTN}"]
    assert plan_query("No known value here", 2, context())["path"] == "vector_first"


def test_conformal_calibration_excludes_none_and_uses_best_true_reading():
    post = [{"a": 0.8, "b": 0.1}] * 20 + [{"a": 0.0}]
    truth = [["a", "b"]] * 20 + [[]]
    calibrated = calibrate_conformal(post, truth)
    assert calibrated["n_segments"] == 20 and calibrated["thresholds"]["0.1"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="labelled"):
        calibrate_conformal([{}], [[]])

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import yaml

from slotspace.artifacts import sha256
from slotspace.planner import condition_key
from slotspace.reference import ScoreConfig
from slotspace.tuning import FeatureSet, build_feature_set, conformal_bank, freeze, tune
from test_feature_cache import CON, HTN, fixture_pipeline


def sources(tmp_path, complete=True):
    pipeline = fixture_pipeline()
    pipeline.ctx.cfg = replace(pipeline.ctx.cfg, r_max=1)
    identity = {"schema_sha256": "fixture-schema", "artifacts": {},
                "conformal_identity": {"schema_sha256": "fixture-schema", "schema_version": "0.1.0",
                                       "lexicon_version": pipeline.index.version, "scorer": {"fixture": True},
                                       "temperatures_sha256": "fixture-temperature"}}
    for split, ids in (("calibration", [2, 4]), ("val", [11, 13])):
        queries = [{"qid": qid, "query": "hypertension", "answers": [91]} for qid in ids]
        silver = {qid: [{"start": 0, "end": 12, "links": pipeline.ctx.linker.link_query("hypertension")[0]["values"],
                          "true_readings": [f"{CON}|{HTN}"]}] for qid in ids}
        build_feature_set(pipeline, queries, silver, tmp_path / split, split, identity,
                          [ScoreConfig()], complete=complete, purpose="fixture")
    cfg = {"dataset": "prime", "seed": 13, "schema": "fixture-schema", "paths": {"data": "fixture", "runs": "fixture"},
           "planner": {"beta": 0.5, "eps": 1.0, "r_max": 1, "window_words": 8, "weight_objective": "p", "alpha": 0.1},
           "embedding": {"ctx_dim": 4, "segment_instruction": pipeline.ctx.cfg.segment_instruction},
           "retrieval": {"K_L": 1, "K_D": 1, "K_V": 1, "final_k": 100},
           "scoring": {"mode": "bound", "combine": "max", "a": 1.0, "b": 1.0, "precision": "float",
                       "lambda_B": 1.0, "lambda_D": 0.0, "lambda_L": 0.0},
           "tuning_grid": {"beta": [0.0, 0.5], "alpha": [0.05, 0.1], "a": [1.0], "b": [1.0],
                           "lambda_B": [1.0], "lambda_L": [0.0]}}
    return pipeline, FeatureSet(tmp_path / "calibration"), FeatureSet(tmp_path / "val"), cfg


def test_staged_tuning_recalibrates_each_beta_and_freezes_without_approving_h5(tmp_path):
    pipeline, cal, val, cfg = sources(tmp_path)
    selection = tune(cal, val, pipeline.ctx.schema, cfg, tmp_path / "tuned")
    assert json.loads((tmp_path / "tuned" / "meta.json").read_text())["status"] == "ok"
    assert selection["accepted"] and selection["metrics"]["top100"]["hit@1"] == 100
    trials = json.loads((tmp_path / "tuned" / "trials.json").read_text())
    assert [r["stage"] for r in trials] == ["beta-alpha"] * 4 + ["a-b", "fusion"] + ["weight-objective"] * 3
    bank = json.loads((tmp_path / "tuned" / "conformal_bank.json").read_text())
    assert condition_key(replace(pipeline.ctx.cfg, beta=0)) in bank["conditions"]
    assert condition_key(replace(pipeline.ctx.cfg, use_pacc=False)) in bank["conditions"]
    frozen = tmp_path / "frozen.yaml"
    receipt = freeze(tmp_path / "tuned" / "selection.json", frozen)
    data = json.loads(receipt.read_text())
    assert data["H5"] == "pending" and data["frozen_sha256"] == sha256(frozen)
    assert yaml.safe_load(frozen.read_text())["artifacts"]["conformal"].endswith("conformal_bank.json")
    assert json.loads((tmp_path / "tuned" / "metrics.json").read_text())["timing_kind"].startswith("feature replay")


def test_partial_validation_cannot_freeze_and_calibration_cannot_use_val(tmp_path):
    pipeline, cal, val, cfg = sources(tmp_path, complete=False)
    selection = tune(cal, val, pipeline.ctx.schema, cfg, tmp_path / "tuned")
    assert not selection["accepted"]
    with pytest.raises(ValueError, match="acceptance"):
        freeze(tmp_path / "tuned" / "selection.json", tmp_path / "bad.yaml")
    with pytest.raises(ValueError, match="validation"):
        conformal_bank(val, pipeline.ctx.schema, pipeline.ctx.cfg, [0.5], [0.1])


def test_production_freeze_cannot_promote_missing_or_changed_capture_inputs(tmp_path):
    pipeline, cal, val, cfg = sources(tmp_path)
    tune(cal, val, pipeline.ctx.schema, cfg, tmp_path / "tuned")
    path = tmp_path / "tuned" / "selection.json"
    selected = json.loads(path.read_text())
    selected["purpose"] = "production"
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="fingerprints"):
        freeze(path, tmp_path / "bad.yaml")
    model = tmp_path / "weights"
    model.write_bytes(b"old")
    selected["input_fingerprints"] = {str(model): {"sha256": sha256(model)}}
    path.write_text(json.dumps(selected))
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        freeze(path, tmp_path / "bad.yaml")

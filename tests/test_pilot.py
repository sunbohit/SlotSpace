from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from slotspace.artifacts import sha256, write_json
from slotspace.facts import NATIVE_SCHEMA, TEXT_SCHEMA
from slotspace.linker import Linker
from slotspace.pilot import features, measure
from slotspace.pilot_data import prepare, select_queries, verify_previous
from slotspace.pipeline import CorpusIndex, Pipeline
from slotspace.planner import PlannerContext, PlannerSettings
from slotspace.reference import Doc, Fact, Lexicon
from slotspace.reporting import generate_pilot_report
from slotspace.retrieve import BM25Index, DenseIndex
from slotspace.schema import Schema
from slotspace.tuning import freeze
from test_schema import SCHEMA

PROJECT = Path(__file__).resolve().parents[1]
IND, CON = "drug.indication.disease", "drug.contraindication.disease"


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_sampler_keeps_complete_answers_and_calibration_is_even_train_only():
    queries = [{"qid": n, "query": "text", "answers": [17, 91]} for n in (2, 4, 6)]
    gold = [{"qid": n, "segments": [{"true_readings": [f"{CON}|n4"]}]} for n in (2, 4, 6)]
    one, _, _ = select_queries(queries, gold, 1, "calibration")
    two, _, _ = select_queries(queries, gold, 2, "calibration")
    assert one[0] == two[0] and one[0]["answers"] == [17, 91]
    with pytest.raises(ValueError, match="even"):
        select_queries([{"qid": 3, "answers": [17]}], [{"qid": 3, "segments": gold[0]["segments"]}], 1, "calibration")
    with pytest.raises(ValueError, match="Cannot select"):
        select_queries(queries, gold, 1, "val", max_answers=1)


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    facts, silver = folder / "facts_prime-0.1.0", folder / "silver_prime-0.1.0"
    facts.mkdir()
    silver.mkdir()
    ids = [4, 17, 91, 101, 102, 103]
    pq.write_table(pa.table({"doc_id": ids, "doc_type": ["disease"] + ["drug"] * 5,
                            "name": ["hypertension", "alpha", "beta", "delta", "epsilon", "zeta"],
                            "text": ["text"] * 6, "details_json": ["{}"] * 6}), folder / "docs.parquet")
    pq.write_table(pa.Table.from_pylist([{"doc_id": 17, "key": f"{IND}|n4"},
                                       {"doc_id": 91, "key": f"{CON}|n4"}], schema=NATIVE_SCHEMA),
                   facts / "native_keys.parquet")
    row = {"row": 0, "fact_id": "17.t0", "doc_id": 17, "field": "details", "sentence": "hypertension",
           "ctx_text": "[disease]", "mentions": [{"value_ids": ["n4"], "value_type": "disease", "start": 0,
                                                "end": 12, "scorer_input": "[TARGET:disease]"}]}
    pq.write_table(pa.Table.from_pylist([row, {**row, "row": 1, "fact_id": "91.t0", "doc_id": 91}],
                                      schema=TEXT_SCHEMA), facts / "text_facts.parquet")
    for split, qids, answer, slot in (("train", [2, 4], 17, IND), ("val", [11, 13], 91, CON)):
        _jsonl(folder / f"queries_{split}.jsonl", [{"qid": n, "query": "hypertension", "answers": [answer]} for n in qids])
        _jsonl(silver / f"silver_{'calibration' if split == 'train' else 'val'}.jsonl", [
            {"qid": n, "segments": [{"start": 0, "end": 12, "links": {"n4": 1.0}, "true_readings": [f"{slot}|n4"]}]}
            for n in qids])
    for name in ("test", "human"):
        (folder / f"queries_{name}.jsonl").write_text("MUST NOT BE PARSED")
    config = yaml.safe_load((PROJECT / "configs" / "prime_v0.yaml").read_text(encoding="utf-8"))
    write_json(facts / "manifest.json", {"settings": config, "limit_docs": None,
               "docs_sha256": sha256(folder / "docs.parquet"), "schema_sha256": sha256(SCHEMA)})
    write_json(silver / "manifest.json", {"complete": True, "docs_sha256": sha256(folder / "docs.parquet"),
               "schema_sha256": sha256(SCHEMA), "native_sha256": sha256(facts / "native_keys.parquet"),
               "query_hashes": {s: sha256(folder / f"queries_{s}.jsonl") for s in ("train", "val")}})
    schedule = {"seed": 13, "max_query_mentions": 2, "max_answers": 4, "max_fact_mentions": 2,
                "stages": {"smoke": {"documents": 4, "calibration": 1, "validation": 1,
                                    "facts_per_doc": 1, "max_seq_length": 16, "repeats": 1},
                           "small": {"documents": 6, "calibration": 2, "validation": 2,
                                    "facts_per_doc": 1, "max_seq_length": 16, "repeats": 1}}}
    return folder, schedule


def test_stage_expansion_requires_verified_parent_and_keeps_cohorts_and_candidates(source, tmp_path):
    folder, schedule = source
    smoke = prepare(folder, SCHEMA, schedule, "smoke", tmp_path / "smoke")
    with pytest.raises(ValueError, match="preceding"):
        prepare(folder, SCHEMA, schedule, "small", tmp_path / "small")
    files = {p.name: sha256(p) for p in smoke.iterdir()}
    write_json(smoke / "completed.json", {"purpose": "pilot", "status": "ok", "engineering_checks": {"parity": True},
                                         "outputs": files})
    small = prepare(folder, SCHEMA, schedule, "small", tmp_path / "small", smoke)
    a = json.loads((smoke / "manifest.json").read_text())
    b = json.loads((small / "manifest.json").read_text())
    for split in a["selected_qids"]:
        assert set(a["selected_qids"][split]) <= set(b["selected_qids"][split])
    assert set(pq.read_table(smoke / "docs.parquet")["doc_id"].to_pylist()) <= set(
        pq.read_table(small / "docs.parquet")["doc_id"].to_pylist())
    assert a["production"] is False and a["dictionary"].startswith("full source KB")
    (smoke / "queries_val.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_previous(smoke)


def test_real_pipeline_replay_and_pilot_report_cannot_create_production_acceptance(tmp_path, monkeypatch):
    schema = Schema.load(SCHEMA)
    documents = [Doc("4", []), Doc("17", [Fact("17.n", {f"{IND}|n4": 1.0}, None, True)]),
                 Doc("91", [Fact("91.n", {f"{CON}|n4": 1.0}, None, True),
                            Fact("91.t", {f"{CON}|n4": 0.8}, np.eye(128)[1])])]
    lex = Lexicon()
    for doc in documents:
        lex.upsert(doc)
    linker = Linker(pd.DataFrame({"doc_id": [4, 17, 91], "name": ["hypertension", "alpha", "beta"],
                                  "doc_type": ["disease", "drug", "drug"]}), common_words=0)

    class Encoder:
        def queries(self, texts, dim, prompt=None):
            return np.tile(np.eye(dim, dtype=np.float32)[1 if dim == 128 else 2], (len(texts), 1))

    class Scorer:
        T = 1.0

        def score(self, contexts, candidates):
            return [{k: 0.8 if k == CON else 0.1 if k == IND else 0.1 / (len(c) - 2) for k in c} for c in candidates]

    context = PlannerContext(schema, linker, Encoder(), Scorer(), lex,
                             PlannerSettings(K_L=3, K_D=3, K_V=3, use_pacc=False), 0)
    pipe = Pipeline(context, CorpusIndex(documents, 128, lex.version),
                    BM25Index(["hypertension", "treat disease", "avoid hypertension"]), DenseIndex(np.eye(256)[:3]))
    cfg = yaml.safe_load((PROJECT / "configs" / "prime_v0.yaml").read_text(encoding="utf-8"))
    cfg["dataset"] = "prime_pilot"
    cfg["planner"]["use_pacc"] = False
    cfg["retrieval"].update(K_L=3, K_D=3, K_V=3)
    cfg["tuning_grid"] = {"beta": [0.5], "alpha": [0.1], "a": [1], "b": [1], "lambda_B": [1], "lambda_L": [0.5]}
    identity = {"artifacts": {}, "conformal_identity": {"schema_sha256": sha256(SCHEMA)}, "purpose": "pilot"}
    for split, qid in (("calibration", 2), ("val", 11)):
        _jsonl(tmp_path / f"queries_{split}.jsonl", [{"qid": qid, "query": "avoid hypertension", "answers": [91]}])
        _jsonl(tmp_path / f"silver_{split}.jsonl", [{"qid": qid, "segments": [
            {"start": 6, "end": 18, "links": {"n4": 1.0}, "true_readings": [f"{CON}|n4"]}]}])
    caches, selection = features(tmp_path, pipe, cfg, identity)
    assert not selection["accepted"] and selection["purpose"] == "pilot"
    changed = deepcopy(selection)
    changed["accepted"] = True
    changed["checks"] = {key: True for key in changed["checks"]}
    write_json(tmp_path / "forged-selection.json", changed)
    with pytest.raises(ValueError, match="Pilot"):
        freeze(tmp_path / "forged-selection.json", tmp_path / "frozen.yaml")
    bank = json.loads((tmp_path / "cache" / "tuning" / "conformal_bank.json").read_text())
    data = measure(pipe, [{"qid": 11, "query": "avoid hypertension", "answers": [91]}], caches,
                   selection["config"], bank, ["bm25", "dense", "hybrid", "bound", "unbound", "fanout"], 1)
    results, metrics, costs, rows, plans, evidence, parity = data
    assert parity[0]["ranking_equal"] and parity[0]["max_abs"] < 1e-5
    assert next(c for c in costs if c["method"] == "fanout")["encoder_texts"] > 1
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "meta.json", {"status": "ok"})
    write_json(run / "pilot.json", {"purpose": "pilot", "formal_acceptance": False, "stage": "fixture",
               "budget": {"documents": 3, "validation": 1, "repeats": 1}, "selection": selection})
    write_json(run / "metrics.json", metrics)
    pq.write_table(pa.Table.from_pylist(rows), run / "per_query.parquet")
    cols = set().union(*(c.keys() for c in costs))
    pq.write_table(pa.table({k: [c.get(k) for c in costs] for k in cols}), run / "counters.parquet")
    monkeypatch.setattr("slotspace.pilot.SlotScorer", lambda *a, **k: pytest.fail("Report loaded a model"))
    report = generate_pilot_report(run, run / "report.md")
    assert report["formal_acceptance"] is False and "NOT a formal M8" in (run / "report.md").read_text()

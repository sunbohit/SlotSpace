import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from slotspace.evaluate import Evaluation, per_query_metrics
from slotspace.experiments import run_baseline


def test_top100_metrics_match_official_with_noncontiguous_ids():
    ev = Evaluation([4, 17, 91])
    ev.add(12, [17, 91, 4], [91, 4])
    summary = ev.summary()
    assert summary["top100"]["hit@1"] == 0
    assert summary["top100"]["mrr"] == 50
    assert max(summary["agreement_points"].values()) < 1e-5
    with pytest.raises(ValueError, match="distinct"):
        per_query_metrics([17, 17], {17})


def test_empty_hard_and_results_have_zero_quality_without_a_fake_placeholder():
    ev = Evaluation([4, 17])
    ev.add(1, [], [4])
    summary = ev.summary()
    assert summary["empty_rankings"] == 1 and not any(summary["top100"].values())
    assert not any(summary["official"].values()) and "extension" in summary["empty_policy"]


def test_baseline_run_maps_positions_and_writes_all_contract_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    pq.write_table(pa.table({"doc_id": [4, 17, 91], "text": ["red shoes", "blue hat", "green boot"]}),
                   data / "docs.parquet")
    (data / "queries_val.jsonl").write_text(json.dumps({"qid": 8, "query": "blue hat", "answers": [17]}) + "\n")
    config = {"dataset": "prime", "seed": 13, "paths": {"data": str(data), "runs": str(tmp_path / "runs")},
              "embedding": {}, "retrieval": {"K_L": 200, "K_D": 200, "final_k": 100}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_baseline(path, "bm25")
    assert {p.name for p in out.iterdir()} == {
        "meta.json", "run.json", "per_query.parquet", "counters.parquet", "plans.jsonl", "metrics.json",
    }
    results = json.loads((out / "run.json").read_text())
    assert next(iter(results["8"])) == "17" and set(results["8"]) == {"4", "17", "91"}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["status"] == "ok" and meta["packages"] and meta["git_commit"]
    costs = pq.read_table(out / "counters.parquet").to_pylist()[0]
    assert costs["dense_searches"] == 0 and costs["bm25_postings"] == 2 and costs["candidates"] == 3
    assert costs["latency_ms"] >= 0
    with pytest.raises(ValueError, match="H5"):
        run_baseline(path, "bm25", split="test")

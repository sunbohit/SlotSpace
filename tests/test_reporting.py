import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from slotspace.artifacts import write_json
from slotspace.experiment_run import execute
from slotspace.reporting import comparisons, generate_report, pass_lines, reliability_svg, scatter_svg
from test_experiment_run import authorized_fixture


def test_unrun_protocol_builds_four_honest_tables_and_placeholder_figures(tmp_path):
    plan = tmp_path / "draft.json"
    write_json(plan, {"draft": True, "frozen_sha256": "not-frozen", "jobs": []})
    out = tmp_path / "report.md"
    summary = generate_report(plan, None, out)
    text = out.read_text()
    assert summary["completed_rows"] == 0 and all(s == "NOT RUN" for s in summary["pass_lines"].values())
    assert all(f"Table {n}" in text for n in range(1, 5))
    assert "DRAFT / NOT RUN" in text and "95%" in text
    for path in (tmp_path / "report_figures").glob("*.svg"):
        assert ET.parse(path).getroot().tag.endswith("svg") and "NOT RUN" in path.read_text()


def test_report_reconstructs_saved_outputs_and_does_not_promote_fixtures(tmp_path, monkeypatch):
    plan, approval = authorized_fixture(tmp_path, monkeypatch)

    class Backend:
        source = "fixture"
        doc_ids = [17, 91]
        storage_per_doc = None

        def search(self, text, qid, gold):
            return {"doc_ids": [17], "scores": [1.0], "plan": {"query_id": qid, "segments": []},
                    "counters": {"bm25_postings": 2, "value_postings": 0, "dense_searches": 0,
                                 "candidates": 1, "facts_scored": 0, "encoder_texts": 0, "latency_ms": 1.0}}

        def close(self):
            pass

    execute(plan, approval, tmp_path / "runs", backend_factory=lambda job: Backend())
    Path(json.loads(plan.read_text())["datasets"]["prime_test"]["path"]).unlink()
    hidden = generate_report(plan, tmp_path / "runs", tmp_path / "hidden.md")
    assert hidden["completed_rows"] == 0
    shown = generate_report(plan, tmp_path / "runs", tmp_path / "fixture.md", include_fixtures=True)
    assert shown["completed_rows"] == 1
    assert "SYNTHETIC FIXTURE REPORT" in (tmp_path / "fixture.md").read_text()


def test_paired_statistics_align_ids_and_never_treat_identical_aliases_as_an_effect():
    a = pd.DataFrame({"qid": [1, 2, 3, 4], "hit@1": [1.0, 0, 1, 0],
                      "mrr": [1.0, 0, 1, 0], "recall@100": [1.0, 0, 1, 0]}).set_index("qid")
    rows = {("A0", "role_swap_test"): {"quality": a, "metrics": {}},
            ("A1", "role_swap_test"): {"quality": a.iloc[::-1], "metrics": {}}}
    result = comparisons(rows, "role_swap_test", ["A0", "A1"])
    assert result["A1"]["metrics"]["hit@1"]["ci95_points"] == [0, 0]
    assert result["A1"]["p_holm"] == 1
    assert pass_lines(rows, result, {})["A1"] == "NOT DETECTED"
    rows[("A1", "role_swap_test")]["quality"] = a.iloc[:3]
    with pytest.raises(ValueError, match="exactly"):
        comparisons(rows, "role_swap_test", ["A0", "A1"])


def test_completed_data_figures_are_valid_svg_and_keep_real_numeric_scales():
    svg = scatter_svg([{"id": "A0", "work": 120, "recall": 75}], "work", "A0 & work")
    root = ET.fromstring(svg)
    assert len(root.findall("{http://www.w3.org/2000/svg}circle")) == 1
    assert "A0 &amp; work" in svg and "Posting-list entries" in svg
    points = [{"style": style, "confidence_before": 0.9, "confidence_after": 0.6, "correct": False}
              for style in ("query", "doc")]
    root = ET.fromstring(reliability_svg(points))
    assert len(root.findall("{http://www.w3.org/2000/svg}circle")) == 4

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from slotspace.artifacts import sha256, write_json
from slotspace.reproduce import ReviewRequired, Step, build_steps, check_review, execute_steps

PROJECT = Path(__file__).resolve().parents[1]


def test_full_recipe_uses_real_cli_options_and_separates_generation_and_h5(tmp_path, monkeypatch):
    generated, steps = build_steps(PROJECT, tmp_path / "Resource", tmp_path / "work")
    assert {s.stage for s in steps} == {f"M{i}" for i in range(9)}
    assert [s.gate for s in steps if s.gate] == ["H1", "H2", "H3", "H4", "H5"]
    assert {s.id for s in steps if s.generation} == {"paraphrase-ft", "paraphrase-a12"}
    assert next(i for i, s in enumerate(steps) if s.gate == "H5") < next(i for i, s in enumerate(steps) if s.id == "execute")
    assert all("queries_test.jsonl" not in str(s.args) for s in steps if s.stage not in {"M0", "M8"})

    class Parsed(Exception):
        pass

    original = argparse.ArgumentParser.parse_args

    def parse(self, args=None, namespace=None):
        original(self, args, namespace)
        raise Parsed()

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    for step in steps:
        if step.gate:
            continue
        argv = ["1.0" if isinstance(arg, dict) else "approval.json" if arg == "@H5" else str(arg) for arg in step.args]
        monkeypatch.setattr(sys, "argv", ["reproduce-test", *argv])
        with pytest.raises(Parsed):
            importlib.import_module(f"slotspace.{step.module}").main()
    assert len(generated) == 2
    main = next(s for s in steps if s.id == "train-ft")
    assert "masked_ft" in str(main.args) and "--a12-docs" not in main.args


def test_receipts_never_adopt_existing_outputs_and_resume_rechecks_bytes(tmp_path):
    output, source = tmp_path / "out.json", tmp_path / "source"
    source.write_text("public input")
    step = Step("one", "M0", "fixture", outputs=(str(output),), external=(str(source),))
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        output.write_text("produced")

    result = execute_steps([step], tmp_path, "same-source", runner=runner)
    assert result["status"] == "ok" and result["source"] == "fixture"
    execute_steps([step], tmp_path, "same-source", resume=True, runner=runner)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="settings changed"):
        execute_steps([step], tmp_path, "changed", resume=True, runner=runner)
    output.write_text("changed")
    with pytest.raises(ValueError, match="artifact/review changed"):
        execute_steps([step], tmp_path, "same-source", resume=True, runner=runner)


def test_failed_generation_and_missing_h5_cannot_turn_into_success(tmp_path):
    step = Step("gen", "M5", "gen.paraphrase_vllm", generation=True)
    with pytest.raises(ReviewRequired, match="separate Linux"):
        execute_steps([step], tmp_path, "s")
    assert json.loads((tmp_path / "reproduction.json").read_text())["status"] == "blocked"
    (tmp_path / "reproduction.json").unlink()
    with pytest.raises(ReviewRequired, match="H5"):
        execute_steps([Step("H5", "M8", gate="H5")], tmp_path, "s")
    assert json.loads((tmp_path / "reproduction.json").read_text())["completed"] == {}
    (tmp_path / "reproduction.json").unlink()

    def failure(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    with pytest.raises(subprocess.CalledProcessError):
        execute_steps([Step("fails", "M0", "fixture")], tmp_path, "s", runner=failure)
    assert json.loads((tmp_path / "reproduction.json").read_text())["status"] == "failed"


def test_generation_resume_pins_the_separate_environment_versions(tmp_path, monkeypatch):
    versions = {"python": "fixture", "packages": [["vllm", "0.34.0"]]}
    monkeypatch.setattr("slotspace.reproduce.generator_identity", lambda exe: versions)
    output = tmp_path / "paraphrases.jsonl"
    step = Step("gen", "M5", "fixture", outputs=(str(output),), generation=True)

    def runner(*args, **kwargs):
        output.write_text("fixture")

    execute_steps([step], tmp_path, "s", gen_python=tmp_path / "separate-python", runner=runner)
    versions["packages"] = [["vllm", "changed"]]
    with pytest.raises(ValueError, match="changed after gen"):
        execute_steps([step], tmp_path, "s", gen_python=tmp_path / "separate-python", resume=True, runner=runner)


def test_human_labels_and_exact_evidence_approval_are_both_required(tmp_path):
    schema, approvals = tmp_path / "schema.yaml", tmp_path / "reviews.json"
    schema.write_text("schema fixture")
    folder = tmp_path / "runs" / "silver-review"
    folder.mkdir(parents=True)
    write_json(tmp_path / "runs" / "m4_silver_stats.json", {"review_run": str(folder)})
    samples = folder / "silver_review.jsonl"

    def labels(correct):
        samples.write_text("".join(json.dumps({"correct": c}) + "\n" for c in correct))

    labels([None] * 100)
    with pytest.raises(ReviewRequired, match="boolean"):
        check_review("H2", tmp_path, schema, approvals, None)
    labels([True] * 79 + [False] * 21)
    with pytest.raises(ReviewRequired, match="below"):
        check_review("H2", tmp_path, schema, approvals, None)
    labels([True] * 80 + [False] * 20)
    evidence = {str(schema): sha256(schema), str(samples): sha256(samples)}
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    with pytest.raises(ReviewRequired, match=digest):
        check_review("H2", tmp_path, schema, approvals, None)
    write_json(approvals, {"H2": {"approved": True, "evidence_sha256": digest}})
    assert check_review("H2", tmp_path, schema, approvals, None) == {"evidence_sha256": digest}
    schema.write_text("changed schema")
    with pytest.raises(ReviewRequired):
        check_review("H2", tmp_path, schema, approvals, None)

from copy import deepcopy
import json

import pytest
import yaml

from slotspace.artifacts import sha256, write_json
from slotspace.feature_cache import digest
from slotspace.operator import approve_h5
from slotspace.protocol import authorize, build_plan, catalog


def frozen_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr("slotspace.protocol._git", lambda *args: "fixture-commit" if args[0] == "rev-parse" else "")
    data = tmp_path / "data"
    data.mkdir()
    for name, count in (("test", 2801), ("human", 98), ("val", 2241)):
        with (data / f"queries_{name}.jsonl").open("w") as out:
            for i in range(count):
                out.write(json.dumps({"qid": i, "query": "fixture", "answers": [17]}) + "\n")
    role = data / "role_swap_prime-0.1.0"
    role.mkdir()
    (role / "role_swap_test.jsonl").write_text(json.dumps({"qid": "rs-fixture", "query": "fixture", "answers": [17]}) + "\n")
    schema = tmp_path / "schema.yaml"
    schema.write_text("fixture: true\n")
    resources = {}
    for name in ("facts", "context_cache", "topic_cache", "lexicon"):
        folder = tmp_path / name
        folder.mkdir()
        write_json(folder / "manifest.json", {})
        resources[name] = str(folder)
    model = tmp_path / "trained" / "final"
    model.mkdir(parents=True)
    write_json(model / "config.json", {})
    (model / "model.safetensors").write_bytes(b"fixture-only")
    write_json(model.parent / "manifest.json", {"variant": "main"})
    resources.update(model=str(model), embedding_model=str(model), kind="ft")
    for name in ("text_keys", "temperatures"):
        path = tmp_path / (name + ".json")
        write_json(path, {})
        resources[name] = str(path)
    report = tmp_path / "tuned"
    report.mkdir()
    bank = report / "conformal_bank.json"
    write_json(bank, {})
    resources["conformal"] = str(bank)
    config = {"dataset": "prime", "seed": 13, "schema": str(schema), "paths": {"data": str(data), "runs": str(tmp_path / "runs")},
              "artifacts": resources, "planner": {"alpha": 0.1, "weight_objective": "p"},
              "scoring": {"mode": "bound", "precision": "float"}, "embedding": {"topic_dim": 256}, "facts": {}}
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(yaml.safe_dump(config))
    write_json(report / "selection.json", {"config": config})
    write_json(frozen.with_suffix(".yaml.freeze.json"), {
        "accepted": True, "purpose": "production", "frozen_sha256": sha256(frozen),
        "bank_sha256": sha256(bank), "selection_sha256": sha256(report / "selection.json")})
    return frozen


def test_catalog_is_complete_and_aliases_do_not_repeat_identical_final_conditions(tmp_path, monkeypatch):
    frozen = frozen_fixture(tmp_path, monkeypatch)
    ids = {e.id.split("_")[0] for e in catalog()}
    assert {f"A{i}" for i in range(15)} <= ids and {f"B{i}" for i in range(8)} <= ids
    plan = build_plan(frozen, tmp_path / "plan.json", only=["B5", "A3"])
    assert len(plan["jobs"]) == 6
    assert all(j["alias_of"] == j["id"].replace("A3", "B5") for j in plan["jobs"] if j["experiment"]["id"] == "A3")
    assert not any(j["dataset"] != "val" and j["experiment"]["oracle"] for j in plan["jobs"])


def test_h5_requires_exact_plan_and_configuration_and_rejects_artifact_changes(tmp_path, monkeypatch):
    frozen = frozen_fixture(tmp_path, monkeypatch)
    path = tmp_path / "plan.json"
    plan = build_plan(frozen, path, only=["B0"])
    assert not any(j["missing"] for j in plan["jobs"])
    approval = tmp_path / "approval.json"
    write_json(approval, {"approved": False, "frozen_sha256": sha256(frozen), "plan_sha256": sha256(path)})
    with pytest.raises(ValueError, match="H5"):
        authorize(path, approval)
    write_json(approval, {"approved": True, "frozen_sha256": sha256(frozen), "plan_sha256": sha256(path)})
    assert authorize(path, approval)["frozen_sha256"] == sha256(frozen)
    (tmp_path / "temperatures.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="artifact changed"):
        authorize(path, approval)


def test_draft_is_descriptive_only_even_if_somebody_writes_an_approval(tmp_path, monkeypatch):
    frozen = frozen_fixture(tmp_path, monkeypatch)
    plan = tmp_path / "draft.json"
    build_plan(frozen, plan, draft=True, only=["B0"])
    with pytest.raises(ValueError, match="Draft"):
        authorize(plan, tmp_path / "does-not-exist.json")


def test_operator_h5_writes_only_explicit_exact_authorization_and_rechecks_artifacts(tmp_path, monkeypatch):
    frozen = frozen_fixture(tmp_path, monkeypatch)
    plan, approval = tmp_path / "plan.json", tmp_path / "h5.json"
    build_plan(frozen, plan, only=["B0"])
    with pytest.raises(ValueError, match="--yes"):
        approve_h5(plan, approval)
    assert not approval.exists()
    entry = approve_h5(plan, approval, confirmed=True)
    assert entry["plan_sha256"] == sha256(plan) and authorize(plan, approval)["frozen_sha256"] == sha256(frozen)
    with pytest.raises(FileExistsError):
        approve_h5(plan, approval, confirmed=True)
    (tmp_path / "temperatures.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="artifact changed"):
        approve_h5(plan, tmp_path / "new-h5.json", confirmed=True)
    assert not (tmp_path / "new-h5.json").exists()


@pytest.mark.parametrize("invalid", ["draft", "pilot", "missing", "changed-implementation"])
def test_operator_h5_never_converts_ineligible_plans_into_approval(tmp_path, monkeypatch, invalid):
    frozen = frozen_fixture(tmp_path, monkeypatch)
    plan, approval = tmp_path / "plan.json", tmp_path / "h5.json"
    result = build_plan(frozen, plan, draft=invalid == "draft", only=["B0"])
    if invalid == "pilot":
        receipt = frozen.with_suffix(".yaml.freeze.json")
        data = json.loads(receipt.read_text())
        data["purpose"] = "pilot"
        write_json(receipt, data)
    elif invalid == "missing":
        result["jobs"][0]["missing"] = ["trained-model:main"]
        write_json(plan, result)
    elif invalid == "changed-implementation":
        monkeypatch.setattr("slotspace.protocol._git",
                            lambda *a: "other-commit" if a[0] == "rev-parse" else "")
    with pytest.raises(ValueError):
        approve_h5(plan, approval, confirmed=True)
    assert not approval.exists()

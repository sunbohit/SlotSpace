from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.check_dist import check
from tools.license_inventory import model_inventory


def _wheel(path, extra=(), missing_notice=False, python=">=3.11,<3.12", omit=()):
    prefix = "slotspace-0.1.0.dist-info/"
    names = ["slotspace/reproduce.py", "slotspace/resources.py", "slotspace/operator.py",
             "slotspace/model_card.py", "slotspace/vespa_app/app/services.xml",
             "slotspace/vespa_app/app/hosts.xml", "slotspace/vespa_app/app/schemas/doc.sd",
             prefix + "licenses/LICENSE", prefix + "licenses/THIRD_PARTY_NOTICES.md"]
    names += ["slotspace-0.1.0.data/data/share/slotspace/" + asset for asset in (
        "schemas/prime.yaml", "environments/research.yaml", "environments/research.constraints.txt",
        "environments/generation.yaml", "environments/generation-linux.lock.txt", "reproducibility/huggingface.csv")]
    if not missing_notice:
        names.append(prefix + "licenses/NOTICE")
    with ZipFile(path, "w") as archive:
        for name in [*names, *extra]:
            if name not in omit:
                archive.writestr(name, "fixture")
        archive.writestr(prefix + "METADATA",
                         f"Name: slotspace\nVersion: 0.1.0\nRequires-Python: {python}\nLicense-Expression: Apache-2.0\n")


def test_distribution_validation_rejects_accidental_data_weights_and_missing_attribution(tmp_path):
    safe = tmp_path / "safe.whl"
    _wheel(safe)
    assert check(safe)["version"] == "0.1.0"
    for name in ("data/prime/stark_qa.csv", "slotspace/leaked-model.safetensors", "../outside", "runs/metrics.json",
                 "C:/outside", "slotspace-0.1.0.data/data/data/prime/records.csv", "slotspace/huggingface.csv"):
        bad = tmp_path / "bad.whl"
        _wheel(bad, extra=[name])
        with pytest.raises(ValueError, match="Forbidden"):
            check(bad)
    bad = tmp_path / "no-notice.whl"
    _wheel(bad, missing_notice=True)
    with pytest.raises(ValueError, match="license/attribution"):
        check(bad)
    _wheel(bad, python=">=3.12")
    with pytest.raises(ValueError, match="Python constraints"):
        check(bad)


@pytest.mark.parametrize("asset", [
    "slotspace/operator.py",
    "slotspace-0.1.0.data/data/share/slotspace/reproducibility/huggingface.csv",
])
def test_release_requires_agentless_tools_and_portable_inputs(tmp_path, asset):
    path = tmp_path / "missing.whl"
    _wheel(path, omit=(asset,))
    with pytest.raises(ValueError, match="missing"):
        check(path)


def test_model_license_inventory_accepts_the_clone_manifest_and_rejects_changed_cards(tmp_path):
    root = tmp_path / "Resource"
    card = root / "models" / "Owner" / "tiny" / "README.md"
    card.parent.mkdir(parents=True)
    card.write_text("---\nlicense: apache-2.0\n---\nSynthetic model card\n")
    manifest = tmp_path / "huggingface.csv"
    with manifest.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=("path", "sha256", "revision", "source"))
        writer.writeheader()
        writer.writerow({"path": "models\\Owner\\tiny\\README.md",
                         "sha256": hashlib.sha256(card.read_bytes()).hexdigest(), "revision": "a" * 40,
                         "source": "https://huggingface.co/Owner/tiny"})
    assert model_inventory(root, manifest)[0]["license"] == "apache-2.0"
    card.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        model_inventory(root, manifest)
import csv
import hashlib

from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.check_dist import check


def _wheel(path, extra=(), missing_notice=False, python=">=3.11,<3.12"):
    prefix = "slotspace-0.1.0.dist-info/"
    names = ["slotspace/reproduce.py", "slotspace/vespa_app/app/services.xml",
             "slotspace/vespa_app/app/hosts.xml", "slotspace/vespa_app/app/schemas/doc.sd",
             "slotspace-0.1.0.data/data/share/slotspace/schemas/prime.yaml",
             prefix + "licenses/LICENSE", prefix + "licenses/THIRD_PARTY_NOTICES.md"]
    if not missing_notice:
        names.append(prefix + "licenses/NOTICE")
    with ZipFile(path, "w") as archive:
        for name in [*names, *extra]:
            archive.writestr(name, "fixture")
        archive.writestr(prefix + "METADATA",
                         f"Name: slotspace\nVersion: 0.1.0\nRequires-Python: {python}\nLicense-Expression: Apache-2.0\n")


def test_distribution_validation_rejects_accidental_data_weights_and_missing_attribution(tmp_path):
    safe = tmp_path / "safe.whl"
    _wheel(safe)
    assert check(safe)["version"] == "0.1.0"
    for name in ("data/prime/stark_qa.csv", "slotspace/leaked-model.safetensors", "../outside", "runs/metrics.json",
                 "C:/outside", "slotspace-0.1.0.data/data/data/prime/records.csv"):
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

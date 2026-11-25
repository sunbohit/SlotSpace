import csv
import hashlib

import pytest

from slotspace.resources import model_resources


def _row(payload, *, v1=False):
    return {"path": ("v1\\" if v1 else "") + "models\\Owner\\tiny\\1_Pooling\\config.json",
            "source": "https://huggingface.co/Owner/tiny/resolve/" + "a" * 40 + "/1_Pooling/config.json",
            "revision": "a" * 40, "bytes": str(len(payload)), "sha256": hashlib.sha256(payload).hexdigest()}


def _manifest(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=("path", "source", "revision", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_fetch_checks_pinned_bytes_and_reuses_existing_inputs_without_overwriting(tmp_path):
    payload = b'{"synthetic":true}\n'
    manifest = _manifest(tmp_path / "manifest.csv", [_row(payload), _row(payload, v1=True)])
    cached = tmp_path / "cached"
    cached.write_bytes(payload)
    root, calls = tmp_path / "Resource", []

    def download(**kwargs):
        calls.append(kwargs)
        return str(cached)

    with pytest.raises(FileNotFoundError, match="Missing model"):
        model_resources(manifest, root)
    assert model_resources(manifest, root, fetch=True, downloader=download) == 1
    dest = root / "models" / "Owner" / "tiny" / "1_Pooling" / "config.json"
    assert dest.read_bytes() == payload
    assert calls[0]["revision"] == "a" * 40 and calls[0]["token"] is False
    assert calls[0]["filename"] == "1_Pooling/config.json"
    model_resources(manifest, root, fetch=True, downloader=download)
    assert len(calls) == 1
    assert model_resources(manifest, root, fetch=True, include_v1=True, downloader=download) == 2
    dest.write_bytes(b"existing incompatible input")
    with pytest.raises(ValueError, match="mismatch"):
        model_resources(manifest, root, fetch=True, downloader=download)
    assert dest.read_bytes() == b"existing incompatible input" and len(calls) == 2
    assert not list(root.rglob(".resource-*"))


def test_failed_download_verification_leaves_no_destination_or_stale_temporary_file(tmp_path):
    payload = b"valid public fixture"
    manifest = _manifest(tmp_path / "manifest.csv", [_row(payload)])
    cached, root = tmp_path / "cached", tmp_path / "Resource"
    cached.write_bytes(b"invalid fixture")
    with pytest.raises(ValueError, match="mismatch"):
        model_resources(manifest, root, fetch=True, downloader=lambda **_: str(cached))
    assert not list(root.rglob("config.json")) and not list(root.rglob(".resource-*"))
    cached.write_bytes(payload)
    model_resources(manifest, root, fetch=True, downloader=lambda **_: str(cached))
    assert next(root.rglob("config.json")).read_bytes() == payload


@pytest.mark.parametrize("change", [
    {"path": "models\\Owner\\tiny\\..\\outside"},
    {"path": "models\\C:\\outside\\config.json"},
    {"path": "models\\Owner\\tiny\\different.json"},
    {"source": "https://example.invalid/Owner/tiny/resolve/" + "a" * 40 + "/1_Pooling/config.json"},
    {"revision": "b" * 40},
    {"sha256": "not-a-digest"},
    {"bytes": "0"},
    {},
])
def test_manifest_validation_precedes_downloads_including_duplicate_destinations(tmp_path, change):
    row = _row(b"fixture")
    manifest = _manifest(tmp_path / "manifest.csv", [row, {**row, **change}])
    calls = []
    with pytest.raises(ValueError):
        model_resources(manifest, tmp_path / "Resource", fetch=True, downloader=lambda **kw: calls.append(kw))
    assert calls == []

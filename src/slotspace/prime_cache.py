"""Inputs to M0, pinned to the public STaRK release (never unpickle arbitrary caches)."""

import hashlib
import json
import shutil
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

REPO = "snap-stanford/stark"
REVISION = "88269e23e90587f99476c5dd74e235a0877e69be"


def _verify(path: Path, expected: str) -> None:
    with path.open("rb") as src:
        if hashlib.file_digest(src, "sha256").hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")


def prepare_prime(root: Path, download: bool = False) -> Path:
    """Return root/prime in the STaRK layout. Only download into a new directory."""
    hashes = json.loads(files("slotspace").joinpath("prime-sha256.json").read_text())
    prime = root / "prime"
    if not prime.exists() and download:
        from huggingface_hub import hf_hub_download

        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".prime-", dir=root) as tmp:
            stage = Path(tmp) / "prime"
            stage.mkdir()
            for name, digest in hashes.items():
                if name.startswith("processed/"):
                    continue
                remote = "skb/prime/processed.zip" if name == "processed.zip" else f"qa/prime/{name}"
                cached = hf_hub_download(REPO, remote, revision=REVISION, repo_type="dataset",
                                         cache_dir=root / ".hub", token=False)
                dest = stage / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, dest)
                _verify(dest, digest)
            with ZipFile(stage / "processed.zip") as archive:
                (stage / "processed").mkdir()
                for name, digest in hashes.items():
                    if name.startswith("processed/"):
                        with archive.open(name) as src, (stage / name).open("xb") as dest:
                            shutil.copyfileobj(src, dest)
                        _verify(stage / name, digest)
            stage.rename(prime)

    if not prime.is_dir():
        raise FileNotFoundError(f"No Prime cache at {prime}; specify --root or opt in to --download")
    expected = {name for name in hashes if name.startswith("processed/")}
    actual = {p.relative_to(prime).as_posix() for p in (prime / "processed").iterdir()}
    if actual != expected:
        raise ValueError(f"Unexpected Prime processed cache contents: {sorted(actual ^ expected)}")
    for name, digest in hashes.items():
        if name != "processed.zip":  # a verified unpacked cache needs no archive
            _verify(prime / name, digest)
    return prime

"""Check actual release archives for required assets and accidental benchmark/weight bundling."""

import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path, PurePosixPath
import tarfile
from zipfile import ZipFile

from packaging.specifiers import SpecifierSet

FORBIDDEN_ROOTS = {"data", "runs", "resource", ".envs", ".venv", ".git", "build", "dist", ".pytest_cache"}
FORBIDDEN_SUFFIXES = {".pkl", ".pt", ".npy", ".npz", ".safetensors", ".onnx", ".bin", ".jsonl"}


def check(path):
    if path.suffix == ".whl":
        with ZipFile(path) as archive:
            names = archive.namelist()
            meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(archive.read(meta_name))
        required = {"slotspace/reproduce.py", "slotspace/resources.py", "slotspace/operator.py",
                    "slotspace/model_card.py", "slotspace/vespa_app/app/services.xml",
                    "slotspace/vespa_app/app/hosts.xml", "slotspace/vespa_app/app/schemas/doc.sd"}
        if not required <= set(names):
            raise ValueError("Wheel is missing implementation/application resources")
        prefix = meta_name.removesuffix("METADATA")
        if not {prefix + f"licenses/{n}" for n in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")} <= set(names):
            raise ValueError("Wheel is missing license/attribution files")
        shared = ("schemas/prime.yaml", "environments/research.yaml", "environments/research.constraints.txt",
                  "environments/generation.yaml", "environments/generation-linux.lock.txt",
                  "reproducibility/huggingface.csv")
        if any(not any(n.endswith("/share/slotspace/" + asset) for n in names) for asset in shared):
            raise ValueError("Wheel is missing the shared research assets")
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            if any(m.issym() or m.islnk() for m in archive.getmembers()):
                raise ValueError("Release source archive must not use symlinked inputs")
            roots = {PurePosixPath(m.name).parts[0] for m in members}
            if len(roots) != 1:
                raise ValueError("Source archive must have one top-level directory")
            names = ["/".join(PurePosixPath(m.name).parts[1:]) for m in members]
            member = next(m for m in members if m.name.endswith("/PKG-INFO") and len(PurePosixPath(m.name).parts) == 2)
            metadata = BytesParser().parsebytes(archive.extractfile(member).read())
        required = {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "CITATION.cff", "CHANGELOG.md",
                    "Makefile", "pyproject.toml", "docs/MODEL_CARD.md", "docs/DECISIONS.md",
                    "docs/dependency-licenses.csv", "configs/prime_v0.yaml", "schemas/prime.yaml",
                    "templates/train/prime_slots.yaml", "templates/test/role_swap.yaml",
                    "src/slotspace/reproduce.py", "src/slotspace/resources.py", "src/slotspace/operator.py",
                    "src/slotspace/model_card.py", "tests/test_reference.py", "tools/check_dist.py",
                    "docs/GPU_SETUP.md", "environments/research.yaml", "environments/research.constraints.txt",
                    "environments/generation.yaml", "environments/generation-linux.lock.txt",
                    "reproducibility/huggingface.csv"}
        if not required <= set(names):
            raise ValueError(f"Source archive is missing release/research assets: {sorted(required - set(names))}")
    else:
        raise ValueError(f"Unsupported release archive: {path}")
    for name in names:
        pure = PurePosixPath(name)
        effective = pure.parts
        if len(effective) >= 3 and effective[0].endswith(".data") and effective[1] == "data":
            effective = effective[2:]
        license_tables = {"dependency-licenses.csv", "generator-licenses.csv", "model-licenses.csv"}
        input_manifest = "/".join(effective) in {
            "reproducibility/huggingface.csv", "share/slotspace/reproducibility/huggingface.csv"}
        if (pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in pure.parts[0]
                or effective[0].casefold() in FORBIDDEN_ROOTS or pure.suffix in FORBIDDEN_SUFFIXES
                or pure.suffix == ".csv" and pure.name not in license_tables and not input_manifest
                or pure.suffix == ".json" and pure.name != "prime-sha256.json"):
            raise ValueError(f"Forbidden release payload: {name}")
    if metadata.get("Name") != "slotspace" or metadata.get("License-Expression") != "Apache-2.0":
        raise ValueError("Release package identity/SPDX license does not match SlotSpace")
    if not metadata.get("Version") or not metadata.get("Requires-Python"):
        raise ValueError("Release package omitted version/Python constraints")
    if SpecifierSet(metadata["Requires-Python"]) != SpecifierSet(">=3.11,<3.12"):
        raise ValueError("Release Python constraints must preserve the documented 3.11 environment")
    return {"path": str(path), "version": metadata["Version"], "files": len(names),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="*")
    parser.add_argument("--dist", type=Path, help="Inspect every wheel/source archive in this directory")
    args = parser.parse_args()
    archives = args.archives + (sorted(args.dist.glob("*.whl")) + sorted(args.dist.glob("*.tar.gz")) if args.dist else [])
    if not archives:
        parser.error("Supply release archives or a nonempty --dist directory")
    for path in archives:
        result = check(path)
        print(f"{result['sha256']}  {path.name} ({result['files']} files; version {result['version']})")


if __name__ == "__main__":
    main()

"""Fetch or verify public pinned inputs; never overwrite mismatched local files."""

import argparse
from collections.abc import Callable, Mapping
import csv
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from .artifacts import sha256
from .prime_cache import prepare_prime


def source_spec(row: Mapping[str, str]) -> tuple[tuple[str, ...], str, str, str]:
    normalized = row["path"].replace("\\", "/")
    parts = tuple(normalized.split("/"))
    if (not parts or any(p in {"", ".", ".."} or ":" in p for p in parts)
            or PurePosixPath(normalized).is_absolute() or PureWindowsPath(normalized).drive):
        raise ValueError("Resource manifest contains an unsafe destination")
    url = urlsplit(row["source"])
    if (url.scheme != "https" or url.netloc != "huggingface.co" or url.query or url.fragment):
        raise ValueError("Resources must come from the pinned public Hugging Face origin")
    remote = url.path.strip("/").split("/")
    kind = "dataset" if remote[0] == "datasets" else "model"
    offset = 1 if kind == "dataset" else 0
    if (len(remote) < offset + 5 or remote[offset + 2] != "resolve"
            or remote[offset + 3] != row["revision"]
            or not re.fullmatch("[0-9a-f]{40}", row["revision"])
            or not re.fullmatch("[0-9a-f]{64}", row["sha256"])
            or any(p in {"", ".", ".."} for p in remote)):
        raise ValueError("Resource URL/revision/checksum is not a pinned manifest entry")
    repo = "/".join(remote[offset:offset + 2])
    name = "/".join(remote[offset + 4:])
    prefix = ("v1", "models") if parts[:2] == ("v1", "models") else ("models",)
    if kind != "model" or parts != (*prefix, *repo.split("/"), *name.split("/")) or int(row["bytes"]) < 1:
        raise ValueError("Model destination, source filename or byte count is incompatible")
    return parts, repo, name, kind


def _check_file(path: Path, row: Mapping[str, str]) -> None:
    if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
        raise ValueError(f"Resource size/SHA-256 mismatch: {path}; no existing input was overwritten")


def model_resources(manifest: Path, root: Path, *, fetch: bool = False, include_v1: bool = False,
                    downloader: Callable[..., str] | None = None) -> int:
    if fetch and downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    with manifest.open(encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if not {"path", "source", "revision", "bytes", "sha256"} <= set(reader.fieldnames or ()):
            raise ValueError("Resource manifest is missing required columns")
        rows = list(reader)
    selected, seen = [], set()
    for row in rows:
        if any(not isinstance(row.get(k), str) for k in ("path", "source", "revision", "bytes", "sha256")):
            raise ValueError("Resource manifest contains an incomplete row")
        normalized = row["path"].replace("\\", "/")
        if not (normalized.startswith("models/") or include_v1 and normalized.startswith("v1/models/")):
            continue
        parts, repo, name, kind = source_spec(row)
        dest = root.joinpath(*parts)
        if not dest.resolve().is_relative_to(root.resolve()) or normalized.casefold() in seen:
            raise ValueError("Resource destination escapes the cache or is duplicated")
        seen.add(normalized.casefold())
        selected.append((row, dest, repo, name, kind))
    if not selected:
        raise ValueError("No model inputs selected by the manifest")
    for row, dest, repo, name, kind in selected:
        if dest.exists():
            _check_file(dest, row)
            continue
        if not fetch:
            raise FileNotFoundError(f"Missing model input: {dest}; run slotspace.resources fetch")
        dest.parent.mkdir(parents=True, exist_ok=True)
        cached = downloader(repo_id=repo, filename=name, revision=row["revision"], repo_type=kind,
                            cache_dir=root / ".hub", token=False)
        with TemporaryDirectory(prefix=".resource-", dir=dest.parent) as temp:
            candidate = Path(temp) / dest.name
            shutil.copyfile(cached, candidate)
            _check_file(candidate, row)
            # Exclusive publication avoids replacing a concurrently created cache entry.
            os.link(candidate, dest)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "verify"))
    parser.add_argument("--manifest", type=Path, default=Path("reproducibility/huggingface.csv"))
    parser.add_argument("--root", type=Path, default=Path("../Resource"))
    parser.add_argument("--include-v1-models", action="store_true")
    args = parser.parse_args()
    count = model_resources(args.manifest, args.root, fetch=args.action == "fetch",
                            include_v1=args.include_v1_models)
    prime = prepare_prime(args.root / "datasets" / "stark", download=args.action == "fetch")
    print(f"Verified {count} model files and the pinned Prime cache at {prime}")


if __name__ == "__main__":
    main()

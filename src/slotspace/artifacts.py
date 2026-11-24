"""Immutable caches and reproducible run metadata."""

import hashlib
import json
import platform
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .stark_io import _git


def sha256(path: Path) -> str:
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


@contextmanager
def new_directory(path: Path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as tmp:
        stage = Path(tmp) / "artifact"
        stage.mkdir()
        yield stage
        stage.rename(path)


@contextmanager
def logged_run(root: Path, label: str, config: dict):
    now = datetime.now(timezone.utc)
    path = root / f"{label}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    path.mkdir(parents=True)
    meta = {
        "config": config, "git_commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--porcelain"), "started_at": now.isoformat(),
        "machine": platform.node(), "platform": platform.platform(), "python": sys.version,
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "status": "failed",
    }
    write_json(path / "meta.json", {**meta, "status": "running"})
    try:
        yield path, meta
        meta["status"] = "ok"
    finally:
        meta["ended_at"] = datetime.now(timezone.utc).isoformat()
        write_json(path / "meta.json", meta)

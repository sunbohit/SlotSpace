"""Inventory exact installed license declarations/text hashes; not legal approval."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
from urllib.request import urlopen

import yaml


def inventory():
    rows = []
    for dist in sorted(distributions(), key=lambda d: d.metadata["Name"].lower()):
        name, version = dist.metadata["Name"], dist.version
        if name.lower().replace("_", "-") == "slotspace":
            continue
        classifiers = [c.removeprefix("License :: ") for c in dist.metadata.get_all("Classifier", [])
                       if c.startswith("License :: ")]
        declaration = dist.metadata.get("License-Expression", "")
        text = dist.metadata.get("License", "")
        license_files = []
        for file in dist.files or []:
            if Path(file).name.lower().startswith(("license", "copying", "notice", "copyright")):
                path = Path(dist.locate_file(file))
                if path.is_file():
                    license_files.append((str(file).replace("\\", "/"), hashlib.sha256(path.read_bytes()).hexdigest()))
        summary = text.splitlines()[0][:180] if text else ""
        status = "declared metadata and bundled-text inventory"
        if not declaration and not classifiers and not text and not license_files:
            status = "UNRESOLVED: no declaration or bundled license text"
        rows.append({"package": name, "version": version, "license_expression": declaration,
                     "license_classifiers": "; ".join(classifiers), "license_metadata_first_line": summary,
                     "license_metadata_sha256": hashlib.sha256(text.encode()).hexdigest() if text else "",
                     "bundled_license_files": "; ".join(p for p, _ in license_files),
                     "bundled_license_sha256": "; ".join(f"{p}={h}" for p, h in license_files),
                     "source": f"https://pypi.org/project/{name}/{version}/", "review": status})
    return rows


def model_inventory(resources):
    with (resources / "manifests" / "huggingface.csv").open(encoding="utf-8-sig", newline="") as src:
        pinned = {r["path"].replace("\\", "/"): r for r in csv.DictReader(src)}
    rows = []
    for base in (resources / "models", resources / "v1" / "models"):
        for card in sorted(base.glob("*/*/README.md")):
            relative = card.relative_to(resources).as_posix()
            record = pinned[relative]
            digest = hashlib.sha256(card.read_bytes()).hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"Cached model card changed: {relative}")
            metadata = yaml.safe_load(card.read_text(encoding="utf-8").split("---", 2)[1])
            if not metadata.get("license"):
                raise ValueError(f"No upstream license declaration: {relative}")
            rows.append({"model": card.parent.parent.name + "/" + card.parent.name,
                         "license": metadata["license"], "revision": record["revision"],
                         "source": record["source"], "card_sha256": digest,
                         "review": "pinned upstream model-card declaration; no weights redistributed"})
    return rows


def generator_inventory(lock):
    from packaging.requirements import Requirement

    specs = [Requirement(line) for line in lock.read_text().splitlines() if line and not line.startswith("#")]

    def metadata(req):
        version = next(iter(req.specifier)).version
        url = f"https://pypi.org/pypi/{req.name}/{version}/json"
        with urlopen(url, timeout=30) as src:
            info = json.load(src)["info"]
        text = info.get("license") or ""
        return {"package": req.name, "version": version, "license_expression": info.get("license_expression") or "",
                "license_classifiers": "; ".join(c for c in info.get("classifiers", []) if c.startswith("License ::")),
                "license_metadata_first_line": text.splitlines()[0][:180] if text else "",
                "metadata_license_sha256": hashlib.sha256(text.encode()).hexdigest() if text else "",
                "source": url, "review": "upstream declaration only; Linux/CUDA wheel contents NOT INSPECTED"}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(metadata, specs))


def write_rows(path, rows):
    if not rows:
        raise ValueError("License inventory cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--models-out", type=Path)
    parser.add_argument("--generator-lock", type=Path)
    parser.add_argument("--generator-out", type=Path)
    args = parser.parse_args()
    rows = inventory()
    write_rows(args.out, rows)
    if args.models_out:
        if args.resources is None:
            parser.error("--models-out requires --resources")
        models = model_inventory(args.resources)
        write_rows(args.models_out, models)
        print(f"{len(models)} pinned model license declarations checked")
    if args.generator_out:
        if args.generator_lock is None:
            parser.error("--generator-out requires --generator-lock")
        generator = generator_inventory(args.generator_lock)
        write_rows(args.generator_out, generator)
        print(f"{len(generator)} Linux generator dependency declarations inventoried (not wheel-license review)")
    unresolved = [r["package"] for r in rows if r["review"].startswith("UNRESOLVED")]
    print(f"{len(rows)} exact distributions inventoried; unresolved declarations/texts: {unresolved}")
    if unresolved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

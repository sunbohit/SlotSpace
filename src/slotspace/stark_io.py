"""Export the Prime corpus, graph and official QA splits without running a retriever."""

import argparse
import csv
import json
import platform
import random
import shutil
import subprocess
import sys
from collections import Counter
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .prime_cache import REVISION, prepare_prime

SPLITS = ("train", "val", "test", "test-0.1", "human")
DOC_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("doc_type", pa.string()), ("name", pa.string()),
    ("text", pa.string()), ("details_json", pa.string()),
])
EDGE_SCHEMA = pa.schema([("src", pa.int64()), ("dst", pa.int64()), ("rel", pa.string())])


def norm_name(s: str) -> str:
    return s.strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")


def _clean_details(details) -> dict[str, str]:
    out = {}
    for key, value in (details or {}).items():
        if key.startswith("_") or "_id" in key or value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(map(str, value))
        text = str(value).strip()
        if text and text.lower() != "nan":
            out[key] = text
    return out


def _text_view(skb):
    # A sparse COO row selection in STaRK scans the relation's entire graph.
    # Index the same sorted, de-duplicated rows; leave its text formatter untouched.
    rows = {}
    for rel, adjacency in skb.sparse_adj_by_type.items():
        src, dst = adjacency.coalesce().indices().numpy()
        ends = np.bincount(src, minlength=len(skb.node_types)).cumsum()
        rows[rel] = (np.concatenate(([0], ends)), dst)

    def neighbors(idx, edge_type="*"):
        if edge_type == "*":
            return skb.get_neighbor_nodes(idx, edge_type)
        offsets, indices = rows[edge_type]
        return indices[offsets[idx]:offsets[idx + 1]].tolist()

    view = copy(skb)
    view.get_neighbor_nodes = neighbors
    return view


def _write_docs(skb, ids: list[int], dest: Path) -> dict[str, int]:
    types = Counter()
    with pq.ParquetWriter(dest, DOC_SCHEMA, compression="zstd") as writer:
        for start in range(0, len(ids), 1024):
            rows = []
            for i in ids[start:start + 1024]:
                node = skb.node_info[i]
                doc_type = norm_name(node["type"])
                types[doc_type] += 1
                rows.append({
                    "doc_id": i, "doc_type": doc_type, "name": str(node["name"]),
                    "text": skb.get_doc_info(i, add_rel=True, compact=True),
                    "details_json": json.dumps(_clean_details(node.get("details")), ensure_ascii=False),
                })
            writer.write_table(pa.Table.from_pylist(rows, schema=DOC_SCHEMA))
    return dict(sorted(types.items()))


def _write_edges(skb, dest: Path, subset: np.ndarray | None) -> int:
    edges, codes = skb.edge_index.numpy(), skb.edge_types.numpy()
    if edges.shape != (2, len(codes)):
        raise ValueError("STaRK edge indices and relation codes are not aligned")
    keys = np.array(sorted(skb.edge_type_dict), dtype=np.int64)
    names = pa.array([skb.edge_type_dict[k] for k in keys], type=pa.string())
    count = 0
    with pq.ParquetWriter(dest, EDGE_SCHEMA, compression="zstd") as writer:
        for start in range(0, len(codes), 131072):
            chunk = slice(start, start + 131072)
            src, dst = edges[:, chunk]
            rel = codes[chunk]
            if subset is not None:
                keep = np.isin(src, subset) & np.isin(dst, subset)
                src, dst, rel = src[keep], dst[keep], rel[keep]
            if not len(rel):
                continue
            idx = keys.searchsorted(rel)
            if idx.max() >= len(keys) or not np.array_equal(keys[idx], rel):
                raise ValueError("Unknown STaRK edge relation code")
            writer.write_table(pa.Table.from_arrays([src, dst, pc.take(names, idx)], schema=EDGE_SCHEMA))
            count += len(rel)
    return count


def _read_qa(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as src:
        for row in csv.DictReader(src):
            qid = int(row["id"])
            answers = json.loads(row["answer_ids"])
            if qid in rows or not isinstance(answers, list) or not answers or any(type(a) is not int for a in answers):
                raise ValueError(f"Invalid QA record {qid} in {path}")
            rows[qid] = {"qid": qid, "query": row["query"], "answers": answers}
    return rows


def _write_queries(prime: Path, out: Path, subset: set[int] | None) -> dict[str, int]:
    # The released load_qa lists the Hub even for a local root. These are its input
    # tables: split entries are query IDs, not offsets into the CSV.
    tables = prime / "stark_qa"
    synth = _read_qa(tables / "stark_qa.csv")
    human = _read_qa(tables / "stark_qa_human_generated_eval.csv")
    counts = {}
    for split in SPLITS:
        if split == "human":
            rows, ids = human, sorted(human)
        else:
            rows = synth
            ids = [int(x) for x in (prime / "split" / f"{split}.index").read_text().splitlines()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate qid in {split}")
        count = 0
        with (out / f"queries_{split}.jsonl").open("x", encoding="utf-8", newline="\n") as f:
            for qid in ids:
                row = rows[qid]
                if subset is None or subset.issuperset(row["answers"]):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
        counts[split] = count
    return counts


def _git(*args: str) -> str | None:
    if not shutil.which("git"):
        return None
    cmd = ["git", "--no-pager", "--no-optional-locks", "-C", str(Path(__file__).resolve().parent), *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def export_prime(out_dir: str | Path = "data/prime", root: str | Path = "data/stark_cache",
                 limit: int | None = None, *, download: bool = False,
                 runs: str | Path = "runs") -> None:
    out = Path(out_dir).resolve()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose a new --out directory")
    start = datetime.now(timezone.utc)
    log_dir = Path(runs) / f"m0-{start:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    log_dir.mkdir(parents=True)
    meta = {
        "started_at": start.isoformat(), "git_commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--porcelain"), "machine": platform.node(),
        "platform": platform.platform(), "python": sys.version,
        "config": {"out": str(out), "root": str(Path(root).resolve()), "limit": limit,
                   "download": download, "seed": 13, "dataset_revision": REVISION},
        "status": "failed",
    }
    try:
        meta["packages"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
        prime = prepare_prime(Path(root), download=download)
        import torch
        from stark_qa import load_skb

        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        skb = load_skb("prime", root=str(prime.parent), download_processed=True)
        ids = [int(i) for i in skb.candidate_ids[:limit]]
        out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".prime-export-", dir=out.parent) as tmp:
            stage = Path(tmp) / "export"
            stage.mkdir()
            counts = {
                "docs": len(ids), "types": _write_docs(_text_view(skb), ids, stage / "docs.parquet"),
                "edges": _write_edges(skb, stage / "edges.parquet", None if limit is None else np.asarray(ids)),
                "queries": _write_queries(prime, stage, None if limit is None else set(ids)),
            }
            stage.rename(out)
        meta.update(status="ok", counts=counts)
        print(json.dumps(counts, sort_keys=True))
    finally:
        meta["ended_at"] = datetime.now(timezone.utc).isoformat()
        (log_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/stark_cache"), help="Parent of the prime/ cache")
    parser.add_argument("--out", type=Path, help="New output directory; defaults to data/prime or data/prime-sample-N")
    parser.add_argument("--limit", type=int, help="First N candidates, induced edges and fully answerable queries")
    parser.add_argument("--download", action="store_true", help="Fetch the pinned release if the cache is absent")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    out = args.out or Path("data") / ("prime" if args.limit is None else f"prime-sample-{args.limit}")
    export_prime(out, args.root, args.limit, download=args.download, runs=args.runs)


if __name__ == "__main__":
    main()

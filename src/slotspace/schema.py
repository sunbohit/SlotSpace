"""Versioned slot schema and the slot inventory derived from Prime edges."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from .stark_io import norm_name


@dataclass(frozen=True)
class Slot:
    id: str
    doc_type: str | None
    value_type: str
    relation: str | None
    direction: str
    description: str
    query_cues: tuple[str, ...] = ()
    doc_cues: tuple[str, ...] = ()
    n_edges: int = 0
    collides_with: tuple[str, ...] = ()


class Schema:
    @classmethod
    def load(cls, path: str | Path, expected_version: str | None = None):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.name, obj.version = data["name"], data["version"]
        if not isinstance(obj.version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", obj.version):
            raise ValueError("Schema version must be a semantic version string")
        if expected_version is not None and obj.version != expected_version:
            raise ValueError(f"Schema version mismatch: {obj.version} != {expected_version}")
        obj.value_types = data["value_types"]
        obj.doc_types = data["doc_types"]
        obj.conventions = data["conventions"]
        obj.slots: list[Slot] = []
        obj._slots: dict[str, Slot] = {}
        for raw in data["slots"]:
            slot = Slot(**{**raw, "query_cues": tuple(raw.get("query_cues", ())),
                           "doc_cues": tuple(raw.get("doc_cues", ())),
                           "collides_with": tuple(raw.get("collides_with", ()))})
            if (slot.doc_type not in obj.doc_types or slot.value_type not in obj.value_types
                    or slot.direction not in {"out", "in", "sym"} or slot.n_edges < 0):
                raise ValueError(f"Invalid slot types, direction or edge count: {slot.id}")
            if slot.relation is not None:
                expected = f"{slot.doc_type}.{norm_name(slot.relation)}.{slot.value_type}"
                if slot.id != expected:
                    raise ValueError(f"Slot ID does not match its role: {slot.id}")
            if any(not 3 <= len(cues) <= 6 or not all(cues) for cues in (slot.query_cues, slot.doc_cues)):
                raise ValueError(f"Each slot needs 3-6 query/doc cues: {slot.id}")
            obj._add(slot)
            obj.slots.append(slot)
        for value_type, description in data["none"].items():
            if value_type not in obj.value_types:
                raise ValueError(f"Unknown none type: {value_type}")
            obj._add(Slot(f"none.{value_type}", None, value_type, None, "none", description))
        if set(data["none"]) != set(obj.value_types):
            raise ValueError("Every value type needs a none outcome")
        return obj

    def _add(self, slot: Slot) -> None:
        if slot.id in self._slots or "|" in slot.id or not slot.description.strip():
            raise ValueError(f"Duplicate/invalid slot ID or empty description: {slot.id}")
        self._slots[slot.id] = slot

    def slot(self, slot_id: str) -> Slot:
        return self._slots[slot_id]

    def candidates(self, doc_type: str | None, value_types: set[str]) -> list[str]:
        if not value_types <= self.value_types.keys() or (doc_type is not None and doc_type not in self.doc_types):
            raise ValueError("Candidate selection received an unknown document/value type")
        return ([s.id for s in self.slots if s.value_type in value_types
                 and (doc_type is None or s.doc_type == doc_type)]
                + [f"none.{t}" for t in sorted(value_types)])

    def slot_text(self, slot_id: str) -> str:
        slot = self.slot(slot_id)
        return f"{slot.id}: {slot.description}"


def derive_slots(docs: pd.DataFrame, edges: pd.DataFrame) -> list[dict]:
    if docs.doc_id.duplicated().any() or docs.doc_id.min() < 0:
        raise ValueError("Schema derivation requires distinct nonnegative document IDs")
    types = docs.set_index("doc_id")["doc_type"]
    e = edges.assign(st=edges.src.map(types), dt=edges.dst.map(types))
    if e.isna().any().any():
        raise ValueError("Edges contain missing types or unknown document IDs")
    stride = int(docs.doc_id.max()) + 1
    if stride > 3_037_000_499:
        raise ValueError("Node IDs exceed the supported int64 reciprocal-edge encoding")
    for relation, group in e.groupby("rel", observed=True):
        src, dst = group.src.to_numpy(), group.dst.to_numpy()
        if not np.array_equal(np.sort(src * stride + dst), np.sort(dst * stride + src)):
            raise ValueError(f"Prime relation is not stored symmetrically: {relation}")
    counts = e.groupby(["st", "rel", "dt"], observed=True).size()
    return [
        {"id": f"{st}.{norm_name(rel)}.{dt}", "doc_type": st, "relation": rel, "value_type": dt,
         "direction": "sym" if st == dt else "out", "n_edges": int(n)}
        for (st, rel, dt), n in counts.items()
    ]


def verify_inventory(schema: Schema, derived: list[dict]) -> None:
    observed = {row["id"]: row for row in derived}
    expected = {s.id: s for s in schema.slots if s.relation is not None}
    if observed.keys() != expected.keys():
        raise ValueError(f"Schema and corpus slot sets differ: {sorted(observed.keys() ^ expected.keys())}")
    for key, row in observed.items():
        slot = expected[key]
        if any(getattr(slot, k) != v for k, v in row.items()):
            raise ValueError(f"Schema role/count differs from the corpus: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("data/prime/docs.parquet"))
    parser.add_argument("--edges", type=Path, default=Path("data/prime/edges.parquet"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
    args = parser.parse_args()
    schema = Schema.load(args.schema)
    derived = derive_slots(pd.read_parquet(args.docs, columns=["doc_id", "doc_type"]), pd.read_parquet(args.edges))
    verify_inventory(schema, derived)
    print(f"{len(derived)} slots, {sum(r['n_edges'] for r in derived)} stored edges; all reciprocal")


if __name__ == "__main__":
    main()

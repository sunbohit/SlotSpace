"""Independent training frames and corpus-derived hard-negative role pairs."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

import pandas as pd
import yaml

from ..artifacts import sha256
from ..lexicon_build import CorpusLexicon
from ..schema import Schema, Slot

QUERY_FRAMES = (
    "Which {ANS} {predicate}?", "What {ANS} {predicate}?",
    "List {ANS} that {predicate}.", "Find {ANS} that {predicate}.",
    "Identify {ANS} that {predicate}.", "Name {ANS} that {predicate}.",
    "Show {ANS} that {predicate}.", "Return {ANS} that {predicate}.",
    "I need {ANS} that {predicate}.", "What are some {ANS} that {predicate}?",
    "Are there any {ANS} that {predicate}?", "Give examples of {ANS} that {predicate}.",
)
DOC_FRAMES = (
    "{clause}.", "The record states that {clause}.", "It is reported that {clause}.",
    "Available information indicates that {clause}.", "The description explains that {clause}.",
    "According to the record, {clause}.", "The evidence states that {clause}.",
    "This account notes that {clause}.", "The available record says that {clause}.",
    "One observation is that {clause}.", "The following relationship is recorded: {clause}.",
    "In the present description, {clause}.",
)


def predicates(slot: Slot) -> tuple[str, str]:
    """Return a plural query predicate and a single-SELF document clause."""
    rel, kind = slot.relation, slot.doc_type
    if rel == "indication":
        return (("are indicated for treating {T}", "{SELF} is indicated for treating {T}") if kind == "drug"
                else ("are treated with {T}", "{SELF} is treated with {T}"))
    if rel == "contraindication":
        return (("must be avoided in people with {T}", "{SELF} is contraindicated in patients with {T}") if kind == "drug"
                else ("make {T} contraindicated", "{SELF} makes the use of {T} contraindicated"))
    if rel == "off-label use":
        return (("are used off-label for {T}", "{SELF} is used off-label for {T}") if kind == "drug"
                else ("are treated off-label with {T}", "{SELF} is treated off-label with {T}"))
    if rel == "side effect":
        return (("cause {T} as a side effect", "{SELF} can cause {T} as a side effect") if kind == "drug"
                else ("are side effects of {T}", "{SELF} is a side effect of {T}"))
    if rel in {"expression present", "expression absent"}:
        neg = "not " if rel == "expression absent" else ""
        if kind == "anatomy":
            return (f"show {'no detectable ' if neg else ''}expression of {{T}}",
                    f"{{T}} is {neg}expressed in {{SELF}}")
        return (f"are {neg}expressed in {{T}}", f"{{SELF}} is {neg}expressed in {{T}}")
    if rel in {"phenotype present", "phenotype absent"}:
        if kind == "disease":
            return (("do not exhibit {T}", "{SELF} does not exhibit {T}") if rel.endswith("absent")
                    else ("exhibit {T}", "{SELF} exhibits the phenotype {T}"))
        return (("are absent from the disease {T}", "{SELF} is absent from the disease {T}") if rel.endswith("absent")
                else ("are present in the disease {T}", "{SELF} is present in the disease {T}"))
    if rel == "target":
        return (("target {T}", "{SELF} directly targets {T}") if kind == "drug"
                else ("are targeted by {T}", "{SELF} is targeted by {T}"))
    if rel == "enzyme":
        return (("are metabolized by {T}", "{SELF} is metabolized by {T}") if kind == "drug"
                else ("metabolize {T}", "{SELF} enzymatically metabolizes {T}"))
    if rel == "transporter":
        return (("are transported by {T}", "{SELF} is transported by {T}") if kind == "drug"
                else ("transport {T}", "{SELF} transports {T} across membranes"))
    if rel == "carrier":
        return (("are carried by {T}", "{SELF} is carried by {T}") if kind == "drug"
                else ("carry {T}", "{SELF} carries {T} in the body"))
    ordinary = {
        "associated with": ("are associated with {T}", "{SELF} is associated with {T}"),
        "linked to": ("are linked to the exposure or condition {T}", "{SELF} is linked to {T}"),
        "interacts with": ("interact functionally with {T}", "{SELF} interacts functionally with {T}"),
        "ppi": ("physically interact with {T}", "{SELF} physically interacts with {T}"),
        "parent-child": ("are broader or narrower categories related to {T}",
                         "{SELF} has a broader or narrower hierarchical relationship with {T}"),
        "synergistic interaction": ("act synergistically with {T}", "{SELF} acts synergistically with {T}"),
    }
    if rel not in ordinary:
        raise ValueError(f"No training predicate for {slot.id}")
    return ordinary[rel]


def mine_collisions(table: pd.DataFrame, schema: Schema) -> dict[str, list[str]]:
    collisions = defaultdict(set)
    for _, group in table[table.hard >= 3].groupby("value_id", sort=False):
        types = defaultdict(list)
        for key in group.slot:
            slot = schema.slot(key)
            types[(slot.doc_type, slot.value_type)].append(key)
        for keys in types.values():
            for a, b in combinations(sorted(keys), 2):
                collisions[a].add(b)
                collisions[b].add(a)
    return {slot.id: sorted(collisions[slot.id]) for slot in schema.slots}


def make_templates(schema: Schema, collisions: dict[str, list[str]]) -> dict:
    result = {}
    for slot in schema.slots:
        query, doc = predicates(slot)
        result[slot.id] = {
            "query": [f.replace("{predicate}", query) for f in QUERY_FRAMES],
            "doc": [f.replace("{clause}", doc) for f in DOC_FRAMES],
            "collides_with": collisions.get(slot.id, []),
        }
    query = "are described in ongoing studies concerning {T}"
    doc = "{SELF} is discussed in the same report as {T}"
    for kind in schema.value_types:
        result[f"none.{kind}"] = {
            "query": [f.replace("{predicate}", query) for f in QUERY_FRAMES],
            "doc": [f.replace("{clause}", doc) for f in DOC_FRAMES], "collides_with": [],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
    parser.add_argument("--lexicon", type=Path, default=Path("data/prime/lexicon_native"))
    parser.add_argument("--out", type=Path, default=Path("templates/train/prime_slots.yaml"))
    args = parser.parse_args()
    schema = Schema.load(args.schema)
    CorpusLexicon.load(args.lexicon)
    meta = json.loads((args.lexicon / "manifest.json").read_text(encoding="utf-8"))
    if meta["schema_sha256"] != sha256(args.schema):
        raise ValueError("Training frames require the matching lexicon/schema")
    table = pd.read_parquet(args.lexicon / meta["file"], columns=["value_id", "slot", "hard"])
    result = make_templates(schema, mine_collisions(table, schema))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as dest:
        yaml.safe_dump(result, dest, sort_keys=False, allow_unicode=False)
    print(f"{len(result)} slot/none template families written; no test templates read")


if __name__ == "__main__":
    main()

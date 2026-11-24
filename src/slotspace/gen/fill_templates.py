"""Fill training-only frames with surrogate tokens (or explicitly marked A12 targets)."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

import pandas as pd
import yaml

from ..artifacts import logged_run, new_directory, sha256, write_json
from ..schema import Schema
from .templates import predicates

PLACEHOLDER = re.compile(r"\{(T|SELF|ANS|O:[a-z_]+)\}")


def _pseudo(rng: random.Random, used: dict) -> str:
    while True:
        word = "".join(rng.choice("bcdfghjklmnprstvz") + rng.choice("aeiou") for _ in range(2))
        token = word.capitalize() + f"{rng.randrange(100):02d}"
        if token not in used:
            return token


def fill_sample(schema: Schema, family: dict, slot_id: str, style: str, index: int, rng: random.Random,
                sample_id: str, *, real_names: dict[str, list[str]] | None = None,
                bindings: dict[str, str] | None = None, second_slot: str | None = None,
                doc_type: str | None = None) -> dict:
    slot = schema.slot(slot_id)
    doc_type = slot.doc_type or doc_type or rng.choice(sorted(schema.doc_types))
    template = family[style][index]
    if second_slot is not None:
        other = schema.slot(second_slot)
        if other.doc_type != doc_type:
            raise ValueError("Secondary mention must describe the same document type")
        q, d = predicates(other)
        clause = q if style == "query" else d.replace("{SELF}", "it")
        template = template.rstrip(".?") + "; " + clause.replace("{T}", f"{{O:{other.value_type}}}") + "."
    tokens, chosen = {}, dict(bindings or {})

    def replace(match):
        key = match.group(1)
        if key == "ANS":
            if key not in chosen:
                chosen[key] = rng.choice(schema.conventions["answer_phrases"][doc_type])
            return chosen[key]
        if key not in chosen:
            if key == "T" and real_names is not None:
                options = real_names.get(slot.value_type, [])
                if not options:
                    raise ValueError(f"A12 has no real names for {slot.value_type}")
                chosen[key] = rng.choice(options)
            else:
                chosen[key] = _pseudo(rng, {**tokens, **dict.fromkeys(chosen.values())})
        token = chosen[key]
        role = "target" if key == "T" else "self" if key == "SELF" else f"other:{key[2:]}"
        tokens[token] = role
        return token

    leftover = PLACEHOLDER.sub("", template)
    if "{" in leftover or "}" in leftover:
        raise ValueError(f"Unknown training placeholder in {slot_id}")
    text = PLACEHOLDER.sub(replace, template)
    return {
        "sample_id": sample_id, "slot": slot_id, "value_type": slot.value_type, "style": style,
        "doc_type": doc_type, "template_id": f"{slot_id}:{style}:{index}",
        "text": text, "tokens": tokens, "bindings": chosen, "other_slot": second_slot,
        "variant": "a12" if real_names is not None else "masked",
    }


def generate_samples(schema: Schema, templates: dict, n: int, *, real_names=None):
    if n < 1:
        raise ValueError("Samples per slot/style must be positive")
    rng = random.Random(13)
    slots = [slot.id for slot in schema.slots] + [f"none.{t}" for t in schema.value_types]
    for slot_id in slots:
        slot = schema.slot(slot_id)
        family = templates[slot_id]
        if any(len(set(family[s])) < 12 for s in ("query", "doc")):
            raise ValueError(f"{slot_id} needs twelve distinct templates per style")
        for style in ("query", "doc"):
            for i in range(n):
                doc_type = slot.doc_type or rng.choice(sorted(schema.doc_types))
                others = [s for s in schema.slots if s.doc_type == doc_type and s.relation != slot.relation]
                second = rng.choice(others).id if others and rng.random() < 0.3 else None
                index = rng.randrange(len(family[style]))
                sample_id = f"{slot_id}:{style}:{i}"
                token_rng = random.Random(int.from_bytes(hashlib.sha256(f"13:{sample_id}".encode()).digest()[:8]))
                yield fill_sample(schema, family, slot_id, style, index, token_rng, sample_id,
                                  real_names=real_names, second_slot=second, doc_type=doc_type)
    pair = 0
    for a in slots:
        for b in templates[a].get("collides_with", []):
            if a >= b:
                continue
            sa, sb = schema.slot(a), schema.slot(b)
            if (sa.doc_type, sa.value_type) != (sb.doc_type, sb.value_type):
                raise ValueError("Minimal pairs must keep document and target types fixed")
            for style in ("query", "doc"):
                base = fill_sample(schema, templates[a], a, style, 0, rng, f"hard:{pair}:a", real_names=real_names)
                alt = fill_sample(schema, templates[b], b, style, 0, rng, f"hard:{pair}:b",
                                  real_names=real_names, bindings=base["bindings"])
                base["hard_pair"], alt["hard_pair"] = str(pair), str(pair)
                yield base
                yield alt
                pair += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
    parser.add_argument("--templates", type=Path, default=Path("templates/train/prime_slots.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-slot", type=int, default=2000)
    parser.add_argument("--a12-docs", type=Path, help="Only A12: explicitly load visible real target names")
    args = parser.parse_args()
    schema = Schema.load(args.schema)
    templates = yaml.safe_load(args.templates.read_text(encoding="utf-8"))
    names = None
    if args.a12_docs is not None:
        docs = pd.read_parquet(args.a12_docs, columns=["doc_type", "name"])
        names = {t: list(g.name) for t, g in docs.groupby("doc_type")}
    config = {"schema_sha256": sha256(args.schema), "templates_sha256": sha256(args.templates),
              "schema_version": schema.version, "samples_per_slot": args.samples_per_slot, "seed": 13,
              "variant": "a12" if names is not None else "masked",
              "a12_docs_sha256": sha256(args.a12_docs) if args.a12_docs else None, "human_reviews": "pending"}
    with logged_run(Path("runs"), "m5-fill", config):
        with new_directory(args.out) as stage:
            count = 0
            with (stage / "filled.jsonl").open("x", encoding="utf-8", newline="\n") as dest:
                for row in generate_samples(schema, templates, args.samples_per_slot, real_names=names):
                    dest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            write_json(stage / "manifest.json", {**config, "samples": count})
    print(f"{count} filled samples; main-model generation never loads KB names")


if __name__ == "__main__":
    main()

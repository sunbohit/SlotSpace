"""Build a test-only KB-grounded role-swap dataset, without any model or LLM."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from .artifacts import logged_run, new_directory, sha256, write_json
from .lexicon_build import CorpusLexicon
from .stats import mde_points
from .text import normalize

DRUG_SLOTS = ("drug.indication.disease", "drug.contraindication.disease", "drug.off_label_use.disease")
SIDE = "drug.side_effect.effect_phenotype"


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)


def eligible(count: int, lo: int = 3, hi: int = 100) -> bool:
    return lo <= count <= hi


def load_postings(path: Path) -> dict[str, tuple[int, ...]]:
    postings = defaultdict(set)
    slots = pa.array([*DRUG_SLOTS, SIDE])
    for batch in pq.ParquetFile(path).iter_batches(batch_size=131072):
        kinds = pc.list_element(pc.split_pattern(batch["key"], pattern="|", max_splits=1), 0)
        selected = batch.filter(pc.is_in(kinds, value_set=slots))
        for row in selected.to_pylist():
            postings[row["key"]].add(row["doc_id"])
    return {key: tuple(sorted(ids)) for key, ids in sorted(postings.items())}


def role_swap_rows(docs: pd.DataFrame, postings: dict[str, tuple[int, ...]], templates: dict,
                   lo: int = 3) -> tuple[list[dict], list[dict]]:
    names = {f"n{i}": name for i, name in docs[["doc_id", "name"]].itertuples(index=False)}
    by_value = defaultdict(set)
    for key, ids in postings.items():
        slot, value = key.split("|", 1)
        if eligible(len(ids), lo):
            by_value[value].add(slot)
    selected = {(v, slot) for v, slots in by_value.items()
                if len(slots.intersection(DRUG_SLOTS)) >= 2 for slot in slots.intersection(DRUG_SLOTS)}
    disease, phenotype = defaultdict(set), defaultdict(set)
    for v, slots in by_value.items():
        name = normalize(names[v])
        if DRUG_SLOTS[0] in slots:
            disease[name].add(v)
        if SIDE in slots:
            phenotype[name].add(v)
    cross_names = disease.keys() & phenotype.keys()
    for name in sorted(cross_names):
        selected.update((v, DRUG_SLOTS[0]) for v in disease[name])
        selected.update((v, SIDE) for v in phenotype[name])
    rows = []
    groups = defaultdict(list)
    for value, slot in sorted(selected):
        name = normalize(names[value])
        group = f"name:{name}" if name in cross_names else f"value:{value}"
        split_key = name if name in cross_names else value
        template_index = h(f"{value}|{slot}") % len(templates[slot])
        row = {"qid": f"rs-{h(f'{value}|{slot}'):040x}", "query": templates[slot][template_index].replace("{V}", names[value]),
               "answers": list(postings[f"{slot}|{value}"]), "value_id": value, "slot": slot,
               "template_id": f"{slot}:{template_index}", "split": "dev" if h(split_key) % 2 == 0 else "test"}
        groups[group].append(len(rows))
        rows.append(row)
    for indices in groups.values():
        for index in indices:
            row = rows[index]
            partners = [rows[j] for j in indices if rows[j]["slot"] != row["slot"]]
            row["partners"] = sorted(q["qid"] for q in partners)
            row["partner_answers"] = sorted({a for q in partners for a in q["answers"]})
    result = []
    for split in ("dev", "test"):
        result.append([{k: v for k, v in row.items() if k != "split"} for row in rows if row["split"] == split])
    return tuple(result)


def build_role_swap(docs: Path, native: Path, lexicon: Path, templates: Path, out: Path, *,
                    runs: Path = Path("runs")) -> Path:
    lex = CorpusLexicon.load(lexicon)
    meta = json.loads((lexicon / "manifest.json").read_text(encoding="utf-8"))
    if lex.mode != "native" or meta["native_sha256"] != sha256(native) or meta["docs_sha256"] != sha256(docs):
        raise ValueError("Role-swap requires the matching complete native-only lexicon")
    if (runs / "m4_role_swap_stats.json").exists():
        raise FileExistsError("Role-swap statistics exist; select a fresh --runs root")
    families = yaml.safe_load(templates.read_text(encoding="utf-8"))
    if set(families) != {*DRUG_SLOTS, SIDE} or any(not v or any("{V}" not in t for t in v) for v in families.values()):
        raise ValueError("Invalid test-only role-swap template families")
    config = {"docs_sha256": sha256(docs), "native_sha256": sha256(native),
              "templates_sha256": sha256(templates), "native_index_version": lex.version,
              "seed": 13, "human_reviews": {"H1": "pending", "H3": "pending"},
              "split_policy": "cross-type name groups take precedence over individual value hashes"}
    with logged_run(runs, "m4-role-swap", config) as (run, metadata):
        frame = pd.read_parquet(docs, columns=["doc_id", "name", "doc_type"])
        postings = load_postings(native)
        for key, ids in postings.items():
            slot, value = key.split("|", 1)
            if lex.hard_support(value, slot) != len(ids):
                raise ValueError(f"Native posting length differs from lexicon: {key}")
        lo = 3
        dev, test = role_swap_rows(frame, postings, families, lo)
        if len(test) < 600:
            lo = 2
            dev, test = role_swap_rows(frame, postings, families, lo)
        if not dev or not test:
            raise ValueError("Role-swap construction produced an empty split")
        with new_directory(out) as stage:
            for split, rows in (("dev", dev), ("test", test)):
                (stage / f"role_swap_{split}.jsonl").write_text(
                    "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows), encoding="utf-8")
            write_json(stage / "manifest.json", {**config, "min_support": lo})
        rng = np.random.default_rng(13)
        review = [{**dev[int(i)], "natural": None} for i in rng.choice(len(dev), min(50, len(dev)), replace=False)]
        (run / "role_swap_review.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in review),
                                                    encoding="utf-8")
        stats = {"dev": len(dev), "test": len(test), "min_support": lo,
                 "fallback_used": lo == 2, "test_size_target_met": len(test) >= 1000,
                 "mde_points": mde_points(len(test), 0.2), "mde_discordant_assumption": 0.2,
                 "per_slot": {s: {k: sum(q["slot"] == k for q in rows) for k in (*DRUG_SLOTS, SIDE)}
                              for s, rows in (("dev", dev), ("test", test))}, "review_run": str(run)}
        write_json(runs / "m4_role_swap_stats.json", stats)
        metadata["counts"] = stats
        print(json.dumps(stats), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("data/prime/docs.parquet"))
    parser.add_argument("--native", type=Path, default=Path("data/prime/facts_prime-0.1.0/native_keys.parquet"))
    parser.add_argument("--lexicon", type=Path, default=Path("data/prime/lexicon_native"))
    parser.add_argument("--templates", type=Path, default=Path("templates/test/role_swap.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/prime/role_swap_prime-0.1.0"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    build_role_swap(args.docs, args.native, args.lexicon, args.templates, args.out, runs=args.runs)


if __name__ == "__main__":
    main()

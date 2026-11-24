"""Reject malformed paraphrases, mask surrogate values, and balance classes."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
from pathlib import Path
import random
import re

from ..artifacts import logged_run, new_directory, read_jsonl, sha256, write_json
from ..schema import Schema
from ..text import normalize

LOG = logging.getLogger(__name__)


def _occurrences(text: str, token: str):
    return list(re.finditer(r"(?<!\w)" + re.escape(token) + r"(?!\w)", text))


def validate_text(text: str, sample: dict, schema: Schema) -> str | None:
    """Return a rejection reason, or None for a valid candidate."""
    if not 4 <= len(text.split()) <= 60:
        return "word_count"
    occurrences = {token: _occurrences(text, token) for token in sample["tokens"]}
    if any(len(locations) != 1 for locations in occurrences.values()):
        return "missing_or_repeated_token"
    targets = [token for token, role in sample["tokens"].items() if role == "target"]
    if len(targets) != 1:
        raise ValueError("A generated sample must have exactly one target token")
    target = occurrences[targets[0]][0]
    center = (target.start() + target.end()) / 2
    own = schema.slot(sample["slot"])
    own_cues = set(own.query_cues + own.doc_cues)
    own_hits = [m for cue in own_cues
                for m in re.finditer(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", text, re.IGNORECASE)]
    other_slot = schema.slot(sample["other_slot"]) if sample.get("other_slot") else None
    other_tokens = [token for token, role in sample["tokens"].items() if role.startswith("other:")]
    for slot in schema.slots:
        if slot.id == own.id or slot.value_type != own.value_type:
            continue
        # Typed slots can share a cue family; shared phrases are not conflicting evidence.
        for cue in set(slot.query_cues + slot.doc_cues) - own_cues:
            for hit in re.finditer(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", text, re.IGNORECASE):
                if any(hit.start() < own_hit.end() and own_hit.start() < hit.end() for own_hit in own_hits):
                    continue
                cue_center = (hit.start() + hit.end()) / 2
                boundaries = [0, *[m.end() for m in re.finditer(r"[;.!?]", text)], len(text)]
                left = max(pos for pos in boundaries if pos <= hit.start())
                right = min(pos for pos in boundaries if pos >= hit.end())
                local_others = [(occurrences[t][0].start() + occurrences[t][0].end()) / 2
                                for t in other_tokens
                                if left <= occurrences[t][0].start() < occurrences[t][0].end() <= right]
                allowed = (other_slot is not None and cue in set(other_slot.query_cues + other_slot.doc_cues)
                           and any(not left <= center <= right or abs(cue_center - other_center) < abs(cue_center - center)
                                   for other_center in local_others))
                if not allowed:
                    return "foreign_target_cue"
    return None


def mask_sample(text: str, sample: dict, *, untyped: bool) -> str:
    replacements = {}
    for token, role in sample["tokens"].items():
        if role == "target":
            replacements[token] = (token if sample["variant"] == "a12" else
                                   "[TARGET]" if untyped else f"[TARGET:{sample['value_type']}]")
        elif role == "self":
            replacements[token] = f"[SELF:{sample['doc_type']}]"
        elif role.startswith("other:"):
            replacements[token] = f"[{role.split(':', 1)[1]}]"
        else:
            raise ValueError(f"Unknown surrogate role: {role}")
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(t) for t in sorted(replacements, key=len, reverse=True)) + r")(?!\w)"
    return re.sub(pattern, lambda m: replacements[m.group()], text)


def validate_samples(samples, schema: Schema, n: int):
    if n < 1:
        raise ValueError("Balanced samples per class/style must be positive")
    groups, rejects, variants, seen_text = defaultdict(list), Counter(), set(), set()
    rng = random.Random(13)
    source_count = paraphrases = 0
    for sample in samples:
        source_count += 1
        if sample["variant"] not in {"masked", "a12"}:
            raise ValueError("Unknown training-data variant")
        variants.add(sample["variant"])
        if len(variants) > 1:
            raise ValueError("Visible-target A12 and masked main training sources cannot be mixed")
        for i, text in enumerate([sample["text"], *sample.get("rewrites", [])]):
            paraphrases += i > 0
            reason = validate_text(text, sample, schema)
            normalized = normalize(text)
            if reason is None and normalized in seen_text:
                reason = "duplicate"
            if reason:
                rejects[reason] += 1
                continue
            seen_text.add(normalized)
            untyped = rng.random() < 0.2
            if "hard_pair" in sample:
                untyped = int.from_bytes(hashlib.sha256(f"13:{sample['hard_pair']}".encode()).digest()[:8]) % 5 == 0
            row = {key: sample[key] for key in ("slot", "value_type", "style", "doc_type", "template_id", "variant")}
            row.update(sample_id=f"{sample['sample_id']}:v{i}", context=mask_sample(text, sample, untyped=untyped),
                       untyped=untyped, hard_pair=sample.get("hard_pair"))
            groups[(sample["slot"], sample["style"])].append(row)
    expected = [(s.id, style) for s in schema.slots for style in ("query", "doc")]
    expected += [(f"none.{t}", style) for t in schema.value_types for style in ("query", "doc")]
    result, resampled = [], {}
    for key in expected:
        rows = groups[key]
        if not rows:
            raise ValueError(f"No valid samples for {key}; inspect cue conflicts/templates")
        hard = [r for r in rows if r["hard_pair"] is not None]
        ordinary = [r for r in rows if r["hard_pair"] is None]
        kept = rng.sample(hard, n) if len(hard) >= n else list(hard)
        need = n - len(kept)
        if need:
            available = ordinary or rows
            if len(available) >= need:
                kept += rng.sample(available, need)
            else:
                LOG.warning("Upsampling %s/%s: %d available, %d requested", *key, len(rows), n)
                kept += available + rng.choices(rows, k=need - len(available))
                resampled["/".join(key)] = {"available": len(rows), "requested": n}
        result.extend({**row, "sample_id": f"{row['sample_id']}:b{i}"} for i, row in enumerate(kept))
    stats = {"input_samples": source_count, "paraphrase_candidates": paraphrases, "output_samples": len(result),
             "rejected": dict(rejects), "upsampled": resampled, "variant": next(iter(variants)),
             "samples_per_slot_per_style": n, "seed": 13}
    return result, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-slot", type=int, default=2000)
    args = parser.parse_args()
    schema = Schema.load(args.schema)
    config = {"input_sha256": sha256(args.input), "schema_sha256": sha256(args.schema),
              "schema_version": schema.version, "samples_per_slot": args.samples_per_slot, "seed": 13,
              "human_reviews": "pending"}
    with logged_run(Path("runs"), "m5-validate", config):
        rows, stats = validate_samples(read_jsonl(args.input), schema, args.samples_per_slot)
        with new_directory(args.out) as stage:
            (stage / "validated.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                   encoding="utf-8")
            write_json(stage / "manifest.json", {**config, **stats})
    print(json.dumps({k: v for k, v in stats.items() if k != "upsampled"}), flush=True)


if __name__ == "__main__":
    main()

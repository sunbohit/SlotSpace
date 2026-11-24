"""Document JSON, including raw signed packed bytes (not quantize_embeddings)."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..artifacts import new_directory, sha256, write_json
from ..reference import TAU_MIN, binarize
from ..schema import Schema
from .application import EXPONENTS, profile_name, write_application
from .codec import document_number, engine_key, unit_vector


def vespa_doc(doc_id, doc_type, text, topic256, native_keys, text_facts, slot_index) -> dict:
    """text_facts contains (safe label, {readable key: pi}, floating unit context)."""
    number = document_number(doc_id)
    if not isinstance(doc_type, str) or not doc_type or not isinstance(text, str):
        raise ValueError("Document type and text must be strings")
    topic = unit_vector(topic256, 256, "topic")
    nk = sorted({engine_key(k, slot_index) for k in native_keys})
    cells, contexts = [], {}
    vkeys = dict.fromkeys(nk, 1)
    for label, keys, ctx in text_facts:
        if not isinstance(label, str) or not re.fullmatch("[a-z0-9_]+", label) or label in contexts:
            raise ValueError("Text fact labels must be distinct [a-z0-9_] strings")
        contexts[label] = binarize(unit_vector(ctx, 128, "fact context")).tolist()
        for key, probability in sorted(keys.items()):
            ek = engine_key(key, slot_index)
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Role probabilities must be finite and in [0, 1]")
            if probability >= TAU_MIN:
                cells.append({"address": {"fact": label, "key": ek}, "value": float(probability)})
                vkeys[ek] = 1
    return {"put": f"id:slotspace:doc::{number}", "fields": {
        "doc_id": number, "doc_type": doc_type, "text": text,
        "topic": {"values": topic.tolist()}, "vkeys": vkeys,
        "nk": {"cells": [{"address": {"key": k}, "value": 1.0} for k in nk]},
        "tk": {"cells": cells}, "tc": {"blocks": contexts},
    }}


def export_feed(docs: Path, schema, index, topics: np.ndarray, out: Path, schema_path: Path) -> Path:
    if topics.shape != (len(index.doc_ids), 256) or index.store.dim != 128:
        raise ValueError("Engine feeding requires aligned 256-d topics and 128-d fact contexts")
    pinned = Schema.load(schema_path)
    if schema.version != pinned.version or schema.slots != pinned.slots:
        raise ValueError("Feed schema differs from its recorded source file")
    slots = {slot.id: i for i, slot in enumerate(schema.slots)}
    with new_directory(out) as stage:
        app_hashes = write_application(stage / "app")
        pos = 0
        with (stage / "feed.jsonl").open("x", encoding="utf-8", newline="\n") as dest:
            for batch in pq.ParquetFile(docs).iter_batches(batch_size=256, columns=["doc_id", "doc_type", "text"]):
                for row in batch.to_pylist():
                    number = document_number(row["doc_id"])
                    if pos >= len(index.doc_ids) or index.doc_ids[pos] != number:
                        raise ValueError("Feed rows do not match the indexed corpus/topic order")
                    key = str(number)
                    facts = [(f"t{i}", f.keys, f.ctx) for i, f in enumerate(index.texts.get(key, []))]
                    op = vespa_doc(number, row["doc_type"], row["text"], topics[pos],
                                   index.store.native.get(key, ()), facts, slots)
                    dest.write(json.dumps(op, ensure_ascii=False, allow_nan=False) + "\n")
                    pos += 1
        if pos != len(index.doc_ids):
            raise ValueError("Feed does not cover every document in the index")
        write_json(stage / "manifest.json", {
            "format": 1, "schema_version": schema.version, "schema_sha256": sha256(schema_path),
            "slot_index": slots, "index_version": index.version, "index_fingerprint": index.fingerprint,
            "docs_sha256": sha256(docs), "documents": pos, "feed_sha256": sha256(stage / "feed.jsonl"),
            "topic_sha256": hashlib.sha256(np.asarray(topics, dtype=np.float32).tobytes()).hexdigest(),
            "application": app_hashes, "context_precision": "binary", "topic_precision": "bfloat16",
            "profiles": [profile_name(a, b) for a, b in EXPONENTS] + ["hybrid"],
        })
    return out


def verify_bundle(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["format"] != 1 or manifest["context_precision"] != "binary"
            or manifest["topic_precision"] != "bfloat16" or manifest["documents"] < 1
            or set(manifest["application"]) != {"services.xml", "hosts.xml", "schemas/doc.sd"}):
        raise ValueError("Invalid Vespa feed/application manifest")
    for name, digest in manifest["application"].items():
        if sha256(path / "app" / Path(name)) != digest:
            raise ValueError(f"Vespa application changed: {name}")
    if sha256(path / "feed.jsonl") != manifest["feed_sha256"]:
        raise ValueError("Vespa feed checksum mismatch")
    return manifest

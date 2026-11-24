"""Compare actual engine match features to the unchanged binary numpy reference."""

import numpy as np

from ..planner import plan_readings
from ..reference import Doc, Fact, ScoreConfig, binarize, binding_score


def binary_document(index, doc_id) -> Doc:
    key = str(doc_id)
    position = np.searchsorted(index.doc_ids, doc_id)
    if position == len(index.doc_ids) or index.doc_ids[position] != doc_id:
        raise ValueError(f"Engine returned a document outside the reference corpus: {doc_id}")
    return Doc(key, [Fact(f"{key}.n.{k}", {k: 1.0}, None, True)
                     for k in sorted(index.store.native.get(key, ()))]
               + [Fact(f.fact_id, f.keys, binarize(f.ctx)) for f in index.texts.get(key, [])])


def compare_binding(hits: list[dict], plan: dict, documents, a=1.0, b=1.0, tolerance=1e-3) -> dict:
    if not hits or not 0 < tolerance <= 1e-3:
        raise ValueError("Parity requires returned candidates and an absolute tolerance at most 1e-3")
    readings = plan_readings(plan)
    cfg = ScoreConfig(a=a, b=b, precision="binary")
    ids = [hit["doc_id"] for hit in hits]
    if len(ids) != len(set(ids)):
        raise ValueError("Parity candidates must have distinct IDs")
    reference = np.array([binding_score(documents(i), readings, cfg) for i in ids])
    engine = np.array([h["bind"] for h in hits])
    if not np.isfinite(engine).all() or not np.isfinite(reference).all():
        raise ValueError("Parity scores must be finite")
    error = float(np.max(np.abs(reference - engine)))
    # Float32 rounding may collapse an actual tie; it must not invert distinct scores.
    tie = 1e-6
    inversions = int(np.sum(((reference[:, None] - reference) > tie) & ((engine[:, None] - engine) < -tie)))
    if error > tolerance or inversions:
        raise ValueError(f"Engine binding parity failed: max_abs={error:.9g}, order inversions={inversions}")
    return {"candidates": len(ids), "max_abs_difference": error, "order_inversions": inversions,
            "absolute_tolerance": tolerance, "tie_tolerance": tie}

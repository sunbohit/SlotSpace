"""Shared candidate selection and fusion for online search and exact feature replay."""

import numpy as np

from .reference import ia_select
from .retrieve import minmax, topk


def select_candidates(lexical, dense, postings: dict[str, np.ndarray], keys: set[str], budgets: dict):
    lexical, dense = np.asarray(lexical), np.asarray(dense)
    if lexical.ndim != 1 or lexical.shape != dense.shape or not len(dense):
        raise ValueError("Lexical/dense candidate scores must be aligned nonempty vectors")
    if any(budgets[k] < 1 for k in ("K_L", "K_D", "K_V")):
        raise ValueError("Candidate budgets must be positive")
    lpos, dpos = topk(lexical, budgets["K_L"]), topk(dense, budgets["K_D"])
    rows = [postings[k] for k in sorted(keys)]
    value = np.unique(np.concatenate(rows)) if rows else np.empty(0, np.int64)
    if len(value) and (value.min() < 0 or value.max() >= len(dense)):
        raise ValueError("Value postings reference unknown score rows")
    if len(value) > budgets["K_V"]:
        value = value[topk(dense[value], budgets["K_V"])]
    return np.union1d(np.union1d(lpos, dpos), value), sum(map(len, rows))


def rank_matches(doc_ids, lexical, dense, matrix, readings, plan, weights, final_k=100, hard_and=False):
    lexical, dense, matrix = np.asarray(lexical), np.asarray(dense), np.asarray(matrix)
    if (matrix.shape != (len(doc_ids), len(readings)) or lexical.shape != (len(doc_ids),)
            or dense.shape != lexical.shape or not np.isfinite(matrix).all() or final_k < 1):
        raise ValueError("Ranking features are not aligned")
    if len(weights) != 3 or any(not np.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Fusion weights must be three finite nonnegative numbers")
    binding = np.ascontiguousarray(matrix, dtype=np.float64) @ np.array([r.weight for r in readings])
    segments = sum(bool(s["readings"]) for s in plan["segments"])
    lb, ld, ll = weights
    fused = lb * binding / max(segments, 1) + ld * minmax(dense) + ll * minmax(lexical)
    allowed = np.ones(len(doc_ids), dtype=bool)
    if hard_and:
        for i, _ in enumerate(plan["segments"]):
            columns = [j for j, r in enumerate(readings) if r.segment == i]
            allowed &= np.any(matrix[:, columns] > 0, axis=1) if columns else False
    positions = np.flatnonzero(allowed)
    order = positions[np.lexsort((np.asarray(doc_ids)[positions], -fused[positions]))[:final_k]]
    if plan["weight_objective"] == "ia" and readings:
        if np.any(matrix < -1e-5) or np.any(matrix > 1 + 1e-5):
            raise ValueError("IA received non-probabilistic matches")
        probabilities = {f"s{i}:{r['key']}": r["p"] for i, s in enumerate(plan["segments"]) for r in s["readings"]}
        bounded = np.clip(matrix, 0, 1)
        coverage = {r.rid: {str(doc_ids[i]): float(bounded[i, j]) for i in order}
                    for j, r in enumerate(readings)}
        ranked = ia_select([str(doc_ids[i]) for i in order], probabilities, coverage, len(order))
        lookup = {str(doc_ids[i]): i for i in order}
        order = np.array([lookup[d] for d in ranked], dtype=np.int64)
    values = (np.arange(len(order), 0, -1, dtype=float)
              if plan["weight_objective"] == "ia" and readings else fused[order])
    return order, values, fused


def fact_work(store, doc_ids, readings, mode):
    result = np.zeros((len(doc_ids), len(readings)), dtype=np.int64)
    for i, doc in enumerate(doc_ids):
        for j, reading in enumerate(readings):
            if mode == "no_role":
                rows = [store.value_rows.get(doc, {}).get(k.split("|", 1)[1], np.empty(0, int))
                        for k in reading.keys]
                result[i, j] = len(np.unique(np.concatenate(rows))) if rows else 0
            else:
                present = [store.key_index.get(doc, {})[k][0] for k in reading.keys if k in store.key_index.get(doc, {})]
                result[i, j] = (len(store.text_rows.get(doc, ())) if present else 0
                                ) if mode == "unbound" else sum(map(len, present))
    return result

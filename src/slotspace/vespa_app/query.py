"""Compile a planner reading set into one engine request and fuse its returned pool."""

import numpy as np

from ..retrieve import minmax
from .application import profile_name
from .codec import engine_key, unit_vector


def vespa_body(plan, query_text, qvec256, slot_index, a=1.0, b=1.0, *, hits=400, match_all=False) -> dict:
    if type(hits) is not int or not 1 <= hits <= 500:
        raise ValueError("Engine candidate hits must lie in [1, 500] (the reranking budget)")
    name = profile_name(a, b)
    if plan.get("weight_objective", "p") not in {"p", "p_over_H"}:
        raise ValueError("Vespa bound/max fusion does not implement IA or hard-AND")
    if not isinstance(query_text, str):
        raise ValueError("Query text must be a string")
    rk, weights, vectors, posting_weights = [], {}, {}, {}
    for segment in plan["segments"]:
        seen = set()
        g = unit_vector(segment["g"], 128, "segment") if segment["readings"] else None
        for reading in segment["readings"]:
            key = reading["key"]
            ek = engine_key(key, slot_index)
            if key in seen:
                raise ValueError("A segment contains a duplicate reading")
            seen.add(key)
            link = segment["links"][key.split("|")[1]]
            weight = reading["w"]
            if not np.isfinite(link) or not 0 < link <= 1 or not np.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("Reading links/weights are invalid")
            rid = f"r{len(weights)}"
            rk.append(f"{{r:{rid},key:{ek}}}:{float(link) ** a:.17g}")
            weights[rid] = float(weight)
            vectors[rid] = g.tolist()
            posting_weights[ek] = max(posting_weights.get(ek, 0), int(round(1000 * weight)))
    budgets = plan["budgets"]
    if any(type(budgets[k]) is not int or budgets[k] < 1 for k in ("K_D", "K_L")):
        raise ValueError("Engine candidate budgets must be positive integers")
    where = f"({{totalTargetHits:{budgets['K_D']}}}nearestNeighbor(topic, qt)) or userQuery()"
    if rk:
        where += " or dotProduct(vkeys, @vk)"
    body = {
        "yql": "select * from doc where " + ("true" if match_all else where),
        "query": query_text, "type": "weakAnd", "ranking": name if rk else "hybrid", "hits": hits,
        "wand.hits": budgets["K_L"], "timeout": "5s", "ranking.softtimeout.enable": False,
        "presentation.timing": True, "input.query(qt)": unit_vector(qvec256, 256, "query topic").tolist(),
    }
    if rk:
        body.update({"input.query(rk)": "{" + ",".join(rk) + "}", "input.query(w)": weights,
                     "input.query(g)": vectors,
                     "vk": "{" + ",".join(f"{k}:{v}" for k, v in sorted(posting_weights.items())) + "}"})
    return body


def fuse_hits(hits: list[dict], plan: dict, weights=(1.0, 1.0, 0.5), final_k=100) -> dict:
    if (len(weights) != 3 or any(not np.isfinite(w) or w < 0 for w in weights)
            or type(final_k) is not int or not 1 <= final_k <= 100):
        raise ValueError("Invalid engine fusion weights/result length")
    if not hits:
        return {"doc_ids": [], "scores": [], "candidates": []}
    ids = np.array([hit["doc_id"] for hit in hits], dtype=np.int64)
    if len(set(ids)) != len(ids):
        raise ValueError("Engine returned duplicate document IDs")
    bound, dense, lexical = (np.array([hit[k] for hit in hits]) for k in ("bind", "dense", "bm25"))
    if not all(np.isfinite(v).all() for v in (bound, dense, lexical)):
        raise ValueError("Engine ranking features must be finite")
    segments = sum(bool(s["readings"]) for s in plan["segments"])
    lb, ld, ll = weights
    fused = (lb * bound / segments if segments else 0) + ld * minmax(dense) + ll * minmax(lexical)
    order = np.lexsort((ids, -fused))[:final_k]
    return {"doc_ids": ids[order].tolist(), "scores": fused[order].tolist(), "candidates": hits}

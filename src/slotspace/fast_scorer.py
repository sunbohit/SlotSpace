"""Fast binding scorer for the research path (DEVELOPMENT.md M3).

Same math as slotspace_reference.binding_score, but documents are stored in
compact arrays and only gated facts are touched. Parity with the reference is
checked by tests/test_fast_scorer.py.

precision='binary' stores each context as unpack_pm1(binarize(ctx)) / sqrt(D),
which gives exactly the reference's binary similarity.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

from slotspace.reference import TAU_MIN, Doc, Reading, ScoreConfig, binarize, unpack_pm1


class FactStore:
    def __init__(self, dim: int = 128, precision: str = "float") -> None:
        self.dim, self.precision = dim, precision
        self._ctx: List[np.ndarray] = []
        self.native: Dict[str, set] = {}
        self.native_values: Dict[str, set] = {}
        self.text_rows: Dict[str, np.ndarray] = {}                 # doc -> all text-fact rows
        self.key_index: Dict[str, Dict[str, tuple]] = {}           # doc -> key -> (rows, pis)
        self.value_rows: Dict[str, Dict[str, np.ndarray]] = {}     # doc -> value -> rows (no_role mode)
        self.ctx = np.zeros((0, dim), dtype=np.float32)

    def add_docs(self, docs: Iterable[Doc]) -> None:
        start = len(self._ctx)
        for d in docs:
            nat, natv = set(), set()
            per_key, per_val, rows = defaultdict(lambda: ([], [])), defaultdict(list), []
            for f in d.facts:
                if f.native:
                    for k in f.keys:
                        nat.add(k)
                        natv.add(k.split("|", 1)[1])
                    continue
                row = len(self._ctx)
                c = np.asarray(f.ctx, dtype=np.float64)
                if self.precision == "binary":
                    c = unpack_pm1(binarize(c)) / np.sqrt(c.shape[0])
                self._ctx.append(c.astype(np.float32))
                rows.append(row)
                for k, pi in f.keys.items():
                    if pi >= TAU_MIN:
                        per_key[k][0].append(row)
                        per_key[k][1].append(pi)
                        per_val[k.split("|", 1)[1]].append(row)
            self.native[d.doc_id], self.native_values[d.doc_id] = nat, natv
            self.text_rows[d.doc_id] = np.array(rows, dtype=np.int64)
            self.key_index[d.doc_id] = {k: (np.array(r, dtype=np.int64), np.array(p, dtype=np.float64)) for k, (r, p) in per_key.items()}
            self.value_rows[d.doc_id] = {v: np.array(sorted(set(r)), dtype=np.int64) for v, r in per_val.items()}
        if len(self._ctx) > start:
            self.ctx = np.vstack(self._ctx).astype(np.float32)

    def scores(self, doc_ids: Sequence[str], readings: List[Reading], cfg: ScoreConfig = ScoreConfig()) -> np.ndarray:
        if cfg.precision != self.precision:
            raise ValueError(f"store built with precision={self.precision}, config asks for {cfg.precision}")
        G = np.stack([r.g for r in readings]).astype(np.float32) if readings else np.zeros((0, self.dim), np.float32)
        out = np.zeros(len(doc_ids), dtype=np.float64)
        for i, d in enumerate(doc_ids):
            nat, keys = self.native.get(d, set()), self.key_index.get(d, {})
            total = 0.0
            for j, r in enumerate(readings):
                # native part
                if cfg.mode == "no_role":
                    vals = {k.split("|", 1)[1] for k in r.keys}
                    n = 1.0 if vals & self.native_values.get(d, set()) else 0.0
                else:
                    best = max((lp for k, lp in r.keys.items() if k in nat), default=0.0)
                    n = best ** cfg.a if best > 0 else 0.0
                # text part
                t = 0.0
                if cfg.mode == "bound":
                    for k, lp in r.keys.items():
                        if k in keys:
                            rows, pis = keys[k]
                            s = np.maximum(self.ctx[rows] @ G[j], 0.0).astype(np.float64)
                            t = max(t, float(np.max((lp * pis) ** cfg.a * s ** cfg.b)))
                elif cfg.mode in ("unbound", "gate_only"):
                    gmax = max((lp * float(keys[k][1].max()) for k, lp in r.keys.items() if k in keys), default=0.0)
                    if gmax > 0:
                        t = gmax ** cfg.a
                        if cfg.mode == "unbound":
                            rows = self.text_rows[d]
                            t *= float(np.max(np.maximum(self.ctx[rows] @ G[j], 0.0))) ** cfg.b
                elif cfg.mode == "no_role":
                    vr = self.value_rows.get(d, {})
                    rows = [vr[k.split("|", 1)[1]] for k in r.keys if k.split("|", 1)[1] in vr]
                    if rows:
                        rows = np.unique(np.concatenate(rows))
                        t = float(np.max(np.maximum(self.ctx[rows] @ G[j], 0.0))) ** cfg.b
                m = max(n, t) if cfg.combine == "max" else n + t
                total += r.weight * m
            out[i] = total
        return out

"""SlotSpace numpy reference implementation (DESIGN.md sections 7-10).

This module is the single source of truth for the scoring math. Faster code
(batched numpy, the Vespa rank profile) must reproduce its numbers on the
parity tests. Only numpy is required. Readability beats speed here.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

TAU_MIN = 0.05  # smallest role probability stored for a text fact (the gate floor)


# ---------------------------------------------------------------- vectors
def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def binarize(v) -> np.ndarray:
    """float[D] (D multiple of 8) -> int8[D/8].

    bit = 1 if component > 0 else 0 (exact zero -> 0). The first component is
    the most significant bit of the first byte ('big' bit order). Bytes are
    reinterpreted as two's-complement int8 (a view; NOT 'minus 128').
    """
    bits = (np.asarray(v) > 0).astype(np.uint8)
    return np.packbits(bits, bitorder="big").view(np.int8)


def unpack_pm1(b) -> np.ndarray:
    """int8[D/8] -> float[D] with values in {-1.0, +1.0}."""
    bits = np.unpackbits(np.asarray(b, dtype=np.int8).view(np.uint8), bitorder="big")
    return bits.astype(np.float64) * 2.0 - 1.0


def sim(g: np.ndarray, ctx: np.ndarray, precision: str = "float") -> float:
    """precision='float': cosine of unit vectors.
    precision='binary': ctx is packed int8; <g, pm1>/sqrt(D) (float query, binary doc)."""
    if precision == "float":
        return float(np.dot(g, ctx))
    pm = unpack_pm1(ctx)
    return float(np.dot(g, pm) / math.sqrt(pm.shape[0]))


def psi(x: float) -> float:
    """Negative context similarity gives no credit."""
    return x if x > 0.0 else 0.0


# ---------------------------------------------------------------- data model
@dataclass
class Fact:
    fact_id: str
    keys: Dict[str, float]          # "slot|value_id" -> pi in [TAU_MIN, 1]
    ctx: Optional[np.ndarray]       # text facts: unit float (or packed int8); native: may be None
    native: bool = False            # native fact = one KB edge: exactly one key with pi = 1.0


@dataclass
class Doc:
    doc_id: str
    facts: List[Fact]


@dataclass
class Reading:
    rid: str                        # e.g. "s0:drug.contraindication.disease|n123"
    segment: int                    # which query segment (value mention) it belongs to
    keys: Dict[str, float]          # "slot|value_id" -> link probability (usually one key)
    weight: float                   # w_r from the planner
    g: np.ndarray                   # unit float vector of the delexicalised segment context


@dataclass
class ScoreConfig:
    mode: str = "bound"             # bound | unbound | gate_only | no_role
    combine: str = "max"            # max | sum  (how native and text evidence combine)
    a: float = 1.0                  # exponent on the gate (role probability x link prob)
    b: float = 1.0                  # exponent on psi(sim)
    precision: str = "float"        # float | binary (text-fact context vectors)


# ---------------------------------------------------------------- scoring
def fact_gate(f: Fact, r: Reading) -> float:
    """max over the reading's keys of linkprob * pi_f(key); cells below TAU_MIN never open."""
    best = 0.0
    for k, lp in r.keys.items():
        pi = f.keys.get(k, 0.0)
        if pi >= TAU_MIN:
            best = max(best, lp * pi)
    return best


def _values(keys: Iterable[str]) -> set:
    return {k.split("|", 1)[1] for k in keys}


def native_part(doc: Doc, r: Reading, cfg: ScoreConfig) -> float:
    """N_r(d): native facts carry pi = 1 and context factor 1, so N = max link prob of a matching key."""
    if cfg.mode == "no_role":  # value in any slot counts
        vals = _values(r.keys)
        return max((1.0 for f in doc.facts if f.native and _values(f.keys) & vals), default=0.0)
    best = 0.0
    for f in doc.facts:
        if f.native:
            best = max(best, fact_gate(f, r))
    return best ** cfg.a if best > 0 else 0.0


def text_part(doc: Doc, r: Reading, cfg: ScoreConfig) -> float:
    """T_r(d) over text facts, for each ablation mode."""
    texts = [f for f in doc.facts if not f.native]
    if not texts:
        return 0.0
    if cfg.mode == "bound":  # gate and context on the SAME fact
        best = 0.0
        for f in texts:
            gt = fact_gate(f, r)
            if gt > 0.0:
                best = max(best, gt ** cfg.a * psi(sim(r.g, f.ctx, cfg.precision)) ** cfg.b)
        return best
    if cfg.mode == "unbound":  # per-segment Chamfer: maximised separately, then multiplied
        best_gate = max((fact_gate(f, r) for f in texts), default=0.0)
        if best_gate == 0.0:
            return 0.0
        best_sim = max(psi(sim(r.g, f.ctx, cfg.precision)) for f in texts)
        return best_gate ** cfg.a * best_sim ** cfg.b
    if cfg.mode == "gate_only":
        best_gate = max((fact_gate(f, r) for f in texts), default=0.0)
        return best_gate ** cfg.a if best_gate > 0 else 0.0
    if cfg.mode == "no_role":  # value present in the fact, role ignored
        vals = _values(r.keys)
        return max((psi(sim(r.g, f.ctx, cfg.precision)) ** cfg.b for f in texts if _values(f.keys) & vals), default=0.0)
    raise ValueError(f"unknown mode {cfg.mode}")


def reading_score(doc: Doc, r: Reading, cfg: ScoreConfig = ScoreConfig()) -> float:
    """M_r(d) in [0, 1]."""
    n, t = native_part(doc, r, cfg), text_part(doc, r, cfg)
    return max(n, t) if cfg.combine == "max" else n + t


def binding_score(doc: Doc, readings: List[Reading], cfg: ScoreConfig = ScoreConfig()) -> float:
    """B(d) = sum_r w_r * M_r(d)."""
    return float(sum(r.weight * reading_score(doc, r, cfg) for r in readings))


# ---------------------------------------------------------------- ambiguity lexicon
def doc_contributions(doc: Doc) -> Dict[Tuple[str, str], float]:
    """(value_id, slot) -> c_d(v, k) = max over the doc's facts of pi (native facts give 1.0)."""
    out: Dict[Tuple[str, str], float] = {}
    for f in doc.facts:
        for key, pi in f.keys.items():
            if pi < TAU_MIN:
                continue
            slot, value = key.split("|", 1)
            out[(value, slot)] = max(out.get((value, slot), 0.0), pi)
    return out


class Lexicon:
    """soft[(v,k)] = sum_d c_d(v,k);  hard[(v,k)] = number of docs with c_d(v,k) >= TAU_MIN.

    Maintained incrementally per document version; a rebuild from scratch must
    give identical numbers (invariant I4)."""

    def __init__(self) -> None:
        self.soft: Dict[Tuple[str, str], float] = defaultdict(float)
        self.hard: Dict[Tuple[str, str], int] = defaultdict(int)
        self._doc: Dict[str, Dict[Tuple[str, str], float]] = {}
        self.version = 0

    def _apply(self, contrib: Dict[Tuple[str, str], float], sign: int) -> None:
        for vk, c in contrib.items():
            self.soft[vk] += sign * c
            self.hard[vk] += sign
            if self.hard[vk] == 0:
                self.soft.pop(vk, None)
                self.hard.pop(vk, None)

    def upsert(self, doc: Doc) -> None:
        if doc.doc_id in self._doc:
            self._apply(self._doc[doc.doc_id], -1)
        contrib = doc_contributions(doc)
        self._apply(contrib, +1)
        self._doc[doc.doc_id] = contrib
        self.version += 1

    def delete(self, doc_id: str) -> None:
        if doc_id in self._doc:
            self._apply(self._doc.pop(doc_id), -1)
            self.version += 1

    def support(self, value: str) -> Dict[str, float]:
        return {k: s for (v, k), s in self.soft.items() if v == value}

    def hard_support(self, value: str, slot: str) -> int:
        return self.hard.get((value, slot), 0)

    def snapshot(self) -> Dict[Tuple[str, str], Tuple[float, int]]:
        return {vk: (round(self.soft[vk], 9), self.hard[vk]) for vk in self.hard}


def pacc_binary(q_mean: float, tpr: float, fpr: float, min_gap: float = 0.05) -> float:
    """PACC for one role: (mean posterior - FPR_pa) / (TPR_pa - FPR_pa), clipped to [0, 1].

    If TPR - FPR < min_gap the correction is unstable: return the raw mean (caller logs a warning)."""
    if tpr - fpr < min_gap:
        return float(np.clip(q_mean, 0.0, 1.0))
    return float(np.clip((q_mean - fpr) / (tpr - fpr), 0.0, 1.0))


# ---------------------------------------------------------------- planner pieces
def reading_posterior(
    C: Dict[str, Dict[str, float]],       # value_type -> {slot: P(slot | context, type)}
    links: Dict[str, Tuple[str, float]],  # value_id -> (value_type, link probability)
    lex: Lexicon,
    beta: float = 0.5,
    eps: float = 1.0,
) -> Dict[str, float]:
    """p(slot, value | q, s) ∝ link(v) * C(slot | ctx, type(v)) * E(slot | v)^beta.

    E(slot | v) = (L_v[slot] + eps) / (sum over compatible slots of L_v + eps * n_slots).
    Readings with hard support 0 are dropped (lossless, see DESIGN.md P2)."""
    scores: Dict[str, float] = {}
    for v, (vtype, lp) in links.items():
        slots = C.get(vtype, {})
        sup = lex.support(v)
        tot = sum(sup.get(k, 0.0) for k in slots)
        for k, c in slots.items():
            if lex.hard_support(v, k) == 0:
                continue
            e = (sup.get(k, 0.0) + eps) / (tot + eps * len(slots))
            scores[f"{k}|{v}"] = lp * c * e ** beta
    z = sum(scores.values())
    return {key: s / z for key, s in scores.items()} if z > 0 else {}


def conformal_threshold(p_true, alpha: float) -> float:
    """Split conformal. Keep readings with p >= t  =>  P(true reading kept) >= 1 - alpha."""
    p_true = np.asarray(p_true, dtype=np.float64)
    n = p_true.shape[0]
    level = math.ceil((n + 1) * (1 - alpha)) / n
    if level > 1.0:
        return 0.0  # too few calibration points: keep everything
    qhat = float(np.quantile(1.0 - p_true, level, method="higher"))
    return 1.0 - qhat


def keep_readings(post: Dict[str, float], t: float, min_keep: int = 1, r_max: int = 6) -> Dict[str, float]:
    """Keep readings with p >= t (always the top min_keep, at most r_max), renormalized to sum 1."""
    ranked = sorted(post.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = [(k, p) for i, (k, p) in enumerate(ranked) if (p >= t or i < min_keep) and i < r_max]
    z = sum(p for _, p in kept)
    return {k: p / z for k, p in kept} if z > 0 else {}


def ia_select(cands: List[str], P: Dict[str, float], V: Dict[str, Dict[str, float]], k: int) -> List[str]:
    """Greedy IA-Select. P: reading -> probability; V[reading][doc] in [0, 1]."""
    U = dict(P)
    chosen: List[str] = []
    remaining = list(cands)
    while remaining and len(chosen) < k:
        best = max(remaining, key=lambda d: (sum(U[c] * V.get(c, {}).get(d, 0.0) for c in U), -cands.index(d)))
        chosen.append(best)
        remaining.remove(best)
        for c in U:
            U[c] *= 1.0 - V.get(c, {}).get(best, 0.0)
    return chosen

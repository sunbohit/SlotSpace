"""Unit tests for the SlotSpace reference implementation (DEVELOPMENT.md Appendix B).
Run: pytest -q tests/test_reference.py
"""
import math

import numpy as np
import pytest

from slotspace.reference import (
    TAU_MIN, Doc, Fact, Lexicon, Reading, ScoreConfig, binarize, binding_score,
    conformal_threshold, doc_contributions, ia_select, keep_readings, pacc_binary,
    reading_posterior, reading_score, sim, unit, unpack_pm1,
)

IND, CON, OFF = "drug.indication.disease", "drug.contraindication.disease", "drug.off_label_use.disease"
SIDE = "drug.side_effect.effect_phenotype"
HTN, HTNP, AST, KF = "n101", "n202", "n103", "n104"
E1, E2, E3, E4 = np.eye(4)
CTX_TREAT = unit(E1 + 0.2 * E4)
CTX_AVOID = unit(E2 + 0.2 * E4)
CTX_NEUTRAL = unit(0.3 * E1 + 0.3 * E2 + 0.9 * E4)
G = unit(E2 + 0.3 * E4)


def text_fact(fid, keys, ctx):
    return Fact(fid, {k: p for k, p in keys.items() if p >= TAU_MIN}, ctx, native=False)


def native_fact(fid, key):
    return Fact(fid, {key: 1.0}, None, native=True)


def toy_docs():
    return [
        Doc("A", [native_fact("A.n1", f"{IND}|{HTN}"),
                  text_fact("A.t1", {f"{IND}|{HTN}": 0.92, f"{CON}|{HTN}": 0.05, f"{OFF}|{HTN}": 0.03}, CTX_TREAT)]),
        Doc("B", [text_fact("B.t1", {f"{CON}|{HTN}": 0.60, f"{IND}|{HTN}": 0.25, f"{OFF}|{HTN}": 0.15}, CTX_AVOID)]),
        Doc("C", [native_fact("C.n1", f"{IND}|{AST}"), native_fact("C.n2", f"{SIDE}|{HTNP}"),
                  text_fact("C.t1", {f"{CON}|{AST}": 0.90, f"{IND}|{AST}": 0.06}, CTX_AVOID),
                  text_fact("C.t2", {f"{IND}|{HTN}": 0.35, f"{CON}|{HTN}": 0.30, f"{OFF}|{HTN}": 0.35}, CTX_NEUTRAL)]),
        Doc("D", [text_fact("D.t1", {f"{CON}|{KF}": 0.88, f"{IND}|{KF}": 0.07}, CTX_AVOID)]),
    ]


def corpus_lexicon():
    lex = Lexicon()
    lex.soft.update({(HTN, IND): 120.0, (HTN, CON): 40.0, (HTNP, SIDE): 300.0})
    lex.hard.update({(HTN, IND): 131, (HTN, CON): 44, (HTNP, SIDE): 300})
    return lex


def toy_posterior():
    C = {"disease": {IND: 0.15, CON: 0.80, OFF: 0.05}, "effect_phenotype": {SIDE: 1.0}}
    links = {HTN: ("disease", 0.9), HTNP: ("effect_phenotype", 0.1)}
    return reading_posterior(C, links, corpus_lexicon(), beta=0.5, eps=1.0)


def toy_readings():
    return [Reading(f"s0:{k}", 0, {k: 1.0}, p, G) for k, p in toy_posterior().items()]


def test_posterior_values_and_pruning():
    post = toy_posterior()
    assert set(post) == {f"{CON}|{HTN}", f"{IND}|{HTN}", f"{SIDE}|{HTNP}"}  # off-label pruned (H = 0)
    assert post[f"{CON}|{HTN}"] == pytest.approx(0.6254, abs=1e-4)
    assert post[f"{IND}|{HTN}"] == pytest.approx(0.2014, abs=1e-4)
    assert post[f"{SIDE}|{HTNP}"] == pytest.approx(0.1732, abs=1e-4)
    assert sum(post.values()) == pytest.approx(1.0)


def test_toy_scores_and_rankings():
    docs, rs = toy_docs(), toy_readings()
    expected = {  # doc -> (bound, unbound, gate_only, no_role)
        "A": (0.2032, 0.2032, 0.2327, 0.8268),
        "B": (0.4237, 0.4237, 0.4256, 0.8232),
        "C": (0.3148, 0.4302, 0.4313, 0.6269),
        "D": (0.0, 0.0, 0.0, 0.0),
    }
    modes = ["bound", "unbound", "gate_only", "no_role"]
    for d in docs:
        for m, want in zip(modes, expected[d.doc_id]):
            assert binding_score(d, rs, ScoreConfig(mode=m)) == pytest.approx(want, abs=1e-4), (d.doc_id, m)
    order = {m: [d.doc_id for d in sorted(docs, key=lambda d: (-binding_score(d, rs, ScoreConfig(mode=m)), d.doc_id))] for m in modes}
    assert order["bound"] == ["B", "C", "A", "D"]
    assert order["unbound"] == ["C", "B", "A", "D"]
    assert order["gate_only"] == ["C", "B", "A", "D"]
    assert order["no_role"] == ["A", "B", "C", "D"]


def test_binding_property_P1():
    # unbound > 0 although no single fact has both the value and a positive context
    f_val = text_fact("x.t1", {f"{CON}|{HTN}": 0.9}, unit(-E2))    # value, opposite context
    f_ctx = text_fact("x.t2", {f"{CON}|{AST}": 0.9}, CTX_AVOID)    # other value, matching context
    d = Doc("X", [f_val, f_ctx])
    r = Reading("r", 0, {f"{CON}|{HTN}": 1.0}, 1.0, G)
    assert reading_score(d, r, ScoreConfig(mode="bound")) == 0.0
    assert reading_score(d, r, ScoreConfig(mode="unbound")) > 0.8


def test_zero_support_pruning_is_lossless_P2():
    docs = toy_docs()
    lex = Lexicon()
    for d in docs:
        lex.upsert(d)
    unsupported = Reading("u", 0, {f"{OFF}|{AST}": 1.0}, 0.3, G)
    assert lex.hard_support(AST, OFF) == 0
    base = toy_readings()
    for d in docs:
        assert reading_score(d, unsupported) == 0.0
        assert binding_score(d, base + [unsupported]) == pytest.approx(binding_score(d, base))


def test_bounds_and_monotonicity_P4_P5():
    d = toy_docs()[1]
    r = Reading("r", 0, {f"{CON}|{HTN}": 1.0}, 1.0, G)
    s0 = reading_score(d, r)
    assert 0.0 <= s0 <= 1.0
    d.facts[0].keys[f"{CON}|{HTN}"] = 0.9  # stronger role probability
    assert reading_score(d, r) >= s0
    d.facts.append(text_fact("B.t2", {f"{CON}|{HTN}": 0.99}, G))  # adding a fact never lowers the score
    assert reading_score(d, r) >= s0


def test_lexicon_incremental_equals_rebuild_I4():
    docs = toy_docs()
    lex = Lexicon()
    for d in docs:
        lex.upsert(d)
    snap = lex.snapshot()
    assert snap[(HTN, IND)] == (pytest.approx(1.6), 3)
    assert snap[(HTN, CON)] == (pytest.approx(0.95), 3)
    assert snap[(HTN, OFF)] == (pytest.approx(0.5), 2)
    lex.delete("B")
    rebuilt = Lexicon()
    for d in docs:
        if d.doc_id != "B":
            rebuilt.upsert(d)
    assert lex.snapshot() == rebuilt.snapshot()
    assert lex.snapshot()[(HTN, CON)] == (pytest.approx(0.35), 2)
    lex.upsert(docs[1])  # re-adding restores the original numbers
    assert lex.snapshot() == snap


def test_doc_contribution_is_max_over_facts():
    c = doc_contributions(toy_docs()[0])
    assert c[(HTN, IND)] == 1.0  # native beats the 0.92 text fact
    assert (HTN, OFF) not in c   # 0.03 < TAU_MIN is never stored


def test_conformal_threshold():
    pt = [0.91, 0.85, 0.77, 0.95, 0.42, 0.66, 0.88, 0.73, 0.59, 0.97, 0.81, 0.35, 0.69, 0.92, 0.54, 0.79, 0.83, 0.61, 0.90]
    assert conformal_threshold(pt, 0.10) == pytest.approx(0.35)
    assert conformal_threshold(pt, 0.20) == pytest.approx(0.54)
    assert conformal_threshold(pt, 0.02) == 0.0  # level > 1: keep everything
    kept = keep_readings(toy_posterior(), 0.35)
    assert list(kept) == [f"{CON}|{HTN}"] and kept[f"{CON}|{HTN}"] == pytest.approx(1.0)
    assert len(keep_readings(toy_posterior(), 0.15)) == 3
    assert len(keep_readings(toy_posterior(), 0.0, r_max=2)) == 2


def test_conformal_coverage_simulation():
    rng = np.random.default_rng(13)
    misses = []
    for _ in range(200):
        cal = rng.beta(5, 2, size=500)
        test = rng.beta(5, 2, size=500)
        t = conformal_threshold(cal, 0.1)
        misses.append(np.mean(test < t))
    assert np.mean(misses) <= 0.10 + 0.01


def test_pacc():
    assert pacc_binary(0.40, 0.80, 0.15) == pytest.approx(0.3846, abs=1e-4)
    assert pacc_binary(0.05, 0.80, 0.15) == 0.0
    assert pacc_binary(0.95, 0.80, 0.15) == 1.0
    assert pacc_binary(0.40, 0.42, 0.40) == pytest.approx(0.40)  # gap < 0.05: raw mean


def test_binarize_and_unpack():
    v = np.array([0.3, -0.1, 0.0, 0.7, -0.2, 0.5, -0.9, 0.1])
    b = binarize(v)
    assert b.dtype == np.int8 and b.tolist() == [-107]          # bits 10010101 = 0x95
    assert b.view(np.uint8).tolist() == [149]
    assert unpack_pm1(b).tolist() == [1, -1, -1, 1, -1, 1, -1, 1]
    rng = np.random.default_rng(0)
    x = unit(rng.normal(size=128))
    y = unit(x + 0.8 * unit(rng.normal(size=128)))
    assert float(x @ y) == pytest.approx(0.7815, abs=1e-4)
    assert sim(x, binarize(y), "binary") == pytest.approx(0.6473, abs=1e-4)


def test_ia_select():
    P = {"ind": 0.3, "con": 0.7}
    V = {"ind": {"A": 0.9, "C": 0.3}, "con": {"B": 0.8, "D": 0.6, "C": 0.2}}
    assert ia_select(["A", "B", "C", "D"], P, V, 3) == ["B", "A", "D"]

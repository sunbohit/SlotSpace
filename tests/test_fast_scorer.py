"""Parity: FactStore.scores == reference binding_score on random documents."""
import numpy as np
from slotspace.reference import Doc, Fact, Reading, ScoreConfig, TAU_MIN, binarize, binding_score, unit
from slotspace.fast_scorer import FactStore


def random_corpus(rng, n_docs=60, n_vals=8, slots=("a", "b", "c"), dim=16):
    """Random documents with native and text facts; text-fact contexts are unit float vectors."""
    docs = []
    for i in range(n_docs):
        facts = []
        for j in range(int(rng.integers(0, 4))):
            v = f"n{int(rng.integers(n_vals))}"
            facts.append(Fact(f"d{i}.n{j}", {f"{rng.choice(slots)}|{v}": 1.0}, None, True))
        for j in range(int(rng.integers(0, 6))):
            keys = {}
            for _ in range(int(rng.integers(1, 3))):
                v = f"n{int(rng.integers(n_vals))}"
                p = rng.dirichlet(np.ones(len(slots)))
                for s, pi in zip(slots, p):
                    if pi >= TAU_MIN:
                        keys[f"{s}|{v}"] = max(keys.get(f"{s}|{v}", 0.0), float(pi))
            facts.append(Fact(f"d{i}.t{j}", keys, unit(rng.normal(size=dim)), False))
        docs.append(Doc(f"d{i}", facts))
    return docs


def test_fast_equals_reference():
    rng = np.random.default_rng(5)
    docs = random_corpus(rng)
    store = FactStore(dim=16)
    store.add_docs(docs)
    for trial in range(30):
        readings = []
        for s in range(int(rng.integers(1, 3))):
            g = unit(rng.normal(size=16))
            v = f"n{int(rng.integers(8))}"
            for slot in ("a", "b", "c"):
                readings.append(Reading(f"s{s}:{slot}", s, {f"{slot}|{v}": float(rng.uniform(0.3, 1.0))}, float(rng.uniform(0.05, 1.0)), g))
        for cfg in [ScoreConfig(), ScoreConfig(mode="unbound"), ScoreConfig(mode="gate_only"), ScoreConfig(mode="no_role"),
                    ScoreConfig(a=0.5, b=2.0), ScoreConfig(combine="sum")]:
            fast = store.scores([d.doc_id for d in docs], readings, cfg)
            ref = np.array([binding_score(d, readings, cfg) for d in docs])
            assert np.allclose(fast, ref, atol=1e-5), (trial, cfg, np.abs(fast - ref).max())


def test_fast_binary_equals_reference_binary():
    rng = np.random.default_rng(6)
    docs = random_corpus(rng)
    store = FactStore(dim=16, precision="binary")
    store.add_docs(docs)
    # reference documents with packed int8 contexts
    ref_docs = [Doc(d.doc_id, [Fact(f.fact_id, f.keys, binarize(f.ctx) if not f.native else None, f.native) for f in d.facts]) for d in docs]
    for trial in range(20):
        g = unit(rng.normal(size=16))
        v = f"n{int(rng.integers(8))}"
        readings = [Reading(f"s0:{s}", 0, {f"{s}|{v}": 1.0}, float(rng.uniform(0.1, 1.0)), g) for s in ("a", "b", "c")]
        cfg = ScoreConfig(precision="binary")
        fast = store.scores([d.doc_id for d in docs], readings, cfg)
        ref = np.array([binding_score(d, readings, cfg) for d in ref_docs])
        assert np.allclose(fast, ref, atol=1e-5), (trial, np.abs(fast - ref).max())


if __name__ == "__main__":
    test_fast_equals_reference()
    test_fast_binary_equals_reference_binary()
    print("fast scorer parity (float and binary): PASS")

"""Synthetic sanity check for the binding claim (DEVELOPMENT.md M3-T6).

A corpus where every fact carries one value in one true role. Documents often
contain the queried value in a different role in one fact and the queried
role's context in another fact (the mis-binding trap). Expected outcome:
'bound' clearly beats 'unbound' and 'gate_only'.
"""
import numpy as np
from slotspace.reference import Doc, Fact, Reading, ScoreConfig, binding_score, unit, TAU_MIN


def make_corpus(rng, n_docs, n_vals, n_roles, dim, sig_ctx, pi_strength):
    protos = np.array([unit(rng.normal(size=dim)) for _ in range(n_roles)])
    zipf = 1.0 / np.arange(1, n_vals + 1) ** 0.8
    zipf /= zipf.sum()
    docs, truth, correct = [], {}, []
    for i in range(n_docs):
        facts = []
        for j in range(int(rng.poisson(5)) + 2):
            v = f"v{int(rng.choice(n_vals, p=zipf))}"
            r = int(rng.integers(n_roles))
            ctx = unit(protos[r] + sig_ctx * rng.normal(size=dim) / 4)
            logits = pi_strength * np.eye(n_roles)[r] + rng.gumbel(size=n_roles)
            pi = np.exp(logits - logits.max()); pi /= pi.sum()
            correct.append(int(np.argmax(pi) == r))
            keys = {f"slot{k}|{v}": float(pi[k]) for k in range(n_roles) if pi[k] >= TAU_MIN}
            facts.append(Fact(f"d{i}.f{j}", keys, ctx))
            truth.setdefault((v, r), set()).add(f"d{i}")
        docs.append(Doc(f"d{i}", facts))
    return docs, truth, protos, float(np.mean(correct))


def run(seed=7, n_docs=2000, n_vals=300, n_roles=4, dim=64, sig_ctx=0.9, pi_strength=1.2,
        sig_q=0.6, c_strength=2.0, n_q=300, cfgs=None):
    rng = np.random.default_rng(seed)
    docs, truth, protos, doc_acc = make_corpus(rng, n_docs, n_vals, n_roles, dim, sig_ctx, pi_strength)
    pairs = sorted(vk for vk, ds in truth.items() if 3 <= len(ds) <= 60)
    pick = rng.choice(len(pairs), size=min(n_q, len(pairs)), replace=False)
    out = {name: {"rprec": [], "mrr": []} for name in cfgs}
    q_acc = []
    for qi in pick:
        v, k = pairs[qi]
        rel = truth[(v, k)]
        g = unit(protos[k] + sig_q * rng.normal(size=dim) / 4)
        logits = c_strength * np.eye(n_roles)[k] + rng.gumbel(size=n_roles)
        C = np.exp(logits - logits.max()); C /= C.sum()
        q_acc.append(int(np.argmax(C) == k))
        readings = [Reading(f"r{j}", 0, {f"slot{j}|{v}": 1.0}, float(C[j]), g) for j in range(n_roles)]
        cand = [d for d in docs if any(key.endswith("|" + v) for f in d.facts for key in f.keys)]
        for name, cfg in cfgs.items():
            ids = [d.doc_id for d in sorted(cand, key=lambda d: (-binding_score(d, readings, cfg), d.doc_id))]
            R = len(rel)
            out[name]["rprec"].append(len(set(ids[:R]) & rel) / R)
            out[name]["mrr"].append(next((1.0 / (i + 1) for i, d in enumerate(ids) if d in rel), 0.0))
    return out, doc_acc, float(np.mean(q_acc)), len(pick)


if __name__ == "__main__":
    cfgs = {
        "bound a=1 b=1": ScoreConfig(),
        "bound a=0.5 b=1": ScoreConfig(a=0.5),
        "unbound": ScoreConfig(mode="unbound"),
        "gate_only": ScoreConfig(mode="gate_only"),
        "no_role": ScoreConfig(mode="no_role"),
    }
    for label, kw in [("default", {}), ("sharper doc-side roles", {"pi_strength": 2.5}), ("weaker context vectors", {"sig_ctx": 2.5})]:
        out, da, qa, nq = run(cfgs=cfgs, **kw)
        print(f"[{label}] queries={nq} doc-side role acc={da:.3f} query-side role acc={qa:.3f}")
        for name, d in out.items():
            print("   %-16s R-prec=%.3f  MRR=%.3f" % (name, np.mean(d["rprec"]), np.mean(d["mrr"])))

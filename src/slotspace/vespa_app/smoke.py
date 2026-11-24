"""Four-document real-engine smoke, safe only on an empty dedicated deployment."""

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from ..reference import Doc, Fact, binarize, unit
from .application import EXPONENTS, profile_name, write_application
from .feed import vespa_doc
from .parity import compare_binding
from .query import vespa_body

IND, CON, OFF = "drug.indication.disease", "drug.contraindication.disease", "drug.off_label_use.disease"
SIDE = "drug.side_effect.effect_phenotype"


def smoke_fixture():
    rng = np.random.default_rng(13)
    slots = {k: i for i, k in enumerate((IND, CON, OFF, SIDE))}
    contexts = [unit(rng.normal(size=128)) for _ in range(5)]
    docs = [
        Doc("1", [Fact("1.n", {f"{IND}|n101": 1.0}, None, True),
                  Fact("1.t", {f"{IND}|n101": 0.92, f"{CON}|n101": 0.05}, contexts[0])]),
        Doc("2", [Fact("2.t", {f"{CON}|n101": 0.60, f"{IND}|n101": 0.25,
                             f"{OFF}|n101": 0.15}, contexts[1])]),
        Doc("3", [Fact("3.n1", {f"{IND}|n103": 1.0}, None, True),
                  Fact("3.n2", {f"{SIDE}|n202": 1.0}, None, True),
                  Fact("3.t1", {f"{CON}|n103": 0.90, f"{IND}|n103": 0.06}, contexts[2]),
                  Fact("3.t2", {f"{IND}|n101": 0.35, f"{CON}|n101": 0.30,
                               f"{OFF}|n101": 0.35}, contexts[3])]),
        Doc("4", [Fact("4.t", {f"{CON}|n104": 0.88, f"{IND}|n104": 0.07}, contexts[4])]),
    ]
    plan = {"segments": [{"g": unit(rng.normal(size=128)).tolist(), "links": {"n101": 0.7, "n202": 0.2},
                          "readings": [{"key": f"{CON}|n101", "w": 0.6254, "p": 0.6254},
                                       {"key": f"{IND}|n101", "w": 0.2014, "p": 0.2014},
                                       {"key": f"{SIDE}|n202", "w": 0.1732, "p": 0.1732}]}],
            "budgets": {"K_L": 200, "K_D": 200}, "weight_objective": "p"}
    topic = unit(rng.normal(size=256))
    operations, reference = [], {}
    for doc in docs:
        native = [k for f in doc.facts if f.native for k in f.keys]
        texts = [(f"t{i}", f.keys, f.ctx) for i, f in enumerate(doc.facts) if not f.native]
        operations.append(vespa_doc(doc.doc_id, "drug", "hypertension", topic, native, texts, slots))
        reference[int(doc.doc_id)] = Doc(doc.doc_id, [
            f if f.native else Fact(f.fact_id, f.keys, binarize(f.ctx)) for f in doc.facts])
    return operations, reference, plan, topic, slots


def smoke(client, *, deploy=False):
    if deploy:
        with TemporaryDirectory(prefix="slotspace-vespa-smoke-") as tmp:
            app = Path(tmp) / "app"
            write_application(app)
            client.deploy(app)
    client.wait_ready()
    if client.document_count() != 0:
        raise ValueError("Smoke requires an EMPTY dedicated Vespa app; existing documents were not touched")
    operations, reference, plan, topic, slots = smoke_fixture()
    try:
        receipt = client.feed(operations)
        client.wait_count(4)
        scores = {}
        for a, b in EXPONENTS:
            response = client.search(vespa_body(plan, "", topic, slots, a, b, match_all=True))
            if {h["doc_id"] for h in response["hits"]} != set(reference):
                raise ValueError("Smoke did not return all four toy documents")
            scores[profile_name(a, b)] = compare_binding(response["hits"], plan, reference.__getitem__, a, b)
        fallback = client.search(vespa_body({**plan, "segments": []}, "", topic, slots, match_all=True))
        if {h["doc_id"] for h in fallback["hits"]} != set(reference):
            raise ValueError("Hybrid profile did not return all toy documents")
        return {"status": "ok", "source": "real Vespa HTTP", "feed": receipt, "profiles": scores}
    finally:
        # Only these four newly fed IDs are removed, never an application's entire corpus.
        for number in reference:
            client.delete_one(number)
        client.wait_count(0)

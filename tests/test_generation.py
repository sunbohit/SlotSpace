from copy import deepcopy
from itertools import groupby
import random
from types import SimpleNamespace

import pytest

from slotspace.gen.fill_templates import fill_sample, generate_samples
from slotspace.gen.paraphrase_vllm import paraphrase_batch
from slotspace.gen.templates import make_templates
from slotspace.gen.validate import mask_sample, validate_samples, validate_text
from slotspace.schema import Schema
from test_schema import SCHEMA

IND, CON = "drug.indication.disease", "drug.contraindication.disease"


def test_target_binding_masking_and_missing_surrogate_rejection():
    schema = Schema.load(SCHEMA)
    templates = make_templates(schema, {})
    sample = fill_sample(schema, templates[CON], CON, "doc", 0, random.Random(13), "a", second_slot=IND)
    assert validate_text(sample["text"], sample, schema) is None
    masked = mask_sample(sample["text"], sample, untyped=False)
    assert "[TARGET:disease]" in masked and "[SELF:drug]" in masked and "[disease]" in masked
    assert all(t not in masked for t in sample["tokens"])
    target = next(t for t, role in sample["tokens"].items() if role == "target")
    other = next(t for t, role in sample["tokens"].items() if role.startswith("other:"))
    swapped = sample["text"].replace(target, "PLACEHOLDER").replace(other, target).replace("PLACEHOLDER", other)
    assert validate_text(swapped, sample, schema) == "foreign_target_cue"
    assert validate_text(sample["text"].replace(target, "missing"), sample, schema) == "missing_or_repeated_token"
    negative = "gene_protein.expression_absent.anatomy"
    absent = fill_sample(schema, templates[negative], negative, "doc", 0, random.Random(13), "b")
    assert validate_text(absent["text"], absent, schema) is None


def test_hard_pairs_share_bindings_and_visible_targets_are_explicit():
    schema = Schema.load(SCHEMA)
    templates = make_templates(schema, {IND: [CON], CON: [IND]})
    rows = list(generate_samples(schema, templates, 2))
    hard = sorted((r for r in rows if "hard_pair" in r), key=lambda r: r["hard_pair"])
    for _, pair in groupby(hard, key=lambda r: r["hard_pair"]):
        a, b = list(pair)
        assert a["bindings"] == b["bindings"] and a["slot"] != b["slot"]
    sample = fill_sample(schema, templates[IND], IND, "query", 0, random.Random(13), "a12",
                         real_names={"disease": ["Hypertension"]})
    assert "Hypertension" in mask_sample(sample["text"], sample, untyped=False) and sample["variant"] == "a12"
    assert all(row["variant"] == "masked" for row in rows)


def test_balancing_covers_every_slot_style_and_does_not_mix_a12():
    schema = Schema.load(SCHEMA)
    templates = make_templates(schema, {})
    source = list(generate_samples(schema, templates, 20))
    assert all(4 <= len(row["text"].split()) <= 60 for row in source)
    rows, stats = validate_samples(source, schema, 12)
    assert len(rows) == 66 * 2 * 12 and stats["paraphrase_candidates"] == 0
    assert len({r["sample_id"] for r in rows}) == len(rows)
    assert all("[TARGET" in r["context"] for r in rows)
    bad = deepcopy(source[0])
    bad["variant"] = "a12"
    with pytest.raises(ValueError, match="cannot be mixed"):
        validate_samples([source[0], bad], schema, 1)


def test_paraphraser_keeps_the_source_mapping_and_checks_output_alignment():
    source = [{"sample_id": "a", "tokens": {"Zore10": "target"}, "text": "Study the effects of Zore10."}]

    class Model:
        def chat(self, messages, params):
            assert "Zore10" in messages[0][0]["content"]
            return [SimpleNamespace(outputs=[SimpleNamespace(text="First sentence.\nSecond sentence.\nThird sentence.")])]

    result = paraphrase_batch(source, Model(), object())
    assert result[0]["tokens"] == source[0]["tokens"] and len(result[0]["rewrites"]) == 3
    assert "rewrites" not in source[0]

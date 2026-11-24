import pytest

from slotspace.text import mask, normalize, split_sentences


def test_sentence_boundaries_long_chunks_and_nfkc_normalization():
    assert normalize("  Hypertension. ") == "hypertension"
    assert normalize("Ｈｙｐｅｒｔｅｎｓｉｏｎ") == "hypertension"
    assert split_sentences("Used e.g. for pain. It works well.") == ["Used e.g. for pain.", "It works well."]
    assert split_sentences("one two three; Four five six.", max_words=3) == ["one two three", "Four five six."]


def test_masks_use_original_offsets_and_no_field_prefix_for_the_scorer():
    sentence = "Do not use Aspirin in asthma."
    mentions = [
        {"start": 11, "end": 18, "types": {"drug"}, "is_self": True},
        {"start": 22, "end": 28, "types": {"disease"}},
    ]
    # Compute offsets from the original sentence to keep this test unambiguous.
    for m, surface in zip(mentions, ("Aspirin", "asthma")):
        m["start"] = sentence.index(surface)
        m["end"] = m["start"] + len(surface)
    assert mask(sentence, mentions, 1, "drug", "description") == (
        "field: description | Do not use [SELF:drug] in [TARGET:disease].")
    assert mask(sentence, mentions, None, "drug", "description") == (
        "field: description | Do not use [SELF:drug] in [disease].")
    assert mask(sentence, mentions, 1, "drug", None) == "Do not use [SELF:drug] in [TARGET:disease]."
    mentions[1]["types"].add("effect_phenotype")
    assert "[TARGET]" in mask(sentence, mentions, 1, "drug", None)
    mentions[1]["start"] = mentions[0]["start"]
    with pytest.raises(ValueError, match="overlap"):
        mask(sentence, mentions, None, "drug", None)

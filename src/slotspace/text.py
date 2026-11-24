"""Sentence splitting and offset-preserving mention masking."""

import re
import string
import unicodedata

_WORD = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9'\-/\.]*[A-Za-z0-9])?")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).strip().strip(string.punctuation + " ")


def split_sentences(text: str, max_words: int = 64) -> list[str]:
    if max_words < 1:
        raise ValueError("Sentence length limit must be positive")
    out = []
    for sentence in _SPLIT.split(text):
        parts = sentence.split(";") if len(sentence.split()) > max_words else [sentence]
        for part in parts:
            words = part.split()
            out.extend(" ".join(words[j:j + max_words]) for j in range(0, len(words), max_words))
    return out


def token_windows(text: str, tokenizer, size: int = 128) -> list[str]:
    if size < 1:
        raise ValueError("Token window size must be positive")
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    offsets = [(int(start), int(end)) for start, end in offsets if end > start]
    if any(not 0 <= start < end <= len(text) for start, end in offsets):
        raise ValueError("Tokenizer returned invalid source offsets")
    return [text[offsets[i][0]:offsets[min(i + size, len(offsets)) - 1][1]]
            for i in range(0, len(offsets), size)]


def visible_target(context: str, surface: str) -> str:
    pattern = r"\[TARGET(?::[a-z_]+)?\]"
    if len(re.findall(pattern, context)) != 1 or not surface:
        raise ValueError("Visible-target inference requires exactly one target marker and its original surface")
    return re.sub(pattern, lambda _: surface, context)


def mask(sentence: str, mentions: list[dict], target: int | None,
         self_type: str, field: str | None) -> str:
    if target is not None and not 0 <= target < len(mentions):
        raise ValueError("Target mention index is outside the sentence")
    end = 0
    for mention in sorted(mentions, key=lambda m: m["start"]):
        if not end <= mention["start"] < mention["end"] <= len(sentence) or not mention["types"]:
            raise ValueError("Mention offsets/types are invalid or overlap")
        end = mention["end"]
    result = sentence
    for i in sorted(range(len(mentions)), key=lambda j: -mentions[j]["start"]):
        m = mentions[i]
        value_type = next(iter(m["types"])) if len(m["types"]) == 1 else None
        if m.get("is_self"):
            token = f"[SELF:{self_type}]"
        elif i == target:
            token = f"[TARGET:{value_type}]" if value_type else "[TARGET]"
        else:
            token = f"[{value_type}]" if value_type else "[entity]"
        result = result[:m["start"]] + token + result[m["end"]:]
    return f"field: {field} | {result}" if field else result

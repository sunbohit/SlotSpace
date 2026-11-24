"""Closed-vocabulary Prime entity linker; fuzzy fallback is query-only."""

from collections import defaultdict

from rapidfuzz import fuzz, process
from wordfreq import top_n_list

from .text import _WORD, normalize


class Linker:
    def __init__(self, docs_df, max_ngram: int = 8, min_chars: int = 3,
                 common_words: int = 5000, fuzzy_threshold: float = 92, fuzzy_mass: float = 0.5):
        if (min(max_ngram, min_chars) < 1 or common_words < 0 or not 0 < fuzzy_mass <= 1
                or not 0 <= fuzzy_threshold <= 100):
            raise ValueError("Invalid linker lengths, fuzzy threshold or probability mass")
        self.max_ngram, self.fuzzy_threshold, self.fuzzy_mass = max_ngram, fuzzy_threshold, fuzzy_mass
        common = set(top_n_list("en", common_words))
        self.norm_index, self.case_index = defaultdict(set), defaultdict(set)
        self.vtype, self.name_by_id = {}, {}
        for doc_id, name, doc_type in docs_df[["doc_id", "name", "doc_type"]].itertuples(index=False):
            vid = f"n{doc_id}"
            if vid in self.vtype:
                raise ValueError(f"Duplicate document ID in linker: {doc_id}")
            self.vtype[vid], self.name_by_id[vid] = doc_type, name
            n = normalize(name)
            if len(n) < min_chars or not any(c.isalpha() for c in n):
                continue
            if " " not in n and n in common:
                if name.isupper() and doc_type == "gene_protein":
                    self.case_index[name].add(vid)
                continue
            self.norm_index[n].add(vid)
        inverted = defaultdict(set)
        for name in self.norm_index:
            for token in set(_WORD.findall(name)):
                inverted[token].add(name)
        self.word_to_names = {w: tuple(sorted(names)) for w, names in inverted.items()}

    def _mention(self, start: int, end: int, text: str, ids: set[str], fuzzy: bool = False) -> dict:
        by_type = defaultdict(list)
        for vid in sorted(ids):
            by_type[self.vtype[vid]].append(vid)
        mass = self.fuzzy_mass if fuzzy else 1.0
        values = {v: mass / len(vs) for vs in by_type.values() for v in vs}
        return {"start": start, "end": end, "surface": text[start:end], "values": values,
                "types": set(by_type), "fuzzy": fuzzy}

    def link(self, text: str) -> list[dict]:
        words = list(_WORD.finditer(text))
        out, i = [], 0
        while i < len(words):
            for n in range(min(self.max_ngram, len(words) - i), 0, -1):
                start, end = words[i].start(), words[i + n - 1].end()
                surface = text[start:end]
                ids = self.norm_index.get(normalize(surface)) or self.case_index.get(surface)
                if ids:
                    out.append(self._mention(start, end, text, ids))
                    i += n
                    break
            else:
                i += 1
        return out

    def link_query(self, text: str) -> list[dict]:
        exact = self.link(text)
        out = list(exact)
        bounds = [(-1, 0), *[(m["start"], m["end"]) for m in exact], (len(text), len(text))]
        for left, right in zip(bounds, bounds[1:]):
            words = [w for w in _WORD.finditer(text) if w.start() >= left[1] and w.end() <= right[0]]
            i = 0
            while i < len(words):
                for n in range(min(6, len(words) - i), 1, -1):
                    start, end = words[i].start(), words[i + n - 1].end()
                    span = normalize(text[start:end])
                    candidates = set()
                    for token in _WORD.findall(span):
                        candidates.update(self.word_to_names.get(token, ())[:5000])
                    found = process.extractOne(span, sorted(candidates)[:5000],
                                               scorer=fuzz.token_set_ratio, score_cutoff=self.fuzzy_threshold)
                    if found is not None:
                        out.append(self._mention(start, end, text, self.norm_index[found[0]], fuzzy=True))
                        i += n
                        break
                else:
                    i += 1
        return sorted(out, key=lambda m: m["start"])

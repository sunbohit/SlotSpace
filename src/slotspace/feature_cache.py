"""Immutable, lossless query features for CPU-only tuning replay."""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .artifacts import new_directory, sha256, write_json
from .planner import PlannerSettings, plan_observations, plan_query, plan_readings
from .ranking import fact_work, rank_matches, select_candidates
from .reference import Lexicon, Reading, ScoreConfig
from .retrieve import topk


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def scoring_spec(cfg: ScoreConfig) -> dict:
    if (cfg.mode not in {"bound", "unbound", "gate_only", "no_role"} or cfg.combine not in {"max", "sum"}
            or cfg.precision not in {"float", "binary"}
            or any(not np.isfinite(x) or x <= 0 for x in (cfg.a, cfg.b))):
        raise ValueError("Invalid cached scoring configuration")
    return {**asdict(cfg), "a": float(cfg.a), "b": float(cfg.b)}


class SnapshotLexicon:
    def __init__(self, value: dict):
        self.value = value
        self.version = value["version"]

    def support(self, value):
        return dict(self.value["soft"].get(value, {}))

    def raw_support(self, value):
        return dict(self.value["raw"].get(value, {}))

    def hard_support(self, value, slot):
        return self.value["hard"].get(value, {}).get(slot, 0)


@dataclass
class QueryFeatures:
    payload: dict
    arrays: dict[str, np.ndarray]
    fingerprint: str = ""
    folder: Path | None = None
    verified_matrices: dict = field(default_factory=dict, repr=False)

    @property
    def lexicon(self):
        return SnapshotLexicon(self.payload["lexicon"])

    def readings(self):
        obs = self.payload["observations"]
        return [Reading(c["rid"], c["segment"], {c["key"]: c["lp"]}, 1.0,
                        np.asarray(obs[c["segment"]]["g"], dtype=np.float32)) for c in self.payload["columns"]]

    def plan(self, schema, cfg: PlannerSettings, threshold: float, oracle=False):
        plan = plan_observations(self.payload["qid"], self.payload["observations"], schema, self.lexicon, cfg, threshold)
        if oracle:
            for i, (segment, truth) in enumerate(zip(plan["segments"], self.payload["true_readings"])):
                by_key = {c["key"]: c for c in self.payload["columns"] if c["segment"] == i}
                if set(truth) - by_key.keys():
                    raise ValueError("Oracle reading is absent from the cached supported inventory")
                segment["readings"] = [{"key": k, "p": 1 / len(truth), "w": 1 / len(truth),
                                        "support_hard": by_key[k]["hard"]} for k in sorted(truth)]
            plan["path"] = "both" if any(s["readings"] for s in plan["segments"]) else "vector_first"
        return plan

    def save(self, folder: Path):
        with new_directory(folder) as stage:
            write_json(stage / "query.json", self.payload)
            np.savez_compressed(stage / "arrays.npz", **self.arrays)
            meta = {"format": 1, "files": {p: sha256(stage / p) for p in ("query.json", "arrays.npz")}}
            meta["fingerprint"] = digest(meta)
            write_json(stage / "manifest.json", meta)
        self.folder, self.fingerprint = folder, meta["fingerprint"]

    @classmethod
    def load(cls, folder: Path):
        meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        base = {k: v for k, v in meta.items() if k != "fingerprint"}
        if (meta["format"] != 1 or set(meta["files"]) != {"query.json", "arrays.npz"}
                or digest(base) != meta["fingerprint"]
                or any(sha256(folder / p) != h for p, h in meta["files"].items())):
            raise ValueError(f"Corrupt query-feature cache: {folder}")
        payload = json.loads((folder / "query.json").read_text(encoding="utf-8"))
        with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        ids = arrays["doc_ids"]
        if (ids.ndim != 1 or not len(ids) or not np.all(ids[:-1] < ids[1:])
                or any(arrays[k].shape != ids.shape for k in ("lexical", "dense", "text_counts"))
                or not all(np.isfinite(arrays[k]).all() for k in ("lexical", "dense"))):
            raise ValueError("Invalid cached query feature shapes")
        return cls(payload, arrays, meta["fingerprint"], folder)


def capture_query(pipeline, query: dict, true_segments: list[dict] | None = None) -> QueryFeatures:
    if pipeline.ctx.cfg.uniform_slots:
        raise ValueError("Capture the learned model first; uniform-model replay does not need new inference")
    start = perf_counter()
    plan = plan_query(query["query"], query["qid"], pipeline.ctx)
    observations = [{k: s[k] for k in ("span", "text", "links", "value_types", "distribution", "g")}
                    for s in plan["segments"]]
    if true_segments is not None and [s["span"] for s in observations] != [
            [s["start"], s["end"]] for s in true_segments]:
        raise ValueError("Silver labels and current linker spans differ")
    if true_segments is not None and any("links" in s and s["links"] != o["links"]
                                         for s, o in zip(true_segments, observations)):
        raise ValueError("Silver labels and current linker values/probabilities differ")
    truth = [list(s["true_readings"]) for s in true_segments] if true_segments is not None else [[] for _ in observations]
    soft, raw, hard, columns = {}, {}, {}, []
    for i, observation in enumerate(observations):
        candidates = pipeline.ctx.schema.candidates(None, set(observation["value_types"].values()))
        for value, lp in observation["links"].items():
            soft[value] = pipeline.ctx.lexicon.support(value)
            raw[value] = (pipeline.ctx.lexicon.support(value) if isinstance(pipeline.ctx.lexicon, Lexicon)
                          else pipeline.ctx.lexicon.raw_support(value))
            hard[value] = {slot: pipeline.ctx.lexicon.hard_support(value, slot) for slot in candidates
                           if not slot.startswith("none.")}
            for slot in candidates:
                if (not slot.startswith("none.") and pipeline.ctx.schema.slot(slot).value_type
                        == observation["value_types"][value] and hard[value][slot] > 0):
                    key = f"{slot}|{value}"
                    columns.append({"rid": f"s{i}:{key}", "segment": i, "key": key, "lp": lp,
                                    "hard": hard[value][slot]})
        if set(truth[i]) - {c["key"] for c in columns if c["segment"] == i}:
            raise ValueError("Silver truth contains an unsupported reading")
    topic = pipeline.ctx.embedder.queries([query["query"]], dim=pipeline.dense.documents.shape[1])[0]
    dense = pipeline.dense.scores(topic)
    lexical, postings_work = pipeline.bm25.scores(query["query"])
    limits = {"K_L": pipeline.ctx.cfg.K_L, "K_D": pipeline.ctx.cfg.K_D}
    universe = np.union1d(topk(lexical, limits["K_L"]), topk(dense, limits["K_D"]))
    original_postings = {}
    for key in sorted({c["key"] for c in columns}):
        rows = pipeline.index.postings.get(key, np.empty(0, np.int64))
        slot, value = key.split("|", 1)
        if len(rows) != pipeline.ctx.lexicon.hard_support(value, slot):
            raise ValueError("Cannot cache an inconsistent posting list")
        original_postings[key] = rows
        # Do not cap the union here: pruning a different reading can expose a
        # different per-query top-K_V posting that must remain in the cache.
        universe = np.union1d(universe, np.searchsorted(pipeline.index.doc_ids, rows))
    ids = pipeline.index.doc_ids[universe]
    payload = {"qid": query["qid"], "query": query["query"], "answers": sorted(set(query["answers"])),
               "observations": observations, "columns": columns, "true_readings": truth,
               "lexicon": {"version": pipeline.ctx.lexicon.version, "soft": soft, "raw": raw, "hard": hard},
               "index_fingerprint": pipeline.index.fingerprint, "limits": limits,
               "fixed_settings": {k: getattr(pipeline.ctx.cfg, k) for k in
                                  ("ctx_dim", "window_words", "segment_instruction")},
               "postings": {k: np.searchsorted(ids, rows).tolist() for k, rows in original_postings.items()},
               "bm25_postings": postings_work, "query_temperature": getattr(pipeline.ctx.scorer, "T", None),
               "feature_extraction_ms": 1000 * (perf_counter() - start)}
    return QueryFeatures(payload, {"doc_ids": ids, "lexical": lexical[universe], "dense": dense[universe],
                                    "topic": topic, "text_counts": np.array([
                                        len(pipeline.index.texts.get(str(doc), ())) for doc in ids], dtype=np.int64)})


def matrix_folder(feature: QueryFeatures, cfg: ScoreConfig) -> Path:
    if feature.folder is None:
        raise ValueError("Persist base features before computing match matrices")
    return feature.folder / "matrices" / digest(scoring_spec(cfg))[:16]


def cache_matches(feature: QueryFeatures, index, cfg: ScoreConfig):
    if (feature.payload["index_fingerprint"] != index.fingerprint
            or feature.payload["lexicon"]["version"] != index.version
            or cfg.precision != index.store.precision):
        raise ValueError("Feature geometry/index/precision mismatch")
    target = matrix_folder(feature, cfg)
    if target.exists():
        return load_matches(feature, cfg)
    readings = feature.readings()
    shape = (len(feature.arrays["doc_ids"]), len(readings))
    if np.prod(shape) * 16 > shutil.disk_usage(target.parent.parent).free - 64 * 1024 ** 2:
        raise OSError("Insufficient disk space for the requested exact match/work matrices")
    with new_directory(target) as stage:
        if not readings:
            np.save(stage / "matches.npy", np.empty(shape, np.float64))
            np.save(stage / "work.npy", np.empty(shape, np.int64))
        else:
            matches = np.lib.format.open_memmap(stage / "matches.npy", mode="w+", dtype=np.float64, shape=shape)
            work = np.lib.format.open_memmap(stage / "work.npy", mode="w+", dtype=np.int64, shape=shape)
            for start in range(0, shape[0], 1024):
                ids = [str(d) for d in feature.arrays["doc_ids"][start:start + 1024]]
                for j, reading in enumerate(readings):
                    matches[start:start + len(ids), j] = index.store.scores(ids, [reading], cfg)
                work[start:start + len(ids)] = fact_work(index.store, ids, readings, cfg.mode)
            matches.flush()
            work.flush()
            del matches, work
        write_json(stage / "manifest.json", {"feature": feature.fingerprint, "scoring": scoring_spec(cfg),
                                             "files": {p: sha256(stage / p) for p in ("matches.npy", "work.npy")}})
    return load_matches(feature, cfg)


def load_matches(feature: QueryFeatures, cfg: ScoreConfig):
    folder = matrix_folder(feature, cfg)
    meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (meta["feature"] != feature.fingerprint or meta["scoring"] != scoring_spec(cfg)
            or set(meta["files"]) != {"matches.npy", "work.npy"}):
        raise ValueError("Match matrix source/scoring/checksum mismatch")
    paths = [folder / p for p in ("manifest.json", "matches.npy", "work.npy")]
    stamps = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    if feature.verified_matrices.get(str(folder)) != stamps:
        if any(sha256(folder / p) != h for p, h in meta["files"].items()):
            raise ValueError("Match matrix checksum mismatch")
        feature.verified_matrices[str(folder)] = stamps
    mode = "r" if feature.payload["columns"] else None
    matrix, work = (np.load(folder / p, mmap_mode=mode, allow_pickle=False) for p in ("matches.npy", "work.npy"))
    shape = (len(feature.arrays["doc_ids"]), len(feature.payload["columns"]))
    if matrix.shape != shape or work.shape != shape or matrix.dtype != np.float64 or work.dtype != np.int64:
        raise ValueError("Invalid cached match matrix shapes/dtypes")
    return matrix, work


def replay(feature: QueryFeatures, schema, settings: PlannerSettings, threshold: float,
           scoring: ScoreConfig, weights=(1.0, 1.0, 0.5), final_k=100, oracle=False, hard_and=False):
    start = perf_counter()
    if any(getattr(settings, k) > feature.payload["limits"][k] for k in ("K_L", "K_D")):
        raise ValueError("Requested lexical/dense budget exceeds the cached maxima")
    if any(getattr(settings, k) != v for k, v in feature.payload["fixed_settings"].items()):
        raise ValueError("Cached segment encoder/window settings differ")
    plan = feature.plan(schema, settings, threshold, oracle)
    if weights[0] == 0:
        plan = {**plan, "segments": [], "path": "vector_first", "weight_objective": "p",
                "planner_encoder_texts": 0, "estimated_value_postings": 0, "binding_disabled": True}
    readings = plan_readings(plan) if weights[0] != 0 else []
    keys = {k for r in readings for k in r.keys}
    postings = {k: np.array(feature.payload["postings"][k], dtype=np.int64) for k in keys}
    a = feature.arrays
    pool, posting_work = select_candidates(a["lexical"], a["dense"], postings, keys, plan["budgets"])
    if readings:
        lookup = {c["rid"]: j for j, c in enumerate(feature.payload["columns"])}
        cols = [lookup[r.rid] for r in readings]
        cached, cached_work = load_matches(feature, scoring)
        matrix = np.ascontiguousarray(cached[np.ix_(pool, cols)], dtype=np.float64)
        work = int(cached_work[np.ix_(pool, cols)].sum())
    else:
        matrix, work = np.empty((len(pool), 0)), 0
    order, scores, fused = rank_matches(a["doc_ids"][pool], a["lexical"][pool], a["dense"][pool],
                                        matrix, readings, plan, weights, final_k, hard_and)
    evidence_work = int(a["text_counts"][pool[order[:10]]].sum()) * len(readings)
    return {"doc_ids": a["doc_ids"][pool[order]].tolist(), "scores": scores.tolist(), "plan": plan,
            "fusion_scores": fused[order].tolist(),
            "counters": {"bm25_postings": feature.payload["bm25_postings"], "dense_searches": 1,
                         "value_postings": posting_work, "candidates": len(pool),
                         "facts_scored": work + evidence_work, "evidence_facts_scored": evidence_work,
                         "encoder_texts": 1 + (plan["planner_encoder_texts"] if weights[0] else 0),
                         "latency_ms": None, "replay_ms": 1000 * (perf_counter() - start)}}

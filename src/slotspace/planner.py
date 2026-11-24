"""One-pass query plans with a joint slot posterior and conformal reading sets."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from .artifacts import logged_run, read_jsonl, sha256, write_json
from .calibrate_slot_scorer import query_records
from .embed import Embedder
from .lexicon_build import CorpusLexicon
from .linker import Linker
from .reference import Lexicon, Reading, conformal_threshold, keep_readings, reading_posterior
from .schema import Schema
from .slot_scorer import SlotScorer
from .text import mask, visible_target


@dataclass(frozen=True)
class PlannerSettings:
    beta: float = 0.5
    eps: float = 1.0
    r_max: int = 6
    window_words: int = 8
    weight_objective: str = "p"
    use_pacc: bool = True
    uniform_slots: bool = False
    segment_instruction: str = (
        "Instruct: Given a constraint taken from a search query, retrieve statements that express the same relation\nQuery:"
    )
    ctx_dim: int = 128
    K_L: int = 200
    K_D: int = 200
    K_V: int = 500

    def __post_init__(self):
        if (not np.isfinite(self.beta) or self.beta < 0 or not np.isfinite(self.eps) or self.eps <= 0
                or min(self.r_max, self.ctx_dim, self.K_L, self.K_D, self.K_V) < 1 or self.window_words < 0
                or self.weight_objective not in {"p", "p_over_H", "ia"}
                or not self.segment_instruction.endswith("Query:")):
            raise ValueError("Invalid planner hyperparameters or segment instruction")

    @classmethod
    def from_config(cls, config: dict):
        planner, embedding, retrieval = config["planner"], config["embedding"], config["retrieval"]
        return cls(beta=planner["beta"], eps=planner["eps"], r_max=planner["r_max"],
                   window_words=planner["window_words"], weight_objective=planner["weight_objective"],
                   use_pacc=planner.get("use_pacc", True), uniform_slots=planner.get("uniform_slots", False),
                   segment_instruction=embedding["segment_instruction"], ctx_dim=embedding["ctx_dim"],
                   K_L=retrieval["K_L"], K_D=retrieval["K_D"], K_V=retrieval["K_V"])


@dataclass
class PlannerContext:
    schema: Schema
    linker: Linker
    embedder: Embedder
    scorer: SlotScorer
    lexicon: CorpusLexicon | Lexicon
    cfg: PlannerSettings
    t_alpha: float


def window(text: str, words: int = 8) -> str:
    if words < 0:
        raise ValueError("Window size cannot be negative")
    tokens = list(re.finditer(r"\S+", text))
    target = [i for i, t in enumerate(tokens) if "[TARGET" in t.group()]
    if len(target) != 1:
        raise ValueError("A segment window needs exactly one target marker")
    center = target[0]
    return text[tokens[max(0, center - words)].start():tokens[min(len(tokens) - 1, center + words)].end()]


def posterior(links: dict[str, float], distribution: dict[str, float], schema: Schema,
              vtype: dict[str, str], lexicon, cfg: PlannerSettings):
    p = np.asarray(list(distribution.values()), dtype=float)
    if not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isclose(p.sum(), 1):
        raise ValueError("Slot scorer must return one normalized joint distribution including none")
    by_type = {}
    for slot, probability in distribution.items():
        if not slot.startswith("none."):
            by_type.setdefault(schema.slot(slot).value_type, {})[slot] = probability
    linked = {value: (vtype[value], probability) for value, probability in links.items()}
    if any(not 0 < probability <= 1 for _, probability in linked.values()):
        raise ValueError("Link probabilities must be in (0, 1]")
    if not cfg.use_pacc:
        raw = lexicon.support if isinstance(lexicon, Lexicon) else lexicon.raw_support

        class RawView:
            support = staticmethod(raw)
            hard_support = staticmethod(lexicon.hard_support)

        lexicon = RawView()
    return reading_posterior(by_type, linked, lexicon, beta=cfg.beta, eps=cfg.eps)


def uniform_distribution(candidates):
    slots = [k for k in candidates if not k.startswith("none.")]
    if not slots:
        raise ValueError("A uniform slot model requires at least one real slot")
    return {k: 1 / len(slots) if k in slots else 0.0 for k in candidates}


def condition_key(cfg: PlannerSettings) -> str:
    return f"beta={float(cfg.beta)},eps={float(cfg.eps)},pacc={int(cfg.use_pacc)},uniform={int(cfg.uniform_slots)}"


def conformal_value(meta: dict, cfg: PlannerSettings, alpha) -> float:
    if "conditions" in meta:
        key = condition_key(cfg)
        if key not in meta["conditions"]:
            raise ValueError(f"No conformal calibration for {key}")
        source = meta["conditions"][key]
    else:
        if (meta["beta"] != cfg.beta or meta["eps"] != cfg.eps
                or meta.get("use_pacc", True) != cfg.use_pacc or meta.get("uniform_slots", False) != cfg.uniform_slots):
            raise ValueError("Conformal calibration conditions differ")
        source = meta
    if alpha == "all":
        return 0.0
    if str(alpha) not in source["thresholds"]:
        raise ValueError(f"No calibrated threshold for alpha={alpha}")
    return float(source["thresholds"][str(alpha)])


def plan_query(query: str, qid, ctx: PlannerContext) -> dict:
    if not np.isfinite(ctx.t_alpha) or not 0 <= ctx.t_alpha <= 1:
        raise ValueError("Conformal threshold must lie in [0, 1]")
    mentions = ctx.linker.link_query(query)
    contexts = [mask(query, mentions, i, "", None) for i in range(len(mentions))]
    candidates = [ctx.schema.candidates(None, mention["types"]) for mention in mentions]
    scorer_contexts = ([visible_target(text, mention["surface"]) for text, mention in zip(contexts, mentions)]
                       if getattr(ctx.scorer, "values_visible", False) else contexts)
    distributions = ([uniform_distribution(c) for c in candidates] if ctx.cfg.uniform_slots
                     else ctx.scorer.score(scorer_contexts, candidates)) if mentions else []
    if len(distributions) != len(mentions):
        raise ValueError("Scorer batch is not aligned with query mentions")
    vectors = (ctx.embedder.queries([window(text, ctx.cfg.window_words) for text in contexts],
                                   dim=ctx.cfg.ctx_dim, prompt=ctx.cfg.segment_instruction)
               if mentions else np.empty((0, ctx.cfg.ctx_dim)))
    vectors = np.asarray(vectors, dtype=np.float32)
    if (vectors.shape != (len(mentions), ctx.cfg.ctx_dim) or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)):
        raise ValueError("Segment vectors must be aligned unit rows")
    observations = [{"span": [m["start"], m["end"]], "text": m["surface"], "links": m["values"],
                     "value_types": {v: ctx.linker.vtype[v] for v in m["values"]},
                     "distribution": d, "g": g.tolist()} for m, d, g in zip(mentions, distributions, vectors)]
    return plan_observations(qid, observations, ctx.schema, ctx.lexicon, ctx.cfg, ctx.t_alpha)


def plan_observations(qid, observations: list[dict], schema, lexicon, cfg: PlannerSettings, threshold: float):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid reading threshold")
    segments, keys = [], set()
    pair_count = 0
    for observation in observations:
        kinds = set(observation["value_types"].values())
        cands = schema.candidates(None, kinds)
        dist = uniform_distribution(cands) if cfg.uniform_slots else observation["distribution"]
        if set(dist) != set(cands):
            raise ValueError("Scorer distribution does not match the joint candidate set")
        pair_count += 0 if cfg.uniform_slots else len(cands)
        post = posterior(observation["links"], dist, schema, observation["value_types"], lexicon, cfg)
        kept = keep_readings(post, threshold, min_keep=1, r_max=cfg.r_max)
        c_none = min(1.0, sum(p for k, p in dist.items() if k.startswith("none.")))
        weights = dict(kept)
        if cfg.weight_objective == "p_over_H":
            weights = {k: p / max(1, lexicon.hard_support(k.split("|")[1], k.split("|")[0]))
                       for k, p in weights.items()}
        normalizer = sum(weights.values())
        weights = {k: (1 - c_none) * p / normalizer for k, p in weights.items()} if normalizer else {}
        zero = [f"{k}|{v}" for v in observation["links"] for k in cands if not k.startswith("none.")
                and schema.slot(k).value_type == observation["value_types"][v] and lexicon.hard_support(v, k) == 0]
        readings = [{"key": k, "p": post[k], "w": w,
                     "support_hard": lexicon.hard_support(k.split("|")[1], k.split("|")[0])}
                    for k, w in weights.items()]
        keys.update(weights)
        segments.append({**observation, "distribution": dist, "c_none": c_none, "readings": readings,
                         "posterior": post, "pruned_zero_support": sorted(zero),
                         "pruned_conformal": sorted(set(post) - set(kept))})
    return {"query_id": qid, "schema_version": schema.version, "lexicon_version": lexicon.version,
            "segments": segments, "path": "both" if keys else "vector_first",
            "budgets": {"K_L": cfg.K_L, "K_D": cfg.K_D, "K_V": cfg.K_V},
            "weight_objective": cfg.weight_objective,
            "estimated_value_postings": sum(lexicon.hard_support(k.split("|")[1], k.split("|")[0]) for k in keys),
            "planner_encoder_texts": len(observations) + pair_count}


def plan_readings(plan: dict) -> list[Reading]:
    return [Reading(f"s{i}:{r['key']}", i, {r["key"]: segment["links"][r["key"].split("|")[1]]},
                    r["w"], np.asarray(segment["g"], dtype=np.float32))
            for i, segment in enumerate(plan["segments"]) for r in segment["readings"]]


def calibrate_conformal(posteriors: list[dict], true_readings: list[list[str]],
                        alphas=(0.05, 0.1, 0.2)) -> dict:
    if len(posteriors) != len(true_readings) or any(not 0 < a < 1 for a in alphas):
        raise ValueError("Invalid conformal calibration inputs")
    values = [max(post.get(k, 0.0) for k in true) for post, true in zip(posteriors, true_readings) if true]
    if not values:
        raise ValueError("Conformal calibration requires labelled train segments")
    if not np.isfinite(values).all() or any(not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid true-reading calibration probabilities")
    return {"n_segments": len(values),
            "thresholds": {str(a): conformal_threshold(values, a) for a in alphas}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate reading sets on even-qid train silver only")
    parser.add_argument("--config", type=Path, default=Path("configs/prime_v0.yaml"))
    parser.add_argument("--silver", type=Path, default=Path("data/prime/silver_prime-0.1.0"))
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperatures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kind", choices=("ft", "zeroshot"), default="ft")
    parser.add_argument("--values-visible", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    schema_path, data = Path(config["schema"]), Path(config["paths"]["data"])
    schema = Schema.load(schema_path)
    cfg, lex = PlannerSettings.from_config(config), CorpusLexicon.load(args.lexicon)
    if lex.mode != "combined":
        raise ValueError("Conformal planning needs the complete text-aware lexicon")
    temperatures = json.loads(args.temperatures.read_text(encoding="utf-8"))
    silver_meta = json.loads((args.silver / "manifest.json").read_text(encoding="utf-8"))
    lex_meta = json.loads((args.lexicon / "manifest.json").read_text(encoding="utf-8"))
    train_path, docs_path = data / "queries_train.jsonl", data / "docs.parquet"
    if (not silver_meta["complete"] or silver_meta["query_hashes"]["train"] != sha256(train_path)
            or silver_meta["schema_sha256"] != sha256(schema_path)
            or silver_meta["docs_sha256"] != sha256(docs_path)
            or lex_meta["docs_sha256"] != sha256(docs_path)
            or lex_meta["schema_sha256"] != sha256(schema_path)
            or temperatures["schema_sha256"] != sha256(schema_path)
            or lex_meta["scorer"]["temperature"] != temperatures["T_doc"]):
        raise ValueError("Conformal calibration source/schema mismatch")
    docs = pd.read_parquet(docs_path, columns=["doc_id", "doc_type"])
    vtypes = {f"n{i}": t for i, t in docs.itertuples(index=False)}
    train_queries = {q["qid"]: q for q in read_jsonl(train_path)}
    records = query_records(read_jsonl(args.silver / "silver_calibration.jsonl"), train_queries,
                            docs.set_index("doc_id").doc_type.to_dict(), schema, args.values_visible)
    scorer = SlotScorer(args.model, schema, temperatures["T_query"],
                        "slot_first" if args.kind == "zeroshot" else "context_first", args.device)
    scorer.values_visible = args.values_visible
    if scorer.metadata()["model"] != temperatures["scorer"]["model"]:
        raise ValueError("Temperature and query scorer model identities differ")
    if scorer.values_visible != temperatures["scorer"].get("values_visible", False):
        raise ValueError("Temperature and query target-visibility settings differ")
    if (lex_meta["scorer"]["model"] != scorer.metadata()["model"]
            or lex_meta["scorer"]["pair_order"] != scorer.order
            or lex_meta["scorer"].get("values_visible", False) != scorer.values_visible):
        raise ValueError("Document and query slot scorers differ")
    meta = {"schema_version": schema.version, "schema_sha256": sha256(schema_path),
            "lexicon_version": lex.version, "scorer": scorer.metadata(),
            "temperatures_sha256": sha256(args.temperatures), "beta": cfg.beta, "eps": cfg.eps,
            "use_pacc": cfg.use_pacc, "uniform_slots": cfg.uniform_slots,
            "silver_sha256": sha256(args.silver / "silver_calibration.jsonl"), "seed": 13,
            "human_reviews": "pending"}
    with logged_run(Path("runs"), "m6-conformal", meta):
        distributions = ([uniform_distribution(r["candidates"]) for r in records] if cfg.uniform_slots else
                         scorer.score([r["context"] for r in records], [r["candidates"] for r in records]))
        post = [posterior(r["links"], c, schema, vtypes, lex, cfg) for r, c in zip(records, distributions)]
        result = {**meta, **calibrate_conformal(post, [r["true_readings"] for r in records])}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as dest:
            json.dump(result, dest, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("n_segments", "thresholds")}))


if __name__ == "__main__":
    main()

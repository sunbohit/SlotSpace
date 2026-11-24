# SlotSpace — Development Guide (v0)

| Item | Value |
|---|---|
| Version | 0.1 (2026-11-24) |
| Read first | `SlotSpace_DESIGN.md` Sections 1, 2, 4 and 7 (in the repository: `docs/DESIGN.md`) |
| Target | STaRK-Prime, Python research path (milestones M0–M8); Vespa engine path (M9) is v1 |
| Hardware | 1 GPU with 24 GB, 64 GB RAM, 200 GB free disk |
| Effort (estimate) | M0–M8 ≈ 7.5 person-weeks; M9–M10 ≈ 4.5 more |

This guide tells you, the implementing agent, exactly what to build, in which order, how to test it, and when it is done. Section numbers like "D7.1" point to `SlotSpace_DESIGN.md`.

---

## 0. Rules for the implementing agent

1. **Work milestone by milestone.** Do not start milestone M(n+1) before M(n) passes its acceptance checks. Commit after each passing milestone with the message `M<n>: <summary>`.
2. **Tests first for math.** The reference implementation (Appendix A) and its tests (Appendix B) are given verbatim. Copy them, run them, and keep them green forever. Never change the math to make a test pass; if a test fails, your copy or your environment is wrong.
3. **Never tune on test data.** Use Prime `train` only for silver labels, `val` for tuning and model selection, `test` and the human-generated set only for final runs, once, with a frozen configuration (M8).
4. **Cache expensive artifacts** (embeddings, facts, slot scores) under `data/` with the schema version and model name in the file name. Never recompute silently; never overwrite a cache with different settings.
5. **Log every run** to `runs/<run_id>/meta.json`: config, git commit, package versions (`pip freeze`), start and end time, machine.
6. **Seeds.** Use seed 13 for Python, numpy and torch everywhere.
7. **Ambiguity.** If this guide does not decide something, choose the simplest option, write the choice and the reason in `docs/DECISIONS.md`, and continue. Stop and ask a human only at the checkpoints in Section 7.
8. **Library calls.** Library versions change. If a call in this guide fails, read the installed library's help (`help(obj)`, `python -c "import x; print(x.__version__)"`), adapt the call, and record the fix in `docs/DECISIONS.md`. Do not change the data flow or the math.
9. **Keep `|` out of ids.** Slot ids and value ids never contain `|` (keys are `slot|value`).
10. **Small steps.** Run each script on 100 documents or 20 queries first (`--limit`), then on everything.

---

## 1. Environment

```bash
# main environment (Python 3.11: stark-qa supports < 3.12)
conda create -n slotspace python=3.11 -y
conda activate slotspace
pip install stark-qa numpy scipy pandas pyarrow pyyaml tqdm rapidfuzz wordfreq pytest datasets
pip install torch            # pick the CUDA build that matches the machine (see pytorch.org)
pip install "transformers>=4.51" sentence-transformers bm25s PyStemmer ranx faiss-cpu
# if importing stark_qa fails on torch_geometric: pip install torch_geometric

# separate environment for synthetic data generation only (vLLM pins its own torch)
conda create -n slotspace-gen python=3.11 -y
conda activate slotspace-gen
pip install vllm pyyaml
```

Version notes (checked 2026-11-24; re-check if something fails):

- Qwen3-Embedding needs `transformers >= 4.51` (older versions fail with `KeyError: 'qwen3'`). ModernBERT needs `transformers >= 4.48`.
- Recent sentence-transformers majors require a recent transformers major. Install them together, then record `pip freeze > requirements.lock`.
- In recent sentence-transformers, `quantize_embeddings` lives in `sentence_transformers.util.quantization` (the old `sentence_transformers.quantization` path still works with a warning). **Do not use it for engine vectors** — see M9 pitfalls.

Models (all downloaded from Hugging Face; licenses checked):

| Use | Model | License |
|---|---|---|
| topic, fact-context and segment vectors | `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 |
| slot scorer (fine-tuned) | `answerdotai/ModernBERT-base` | Apache-2.0 |
| zero-shot slot scorer (A13) | `BAAI/bge-reranker-v2-m3` | Apache-2.0 |
| paraphrasing training data (offline only) | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 |

---

## 2. Repository layout

```
slotspace/
  README.md
  pyproject.toml
  requirements.lock
  configs/
    prime_v0.yaml              # all hyperparameters (Section 3)
  schemas/
    prime.yaml                 # the schema (D5); slots derived in M2, descriptions written by hand
  templates/
    train/prime_slots.yaml     # training templates per slot (M5)
    test/role_swap.yaml        # TEST-ONLY templates (Appendix D); never read by training code
  src/slotspace/
    __init__.py
    config.py                  # load and validate YAML config
    schema.py                  # Schema, Slot; slot-id normalization; version checks
    stark_io.py                # export STaRK to our parquet/jsonl files (M0)
    text.py                    # normalization, sentence splitting, masking
    linker.py                  # dictionary linker (D6.3)
    facts.py                   # native keys and text facts (D6.1, D6.2)
    embed.py                   # Qwen3-Embedding wrapper: topic, fact ctx, segment vectors
    reference.py               # Appendix A, verbatim
    fast_scorer.py             # Appendix C, verbatim
    lexicon_build.py           # build/update the lexicon from facts; PACC (D8)
    slot_scorer.py             # cross-encoder wrapper, zero-shot wrapper, temperature (D9.3, D9.4)
    gen/                       # synthetic data generation (M5)
      fill_templates.py
      paraphrase_vllm.py       # runs in the slotspace-gen environment
      validate.py
    train_slot_scorer.py       # fine-tuning (M5)
    silver.py                  # silver labels from KB edges (D11.4)
    role_swap.py               # role-swap test set builder (D13.2)
    planner.py                 # posterior, conformal, weights, plan JSON (D9.5, D10)
    retrieve.py                # BM25, dense, postings, candidate pool, fusion, counters
    pipeline.py                # end-to-end search for one query
    evaluate.py                # STaRK evaluator + ranx per-query metrics, role confusion
    stats.py                   # paired tests, Holm correction, bootstrap, MDE
    experiments.py             # baselines B0–B7 and ablations A0–A14
    vespa_app/                 # M9 (v1)
  scripts/                     # thin CLI wrappers: python -m slotspace.<module> ...
  tests/
    test_reference.py          # Appendix B, verbatim
    test_fast_scorer.py        # Appendix C, verbatim
    test_schema.py  test_text.py  test_linker.py  test_facts.py  test_planner.py  test_stats.py
  data/                        # caches (git-ignored)
  runs/                        # run outputs (git-ignored)
  docs/
    DESIGN.md                  # copy of SlotSpace_DESIGN.md
    DEVELOPMENT.md             # this file
    DECISIONS.md               # every deviation and choice, dated
```

---

## 3. Configuration (`configs/prime_v0.yaml`)

```yaml
dataset: prime
schema: schemas/prime.yaml
seed: 13
paths: {data: data/prime, runs: runs}

embedding:
  model: Qwen/Qwen3-Embedding-0.6B
  max_seq_length: 512
  topic_dim: 256              # also run 1024 for B1
  ctx_dim: 128
  batch_size: 32
  segment_instruction: "Instruct: Given a constraint taken from a search query, retrieve statements that express the same relation\nQuery:"

linking:
  max_ngram: 8
  min_name_chars: 3
  common_words: 5000          # wordfreq top-n English words treated as non-entities (single words)
  fuzzy_threshold: 92         # rapidfuzz token_set_ratio, queries only
  fuzzy_mass: 0.5

facts:
  tau_min: 0.05
  max_sentence_words: 64
  max_text_facts_per_doc: 256

slot_scorer:
  model: answerdotai/ModernBERT-base
  zero_shot_model: BAAI/bge-reranker-v2-m3
  max_length: 128
  epochs: 2
  lr: 2.0e-5
  batch_size: 32
  warmup_ratio: 0.1
  samples_per_slot: 2000
  heldout_slot: drug.off_label_use.disease   # only for the A14 model

planner:
  beta: 0.5
  eps: 1.0
  alpha: 0.1
  r_max: 6
  weight_objective: p         # p | p_over_H | ia
  window_words: 8

retrieval:
  K_L: 200
  K_D: 200
  K_V: 500
  final_k: 100

scoring:
  mode: bound                 # bound | unbound | gate_only | no_role
  combine: max                # max | sum
  a: 1.0
  b: 1.0
  precision: float            # float | binary
  lambda_B: 1.0
  lambda_D: 1.0
  lambda_L: 0.5

tuning_grid:                  # searched on val only (M7)
  beta: [0.0, 0.25, 0.5, 1.0]
  alpha: [0.05, 0.1, 0.2]
  a: [0.5, 1.0]
  b: [0.5, 1.0, 2.0]
  lambda_B: [0.5, 1.0, 2.0]
  lambda_L: [0.0, 0.25, 0.5]
```

---

## 4. Data contracts

All files live under `data/prime/`. Types are pyarrow/pandas types. `doc_id` is the STaRK node index (int). A value id is the string `n<doc_id>`.

| File | Columns / content |
|---|---|
| `docs.parquet` | `doc_id` int64, `doc_type` str (normalized, e.g. `gene_protein`), `name` str, `text` str (`get_doc_info(idx, add_rel=True, compact=True)`), `details_json` str (the `details` dict as JSON, string values only) |
| `edges.parquet` | `src` int64, `dst` int64, `rel` str (exact STaRK name, e.g. `off-label use`) — every stored edge, both directions |
| `queries_<split>.jsonl` | `{"qid": int, "query": str, "answers": [int, ...]}` for `train`, `val`, `test`, `test-0.1`, `human` |
| `native_keys.parquet` | `doc_id` int64, `key` str (`slot|n<id>`) |
| `text_facts.parquet` | `row` int64 (row in the ctx matrix), `fact_id` str (`<doc_id>.t<j>`), `doc_id` int64, `field` str, `sentence` str, `ctx_text` str (all mentions masked), `mentions` list<struct{value_ids: list<str>, value_type: str, start: int, end: int, scorer_input: str}> |
| `text_fact_ctx_<model>_<dim>.npy` | float16 `[n_text_facts, 128]`, unit rows, aligned with `row` |
| `text_fact_keys_<scorer>.parquet` | `row` int64, `key` str, `pi` float32 (only `pi >= tau_min`) |
| `topic_<model>_<dim>.npy` | float16 `[n_docs, dim]`, unit rows, row = position in `docs.parquet` |
| `lexicon_<scorer>.parquet` | `value_id` str, `slot` str, `soft` float64, `hard` int64, `soft_pacc` float64, `evidence` list<str> (≤ 5 fact ids) |
| `silver_<split>.jsonl` | `{"qid", "segments": [{"start", "end", "links": {value_id: p}, "true_slots": [slot, ...]}]}` |
| `role_swap_<dev|test>.jsonl` | `{"qid": "rs-…", "query", "answers", "value_id", "slot", "template_id", "partners": [qid, ...], "partner_answers": [int, ...]}` |

Run outputs under `runs/<run_id>/`:

| File | Content |
|---|---|
| `meta.json` | config, git commit, versions, times |
| `run.json` | `{qid: {doc_id: score}}` top 100 per query |
| `per_query.parquet` | one row per query: every metric |
| `counters.parquet` | one row per query: the counters of D10.6 |
| `plans.jsonl` | the plan of every query (D10.1) |
| `metrics.json` | means over queries |

---

## 5. Milestones

Each milestone lists: goal, steps, code to write, tests, acceptance checks, pitfalls. Code blocks marked **(verbatim)** must be copied exactly. Other code blocks are skeletons: complete them, keep the names and signatures.

### M0 — Environment and data export (≈ 2 days)

**Goal.** Load STaRK-Prime once and export it to the files of Section 4, so that nothing else depends on the STaRK API.

**Facts you need (verified from the STaRK code, package `stark-qa`):**

- `from stark_qa import load_qa, load_skb`
- `load_skb(name, root=None, download_processed=True)` returns the knowledge base `skb`.
  - `skb.candidate_ids`: sorted list of candidate node ids. For Prime: all 129,375 nodes.
  - `skb.node_info[i]`: dict with `id`, `type`, `name`, `source` and an optional `details` dict.
  - `skb.edge_index`: LongTensor `[2, E]` (row 0 = source, row 1 = target); `skb.edge_types`: LongTensor `[E]`; `skb.edge_type_dict`: `{int: relation name}`; `skb.node_type_dict`: `{int: type name}`.
  - Edges are stored in both directions: E = 8,100,498 for Prime.
  - `skb.get_doc_info(idx, add_rel=False, compact=False, n_rel=-1)` (Prime) returns the document text. Always pass keyword arguments: the argument order differs between the three datasets.
- `load_qa(name, root=None, human_generated_eval=False)` returns the query set `qa`.
  - `qa.get_idx_split()` returns `{'train', 'val', 'test', 'test-0.1'}` → LongTensor of **positions**; use them as `qa[i]`.
  - `qa[i]` returns `(query: str, q_id: int, answer_ids: list[int], meta_info)`; `meta_info` is `None` (STaRK removed metadata to prevent answer leakage).
  - Human-written set: `load_qa(name, human_generated_eval=True).get_idx_split()` returns only `{'human_generated_eval': ...}`.
  - Answer ids are SKB node ids.

**Code: `src/slotspace/stark_io.py` (skeleton).**

```python
import json
import pathlib

import numpy as np
import pandas as pd
from stark_qa import load_qa, load_skb


def norm_name(s: str) -> str:
    """'gene/protein' -> 'gene_protein', 'off-label use' -> 'off_label_use'."""
    return s.strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")


def _clean_details(details) -> dict:
    out = {}
    for k, v in (details or {}).items():
        if k.startswith("_") or "_id" in k or v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        v = str(v).strip()
        if v and v.lower() != "nan":
            out[k] = v
    return out


def export_prime(out_dir: str = "data/prime", root: str = "data/stark_cache", limit: int | None = None) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    skb = load_skb("prime", root=root, download_processed=True)
    ids = list(skb.candidate_ids)[:limit] if limit else list(skb.candidate_ids)
    rows = []
    for i in ids:
        node = skb.node_info[i]
        rows.append({
            "doc_id": int(i),
            "doc_type": norm_name(node["type"]),
            "name": str(node["name"]),
            "text": skb.get_doc_info(i, add_rel=True, compact=True),
            "details_json": json.dumps(_clean_details(node.get("details")), ensure_ascii=False),
        })
    pd.DataFrame(rows).to_parquet(out / "docs.parquet", index=False)

    ei = skb.edge_index.numpy()
    et = skb.edge_types.numpy()
    rel_names = [skb.edge_type_dict[k] for k in range(len(skb.edge_type_dict))]
    edges = pd.DataFrame({"src": ei[0], "dst": ei[1], "rel": pd.Categorical.from_codes(et, categories=rel_names)})
    edges.to_parquet(out / "edges.parquet", index=False)

    qa = load_qa("prime", root=root)
    for split, idx in qa.get_idx_split().items():
        _write_queries(qa, idx.tolist(), out / f"queries_{split}.jsonl")
    human = load_qa("prime", root=root, human_generated_eval=True)
    _write_queries(human, human.get_idx_split()["human_generated_eval"].tolist(), out / "queries_human.jsonl")


def _write_queries(qa, positions, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in positions:
            query, qid, answers, _ = qa[i]
            f.write(json.dumps({"qid": int(qid), "query": query, "answers": [int(a) for a in answers]}) + "\n")
```

If `edge_type_dict` keys are not `0..17`, build `rel_names` by sorting its items by key and map codes accordingly.

**Acceptance checks** (write them as `tests/test_data_export.py`, marked slow):

| Check | Expected |
|---|---|
| rows in `docs.parquet` | 129,375 |
| rows in `edges.parquet` | 8,100,498 |
| distinct `rel` values | 18, exactly the names in D5.3 |
| per-type document counts | disease 17,080; gene_protein 27,671; molecular_function 11,169; drug 7,957; pathway 2,516; anatomy 14,035; effect_phenotype 15,311; biological_process 28,642; cellular_component 4,176; exposure 818 |
| queries per split | train 6,162; val 2,241; test 2,801; test-0.1 280; human 98 |
| every answer id is a `doc_id` | yes |
| for every edge (s, r, d) the edge (d, r, s) exists | yes (check on a 100,000-edge sample) |

**Pitfalls.** `stark-qa` supports Python < 3.12. The relation names contain spaces and hyphens; keep the exact names in `edges.parquet` and normalize only when building slot ids. Hugging Face downloads can be slow; set `root` to a disk with space.

**Human checkpoint H0:** send the table of counts.

---

### M1 — Baselines, evaluation and counters (≈ 4 days)

**Goal.** BM25, dense and hybrid baselines with the official evaluator, our per-query metrics and the cost counters. These are B0–B2, and the hybrid is also the fallback path of SlotSpace.

**BM25 (`retrieve.py`).** Use `bm25s` with English stopwords and the PyStemmer English stemmer, over `docs.parquet:text`.

```python
import bm25s
import Stemmer

class BM25Index:
    def __init__(self, texts: list[str]):
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = bm25s.BM25()
        self.retriever.index(bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer))

    def search(self, queries: list[str], k: int):
        tokens = bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer)
        positions, scores = self.retriever.retrieve(tokens, k=k)   # both shape (n_queries, k)
        return positions, scores                                    # positions index into `texts`
```

For fusion SlotSpace needs a BM25 score for every candidate, not only the BM25 top-K. If your `bm25s` version offers a per-query score vector over all documents (check `help(bm25s.BM25)` for a `get_scores`-like method), use it. Otherwise retrieve `k = 1000` and give 0 to documents outside that list. Record the choice in `docs/DECISIONS.md`.

`bm25_postings` counter: sum of document frequencies of the distinct query tokens (compute document frequencies once from the tokenized corpus).

**Dense (`embed.py`).**

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, name: str, max_seq_length: int = 512, device: str = "cuda"):
        self.model = SentenceTransformer(name, device=device)
        self.model.max_seq_length = max_seq_length

    @staticmethod
    def _cut(e: np.ndarray, dim: int) -> np.ndarray:
        e = e[:, :dim].astype(np.float32)               # Matryoshka truncation
        e /= np.linalg.norm(e, axis=1, keepdims=True)   # re-normalize AFTER truncation
        return e

    def documents(self, texts, dim, batch_size=32):
        e = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        return self._cut(e, dim).astype(np.float16)

    def queries(self, texts, dim, prompt: str | None = None, batch_size=32):
        kw = {"prompt": prompt} if prompt else {"prompt_name": "query"}
        e = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, **kw)
        return self._cut(e, dim)
```

- Documents: no prompt. Queries: the model's built-in `query` prompt. Segment vectors (M6): the custom `segment_instruction` from the config passed as `prompt`; it must end with `Query:` exactly as in the config.
- Encode all documents once at 1024 dimensions, save `topic_qwen3e06_1024.npy`, and derive the 256-dimensional file by truncating and re-normalizing (no second pass).
- Exact search: keep the document matrix as float32 in memory and compute `Q @ D.T` in blocks of 64 queries; take the top-K with `np.argpartition`, then sort.

**Evaluation (`evaluate.py`).**

- Official numbers: `from stark_qa.evaluator import Evaluator`; `ev = Evaluator(skb.candidate_ids)`; per query `ev(pred_dict, torch.LongTensor(answers), metrics=["hit@1", "hit@5", "recall@20", "recall@100", "mrr", "rprecision"])` where `pred_dict` maps doc id → score for the top 100. Unscored candidates are placed below all scored ones. Report means × 100.
- Per-query vectors for statistics: compute them ourselves from the top-100 list (definitions below). They MUST agree with the official means within 0.1 points (MRR may differ slightly because ours stops at rank 100).

```python
def per_query_metrics(ranked: list[int], answers: set[int]) -> dict:
    first = next((i for i, d in enumerate(ranked) if d in answers), None)
    def recall(k): return len(set(ranked[:k]) & answers) / len(answers)
    return {
        "hit@1": float(first is not None and first < 1),
        "hit@5": float(first is not None and first < 5),
        "recall@20": recall(20),
        "recall@100": recall(100),
        "mrr": 0.0 if first is None else 1.0 / (first + 1),
        "rprecision": len(set(ranked[:len(answers)]) & answers) / len(answers),
    }
```

**Hybrid (B2).** Candidate pool = BM25 top-200 ∪ dense top-200. Score = `lambda_D * minmax(dense) + lambda_L * minmax(bm25)` over the pool, where `minmax` is computed per query over the pool (if max = min, use 0). Tune `lambda_L ∈ {0, 0.25, 0.5, 1}` with `lambda_D = 1` on `val`.

**Run all three on `val`** and save runs in the format of Section 4.

**Acceptance checks.**

- Official and per-query means agree (within 0.1 points) for every metric except MRR (within 0.5).
- B0 on Prime `val` is in the same range as the STaRK paper's BM25 on Prime `test` (Hit@1 12.75, Recall@20 31.25). A difference of more than 10 points means a bug (wrong id mapping is the usual cause).
- Every run directory has the six files of Section 4, and `counters.parquet` has `bm25_postings`, `dense_searches`, `candidates`, `latency_ms`.

**Pitfalls.** `bm25s` returns positions in the corpus list, not doc ids; map them through `docs.parquet`. Do not pass `normalize_embeddings=False` and forget to normalize. Use float32 for the matrix product.

---

### M2 — Schema, native keys, linker and text facts (≈ 5 days)

**Goal.** Everything on the document side except the slot probabilities: the schema file, native keys, the dictionary linker, sentences, masking, and context vectors.

**Step 1 — derive slots (`schema.py`).**

```python
edges = pd.read_parquet("data/prime/edges.parquet")
types = pd.read_parquet("data/prime/docs.parquet").set_index("doc_id")["doc_type"]
e = edges.assign(st=edges.src.map(types), dt=edges.dst.map(types))
counts = e.groupby(["st", "rel", "dt"], observed=True).size().reset_index(name="n_edges")
counts["slot_id"] = counts.st + "." + counts.rel.astype(str).map(norm_name) + "." + counts.dt
counts["direction"] = np.where(counts.st == counts.dt, "sym", "out")
```

Write `schemas/prime.yaml` with one entry per row: `id`, `doc_type`, `relation` (exact name), `value_type`, `direction`, `n_edges`, and empty `description`, `query_cues`, `doc_cues`. Then write every `description` by hand in plain English from the document's point of view (style: D5.3), and 3–6 cues per style for each slot. Add the `none` entries (one per value type) and `conventions` (D5.1). Set `version: 0.1.0`.

`schema.py` provides: `Schema.load(path)`, `schema.slots` (ordered list), `schema.slot(id)`, `schema.candidates(doc_type: str | None, value_types: set[str]) -> list[str]` (slot ids plus `none.<type>` for each type), `schema.slot_text(id)` (`"<id>: <description>"`), `schema.version`. Test: loading fails if two slots share an id, if an id contains `|`, or if a description is empty.

**Step 2 — native keys (`facts.py`).** For every stored edge `(src, rel, dst)`: key `f"{type(src)}.{norm_name(rel)}.{type(dst)}|n{dst}"` for document `src`. Because every edge is stored in both directions, this one loop gives both ends their keys. Write `native_keys.parquet`. Check: the number of rows equals the number of edges (8,100,498) minus exact duplicates.

**Step 3 — linker (`linker.py`).**

```python
import re
import string
import unicodedata
from collections import defaultdict

from wordfreq import top_n_list

_WORD = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9'\-/\.]*[A-Za-z0-9])?")   # tokens end with a letter or digit

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(string.punctuation + " ")

class Linker:
    def __init__(self, docs_df, max_ngram=8, min_chars=3, common_words=5000):
        self.max_ngram = max_ngram
        common = set(top_n_list("en", common_words))
        self.norm_index = defaultdict(set)   # normalized name -> {value_id}
        self.case_index = defaultdict(set)   # exact all-caps symbol that is also a common word -> {value_id}
        self.vtype = {}
        for doc_id, name, doc_type in docs_df[["doc_id", "name", "doc_type"]].itertuples(index=False):
            vid = f"n{doc_id}"
            self.vtype[vid] = doc_type
            n = normalize(name)
            if len(n) < min_chars or n.replace(" ", "").isdigit():
                continue
            if " " not in n and n in common:
                if name.isupper():
                    self.case_index[name].add(vid)
                continue
            self.norm_index[n].add(vid)

    def link(self, text: str) -> list[dict]:
        """Longest-match-first, left to right, no overlaps.
        Returns [{"start", "end", "surface", "values": {value_id: within_type_prob}, "types": set}]."""
        words = list(_WORD.finditer(text))
        out, i = [], 0
        while i < len(words):
            hit = None
            for n in range(min(self.max_ngram, len(words) - i), 0, -1):
                start, end = words[i].start(), words[i + n - 1].end()
                surface = text[start:end]
                ids = self.norm_index.get(normalize(surface)) or self.case_index.get(surface)
                if ids:
                    hit = (n, start, end, surface, ids)
                    break
            if hit is None:
                i += 1
                continue
            n, start, end, surface, ids = hit
            by_type = defaultdict(list)
            for v in sorted(ids):
                by_type[self.vtype[v]].append(v)
            values = {v: 1.0 / len(vs) for vs in by_type.values() for v in vs}
            out.append({"start": start, "end": end, "surface": surface, "values": values, "types": set(by_type)})
            i += n
        return out
```

Query-only fuzzy fallback, exposed as `link_query(text)` (= `link(text)` plus this fallback): for word spans of 2–6 words between exact matches, find candidate names through an inverted index (word → names containing it, at most 5,000 candidates), then `rapidfuzz.process.extractOne(normalize(span), candidates, scorer=rapidfuzz.fuzz.token_set_ratio, score_cutoff=92)`. Fuzzy values get half the probability mass of an exact match at the same position (`fuzzy_mass`).

**Step 4 — sentences and masking (`text.py`).**

```python
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

def split_sentences(text: str, max_words: int = 64) -> list[str]:
    out = []
    for s in _SPLIT.split(text):
        parts = [p.strip() for p in s.split(";")] if len(s.split()) > max_words else [s.strip()]
        for p in parts:
            w = p.split()
            out.extend(" ".join(w[j:j + max_words]) for j in range(0, len(w), max_words))
    return [s for s in out if s]

def mask(sentence: str, mentions: list[dict], target: int | None, self_type: str, field: str | None) -> str:
    """Replace mentions right-to-left. target index -> [TARGET:t] (or [TARGET] if several types);
    mentions of the document itself -> [SELF:self_type]; other mentions -> [t] (or [entity])."""
    s = sentence
    for j in sorted(range(len(mentions)), key=lambda j: -mentions[j]["start"]):
        m = mentions[j]
        t = next(iter(m["types"])) if len(m["types"]) == 1 else None
        if m.get("is_self"):
            tok = f"[SELF:{self_type}]"
        elif j == target:
            tok = f"[TARGET:{t}]" if t else "[TARGET]"
        else:
            tok = f"[{t}]" if t else "[entity]"
        s = s[: m["start"]] + tok + s[m["end"]:]
    return f"field: {field} | {s}" if field else s
```

Use `field=<field name>` for `ctx_text` (context vectors) and `field=None` for `scorer_input` (the slot scorer never sees field names; D6.2).

**Step 5 — text facts (`facts.py`).** For each document, for each `details` field (`details_json`), for each sentence:

1. `mentions = linker.link(sentence)`; mark a mention `is_self` if its values contain the document's own id.
2. Drop non-self mentions whose values contain **several ids of the same type** (document-side ambiguity; count them).
3. If no non-self mention remains, skip the sentence.
4. Create a text fact: `fact_id = f"{doc_id}.t{j}"`, `ctx_text = mask(sentence, mentions, target=None, self_type=doc_type, field=field)`, and for every non-self mention a `scorer_input = mask(sentence, mentions, target=index, self_type=doc_type, field=None)`.
5. Stop at `max_text_facts_per_doc` facts per document.

Write `text_facts.parquet`. Then embed all `ctx_text` values with `Embedder.documents(..., dim=1024)`, truncate to 128 and re-normalize, and save `text_fact_ctx_qwen3e06_128.npy` (float16).

**Tests (`tests/test_text.py`, `tests/test_linker.py`, `tests/test_facts.py`).** At minimum:

- `normalize("  Hypertension. ")` → `"hypertension"`.
- A toy linker over names {"hypertension" (disease n1), "hypertension" (effect_phenotype n2), "pulmonary hypertension" (disease n3), "CAT" (gene_protein n4)}: `"Pulmonary hypertension and hypertension"` gives two mentions, the first is n3 only (longest match), the second has values {n1: 1.0, n2: 1.0} with types {disease, effect_phenotype}. `"the cat sat"` gives no mention; `"CAT activity"` gives n4 (if "cat" is in the common-word list).
- `mask("Do not use Aspirin in asthma.", ...)` with Aspirin = self (drug), asthma = target disease and `field="description"` gives `"field: description | Do not use [SELF:drug] in [TARGET:disease]."`; with `target=None` it gives `"field: description | Do not use [SELF:drug] in [disease]."`; with `field=None` the prefix is absent.
- `split_sentences("Used e.g. for pain. It works well.")` → `["Used e.g. for pain.", "It works well."]`.

(The linker, masking and splitting code above was run on these examples while this guide was written; the outputs above are the observed ones.)

**Acceptance checks.** Report in `runs/m2_stats.json`: number of slots; text facts per document (mean, p50, p95, max) by document type; documents hitting the cap; mentions per fact; share of mentions skipped for same-type ambiguity; linked mentions per query on `val` (share of queries with ≥ 1 segment; target ≥ 0.8).

**Human checkpoint H1:** a person reviews 100 random linked query mentions and 100 random document mentions (precision), and the hand-written slot descriptions.

**Pitfalls.** Gene symbols and short names cause most false links; prefer precision. Do not link inside the field name prefix. Keep character offsets relative to the original sentence.

---

### M3 — Reference scorer, fast scorer, lexicon (≈ 3 days)

**Goal.** The scoring math and the lexicon, tested against expected values, before any model exists.

**Steps.**

1. Copy Appendix A to `src/slotspace/reference.py` and Appendix B to `tests/test_reference.py` (imports already use the package path `slotspace.reference`). Run `pytest -q tests/test_reference.py`. All 12 tests MUST pass.
2. Copy Appendix C.1 to `src/slotspace/fast_scorer.py` and C.2 to `tests/test_fast_scorer.py`. Both parity tests (float and binary context vectors) MUST pass.
3. Run the synthetic binding check (Appendix C.3, `python -m slotspace.sim_binding`). Expected output for the default setting *(reference run, seed 7)*: `bound` R-precision 0.835, `unbound` 0.483, `gate_only` 0.436, `no_role` 0.880. Acceptance: bound / unbound ≥ 1.4 and bound − gate_only ≥ 0.2 (exact numbers may move slightly across numpy versions).
4. `lexicon_build.py`:
   - Build `Doc` objects from `native_keys.parquet` and `text_fact_keys_<scorer>.parquet`. Until the fine-tuned scorer exists (M5), produce `text_fact_keys_zeroshot.parquet` with the zero-shot scorer (M5 step 6) so the pipeline can run end to end.
   - Fill a `Lexicon` with `upsert` for every document. Save `lexicon_<scorer>.parquet` with `soft`, `hard` and up to 5 evidence fact ids per (value, slot).
   - PACC (D8.6): build the distant-supervision set (text mentions in document $d$ of value $v$ where the KB has an edge between $d$ and $v$; the true slot set = the native slots of `k|v` in $d$), measure TPR and FPR per slot with `c_d(v,k)` from text facts only, and fill `soft_pacc`.
5. Incremental test: shuffle documents, build the lexicon in two halves with `upsert`, delete 1,000 random documents, re-insert them; the snapshot MUST equal a fresh build (I4).

**Acceptance checks.** All tests green; lexicon size (number of (value, slot) pairs) and the top 20 collision values (most slots with $H \ge 3$) written to `runs/m3_lexicon_stats.json`; PACC rates per slot reported.

**Pitfalls.** Use the reference only for tests and small runs; use the fast scorer in pipelines. Keep float64 in the reference; float32 context vectors are fine in the fast scorer.

---

### M4 — Silver labels and the role-swap test set (≈ 3 days)

**Goal.** Labels for calibration and selection (from `train` and `val`), and the KB-grounded role-swap dev/test sets (D11.4, D13.2).

**Silver labels (`silver.py`).**

```python
import math

def silver_segments(query: dict, linker, schema, native_by_doc: dict[int, set[str]], doc_type: dict[int, str]) -> list[dict]:
    """For one train/val query: link its mentions and attach the true slots from the KB edges of its answers."""
    answers = query["answers"]
    need = max(1, math.ceil(0.5 * len(answers)))
    out = []
    for m in linker.link_query(query["query"]):          # exact + fuzzy (M2)
        true = set()
        for v in m["values"]:
            counts = {}
            for a in answers:
                for k in schema.candidates(doc_type[a], {linker.vtype[v]}):
                    if f"{k}|{v}" in native_by_doc.get(a, ()):
                        counts[k] = counts.get(k, 0) + 1
            true |= {k + "|" + v for k, c in counts.items() if c >= need}
        out.append({"start": m["start"], "end": m["end"], "links": m["values"], "true_readings": sorted(true)})
    return out
```

- `true_readings` holds full keys (`slot|value`), because the true reading also fixes the value when a name is shared.
- Write `silver_train.jsonl` and `silver_val.jsonl`. Split `silver_train` by `qid` parity: even → calibration (temperature, conformal threshold), odd → spare. `silver_val` is the selection set for plan regret.
- Report: share of segments with a non-empty label, label distribution over slots, number of segments with more than one true reading.

**Human checkpoint H2:** a person checks 100 random silver segments (is the label what the query means?) and records the agreement rate in `docs/DECISIONS.md`. If agreement is below 80%, stop and discuss.

**Role-swap sets (`role_swap.py`).** Implement D13.2 exactly:

```python
import hashlib

DRUG_SLOTS = ["drug.indication.disease", "drug.contraindication.disease", "drug.off_label_use.disease"]

def h(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)

def eligible(H: int, lo: int = 3, hi: int = 100) -> bool:
    return lo <= H <= hi
```

1. Native hard support per (value, slot) comes from the lexicon built **from native keys only** (so it does not depend on any model).
2. Same-type collisions: disease values with ≥ 2 eligible slots among `DRUG_SLOTS`.
3. Cross-type collisions: normalized names that belong to a disease node with `drug.indication.disease` eligible **and** to an effect/phenotype node with `drug.side_effect.effect_phenotype` eligible.
4. For each (value, eligible slot): template index = `h(f"{value_id}|{slot}") % len(templates[slot])`; query = template with `{V}` replaced by the node's name; answers = documents whose `native_keys` contain `slot|value`.
5. Partners: all other queries built for the same value (same-type) or the same normalized name (cross-type). Store their ids and the union of their answers.
6. Split: `h(value_id) % 2` (cross-type: `h(normalized_name) % 2`); 0 → dev, 1 → test.
7. Templates come only from `templates/test/role_swap.yaml` (Appendix D). The file is read by `role_swap.py` and by nothing else; add a test that greps the code base and fails if any other module opens it.

**Acceptance checks.** Number of dev and test queries per slot; MDE for the test size (`stats.mde_points(n, 0.2)`); if n_test < 600 apply the fallback of D13.2 step 6. **Human checkpoint H3:** a person reads 50 random role-swap queries.

---

### M5 — The slot scorer (≈ 7 days)

**Goal.** A calibrated slot scorer (D9.3, D11) trained only on schema-generated, value-masked data; plus the zero-shot scorer for A13.

**Step 1 — training templates.** In `templates/train/prime_slots.yaml`, for every slot: `query:` (≥ 12 templates) and `doc:` (≥ 12 templates); for every value type a `none` entry (≥ 12 per style). Placeholders: `{T}` target, `{O:<type>}` other mention, `{SELF}` the document entity (doc style), `{ANS}` answer phrase (query style). Example entries:

```yaml
drug.indication.disease:
  query:
    - "Which {ANS} are used to treat {T}?"
    - "What can be prescribed for {T}?"
    - "List {ANS} approved for {T} that also act on {O:gene_protein}."
  doc:
    - "{SELF} is indicated for the treatment of {T}."
    - "{SELF} is used to manage {T} and {O:disease}."
drug.contraindication.disease:
  query:
    - "Which {ANS} should not be given to people with {T}?"
    - "What must be avoided in patients who have {T}?"
  doc:
    - "{SELF} is contraindicated in patients with {T}."
    - "Do not use {SELF} if the patient has {T}."
none.disease:
  query:
    - "Tell me about {ANS} studied in people with {T}."
  doc:
    - "{T} was reported in a small number of patients during trials of {SELF}."
```

Answer phrases per document type (`{ANS}`): drug → "drugs", "medications", "medicines", "compounds"; disease → "diseases", "conditions", "disorders"; gene_protein → "genes", "proteins"; and so on for every type. The `none` templates must not contain any slot cue of that value type.

**Step 2 — fill with surrogates (`gen/fill_templates.py`).** For each slot and style, sample templates uniformly and fill every placeholder with a fresh pseudo-word (e.g. `Zorvex17`: two random consonant-vowel syllables, a capital letter, two digits; unique within the sample). Keep a map pseudo-word → role (`target`, `other:<type>`, `self`). `{ANS}` is filled directly with a phrase of the slot's `doc_type` (`none`: any type). Output JSONL: `sample_id, slot, value_type, style, template_id, text, tokens`.

**Step 3 — paraphrase (`gen/paraphrase_vllm.py`, environment `slotspace-gen`).**

```python
from vllm import LLM, SamplingParams

PROMPT = ("Rewrite the sentence below in 3 different ways. Keep its meaning. "
          "Keep each of these words exactly as written, each exactly once: {tokens}. "
          "Vary the wording and the sentence structure. "
          "Return one rewrite per line, with no numbering and no extra text.\n\nSentence: {text}")

llm = LLM(model="Qwen/Qwen3-4B-Instruct-2507", max_model_len=4096)
params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=256)
outputs = llm.chat([[{"role": "user", "content": PROMPT.format(tokens=", ".join(s["tokens"]), text=s["text"])}] for s in samples], params)
rewrites = [o.outputs[0].text.strip().split("\n") for o in outputs]   # outputs keep the input order
```

If your vLLM version does not accept `max_model_len` in the constructor, check `help(LLM)` and record the change.

**Step 4 — validate and mask (`gen/validate.py`).** Keep the original filled sentence plus valid rewrites. Reject a text if: a pseudo-word is missing or occurs twice; fewer than 4 or more than 60 words; contains a cue phrase (from the schema's `query_cues`/`doc_cues`) of another slot of the same value type; duplicate after normalization. Then replace pseudo-words: target → `[TARGET:<type>]` (80%) or `[TARGET]` (20%); others → `[<type>]`; self → `[SELF:<doc_type>]`. Balance to `samples_per_slot` per slot and style (down-sample; if short, up-sample with replacement and log it).

**Step 5 — pairs and training (`train_slot_scorer.py`).**

- Candidates of a sample: query style → all slots with the sample's value type (any doc type) + `none.<type>`; doc style → slots with the SELF doc type and the value type + `none.<type>`. For `[TARGET]` samples whose type is `disease` or `effect_phenotype`, add the other type's candidates too (these two types share many names in Prime).
- One row per (sample, candidate): columns `context` (masked text), `slot_text` (`schema.slot_text(candidate)`), `label` (1.0 for the true slot, else 0.0). Column order matters: inputs are matched by position, the label by its name.
- Hold out 10% of **template ids** per slot as the dev set.

```python
import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

model = CrossEncoder("answerdotai/ModernBERT-base", num_labels=1, max_length=128)
train = Dataset.from_dict({"context": ctx, "slot_text": slot_txt, "label": labels})
loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(neg_per_pos))
args = CrossEncoderTrainingArguments(
    output_dir="data/prime/models/slot_scorer_ft", num_train_epochs=2, per_device_train_batch_size=32,
    learning_rate=2e-5, warmup_ratio=0.1, bf16=True, eval_strategy="steps", eval_steps=1000,
    save_steps=1000, load_best_model_at_end=True, metric_for_best_model="eval_loss", seed=13)
CrossEncoderTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev, loss=loss).train()
model.save_pretrained("data/prime/models/slot_scorer_ft/final")
```

If your version rejects `warmup_ratio`, use `warmup_steps=0.1`. Train a second model with the `heldout_slot` removed from the training rows (A14), and a third with values visible (A12: fill the target with a real KB name of the right type instead of the mask, same templates).

**Step 6 — inference wrapper (`slot_scorer.py`).**

```python
import numpy as np
import torch
from sentence_transformers.cross_encoder import CrossEncoder

class SlotScorer:
    def __init__(self, path: str, schema, temperature: float = 1.0, pair_order: str = "context_first"):
        self.model, self.schema, self.T, self.order = CrossEncoder(path), schema, temperature, pair_order

    def logits(self, contexts: list[str], candidates: list[list[str]]) -> list[np.ndarray]:
        pairs, owner = [], []
        for i, (c, cands) in enumerate(zip(contexts, candidates)):
            for k in cands:
                pairs.append((c, self.schema.slot_text(k)) if self.order == "context_first" else (self.schema.slot_text(k), c))
                owner.append(i)
        z = self.model.predict(pairs, batch_size=256, activation_fn=torch.nn.Identity(), convert_to_numpy=True)
        out, pos = [], 0
        for cands in candidates:
            out.append(np.asarray(z[pos:pos + len(cands)], dtype=np.float64)); pos += len(cands)
        return out

    def score(self, contexts, candidates, temperature=None) -> list[dict]:
        T = temperature or self.T
        res = []
        for z, cands in zip(self.logits(contexts, candidates), candidates):
            p = np.exp((z - z.max()) / T); p /= p.sum()
            res.append(dict(zip(cands, p.tolist())))
        return res
```

- If `predict` has no `activation_fn` argument in your version, it returns sigmoid probabilities $q$; convert them to logits with $\log(q/(1-q))$ (clip $q$ to $[10^{-6}, 1-10^{-6}]$).
- Zero-shot scorer (A13): `SlotScorer("BAAI/bge-reranker-v2-m3", schema, pair_order="slot_first")` — a reranker expects (query, passage), so the slot text goes first.

**Step 7 — temperature and ECE.**

```python
from scipy.optimize import minimize_scalar

def fit_temperature(logit_list, true_index) -> float:
    def nll(logT):
        T, tot = np.exp(logT), 0.0
        for z, y in zip(logit_list, true_index):
            a = z / T
            tot -= a[y] - (a.max() + np.log(np.exp(a - a.max()).sum()))
        return tot / len(true_index)
    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded").x))

def ece(conf, correct, bins: int = 15) -> float:
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)
```

- $T_{query}$: on the calibration half of `silver_train`, using each segment's best true reading's slot as the label (segments with an empty label use `none`).
- $T_{doc}$: on the distant-supervision set of M3.
- Save both in `data/prime/models/slot_scorer_ft/temperature.json`.

**Step 8 — score document mentions.** Run the fine-tuned scorer (style `doc`, $T_{doc}$) on every `scorer_input` of `text_facts.parquet`; write `text_fact_keys_ft.parquet` (only $\pi \ge 0.05$); rebuild the lexicon (M3 step 4) as `lexicon_ft.parquet`.

**Acceptance checks.**

| Check | Target |
|---|---|
| accuracy on held-out templates (synthetic dev) | ≥ 0.90 (sanity only) |
| slot accuracy on `silver_val` (query style), fine-tuned vs zero-shot | report both; fine-tuned should be higher |
| ECE after temperature scaling on `silver_val` | ≤ 0.05 |
| A12 and A14 models trained and saved | yes |

**Human checkpoint H4:** send the table above and 30 random scored mentions (doc style) with their top-3 slots.

**Pitfalls.** Never put real KB names into training data for the main model. Never read `templates/test/`. Cross-encoders overfit quickly: keep the best checkpoint by dev loss.

---

### M6 — The planner (≈ 3 days)

**Goal.** `planner.py` turns a query into a plan (D9.5, D10.1–10.4).

```python
def plan_query(query: str, qid, ctx) -> dict:
    """ctx bundles: schema, linker, embedder, slot scorer (+ temperature), lexicon, config, t_alpha."""
    segments = []
    for m in ctx.linker.link_query(query):
        types = m["types"]
        cands = ctx.schema.candidates(None, types)                      # all doc types + none.<type>
        x = mask(query, [m], target=0, self_type=None, field=None)       # other mentions masked too, see M2
        C = ctx.scorer.score([x], [cands])[0]                            # joint distribution incl. none
        c_none = sum(p for k, p in C.items() if k.startswith("none."))
        C_by_type = {}
        for k, p in C.items():
            if not k.startswith("none."):
                C_by_type.setdefault(ctx.schema.slot(k).value_type, {})[k] = p
        links = {v: (ctx.linker.vtype[v], lp) for v, lp in m["values"].items()}
        post = reading_posterior(C_by_type, links, ctx.lexicon, beta=ctx.cfg.beta, eps=ctx.cfg.eps)
        kept = keep_readings(post, ctx.t_alpha, min_keep=1, r_max=ctx.cfg.r_max)
        weights = {k: (1.0 - c_none) * p for k, p in kept.items()}
        g = ctx.embedder.queries([window(x, ctx.cfg.window_words)], dim=128, prompt=ctx.cfg.segment_instruction)[0]
        segments.append({"span": [m["start"], m["end"]], "text": m["surface"], "links": m["values"], "c_none": c_none,
                         "readings": [{"key": k, "p": post[k], "w": w, "support_hard": ctx.lexicon.hard_support(k.split("|")[1], k.split("|")[0])} for k, w in weights.items()],
                         "pruned_conformal": sorted(set(post) - set(kept)), "g": g})
    path = "both" if any(s["readings"] for s in segments) else "vector_first"
    return {"query_id": qid, "segments": segments, "path": path, "weight_objective": ctx.cfg.weight_objective}
```

Notes:

- When masking the query for a segment, mask **all** linked mentions: the target as `[TARGET:...]`, the others as `[type]` (same `mask` function as M2, with the full mention list and `target` = this segment's index). The snippet above shows the call shape only.
- `window(x, W)` keeps W words on each side of the target token.
- `reading_posterior` and `keep_readings` come from `reference.py` (they are fast enough).
- `weight_objective = p_over_H`: apply the conformal threshold to `p` as usual (the threshold was calibrated on `p`), then set the kept readings' weights proportional to `p / max(1, H)` and renormalize them to `1 - c_none`. `ia` is applied after ranking (M7).
- Zero-support readings are removed inside `reading_posterior`; log them as `pruned_zero_support` by recomputing which (slot, value) pairs had `H == 0`.

**Conformal calibration (`planner.calibrate_conformal`).** For each segment of the calibration half of `silver_train` with a non-empty label: `p_true = max(post.get(key, 0) for key in true_readings)`. Compute `t_alpha` for α ∈ {0.05, 0.1, 0.2} with `conformal_threshold`. Save `data/prime/conformal_ft.json` (and one per scorer variant).

**Tests (`tests/test_planner.py`).** A fake scorer returning the joint distribution {ind 0.135, con 0.72, off 0.045, side 0.1, none 0} with within-type links of 1.0 and the corpus lexicon of Appendix B MUST give the posterior of D7.5 (0.6254, 0.2014, 0.1732); with `c_none = 0.2` the weights MUST sum to 0.8; a query without mentions MUST give `path = "vector_first"`; the plan MUST serialize to JSON (convert `g` to a list or drop it when logging).

---

### M7 — End-to-end runs on `val` and tuning (≈ 5 days)

**Goal.** The complete SlotSpace search (D10.8), cost counters, plan regret, and tuned hyperparameters — all on `val`.

**Pipeline (`pipeline.py`).**

1. `plan = plan_query(q)`.
2. Candidates: BM25 top-`K_L` ∪ dense top-`K_D` ∪ postings of kept keys. Postings: a dict `key → np.array(doc ids)` built from `native_keys.parquet` and `text_fact_keys_<scorer>.parquet`. If the union of postings is larger than `K_V`, keep the `K_V` with the highest dense score.
3. Readings for scoring: one `Reading` per kept key with `g` from its segment and `weight` from the plan.
4. `B = FactStore.scores(candidates, readings, cfg)`; dense and BM25 scores for all candidates; fusion (D7.1) with per-query min–max over the pool and $B / \max(m, 1)$.
5. Top 100; for `ia`, re-order them with `ia_select` using $V_r(d) = M_r(d)$ (add a method to `FactStore` that returns per-reading scores).
6. Evidence: for each of the top 10 documents and each reading, the fact id with the largest term.
7. Counters (D10.6) and latency per stage.

**Feature cache for tuning.** $M_r(d)$ does not depend on the weights. For each `val` query store: candidate ids, dense and BM25 scores, and $M_r(d)$ for **all** readings with $H > 0$ (before conformal pruning), for each $(a, b)$ in the grid. Then every $(\beta, \alpha, \lambda)$ setting is a cheap re-weighting.

**Plan regret.** For each `val` query with silver labels: oracle readings = the silver true readings with equal weights (sum 1 per segment); run the same fusion; regret = RR(oracle) − RR(plan). Report the mean and a 95% bootstrap interval.

**Tuning order** (record every result table in `runs/m7_tuning/`):

1. $\beta$ and $\alpha$ by mean plan regret (defaults for the rest).
2. $a$ and $b$ by `val` MRR.
3. $\lambda_B$ and $\lambda_L$ by `val` MRR.
4. Weight objective (`p`, `p_over_H`, `ia`) by `val` MRR and Recall@100 (A9 reports all three).

**Acceptance checks.** SlotSpace (A0) on `val` is not worse than B2 (hybrid) on Recall@100 by more than 1 point; the cost counters are filled for every query; the conformal miss rate on `silver_val` is ≤ α + 0.02; `configs/prime_v0_frozen.yaml` written with the chosen values and its SHA-256 recorded.

**Human checkpoint H5:** approve the frozen configuration before any test run.

---

### M8 — Experiments, statistics and the report (≈ 5 days)

**Goal.** One frozen run of every baseline and ablation on the test sets, statistics, and the report of D13.7.

**Runs.**

- Sets: Prime `test` (2,801), human-generated (98), role-swap `test`. B7 (oracle) on `val` only.
- Baselines B0–B6 (D13.4). B3 multi-vector chunks: dense max over the document's sentence vectors (the 128-d `ctx` vectors are **not** used — embed sentences with the same 256-d topic setting), fused with BM25. B4 field-weighted: fields = name, details, and one field per relation listing (names of neighbors per relation); one dense and one BM25 score per field over the candidate pool; field weights by coordinate ascent on `val` MRR.
- B6 fan-out: for each kept reading: `rewrite = f"{query} ({description_with_value})"` where `description_with_value` is the slot description with "the <value type>" replaced by the value's name (if no such phrase, append `" Value: <name>."`); run B2 for each rewrite; merge with weighted reciprocal rank fusion `score(d) = Σ_r w_r / (60 + rank_r(d))`; sum the counters over rewrites.
- Ablations A0–A14 (D13.5); each changes exactly one setting of the frozen configuration. Implementation notes:
  - A1–A3: `scoring.mode`; A4: `planner.beta = 0`; A5: replace the scorer output by a uniform distribution over the candidate slots (no `none`); A6: after scoring, drop candidates with $M_r(d) = 0$ for every reading of some segment.
  - A7: rebuild text facts from 128-token windows (stride 128) instead of sentences, re-embed them, re-score their mentions, rebuild the lexicon; everything else unchanged.
  - A8: use `soft` instead of `soft_pacc` in the posterior.
  - A10: `planner.alpha` ∈ {0.05, 0.1, 0.2} and "keep all" (`t = 0`).
  - A11: build the `FactStore` with `precision="binary"` and set `scoring.precision: binary` (Appendix C.1).
  - A12, A13, A14: swap the slot scorer (M5) and its temperature and conformal files; rebuild the document-side keys and the lexicon with the same scorer.

**Statistics (`stats.py`, Appendix E).** For each table: per-query vectors → paired t-test and randomization test against the reference system (A0 for ablations, A0 vs each baseline for Table 1) → Holm correction within the table → bootstrap intervals → measured discordant rate and MDE.

**Report.** `runs/report_v0.md` with Tables 1–4 and Figures 1–2 (D13.7), generated by `python -m slotspace.experiments report`. Include the negative-results paragraph.

**Acceptance checks.** Every pass line of D13.5 evaluated and marked PASS / FAIL / NOT DETECTED; the report regenerates from the run files alone.

**Human checkpoint H6:** review the report.
---

### M9 — Vespa engine path (v1, ≈ 3 weeks)

**Goal.** The same binding score inside Vespa's second phase (D12), checked by a smoke test and a parity test, then latency and memory measurements.

**Facts you need (verified against docs.vespa.ai and the Vespa sample apps, 2026-11):**

- Local single node: `docker run --detach --name vespa --hostname vespa-container --publish 8080:8080 --publish 19071:19071 vespaengine/vespa`. Give Docker at least 4 GB (use 16 GB for all of Prime). Port 8080 serves search and feed; 19071 serves deployment.
- CLI: `vespa config set target local`, `vespa deploy --wait 300 ./app`, `vespa feed feed.jsonl` (one operation per line).
- Application package: `app/services.xml` and `app/schemas/<schema-name>.sd` (the file name must equal the schema name).
- `unpack_bits(t, float)` turns `tensor<int8>(fact{}, x[16])` into `tensor<float>(fact{}, x[128])` with **0/1** cells, big-endian bit order by default, keeping the dimension name.
- Joins on shared mapped dimensions are **inner joins**: cells present in only one argument are dropped.
- `tensorFromWeightedSet` exists but costs time linear in the set size at ranking time; use native tensor fields for ranking.
- `closeness(field, topic)` is defined only for documents matched by `nearestNeighbor`.
- Query tensors: an indexed tensor as a JSON array; a mapped tensor `tensor<float>(r{})` as a JSON object `{"r0": 0.6}`; a mixed tensor `tensor<float>(r{}, x[128])` as an object of label → array; a tensor with two mapped dimensions as a **string** in tensor literal form `"{{r:r0,key:s7_n101}:1.0}"`.
- Weighted-set parameters for `dotProduct(vkeys, @vk)` are passed as a string `"{s7_n101:625,s3_n101:201}"`; weights are integers.
- Default `hits` limit is 400; default query timeout is short, so set `"timeout": "5s"`.

**`app/services.xml`:**

```xml
<?xml version="1.0" encoding="utf-8" ?>
<services version="1.0">
  <container id="default" version="1.0">
    <search/>
    <document-api/>
  </container>
  <content id="content" version="1.0">
    <min-redundancy>1</min-redundancy>
    <documents>
      <document type="doc" mode="index"/>
    </documents>
    <nodes>
      <node distribution-key="0" hostalias="node1"/>
    </nodes>
  </content>
</services>
```

(If your Vespa version rejects `min-redundancy`, use `<redundancy>1</redundancy>`.)

**`app/schemas/doc.sd`:**

```
schema doc {
    document doc {
        field doc_id type int {
            indexing: attribute | summary
        }
        field doc_type type string {
            indexing: attribute | summary
        }
        field text type string {
            indexing: index | summary
            index: enable-bm25
        }
        field topic type tensor<bfloat16>(x[256]) {
            indexing: attribute | index
            attribute {
                distance-metric: prenormalized-angular
            }
        }
        field vkeys type weightedset<string> {
            indexing: attribute
            attribute: fast-search
        }
        field nk type tensor<float>(key{}) {
            indexing: attribute
            attribute: fast-rank
        }
        field tk type tensor<float>(fact{}, key{}) {
            indexing: attribute
            attribute: fast-rank
        }
        field tc type tensor<int8>(fact{}, x[16]) {
            indexing: attribute
        }
    }

    fieldset default {
        fields: text
    }

    rank-profile slotspace inherits default {
        inputs {
            query(rk) tensor<float>(r{}, key{})
            query(w) tensor<float>(r{})
            query(g) tensor<float>(r{}, x[128])
            query(qt) tensor<float>(x[256])
        }
        function dense() {
            expression: sum(query(qt) * cell_cast(attribute(topic), float))
        }
        function native() {
            expression: reduce(query(rk) * attribute(nk), max, key)
        }
        function gate() {
            expression: reduce(query(rk) * attribute(tk), max, key)
        }
        function simf() {
            expression: reduce(query(g) * (2 * unpack_bits(attribute(tc), float) - 1), sum, x) / 11.313708498984761
        }
        function textpart() {
            expression: reduce(gate * map(simf, f(x)(max(0, x))), max, fact)
        }
        function bind() {
            expression: sum(query(w) * native) + sum(query(w) * textpart) - sum(query(w) * join(native, textpart, f(x,y)(min(x,y))))
        }
        first-phase {
            expression: bm25(text) + 10 * dense + 0.002 * rawScore(vkeys)
        }
        second-phase {
            total-rerank-count: 500
            expression: bind
        }
        match-features: bind dense bm25(text)
    }
}
```

`11.313708498984761` = $\sqrt{128}$. For other $(a, b)$ settings generate extra profiles from this text (D12.5). Add a profile `hybrid` (same inputs, `second-phase` expression `dense`) for B2 latency comparisons.

**Feed (`vespa_app/feed.py`).** One JSON line per document:

```python
def engine_key(key: str, slot_index: dict) -> str:
    slot, value = key.split("|", 1)
    return f"s{slot_index[slot]}_{value}"           # e.g. s7_n101

def vespa_doc(doc_id, doc_type, text, topic256, native_keys, text_facts, slot_index) -> dict:
    """text_facts: list of (fact_label, {key: pi}, ctx128_float). Fact labels: 't0', 't1', ..."""
    nk = [engine_key(k, slot_index) for k in native_keys]
    tk_cells, tc = [], {}
    vkeys = {ek: 1 for ek in nk}
    for label, keys, ctx in text_facts:
        for k, pi in keys.items():
            ek = engine_key(k, slot_index)
            tk_cells.append({"address": {"fact": label, "key": ek}, "value": float(pi)})
            vkeys[ek] = 1
        tc[label] = binarize(ctx).tolist()          # 16 int8 values; reference.binarize
    return {"put": f"id:slotspace:doc::{doc_id}", "fields": {
        "doc_id": int(doc_id), "doc_type": doc_type, "text": text,
        "topic": {"values": [float(x) for x in topic256]},
        "vkeys": vkeys,
        "nk": {"cells": [{"address": {"key": ek}, "value": 1.0} for ek in nk]},
        "tk": {"cells": tk_cells},
        "tc": tc,
    }}
```

**Query (`vespa_app/query.py`).**

```python
def vespa_body(plan, query_text, qvec256, slot_index, a=1.0) -> dict:
    rk, w, g, vk, j = [], {}, {}, {}, 0
    for seg in plan["segments"]:
        for r in seg["readings"]:
            rid, ek = f"r{j}", engine_key(r["key"], slot_index)
            j += 1
            link = seg["links"][r["key"].split("|", 1)[1]]
            rk.append(f"{{r:{rid},key:{ek}}}:{link ** a:.6f}")
            w[rid] = float(r["w"])
            g[rid] = [float(x) for x in seg["g"]]
            vk[ek] = max(vk.get(ek, 0), int(round(1000 * r["w"])))
    yql = "select * from doc where ({totalTargetHits:200}nearestNeighbor(topic, qt)) or userQuery()"
    body = {"yql": yql, "query": query_text, "type": "weakAnd", "ranking": "slotspace", "hits": 400,
            "timeout": "5s", "presentation.timing": True, "input.query(qt)": [float(x) for x in qvec256]}
    if rk:
        body["yql"] += " or dotProduct(vkeys, @vk)"
        body["vk"] = "{" + ",".join(f"{k}:{v}" for k, v in vk.items()) + "}"
        body.update({"input.query(rk)": "{" + ",".join(rk) + "}", "input.query(w)": w, "input.query(g)": g})
    return body
```

Send with `curl -s -H "Content-Type: application/json" --data @body.json http://localhost:8080/search/` or with pyvespa: `Vespa(url="http://localhost", port=8080).query(body=body)`. Read `root.children[i].fields.matchfeatures` for `bind`, `dense`, `bm25(text)`, and `timing.searchtime`.

**Smoke test (`tests/test_vespa_smoke.py`, needs Docker).** Feed the four toy documents of Appendix B with random 128-d context vectors (seed 13) for their text facts and native keys as `nk`; query with the toy readings. Expected: deploy succeeds; all four documents are returned for a query that matches everything (`select * from doc where true`); `bind` for each document equals `binding_score(doc, readings, ScoreConfig(precision="binary"))` from the reference within $10^{-3}$, where the reference documents use `binarize(ctx)` as their context. Any mismatch is a syntax or format problem: fix the engine side, never the math.

**Parity test.** 200 fixed `val` queries, all returned hits: the same comparison as the smoke test. Record the maximum absolute difference.

**Measurements.** Index memory (from the container's metrics or `docker stats`), feed time, p50/p95 `searchtime` and client latency over 1,000 `val` queries for profiles `hybrid` and `slotspace`, and for B6 fan-out (sum and max over its queries). Report quality on `val` for the engine path next to the research path (first-phase decides the candidate pool here, so small differences are expected).

**Pitfalls.**

- `sentence_transformers` `quantize_embeddings(..., precision="binary")` subtracts 128 from the packed bytes. Vespa's `unpack_bits` reads the raw two's-complement bits. Use `reference.binarize` (`np.packbits(...).view(np.int8)`) for engine vectors.
- A plain element-wise `max` join between `native` and `textpart` drops readings that have only one of the two parts; keep the `N + T − min(N, T)` formula.
- Fact labels and engine keys must use only `[a-z0-9_]`.
- `rerank-count` is per content node; `total-rerank-count` is the total.

---

### M10 — Release (≈ 1.5 weeks)

- `README.md`: what SlotSpace is (three paragraphs from D1), quick start, how to reproduce v0 (`make reproduce-v0` runs M0–M8 from cached downloads), results table, limitations (D14).
- License: Apache-2.0 for code. `NOTICE`: STaRK data is CC-BY-4.0 — cite the STaRK paper and link the dataset; do **not** re-host raw STaRK files. The role-swap set is released as query ids, templates, value ids and the build script, with the same attribution.
- Model card for the fine-tuned slot scorer: base model and license, training data (synthetic, generated from the schema; no real KB names), intended use, metrics from M5, limitations.
- `CITATION.cff`; changelog; `docs/DECISIONS.md` included.
- Check every dependency and model license again before publishing.

---

## 6. Schedule

| Milestone | Effort (days) | Cumulative (weeks) |
|---|---|---|
| M0 data export | 2 | 0.4 |
| M1 baselines and evaluation | 4 | 1.2 |
| M2 schema, linker, facts | 5 | 2.2 |
| M3 scorer and lexicon | 3 | 2.8 |
| M4 silver labels and role-swap set | 3 | 3.4 |
| M5 slot scorer | 7 | 4.8 |
| M6 planner | 3 | 5.4 |
| M7 end-to-end and tuning | 5 | 6.4 |
| M8 experiments and report | 5 | 7.4 |
| M9 Vespa engine path (v1) | 15 | 10.4 |
| M10 release | 8 | 12.0 |

GPU time for v0 (estimate): document and fact embeddings about 2.5 GPU-hours per full pass, about 7.5 with reruns; paraphrasing under 1 GPU-hour; slot-scorer training about 1.5 GPU-hours per model (three models: main, A12, A14). Total roughly 10–15 GPU-hours.

---

## 7. Human checkpoints

| Id | After | What the human checks | Stop condition |
|---|---|---|---|
| H0 | M0 | the count table | any count differs from Section M0 |
| H1 | M2 | 100 query mentions, 100 document mentions, slot descriptions | linking precision below 85% |
| H2 | M4 | 100 silver segments | agreement below 80% |
| H3 | M4 | 50 role-swap queries | unnatural or wrong queries above 10% |
| H4 | M5 | scorer metrics, 30 scored mentions | ECE above 0.05 after scaling |
| H5 | M7 | the frozen configuration | — (approval needed before test runs) |
| H6 | M8 | the report | — |

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'qwen3'` | transformers too old | `pip install -U "transformers>=4.51"` |
| `import stark_qa` fails | missing graph library or Python 3.12 | install `torch_geometric`; use Python 3.11 |
| BM25 far from the paper's numbers | doc id mapping or tokenization | map `bm25s` positions through `docs.parquet`; same stopwords and stemmer for corpus and queries |
| official and own metrics disagree | `pred_dict` keys not ints, or wrong candidate list | use `int` keys and `skb.candidate_ids` |
| GPU out of memory while encoding | long documents | lower `batch_size`, keep `max_seq_length` 512, load the model in fp16 |
| many false links | short names, gene symbols, common words | raise `min_name_chars`, extend the common-word list, inspect the 50 most frequent linked names |
| all document π equal to 1 | `none` missing from candidates | add `none.<type>` to every candidate list (D6.2) |
| conformal miss rate above α + 0.02 | label noise or a different calibration distribution | re-check silver labels (H2), refit temperatures, recompute thresholds on the right split |
| `Expected 'query(...)' to be a tensor` (Vespa) | input not declared | declare it in the profile's `inputs` |
| engine `bind` lower than the reference for some documents | element-wise max join dropped readings, or wrong binarization | use the `N + T − min` formula and `reference.binarize` |
| vLLM out of memory | context length | lower `max_model_len` to 4096 and the GPU memory fraction |

---

## 9. Definition of done (v0)

- All tests pass (`pytest -q`), including Appendix B and C tests.
- Acceptance checks of M0–M8 met or failures documented.
- `runs/report_v0.md` contains Tables 1–4 and Figures 1–2 of D13.7 with n, MDE and Holm-adjusted p-values, and every pass line of D13.5 is marked.
- `docs/DECISIONS.md` lists every deviation, with dates.
- A fresh clone reproduces the report with the documented commands (allowing for download time).

---

## Appendix A — Reference implementation (verbatim) → `src/slotspace/reference.py`

Only numpy is required. Every faster implementation must match it on the tests.

```python
"""SlotSpace numpy reference implementation (DESIGN.md sections 7-10).

This module is the single source of truth for the scoring math. Faster code
(batched numpy, the Vespa rank profile) must reproduce its numbers on the
parity tests. Only numpy is required. Readability beats speed here.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

TAU_MIN = 0.05  # smallest role probability stored for a text fact (the gate floor)


# ---------------------------------------------------------------- vectors
def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def binarize(v) -> np.ndarray:
    """float[D] (D multiple of 8) -> int8[D/8].

    bit = 1 if component > 0 else 0 (exact zero -> 0). The first component is
    the most significant bit of the first byte ('big' bit order). Bytes are
    reinterpreted as two's-complement int8 (a view; NOT 'minus 128').
    """
    bits = (np.asarray(v) > 0).astype(np.uint8)
    return np.packbits(bits, bitorder="big").view(np.int8)


def unpack_pm1(b) -> np.ndarray:
    """int8[D/8] -> float[D] with values in {-1.0, +1.0}."""
    bits = np.unpackbits(np.asarray(b, dtype=np.int8).view(np.uint8), bitorder="big")
    return bits.astype(np.float64) * 2.0 - 1.0


def sim(g: np.ndarray, ctx: np.ndarray, precision: str = "float") -> float:
    """precision='float': cosine of unit vectors.
    precision='binary': ctx is packed int8; <g, pm1>/sqrt(D) (float query, binary doc)."""
    if precision == "float":
        return float(np.dot(g, ctx))
    pm = unpack_pm1(ctx)
    return float(np.dot(g, pm) / math.sqrt(pm.shape[0]))


def psi(x: float) -> float:
    """Negative context similarity gives no credit."""
    return x if x > 0.0 else 0.0


# ---------------------------------------------------------------- data model
@dataclass
class Fact:
    fact_id: str
    keys: Dict[str, float]          # "slot|value_id" -> pi in [TAU_MIN, 1]
    ctx: Optional[np.ndarray]       # text facts: unit float (or packed int8); native: may be None
    native: bool = False            # native fact = one KB edge: exactly one key with pi = 1.0


@dataclass
class Doc:
    doc_id: str
    facts: List[Fact]


@dataclass
class Reading:
    rid: str                        # e.g. "s0:drug.contraindication.disease|n123"
    segment: int                    # which query segment (value mention) it belongs to
    keys: Dict[str, float]          # "slot|value_id" -> link probability (usually one key)
    weight: float                   # w_r from the planner
    g: np.ndarray                   # unit float vector of the delexicalised segment context


@dataclass
class ScoreConfig:
    mode: str = "bound"             # bound | unbound | gate_only | no_role
    combine: str = "max"            # max | sum  (how native and text evidence combine)
    a: float = 1.0                  # exponent on the gate (role probability x link prob)
    b: float = 1.0                  # exponent on psi(sim)
    precision: str = "float"        # float | binary (text-fact context vectors)


# ---------------------------------------------------------------- scoring
def fact_gate(f: Fact, r: Reading) -> float:
    """max over the reading's keys of linkprob * pi_f(key); cells below TAU_MIN never open."""
    best = 0.0
    for k, lp in r.keys.items():
        pi = f.keys.get(k, 0.0)
        if pi >= TAU_MIN:
            best = max(best, lp * pi)
    return best


def _values(keys: Iterable[str]) -> set:
    return {k.split("|", 1)[1] for k in keys}


def native_part(doc: Doc, r: Reading, cfg: ScoreConfig) -> float:
    """N_r(d): native facts carry pi = 1 and context factor 1, so N = max link prob of a matching key."""
    if cfg.mode == "no_role":  # value in any slot counts
        vals = _values(r.keys)
        return max((1.0 for f in doc.facts if f.native and _values(f.keys) & vals), default=0.0)
    best = 0.0
    for f in doc.facts:
        if f.native:
            best = max(best, fact_gate(f, r))
    return best ** cfg.a if best > 0 else 0.0


def text_part(doc: Doc, r: Reading, cfg: ScoreConfig) -> float:
    """T_r(d) over text facts, for each ablation mode."""
    texts = [f for f in doc.facts if not f.native]
    if not texts:
        return 0.0
    if cfg.mode == "bound":  # gate and context on the SAME fact
        best = 0.0
        for f in texts:
            gt = fact_gate(f, r)
            if gt > 0.0:
                best = max(best, gt ** cfg.a * psi(sim(r.g, f.ctx, cfg.precision)) ** cfg.b)
        return best
    if cfg.mode == "unbound":  # per-segment Chamfer: maximised separately, then multiplied
        best_gate = max((fact_gate(f, r) for f in texts), default=0.0)
        if best_gate == 0.0:
            return 0.0
        best_sim = max(psi(sim(r.g, f.ctx, cfg.precision)) for f in texts)
        return best_gate ** cfg.a * best_sim ** cfg.b
    if cfg.mode == "gate_only":
        best_gate = max((fact_gate(f, r) for f in texts), default=0.0)
        return best_gate ** cfg.a if best_gate > 0 else 0.0
    if cfg.mode == "no_role":  # value present in the fact, role ignored
        vals = _values(r.keys)
        return max((psi(sim(r.g, f.ctx, cfg.precision)) ** cfg.b for f in texts if _values(f.keys) & vals), default=0.0)
    raise ValueError(f"unknown mode {cfg.mode}")


def reading_score(doc: Doc, r: Reading, cfg: ScoreConfig = ScoreConfig()) -> float:
    """M_r(d) in [0, 1]."""
    n, t = native_part(doc, r, cfg), text_part(doc, r, cfg)
    return max(n, t) if cfg.combine == "max" else n + t


def binding_score(doc: Doc, readings: List[Reading], cfg: ScoreConfig = ScoreConfig()) -> float:
    """B(d) = sum_r w_r * M_r(d)."""
    return float(sum(r.weight * reading_score(doc, r, cfg) for r in readings))


# ---------------------------------------------------------------- ambiguity lexicon
def doc_contributions(doc: Doc) -> Dict[Tuple[str, str], float]:
    """(value_id, slot) -> c_d(v, k) = max over the doc's facts of pi (native facts give 1.0)."""
    out: Dict[Tuple[str, str], float] = {}
    for f in doc.facts:
        for key, pi in f.keys.items():
            if pi < TAU_MIN:
                continue
            slot, value = key.split("|", 1)
            out[(value, slot)] = max(out.get((value, slot), 0.0), pi)
    return out


class Lexicon:
    """soft[(v,k)] = sum_d c_d(v,k);  hard[(v,k)] = number of docs with c_d(v,k) >= TAU_MIN.

    Maintained incrementally per document version; a rebuild from scratch must
    give identical numbers (invariant I4)."""

    def __init__(self) -> None:
        self.soft: Dict[Tuple[str, str], float] = defaultdict(float)
        self.hard: Dict[Tuple[str, str], int] = defaultdict(int)
        self._doc: Dict[str, Dict[Tuple[str, str], float]] = {}
        self.version = 0

    def _apply(self, contrib: Dict[Tuple[str, str], float], sign: int) -> None:
        for vk, c in contrib.items():
            self.soft[vk] += sign * c
            self.hard[vk] += sign
            if self.hard[vk] == 0:
                self.soft.pop(vk, None)
                self.hard.pop(vk, None)

    def upsert(self, doc: Doc) -> None:
        if doc.doc_id in self._doc:
            self._apply(self._doc[doc.doc_id], -1)
        contrib = doc_contributions(doc)
        self._apply(contrib, +1)
        self._doc[doc.doc_id] = contrib
        self.version += 1

    def delete(self, doc_id: str) -> None:
        if doc_id in self._doc:
            self._apply(self._doc.pop(doc_id), -1)
            self.version += 1

    def support(self, value: str) -> Dict[str, float]:
        return {k: s for (v, k), s in self.soft.items() if v == value}

    def hard_support(self, value: str, slot: str) -> int:
        return self.hard.get((value, slot), 0)

    def snapshot(self) -> Dict[Tuple[str, str], Tuple[float, int]]:
        return {vk: (round(self.soft[vk], 9), self.hard[vk]) for vk in self.hard}


def pacc_binary(q_mean: float, tpr: float, fpr: float, min_gap: float = 0.05) -> float:
    """PACC for one role: (mean posterior - FPR_pa) / (TPR_pa - FPR_pa), clipped to [0, 1].

    If TPR - FPR < min_gap the correction is unstable: return the raw mean (caller logs a warning)."""
    if tpr - fpr < min_gap:
        return float(np.clip(q_mean, 0.0, 1.0))
    return float(np.clip((q_mean - fpr) / (tpr - fpr), 0.0, 1.0))


# ---------------------------------------------------------------- planner pieces
def reading_posterior(
    C: Dict[str, Dict[str, float]],       # value_type -> {slot: P(slot | context, type)}
    links: Dict[str, Tuple[str, float]],  # value_id -> (value_type, link probability)
    lex: Lexicon,
    beta: float = 0.5,
    eps: float = 1.0,
) -> Dict[str, float]:
    """p(slot, value | q, s) ∝ link(v) * C(slot | ctx, type(v)) * E(slot | v)^beta.

    E(slot | v) = (L_v[slot] + eps) / (sum over compatible slots of L_v + eps * n_slots).
    Readings with hard support 0 are dropped (lossless, see DESIGN.md P2)."""
    scores: Dict[str, float] = {}
    for v, (vtype, lp) in links.items():
        slots = C.get(vtype, {})
        sup = lex.support(v)
        tot = sum(sup.get(k, 0.0) for k in slots)
        for k, c in slots.items():
            if lex.hard_support(v, k) == 0:
                continue
            e = (sup.get(k, 0.0) + eps) / (tot + eps * len(slots))
            scores[f"{k}|{v}"] = lp * c * e ** beta
    z = sum(scores.values())
    return {key: s / z for key, s in scores.items()} if z > 0 else {}


def conformal_threshold(p_true, alpha: float) -> float:
    """Split conformal. Keep readings with p >= t  =>  P(true reading kept) >= 1 - alpha."""
    p_true = np.asarray(p_true, dtype=np.float64)
    n = p_true.shape[0]
    level = math.ceil((n + 1) * (1 - alpha)) / n
    if level > 1.0:
        return 0.0  # too few calibration points: keep everything
    qhat = float(np.quantile(1.0 - p_true, level, method="higher"))
    return 1.0 - qhat


def keep_readings(post: Dict[str, float], t: float, min_keep: int = 1, r_max: int = 6) -> Dict[str, float]:
    """Keep readings with p >= t (always the top min_keep, at most r_max), renormalized to sum 1."""
    ranked = sorted(post.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = [(k, p) for i, (k, p) in enumerate(ranked) if (p >= t or i < min_keep) and i < r_max]
    z = sum(p for _, p in kept)
    return {k: p / z for k, p in kept} if z > 0 else {}


def ia_select(cands: List[str], P: Dict[str, float], V: Dict[str, Dict[str, float]], k: int) -> List[str]:
    """Greedy IA-Select. P: reading -> probability; V[reading][doc] in [0, 1]."""
    U = dict(P)
    chosen: List[str] = []
    remaining = list(cands)
    while remaining and len(chosen) < k:
        best = max(remaining, key=lambda d: (sum(U[c] * V.get(c, {}).get(d, 0.0) for c in U), -cands.index(d)))
        chosen.append(best)
        remaining.remove(best)
        for c in U:
            U[c] *= 1.0 - V.get(c, {}).get(best, 0.0)
    return chosen
```

## Appendix B — Reference tests (verbatim) → `tests/test_reference.py`

The expected values below were produced by running Appendix A while this guide was written (12 tests, all passing). The toy documents are the worked example of D7.5.

```python
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
```

## Appendix C — Fast scorer, parity tests and synthetic binding check (verbatim)

### C.1 `src/slotspace/fast_scorer.py`

Same math as the reference; only facts that hold a reading's key are touched. Build it with `precision="float"` (default) or `precision="binary"` (ablation A11, engine parity at scale).

```python
"""Fast binding scorer for the research path (DEVELOPMENT.md M3).

Same math as slotspace_reference.binding_score, but documents are stored in
compact arrays and only gated facts are touched. Parity with the reference is
checked by tests/test_fast_scorer.py.

precision='binary' stores each context as unpack_pm1(binarize(ctx)) / sqrt(D),
which gives exactly the reference's binary similarity.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

from slotspace.reference import TAU_MIN, Doc, Reading, ScoreConfig, binarize, unpack_pm1


class FactStore:
    def __init__(self, dim: int = 128, precision: str = "float") -> None:
        self.dim, self.precision = dim, precision
        self._ctx: List[np.ndarray] = []
        self.native: Dict[str, set] = {}
        self.native_values: Dict[str, set] = {}
        self.text_rows: Dict[str, np.ndarray] = {}                 # doc -> all text-fact rows
        self.key_index: Dict[str, Dict[str, tuple]] = {}           # doc -> key -> (rows, pis)
        self.value_rows: Dict[str, Dict[str, np.ndarray]] = {}     # doc -> value -> rows (no_role mode)
        self.ctx = np.zeros((0, dim), dtype=np.float32)

    def add_docs(self, docs: Iterable[Doc]) -> None:
        start = len(self._ctx)
        for d in docs:
            nat, natv = set(), set()
            per_key, per_val, rows = defaultdict(lambda: ([], [])), defaultdict(list), []
            for f in d.facts:
                if f.native:
                    for k in f.keys:
                        nat.add(k)
                        natv.add(k.split("|", 1)[1])
                    continue
                row = len(self._ctx)
                c = np.asarray(f.ctx, dtype=np.float64)
                if self.precision == "binary":
                    c = unpack_pm1(binarize(c)) / np.sqrt(c.shape[0])
                self._ctx.append(c.astype(np.float32))
                rows.append(row)
                for k, pi in f.keys.items():
                    if pi >= TAU_MIN:
                        per_key[k][0].append(row)
                        per_key[k][1].append(pi)
                        per_val[k.split("|", 1)[1]].append(row)
            self.native[d.doc_id], self.native_values[d.doc_id] = nat, natv
            self.text_rows[d.doc_id] = np.array(rows, dtype=np.int64)
            self.key_index[d.doc_id] = {k: (np.array(r, dtype=np.int64), np.array(p, dtype=np.float64)) for k, (r, p) in per_key.items()}
            self.value_rows[d.doc_id] = {v: np.array(sorted(set(r)), dtype=np.int64) for v, r in per_val.items()}
        if len(self._ctx) > start:
            self.ctx = np.vstack(self._ctx).astype(np.float32)

    def scores(self, doc_ids: Sequence[str], readings: List[Reading], cfg: ScoreConfig = ScoreConfig()) -> np.ndarray:
        if cfg.precision != self.precision:
            raise ValueError(f"store built with precision={self.precision}, config asks for {cfg.precision}")
        G = np.stack([r.g for r in readings]).astype(np.float32) if readings else np.zeros((0, self.dim), np.float32)
        out = np.zeros(len(doc_ids), dtype=np.float64)
        for i, d in enumerate(doc_ids):
            nat, keys = self.native.get(d, set()), self.key_index.get(d, {})
            total = 0.0
            for j, r in enumerate(readings):
                # native part
                if cfg.mode == "no_role":
                    vals = {k.split("|", 1)[1] for k in r.keys}
                    n = 1.0 if vals & self.native_values.get(d, set()) else 0.0
                else:
                    best = max((lp for k, lp in r.keys.items() if k in nat), default=0.0)
                    n = best ** cfg.a if best > 0 else 0.0
                # text part
                t = 0.0
                if cfg.mode == "bound":
                    for k, lp in r.keys.items():
                        if k in keys:
                            rows, pis = keys[k]
                            s = np.maximum(self.ctx[rows] @ G[j], 0.0).astype(np.float64)
                            t = max(t, float(np.max((lp * pis) ** cfg.a * s ** cfg.b)))
                elif cfg.mode in ("unbound", "gate_only"):
                    gmax = max((lp * float(keys[k][1].max()) for k, lp in r.keys.items() if k in keys), default=0.0)
                    if gmax > 0:
                        t = gmax ** cfg.a
                        if cfg.mode == "unbound":
                            rows = self.text_rows[d]
                            t *= float(np.max(np.maximum(self.ctx[rows] @ G[j], 0.0))) ** cfg.b
                elif cfg.mode == "no_role":
                    vr = self.value_rows.get(d, {})
                    rows = [vr[k.split("|", 1)[1]] for k in r.keys if k.split("|", 1)[1] in vr]
                    if rows:
                        rows = np.unique(np.concatenate(rows))
                        t = float(np.max(np.maximum(self.ctx[rows] @ G[j], 0.0))) ** cfg.b
                m = max(n, t) if cfg.combine == "max" else n + t
                total += r.weight * m
            out[i] = total
        return out
```

### C.2 `tests/test_fast_scorer.py`

Both tests passed when this guide was written.

```python
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
```

### C.3 `src/slotspace/sim_binding.py`

Run with `python -m slotspace.sim_binding`. Observed output when this guide was written (numpy 2.x, about 1 minute):

```
[default] queries=300 doc-side role acc=0.526 query-side role acc=0.690
   bound a=1 b=1    R-prec=0.835  MRR=0.984
   bound a=0.5 b=1  R-prec=0.842  MRR=0.995
   unbound          R-prec=0.483  MRR=0.726
   gate_only        R-prec=0.436  MRR=0.673
   no_role          R-prec=0.880  MRR=0.997
[sharper doc-side roles] queries=300 doc-side role acc=0.803 query-side role acc=0.690
   bound a=1 b=1    R-prec=0.888  MRR=0.969
   bound a=0.5 b=1  R-prec=0.891  MRR=0.990
   unbound          R-prec=0.635  MRR=0.785
   gate_only        R-prec=0.601  MRR=0.754
   no_role          R-prec=0.880  MRR=0.997
[weaker context vectors] queries=300 doc-side role acc=0.526 query-side role acc=0.690
   bound a=1 b=1    R-prec=0.573  MRR=0.871
   bound a=0.5 b=1  R-prec=0.560  MRR=0.872
   unbound          R-prec=0.447  MRR=0.692
   gate_only        R-prec=0.436  MRR=0.673
   no_role          R-prec=0.536  MRR=0.853
```

Read it as D7.6 does: binding beats unbound and gate-only in every setting; whether `no_role` or `bound` wins depends on the quality of the document-side role model versus the context vectors.

```python
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
```

## Appendix D — Test-only role-swap templates (`templates/test/role_swap.yaml`)

This file is read **only** by `role_swap.py`. It must never be used for training, paraphrasing or tuning. `{V}` is replaced by the value's KB name.

```yaml
# TEST-ONLY. Never read by training or data-generation code.
drug.indication.disease:
  - "My patient was just diagnosed with {V}. Which medicines are approved to treat it?"
  - "Name the pharmaceutical agents whose labeled use is the treatment of {V}."
  - "For {V}, which therapies are listed as indicated?"
  - "I am looking for compounds indicated in {V}."
  - "Which drugs have {V} among their therapeutic indications?"
  - "What medication is given to patients to treat {V}?"
drug.contraindication.disease:
  - "Which medications must be avoided when a patient has {V}?"
  - "List drugs that are contraindicated for someone with {V}."
  - "A patient with {V} needs a prescription: which drugs are unsafe for them?"
  - "Which agents carry a contraindication for {V}?"
  - "Which drugs should never be prescribed if {V} is present?"
  - "What are the drugs to stay away from in cases of {V}?"
drug.off_label_use.disease:
  - "Which drugs are sometimes used off-label for {V}?"
  - "Without formal approval, what medicines do clinicians use to manage {V}?"
  - "List drugs prescribed outside their approved indications for {V}."
  - "What unapproved drug uses exist for {V}?"
  - "Which medications are used for {V} even though they are not licensed for it?"
  - "Find drugs with off-label use in {V}."
drug.side_effect.effect_phenotype:
  - "Which drugs list {V} as a possible adverse effect?"
  - "What medications can lead to {V} in patients who take them?"
  - "Which treatments are known to induce {V}?"
  - "List drugs where {V} is reported as a side effect."
  - "I developed {V} after starting a new medicine. Which drugs could cause this?"
  - "Which compounds have {V} among their adverse reactions?"
```

Training templates (`templates/train/prime_slots.yaml`, M5) must be written independently, after this file is frozen, and must not copy any of these sentences.

## Appendix E — Statistics helpers (verbatim) → `src/slotspace/stats.py`

Checked while writing this guide: Holm on {x: 0.01, y: 0.04, z: 0.03} gives {x: 0.03, z: 0.06, y: 0.06}; `mde_points(2801, 0.2)` = 2.37 and with `m=6` = 2.94.

```python
"""Statistics for SlotSpace reports (DEVELOPMENT.md M8)."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from scipy import stats as st


def paired_t(a, b) -> float:
    """Two-sided paired t-test p-value for per-query metric vectors a, b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    return float(st.ttest_rel(a, b).pvalue)


def randomization_test(a, b, n_perm: int = 10000, seed: int = 13) -> float:
    """Two-sided paired randomization (sign-flip) test on the mean difference."""
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(perm >= obs - 1e-12)) / (n_perm + 1))


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def bootstrap_ci(a, b, n_boot: int = 1000, seed: int = 13, level: float = 0.95):
    """Percentile bootstrap interval for mean(a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d.mean()), float(lo), float(hi)


def mde_points(n: int, p_discordant: float, alpha: float = 0.05, power: float = 0.8, m: int = 1) -> float:
    """Minimum detectable difference (in points) for a paired binary metric such as Hit@1."""
    z = st.norm.ppf(1 - alpha / (2 * m)) + st.norm.ppf(power)
    return 100.0 * z * math.sqrt(p_discordant / n)


def discordant_rate(a, b) -> float:
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    return float(np.mean(a != b))
```

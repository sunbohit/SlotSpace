# SlotSpace — Design Document

**Ambiguity-aware, fact-level retrieval with a shared slot space**

| Item | Value |
|---|---|
| Version | 0.1 (2026-11-24) |
| Status | Ready for v0 implementation |
| Companion | `SlotSpace_DEVELOPMENT.md` (build plan, code contracts, tests); in the repository the two files are `docs/DESIGN.md` and `docs/DEVELOPMENT.md` |
| License intent | Code: Apache-2.0. Data: each dataset keeps its own license (STaRK: CC-BY-4.0, attribution required) |
| v0 scope | STaRK-Prime, Python research path (numpy), open models only, one GPU |

---

## 0. How to read this document

This document explains **what** SlotSpace is, **why** each design choice was made, and **how** the math works. `DEVELOPMENT.md` explains **how to build it**, step by step, with tests and expected values.

Read in this order:

1. Section 1 (summary) and Section 2 (background). Do not skip Section 2 if any term is unfamiliar.
2. Section 4 (notation). Keep it open while reading the rest.
3. Sections 5–10 specify the system. They are normative.
4. Sections 11–13 specify the small model, the engine mapping and the evaluation.

Conventions:

- **MUST** = required for v0. **SHOULD** = strongly recommended; deviate only with a written reason in `docs/DECISIONS.md`. **MAY** = optional.
- **v0** = the first deliverable. **v1** = a planned extension. Do not build v1 items before v0 passes its acceptance tests.
- Numbers marked *(simulation)* come from small synthetic programs. They show that the code behaves as intended. They are not evidence about real data.
- If this document and `DEVELOPMENT.md` disagree on **math**, this document wins. If they disagree on **file names, commands or library calls**, `DEVELOPMENT.md` wins.

---

## 1. Summary

### 1.1 The problem in one example

Three queries against a biomedical knowledge base:

- Q1 "Which drugs treat hypertension?"
- Q2 "Which drugs should be avoided in patients with hypertension?"
- Q3 "Which drugs can cause hypertension?"

All three contain the same value, *hypertension*. Their correct answers are almost disjoint. What changes is the **role** the value plays: the indication of a drug (Q1), a contraindication (Q2), or a side effect (Q3). In Q3 the value is not even the same kind of thing: it is an observed effect (a phenotype), not a disease entry.

Standard retrievers see the value and the general topic. A drug document that says "contraindicated in hypertension" looks very similar to one that says "indicated for hypertension". A single-vector dense retriever pools the whole document into one vector, so it cannot tell which statement holds the value. BM25 matches the word wherever it is.

### 1.2 The idea

SlotSpace makes the **role** a first-class object that every component shares:

1. A **schema** lists the *slots* (roles) of a domain, such as `drug.indication.disease` or `drug.side_effect.effect_phenotype`. The schema is a scene hyperparameter: written once per domain, versioned, and read by all components.
2. Documents are stored as **facts**. A fact is a small unit of evidence: which values it mentions, a probability distribution over the slots each value fills, and a context vector of the surrounding words with the values masked.
3. A query is split into **readings**. A reading is one hypothesis, "this mention means value *v* in slot *k*", with a probability.
4. A document scores for a reading only if **one single fact** contains the value, gives it that slot, and has a compatible context. This is the **binding rule**. The best fact wins (max), and readings add up with their probabilities.
5. An **Ambiguity Lexicon** counts, for every value, how many documents use it in each slot. It is a view of the index and is updated when documents change. It tells the planner which readings are possible in *this* corpus.
6. A **planner** combines query-side evidence (how the words around the mention fit each slot) with corpus evidence (the lexicon). It removes impossible readings at no loss, keeps a calibrated set of plausible readings, and runs **all readings in one retrieval pass**.

### 1.3 Why Coverage and Efficiency can improve together

Coverage (finding the relevant documents):

- Plausible readings are kept with a statistical guarantee (conformal reading sets, Section 10.2), so the intended reading is rarely dropped.
- Scores are soft (weighted sums), not hard filters, and a lexical plus dense fallback is always in the candidate pool.
- Fact-level binding ranks documents whose evidence is in the right role above documents that only mention the value.

Efficiency (work per query):

- One retrieval pass serves every reading. "Rewrite the query once per reading" repeats the whole search once per reading.
- Readings with zero corpus support are removed before execution. By construction this changes no score (Section 7.4, P2).
- The expensive binding score runs only on a bounded candidate set.

Section 7.4 states these properties precisely and proves them.

### 1.4 What v0 delivers

- A Python package `slotspace`: STaRK loaders, schema, fact extraction, lexicon, a fine-tuned slot scorer (< 1B parameters), planner, numpy scorer, baselines and an experiment runner.
- Results on STaRK-Prime (synthesized test split and the human-written set) and on a new **KB-grounded role-swap test set** built from Prime's own edges.
- A cost report (work per query) next to every quality number.

v1 adds a Vespa application package (the same math inside a search engine, checked by parity tests), STaRK-MAG and STaRK-Amazon, open-vocabulary values with GLiNER2, and multilingual queries.

### 1.5 What is new, stated honestly

Most building blocks exist (Section 2.8). What SlotSpace adds is a **combination around one shared slot space**:

- **N1 Binding score.** Exact value match, role probability and context similarity must come from the same fact; the best fact wins.
- **N2 Corpus-level lexicon that drives planning.** Incrementally maintained counts of value-in-slot evidence are used for the reading posterior, lossless pruning and cost estimates.
- **N3 Single-pass execution of all readings,** with a conformal guarantee on which readings are kept.
- **N4 A small model that learns only context.** It is trained on schema-generated, value-masked data, so the corpus (not the model) supplies the value priors.

Treat N1–N4 as a system design. The experiments decide how much each part is worth.

---

## 2. Background for implementers

### 2.1 Retrieval building blocks used here

- **BM25.** A lexical score. A document scores higher when it contains rare query words, normalized by document length. Fast and exact on words; blind to meaning and roles.
- **Dense retrieval (bi-encoder).** A neural encoder maps the query and each document to one vector each; the score is their cosine. It captures meaning, but the whole document becomes one vector ("pooling"), so the link between a value and the statement around it is lost.
- **Approximate nearest neighbor (ANN) search.** Finds the top-k vectors without comparing against all of them; HNSW is a common graph-based ANN index. v0 uses exact search with numpy (Prime is small enough). ANN matters only in v1.
- **Multi-vector late interaction.** Store several vectors per document (one per token or passage). For each query vector take the max similarity over the document's vectors, then sum over query vectors. ColBERT calls this MaxSim. SlotSpace's fact layer is a multi-vector design where each vector belongs to one fact.
- **Max over passages (MaxP).** A long document scores as its best passage. Facts use the same "existential" logic: one supporting fact is enough.
- **Phased ranking.** A cheap first-phase score runs on many candidates; an expensive second-phase score runs only on the top N. The binding score is a second-phase score.
- **Weighted-set retrieval.** A document field holds a set of string keys (for example `drug.indication.disease|n123`). Finding all documents with a key is an inverted-index lookup. The list of documents for a key is its **posting list**; its length is the key's **document frequency** (df).
- **Binary quantization.** Keep only the sign of each vector component (1 bit), so a 128-dimensional vector takes 16 bytes. The dot product of a float query vector with a ±1 document vector approximates the cosine (Section 6.4).

### 2.2 Corpora backed by a knowledge base

Many corpora mix structured relations and text. In STaRK (Section 13.1) every document is a knowledge-base (KB) entity: a drug, a disease, a gene, and so on. Each entity has text (name, description, other details) and typed edges to other entities; for example a drug has `indication` edges to diseases. Queries ask for entities that satisfy a mix of relational requirements ("treats X", "targets gene Y") and textual properties.

Two kinds of evidence therefore exist:

- **Native evidence**: an explicit KB edge. Its role is certain.
- **Text evidence**: a sentence in a description. Its role must be inferred from the words around the value ("used to treat", "should not be used in", "may cause").

### 2.3 Three kinds of ambiguity

Let a query contain a **mention** (a span of text) such as "hypertension".

1. **Linking ambiguity.** Which KB value does the mention refer to? "Hypertension" can be a disease node or an effect/phenotype node with the same name. Represented by a probability distribution over value ids.
2. **Slot ambiguity.** Which slot does the value fill? For a disease value and a drug answer, the slot can be indication, contraindication or off-label use. Represented by a distribution over slots.
3. **Schema collision.** The same value is legitimately used in two or more slots **in this corpus**: hypertension is the indication of some drugs and a contraindication of others. Collision is a property of the corpus, not of the query. It is what makes slot ambiguity matter: without a collision, the corpus itself resolves the ambiguity.

### 2.4 The binding problem

A drug document has two sentences:

- f1: "Do not use in patients with **asthma**."
- f2: "**Hypertension** was reported during clinical trials."

For Q2 ("avoided in hypertension") this document should not score high: no single statement says "avoid in hypertension". A scorer that looks at the parts separately finds the value (in f2) and a strong "avoid" context (in f1) and combines them. It **mixes evidence across facts**. This is the binding problem.

| Approach | What it does | Failure on the example |
|---|---|---|
| Single vector (pooling) | one vector per document | the value and "avoid" are both inside the pooled vector |
| BM25 | words anywhere | "hypertension" matches; roles are invisible |
| Per-field scoring | score each field, weight the fields | helps only if roles live in separate fields; descriptions mix roles in one field |
| Per-segment max ("unbound") | max over facts for the value, max over facts for the context, then combine | combines f2's value with f1's context |
| **Bound (SlotSpace)** | value, role probability and context must come from **the same fact**; max over facts | f1 lacks the value; f2 has a neutral context, so the score stays low |

Section 7.5 shows this with numbers. The synthetic check in `DEVELOPMENT.md` (milestone M3) shows it at scale.

### 2.5 Why the query alone cannot decide the slot

The plausibility of slot $k$ for a mention has two factors:

- $C(k \mid q)$, **context compatibility**: do the query words around the mention fit slot $k$? It depends only on the query and the schema. A model can learn it.
- $E(k \mid v)$, **corpus evidence**: is value $v$ used in slot $k$ in this corpus, and how often? It depends on the corpus and changes when documents change. A query-only model cannot know it.

A reading is useful only if both hold: the user could mean it, and the corpus can satisfy it. Consequences:

- A query-only predictor must guess $E$ from its training data. When the corpus differs or changes, those guesses are wrong, and more training cannot fix that because the corpus is not an input.
- So SlotSpace learns **only** $C$ with the model and reads $E$ from the lexicon at query time (Section 9.5).
- To make the model learn only $C$, the value is **masked** in its input (Section 11.1). If the model saw "hypertension" it could learn "hypertension usually means indication" from its training data. That is a corpus prior, and it belongs to $E$.

### 2.6 The cost of rewriting the query once per reading

A common way to handle ambiguity is fan-out: rewrite the query once per reading, run each rewrite as a separate search, then merge the result lists. If a query has $m$ mentions with $R_1,\dots,R_m$ readings, enumerating all combinations needs $\prod_i R_i$ searches; one rewrite per reading still needs $\sum_i R_i$ full searches, each with its own lexical pass, vector search and re-ranking.

SlotSpace runs one search. All readings share candidate generation; readings only change the weights of the second-phase score (cost model in Section 10.6).

### 2.7 Calibration, conformal prediction and quantification

- **Calibration.** A classifier is calibrated if, among predictions made with probability 0.8, about 80% are correct. **Temperature scaling** divides the logits by one learned number $T$ before the softmax. **Expected calibration error (ECE)** measures the gap between confidence and accuracy.
- **Split conformal prediction.** Hold out $n$ labeled calibration examples and record the model's probability of the true label for each. From these numbers compute a threshold $t$ (Section 10.2). At test time keep every label whose probability is at least $t$. Then the true label is kept with probability at least $1-\alpha$, for any model, provided calibration and test examples are exchangeable. SlotSpace uses it to decide which readings to keep.
- **Quantification.** Estimating how many items belong to each class instead of classifying each item. Summing a biased classifier's probabilities gives a biased count. **PACC** (probabilistic adjusted classify and count) corrects the sum using the classifier's measured true-positive and false-positive rates. SlotSpace uses it to correct lexicon counts that come from text (Section 8.6).

### 2.8 Prior work and positioning

Every building block below exists. Do **not** claim novelty for any single one of them; the claim is their combination around one slot space (1.5).

| Work | What it does | How SlotSpace relates |
|---|---|---|
| PRMS — Kim, Xue, Croft, ECIR 2009 | For each query term, estimates a distribution over document fields from collection statistics (no labels) and scores with a field-weighted mixture of field language models. | Closest ancestor of the lexicon. $E(k \mid v)$ is a value-level analogue of PRMS's field mapping, with roles instead of fields, document-level max counting, and a learned context model $C$ next to it. |
| mFAR — Li, Chen, Van Durme, Xia, ICLR 2025 | Splits documents into fields; scores each field with lexical and dense scorers; learns query-conditioned field weights; sums the weighted scores. | Strongest "per-field" baseline (B4 is a simplified, re-implemented version; paper numbers are cited). It weights fields per query, not readings per mention, and has no binding inside a field. |
| COIL — Gao, Dai, Callan, NAACL 2021 | Contextualized exact match: only identical tokens are compared, using their contextual vectors, keeping the best occurrence. | SlotSpace's gate is the same pattern at the level of value ids instead of tokens, with a role probability added. |
| CITADEL — Li et al., ACL 2023 | A learned router assigns token vectors to lexical keys; tokens interact only within shared keys. | SlotSpace keys (`slot|value`) are explicit, typed routes defined by the schema instead of learned ones. |
| ColBERT — Khattab, Zaharia, SIGIR 2020; ColBERTv2 — Santhanam et al., NAACL 2022 | Late interaction: sum over query vectors of the max similarity over document vectors; v2 compresses vectors with centroids plus quantized residuals. | The fact layer is a multi-vector representation; binary context vectors follow the same storage idea. |
| MaxP — Dai, Callan, SIGIR 2019 | Scores passages separately; the document score is the best passage. | Existential max over facts. |
| MUVERA — Dhulipala et al., NeurIPS 2024 | Fixed dimensional encodings that approximate multi-vector (Chamfer) similarity with one vector, then re-rank exactly. | v1 option for a single ANN traversal over facts; not in v0. |
| IA-Select — Agrawal, Gollapudi, Halverson, Ieong, WSDM 2009 | Chooses results that maximize the probability that a user with an uncertain intent finds something useful; greedy with a $(1-1/e)$ guarantee. | The `ia` weight objective, with readings as intents. |
| Sarkas, Paparizos, Tsaparas, SIGMOD 2010 | Annotates keyword queries with a table and attribute values; a generative model scores annotations; an open-language-model score filters implausible ones. | Query-side slot annotation with a "none" outcome. SlotSpace adds corpus counts per value and fact-level binding at retrieval time. |
| Saerens, Latinne, Decaestecker, Neural Computation 2002 | Re-weights classifier posteriors when class priors change; EM when the new prior is unknown. | Justifies $p \propto C \cdot E^{\beta}$ when $C$ is trained with uniform priors (9.5). |
| Guo, Pleiss, Sun, Weinberger, ICML 2017 | Temperature scaling; reports ECE (a metric credited to Naeini et al., 2015). | Calibration of $C$ and $\pi$ (9.4). |
| Angelopoulos, Bates, 2021 (arXiv 2107.07511); Angelopoulos et al., "Conformal Risk Control", ICLR 2024 | Split conformal prediction sets with a coverage guarantee; risk control generalizes it, e.g. to false-negative rate. | Reading sets (10.2). |
| Bella, Ferri, Hernández-Orallo, Ramírez-Quintana, ICDM 2010 | Quantification with probability estimators; the scaled probability average, later called PACC. | Lexicon count correction (8.6). |
| Iyer, Bilmes, NeurIPS 2013 | Submodular cover / submodular knapsack (SCSK) optimization. | A possible formal frame for budgeted reading selection; v0 uses the simpler rules of 10.2. |
| STaRK — Wu et al., NeurIPS 2024 Datasets and Benchmarks | Retrieval benchmark over semi-structured knowledge bases (products, papers, precision medicine) with synthesized and human-written queries. | The evaluation benchmark (13.1). |
| AmbigQA — Min et al., EMNLP 2020 | Open-domain questions with several plausible answers, each needing its own disambiguated question. | A different kind of ambiguity (answer-level); not used in v0. |

---

## 3. Goals, non-goals and hypotheses

### 3.1 Goals

- **G1.** Rank documents by evidence in the intended role, using fact-level binding.
- **G2.** Handle slot and linking ambiguity with calibrated probabilities from query context plus corpus evidence.
- **G3.** Execute all readings of a query in one retrieval pass, with less work than per-reading fan-out at equal or better coverage.
- **G4.** Keep the schema as the only domain-specific input. A new domain means: a new schema file, generated training data and re-scoring.
- **G5.** Be fully reproducible with open data, open models (< 1B parameters at query time) and open-source software, on one GPU.

### 3.2 Non-goals for v0

- **NG1.** No new search engine and no engine fork. (v1 uses Vespa unmodified.)
- **NG2.** No closed LLM APIs anywhere and no LLM at query time.
- **NG3.** No time or date ambiguity (STaRK has no temporal relations). Granularity ambiguity ("lymphoma" vs its subtypes) is handled only through linking distributions; hierarchy expansion is v1.
- **NG4.** No training of the main dense retriever. Off-the-shelf embeddings only.
- **NG5.** No ANN index in v0. Exact numpy search is fast enough on Prime.
- **NG6.** No private, internal or user data. Public benchmarks only.

### 3.3 Hypotheses and how they can fail

| ID | Hypothesis | Main test | It is false if ... |
|---|---|---|---|
| H1 | Binding helps role-sensitive queries | A0 vs A3 (unbound) on the role-swap test set | the Hit@1 gain is below the MDE, or Role-Confusion@10 is not lower |
| H2 | Corpus evidence improves slot decisions beyond query context | A0 vs A4 (β = 0) | plan regret on validation is not lower, and the Hit@1 gain on role-swap is below the MDE |
| H3 | Single-pass execution matches fan-out coverage at lower cost | A0 vs B6 (fan-out) | Recall@100 is lower by more than 1 point, or the work counters are not lower |
| H4 | Conformal reading sets bound the loss of the intended reading | true-reading miss rate on test | the miss rate exceeds α + 0.02 |
| H5 | No regression on ordinary queries | full STaRK-Prime test split | Recall@20 or MRR is more than 1 point below the best baseline |

MDE = minimum detectable effect (Section 13.5). A gain smaller than the MDE is reported as "not detected", never as "no effect" and never as "improvement".

---

## 4. Concepts and notation

| Symbol | Name | Meaning |
|---|---|---|
| $\mathcal{S}$ | schema | versioned list of value types and slots |
| $t$ | value type | a node type, e.g. `disease`, `drug`, `effect_phenotype` |
| $k$ | slot | a directed role seen from the document: `<doc_type>.<relation>.<value_type>` |
| $v$ | value | a value id; in v0 a KB node id written `n<index>` |
| $\tau(v)$ | type of $v$ | |
| $d$ | document | a candidate entity with its text and edges |
| $f$ | fact | a unit of evidence in $d$: native (one edge) or text (one sentence) |
| $\kappa$ | key | the string `k|v` |
| $\pi_f(\kappa)$ | role probability | probability that value $v$ fills slot $k$ in fact $f$ (1 for native facts) |
| $c_f$ | fact context vector | unit vector of the fact's text with all values masked |
| $q$ | query | |
| $s$ | segment | one value mention in the query |
| $\ell_s(v)$ | link probability | probability that mention $s$ refers to value $v$ |
| $g_s$ | segment vector | unit vector of the words around mention $s$, value masked |
| $r=(s,k,v)$ | reading | one hypothesis for segment $s$; its key is $\kappa_r$ |
| $C(k \mid x_s)$ | context compatibility | slot-scorer output for the masked context $x_s$: one distribution over the candidate slots of the mention's possible value types plus `none` |
| `none` | no-slot outcome | the mention does not express any slot (e.g. "hypertension is common"); its probability $C(\texttt{none})$ lowers the segment's weight |
| $L_v[k]$ | soft support | lexicon: sum over documents of the document's strongest evidence for $(v,k)$ |
| $H_v[k]$ | hard support | lexicon: number of documents with any stored evidence for $(v,k)$ |
| $E(k \mid v)$ | corpus evidence | smoothed share of $v$'s support that is in slot $k$ |
| $p_r$ | reading posterior | normalized $\ell \cdot C \cdot E^{\beta}$ |
| $w_r$ | reading weight | weight used in scoring; $w_r = p_r$ by default |
| $M_r(d)$ | reading match | best single-fact evidence for $r$ in $d$, in $[0,1]$ |
| $B(d)$ | binding score | $\sum_r w_r M_r(d)$ |
| $\psi$ | context transform | $\psi(x)=\max(0,x)$ |
| $\tau_{\min}$ | gate floor | 0.05; role probabilities below it are not stored |
| $a, b$ | exponents | weight of the gate and of the context term in $M_r$ (default 1, 1) |
| $\beta$ | evidence exponent | how much corpus evidence counts (0 = ignore the corpus) |
| $\alpha$ | miss rate | target probability of dropping the intended reading |

Key format: `slot|value`, for example `drug.indication.disease|n10432`. The character `|` MUST NOT occur inside slot ids or value ids.

---

## 5. The schema (the scene hyperparameter)

### 5.1 What a schema contains

A schema is one YAML file per domain (`schemas/prime.yaml` in v0). It contains:

- `name`, `version` (semantic version; any change to slots bumps it).
- `value_types`: the node types that can be values, and the masking token for each (e.g. `disease` → `[disease]`).
- `doc_types`: the node types that can be documents (retrieval candidates).
- `slots`: a list. Each slot has
  - `id`: `<doc_type>.<relation>.<value_type>`, lowercase, spaces, `/` and `-` replaced by `_` (for an inverse direction see 6.1);
  - `doc_type`, `value_type`;
  - `relation`: the KB relation name **exactly as it appears in the data** (needed to build native facts);
  - `direction`: `out` (edge goes document → value), `in` (value → document) or `sym`;
  - `description`: one or two plain sentences, written from the document's point of view ("The drug is used to treat the disease."). Used by the slot scorer and by the data generator;
  - `query_cues`, `doc_cues`: short phrases that express the slot ("used to treat", "indicated for"). Used **only** by the training-data generator, never at query time;
  - `collides_with`: slots with the same value type that the corpus actually confuses; filled automatically from the lexicon (Section 8) and used for hard negatives.
- `conventions`: text normalization (Unicode NFKC, lowercase, collapse whitespace, strip punctuation at both ends), masking tokens (`[TARGET:<type>]`, `[<type>]`, `[SELF:<doc_type>]`), minimum name length for linking, and the list of answer-type phrases used by templates.

### 5.2 Rules

- One schema per domain. Every component loads the same file and checks `version`. A version mismatch is a hard error.
- The schema is **closed and versioned** (lexicon layer L0, Section 8.3). Values are **open** and come from the corpus.
- Changing a description or adding a slot is cheap: regenerate training data for the affected value type, fine-tune (or score zero-shot from the new description), re-score text facts, rebuild the lexicon. Documents are **not** re-embedded, because context vectors do not depend on slots.
- Slots whose relation never occurs in the data are allowed (text-only slots) but MUST have `relation: null`.

### 5.3 Prime slots used in v0

Facts about STaRK-Prime that shape the schema (verified from the STaRK code, `stark_qa/skb/prime.py`):

- 10 node types: `disease`, `gene/protein`, `molecular_function`, `drug`, `pathway`, `anatomy`, `effect/phenotype`, `biological_process`, `cellular_component`, `exposure`. **All 129,375 nodes are retrieval candidates.**
- 18 relation names (with spaces, exactly as in the data): `ppi`, `carrier`, `enzyme`, `target`, `transporter`, `contraindication`, `indication`, `off-label use`, `synergistic interaction`, `associated with`, `parent-child`, `phenotype absent`, `phenotype present`, `side effect`, `interacts with`, `linked to`, `expression present`, `expression absent`.
- Every edge is stored **in both directions** with the same relation name (8,100,498 stored edges = 2 × 4,050,249). So a drug document gets `drug.indication.disease|n<disease>` and the disease document gets `disease.indication.drug|n<drug>` from the two stored copies. For relations between two nodes of the **same** type (`parent-child`, `ppi`, `synergistic interaction`) the direction cannot be recovered and the slot is `sym`.
- Type and relation names are normalized for slot ids: `gene/protein` → `gene_protein`, `effect/phenotype` → `effect_phenotype`, `off-label use` → `off_label_use`, `parent-child` → `parent_child`, and so on.

The full slot list is **derived from the data**, not typed by hand: count edges by (type of source, relation, type of target) and create one slot per combination with at least one edge (`DEVELOPMENT.md` M2). Descriptions are then written for every slot. The slots that matter most for v0 are below; the relation-to-type pairs listed for them are what PrimeKG is expected to contain and MUST be confirmed by the derivation script.

| Slot id | Relation | Description (document's point of view) |
|---|---|---|
| `drug.indication.disease` | indication | The drug is used to treat the disease. |
| `drug.contraindication.disease` | contraindication | The drug should not be used in patients who have the disease. |
| `drug.off_label_use.disease` | off-label use | The drug is used for the disease without formal approval for that use. |
| `drug.side_effect.effect_phenotype` | side effect | The drug can cause the effect or phenotype as an adverse effect. |
| `drug.target.gene_protein` | target | The drug binds to or acts on the protein as its target. |
| `drug.enzyme.gene_protein` | enzyme | The protein is an enzyme that metabolizes the drug. |
| `drug.transporter.gene_protein` | transporter | The protein transports the drug across membranes. |
| `drug.carrier.gene_protein` | carrier | The protein carries the drug in the body. |
| `drug.synergistic_interaction.drug` | synergistic interaction (sym) | The drug has a synergistic interaction with the other drug. |
| `disease.indication.drug` | indication | The disease is treated by the drug. |
| `disease.contraindication.drug` | contraindication | Patients with the disease should not take the drug. |
| `disease.off_label_use.drug` | off-label use | The drug is used off-label for the disease. |
| `disease.phenotype_present.effect_phenotype` | phenotype present | The disease shows the phenotype. |
| `disease.phenotype_absent.effect_phenotype` | phenotype absent | The disease does not show the phenotype. |
| `disease.associated_with.gene_protein` | associated with | The gene or protein is associated with the disease. |
| `disease.parent_child.disease` | parent-child (sym) | The disease is a broader or narrower form of the other disease. |
| `disease.linked_to.exposure` | linked to | The exposure is linked to the disease. |
| `effect_phenotype.side_effect.drug` | side effect | The phenotype is a side effect of the drug. |
| `gene_protein.ppi.gene_protein` | ppi (sym) | The protein physically interacts with the other protein. |
| `gene_protein.target.drug` | target | The protein is a target of the drug. |
| `gene_protein.expression_present.anatomy` | expression present | The gene is expressed in the anatomical structure. |
| `gene_protein.interacts_with.biological_process` | interacts with | The protein takes part in the biological process. |

The four drug-side slots for a disease or phenotype value (indication, contraindication, off-label use, side effect) are the main collision family and the basis of the role-swap test set (13.2).

---

## 6. Document side: from entities to facts

### 6.1 Native facts (from KB edges)

For every KB edge that touches document $d$, create one native fact:

- key $\kappa$ = `k|v`, where $k$ is the slot seen from $d$ and $v$ the other end of the edge;
- $\pi = 1$; no context vector is needed (Section 7.1 uses a constant context factor of 1 for native facts);
- stored as a string in the set `native_keys(d)`.

Direction. Slot ids are always written from the document's side. For an edge $u \xrightarrow{rel} w$:

- if the KB also stores $w \xrightarrow{rel} u$ (the relation is stored in both directions), each stored copy already gives one document its key; do nothing extra;
- if the relation is stored in one direction only, document $u$ gets `type(u).rel.type(w)|w` and document $w$ gets the inverse slot `type(w).rel_inv.type(u)|u`;
- if both directions are stored under the same name for a relation between two nodes of the **same** type (e.g. a parent–child relation between diseases), the direction cannot be recovered; treat the relation as `sym`.

`DEVELOPMENT.md` M2 gives the check that decides this per relation (count reciprocal edges).

### 6.2 Text facts (from sentences)

- Source: every string-valued text field of the entity (description and detail fields), **excluding** any generated listing of the entity's relations (that listing duplicates native facts).
- Split each field into sentences with a rule-based splitter. A sentence longer than 64 words is split at `;`, then cut into 64-word pieces.
- A sentence becomes a text fact only if it contains at least one linked value mention (Section 6.3). Other sentences still feed BM25 and the topic vector.
- For each linked mention the slot scorer (Section 9.3) returns one distribution over the candidate slots — every slot with `doc_type = type(d)` and a `value_type` among the mention's possible types — plus `none`. Its input is the sentence with this mention replaced by `[TARGET:t]` (or `[TARGET]` when the mention has candidates of several types), the other mentions by `[t']`, and the entity's own name by `[SELF:type(d)]`: `Used for [TARGET:disease] in adults.` The field name is **not** part of the scorer input (synthetic training data cannot reproduce the real link between fields and slots); it is part of the context-vector text below.
- `none` is essential: without it, a value type with a single candidate slot (for example a phenotype in a drug document, whose only slot is *side effect*) would always receive probability 1.
- For a slot $k$ and a candidate value $v$ of type `value_type(k)`, $\pi_f(\texttt{k|v})$ is the probability of $k$. On the document side, a surface form that maps to **several nodes of the same type** is skipped (counted and reported), so that lexicon counts stay clean.
- Store only cells with $\pi \ge \tau_{\min} = 0.05$. A sentence with two mentions yields **one** fact with keys for both values.
- Context vector $c_f$: embed the sentence with **all** mentions masked as `[t]` and the own name as `[SELF:type(d)]`, prefixed by the field name. No `TARGET` marker, because the vector belongs to the whole fact.
- Cap: at most `F_text_max = 256` text facts per document (keep the first 256 in field order). Report how many documents hit the cap.

### 6.3 Mention detection and linking

v0 values are KB nodes, so linking is dictionary-based:

1. Build a dictionary from every node's name: normalized name → list of node ids. Normalization follows `conventions` (5.1).
2. Scan text left to right; at each word position try the longest n-gram (up to 8 words) first; accept the longest match; do not allow overlaps.
3. Precision rules (MUST): ignore names shorter than 3 characters; ignore single-word names that are common English words (use a stop list of the 5,000 most frequent English words) unless the match is case-identical to an all-caps gene symbol; ignore pure numbers.
4. A surface form shared by several nodes yields several candidate values. Within one type they get equal link probability $\ell(v \mid t) = 1/n_t$. Candidates of **different** types (a disease and a phenotype with the same name) are not weighted by the linker: the slot scorer decides between them, because its candidate slots cover all the mention's types (9.3).
5. Queries use the same dictionary plus a fuzzy fallback (token-set similarity ≥ 92 on n-grams of 1–6 words) for spans that have no exact match; fuzzy candidates get half the probability mass of exact ones.

v1 replaces step 2 with GLiNER2 span detection for open-vocabulary values (no dictionary).

### 6.4 Vectors and binarization

- **Topic vector** (one per document): embedding of the document text (name + text fields + relation listing), truncated to 256 dimensions (Matryoshka truncation), L2-normalized. Used for dense retrieval and in the final fusion.
- **Fact context vectors** $c_f$ and **segment vectors** $g_s$: same encoder, truncated to 128 dimensions, L2-normalized. Queries and segments use the encoder's query instruction; facts use none.
- **Binary storage (v1 / engine path).** Store $\mathrm{sign}(c_f)$ as 16 bytes (bit = 1 if the component is > 0; first component = most significant bit of the first byte; bytes reinterpreted as two's-complement int8). Similarity with a float query vector: $\mathrm{sim}_{bin}(g, c) = \langle g, \pm 1 \rangle / \sqrt{D}$. For random-like vectors this is close to $\sqrt{2/\pi}\cos \approx 0.8\cos$ *(simulation: cosine 0.782 vs 0.647 for one pair)*. It is a monotone proxy, not the cosine: tune thresholds and the exponent $b$ with the same precision that will be used in production. The numpy path uses float vectors by default and binary vectors for parity tests and ablation A11.

### 6.5 The document record

Each document is stored as:

| Field | Type | Use |
|---|---|---|
| `doc_id` | int (the KB node index) | identity |
| `doc_type` | string | slot compatibility |
| `name` | string | linking, display |
| `text` | string | BM25 |
| `topic` | float[256] | dense retrieval, fusion |
| `native_keys` | set of strings | native facts, value postings |
| `text_facts` | list of {`fact_id`, `keys`: {κ: π}, `ctx`: float[128], `field`, `sentence`} | binding score, evidence display |

The posting index for value-first retrieval maps each key κ (native or text) to the documents that contain it.

---

## 7. Scoring

### 7.1 Definitions

For a reading $r=(s,k,v)$ with key $\kappa_r$ = `k|v` and link probability $\ell_r=\ell_s(v)$:

**Gate of a fact** (does the fact hold this value in this slot, and how sure are we?):

$$\gamma_f(r) = \begin{cases} \ell_r \cdot \pi_f(\kappa_r) & \text{if } \pi_f(\kappa_r) \ge \tau_{\min} \\ 0 & \text{otherwise} \end{cases}$$

**Native part** (native facts have $\pi = 1$ and a constant context factor 1):

$$N_r(d) = \ell_r^{\,a} \cdot \mathbb{1}[\kappa_r \in \texttt{native\_keys}(d)]$$

**Text part, bound** (all factors from the same fact; best fact wins):

$$T_r(d) = \max_{f \in \text{text}(d)} \; \gamma_f(r)^{a} \cdot \psi\big(\langle g_s, c_f \rangle\big)^{b}, \qquad \psi(x) = \max(0, x)$$

**Reading match** and **binding score**:

$$M_r(d) = \max\big(N_r(d),\, T_r(d)\big), \qquad B(d) = \sum_{r \in \mathcal{R}(q)} w_r \, M_r(d)$$

**Final score** (soft fusion; tildes are per-query min–max normalization over the candidate pool; $m$ = number of segments with at least one reading):

$$S(d) = \lambda_B \frac{B(d)}{\max(m,1)} + \lambda_D \,\widetilde{D}(d) + \lambda_L \,\widetilde{L}(d)$$

where $D$ is the dense cosine with the topic vector and $L$ the BM25 score. If $m = 0$ the $B$ term is absent and SlotSpace reduces to the hybrid baseline.

Why the native context factor is a constant. For a native fact the slot is certain and the slot is already weighted by $w_r$, which contains $C(k \mid x_s)$. Comparing $g_s$ with a slot description vector would count the same query evidence twice. With the constant, $w_r N_r(d)$ is simply "probability the user means $r$" × "the document certainly satisfies $r$". The alternative (`native_ctx: slot_vector`) MAY be tried as a variant.

### 7.2 Defaults

$\tau_{\min}=0.05$, $a=b=1$, combine = `max`, $\lambda_B=1$, $\lambda_D=1$, $\lambda_L=0.5$. Tune $a \in \{0.5, 1\}$, $b \in \{0.5, 1, 2\}$, $\lambda_B \in \{0.5, 1, 2\}$, $\lambda_L \in \{0, 0.25, 0.5\}$ on validation only.

### 7.3 Interpretation

If $w_r$ is the probability that $r$ is the intended reading and $M_r(d)$ is read as the probability that $d$ satisfies $r$, then $B(d)$ is the **expected number of query segments that $d$ satisfies**. Ranking by $B$ follows the probability ranking principle under reading uncertainty (assuming independence between segments). This is why $w=p$ is the default weight (Section 10.3).

### 7.4 Properties (with proofs)

**P1 — Binding** (for $b > 0$). $T_r(d) > 0$ if and only if one single text fact $f$ of $d$ stores $\kappa_r$ with $\pi \ge \tau_{\min}$, $\ell_r > 0$, and $\langle g_s, c_f\rangle > 0$.
*Proof.* $T_r$ is a maximum over facts of a product whose factors are all evaluated on the same $f$. A product of non-negative factors is positive iff every factor is positive; a maximum is positive iff some term is. ∎
The unbound variant $\max_f \gamma_f(r)^a \cdot \max_{f'} \psi(\cdot)^b$ can be positive with no such fact (the value from one fact, the context from another), which is the failure in Section 2.4.

**P2 — Zero-support pruning is lossless.** If $H_v[k]=0$ then $M_r(d)=0$ for every document $d$ and every reading $r$ with key `k|v`. Hence removing $r$ leaves every $\sum_{r' \ne r} w_{r'} M_{r'}(d)$ unchanged.
*Proof.* $H_v[k]$ counts documents that store $\kappa_r$ natively or in a text fact with $\pi \ge \tau_{\min}$ (Section 8.1). If it is 0, no document stores the key, every gate is 0, so $N_r = T_r = 0$. ∎
Two conditions make this hold in practice: the lexicon is never older than the index, and $H$ never undercounts (invariant I3, Section 8.5). Renormalizing the remaining weights after pruning is a separate, deliberate choice (Section 10.2).

**P3 — One pass versus fan-out.** In single-pass execution each distinct key's posting list is read once, and the BM25 and dense passes run once. Let segment $i$ have $R_i$ readings with keys $\kappa_{ij}$, and let posting work be $W_{\text{single}} = \sum_{i,j} df(\kappa_{ij})$ (distinct keys). Enumerating all reading combinations as separate searches costs
$$W_{\text{DNF}} = \sum_{i}\sum_{j} df(\kappa_{ij}) \prod_{i' \ne i} R_{i'},$$
so $1 \le W_{\text{DNF}}/W_{\text{single}} \le \max_i \prod_{i'\ne i} R_{i'} \le \prod_i R_i$. One rewrite per reading repeats the BM25 pass, the dense search and the re-ranking $\sum_i R_i$ times instead of once.
*Proof.* Key $\kappa_{ij}$ occurs in exactly $\prod_{i'\ne i} R_{i'}$ combinations. The ratio is a weighted average of these multiplicities, with weights $df(\kappa_{ij})$, so it lies between their minimum (≥ 1) and maximum. ∎

**P4 — Bounds.** With $a,b \ge 0$ and combine = `max`, $0 \le M_r(d) \le 1$. The weights of each segment sum to at most 1 (exactly $1 - C(\texttt{none})$, Section 10.2), so $0 \le B(d) \le m$.

**P5 — Monotonicity.** $M_r(d)$ never decreases when a role probability, a link probability or a context similarity increases, and never decreases when a fact is added to $d$. (Max of non-decreasing functions.) Unit tests check this.

### 7.5 Worked example

Four drug documents, one query segment ("Which drugs should be avoided in patients with **hypertension**?"). Vectors are 4-dimensional for readability: $e_1$ = "treat", $e_2$ = "avoid", $e_3$ = "cause", $e_4$ = neutral. $g = \mathrm{unit}(e_2 + 0.3 e_4)$.

| Doc | Facts |
|---|---|
| A | native `drug.indication.disease`→HTN; text "used to treat [disease]" (π: indication 0.92, contraindication 0.05) with $c = \mathrm{unit}(e_1+0.2e_4)$ |
| B (relevant) | text "do not use in patients with [disease]" (π: contraindication 0.60, indication 0.25, off-label 0.15) with $c = \mathrm{unit}(e_2+0.2e_4)$ |
| C (trap) | native indication→asthma; native side effect→HTN (phenotype); text f1 "do not use in patients with [disease]" about **asthma** with $c=\mathrm{unit}(e_2+0.2e_4)$; text f2 "[disease] was reported in trials" about **HTN** (π: 0.35/0.30/0.35) with $c=\mathrm{unit}(0.3e_1+0.3e_2+0.9e_4)$ |
| D | text "do not use in patients with [disease]" about kidney failure |

Readings after the planner (illustrative link probabilities 0.9 disease / 0.1 phenotype — the v0 linker would give 0.5 / 0.5 for two nodes with the same name; $C$ = 0.15 / 0.80 / 0.05 over indication / contraindication / off-label; lexicon support 120 / 40 / 0 for the disease and 300 for the phenotype side effect; β = 0.5; ε = 1): off-label is pruned (hard support 0); $p$ = contraindication **0.6254**, indication **0.2014**, side effect **0.1732**.

Scores *(computed with the reference implementation)*:

| Doc | bound (SlotSpace) | unbound | gate only | no role |
|---|---|---|---|---|
| A | 0.2032 | 0.2032 | 0.2327 | **0.8268** |
| B | **0.4237** | 0.4237 | 0.4256 | 0.8232 |
| C | 0.3148 | **0.4302** | **0.4313** | 0.6269 |
| D | 0 | 0 | 0 | 0 |
| Ranking | **B, C, A, D** | C, B, A, D | C, B, A, D | A, B, C, D |

What the table shows:

- **Unbound** ranks the trap C first: for the contraindication reading it takes the value from f2 (gate 0.30) and the "avoid" context from f1 (similarity 0.9956).
- **Gate only** also ranks C first: the native side-effect fact opens a gate and nothing checks whether the context is right.
- **No role** ranks A first: A's indication evidence counts for the contraindication reading.
- **Bound** ranks the relevant document first.

`DEVELOPMENT.md` reproduces this table as a unit test.

### 7.6 Variants used by ablations

| Variant | $T_r(d)$ (native part unchanged unless noted) |
|---|---|
| `bound` (default) | $\max_f \gamma_f^a \psi^b$ on the same fact |
| `unbound` | $\max_f \gamma_f^a \cdot \max_{f'} \psi(\langle g_s, c_{f'}\rangle)^b$ over text facts |
| `gate_only` | $\max_f \gamma_f^a$ |
| `no_role` | $\max_{f \ni v} \psi^b$, and the native part counts $v$ in **any** slot |
| combine `sum` | $M_r = N_r + T_r$ instead of the max |

*(simulation, 2,000 documents, 300 queries, seed 7)* `bound` beat `unbound` in all three settings tried (R-precision 0.835 vs 0.483; 0.888 vs 0.635 with a sharper document-side role model; 0.573 vs 0.447 with noisier context vectors). Whether the explicit role distribution adds to the context vector depends on their relative quality: with a weak document-side role model (52.6% accuracy) `no_role` was slightly better (0.880 vs 0.835); with a better role model (80.3%) or weaker context vectors, `bound` was better (0.888 vs 0.880; 0.573 vs 0.536). Ablation A1 therefore answers a real question, and $a, b$ are tuned.

### 7.7 Co-constraint groups (v1)

Some queries need several mentions satisfied by the **same** fact (an n-ary statement, e.g. "approved for X in children"). For a group $G$ of segments:
$$M_G(d) = \max_f \sum_{s \in G}\sum_{r \in s} w_r\, \gamma_f(r)^a\, \psi(\langle g_s, c_f\rangle)^b$$
(sum over the group first, then max over facts). Not in v0 because STaRK's relational requirements are mostly separate edges.

---

## 8. The Ambiguity Lexicon

### 8.1 Definition

For document $d$, value $v$ and slot $k$, the document's contribution is its strongest evidence:
$$c_d(v,k) = \max_{f \in d} \pi_f(\texttt{k|v}) \qquad (\text{native facts give } 1)$$
The lexicon stores two numbers per $(v,k)$:
$$L_v[k] = \sum_{d} c_d(v,k) \quad (\text{soft support}), \qquad H_v[k] = \big|\{d : c_d(v,k) \ge \tau_{\min}\}\big| \quad (\text{hard support}).$$

Document-level aggregation (max inside a document, sum across documents) is deliberate: the planner needs "how many documents could satisfy this reading", not "how many sentences mention it".

### 8.2 What it is used for

1. **Corpus evidence** $E(k \mid v)$ in the reading posterior (Section 9.5).
2. **Lossless pruning**: readings with $H_v[k]=0$ are removed (P2).
3. **Cost estimates**: $H_v[k]$ is the posting length of key `k|v` (Section 10.6).
4. **Diagnostics**: which values collide in this corpus, and how strongly.
5. **Test generation**: corpus-grounded collisions drive the role-swap test set (Section 13.2).

### 8.3 Layers: what is closed and what is open

| Layer | Content | Open or closed | Changed by |
|---|---|---|---|
| L0 | schema and conventions | closed, versioned | a person editing the schema |
| L1 | generic priors per value type (uniform in v0) | closed | configuration |
| L2 | values and their counts $L, H$ | **open**, grows with the corpus | ingest, at commit |
| L3 | evidence pointers: up to 5 fact ids per $(v,k)$ | **open** | ingest, at commit |
| L4 | mined aliases (surface forms) | open but gated (added only after a precision check) | v1 |

The lexicon is built from the documents themselves, not from generic simulated data. Simulated data is used only for tests.

### 8.4 Incremental maintenance

- On ingest of a document version: compute its contribution record $\{(v,k): c_d(v,k)\}$, store the record by `doc_id`, and add it to $L$ and $H$.
- On update: subtract the old record, add the new one. On delete: subtract the old record.
- Every change increases `lexicon.version`. The index and the lexicon commit together. A query reads one consistent `(index_version, lexicon_version)` pair.

### 8.5 Invariants (each has a unit test)

- **I1 (sum).** $L_v[k] = \sum_{d\ \text{live}} c_d(v,k)$ and $H_v[k] = |\{d\ \text{live}: c_d(v,k) \ge \tau_{\min}\}|$.
- **I2 (visibility).** A query never sees a lexicon version older than the index version it searches.
- **I3 (no undercount).** $H_v[k]$ is never smaller than the true count at the visible version. Late deletes are allowed (they overcount); early deletes are not. This keeps pruning lossless.
- **I4 (determinism).** Rebuilding from scratch gives the incremental result: $H$ exactly, $L$ within $10^{-9}$.

### 8.6 Correcting text counts with PACC

Soft counts from text facts inherit the slot scorer's bias: if it over-predicts "indication", $L_v[\text{indication}]$ is inflated. PACC corrects this per slot.

1. Build a **distant-supervision set**: text mentions $(d, v)$ where the KB has an edge between $d$ and $v$. The edge's slot is the truth. (The KB is incomplete, so this set is noisy; it is still independent of the scorer's training data.)
2. For each slot $k$, measure on that set
   $\mathrm{TPR}_k$ = mean of $c_d(v,k)$ over pairs whose true slots include $k$, and
   $\mathrm{FPR}_k$ = mean of $c_d(v,k)$ over pairs whose true slots do not include $k$.
3. For value $v$, let $P$ be the documents where $v$ appears in a text fact but `k|v` is not native, $n=|P|$, and $\bar q$ the mean text contribution over $P$. Then
$$\hat L_v[k] = \underbrace{\#\{d: \texttt{k|v} \in \texttt{native\_keys}(d)\}}_{\text{exact}} + n \cdot \mathrm{clip}\!\left(\frac{\bar q - \mathrm{FPR}_k}{\mathrm{TPR}_k - \mathrm{FPR}_k},\, 0,\, 1\right).$$
   Example: $\bar q = 0.40$, TPR = 0.80, FPR = 0.15, $n = 200$ gives $200 \times 0.3846 = 76.9$ (the raw sum would be 80).
4. Use $\hat L$ in the posterior. Keep the raw $H$ for pruning (pruning must stay lossless, not accurate).

If $\mathrm{TPR}_k - \mathrm{FPR}_k < 0.05$ the correction is unstable; fall back to the raw mean and log a warning.

### 8.7 Out-of-vocabulary values

- **Value OOV**: the mention links to nothing, or to values with $H=0$ in every compatible slot. The segment gets no readings and is served by BM25 and dense search. Log it; a high rate means the dictionary or the schema is missing something.
- **Surface OOV**: the value exists but this surface form is unknown. The fuzzy fallback (6.3, step 5) handles small spelling differences; alias mining (L4) is v1.

---

## 9. Query understanding

### 9.1 Segments

Run the linker (6.3) on the query. Every mention with at least one candidate value becomes a **segment** $s$ with a link distribution $\ell_s(v)$. Longest match wins; overlapping mentions are not allowed.

### 9.2 Segment vector $g_s$

1. Mask the query: the target mention becomes `[TARGET:t]` (`[TARGET]` without a type if its candidate values have different types, as in 6.2); every other mention becomes `[t']` (`[entity]` if it has several types).
2. Keep a window of $W = 8$ words on each side of the target (the whole query if it is shorter).
3. Embed with the query instruction `Instruct: Given a constraint taken from a search query, retrieve statements that express the same relation\nQuery:`; truncate to 128 dimensions; L2-normalize again after truncation.

### 9.3 The slot scorer (context compatibility $C$)

Interface (both implementations MUST follow it):

```
score(masked_contexts: list[str], candidates: list[list[slot_id]], temperature: float)
    -> list[dict[slot_id, probability]]     # one distribution per input, over its candidates
```

Candidates come from the schema: `schema.candidates(doc_type or None, value_types)` returns the slot ids plus `none.<type>` for each type. The caller passes the temperature of the style (`query` or `doc`).

- Candidate slots for a query mention: every slot whose `value_type` is one of the mention's possible types (any `doc_type`), plus `none`. For queries this lets the context decide the answer type ("Which **drugs** ..." puts mass on `drug.*` slots) and, for a name shared by several types, the value type. For document mentions the `doc_type` is fixed to the document's type.
- The output is one joint distribution over these candidates. The planner reads it per value type as $C(k \mid x_s)$ for slots $k$ of that type, and $C(\texttt{none} \mid x_s)$ separately.
- v0 implementation: a **cross-encoder** scores each pair (masked context, slot text), where slot text = `"<slot id>: <description>"`. The scores $z_k$ of the candidate slots become a distribution with a temperature: $C(k \mid x) = \mathrm{softmax}_k(z_k / T)$.
- Base model: `answerdotai/ModernBERT-base` (149M parameters, Apache-2.0), fine-tuned as in Section 11. The zero-shot ablation (A13) uses `BAAI/bge-reranker-v2-m3` (Apache-2.0, multilingual, already trained as a reranker) without fine-tuning.
- The same model serves both styles: query mentions (`style=query`) and document mentions (`style=doc`, producing $\pi$ for text facts). Each style has its own temperature.
- The input never contains the value string (Section 11.1).

### 9.4 Calibration

- Fit the temperature $T_{query}$ on the silver calibration split (Section 11.4) by minimizing the negative log-likelihood of the true slot among the candidate slots. Fit $T_{doc}$ the same way on the distant-supervision set (Section 8.6).
- Report ECE with 15 equal-width bins before and after scaling.
- Temperature scaling does not change the argmax, so it cannot change slot accuracy; it only makes the probabilities usable as weights and for conformal thresholds.

### 9.5 The reading posterior

Assume the scorer was trained with a **uniform prior** over the candidate slots of each value type (balanced data, Section 11.2). Then $C(k \mid x) \propto P(x \mid k)$: it measures only how well the context fits slot $k$.

The intended slot also depends on the value. The prior-shift rule of Saerens et al. (2002) re-weights a posterior trained under prior $\pi_{train}$ to a new prior $\pi_{new}$: $p_{new}(k \mid x) \propto p_{train}(k \mid x)\, \pi_{new}(k) / \pi_{train}(k)$. With a uniform $\pi_{train}$ and $\pi_{new}(k) \propto E(k \mid v)^{\beta}$, and with the link probability for the value:

$$p_r = p(k, v \mid q, s) = \frac{\ell_s(v)\; C(k \mid x_s)\; E(k \mid v)^{\beta}}{\sum_{(k',v')} \ell_s(v')\; C(k' \mid x_s)\; E(k' \mid v')^{\beta}}, \qquad E(k \mid v) = \frac{\hat L_v[k] + \varepsilon}{\sum_{k'} \hat L_v[k'] + \varepsilon\, n_t}$$

- $\ell_s(v)$ is the **within-type** link probability $\ell(v \mid \tau(v))$ (6.3); the choice between types is already inside $C$. (The worked example in 7.5 uses illustrative numbers in which this product is written as link × $C$; the formula is the same.)
- The sums run over the candidate slots of each value's type; $n_t$ is their number; $\varepsilon = 1$.
- The segment as a whole is a slot constraint with probability $1 - C(\texttt{none} \mid x_s)$; this becomes the segment's total weight in 10.2.
- Readings with $H_v[k]=0$ get $p_r = 0$ and are removed (lossless by P2).
- $\beta$ controls trust in the corpus. $\beta = 1$: the user's intent follows the corpus distribution of the value's slots. $\beta = 0$: ignore the corpus. $0 < \beta < 1$ (default 0.5): tempered, so a rare but intended reading is not crushed. Role-swap queries ask for the rare slot half of the time, which is exactly where $\beta = 1$ would hurt.
- Tune $\beta \in \{0, 0.25, 0.5, 1\}$ by plan regret on validation (10.7).

Worked example: Section 7.5 (posterior 0.6254 / 0.2014 / 0.1732 after pruning the unsupported off-label reading).

### 9.6 Answer type

No separate answer-type model in v0. The slot ids carry the document type and the scorer sees the whole masked query, so the answer type is decided jointly with the slot. v1 MAY add an answer-type prior.

---

## 10. Planner and execution

### 10.1 The plan object

The planner turns a query into a plan. The plan is logged for every query (it is the main debugging tool).

```json
{
  "query_id": "prime-test-00042",
  "schema_version": "0.1.0",
  "lexicon_version": 129375,
  "segments": [
    {
      "span": [36, 48], "text": "hypertension",
      "links": {"n101": 0.9, "n202": 0.1},
      "c_none": 0.0,
      "readings": [
        {"key": "drug.contraindication.disease|n101", "p": 0.6254, "w": 0.6254, "support_hard": 44},
        {"key": "drug.indication.disease|n101",       "p": 0.2014, "w": 0.2014, "support_hard": 131},
        {"key": "drug.side_effect.effect_phenotype|n202", "p": 0.1732, "w": 0.1732, "support_hard": 300}
      ],
      "pruned_zero_support": ["drug.off_label_use.disease|n101"],
      "pruned_conformal": []
    }
  ],
  "path": "both",
  "budgets": {"K_L": 200, "K_D": 200, "K_V": 500},
  "weight_objective": "p",
  "estimated_value_postings": 475
}
```

### 10.2 Choosing the reading set

1. Compute $p_r$ for every reading of each segment (9.5). Readings with zero hard support are already gone.
2. **Conformal threshold.** On the calibration split, for each segment with a silver true slot set $J^*$, record $p_{true} = \max_{r \in J^*} p_r$. Compute
   $$t_\alpha = 1 - \hat q, \qquad \hat q = \text{the } \lceil (n+1)(1-\alpha) \rceil / n \text{ empirical quantile of } \{1 - p_{true,i}\}_{i=1}^n$$
   (if that level exceeds 1, set $t_\alpha = 0$ and keep everything).
3. Keep $\{r : p_r \ge t_\alpha\}$; always keep the top-1 reading; keep at most $R_{\max} = 6$ per segment.
4. Renormalize the kept readings within each segment so that they sum to $1 - C(\texttt{none} \mid x_s)$, the probability that the segment is a slot constraint at all. (A deliberate choice: real constraints count equally in $B$, and a mention the scorer reads as `none` contributes little.)

**Guarantee.** If calibration and test segments are exchangeable, the best true reading is kept with probability at least $1-\alpha$ (marginally over segments). STaRK's synthesized test queries come from the same generator as its training and validation queries, so the assumption is plausible there. For human-written queries it is not guaranteed; report the observed miss rate. Default $\alpha = 0.1$; ablation A10 sweeps $\{0.05, 0.1, 0.2\}$ and "keep all".

Worked example *(reference implementation)*: with 19 calibration values and $\alpha = 0.1$ the level is $\lceil 20 \times 0.9 \rceil/19 = 0.947$ and $t = 0.35$; the example query of 7.5 keeps only the contraindication reading. With $\alpha = 0.2$, $t = 0.54$. With $\alpha = 0.02$ the level exceeds 1, so $t = 0$.

v1: when a segment has several true readings that must all be kept, use conformal risk control with a false-negative-rate loss instead.

### 10.3 Reading weights by objective

| `weight_objective` | Weights | What it optimizes |
|---|---|---|
| `p` (default) | $w_r = p_r$ | expected number of satisfied segments (7.3) |
| `p_over_H` | $w_r \propto p_r / \max(1, H_v[k])$, renormalized per segment | expected per-reading recall: a reading with few matching documents gets more weight per document |
| `ia` | rank with $w=p$, then re-order the top 100 with IA-Select | coverage: probability that the user's intended reading has at least one good result near the top |

For `ia`, readings play the role of IA-Select's intent categories: $P(c \mid q) = p_r$, $V(d \mid q, c) = M_r(d) \in [0,1]$. Greedy step: pick $d^* = \arg\max_d \sum_r U_r V_r(d)$, then $U_r \leftarrow U_r (1 - V_r(d^*))$, starting from $U_r = p_r$. Worked example *(reference implementation)*: with $P$ = {ind 0.3, con 0.7} and $V$ = {ind: A 0.9, C 0.3; con: B 0.8, D 0.6, C 0.2}, IA-Select returns B, A, D while ranking by expected relevance returns B, D, A, C.

### 10.4 Execution paths and fallbacks

- No segment, or no segment with readings: path `vector_first`. Candidates = BM25 top-$K_L$ ∪ dense top-$K_D$. $S$ = hybrid score only.
- Otherwise: path `both`. Candidates = BM25 top-$K_L$ ∪ dense top-$K_D$ ∪ value postings of the kept readings' keys.
- If the union of value postings exceeds $K_V$, keep the $K_V$ documents with the highest dense score among them.
- BM25 and dense candidates are **always** included. This is the coverage fallback when linking or slot decisions are wrong.

Defaults: $K_L = K_D = 200$, $K_V = 500$. All candidates are scored with $S(d)$ in the numpy path; the top 100 are returned.

### 10.5 Hard AND versus soft fusion

A hard AND keeps only documents that satisfy at least one reading of **every** segment ($M_r(d) > 0$). It is cheaper to score but drops relevant documents whenever linking or the slot decision is wrong, or the KB is incomplete. SlotSpace uses soft fusion (7.1) and keeps hard AND only as ablation A6.

### 10.6 Cost model and counters

Each query run records:

| Counter | Definition |
|---|---|
| `bm25_postings` | sum over distinct query terms of their document frequency |
| `dense_searches` | number of dense search passes (1 for SlotSpace) |
| `value_postings` | sum over distinct kept keys of $H_v[k]$ |
| `candidates` | size of the candidate pool |
| `facts_scored` | number of (candidate, reading, text fact) triples evaluated |
| `encoder_texts` | texts encoded at query time (query, segments, cross-encoder pairs) |
| `latency_ms` | per stage, median of 3 warm runs |

For the fan-out baseline (B6) the same counters are summed over its rewrites. Always report quality next to these counters, and plot Recall@100 against `bm25_postings + value_postings` and against latency.

In single-pass mode, pruning removes little work (a pruned key would add only its own postings); its main effect is on the weights. In fan-out mode, every pruned reading saves a whole search. Section 7.4 P3 gives the ratio.

### 10.7 Plan regret (the selection metric for models and hyperparameters)

$$\mathrm{Regret}(q) = \mathrm{RR}(q;\ \text{oracle plan}) - \mathrm{RR}(q;\ \text{plan})$$

- RR = reciprocal rank of the first relevant document.
- The **oracle plan** uses the same pipeline but replaces each segment's readings by its silver true readings with equal weights.
- Model checkpoints, temperatures, $\beta$, $\alpha$, $a$, $b$ and $\lambda$ are selected by the lowest mean regret on validation, subject to a cost budget. Report mean regret with a 95% bootstrap interval (1,000 resamples).

Why not slot accuracy: a slot mistake that does not change the ranking costs nothing, and one that moves a relevant document out of the top 10 costs a lot. Regret measures what the planner is for.

### 10.8 Pseudocode

```
def search(q):
    segments = link(q)                                   # 6.3
    for s in segments:
        x_s = mask_query(q, s)                           # 9.2
        s.g = embed_query(window(x_s, W=8), dim=128)     # 9.2
        C_s = slot_scorer.score([x_s], s.types, "query") # 9.3 (calibrated)
        post = reading_posterior(C_s, s.links, lexicon, beta)   # 9.5, prunes H = 0
        s.readings = keep_readings(post, t_alpha, min_keep=1, r_max=6)  # 10.2
    readings = set_weights(segments, objective)           # 10.3
    cand = bm25_top(q, K_L) | dense_top(q, K_D) | postings(keys(readings), K_V)  # 10.4
    scores = {d: fuse(B(d, readings), dense(q, d), bm25(q, d)) for d in cand}   # 7.1
    ranked = top(scores, 100)
    if objective == "ia": ranked = ia_select(ranked, readings)
    return ranked, evidence(ranked, readings), counters
```

`evidence(d, r)` returns the fact (id, field, sentence or edge) that produced $M_r(d)$. It is free (the max already found it) and is the system's explanation.

---

## 11. The small model: schema-driven data and fine-tuning

### 11.1 What the model learns, and what it must not learn

- It learns $P(\text{slot} \mid \text{masked context}, \text{value type})$.
- It never sees the value string: the target is `[TARGET:t]`, other mentions are `[t]`, the document's own name is `[SELF:doc_type]`.
- Reason: value-specific priors ("hypertension is usually an indication") are corpus facts. They come from the lexicon at query time (9.5) and update when the corpus changes. A model that memorized them would double-count them and would go stale.
- Test of this rule: ablation A12 trains the same model with values visible and compares on the role-swap set (where intended slots are balanced, unlike corpus frequencies).

### 11.2 Data generation (per schema)

1. **Templates.** For every slot write at least 12 query-style templates and 12 document-style templates using the slot's cues. Placeholders: `{T}` (target), `{O:<type>}` (another mention), `{SELF}`, `{ANS}` (answer-type phrase for queries, e.g. "drugs", "medications").
   - Query example (`drug.contraindication.disease`): "Which {ANS} should not be given to people with {T}?"
   - Document example: "{SELF} is contraindicated in patients with {T} or {O:disease}."
2. **Surrogate tokens.** Fill each placeholder with a unique pseudo-word (e.g. `Zorvex17`, `Quilamid4`) so that a paraphraser keeps them intact and they can be found again. After paraphrasing, replace them with the mask tokens of 11.1.
3. **Paraphrasing.** A local Apache-2.0 instruct model (default `Qwen/Qwen3-4B-Instruct-2507`, offline with vLLM) writes 3 paraphrases per filled template (temperature 0.7, top-p 0.8). Prompt: keep every pseudo-word exactly once, keep the meaning, vary the wording.
4. **Validation (rule-based, no LLM judge).** Reject a sample if: a pseudo-word is missing or repeated; it has fewer than 4 or more than 60 words; it contains a cue phrase of a colliding slot (lexical check); it duplicates another sample after normalization.
5. **Hard negatives.** For every colliding slot pair $(k_1,k_2)$ create minimal pairs: the same frame with only the cue changed ("used to treat {T}" / "should not be used in {T}").
6. **Multiple mentions.** 30% of samples contain a second mention `{O:type}` with a different slot cue, so the model learns to read the cue attached to the target.
7. **`none` samples.** For every value type, templates where the value is mentioned without any slot cue ("{T} is a common condition", "Studies of {T} are ongoing"), in both styles, with the same count as a slot.
8. **Untyped targets.** In 20% of samples write the target mask without its type (`[TARGET]`), so the model handles names shared by several types.
9. **Balance.** The same number of samples per slot (and for `none`) within each value type (uniform training prior; required by 9.5).
10. **Size.** 2,000 samples per slot (1,000 per style). With about 50 slots plus one `none` per value type that is roughly 120,000 samples; reduce to 1,000 per slot if generation time is a problem.
11. **Held-out slot.** Generate data for all slots, but train a second model without one chosen slot (default: the off-label-use slot) to measure zero-shot transfer through its description.

### 11.3 Training

- Cross-encoder on pairs (masked context, slot text). For each sample: one positive pair (its slot or `none`, label 1) and one negative pair for every other candidate of the same value type, including `none` (label 0). The slot text of `none` is `"none: The text mentions the <value type> but states none of the relations listed for it."`
- Loss: binary cross-entropy with `pos_weight` = negatives per positive. (A listwise softmax loss is an allowed alternative; keep whichever has lower validation regret.)
- Defaults: 2 epochs, learning rate 2e-5, batch size 32, maximum length 128 tokens, 10% warmup, bf16, evaluate every 1,000 steps on held-out templates, keep the best checkpoint.
- At inference: scores of the candidate slots → softmax with temperature (9.4).

### 11.4 Silver labels from the benchmark itself

For a STaRK **train or validation** query with answer set $A$ and a linked segment value $v$:
$$J^*(s) = \big\{k : |\{a \in A : \texttt{k|v} \in \texttt{native\_keys}(a)\}| \ge \max(1, \lceil 0.5 |A| \rceil)\big\}$$
If $J^*$ is empty, the segment is not a relational constraint: it is a `none` example for temperature fitting and is excluded from the conformal threshold (which concerns readings). These labels come from the knowledge base, not from any LLM, and they are never computed on test queries. Uses: temperature (9.4), conformal threshold (10.2), plan regret (10.7), slot accuracy reports. Before use, a person checks 100 random silver labels and records the agreement rate in `docs/DECISIONS.md`.

Split the silver set: 50% calibration (temperature, conformal), 50% selection (regret).

### 11.5 Rules against self-validation

- The role-swap **test** templates (13.2) are written separately, stored in their own file, and never used for training or paraphrasing.
- No paraphraser output ever enters a test set.
- The main results are on STaRK's own queries, which our generator did not write.
- Report the held-out-slot model (11.2 step 9) to show that schema descriptions transfer.

### 11.6 v1 options

- **Bi-encoder distillation.** Train a bi-encoder to imitate the cross-encoder. Store one span vector per text mention; a schema change then needs only a matrix product with the new slot vectors, not a new pass of the cross-encoder.
- **GLiNER2** for open-vocabulary values (Amazon attributes, MAG topics): one model for span detection and slot labels with descriptions (`fastino/gliner2-base-v1`, 205M, Apache-2.0).
- **Per-schema LoRA adapters** on one base model.
- **Multilingual**: `jhu-clsp/mmBERT-base` (MIT) or `BAAI/bge-reranker-v2-m3` as the base.

---

## 12. Execution paths: numpy research path and Vespa engine path

### 12.1 Two paths, one math

- **Research path (v0, MUST).** Pure Python: exact dense search with numpy, BM25 with `bm25s`, Python dictionaries for postings, and a batched numpy scorer that reproduces the reference implementation. All v0 results come from this path.
- **Engine path (v1, SHOULD after v0 passes).** A Vespa application package that computes the same binding score inside the engine's second phase. Used for latency, memory and scaling measurements, and to show the design works in a real search engine without forking it.
- **Parity rule.** On 200 fixed validation queries, for every candidate the engine returns, the engine's `bind` equals the reference implementation's $B(d)$ computed with **binary** context vectors within $10^{-3}$ (absolute), and the order of documents by $B$ is identical except for ties.

The client (Python) always does linking, slot scoring, the posterior and the plan. The engine receives the plan as query tensors.

### 12.2 Engine keys

Vespa tensor labels and YQL maps are safest with `[a-z0-9_]`. Map each readable key `slot|value` to an **engine key** `s<slot_index>_<value_id>`, e.g. `drug.contraindication.disease|n101` → `s7_n101`. `slot_index` is the slot's position in the schema file (stable within a schema version).

### 12.3 Document fields

| Field | Vespa type | Indexing | Purpose |
|---|---|---|---|
| `text` | `string` | `index \| summary`, `enable-bm25` | BM25 |
| `topic` | `tensor<bfloat16>(x[256])` | `attribute \| index`, `prenormalized-angular` | dense retrieval (HNSW) |
| `vkeys` | `weightedset<string>` | `attribute`, `fast-search` | value-first retrieval: all native and text keys, weight 1 |
| `nk` | `tensor<float>(key{})` | `attribute`, `fast-rank` | native keys, cell value 1.0 |
| `tk` | `tensor<float>(fact{}, key{})` | `attribute`, `fast-rank` | text-fact role probabilities ≥ $\tau_{\min}$ |
| `tc` | `tensor<int8>(fact{}, x[16])` | `attribute` | binarized text-fact context vectors (128 bits) |

`vkeys` and `nk` hold the same native keys twice on purpose: weighted sets are fast for matching, but converting a weighted set into a tensor at ranking time costs time linear in its size, and Vespa recommends native tensor fields for ranking.

### 12.4 Query inputs

| Input | Type | Content |
|---|---|---|
| `query(rk)` | `tensor<float>(r{}, key{})` | reading → its engine key(s), value $\ell_r^{a}$ |
| `query(w)` | `tensor<float>(r{})` | reading weights $w_r$ |
| `query(g)` | `tensor<float>(r{}, x[128])` | segment vector, repeated for each reading of that segment |
| `query(qt)` | `tensor<float>(x[256])` | dense query vector |

Matching (YQL): `nearestNeighbor` on `topic` OR `userQuery()` (weakAnd over the text) OR `dotProduct(vkeys, @vk)`, where `vk` maps the kept readings' engine keys to integer weights `round(1000 * w_r)`.

### 12.5 The rank expression

The binding score of 7.1 in tensor algebra. Mapped dimensions join as **inner joins**: a cell that exists in only one argument disappears. The formula below relies on this.

- `native(r)` = `reduce(query(rk) * attribute(nk), max, key)` — $N_r$; only readings with a native match get a cell.
- `gate(r, fact)` = `reduce(query(rk) * attribute(tk), max, key)` — $\gamma_f(r)$; only facts that store the key get a cell.
- `simf(r, fact)` = `reduce(query(g) * (2 * unpack_bits(attribute(tc), float) - 1), sum, x) / sqrt(128)` — `unpack_bits` yields 0/1 cells, so `2·bits − 1` gives ±1.
- `textpart(r)` = `reduce(gate * map(simf, f(x)(max(0, x))), max, fact)` — $T_r$; the inner join keeps only gated facts.
- `bind` = `sum(query(w) * native) + sum(query(w) * textpart) - sum(query(w) * join(native, textpart, f(x,y)(min(x,y))))`

The last line computes $\sum_r w_r \max(N_r, T_r)$ using $\max(x,y) = x + y - \min(x,y)$. The `min` join is an inner join, so it is non-zero only for readings that have both parts, which is exactly where the correction is needed; a reading with only one part contributes that part. A plain element-wise `max` join would silently drop readings that have only one part.

For $a \ne 1$ or $b \ne 1$, generate a rank profile per setting: replace `attribute(tk)` by `map(attribute(tk), f(x)(pow(x, A)))` and `max(0,x)` by `pow(max(0,x), B)` with literal numbers.

Phases:

- first-phase (every matched document, must be cheap): `bm25(text) + 10 * dense + 0.002 * rawScore(vkeys)` where `dense = sum(query(qt) * cell_cast(attribute(topic), float))`. The constants only decide which documents reach the second phase.
- second-phase (`total-rerank-count: 500`): `bind`.
- `match-features`: `bind`, `dense`, `bm25(text)`. The client normalizes and fuses them exactly as the research path does (7.1).

`dense` is computed explicitly instead of using `closeness(...)`, because `closeness` is only defined for documents matched by `nearestNeighbor`, and the fusion needs a dense score for every candidate.

The exact `.sd`, `services.xml`, feed and query formats are in `DEVELOPMENT.md` M9, together with a smoke test. Any syntax that fails the smoke test is fixed there, never by changing the math.

---

## 13. Evaluation plan

### 13.1 Datasets and splits

| Set | Size | Use |
|---|---|---|
| STaRK-Prime knowledge base | 129,375 entities, all candidates; 8,100,498 stored edges | the corpus |
| Prime `train` | 6,162 queries | silver labels for calibration (11.4); never for reported numbers |
| Prime `val` | 2,241 | tuning and model selection |
| Prime `test` | 2,801 | main results (H5) |
| Prime `test-0.1` | 280 | quick checks; the subset on which the STaRK paper ran LLM rerankers |
| Prime human-generated | 98 | directional evidence only |
| Role-swap dev / test (ours, 13.2) | target ≥ 1,000 test queries | H1, H2, H4 |
| v1: STaRK-MAG, STaRK-Amazon | papers (about 0.7M candidates) / products (about 0.9M candidates) | generality |

STaRK queries carry no metadata (no relation template, no sampled entities); everything SlotSpace needs about relations comes from the knowledge base.

### 13.2 The KB-grounded role-swap test set

Purpose: measure binding and slot decisions on queries where the same value appears in different slots, with exact answers taken from KB edges (no LLM in the loop).

Construction:

1. **Same-type collisions.** For every disease value $v$, take the native hard support $H_v[k]$ for $k \in$ {`drug.indication.disease`, `drug.contraindication.disease`, `drug.off_label_use.disease`}. A slot is eligible if $3 \le H_v[k] \le 100$. $v$ is a collision value if at least two slots are eligible.
2. **Cross-type collisions.** A normalized name shared by a disease node and an effect/phenotype node, with `drug.indication.disease` eligible for the disease node and `drug.side_effect.effect_phenotype` eligible for the phenotype node.
3. **Queries.** For each collision value and each eligible slot, write one query from the **test-only** template family of that slot (`DEVELOPMENT.md` Appendix D), choosing the template by a hash of (value id, slot) so the choice is deterministic. Insert the value's KB name. The answers are all documents with the native key `k|v`.
4. **Pairs.** Store, for each query, the ids of its partner queries (same value or same name, other slots) and their answer sets.
5. **Split by value.** `hash(value_id) mod 2`: 0 → dev, 1 → test. All queries of one value go to the same split.
6. **Size check.** Report the number of test queries and the MDE for that size (13.6). If there are fewer than 600, lower the bound in step 1 to 2 and record the change in `docs/DECISIONS.md`.
7. **Human check.** A person reads 50 random queries for naturalness and records problems.

Role-swap metric in addition to the standard ones:

$$\text{Role-Confusion@10}(q) = \frac{\big|\text{top10}(q) \cap \big(\bigcup_{q' \in \text{partners}(q)} A(q') \setminus A(q)\big)\big|}{10}$$

the share of the top 10 that answers a partner query (the wrong role) but not this one. Lower is better.

### 13.3 Metrics

- **Quality** (official STaRK evaluator): `hit@1`, `hit@5`, `recall@20`, `mrr`; plus `recall@100` (coverage) and `rprecision`.
- **Role-swap**: the same plus Role-Confusion@10.
- **Planner**: miss rate of the best true reading (conformal), ECE of $C$ before and after temperature scaling, slot accuracy, mean plan regret.
- **Lexicon**: relative error of text-derived counts (raw and PACC) against native counts on the distant-supervision subset.
- **Cost**: the counters of 10.6 and latency.

### 13.4 Baselines

Re-run with the same embedding model, the same BM25 and the same candidate set rules:

| ID | Baseline | Notes |
|---|---|---|
| B0 | BM25 | `bm25s` over the document text |
| B1 | Dense | Qwen3-Embedding-0.6B topic vector (256 dimensions; 1024 also reported) |
| B2 | Hybrid | min–max fusion of B0 and B1, weight tuned on `val`; identical to SlotSpace with $\lambda_B = 0$ |
| B3 | Multi-vector chunks (MaxP style) | dense max over sentence chunks, fused with B0 |
| B4 | Field-weighted ("mFAR-lite") | one dense and one BM25 score per field (name, details, one field per relation listing), field weights tuned on `val`; no query conditioning, no fine-tuning |
| B5 | Unbound SlotSpace | same as ablation A3 |
| B6 | Fan-out | one rewritten query per kept reading (original query + the reading in words, e.g. "The drug is contraindicated in hypertension."), each run as B2, merged by weighted reciprocal rank fusion (weights $p_r$, $k = 60$). Same plan as SlotSpace, so only the execution differs |
| B7 | Oracle readings | SlotSpace with the silver true readings; `val` only (upper bound for plan regret) |

Cited for context, not re-run (different encoders; mFAR fine-tunes its encoder and uses documents with up to two hops):

| Source (Prime, synthesized test) | Hit@1 | Hit@5 | Recall@20 | MRR |
|---|---|---|---|---|
| STaRK paper, BM25 | 12.75 | 27.92 | 31.25 | 19.84 |
| STaRK paper, ada-002 | 12.63 | 31.49 | 36.00 | 21.41 |
| STaRK paper, multi-ada-002 | 15.10 | 33.56 | 38.05 | 23.49 |
| mFAR (All+2) | 40.0 | — | 72.6 | 52.0 |

### 13.5 Ablations

| ID | Change from A0 | Question | Pass line |
|---|---|---|---|
| A0 | none (defaults, tuned on `val`) | — | — |
| A1 | `no_role` | does the role distribution add to the context vector? | role-swap Hit@1 of A0 − A1 ≥ MDE; otherwise report "context vectors suffice" |
| A2 | `gate_only` | does the context check matter? | Role-Confusion@10 lower for A0 |
| A3 | `unbound` | binding (H1) | role-swap Hit@1 gain ≥ MDE and lower Role-Confusion@10 |
| A4 | $\beta = 0$ | corpus evidence (H2) | lower plan regret on `val` and role-swap Hit@1 gain ≥ MDE |
| A5 | $C$ uniform over the candidate slots, $C(\texttt{none}) = 0$ | query model | lower plan regret for A0 |
| A6 | hard AND | soft fusion | Recall@100 of A0 ≥ A6 + 1 point on Prime `test` |
| A7 | 128-token chunks instead of sentence facts | fact granularity | role-swap Hit@5 of A0 ≥ A7 + 3 points; Recall@100 drop ≤ 1 point |
| A8 | lexicon without PACC | count correction | PACC relative error ≤ 5% and lower than raw |
| A9 | weights `p` / `p_over_H` / `ia` | objective | report the trade-off (no pass line) |
| A10 | $\alpha \in$ {0.05, 0.1, 0.2, keep all} | conformal pruning (H4) | miss rate ≤ α + 0.02; work decreases as α grows |
| A11 | binary context vectors | storage versus quality | Hit@1 drop ≤ 1 point; added storage ≤ 2 KiB per document on average |
| A12 | value visible in the slot-scorer input | "learn C only" (11.1) | A0 ≥ A12 on role-swap Hit@1 |
| A13 | zero-shot slot scorer (bge-reranker-v2-m3, no fine-tuning) | value of schema-driven fine-tuning | lower plan regret for A0 |
| A14 | slot held out of training (off-label use) | schema transfer through descriptions | report accuracy on the held-out slot |

v1 ablations: fixed dimensional encodings (MUVERA-style) for a single ANN pass over facts; co-constraint groups (7.7); the engine path versus the research path.

### 13.6 Statistics

- Keep per-query metric vectors for every run.
- Paired tests between two systems on the same queries: two-sided paired t-test (`ranx` `student`) and randomization test (`ranx` `fisher`). `ranx` has no multiple-testing correction, so apply Holm–Bonferroni ourselves across each table's family of comparisons.
- 95% bootstrap intervals (1,000 resamples) for every reported difference.
- Minimum detectable Hit@1 difference (α = 0.05 two-sided, power 0.8, $\text{MDE} \approx (z_{1-\alpha/2m} + z_{0.8})\sqrt{p_d/n}$ with discordant-pair rate $p_d$ and $m$ comparisons) *(computed)*:

| Set | n | $p_d$ = 0.1 | $p_d$ = 0.2 | $p_d$ = 0.3 | $p_d$ = 0.2, 6 comparisons |
|---|---|---|---|---|---|
| Prime human-generated | 98 | 8.9 | 12.7 | 15.5 | 15.7 |
| Prime test-0.1 | 280 | 5.3 | 7.5 | 9.2 | 9.3 |
| Prime val | 2,241 | 1.9 | 2.6 | 3.2 | 3.3 |
| Prime test | 2,801 | 1.7 | 2.4 | 2.9 | 2.9 |
| Role-swap test (target) | 1,000 | 2.8 | 4.0 | 4.9 | 4.9 |
| Role-swap test (minimum) | 600 | 3.6 | 5.1 | 6.3 | 6.4 |

Values are Hit@1 points. Measure the actual $p_d$ from the runs and recompute. Main claims rest on Prime `test` and the role-swap test set; the human-generated set is directional only.

### 13.7 What every report contains

- Table 1: Prime `test` — baselines and SlotSpace: Hit@1, Hit@5, Recall@20, MRR, Recall@100, cost counters.
- Table 2: role-swap `test` — the same plus Role-Confusion@10.
- Table 3: ablations A0–A14.
- Table 4: planner — ECE before and after scaling, miss rate for each α, mean plan regret.
- Figure 1: Recall@100 against work (postings) and against latency for SlotSpace, B2 and B6.
- Figure 2: reliability diagrams for $C$ (query style) and $\pi$ (document style).
- Every table shows n, the MDE, and Holm-adjusted p-values.
- A "negative results" paragraph: every pass line that failed.

---

## 14. Risks and limitations

| Risk | Effect | Mitigation |
|---|---|---|
| STaRK's synthesized queries are written by LLMs | style bias; the generator's habits may favor some systems | also report the human-written set (directional, small n) and the KB-grounded role-swap set |
| Self-validation | our own templates could appear in training and testing | rules in 11.5; separate files; held-out slot |
| Noisy silver labels | wrong temperature, conformal threshold or regret | 100-sample human check; silver labels are used for calibration and selection only, never as test truth |
| Incomplete knowledge base | missing edges look like irrelevant documents | text facts and the BM25/dense fallback; the role-swap answer sets inherit the gap equally for all systems |
| Linking errors (gene symbols, common words) | wrong values, wrong readings | precision rules (6.3); linking statistics in every report; human check of 100 linked queries |
| Strong baselines already capture roles from text | small gains on the full test | this is a valid outcome; H1 is tested on the role-swap set, H5 on the full set |
| Over-claiming novelty | reviewers reject | Section 2.8; claim the combination and the measured effects only |
| Engine syntax changes across Vespa versions | the engine path breaks | smoke tests and parity tests; the numpy path stays the reference |
| Data license (STaRK is CC-BY-4.0) | redistribution obligations | attribute STaRK; publish code, templates, and test-set ids plus build scripts; do not re-host the raw data |
| Compute | one 24 GB GPU | v0 on Prime needs roughly 10 GPU-hours including reruns (estimate); embeddings are cached once |

Limitations to state in any write-up: values must be linkable (v0 uses the KB dictionary); time ambiguity is not covered; $C$ and $E$ are combined with a fixed functional form; the conformal guarantee is marginal, not per query.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| Binding | Requiring that value, role probability and context come from the same fact |
| Candidate pool | Documents scored in the second phase: BM25 top-K ∪ dense top-K ∪ value postings |
| Collision (schema collision) | A value used in two or more slots in this corpus |
| Context compatibility $C$ | Slot-scorer probability that the masked context expresses a slot |
| Corpus evidence $E$ | Share of a value's lexicon support that falls in a slot |
| Distant supervision | Labels taken from KB edges instead of human annotation |
| Fact | Smallest unit of evidence: one KB edge (native) or one sentence (text) |
| Fan-out | Running one rewritten query per reading and merging the results |
| Gate | $\gamma_f(r)$: whether and how strongly a fact holds the reading's value in its slot |
| Hard / soft support | $H$: number of documents with stored evidence; $L$: sum of their strongest probabilities |
| Lexicon (Ambiguity Lexicon) | Per value and slot: $L$, $H$ and evidence pointers; a view of the index |
| MDE | Minimum detectable effect: the smallest difference a test of this size would detect with 80% power |
| Plan regret | Reciprocal-rank loss of a plan compared with the oracle plan |
| Reading | Hypothesis (segment, slot, value) with a probability |
| Segment | A value mention in the query |
| Silver label | A slot label for a train/validation query derived from KB edges of its answers |
| Slot | A directed role from the document's perspective: `<doc_type>.<relation>.<value_type>` |
| Value | A KB node id (v0) |

---

## 16. References

Papers:

- Agrawal, Gollapudi, Halverson, Ieong. Diversifying Search Results. WSDM 2009. https://www.microsoft.com/en-us/research/wp-content/uploads/2009/02/diversifying-wsdm09.pdf
- Angelopoulos, Bates. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511. https://arxiv.org/abs/2107.07511
- Angelopoulos, Bates, Fisch, Lei, Schuster. Conformal Risk Control. ICLR 2024. https://openreview.net/forum?id=33XGfHLtZg
- Bella, Ferri, Hernández-Orallo, Ramírez-Quintana. Quantification via Probability Estimators. ICDM 2010. https://doi.org/10.1109/ICDM.2010.75
- Dai, Callan. Deeper Text Understanding for IR with Contextual Neural Language Modeling. SIGIR 2019. https://arxiv.org/abs/1905.09217
- Dhulipala, Hadian, Jayaram, Lee, Mirrokni. MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings. NeurIPS 2024. https://arxiv.org/abs/2405.19504
- Gao, Dai, Callan. COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List. NAACL 2021. https://aclanthology.org/2021.naacl-main.241/
- Guo, Pleiss, Sun, Weinberger. On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599
- Iyer, Bilmes. Submodular Optimization with Submodular Cover and Submodular Knapsack Constraints. NeurIPS 2013. https://proceedings.neurips.cc/paper_files/paper/2013/file/a1d50185e7426cbb0acad1e6ca74b9aa-Paper.pdf
- Khattab, Zaharia. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR 2020. https://arxiv.org/abs/2004.12832
- Kim, Xue, Croft. A Probabilistic Retrieval Model for Semistructured Data. ECIR 2009. https://doi.org/10.1007/978-3-642-00958-7_22
- Li, Chen, Van Durme, Xia. Multi-Field Adaptive Retrieval. ICLR 2025. https://arxiv.org/abs/2410.20056 (code: https://github.com/microsoft/multifield-adaptive-retrieval)
- Li, Lin, Oguz, Ghoshal, Lin, Mehdad, Yih, Chen. CITADEL: Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval. ACL 2023. https://aclanthology.org/2023.acl-long.663/
- Min, Michael, Hajishirzi, Zettlemoyer. AmbigQA: Answering Ambiguous Open-domain Questions. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.466/
- Saerens, Latinne, Decaestecker. Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure. Neural Computation 14(1), 2002. https://doi.org/10.1162/089976602753284446
- Santhanam, Khattab, Saad-Falcon, Potts, Zaharia. ColBERTv2. NAACL 2022. https://aclanthology.org/2022.naacl-main.272/
- Sarkas, Paparizos, Tsaparas. Structured Annotations of Web Queries. SIGMOD 2010.
- Wu, Zhao, Yasunaga, Huang, Cao, Huang, Ioannidis, Subbian, Zou, Leskovec. STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2404.13207 (code: https://github.com/snap-stanford/stark)

Software and models (check each license file before release):

- Vespa — https://docs.vespa.ai (Apache-2.0)
- Qwen3-Embedding-0.6B — https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (Apache-2.0)
- ModernBERT-base — https://huggingface.co/answerdotai/ModernBERT-base (Apache-2.0)
- bge-reranker-v2-m3 — https://huggingface.co/BAAI/bge-reranker-v2-m3 (Apache-2.0)
- Qwen3-4B-Instruct-2507 — https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0)
- GLiNER2 — https://github.com/fastino-ai/GLiNER2 (Apache-2.0)
- sentence-transformers — https://www.sbert.net
- bm25s — https://github.com/xhluca/bm25s (MIT)
- ranx — https://github.com/AmenRa/ranx (MIT)
- vLLM — https://docs.vllm.ai
- faiss-cpu — https://pypi.org/project/faiss-cpu/ (MIT)

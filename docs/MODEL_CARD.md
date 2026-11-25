---
language: en
license: apache-2.0
base_model: answerdotai/ModernBERT-base
library_name: sentence-transformers
pipeline_tag: text-classification
---

# SlotSpace slot scorer - unreleased status card

**No fine-tuned SlotSpace checkpoint has been trained or released.**
This card documents the specified model and its current evidence, not an
evaluated downloadable model. The planned base is ModernBERT-base (149M
parameters), whose model card declares Apache-2.0.

## Intended use and interface

Score a masked query/document context against schema slot descriptions,
including the applicable `none` outcomes. One joint softmax covers all
candidate types for a mention. Separate query/document temperatures calibrate
that distribution. Corpus priors come from the index lexicon, not this model.

This is a retrieval research component, not a medical diagnostic or treatment
system. It is not designed for patient records or private data. English,
closed-vocabulary STaRK-Prime is the initial scope; multilingual and
open-vocabulary behavior is unvalidated.

## Training specification

- Schema-generated query/document templates; fresh surrogate values and
  masked target/other/self mentions for the main model.
- Offline Qwen3-4B paraphrases, validated before masking; no closed-model API.
- Template-disjoint train/dev splits, seed 13 and BCE fine-tuning.
- STaRK train/even-qid silver segments calibrate query temperatures. KB-linked
  document mentions calibrate document temperatures. Never tune on test.
- A12 (visible real targets) and A14 (held-out slot) are distinct variants;
  neither is evidence about a trained main checkpoint.

The CPU generation smoke produced 1,584 masked examples. It ran no paraphraser
or neural optimization and is not the final training corpus.

## Evaluation status

| Measure | Observed main-model result |
|---|---|
| Fine-tuning / checkpoint | NOT RUN / absent |
| Query/document slot accuracy | NOT MEASURED |
| ECE before/after temperature scaling | NOT MEASURED |
| Held-out-slot accuracy / reading coverage | NOT MEASURED |
| Retrieval quality on official test/human queries | NOT RUN |

ECE at most 0.05 is an acceptance **target**, not an achieved result. Before
shipping weights, replace this status card using the actual training manifest,
model hashes, corpus/schema version, calibration JSON and provenance-matched
M5/M8 metrics; retain any failed acceptance and limitations.
Use `python -m slotspace.model_card --model <final-checkpoint> --calibration
<temperature-json> --out <new-card-path>` after a real run. It validates
checkpoint/schema/train identities and recomputes M5 metrics from the matching
`.points.parquet` file. It does not replace this status card automatically or
turn calibration/card generation into human/experiment approval.

## Limitations

Synthetic phrasing and distant-supervision labels may not match natural
queries. Calibration and conformal coverage depend on the calibration/test
distribution. Same-name ambiguity, missing knowledge-base edges, rare roles
and the 256-fact cap can affect retrieval. A masked input prevents simple value
memorization but does not prove generalization or remove all corpus bias.

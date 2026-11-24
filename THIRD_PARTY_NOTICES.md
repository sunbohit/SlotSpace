# Third-party provenance and redistribution

SlotSpace's own source is Apache-2.0. That does **not** relicense its inputs,
optional services or installed dependencies. The Python distribution does not
vendor those dependencies or bundle datasets, model weights or experiment runs.

## Data and models

| Input | Upstream license declaration | Source |
|---|---|---|
| STaRK code | MIT | <https://github.com/snap-stanford/stark> |
| STaRK benchmark data | CC BY 4.0 | <https://huggingface.co/datasets/snap-stanford/stark> |
| Qwen3-Embedding-0.6B | Apache-2.0 | <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B> |
| ModernBERT-base | Apache-2.0 | <https://huggingface.co/answerdotai/ModernBERT-base> |
| bge-reranker-v2-m3 | Apache-2.0 | <https://huggingface.co/BAAI/bge-reranker-v2-m3> |
| Qwen3-4B-Instruct-2507 | Apache-2.0 | <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507> |
| Optional GLiNER2-base-v1 | Apache-2.0 | <https://huggingface.co/fastino/gliner2-base-v1> |
| Optional mmBERT-base | MIT | <https://huggingface.co/jhu-clsp/mmBERT-base> |
| Optional Vespa engine | Apache-2.0 | <https://github.com/vespa-engine/vespa> |

Preserve the benchmark attribution in [NOTICE](NOTICE). Download source data
from its publisher; do not add raw records, answers or weights to a release
archive. Derived role-swap distribution is limited by project convention to
IDs/templates/value IDs and the build code. Underlying source collections and
any future inputs may impose additional terms.

## Exact local dependency inventory

[The dependency inventory](docs/dependency-licenses.csv) records the installed
versions, declared metadata and hashes/locations of bundled license texts.
It is a metadata/text inventory, **not** a legal compatibility certification.
The lock files identify the tested environment; other platforms, wheels and
CUDA/vLLM builds require a fresh inventory. The optional generator has its own
dependency graph and must not be inferred from the CPU inventory.

[Six model-card declarations](docs/model-licenses.csv) were rechecked against
the cached revision/hash manifest. [The generator inventory](docs/generator-licenses.csv)
records declarations for all 198 pinned Linux dependencies; those GPU wheels
and their bundled licenses were **not** installed or inspected. Inventories
and research assets are also carried in a wheel under `share/slotspace/`.

Items requiring care before any public/composite distribution:

- `fuzzywuzzy` 0.18.0, pulled in by PyTDC, ships GPL version 2 text. Do not
  infer compatibility with an Apache-licensed combined distribution.
- `locate` 1.1.1 has an Other/Proprietary metadata classifier, but its bundled
  `LICENSE` is MIT. The raw declaration and file hash are retained rather than
  silently rewriting the publisher's metadata.
- `docutils` (a release-tool dependency) has mixed public-domain/BSD/GPL
  component notices. It is not bundled in SlotSpace.
- The Linux generator lock includes NVIDIA proprietary runtime/toolchain
  declarations. An open-model license is not a license to redistribute the
  complete CUDA stack.

These are review findings, not legal approval. Public distribution remains
subject to an appropriate dependency/binary/container license review.

In particular, `wordfreq` combines Apache-licensed code with word-frequency
data under CC BY-SA 4.0; retain its full upstream notices. Native scientific
wheels can contain additional BLAS/compiler/runtime licenses beyond their
top-level Python package declaration. Any vendored binary or container image
must preserve its complete third-party notices and receive a separate review.
GNU Make, if used, is an external development tool, not bundled software.

No package or model has been uploaded to a registry by this preparation work.

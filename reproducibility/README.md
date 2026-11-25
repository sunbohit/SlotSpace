# Portable inputs and preliminary data

- `huggingface.csv`: original upstream URLs, fixed revisions and file hashes.
  Obtain raw datasets/model weights from those publishers; they are not hosted
  in this repository.
  From the clone, `python -m slotspace.resources fetch --root ../Resource`
  acquires the four v0 models and required Prime cache; `verify` is offline-only.
- `role_swap_ids.jsonl`: provisional dev/test query IDs, slot/value/template IDs
  and partner IDs only. It contains **no query text, answers or partner answers**.
  Regenerate the full set using `slotspace.role_swap`. H3 semantic review and
  the original 1,000-query target have not passed (the test split has 667 IDs).
- `pilot_smoke/`: portable measured per-query metrics/counters and the
  preliminary report from clean code `cf43e84`. It is not a formal M8 result.
  Machine-specific paths, environments, model weights and run logs are omitted.

STaRK-derived artifacts retain the CC BY 4.0 attribution in [NOTICE](../NOTICE).
Do not re-host the original benchmark. GPU installation and input acquisition
are documented in [GPU_SETUP](../docs/GPU_SETUP.md).

# GPU machine handoff - no AI agent required

This runbook covers acquisition, M0-M8 with real review gates, optional M9,
and M10 release preparation using terminal commands. Development/mock tests
may assume successful experiments; **real experiment status does not**.
No fine-tuned main checkpoint, accepted production freeze, formal M8 result
or real Vespa measurement is supplied by this clone.

Code, schemas, templates and portable configuration live in this repository.
Raw STaRK files, model weights, environments and machine-specific run logs are
not Git payloads. Do not re-host raw benchmark data.

## Clone and prerequisites

Use Linux, Conda, Git and a CUDA-capable machine. The design target is one
24 GiB GPU, 64 GiB RAM and 200 GiB free disk. These are planning estimates,
not measured GPU requirements. Leave system GPU drivers unchanged.

```bash
git clone https://github.com/sunbohit/SlotSpace.git
cd SlotSpace
nvidia-smi
git status --short
```

The code checkout must stay clean during reproduction and H5/M8. Outputs
belong under the ignored `runs/` tree. Record the cloned commit; do not update
code or installed packages midway through a reproduction workspace.

## Linux research environment

Run from the cloned repository. Install release tools now, not during a later
resume: environment/package identities are bound to reproduction receipts.

```bash
conda env create -f environments/research.yaml
conda activate slotspace
# CUDA 12.8 example: verify compatibility with the machine's driver first.
python -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c environments/research.constraints.txt -e ".[test,release]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The official index has a Python 3.11 Linux x86_64 CUDA 12.8 wheel for Torch
2.10.0. This is a prepared configuration, **not a GPU-tested environment**.
Select a different official build if the actual driver/GPU requires it;
deliberately update the Torch constraint to match that choice before starting.
Training extras are declared in the package requirements, not in constraints
(pip rejects extras inside a constraints file).

## Separate Linux generation environment

```bash
conda env create -f environments/generation.yaml
conda activate slotspace-gen
python -m pip install -r environments/generation-linux.lock.txt
python -m pip install --no-deps -e .
```

This pinned generator stack uses vLLM 0.34.0 and its own Torch/CUDA dependencies.
It must not be mixed into the research environment. Its NVIDIA/runtime license
and driver requirements need review on the actual machine.

## Fetch and verify inputs

Activate the research environment. All revisions and checksums are recorded in
[the original resource manifest](../reproducibility/huggingface.csv).

```bash
conda activate slotspace
export HF_HUB_DISABLE_TELEMETRY=1
python -m slotspace.resources fetch --root ../Resource
python -m slotspace.resources verify --root ../Resource
```

Default acquisition covers the four v0 snapshots (36 manifest files) and
the required official Prime QA/processed cache. Fixed revisions, filenames,
byte counts and SHA-256 hashes are checked. Prime preparation extracts only
the six expected processed members, verifying them before deserialization.
Existing incompatible inputs fail without being overwritten. Downloaded model
files are staged, verified and published exclusively; failed staging is cleaned
up. Use a local filesystem supporting hard links (normal ext4/NTFS does).

Hub caching and the standalone snapshots can occupy two copies of the weights.
The raw Prime archive, papers, upstream source clones, optional MAG/Amazon
datasets and v1 algorithms are not required to reproduce v0. Optional
`--include-v1-models` acquires the two extra model snapshots only; it does not
implement those future algorithms.

Once acquisition succeeds, disable implicit Hub access for research jobs:

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
```

Before fetching a missing input later, unset the two offline flags explicitly.
`verify` never downloads anything.

## Reproduce M0-M8 and perform H1-H6 yourself

In each new terminal/SSH session, activate the research environment, return to
the checkout and recreate these variables. The generator executable is
discovered from its separate Conda environment:

```bash
conda activate slotspace
ROOT="$(pwd)"
RES="$(cd .. && pwd)/Resource"
WORK="$ROOT/runs/reproduce-v0"
GEN_PY="$(conda run -n slotspace-gen python -c 'import sys; print(sys.executable)')"
REVIEWS="$WORK/reviews.json"
H5="$WORK/h5.json"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled

python -m pytest -q -m "not slow and not vespa"
python -m slotspace.reproduce --resources "$RES" --work "$WORK" --dry-run

run_v0() {
  python -m slotspace.reproduce --resources "$RES" --work "$WORK" \
    --gen-python "$GEN_PY" --reviews "$REVIEWS" --approval "$H5" "$@"
}
run_v0 --through M0
```

`--dry-run` prints commands, not results. The real recipe has 56 command/gate
entries. It generates all missing full-corpus/model/calibration/ablation inputs;
downloaded ModernBERT-base is **not** the trained main scorer.

After M0, run H0's actual data checks and continue:

```bash
export SLOTSPACE_STARK_ROOT="$RES/datasets/stark"
export SLOTSPACE_PRIME_EXPORT="$WORK/data/prime"
python -m pytest -q tests/test_data_export.py
run_v0 --resume --through M8
```

A review stop exits with code 2 and retains completed receipts. It is expected,
not a request to skip a checkpoint. Keep the same `WORK`, `GEN_PY`, code and
environment when resuming. The work layout differs from older README/manual
examples; use the generated paths rather than copying old cache filenames.

### H1-H4: inspect, label, approve, resume

```bash
python -m slotspace.operator status --work "$WORK"
python -m slotspace.operator review-info --work "$WORK" --gate H1
```

`review-info` works before labels are filled. It prints the exact sample paths
and requirements, with `validated: false`; inspection is **not approval**.
Use a text editor to review each example and change only its `correct` or
`natural` field from `null` to JSON `true`/`false`. Retain the records, their
other fields and ordering. Do not mass-fill positive labels.

| Gate | Review requirement |
|---|---|
| H1 | At least 100 query and 100 document mentions, each accuracy >= 0.85; also inspect schema descriptions |
| H2 | At least 100 silver examples, agreement >= 0.80 |
| H3 | At least 50 role-swap examples, natural/correct-query rate >= 0.90 |
| H4 | At least 30 provenance-matched scorer examples labelled; both measured post-scaling ECE <= 0.05 |

After actually reviewing H1 and its schema:

```bash
python -m slotspace.operator approve-review --work "$WORK" --gate H1 \
  --out "$REVIEWS" --yes
run_v0 --resume --through M8
```

Repeat `review-info` / editing / `approve-review` for H2, H3 and H4 at the
corresponding stops, using the same approval file. `--yes` records the human's
explicit approval; the helper still validates real labels/thresholds and binds
the exact evidence hash. Changed evidence is not silently re-approved. Existing
approvals remain invalid until deliberately revoked and reviewed again.

If linking/silver/naturalness or ECE fails, preserve the negative result.
Investigate and use a new, explicitly configured run; never edit measured
metrics, fabricate labels or lower gates merely to continue.

### M7/H5: production freeze and final test authorization

M7 freezes only complete accepted production evidence. Small caches, pilots
and fixture/test outputs cannot substitute. The recipe prepares the final
protocol and stops at H5 before evaluating test/human queries.

```bash
python -m slotspace.operator status --work "$WORK"
python -c 'import json,sys; p=json.load(open(sys.argv[1])); print([(j["id"], j["missing"]) for j in p["jobs"] if j["missing"]])' "$WORK/protocol.json"
```

Review the actual `frozen.yaml`, its freeze receipt and `protocol.json`,
including experiment scope and all artifact paths. An empty missing list
alone is not approval. Only after that human review:

```bash
python -m slotspace.operator approve-h5 --plan "$WORK/protocol.json" --out "$H5" --yes
run_v0 --resume --through M8
```

H5 reuses the existing authorization validator: genuine production freeze,
exact configuration/plan hashes, unchanged clean implementation and intact
inputs. Drafts, pilots and missing/stale artifacts are rejected without writing
an approval. No final-test permission follows from development assumptions.

H6 is your review of `report_v0.md`, its JSON and figures. Inspect PASS / FAIL /
NOT DETECTED / NOT RUN and the negative-results paragraph, not just top-line
quality. The report regenerates from saved artifacts rather than inference.

### Interrupted/failed work

Use `operator status` and the failing step's `meta.json` for diagnostics.
`--resume` verifies receipts; it never adopts partial/unowned outputs.
Do not delete or overwrite a failed formal execution ledger to rerun test
queries. A changed environment/source requires a new reproduction workspace.
An approved scientific redesign needs its own new plan and H5 decision.

## M9: optional stock-Vespa path after M8

This is optional runtime work, not a dependency for M8. It needs Docker already
available on the VM and a **new dedicated disposable** container. It implements
binary bound/max with 256-d topic/128-d context vectors, supported exponent
profiles, and `p`/`p_over_H` weights. It does not implement IA or hard-AND.
If the frozen setup is unsupported, stop or prepare a separately documented,
validation-selected compatible M9 configuration; do not mutate the M8 freeze.

Acquire the public image, resolve it to a digest and record that digest before
starting. No particular Vespa image/version has been certified on this host:

```bash
docker pull vespaengine/vespa
VESPA_IMAGE="$(docker image inspect vespaengine/vespa --format '{{index .RepoDigests 0}}')"
printf '%s\n' "$VESPA_IMAGE" > "$WORK/vespa-image.txt"
docker run --detach --name slotspace-vespa --hostname vespa-container \
  --memory 16g --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 "$VESPA_IMAGE"
docker logs --tail 30 slotspace-vespa
```

If the name/ports are occupied, choose other dedicated ones and pass the
matching `--endpoint`/`--deploy-endpoint`; never replace a shared application.
Wait for container startup, then perform the real four-document smoke:

```bash
python -m slotspace.vespa_app smoke --deploy --allow-replace-application --runs "$WORK/runs"
```

Smoke checks every exponent profile at absolute parity tolerance `1e-3`, the
hybrid fallback and cleanup of its four newly fed IDs. A successful mocked HTTP
unit test does not certify the real schema compiler or this deployment.

With a compatible frozen configuration and an empty dedicated app:

```bash
python -m slotspace.vespa_app export --config "$WORK/frozen.yaml" --out "$WORK/vespa_binary"
python -m slotspace.vespa_app feed --bundle "$WORK/vespa_binary" \
  --deploy --allow-replace-application --runs "$WORK/runs"
```

Record the **actual directory printed by feed**. Set `FEED_RUN` to it; do not
guess a successful receipt after a failed command. Start with a small
validation-only measurement, then increase:

```bash
FEED_RUN="/absolute/path/printed/by/feed"
python -m slotspace.vespa_app measure --config "$WORK/frozen.yaml" \
  --bundle "$WORK/vespa_binary" --feed-receipt "$FEED_RUN/feed.json" \
  --queries 20 --parity-queries 4 --container slotspace-vespa --runs "$WORK/runs"
python -m slotspace.vespa_app measure --config "$WORK/frozen.yaml" \
  --bundle "$WORK/vespa_binary" --feed-receipt "$FEED_RUN/feed.json" \
  --queries 1000 --parity-queries 200 --container slotspace-vespa --runs "$WORK/runs"
```

The full criterion needs 200 parity queries and 1,000 validation measurements.
Inspect parity, quality, p50/p95 client/server timings and unstable rankings.
Fan-out records sum/max RPC/server times. Docker RAM is **whole-container
memory**, not isolated index RAM. A partial failed feed is preserved rather
than silently retried or erased. Stop only your own container when finished:
`docker stop slotspace-vespa`.

## M10: measured cards, license review and local packages

Generate cards into a separate artifact directory, not the input checkpoints
or the committed untrained status card:

```bash
for kind in ft a12 a14; do
  python -m slotspace.model_card --model "$WORK/data/prime/models/slot_scorer_$kind/final" \
    --calibration "$WORK/data/prime/temperature_$kind.json" --out "$WORK/model_cards/$kind.md"
done
```

The generator validates schema/train/checkpoint/calibration identities,
template separation, classifier safetensors and recomputed per-example M5
metrics. It keeps failed ECE targets and pending human review visible. It does
not claim H4/M7/M8 acceptance and refuses to overwrite a card.

Re-inventory the actual research environment and downloaded model cards:

```bash
python tools/license_inventory.py --out "$WORK/license-review/dependency-licenses.csv" \
  --resources "$RES" --input-manifest reproducibility/huggingface.csv \
  --models-out "$WORK/license-review/model-licenses.csv" \
  --generator-lock environments/generation-linux.lock.txt \
  --generator-out "$WORK/license-review/generator-licenses.csv"
```

Generator declaration refresh contacts public PyPI metadata. Review its actual
Linux/CUDA wheel terms separately, including NVIDIA runtime terms. Existing
GPL/proprietary inventory findings are documented in
[the third-party notices](../THIRD_PARTY_NOTICES.md); an inventory is not legal
compatibility approval.

```bash
python -m pytest -q -m "not vespa"
python -m build --no-isolation --outdir "$WORK/dist"
python -m twine check --strict "$WORK"/dist/*.whl "$WORK"/dist/*.tar.gz
python tools/check_dist.py --dist "$WORK/dist"
# Equivalent package checks, if Make is available:
make dist DIST="$WORK/dist"
```

Source and wheel contain implementation, application resources, environment
recipes and the pinned public-input manifest, not benchmark records, weights
or run outputs. These commands do **not** publish to a registry. Review H6,
real runtime results, model/data licensing and limitations before calling any
candidate an empirical/model release.

## Verified engineering evidence for this handoff

- Full CPU regression: 153 passed; one real-Vespa test explicitly skipped.
- Actual local verification: all 36 v0 model files and the pinned Prime cache.
- Corrected install recipe: offline pip dry-run resolves with no dependency
  installation into the CPU environment.
- Source/wheel: strict Twine and required-asset/raw-payload checks pass.
- Wheel rebuilt from the sdist: identical member contents; 70 tests pass using
  the installed package outside the checkout, with temporary files cleaned.
- License recheck: 188 exact installed distributions, six cached model-card
  declarations, no unresolved metadata/text inventories. Compatibility and
  actual Linux/CUDA wheel-term review are still separate obligations.

These are development/package checks, not GPU, final-benchmark or real-engine
acceptance. The earlier real pilot's negative findings remain unchanged.

H1-H4 need human reviews. M7 production acceptance must pass before a frozen
configuration is produced, and H5 must authorize its exact protocol before
formal test/human experiments. The committed pilot summaries are preliminary:
four validation queries, zero-shot T=1, high ECE and no demonstrated CPU
efficiency gain. They do not authorize full M8.

# mmd: modality discordance across a model zoo

`mmd` runs the same swap analysis for every multimodal model on one frozen MIMIC-IV cohort, so models can be
compared. It is the shared core of `modality_eda/cxr/LLaVA_Med_MIMIC_ablations.ipynb` and
`modality_eda/ecg/OpenTSLM_MIMIC_ablations.ipynb` (both unchanged); `mmd analyse` reproduces their `swap4_*` tables.
The design is in `dev/plan.md`.

**How it works.**
- **Embed once.** The GPU pass (`mmd perturb`) runs each perturbation record through the model once. It stores the readouts (P(Yes), hidden states, answer embedding) in the bucket and indexes them.
- **Analyse many times.** Every later step (`analyse`, `zoo`, the explore notebook) reads only the stored embeddings, never the model, so you can come back to the results without recomputing anything.
- **Resumable.** `perturb` skips records already in the index, so re-running it only fills gaps.
- **Existing runs reused.** Embeddings from the notebooks are imported read-only.

## What every model gets
Each model runs the same configurations on its modality pair: one row per stay and tier (L1–L3), with the same folds and donors for every model.

| config | family | what changes |
|---|---|---|
| `ref` | reference | own baseline signal + own baseline note |
| `A` / `B` | control | signal only (note removed) / note only (signal removed) |
| `F` | temporal | both modalities from the stay's last record |
| `temporal_<m>` | temporal | modality *m* from the stay's last record |
| `matched_<m>` | matched | *m* from a donor with the same tier, age group × sex and **same** outcome |
| `counterfactual_<m>` | counterfactual | *m* from a donor like the above with the **opposite** outcome |
| `mean_<m>` | mean | *m* replaced by the cohort-mean encoder output (for notes: removed) |
| `…@order=flip`, `…@para=1/2` | prompt factors | other slot order / paraphrases (L1 `ref` and `counterfactual` only) |

Models that cannot take the mean swap (CheXagent) skip `mean_<signal>`, and the zoo table marks it missing.

## Models and modalities
Modalities: `e` ECG, `x` CXR image, `v` vitals, `l` labs, `n` note. The record grammar
(`<stay>|<mod>:<slot>…`, see `cohort.py`) covers all five.

| model | pair | modalities | uv group | mean swap |
|---|---|---|---|---|
| `llava-med` | cxr_notes | x + n | llava-med | projector output |
| `medgemma-4b` | cxr_notes | x + n | hf-modern | projector output |
| `chexagent` | cxr_notes | x + n | chexagent | not offered |
| `opentslm-sp` (Llama-3.2-1B) | ecg_notes | e + n | opentslm | projector output + label statistics |
| `opentslm-sp-gemma` (Gemma-3-1B) | ecg_notes | e + n | opentslm | projector output + label statistics |

Both OpenTSLM models run the same SP code. The Llama run is imported from the notebook. Gemma has no notebook run, so every Gemma record is generated, and its ECG prefix mean comes from its own records. Before the Gemma run, accept the terms of `google/gemma-3-1b-pt` on Hugging Face.

**Vitals (`v`) and labs (`l`) are not run yet.** They have grammar keys but no pair, loader or model. See *Known gaps*.

## Setup, step by step

### 0. Prerequisites (you create these; mmd never does)
- A **private** GCS bucket, in the same region as the GPUs, and a budget alert on the project.
- Access to the phase0 BigQuery dataset (`sql/phase0_dataform`).
- PhysioNet credentials covering MIMIC-IV, MIMIC-CXR and MIMIC-IV-ECG. The CXR JPGs and ECG files must be readable under `raw_root`.
- A Hugging Face token whose account has accepted the MedGemma terms (`huggingface-cli login`, or `HF_TOKEN`).
- On the laptop: `gcloud auth application-default login`. On macOS, `brew install libomp` for XGBoost. Without it, the analysis runs logistic regression only and says so.

### 1. Install (laptop, core only)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # once
uv sync                                                 # Python 3.12, core: cohort, analysis, zoo
```

### 2. Settings
Create `mmd.toml` in the repo root. Don't commit it. `MMD_<NAME>` env vars override any field.
```toml
[mmd]
root = "gs://<your-private-bucket>"
cohort_version = "v1"
# raw_root = "gs://<bucket>/physionet"   # CXR JPG / ECG files (default {root}/physionet)
```
Then check that the bucket and BigQuery are reachable:
```bash
uv run mmd setup check
```

### 3. Data snapshot
Freeze the phase0 tables as parquet. If you already have Drive copies, copy them over (copy only; Drive stays the backup):
```bash
uv run mmd data stage
# or, from Colab with Drive mounted:
uv run mmd migrate drive --src /content/drive/MyDrive/mimic_mmd [--raw] [--dry-run]
```
`migrate drive` also copies the notebooks' `embeddings/` to `{root}/emb/legacy/`.

### 4. Frozen cohorts (one per pair)
Build each cohort from the notebooks' saved donor plans (and CXR slot pins), so records, folds and donors match the existing runs exactly:
```bash
uv run mmd cohort build --pair cxr_notes \
    --donor-plans gs://<bucket>/emb/legacy/llava-med-v1.5-swap4 \
    --pins gs://<bucket>/emb/legacy/llava-med-v1.5-swap4/slot_pins_x.csv
uv run mmd cohort build --pair ecg_notes --donor-plans gs://<bucket>/emb/legacy/<embeddings>/swap4
```
Without `--donor-plans`, donors are drawn fresh (seeded). The manifest's hash is checked by `perturb` and `analyse`.

### 5. Register existing embeddings (no recompute)
```bash
uv run mmd index import --model llava-med \
    --legacy gs://<bucket>/emb/legacy/llava-med-v1.5-cxr-notes \
    --mean-from gs://<bucket>/emb/legacy/llava-med-v1.5-swap4/cohort_mean_projector.npz
uv run mmd index import --model opentslm-sp \
    --legacy gs://<bucket>/emb/legacy/<embeddings>/fused-llama-3.2-1b-ecg-sp-prefix-swap4 \
    --mean-from gs://<bucket>/emb/legacy/<embeddings>/swap4/cohort_mean_sp.npz \
    --lookup sp-prefix=gs://<bucket>/emb/legacy/<embeddings>/llama-3.2-1b-ecg-sp-prefix \
    --lookup flamingo=gs://<bucket>/emb/legacy/<embeddings>/llama-3.2-1b-ecg-flamingo
uv run mmd status --pair cxr_notes                      # stored / needed per model and family
uv run mmd status --pair ecg_notes
```
Legacy folders are only read; `sources.json` points the model at them. You can check parity now:
`uv run mmd analyse --model llava-med` (or `opentslm-sp`) should match the notebook's `swap4_*.csv`.

### 6. Generate missing records (GPU VM)
Use a Deep Learning VM image (NVIDIA driver included; torch brings its own CUDA). Start with `g2-standard-8` (1× L4).
Give each model family its own environment folder, so switching models doesn't reinstall anything. Always run with `--no-sync`: a plain `uv run` re-syncs to core and removes the model packages.
```bash
# once per VM
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo> && cd mimic-multimodal-discordance      # copy mmd.toml here

# once per model family (one at a time; the groups conflict)
export UV_PROJECT_ENVIRONMENT=.venv-hf-modern            # medgemma-4b
uv sync --frozen --group hf-modern
#   llava-med:   .venv-llava-med,  --group llava-med,  then: uv run --no-sync mmd setup llava-med --commit <sha>
#   chexagent:   .venv-chexagent,  --group chexagent
#   opentslm-sp, opentslm-sp-gemma: .venv-opentslm, --group opentslm

# per model
uv run --no-sync mmd weights stage --model medgemma-4b                       # bucket copy, else Hugging Face
uv run --no-sync mmd perturb --model medgemma-4b --smoke 3                   # check ids/spans; nothing stored
uv run --no-sync mmd perturb --model medgemma-4b --plan --sec-per-record 0.6 # records to do, GPU-hours
uv run --no-sync mmd perturb --model medgemma-4b                             # all levels and families
#   options: --levels L1  --families temporal,counterfactual  --no-factors  --shard 0/2
```
`perturb` takes the pair from the model. It writes chunks to `emb/<version>/<model>/` and refreshes the index. The cohort mean for the mean swap is computed once and saved next to the run. Stop the VM when you're done.

### 7. Analyse each model (laptop, core only)
```bash
uv run mmd analyse --model llava-med
uv run mmd analyse --model medgemma-4b
uv run mmd analyse --model chexagent
uv run mmd analyse --model opentslm-sp
uv run mmd analyse --model opentslm-sp-gemma
```
For every readout, this runs:
- the shuffled-label self-test;
- probe transfer (reference-only training; swaps are evaluation only);
- raw P(Yes) with no probe;
- shifts against the reference;
- AUROC (donor mean and single donor);
- temporal pairs;
- paired signal-vs-note dominance;
- informativeness per arm.

Every vector readout is probed with **both logistic regression and XGBoost**, giving one row per classifier in every table. P(Yes) also gets a `no probe (raw P(Yes))` row.

`--no-charts` / `--no-selftest` skip those steps.

### 8. Aggregate, compare models (zoo) and explore
```bash
uv run mmd collect                                      # every model x pair -> results/<version>/all/<table>.parquet
uv run mmd zoo --pair cxr_notes --models llava-med,medgemma-4b,chexagent \
    --split-file <mimic-cxr-2.0.0-split.csv.gz>         # adds the official-split subset for MIMIC-exposed models
uv run mmd zoo --pair ecg_notes --models opentslm-sp
uv run mmd bq link --pair cxr_notes                     # optional: BigQuery external tables over the parquet
```
**How the ablations add up.** Every table is long, with one row per model × pair × level × readout × classifier × swap family × swapped modality (`slot`). `collect` stacks them into one table per kind:
- `summary`
- `auroc`
- `auroc_by_draw`
- `reference_shifts`
- `paired_dominance`
- `temporal_paired`
- `selftest`
- `informativeness`
- `zoo`

Each row carries `model`, `pair` and `signal`, so any slice is a single filter. For example, the counterfactual note swap, XGBoost, L1, for every model is `family == 'counterfactual' & slot == 'n' & clf == 'XGBoost' & level_id == 'L1'`.

`zoo` reduces each model to three numbers:
- **information share:** from the single-modality AUROCs;
- **reliance share:** from the counterfactual |Δ risk|;
- **bias:** reliance minus information.

These are the numbers to compare across models and pairs. `notebooks/explore_results.ipynb` (core kernel) reads the zoo table, the per-model tables and the figures.

## Where everything goes (`root`)
```
data/<version>/*.parquet                      phase0 snapshot
cohort/<version>/<pair>/                      frozen manifest: slots, tiers + folds, donor plans, configs, hash
emb/<version>/<model>/                        chunk_*.npz (+ .records.txt), index.parquet, runs.parquet,
                                              manifest.json, cohort_mean_<sig>.npz, sources.json (legacy links)
emb/legacy/…                                  the notebooks' runs, read only
results/<version>/<model>/<pair>/             model.json + every table as .csv (notebook names) and .parquet:
    swap4_summary_<pair>[_clean]  swap4_per_stay_<pair>  swap4_selftest_<pair>  swap4_score_frames_<pair>
    swap4_reference_shifts_<pair>  swap4_auroc_<pair>  swap4_auroc_by_draw_<pair>
    swap4_temporal_paired_<pair>  swap4_paired_dominance_<pair>  informativeness[_per_stay]_<pair>
results/<version>/zoo/zoo_<pair>.csv|parquet  cross-model table
results/<version>/all/<kind>.csv|parquet      `mmd collect`: every model x pair stacked (+ pair, signal columns)
figures/<version>/<model>/<pair>/*.png        swap4_vsref_<readout>_<clf>   shift from the reference per swap
                                              swap4_auroc_<readout>_<clf>   AUROC at reference vs each swap
                                              swap4_temporal_paired_<readout>_<clf>
                                              swap4_paired_dominance_<clf>
weights/<model>@<revision>/                   pinned weights
```
Every table has a `model` column, and every figure title names the model, readout and classifier. The store holds derived data only (embeddings, scores, record ids, results). Note text, images and signals are read at run time and never written there, as the MIMIC data use agreement requires.

## Adding a model
1. Write `mmd/models/<name>.py` with a class registered via `@register`, setting `name`, `hf_id`, `revision`, `group`, `pair`, `readouts`, `meta`.
   - Subclass `HFImageTextAdapter` when the model loads with `AutoModelForImageTextToText`.
   - Otherwise subclass `Adapter` and implement `load()` and `forward()`. `forward()` returns the common readouts (`p_yes`, `post_last`, `post_all`, `post_signal`, `ans_emb`, `answer`).
2. Implement `signal_mean()` / `inject_mean()` (and set `supports_mean = True`) if it can take the mean swap. Implement `extra_readouts()` for pre-fusion lookups.
3. Add the module name to `models/base.get()`, and a dependency group in `pyproject.toml` if it needs its own pins.
4. Steps 6–8 as above, starting with `--smoke 3`.

## Known gaps
- **Vitals and labs.** To cover all five modalities, they need:
  - a pair in `cohort.PAIRS` (e.g. `ecg_vitals`, `vitals_labs`);
  - slot tables and loaders in `data.py`;
  - `swap_configs` generalised beyond two modalities (A/B/F are pair-only);
  - an adapter that takes them, e.g. OpenTSLM with vitals as extra time series, or labs serialised as text.

  A model then gets one run per pair. Results and figures are already separated by pair; `emb/<version>/<model>/manifest.json` still holds one cohort hash per model and must become per pair.
- **On the first VM run of every new model,** read the `--smoke` output: Yes/No token ids, the signal span and module names.
- **CheXagent:** the image-token span is unverified, and the mean swap is skipped.
- **Pinning:** pin `revision` for each model, and `LLAVA_MED_COMMIT`, once the smoke test passes.
- **OpenTSLM-Gemma:** check that `opentslm`'s `load_pretrained` loads the Gemma checkpoint, and check its Yes/No ids, in `--smoke`.
- **OpenTSLM batch size:** new records run one at a time, while the notebook ran batches of 4. P(Yes) can differ in the last digits, and only the new prompt-factor records are affected.

# mmd: modality discordance across a model zoo

**The question:** when a multimodal model predicts 1-year mortality for a cardiac ICU stay, how much does its answer
rely on each modality (CXR image, ECG, radiology note, …), compared with how much that modality tells it on its own?

`mmd` gives every model exactly the same **perturbed inputs** (a modality removed, swapped for a later record, for a
donor patient's, or for a cohort average). It measures how far each perturbation moves the model's risk and puts
every model in one table. It is the shared core of the notebooks `modality_eda/cxr/LLaVA_Med_MIMIC_ablations.ipynb`
and `modality_eda/ecg/OpenTSLM_MIMIC_ablations.ipynb`, whose embeddings it reuses.

---

## 1. How it fits together

### Four steps
```
 1. DEFINE                    2. RUN (GPU)                 3. ANALYSE (CPU)            4. COMPARE
 what is studied              each input through a model   per model, per set           across models

 modsets.py  modality set     runner.py                    analysis/probe.py            zoo.py
 data.py     slot tables  ─►  models/  (adapters)      ─►  analysis/effects.py      ─►  information vs
 cohort.py   recipients,      store.py (embeddings)        analysis/charts.py           reliance, per
             folds, donors                                 analysis/report.py           modality
 perturb.py  strategies ──► frozen manifest

 zoo.Zoo runs every step for every model ·  cli.py exposes each step as an `mmd` command
 shared: config.py (settings, paths) · fsio.py (local / gs:// IO) · records.py (record ids) · prompts.py
```
- **Steps 1 and 2 run once.** The manifest is frozen, and each input goes through each model once.
- **Steps 3 and 4 read only stored embeddings.** They run on a laptop and can be rerun freely.
- **Step 2 is resumable.** Rerunning it only fills in missing inputs, so it never redoes work.

### The modules
| step | module | what it owns |
|---|---|---|
| define | `modsets.py` | modality keys (`e` ECG, `x` CXR, `v` vitals, `l` labs, `n` note), `ModalitySet`, the registry of named sets |
| define | `records.py` | the record id: one string naming one complete model input |
| define | `data.py` | the phase0 snapshot; per modality, which record fills each stay's `first` (baseline) and `last` (follow-up) slot; image / ECG / note loaders |
| define | `cohort.py` | `TierCohort` (recipients and 5 fixed folds per tier), `DonorPlan` (donors per recipient), `Manifest` (all of it frozen and hashed) |
| define | `perturb.py` | the perturbation strategies, the configuration list they generate, `build_manifest` |
| run | `models/` | one `Adapter` per model (contract in `models/base.py`): a `Sample` in, readouts out |
| run | `runner.py` | the manifest's missing records → adapter → store; plan, smoke test, mean injection |
| run | `store.py` | `EmbeddingStore`: one folder per model of readouts keyed by record id, its index and run log |
| analyse | `analysis/probe.py` | `Embeddings` and `Probe`: fit on the reference embeddings per fold, score every perturbation |
| analyse | `analysis/effects.py` | every effect table: Δ risk, AUROC, within-patient, reliance and information shares |
| analyse | `analysis/charts.py`, `report.py` | the figures; `analyse_model` runs probe → effects → charts and writes the tables |
| compare | `zoo.py` | `Zoo`: the orchestration wrapper every model plugs into, and the cross-model table |
| — | `cli.py` | the `mmd` commands, each a thin call into `Zoo` |
| — | `config.py`, `fsio.py`, `prompts.py` | settings and paths, local / `gs://` file access, the shared question wording |

### What ties the steps together
1. **The record id.** `12345|e:first|x:d9876:first|n:first` means stay 12345 with its own baseline ECG and note, but the CXR of donor stay 9876. That one string is the manifest row, the store key, the resume check and the probe's lookup.
2. **The manifest hash.** Each modality set has one frozen manifest. `perturb` records its hash and `analyse` checks it, so every model is run and analysed on exactly the same inputs.
3. **The adapter contract.** Analysis never sees a model, only named arrays (`yes_probability`, `answer_mean_state`, …). Adding a model means writing one adapter.

---

## 2. Concepts

### Modality sets
A set is a name plus two or more modality keys. `cxr_notes = (x, n)` and `ecg_notes = (e, n)` are built in. Add your own in `mmd.toml` (`[mmd.modsets] cxr_ecg_notes = ["e", "x", "n"]`), with `--define cxr_ecg_notes=e,x,n` on any command, or `MMD_MODSETS=...`.
- Each set gets its own manifest, run records and results.
- A recipient needs a baseline record of **every** modality in the set, so larger sets have fewer recipients.

### Perturbation strategies
Each strategy replaces a **subset** of the set's modalities and keeps the rest at the stay's own baseline record.

| strategy | family (in tables) | the perturbed modalities become |
|---|---|---|
| `ablate` | `ablate` | removed from the input |
| `within_patient` | `temporal` | the stay's own latest follow-up record |
| `matched_patient` | `matched` | a same-outcome donor's baseline record |
| `matched_label_swap` | `counterfactual` | an opposite-outcome donor's baseline record |
| `cohort_mean` | `mean` | signals: the cohort-mean encoder output; the note is removed |

- **`max_order` (default 2)** caps how many modalities are perturbed at once. Subsets are combinations: k modalities give Σ_{r ≤ max_order} C(k, r) subsets (k = 5: 5 + 10 = 15).
- **Donor, mean and ablate subsets stop at k − 1**, because replacing everything leaves no patient in the input. Within-patient may replace all k.
- **A joint donor swap takes every perturbed modality from the same donor.**
- **Donors** come from the recipient's own test fold and age group × sex cell, so the probe never trains on its donors. There are `n_draws` (20) per recipient; `n_draws_model` (5) of them get a model pass.
- **Core arms always run:** the reference and each "only modality m" input, because the information share needs them.
- **Inputs per recipient** (all strategies, max_order 2): 28 for two modalities, 79 for three, 201 for five. `mmd modsets` prints this per set.
- **Prompt factors:** tier L1's reference and single-modality label swaps are also run with the other slot order and two paraphrases. These are marked in a `factor` column and left out of the standard analysis.

### Readouts (what a model pass stores)
All are final-layer hidden states unless noted:

| readout | meaning |
|---|---|
| `yes_probability` | P(Yes) at "Answer:" |
| `prompt_last_state` | state at the last prompt token |
| `prompt_mean_state` | mean over every prompt token |
| `signal_mean_state` | mean over the image or ECG tokens |
| `answer_mean_state` | mean over the generated answer |
| `answer_text` | the decoded answer |
| `encoder_mean`, `projector_mean` | pre-fusion signal vectors |

The notebooks' chunks (`p_yes`, `ans_emb`, `post_image`, …) are renamed to these on load.

### The result, per model and modality m of the set
- **information share:** (AUROC of "only m" − 0.5)⁺ over the sum across modalities. What m tells the model on its own.
- **reliance share:** m's share of the summed |Δ risk| when m alone gets the label swap. How much the model leans on m.
- **bias = reliance − information.** Above 0, the model relies on m more than m alone justifies.

Each vector readout is probed with logistic regression and XGBoost; `yes_probability` is also read raw.

---

## 3. How to run it

### Prerequisites (you create these; mmd never does)
- A **private** GCS bucket in the GPU region, with a budget alert.
- The phase0 BigQuery dataset (`sql/phase0_dataform`).
- PhysioNet access to MIMIC-IV, MIMIC-CXR and MIMIC-IV-ECG.
- A Hugging Face token whose account has accepted the MedGemma terms (and `google/gemma-3-1b-pt` for OpenTSLM-Gemma).
- Laptop: `gcloud auth application-default login`. On macOS, also `brew install libomp` for XGBoost; without it the analysis uses logistic regression only and says so.

### Step 0: install and configure (laptop)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # once
uv sync                                                 # core only: define, analyse, compare
```
Create `mmd.toml` in the repo root (keep it out of git):
```toml
[mmd]
root = "gs://<your-private-bucket>"
project = "<your-gcp-project>"
cohort_version = "v1"
# raw_root = "gs://<bucket>/physionet"   # CXR JPG / ECG files (default {root}/physionet)
# strategies = ["all"]                   # which strategies the manifests hold
# max_order = 2                          # at most this many modalities perturbed at once

[mmd.modsets]                            # optional: sets beyond cxr_notes / ecg_notes
# cxr_ecg_notes = ["e", "x", "n"]
```
```bash
uv run mmd setup check                  # settings, bucket and BigQuery reachable
uv run mmd modsets                      # sets, which models can run each, inputs per recipient
```

### Step 1: define
```bash
uv run mmd data stage                   # phase0 BigQuery tables -> data/v1/*.parquet
# or, from Colab with Drive mounted (also copies the notebooks' embeddings to emb/legacy/):
uv run mmd migrate drive --src /content/drive/MyDrive/mimic_mmd [--raw] [--dry-run]
```
Upload the CXR JPGs / ECG files to `raw_root` **before** building a CXR cohort, because only images that exist are used. For the two original sets, build from the notebooks' donor plans and pins, so the records match the notebooks' embeddings:
```bash
uv run mmd cohort build --modset cxr_notes \
    --donor-plans gs://<bucket>/emb/legacy/llava-med-v1.5-swap4 \
    --pins gs://<bucket>/emb/legacy/llava-med-v1.5-swap4/slot_pins_x.csv
uv run mmd cohort build --modset ecg_notes --donor-plans gs://<bucket>/emb/legacy/swap4
uv run mmd cohort build --modset cxr_ecg_notes --strategies all --max-order 2   # a new set: donors drawn fresh
```

### Step 2a: register the notebooks' embeddings (no GPU)
Each notebook run folder, copied to `emb/legacy/` by `migrate drive`, is registered read-only with its model. Its records then count and are never regenerated:
```bash
uv run mmd index import --model llava-med \
    --run-folder gs://<bucket>/emb/legacy/llava-med-v1.5-cxr-notes \
    --mean-from gs://<bucket>/emb/legacy/llava-med-v1.5-swap4/cohort_mean_projector.npz
uv run mmd index import --model opentslm-sp \
    --run-folder gs://<bucket>/emb/legacy/fused-llama-3.2-1b-ecg-sp-prefix-swap4 \
    --mean-from gs://<bucket>/emb/legacy/swap4/cohort_mean_sp.npz \
    --lookup sp-prefix=gs://<bucket>/emb/legacy/llama-3.2-1b-ecg-sp-prefix \
    --lookup flamingo=gs://<bucket>/emb/legacy/llama-3.2-1b-ecg-flamingo
uv run mmd status --modset cxr_notes    # records needed vs stored, per model and strategy
```

### Step 2b: run the missing inputs (GPU VM)
Use a Deep Learning VM image and start with `g2-standard-8` (1× L4). Each model family has its own environment (their library versions conflict). **Always use `--no-sync`**: a plain `uv run` re-syncs to core and removes the model packages.
```bash
# once per model family
export UV_PROJECT_ENVIRONMENT=.venv-hf-modern            # medgemma-4b
uv sync --frozen --group hf-modern
#   llava-med: .venv-llava-med --group llava-med, then: uv run --no-sync mmd setup llava-med --commit <sha>
#   chexagent: .venv-chexagent --group chexagent  ·  opentslm-sp / -gemma: .venv-opentslm --group opentslm

# per model and set
uv run --no-sync mmd weights stage --model medgemma-4b
uv run --no-sync mmd perturb --model medgemma-4b --smoke 3                    # check ids / spans; nothing stored
uv run --no-sync mmd perturb --model medgemma-4b --plan --sec-per-record 0.6  # records to do, GPU-hours
uv run --no-sync mmd perturb --model medgemma-4b                              # default set: the model's own
#   --modset cxr_notes  --levels L1  --strategies within_patient,matched_label_swap  --max-order 1
#   --no-factors  --shard 0/2
```
Stop the VM when you're done.

### Step 3: analyse (laptop)
```bash
uv run mmd analyse --model llava-med --modset cxr_notes    # --no-charts / --no-selftest to skip those
```

### Step 4: compare
```bash
uv run mmd zoo --modset cxr_notes [--split-file <mimic-cxr-2.0.0-split.csv.gz>] [--exclude-exposed]
uv run mmd collect                       # every model x set stacked into results/<v>/all/<table>
uv run mmd bq link --modset cxr_notes    # optional: BigQuery external tables over the parquet
```
`--split-file` adds a row per model restricted to stays whose baseline film is outside the MIMIC-CXR training split, for models trained on MIMIC-CXR.

### From Python (e.g. a notebook)
```python
from mmd import config
from mmd.zoo import Zoo
zoo = Zoo(config.load(), specs=['cxr_ecg_notes=e,x,n'])
zoo.compatibility()                               # model x set
zoo.analyse('llava-med', 'cxr_notes')
zoo.table(modset='cxr_notes')                     # info_share, reliance_share, bias per model and modality
```

---

## 4. Where everything goes (`root`)
```
physionet/<project>/<version>/...             raw CXR JPG / ECG files (you upload; raw_root)
data/<version>/*.parquet                      phase0 snapshot                          ← data stage
cohort/<version>/<set>/                       slots, tiers + folds, donor plans,       ← cohort build
                                              configs, manifest.json (hash)
emb/legacy/…                                  the notebooks' runs, read only           ← migrate drive
emb/<version>/<model>/                        chunk_*.npz (+ .records.txt), index,     ← perturb / index import
                                              runs.parquet, manifest.json (hash per set),
                                              cohort_mean_<mod>_<set>.npz, sources.json
weights/<model>@<revision>/                   pinned weights                           ← weights stage
results/<version>/<model>/<set>/              model.json + tables (.csv, .parquet):    ← analyse
    summary  summary_clean  per_stay  scores  selftest  reference_shifts  auroc  auroc_by_draw  temporal
    reliance  pairwise_dominance  informativeness  informativeness_per_stay
figures/<version>/<model>/<set>/*.png         reference_shifts_*, auroc_*, temporal_*, reliance_*
results/<version>/zoo/<set>.csv|parquet       the cross-model table                    ← zoo
results/<version>/all/<table>.csv|parquet     every model x set stacked                ← collect
```
The store holds derived data only (embeddings, scores, record ids, results). Note text, images and signals are read at run time and never written there, as the MIMIC data use agreement requires.

## 5. Models
| model | takes | uv group | mean swap |
|---|---|---|---|
| `llava-med` | x + n | llava-med | projector output |
| `medgemma-4b` | x + n | hf-modern | projector output |
| `chexagent` | x + n | chexagent | not offered |
| `opentslm-sp` (Llama-3.2-1B) | e + n | opentslm | projector output + label statistics |
| `opentslm-sp-gemma` (Gemma-3-1B) | e + n | opentslm | projector output + label statistics |

A model runs every set inside the modalities it takes. `mmd modsets` shows which.

### Adding a model
1. Write `mmd/models/<name>.py`: a class decorated with `@register` that sets `name`, `hf_id`, `revision`, `group`, `signal`, `modalities`, `default_modset`, `readouts` (`ReadoutSpec`s) and `meta`.
   - Subclass `HFImageTextAdapter` if it loads with `AutoModelForImageTextToText`.
   - Otherwise subclass `Adapter` and implement `load()` and `forward()`.
   - A model taking several signals reads `sample.signals[mod]`.
2. Optional: `signal_mean(signals, mod)` / `inject_mean(mean, mod)` with `supports_mean = True` for the mean swap, and `extra_readouts()` for pre-fusion lookups.
3. Add it to `MODULES` in `models/base.py`, plus a dependency group in `pyproject.toml` if it needs its own pins.
4. Run steps 2b–4, starting with `--smoke 3`.

## 6. Known gaps
- **No adapter takes two signals yet**, so sets such as `cxr_ecg_notes` build and count but cannot run.
- **Vitals and labs** have modality keys but no slot table or loader. Add a builder to `data.SLOT_BUILDERS` and a loader to `data.signal_loader`.
- **First VM run of every model:** read the `--smoke` output (Yes/No token ids, the signal span, module names).
- **CheXagent:** the image-token span is unverified, and the mean swap is skipped.
- **Pinning:** pin `revision` for each model, and `LLAVA_MED_COMMIT`, once the smoke test passes.
- **OpenTSLM-Gemma:** check in `--smoke` that `load_pretrained` loads the Gemma checkpoint, and check its Yes/No ids.
- **OpenTSLM batch size:** new records run one at a time, while the notebook ran batches of 4. P(Yes) can differ in the last digits.

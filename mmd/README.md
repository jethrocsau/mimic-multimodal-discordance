# mmd: modality discordance across a model zoo

**The question:** when a multimodal model predicts 1-year mortality for a cardiac ICU stay, how much does its answer
rely on each modality (chest X-ray, ECG, radiology note, …), compared with how much that modality tells it on its own?

`mmd` gives every model exactly the same **perturbed inputs**: a modality removed, replaced by the patient's own later
record, by another patient's record, or by a cohort average. It measures how far each perturbation moves the model's
risk estimate, and puts every model in one comparable table.

Contents: [1 How it fits together](#1-how-it-fits-together) · [2 Concepts](#2-concepts) ·
[3 Setup](#3-setup) · [4 Running it](#4-running-it) · [5 Where everything goes](#5-where-everything-goes) ·
[6 Models](#6-models) · [7 Known gaps](#7-known-gaps)

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
- **Steps 3 and 4 read only stored embeddings.** They need no GPU and can be rerun freely.
- **Step 2 is resumable.** Rerunning it only fills in missing inputs.

### The modules
| step | module | what it owns |
|---|---|---|
| define | `modsets.py` | modality keys (`e` ECG, `x` CXR, `v` vitals, `l` labs, `n` note), `ModalitySet`, the registry of named sets |
| define | `records.py` | the record id: one string naming one complete model input |
| define | `data.py` | the cohort tables; per modality, which record fills each stay's `first` (baseline) and `last` (follow-up) slot; image / ECG / note loaders |
| define | `cohort.py` | `TierCohort` (recipients and fixed folds per tier), `DonorPlan` (donors per recipient), `Manifest` (all of it frozen and hashed) |
| define | `perturb.py` | the perturbation strategies, the configurations they generate, `build_manifest` |
| run | `models/` | one `Adapter` per model (contract in `models/base.py`): a `Sample` in, readouts out |
| run | `runner.py` | the manifest's missing records → adapter → store; plan, smoke test, cohort-mean injection |
| run | `store.py` | `EmbeddingStore`: one folder per model of readouts keyed by record id, with its index and run log |
| analyse | `analysis/probe.py` | `Embeddings` and `Probe`: fit on the reference embeddings per fold, score every perturbation |
| analyse | `analysis/effects.py` | every effect table: Δ risk, AUROC, within-patient effects, reliance and information shares |
| analyse | `analysis/charts.py`, `report.py` | the figures; `analyse_model` runs probe → effects → charts and writes the tables |
| compare | `zoo.py` | `Zoo`: the orchestration layer every model plugs into, and the cross-model table |
| — | `cli.py` | the `mmd` commands, each a thin call into `Zoo` |
| — | `config.py`, `fsio.py`, `prompts.py` | settings and paths, local / `gs://` file access, the shared question wording |

### What ties the steps together
1. **The record id.** `12345|e:first|x:d9876:first|n:first` means stay 12345 with its own baseline ECG and note, but
   the CXR of donor stay 9876. That one string is the manifest row, the store key, the resume check and the probe's
   lookup.
2. **The manifest hash.** Each modality set has one frozen manifest. `perturb` records its hash and `analyse` checks
   it, so every model is run and analysed on exactly the same inputs.
3. **The adapter contract.** Analysis never sees a model, only named arrays (`yes_probability`,
   `answer_mean_state`, …). Adding a model means writing one adapter.

---

## 2. Concepts

### Tiers and folds
The cohort has three nested tiers, `L1 ⊂ L2 ⊂ L3`, from the narrowest cardiac ICU definition to the broadest. Every
analysis runs per tier. Within a tier, stays are split into `n_splits` (5) fixed folds, stratified by outcome; every
probe is fitted on four folds and scored on the fifth.

### Modality sets
A set is a name plus two or more modality keys. `cxr_notes = (x, n)` and `ecg_notes = (e, n)` are built in. Add your
own in `mmd.toml` (`[mmd.modsets] cxr_ecg_notes = ["e", "x", "n"]`), with `--define cxr_ecg_notes=e,x,n` on any
command, or with `MMD_MODSETS="cxr_ecg_notes=e,x,n"`.
- Each set gets its own manifest, run records and results.
- A recipient needs a baseline record of **every** modality in the set, so larger sets have fewer recipients.

### Slots
For each modality, every stay has a `first` record from its baseline window and, when one exists, a `last` record
(the latest one in its follow-up window):
- **x (CXR):** the latest frontal baseline film (a lateral one only when there is no frontal film);
- **e (ECG):** the earliest baseline ECG;
- **n (note):** the earliest baseline radiology report, cut to `note_chars` characters in the prompt.

### Perturbation strategies
Each strategy replaces a **subset** of the set's modalities and keeps the rest at the stay's own baseline record.

| strategy | family (in tables) | the perturbed modalities become |
|---|---|---|
| (always) | `reference` | nothing perturbed: the stay's own baseline records |
| `ablate` | `ablate` | removed from the input |
| `within_patient` | `temporal` | the stay's own latest follow-up record |
| `matched_patient` | `matched` | a same-outcome donor's baseline record |
| `matched_label_swap` | `counterfactual` | an opposite-outcome donor's baseline record |
| `cohort_mean` | `mean` | signals: the cohort-mean encoder output; the note is removed |

- **`max_order` (default 2)** caps how many modalities are perturbed at once. Subsets are combinations: k modalities
  give Σ_{r ≤ max_order} C(k, r) subsets (k = 5: 5 + 10 = 15).
- **Donor, mean and ablate subsets stop at k − 1**, because replacing everything leaves nothing of the patient in the
  input. Within-patient may replace all k.
- **A joint donor swap takes every perturbed modality from the same donor.**
- **Donors** come from the recipient's own test fold and age group × sex cell, so a probe never trains on its donors.
  Each recipient has `n_draws` (20) donors; the first `n_draws_model` (5) get a model pass.
- **Core arms always run:** the reference and each "only modality m" input, because the information share needs them.
- **Inputs per recipient** (all strategies, max_order 2): 28 for two modalities, 79 for three, 201 for five.
  `mmd modsets` prints this per set.
- **Prompt factors:** tier L1's reference and single-modality label swaps are also run with the other slot order and
  two paraphrased questions, to check that results do not hinge on the wording. They are marked in a `factor` column
  and left out of the standard analysis; skip them with `perturb --no-factors`.

### Readouts (what a model pass stores)
All are final-layer hidden states unless noted:

| readout | meaning |
|---|---|
| `yes_probability` | P(Yes) / (P(Yes) + P(No)) for the first answer token after "Answer:" |
| `prompt_last_state` | state at the last prompt token |
| `prompt_mean_state` | mean over every prompt token |
| `signal_mean_state` | mean over the image or ECG tokens |
| `answer_mean_state` | mean over the generated answer tokens |
| `answer_text` | the decoded answer |
| `encoder_mean`, `projector_mean` | the signal's vector before fusion: the vision-encoder output, and the projector output fed to the language model |

### The result, per model and modality m of the set
- **information share:** (AUROC of "only m" − 0.5)⁺ over the sum across modalities. What m tells the model on its own.
- **reliance share:** m's share of the summed |Δ risk| when m alone gets the label swap. How much the model leans on m.
- **bias = reliance − information.** Above 0, the model relies on m more than m alone justifies.

Each vector readout is reduced with PCA (fitted on the training folds) and probed with logistic regression and
XGBoost; `yes_probability` is also read raw, as the model's own risk score. For every pair of modalities a, b,
`pairwise_dominance` reports (b − a) / (|a| + |b|) per measure, in [−1, 1]: +1 only b matters, −1 only a.

---

## 3. Setup

### Prerequisites
- The cohort tables in BigQuery (`phase0`, built by [sql/phase0_dataform](../sql/phase0_dataform/)).
- The CXR JPGs and ECG files in the bucket (see [Raw files](#raw-files)).
- Credentialed PhysioNet access to MIMIC-IV, MIMIC-CXR(-JPG) and MIMIC-IV-ECG.
- A Hugging Face token whose account has accepted the terms of every gated model you run (`google/gemma-3-4b-it`,
  and `google/gemma-3-1b-pt` for `opentslm-sp-gemma`).

### Google Cloud: project, bucket, machines
```bash
gcloud auth login
gcloud config set project <your-gcp-project>
gcloud services enable compute.googleapis.com storage.googleapis.com bigquery.googleapis.com

# a private bucket in the GPU region
gcloud storage buckets create gs://<bucket> --location us-central1 \
    --uniform-bucket-level-access --public-access-prevention
# access for collaborators (PhysioNet-credentialed only), ideally through a group
gcloud storage buckets add-iam-policy-binding gs://<bucket> \
    --member group:<group>@<domain> --role roles/storage.objectAdmin
```
Set a budget alert on the project (Billing → Budgets & alerts).

Two machines cover everything:

| machine | for | create |
|---|---|---|
| CPU VM | define, analyse, compare | `gcloud compute instances create mmd-cpu --zone us-central1-a --machine-type e2-highmem-8 --image-family debian-12 --image-project debian-cloud --boot-disk-size 50GB --scopes cloud-platform` |
| GPU VM (Spot, 1× L4) | step 2, `perturb` | `gcloud compute instances create mmd-gpu --zone us-central1-a --machine-type g2-standard-8 --accelerator type=nvidia-l4,count=1 --provisioning-model SPOT --instance-termination-action STOP --maintenance-policy TERMINATE --boot-disk-size 200GB --image-family pytorch-latest-gpu --image-project deeplearning-platform-release --metadata install-nvidia-driver=True --scopes cloud-platform` |

- **Connect** with `gcloud compute ssh <name> --zone us-central1-a`. Run long commands inside `tmux`
  (`sudo apt-get install -y tmux`, `tmux new -s mmd`; detach with Ctrl-b d, reattach with `tmux attach -t mmd`), so
  they survive a dropped connection.
- **Stop each VM when it is idle:** `gcloud compute instances stop <name> --zone us-central1-a`. A stopped VM bills
  only its disk.
- **GPU errors:** a quota error means requesting `GPUS_ALL_REGIONS` = 1 under IAM & Admin → Quotas; "resource not
  available" means the zone has no L4s left, so try `us-central1-b` or `-c`.
- **A laptop works as the CPU machine** too, after `gcloud auth application-default login` (on macOS also
  `brew install libomp` for XGBoost; without it the analysis uses logistic regression only and says so).

### Raw files
`mmd` reads each image or waveform from `{raw_root}/<physionet project>/<version>/<relative path>`, with the
project, version and relative path taken from `phase0.x_file_manifest`. For example:
```
gs://<bucket>/physionet/mimic-cxr-jpg/2.0.0/files/p10/p10000032/s50414267/<dicom_id>.jpg
gs://<bucket>/physionet/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/40689238.hea  (+ .dat)
```
[download/download_all.sh](../download/download_all.sh) fetches exactly the files on that manifest. Upload them
under `gs://<bucket>/physionet/`, extracted and keeping their relative paths, **before** building a CXR cohort: only
films that exist are used.

### Install and configure (each machine)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone <repo-url> && cd mimic-multimodal-discordance
uv sync                                  # core only: define, analyse, compare (no torch)
```
Create `mmd.toml` in the repository root. It is git-ignored, and the `[mmd]` header is required:
```toml
[mmd]
root = "gs://<bucket>"                   # everything mmd reads and writes lives under here
project = "<your-gcp-project>"
cohort_version = "v1"                    # a new value starts a fresh, separate cohort
# dataset = "phase0"                     # the BigQuery dataset with the cohort tables
# raw_root = "gs://<bucket>/physionet"   # CXR / ECG files (default {root}/physionet)
# strategies = ["all"]                   # which strategies the manifests hold
# max_order = 2                          # at most this many modalities perturbed at once

[mmd.modsets]                            # optional: sets beyond cxr_notes / ecg_notes
# cxr_ecg_notes = ["e", "x", "n"]
```
Every key can also be set as an environment variable, `MMD_<KEY>` (e.g. `MMD_ROOT`). The remaining settings, with
their defaults:

| key | default | meaning |
|---|---|---|
| `levels` | L1, L2, L3 | the tiers analysed |
| `n_splits` | 5 | folds per tier |
| `n_draws` | 20 | donors drawn per recipient |
| `n_draws_model` | 5 | of those, how many get a model pass |
| `note_chars` | 1200 | the note is cut to this many characters in the prompt |
| `chunk` | 200 | records per stored chunk file |
| `pca_dim` | 512 | vector readouts are reduced to at most this many PCA components before probing |

Check the setup:
```bash
uv run mmd setup check                   # settings; bucket and BigQuery reachable; torch / CUDA if installed
uv run mmd modsets                       # sets, which models can run each, inputs per recipient
```

---

## 4. Running it

### Step 1: define (CPU)
```bash
uv run mmd data stage                    # BigQuery cohort tables -> {root}/data/v1/*.parquet
uv run mmd cohort build --modset cxr_notes
uv run mmd cohort build --modset ecg_notes
uv run mmd cohort build --modset cxr_ecg_notes --define cxr_ecg_notes=e,x,n --strategies all --max-order 2
```
`cohort build` prints the recipients per tier, the donor balance (standardised mean differences of age and sex) and
the configurations per strategy and order, then saves the manifest and its hash. Options:
- `--donor-plans <folder>`: reuse saved `donor_plan_<set>_<L>.csv` files instead of drawing donors, so the records
  match embeddings computed against those donors;
- `--pins <csv>`: fix which film fills a stay's CXR slot (`stay_id`, `dicom_id`);
- `--note-chest-only`: use chest-radiograph reports only for the note slot;
- `--force`: rebuild a manifest that already exists.

### Step 2: run the models (GPU VM)
Each model family has its own environment, because their library versions conflict. **Always use
`uv run --no-sync`** on the GPU VM: a plain `uv run` re-syncs to the core install and removes the model packages.
```bash
# once per model family
export UV_PROJECT_ENVIRONMENT=.venv-hf-modern HF_TOKEN=hf_...     # gemma3-4b, qwen2.5-vl-7b
uv sync --frozen --group hf-modern
#   llava-med:            .venv-llava-med  --group llava-med, then: uv run --no-sync mmd setup llava-med
#   opentslm-sp / -gemma: .venv-opentslm   --group opentslm
uv run --no-sync mmd setup check                                  # CUDA: True

# per model
uv run --no-sync mmd weights stage --model gemma3-4b              # Hugging Face -> local + {root}/weights/
uv run --no-sync mmd perturb --model gemma3-4b --smoke 3          # one input per arm, nothing stored
uv run --no-sync mmd perturb --model gemma3-4b --plan --sec-per-record 0.6   # records to do, GPU-hours
uv run --no-sync mmd perturb --model gemma3-4b                    # the full run, on the model's default set
```
- **Read the smoke output first:** the answers should look like Yes / No, `yes_probability` should lie in (0, 1), and
  the signal span should be found. It also reports seconds per record, for `--plan`.
- **Narrow a run:** `--modset cxr_notes`, `--levels L1`, `--strategies within_patient,matched_label_swap`,
  `--max-order 1`, `--no-factors`. Widen it later by rerunning without them; stored records are skipped.
- **Split across GPUs:** `--shard 0/2` on one VM and `--shard 1/2` on another.
- **Spot preemption:** start the VM again and rerun the same command; it resumes.

**Embeddings computed elsewhere.** A run folder of `chunk_*.npz` files keyed by the same record ids can be
registered read-only with a model instead of being recomputed; its records then count as stored:
```bash
uv run mmd index import --model <model> --run-folder gs://<bucket>/<run folder> \
    [--mean-from <cohort-mean .npz>] [--lookup <name>=<folder>]
uv run mmd status --modset cxr_notes     # records needed vs stored, per model and strategy
```
`--mean-from` supplies the cohort-mean signal that run injected for the `mean` strategy. `--lookup` registers
single-modality runs a model reads pre-fusion vectors from (OpenTSLM: `sp-prefix`, `flamingo`). Nothing is written
into the imported folders.

### Step 3: analyse (CPU)
```bash
uv run mmd analyse --model gemma3-4b --modset cxr_notes     # --no-charts / --no-selftest to skip those
```
It first runs a label-shuffle test (probes trained on shuffled outcomes should score AUROC ≈ 0.5), then probes every
readout × classifier × tier and writes the tables and figures. Records not stored yet are skipped.

### Step 4: compare
```bash
uv run mmd zoo --modset cxr_notes [--models llava-med,gemma3-4b] [--split-file <mimic-cxr-2.0.0-split.csv.gz>]
uv run mmd collect                       # every model × set stacked into results/v1/all/<table>
uv run mmd bq link --modset cxr_notes    # optional: BigQuery external tables over the parquet results
```
`--split-file` adds a row per model restricted to stays whose baseline film is outside the MIMIC-CXR training split.

### From Python (e.g. a notebook)
```python
from mmd import config
from mmd.zoo import Zoo
zoo = Zoo(config.load(), specs=['cxr_ecg_notes=e,x,n'])
zoo.compatibility()                               # model × set
zoo.analyse('gemma3-4b', 'cxr_notes')
zoo.table(modset='cxr_notes')                     # info_share, reliance_share, bias per model and modality
```
[notebooks/explore_results.ipynb](../notebooks/explore_results.ipynb) loads the result tables.

---

## 5. Where everything goes
Everything lives under `root`, so every machine with access to the bucket sees the same state, and VMs can be deleted
without losing anything.
```
physionet/<project>/<version>/...             raw CXR JPG / ECG files (you upload; raw_root)
data/<version>/*.parquet                      cohort tables snapshot                   ← data stage
cohort/<version>/<set>/                       slots, tiers + folds, donor plans,       ← cohort build
                                              configs.parquet, manifest.json (hash)
emb/<version>/<model>/                        chunk_*.npz (+ .records.txt), index,     ← perturb / index import
                                              runs.parquet, manifest.json (hash per set),
                                              cohort_mean_<mod>_<set>.npz, sources.json
weights/<model>@<revision>/                   pinned weights                           ← weights stage
results/<version>/<model>/<set>/              model.json + tables (.csv and .parquet)  ← analyse
figures/<version>/<model>/<set>/*.png         reference_shifts_*, auroc_*, temporal_*, reliance_*
results/<version>/zoo/<set>.csv|parquet       the cross-model table                    ← zoo
results/<version>/all/<table>.csv|parquet     every model × set stacked                ← collect
```
The store holds derived data only (embeddings, scores, record ids, results). Note text, images and signals are read
at run time and never written there, as the MIMIC data use agreement requires.

The tables `analyse` writes:

| table | what it holds |
|---|---|
| `summary`, `summary_clean` | per tier × readout × classifier × perturbation: Δ risk in decedents and survivors, their gap (bootstrap CI), ΔAUROC; `_clean` drops rows a readout cannot react to |
| `reference_shifts`, `auroc` | each perturbation against the reference, by outcome group; AUROC after each perturbation |
| `auroc_by_draw` | donor perturbations scored per single donor vs the mean over donors |
| `temporal` | within-patient swaps: representation similarity and outcome pull per modality |
| `reliance`, `pairwise_dominance` | each modality's share of the effect; for every pair, which one dominates |
| `informativeness`, `informativeness_per_stay` | AUROC of each "only m" arm and of all modalities together |
| `selftest` | the label-shuffle test |
| `per_stay`, `scores` | the per-stay long tables everything else is computed from |

---

## 6. Models
| model | takes | uv group | mean swap | MIMIC exposure |
|---|---|---|---|---|
| `llava-med` | x + n | llava-med | projector output | none (PubMed Central figures) |
| `gemma3-4b` | x + n | hf-modern | projector output | none (general web data) |
| `qwen2.5-vl-7b` | x + n | hf-modern | not offered | none (general web data) |
| `opentslm-sp` (Llama-3.2-1B) | e + n | opentslm | projector output + label statistics | none |
| `opentslm-sp-gemma` (Gemma-3-1B) | e + n | opentslm | projector output + label statistics | none |

A model runs every set inside the modalities it takes; `mmd modsets` shows which.

**Why no MIMIC-trained models.** A model trained on MIMIC-CXR (for example MedGemma, CheXagent, Lingshu, RadVLM or
MAIRA-2) may have seen this cohort's films and reports, so its reliance could reflect memorisation rather than
reading. Every model above is free of MIMIC training as far as its papers and model cards report. If one is ever
added, set `mimic_exposure` in its `meta`: `mmd zoo --exclude-exposed` then drops it, and `--split-file` re-scores
every model on stays whose baseline film is outside the MIMIC-CXR training split.

### Adding a model
1. Write `mmd/models/<name>.py`: a class decorated with `@register` that sets `name`, `hf_id`, `revision`, `group`,
   `signal`, `modalities`, `default_modset`, `readouts` (`ReadoutSpec`s) and `meta`.
   - Subclass `HFImageTextAdapter` (`models/hf_vlm.py`) if it loads with `AutoModelForImageTextToText`.
     `models/gemma.py` is a complete example; `models/qwen_vl.py` also overrides the input handling.
   - Otherwise subclass `Adapter` and implement `load()` and `forward()`.
   - A model taking several signals reads `sample.signals[mod]`.
2. Optional: `signal_mean(signals, mod)` / `inject_mean(mean, mod)` with `supports_mean = True` for the mean swap;
   `extra_readouts()` for pre-fusion lookups; `keymap` to read stored keys under other names.
3. Add it to `MODULES` in `models/base.py`, plus a dependency group in `pyproject.toml` if it needs its own pins.
4. Run step 2, starting with `--smoke 3`.

## 7. Known gaps
- **No adapter takes two signals yet**, so sets such as `cxr_ecg_notes` build and count but cannot run.
- **Vitals and labs** have modality keys but no slot table or loader. Add a builder to `data.SLOT_BUILDERS` and a
  loader to `data.signal_loader`.
- **First run of every model:** read the `--smoke` output (Yes / No token ids, the signal span, module names).
- **Qwen2.5-VL:** films are resized to a fixed 448×448 so every film gives the same number of image tokens; no mean
  swap yet.
- **Pinning:** pin `revision` for each model, and `LLAVA_MED_COMMIT`, once its smoke test passes.
- **OpenTSLM-Gemma:** check in `--smoke` that `load_pretrained` loads the Gemma checkpoint, and check its Yes / No ids.
- **OpenTSLM batch size:** records run one at a time. Embeddings imported from a batched run can differ from these in
  the last digits of P(Yes).

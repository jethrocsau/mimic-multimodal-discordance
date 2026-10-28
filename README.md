# mimic-multimodal-discordance

When a multimodal model predicts 1-year mortality for a cardiac ICU patient from MIMIC-IV, how much does its answer
rely on each kind of data (chest X-ray, ECG, radiology note, …), compared with how much that data tells it on its own?

This repository builds the patient cohort, collects each patient's data, and runs a set of multimodal models on
systematically perturbed inputs (a modality removed, replaced by a later record, by another patient's record, or by
a cohort average) to measure each model's reliance on each modality.

## What is in the repository

| folder | what it does | start here |
|---|---|---|
| [sql/phase0_dataform/](sql/phase0_dataform/) | BigQuery Dataform project: builds the cardiac ICU cohort from MIMIC-IV and extracts vitals, labs, ECGs, radiology notes and chest X-rays per patient, plus a download list of the image and waveform files | [its README](sql/phase0_dataform/README.md) |
| [download/](download/) | downloads the files on that list (CXR JPGs, ECG waveforms, reports) from PhysioNet, verified against PhysioNet's checksums | header of [download_all.sh](download/download_all.sh) |
| [mmd/](mmd/) | the experiment: defines the perturbed inputs, runs every model on them, analyses and compares the models (`mmd` command line) | [its README](mmd/README.md) |
| [notebooks/](notebooks/) | `explore_results.ipynb`: loads the result tables `mmd` writes | |
| [modality_eda/](modality_eda/) | exploratory notebooks per modality (CXR, ECG, EHR, vitals, coverage) | |

The Dataform project also sits at the repository root on the `dataform` branch, which is the branch to link to
BigQuery Dataform.

## End to end

```
 1. cohort (BigQuery)          2. raw files                  3. experiment (mmd)
 sql/phase0_dataform     ─►    download/download_all.sh  ─►  mmd data stage → cohort build → perturb (GPU)
 phase0.* tables               CXR / ECG files in a          → analyse → zoo
                               private bucket                results in the same bucket
```

1. **Build the cohort.** Run the Dataform project into a BigQuery dataset (`phase0`). See
   [sql/phase0_dataform/README.md](sql/phase0_dataform/README.md).
2. **Get the files.** Download the CXR images and ECG waveforms on `phase0.x_file_manifest` and place them in a
   private Cloud Storage bucket as `physionet/<project>/<version>/<relative_path>` (the paths PhysioNet uses).
3. **Run the experiment.** Follow [mmd/README.md](mmd/README.md): install with `uv`, point `mmd.toml` at the bucket,
   build the cohort manifest, run the models on a GPU VM, then analyse and compare on any machine.

## Requirements

- Credentialed PhysioNet access to MIMIC-IV, MIMIC-CXR(-JPG) and MIMIC-IV-ECG, with the Google account linked in
  your PhysioNet profile (this gives BigQuery access to `physionet-data`).
- A Google Cloud project with BigQuery, Cloud Storage and (for model runs) a GPU quota.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).

## Data use

MIMIC data are covered by the PhysioNet credentialed data use agreement. Keep the bucket private and share it only
with credentialed collaborators. `mmd` writes only derived data (embeddings, scores, record ids, results) and reads
notes, images and signals at run time. Never put MIMIC data, or anything derived from individual records, in a
public repository or on a public model hub.

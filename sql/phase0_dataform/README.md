# MIMIC-IV multimodal cardiac ICU cohort (Dataform)

A BigQuery Dataform project that builds a cardiac ICU cohort from MIMIC-IV and extracts
five data modalities for each patient: **vitals, labs, ECG, radiology notes and chest
X-rays**. It produces:

- one analysis table with one row per patient (outcome, demographics, which modalities
  are available),
- one record-level table per modality,
- a download list for the CXR images and ECG waveforms, which are files on PhysioNet and
  not in BigQuery,
- report tables with cohort attrition and modality coverage.

The outcome is death within 1 year of ICU admission.

## Pipeline

```
cohort_eligibility ─► cohort ─► x_stays ─► x_vitals, x_labs, x_ecg, x_notes, x_cxr ─► stay_summary ─► analysis_dataset ─► report_*
                                                  └─► x_file_manifest (download list)
```

| folder | tag | tables | what it holds |
|---|---|---|---|
| `definitions/00_sources` | — | — | declarations of the `physionet-data` source tables |
| `definitions/01_cohort` | `cohort` | `cohort_eligibility`, `cohort` | who is in the study |
| `definitions/02_extract` | `extract` | `x_*` | the records for each modality |
| `definitions/03_stay` | `stay` | `stay_summary`, `analysis_dataset` | one row per stay |
| `definitions/04_memo` | `memo` | `report_*`, `modality_combos` | summary tables |
| `definitions/05_checks` | `checks` | `assert_*` | data checks |

`workflow_settings.yaml` holds the project, dataset and time-window settings.
`includes/constants.js` holds shared names and SQL helpers.

## Before you start

- PhysioNet credentialed access to MIMIC-IV, MIMIC-IV-Note, MIMIC-IV-ECG, MIMIC-CXR and
  MIMIC-CXR-JPG, with BigQuery access to `physionet-data` granted to your Google account.
- A Google Cloud project with billing enabled.

## Load and run

The project sits at the repository root on the `dataform` branch, and under
`sql/phase0_dataform/` on `main`. BigQuery Dataform needs the project at the repository
root, so link it to the `dataform` branch. The command line works from either.

In both cases, first set `defaultProject` in `workflow_settings.yaml` to your Google
Cloud project ID.

### Option A: BigQuery Dataform (in the browser)

1. In the Google Cloud console, open **BigQuery → Dataform** and click **Create
   repository**. Pick a US region, for example `us-central1`.
2. Give the repository's service account the **BigQuery Job User** and **BigQuery Data
   Editor** roles on your project. The console shows the account name when the
   repository is created.
3. Open the repository, go to **Settings → Connect with Git** and enter:
   - the remote URL of this repository,
   - default branch `dataform`,
   - a GitHub personal access token, stored as a secret in Secret Manager. The service
     account needs the **Secret Manager Secret Accessor** role on that secret.
4. Create a **development workspace** and click **Pull from default branch**.
5. Set `defaultProject` in `workflow_settings.yaml`.
6. Click **Start execution** and choose **All actions**, or pick a tag.

You can also skip step 3 and copy the files into the workspace by hand.

PhysioNet grants BigQuery access to your personal Google account, not to service
accounts. If the run fails with a permission error on `physionet-data`, use Option B,
which runs as you.

### Option B: command line

```bash
npm i -g @dataform/cli
gcloud auth application-default login

cd sql/phase0_dataform       # on main; skip this on the dataform branch
dataform init-creds          # choose ADC, your project, location US
dataform compile             # checks the project for errors
dataform run                 # builds everything
```

To build one stage, use its tag, in this order:

```bash
dataform run --tags cohort
dataform run --tags extract   # the large step
dataform run --tags stay
dataform run --tags memo
dataform run --tags checks
```

`init-creds` writes `.df-credentials.json`. It is in `.gitignore`; keep it out of git.

The output tables are written to the `phase0` dataset, and failed checks to
`phase0_assertions`.

## Settings

Set in `workflow_settings.yaml` under `vars`:

| setting | default | meaning |
|---|---|---|
| `pre_h` | 72 | the baseline window starts this many hours before ICU admission |
| `post_h` | 24 | the baseline window ends this many hours after ICU admission |
| `follow_tail_h` | 24 | follow-up ends this many hours after ICU discharge or in-hospital death, whichever is first |
| `cxr_jpg_version`, `cxr_version`, `ecg_version` | 2.1.0, 2.1.0, 1.0 | PhysioNet dataset versions used to build download URLs |

After changing a window setting, rerun from `extract` onwards. The cohort does not depend
on the windows.

## Cohort

Every level requires age ≥ 18, an ICU stay of at least 24 hours, and no in-hospital death
in the first 24 hours of the ICU stay. Each patient contributes their first eligible stay.

| `level_id` | definition |
|---|---|
| L1 | coronary care unit (CCU) stay with a heart failure diagnosis code (ICD-9 428.x or ICD-10 I50.x, any position) |
| L2 | CCU stay, any diagnosis |
| L3 | CCU or cardiac vascular ICU (CVICU) stay, any diagnosis |

Every L1 patient is in L2, and every L2 patient is in L3. Each level picks its own first
eligible stay, so the same patient can have a different index stay in different levels.
Always filter on one level.

- `cohort_eligibility`: one row per ICU stay in MIMIC-IV, with a yes/no flag for each rule.
- `cohort`: one row per level and patient, with the index stay.

## Modality records (`x_*`)

| table | one row per | source | counted as |
|---|---|---|---|
| `x_vitals` | vital-sign measurement | ICU `chartevents`, routine vital signs | distinct chart time |
| `x_labs` | lab result | `labevents`, including ED and pre-admission draws | distinct specimen |
| `x_ecg` | 12-lead ECG | MIMIC-IV-ECG record list | distinct study |
| `x_notes` | radiology report, with full text | MIMIC-IV-Note `radiology` | distinct note |
| `x_cxr` | chest X-ray image | MIMIC-CXR-JPG metadata | distinct study |

`x_stays` lists the stays and their time windows. Each modality table holds every record
from the start of the baseline window to the end of follow-up, matched by patient and
time, with these columns:

- `in_l1_ccu_hf`, `in_l2_ccu`, `in_l3_ccu_cvicu`: which cohort levels the stay belongs to.
- `phase`: `baseline` (inside the baseline window) or `follow_up` (after it).
- `hours_from_icu_admit`: lets you narrow the window without extracting again.

## Analysis dataset

`analysis_dataset` is the table to work from. Filter on `level_id` to get one row per
patient.

| columns | meaning |
|---|---|
| `died_1y`, `days_to_death` | death within 365 days of ICU admission, and days to death |
| `age`, `sex`, `bmi`, `anchor_year_group` | demographics |
| `n_<modality>_base`, `n_<modality>_follow` | record counts in the baseline window and in follow-up |
| `has_vitals`, `has_labs`, `has_ecg`, `has_notes`, `has_cxr`, `has_all5` | at least one record in the baseline window |
| `base_mask`, `rep_mask` | the same information as bitmasks (Vitals 1, Labs 2, ECG 4, Notes 8, CXR 16). `rep_mask` needs a record in baseline and in follow-up. |

`stay_summary` holds the same columns with one row per stay, before the join to cohort
levels.

## Report tables

Every report table has a `level_id` column; filter on it.

| table | contents |
|---|---|
| `report_attrition` | patient counts after each inclusion rule (L1), and counts under stricter heart failure definitions |
| `report_level_overlap` | size of each level and how the levels overlap |
| `report_coverage_long` | patients and 1-year mortality for every combination of modalities, for baseline and for repeat (baseline and follow-up) availability |
| `report_coverage_baseline` | baseline rows of `report_coverage_long`, without the two-modality combinations |
| `report_coverage_repeat` | repeat rows of `report_coverage_long` |
| `report_coverage_base_vs_repeat` | baseline and repeat coverage side by side, with the loss between them |
| `report_demographics_by_outcome` | demographics of patients who died and who survived, with standardised mean differences |
| `report_swap_strata` | coverage and outcome counts in each age × sex × BMI group |
| `report_swap_readiness` | the groups in `report_swap_strata` graded by whether they hold enough died and surviving patients with all five modalities |
| `report_follow_window` | length of follow-up, by outcome |
| `report_extraction_summary` | rows and stays per modality |
| `report_download_summary` | number of files per PhysioNet project, file type and phase |

A row in the coverage tables counts every stay that has at least the listed modalities.
`modality_combos` is the lookup of the 32 combinations.

## Getting the data

The examples use L3 and the baseline window.

**Cohort, vitals, labs and notes** are complete in BigQuery:

```sql
SELECT * FROM phase0.analysis_dataset WHERE level_id = 'L3';
SELECT * FROM phase0.x_vitals WHERE in_l3_ccu_cvicu AND phase = 'baseline';
SELECT * FROM phase0.x_labs   WHERE in_l3_ccu_cvicu AND phase = 'baseline';
SELECT * FROM phase0.x_notes  WHERE in_l3_ccu_cvicu AND phase = 'baseline';
```

**CXR images, CXR reports and ECG waveforms** are files on PhysioNet. `x_file_manifest`
lists them, one row per stay and file:

| `file_role` | PhysioNet project | file |
|---|---|---|
| `cxr_image_jpg` | `mimic-cxr-jpg` | image, `.jpg` |
| `cxr_image_dicom` | `mimic-cxr` | image, `.dcm` |
| `cxr_report_txt` | `mimic-cxr` | report, `.txt`, one per study |
| `ecg_header`, `ecg_signal` | `mimic-iv-ecg` | `.hea` and `.dat`; both are needed to read a record |

Check the size of the download first:

```sql
SELECT file_role, n_stays, n_files
FROM phase0.report_download_summary
WHERE level_id = 'L3' AND phase = 'baseline';
```

Then export the URLs and download them:

```bash
# PhysioNet login, so wget does not prompt for each file
echo "machine physionet.org login YOUR_USER password YOUR_PASSWORD" >> ~/.netrc
chmod 600 ~/.netrc

bq query --use_legacy_sql=false --format=csv --max_rows=100000000 '
  SELECT DISTINCT url FROM phase0.x_file_manifest
  WHERE in_l3_ccu_cvicu AND phase = "baseline"
    AND file_role IN ("cxr_image_jpg", "cxr_report_txt", "ecg_header", "ecg_signal")' \
  | tail -n +2 > urls.txt

xargs -P 8 -n 50 wget -q -N -c -x -nH --cut-dirs=1 < urls.txt
```

Files are saved as `<physionet_project>/<version>/<relative_path>`, which matches the
columns in `x_file_manifest`, so you can join files back to `stay_id`. Rerunning the
`wget` line fetches only the files that are missing.

Keep downloaded data in a place covered by your PhysioNet data use agreement, and out of
git.

## Checks

A check fails if it returns any rows. The failing rows are saved in `phase0_assertions`.

- `assert_levels_nested`: every L1 patient is in L2, and every L2 patient is in L3.
- `assert_denominators`: every cohort stay has a summary row, and coverage totals equal
  cohort sizes.
- `assert_cxr_timestamps_parse`: CXR images whose date or time could not be read. They
  would otherwise drop out of every window.

Several tables also check for unique keys and missing values when they are built.

## Cost

`x_vitals` and `x_labs` are the large tables, because they cover the whole ICU stay for
every stay in the three levels. Rerunning only the `memo` stage reads small tables and
costs almost nothing.

## Definitions and limits

- Heart failure means an ICD-9 428.x or ICD-10 I50.x code in any position, acute or chronic.
- Notes are radiology reports only. `x_notes.is_chest_radiograph_report` marks reports of
  a chest X-ray, which describe the same exam as a CXR image.
- Vitals come from ICU charting only; emergency department vitals are not included.
- Records are matched by patient and time, so a record from another encounter counts if
  it falls inside the window.
- MIMIC-CXR covers 2011–2016 and was built from patients with an emergency department
  chest X-ray, so most missing CXRs are missing because of how the dataset was built.
- Death dates after discharge come from `patients.dod`. Confirm that they cover 365 days
  after ICU admission in your MIMIC-IV version before relying on `died_1y`.

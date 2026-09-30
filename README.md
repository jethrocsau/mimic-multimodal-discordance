# Phase 0 counting memo — Dataform project

BigQuery Dataform version of `phase0_counting_memo_v3.sql`. Same cohort logic, same
windows, same 1-year outcome; reorganised so each stage is a saved table that can be
rerun on its own, every modality is extracted once, and all counts come from the
extraction tables.

## Layout

```
workflow_settings.yaml     project, dataset, and the window parameters (vars)
includes/constants.js      unit names, cohort levels, modality bits, shared SQL builders
definitions/
  00_sources/              declarations of the physionet-data source tables
  01_cohort/   cohort_eligibility, cohort       who is in each cohort level
  02_extract/  x_*                              record-level extraction, one table per modality
  03_stay/     stay_summary, analysis_dataset   one row per stay: counts, bitmasks, outcome, demographics
  04_memo/     report_*, modality_combos        the memo tables
  05_checks/   assert_*                         assertions (plus inline ones in the config blocks)
```

Tags: `cohort`, `extract`, `stay`, `memo`, `checks`. Run a stage with its tag.

## Parameters (`workflow_settings.yaml` → `vars`)

| var | default | meaning |
|---|---|---|
| `pre_h` | 72 | modality window starts at ICU intime − `pre_h` hours |
| `post_h` | 24 | modality window ends at ICU intime + `post_h` hours (exclusive) |
| `follow_tail_h` | 24 | follow-up ends at LEAST(ICU outtime, in-hospital deathtime) + `follow_tail_h` |

Change a value and rerun from `02_extract` onwards; the cohort stage does not depend on them.
Set `defaultProject` to your own GCP project before running.

## Cohort levels (`cohort`, filter on `level_id`)

| level_id | definition |
|---|---|
| L1 | first eligible CCU stay with an HF ICD code (primary; = `report_attrition` row 8, memo Table A) |
| L2 | first eligible CCU stay, any diagnosis |
| L3 | first eligible CCU or CVICU stay, any diagnosis |

Common criteria: age ≥ 18, ICU LOS ≥ 24 h, no in-hospital death within 24 h of ICU
admission. Each level picks its own first eligible stay, so a patient can be anchored to
different stays in different levels (`report_level_overlap` rows 7–8 count this).

## Extraction (`x_*`)

`x_stays` is the union of every level's stays, with one flag per level
(`in_l1_ccu_hf`, `in_l2_ccu`, `in_l3_ccu_cvicu`) and the windows. Each modality table
(`x_vitals`, `x_labs`, `x_ecg`, `x_notes`, `x_cxr`) holds every record for those stays in
`[w_start, span_end)`, matched by `subject_id` + time, with:

- `phase` = `'baseline'` (inside the modality window) or `'follow_up'` (after it, up to `f_end`)
- `hours_from_icu_admit`, so you can narrow the window later (e.g. ±24 h) without re-extracting
- the level flags, so one extraction serves all three levels

```sql
-- primary cohort, frontal CXR images within ±24 h (download list)
SELECT stay_id, study_id, dicom_id, jpg_path
FROM phase0.x_cxr
WHERE in_l1_ccu_hf AND ViewPosition IN ('AP', 'PA')
  AND hours_from_icu_admit >= -24 AND hours_from_icu_admit < 24;
```

A patient can appear under more than one `stay_id` (see above); always filter by a level
flag to get one stay per patient.

Counting units: vitals = distinct charttime, labs = distinct specimen_id,
ECG = distinct study_id, notes = distinct radiology note_id, CXR = distinct study_id.

## Download manifest (`x_file_manifest`)

One row per (stay, file) with `physionet_project`, `physionet_version`, `file_role`,
`relative_path` and full `url`, plus the level flags, `phase` and `hours_from_icu_admit`
so you can filter exactly like the extraction tables.

| modality | project | file_role | path |
|---|---|---|---|
| CXR | `mimic-cxr-jpg` | `cxr_image_jpg` | `files/pXX/pSUBJECT/sSTUDY/DICOM_ID.jpg` |
| CXR | `mimic-cxr` | `cxr_image_dicom` | `files/pXX/pSUBJECT/sSTUDY/DICOM_ID.dcm` |
| CXR | `mimic-cxr` | `cxr_report_txt` | `files/pXX/pSUBJECT/sSTUDY.txt` (one per study) |
| ECG | `mimic-iv-ecg` | `ecg_header`, `ecg_signal` | `record_list.path` + `.hea` / `.dat` (both needed to read a WFDB record) |
| Notes | `mimic-iv-note` | `bigquery_row` | none: MIMIC-IV-Note is one CSV, not per-note files; the text is already in `x_notes.text` |

Vitals and labs have no files either; they are rows in `x_vitals` / `x_labs`.
Dataset versions are set in `workflow_settings.yaml` (`cxr_jpg_version`, `cxr_version`,
`ecg_version`). `report_download_summary` counts distinct files per level, project, role and
phase, so you can size a download first.

**Download a subset** (one project per list, because `--base` differs):

```bash
# 1. path list: primary cohort, frontal JPGs within +/-24 h
bq query --use_legacy_sql=false --format=csv --max_rows=10000000 '
  SELECT DISTINCT relative_path FROM phase0.x_file_manifest
  WHERE in_l1_ccu_hf AND file_role = "cxr_image_jpg"
    AND view_position IN ("AP", "PA")
    AND hours_from_icu_admit >= -24 AND hours_from_icu_admit < 24' | tail -n +2 > cxr_paths.txt

# 2. download only those files
wget -r -N -c -np -nH --cut-dirs=1 --user YOUR_PHYSIONET_USERNAME --ask-password \
  -i cxr_paths.txt --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/

# ECG: same pattern with file_role IN ("ecg_header", "ecg_signal")
#      and --base=https://physionet.org/files/mimic-iv-ecg/1.0/
```

The same file can appear on several rows (one patient anchored to different stays in
different levels), so always `SELECT DISTINCT relative_path`. Keep downloaded files in
an environment covered by your PhysioNet data use agreement.

## Analysis dataset (`stay_summary`, `analysis_dataset`)

Per stay: record counts per modality and phase, then two bitmasks
(bit 1 = Vitals, 2 = Labs, 4 = ECG, 8 = Notes, 16 = CXR):

- `base_mask`: modalities with ≥ 1 record in the modality window
- `rep_mask`: modalities with ≥ 1 record in the window **and** ≥ 1 in follow-up

`died_1y` = death (patients.dod, else in-hospital deathtime) within 365 days of ICU
admission; `days_to_death` lets you derive other horizons. `analysis_dataset` joins this to
the cohort levels and adds `has_vitals` … `has_cxr`, `has_all5`. **`analysis_dataset` is the
table to work from**: filter on `level_id` to get one row per patient.

## Memo tables (`report_*`)

| table | memo label | contents |
|---|---|---|
| `report_attrition` | Table A | L1 CONSORT attrition + HF rungs |
| `report_level_overlap` | — | level sizes and overlap |
| `report_coverage_long` | — | **long table**: level × definition (`baseline`/`repeat`) × all 32 combinations, with 1-year mortality |
| `report_coverage_baseline` (view) | Table C + C-S1 | `report_coverage_long` where baseline, Table C layout |
| `report_coverage_repeat` (view) | C-2 | `report_coverage_long` where repeat, all 32 combinations |
| `report_coverage_base_vs_repeat` (view) | C-S4 | baseline vs repeat side by side, with attrition |
| `report_demographics_by_outcome` | C-S2 | demographics by 1-year mortality, per set |
| `report_swap_strata`, `report_swap_readiness` | C-S3 | age × sex × BMI strata, swap readiness |
| `report_follow_window` | — | follow-up length by outcome |
| `report_extraction_summary` | — | rows and stays per modality and level |
| `report_download_summary` | — | distinct PhysioNet files per level, project, role and phase |

A stay covers a combination when `(stay_mask & combo_mask) = combo_mask`, so Table C,
C-S1, C-2 and C-S4 are consistent by construction. Every `report_*` table carries `level_id`;
filter on it.

## Checks

Inline (in config blocks): unique stay per level and patient per level (`cohort`),
unique stay (`cohort_eligibility`, `x_stays`, `stay_summary`), non-null outcome and masks
(`stay_summary`), one row per level × definition × combination (`report_coverage_long`).

Standalone (`05_checks`):
- `assert_levels_nested`: every L1 patient is in L2, every L2 patient is in L3
- `assert_denominators`: every cohort stay has a summary; coverage denominators equal cohort sizes
- `assert_cxr_timestamps_parse`: CXR images whose StudyDate/StudyTime failed to parse

An assertion fails if it returns any rows; the failing rows are in the
`phase0_assertions` dataset.

## Running

**BigQuery Studio → Dataform:** create a repository and workspace in the **US** region,
add these files, set `defaultProject`, then *Start execution* (all actions, or by tag).

**Access to physionet-data.** PhysioNet grants BigQuery access to your own Google
account. A Dataform workflow run as a service account will not have that access. If an
execution fails with a permission error on `physionet-data`, run with your own
credentials instead, from the CLI:

```bash
npm i -g @dataform/cli
cd phase0_dataform
gcloud auth application-default login
dataform init-creds          # choose "ADC", project = your defaultProject, location US
dataform compile             # check
dataform run --tags cohort   # or: extract, stay, memo, checks; no flag = everything
```

`.df-credentials.json` is created by `init-creds`; keep it out of version control.

**Cost/size.** `x_vitals` and `x_labs` are the large tables: they hold every record for
the whole ICU stay of every stay across the three levels (the follow-up phase is what
makes them big). They are clustered on `stay_id, phase`. Rerunning only `memo` reads the
small `stay_summary` table and costs almost nothing.

## Notes carried over from the memo

- HF = any-position ICD-9 428.x / ICD-10 I50.x (acute **and** chronic codes).
- Rung 2 uses itemid 50963 = NT-proBNP, > 125 pg/mL.
- Notes = radiology reports only; `x_notes.is_chest_radiograph_report` flags reports of a
  chest film, which describe the same exam as a CXR image.
- Vitals come from ICU chartevents only (mimiciv_ed not used).
- MIMIC-CXR covers 2011–2016 and was built from patients with an ED chest film in that
  period, so most CXR missingness is structural (see the loss breakdown).
- Verify that out-of-hospital death ascertainment in MIMIC-IV 3.1 covers 365 days after
  ICU admission before relying on `died_1y` (protocol Table B).

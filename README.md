# mimic-multimodal-discordance

Code for a study of multimodal data in cardiac ICU patients from MIMIC-IV.

| folder | contents |
|---|---|
| [sql/phase0_dataform](sql/phase0_dataform/) | BigQuery Dataform project that builds the cohort and extracts vitals, labs, ECG, radiology notes and chest X-rays. Start with its [README](sql/phase0_dataform/README.md) for setup and table descriptions. |

The same Dataform project sits at the repository root on the `dataform` branch, which is
the branch to link to BigQuery Dataform.

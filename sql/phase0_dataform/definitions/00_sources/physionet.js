// Source tables in physionet-data (read-only). Declared once so every SQLX file
// can use ${ref("<name>")} and Dataform can draw the dependency graph.

const SOURCES = {
  mimiciv_3_1_icu: ["icustays", "chartevents", "d_items"],
  mimiciv_3_1_derived: ["icustay_detail"],
  mimiciv_3_1_hosp: [
    "admissions",
    "patients",
    "diagnoses_icd",
    "labevents",
    "d_labitems",
    "emar",
    "pharmacy",
    "omr",
  ],
  mimiciv_ecg: ["record_list"],
  mimiciv_note: ["radiology", "radiology_detail"],
  mimic_cxr_jpg: ["metadata"],
};

Object.entries(SOURCES).forEach(([schema, names]) => {
  names.forEach((name) => {
    declare({ database: "physionet-data", schema, name });
  });
});

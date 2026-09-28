// Shared constants and small SQL builders used across the project.
// Files in includes/ are available in every SQLX file under their file name,
// e.g. ${constants.cxrStudyDt("StudyDate", "StudyTime")}.

const CCU = "Coronary Care Unit (CCU)";
const CVICU = "Cardiac Vascular Intensive Care Unit (CVICU)";

// Cohort levels. `flag` is the boolean column carried on x_stays and every x_* table.
const LEVELS = [
  { id: "L1", label: "L1. CCU + HF ICD (primary)",  flag: "in_l1_ccu_hf" },
  { id: "L2", label: "L2. CCU (no HF ICD)",         flag: "in_l2_ccu" },
  { id: "L3", label: "L3. CCU + CVICU (no HF ICD)", flag: "in_l3_ccu_cvicu" },
];

// Modalities. `bit` is the value used in base_mask / rep_mask.
const MODALITIES = [
  { key: "vitals", bit: 1,  label: "Vitals" },
  { key: "labs",   bit: 2,  label: "Labs" },
  { key: "ecg",    bit: 4,  label: "ECG" },
  { key: "notes",  bit: 8,  label: "Notes" },
  { key: "cxr",    bit: 16, label: "CXR" },
];
const ALL_MODALITIES_MASK = MODALITIES.reduce((acc, m) => acc | m.bit, 0); // 31

function bit(key) {
  return MODALITIES.find((m) => m.key === key).bit;
}

// CASE expression mapping level_id -> readable label.
function levelLabelCase(col) {
  return `CASE ${col} ${LEVELS.map((l) => `WHEN '${l.id}' THEN '${l.label}'`).join(" ")} END`;
}

// MIMIC-CXR-JPG StudyDate (YYYYMMDD) + StudyTime (HHMMSS.fff) -> DATETIME.
// FLOOR drops fractional seconds (never rounds up to an invalid time such as 23:60:00);
// LPAD restores the leading zeros lost for studies before 10:00.
function cxrStudyDt(dateCol, timeCol) {
  return `SAFE.PARSE_DATETIME('%Y%m%d%H%M%S', CONCAT(CAST(${dateCol} AS STRING),
    LPAD(CAST(CAST(FLOOR(SAFE_CAST(${timeCol} AS FLOAT64)) AS INT64) AS STRING), 6, '0')))`;
}

// Signed hours from an anchor time, 2 dp.
function hoursFrom(tCol, anchorCol) {
  return `ROUND(DATETIME_DIFF(${tCol}, ${anchorCol}, MINUTE) / 60.0, 2)`;
}

// Key columns every extraction table carries (alias = x_stays alias).
function extractKeys(alias) {
  return [
    `${alias}.stay_id`,
    `${alias}.subject_id`,
    `${alias}.hadm_id`,
    ...LEVELS.map((l) => `${alias}.${l.flag}`),
    `${alias}.intime AS anchor_icu_intime`,
  ].join(",\n  ");
}

// Same key columns, selected from a table that already carries them (an x_* table).
function carryKeys(alias) {
  return [
    `${alias}.stay_id`,
    `${alias}.subject_id`,
    `${alias}.hadm_id`,
    ...LEVELS.map((l) => `${alias}.${l.flag}`),
    `${alias}.anchor_icu_intime`,
  ].join(",\n    ");
}

// Record time inside the extraction span [w_start, span_end).
function inSpan(tCol, alias) {
  return `${tCol} >= ${alias}.w_start AND ${tCol} < ${alias}.span_end`;
}

// 'baseline' = inside the modality window [w_start, w_end);
// 'follow_up' = after it, up to f_end (the span only extends past w_end when f_end > w_end).
function phase(tCol, alias) {
  return `CASE WHEN ${tCol} < ${alias}.w_end THEN 'baseline' ELSE 'follow_up' END`;
}

// Bitmask over modalities from a per-modality boolean SQL condition.
function mask(conditionFor) {
  return MODALITIES.map((m) => `IF(${conditionFor(m.key)}, ${m.bit}, 0)`).join("\n    | ");
}

module.exports = {
  CCU,
  CVICU,
  LEVELS,
  MODALITIES,
  ALL_MODALITIES_MASK,
  bit,
  levelLabelCase,
  cxrStudyDt,
  hoursFrom,
  extractKeys,
  carryKeys,
  inSpan,
  phase,
  mask,
};

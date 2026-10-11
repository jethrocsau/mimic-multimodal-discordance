#!/usr/bin/env bash
# Download every modality for the whole phase0 cohort into one Google Drive folder.
# Written for Google Colab with Drive mounted.
#
# L1 is a subset of L2, which is a subset of L3, so nothing is split by level: every file
# list is the union of all three levels, baseline and follow-up, and each file is
# downloaded once. The level flags (in_l1_ccu_hf, in_l2_ccu, in_l3_ccu_cvicu) stay as
# columns in the exported tables and in x_file_manifest, so a level is a filter later.
#
#   BigQuery tables  -> $OUT/tables/<table>.parquet   (vitals, labs, notes, cohort, manifest, ...)
#   PhysioNet files  -> $OUT/physionet/<project>/<version>/shards/<shard>.tar (+ <shard>.txt index)
#                       mimic-iv-ecg (.hea + .dat), mimic-cxr (report .txt), mimic-cxr-jpg (.jpg)
#
# Files are fetched from the paths in phase0.x_file_manifest, from PhysioNet's Google Cloud
# Storage mirror when this account has access (much faster from Colab), else from
# physionet.org. Each chunk is downloaded to local disk, checked against SHA256SUMS.txt, and
# written to Drive as one .tar, because Drive is slow with many small files. Inside a .tar
# the paths are the manifest's relative_path. Rerunning skips files already on Drive, so a
# dropped Colab session just needs the same command again. The extract part unpacks the
# shards onto local disk for analysis.
#
# Cloud Storage access: link this Google account in your PhysioNet profile, then request
# access on each project page ("Files" section). The buckets are requester pays, billed
# to $PROJECT.
#
# Settings and the PhysioNet login are read from a .env file (see .env.example):
#   ENV_FILE=/content/drive/MyDrive/mimic_mmd/.env bash download_all.sh [part ...]
# ENV_FILE defaults to .env next to this script. Values already set in the environment win.
#
# The script runs in parts, in this order when none are named:
#   size      print file counts and estimated GB (downloads nothing)
#   tables    export the BigQuery tables to Parquet
#   lists     build the list of PhysioNet files to fetch
#   download  fetch the files
#   verify    compare Drive with the lists (DEEP_VERIFY=1 also reads every shard back)
#   extract   unpack the shards to $EXTRACT_TO on local disk (only when named)
#
# Test on a few files first:  LIMIT=50 bash download_all.sh lists download verify
#
# This script wires the steps together; the Python helpers next to it do the work:
#   bq_export.py      size, tables and lists (BigQuery)
#   check_chunk.py    checks a downloaded chunk against SHA256SUMS.txt
#   restore_paths.py  puts files copied flat from Cloud Storage back at their paths
# Keep all four files in the same folder.

set -euo pipefail
export LC_ALL=C   # comm and sort must agree on ordering

ENV_FILE="${ENV_FILE:-$(dirname "$0")/.env}"
if [ -f "$ENV_FILE" ]; then
  # Keep anything set on the command line; take the rest from the .env file.
  saved_env=$(export -p)
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
  eval "$saved_env"
else
  echo "No .env at $ENV_FILE; using the environment only." >&2
fi

: "${PROJECT:?set PROJECT (the GCP project that holds the phase0 dataset) in .env}"
DATASET="${DATASET:-phase0}"
OUT="${OUT:-/content/drive/MyDrive/mimic_mmd}"           # final location, on Drive
STAGE="${STAGE:-/content/stage}"                         # local disk, emptied after each chunk
IMAGE_FILTER="${IMAGE_FILTER:-frontal_one_per_study}"    # frontal_one_per_study | frontal | all
SOURCE="${SOURCE:-auto}"                                 # auto (Cloud Storage, else physionet.org) | gcs | physionet
WORKERS="${WORKERS:-6}"                                  # parallel wget processes for physionet.org; keep <= 8
CHUNK="${CHUNK:-5000}"                                   # files per shard (one .tar on Drive)
LIMIT="${LIMIT:-0}"                                      # >0: download at most this many new files per project
EXTRACT_TO="${EXTRACT_TO:-/content/data}"                # local disk target for the extract part
DEEP_VERIFY="${DEEP_VERIFY:-0}"                          # 1: verify lists the contents of every shard

# PhysioNet projects to download, verify and extract, smallest files first; e.g.
# PROJECTS="mimic-iv-ecg" for ECG only. lists always writes all three.
read -ra PROJECTS <<< "${PROJECTS:-mimic-cxr mimic-iv-ecg mimic-cxr-jpg}"

PARTS=("$@")
[ ${#PARTS[@]} -eq 0 ] && PARTS=(size tables lists download verify)

mkdir -p "$OUT"/{tables,lists,logs,physionet} "$STAGE"

# The Python helpers live next to this script and read their settings from the environment.
HERE="$(cd "$(dirname "$0")" && pwd)"
export PROJECT DATASET OUT STAGE IMAGE_FILTER

part_size() { python3 "$HERE/bq_export.py" size; }

part_tables() {
  python3 -c "import google.cloud.bigquery_storage" 2>/dev/null || pip -q install google-cloud-bigquery-storage
  python3 "$HERE/bq_export.py" tables
}

part_lists() { python3 "$HERE/bq_export.py" lists; }

version_of() { awk -F, -v p="$1" '$1 == p {print $2}' "$OUT/lists/versions.csv"; }

# Bucket names follow <project>-<version>.physionet.org; override one with e.g.
# GCS_BUCKET_mimic_cxr_jpg=other-bucket-name.
gcs_bucket() {
  local var="GCS_BUCKET_${1//-/_}"
  echo "${!var:-$1-$2.physionet.org}"
}

gcs() { gcloud storage --billing-project="$PROJECT" --verbosity=error "$@"; }

# The PhysioNet login is optional when every project comes from Cloud Storage.
setup_physionet_auth() {
  [ -n "${PHYSIONET_USER:-}" ] && [ -n "${PHYSIONET_PASS:-}" ] || return 0
  # A private wgetrc on local disk keeps the password off the command line.
  export WGETRC="$STAGE/wgetrc"
  (umask 077; printf 'user = %s\npassword = %s\nauth_no_challenge = on\n' \
    "$PHYSIONET_USER" "$PHYSIONET_PASS" > "$WGETRC")
  PHYSIONET_AUTH=1
}

# Prints gcs or physionet for a project.
pick_source() {
  local project=$1 bucket=$2 list=$3
  [ "$SOURCE" = physionet ] && { echo physionet; return; }
  # Cloud Storage copies flatten paths into one folder, so file names must be unique.
  if [ "$(awk -F/ '{print $NF}' "$list" | sort | uniq -d | wc -l)" -ne 0 ]; then
    echo "$project: file names repeat across folders, so Cloud Storage is not used" >&2
  elif gcs ls "gs://$bucket/$(head -n 1 "$list")" > /dev/null 2>&1; then
    echo gcs; return
  else
    echo "$project: no access to gs://$bucket (link your Google account and request Cloud access on PhysioNet)" >&2
  fi
  [ "$SOURCE" = gcs ] && { echo none; return; }
  echo physionet
}

download_project() {
  local project=$1 version base bucket src dest list done_list failed sums work
  version=$(version_of "$project")
  base="https://physionet.org/files/$project/$version"
  bucket=$(gcs_bucket "$project" "$version")
  dest="$OUT/physionet/$project/$version/shards"
  list="$OUT/lists/$project.txt"
  done_list="$OUT/logs/$project.done"
  failed="$OUT/logs/$project.failed"
  sums="$OUT/lists/$project.SHA256SUMS.txt"
  work="$STAGE/$project"

  mkdir -p "$dest"
  touch "$done_list"
  : > "$failed"
  rm -rf "$work"
  mkdir -p "$work"

  src=$(pick_source "$project" "$bucket" "$list")
  case "$src" in
    none) echo "$project: SOURCE=gcs but Cloud Storage cannot be used" >&2; exit 1 ;;
    physionet)
      if [ -z "${PHYSIONET_AUTH:-}" ]; then
        echo "$project: physionet.org needs PHYSIONET_USER and PHYSIONET_PASS in .env" >&2; exit 1
      fi
      if ! wget -q -O /dev/null "$base/$(head -n 1 "$list")"; then
        echo "$project: PhysioNet refused the first file. Check PHYSIONET_USER/PASS and that you signed this project's DUA." >&2
        exit 1
      fi ;;
  esac

  if [ ! -s "$sums" ] && [ "$src" = gcs ]; then
    gcs cp "gs://$bucket/SHA256SUMS.txt" "$sums" > /dev/null 2>&1 || rm -f "$sums"
  fi
  if [ ! -s "$sums" ] && [ -n "${PHYSIONET_AUTH:-}" ]; then
    wget -q -O "$sums" "$base/SHA256SUMS.txt" || rm -f "$sums"
  fi
  [ -s "$sums" ] || echo "$project: no SHA256SUMS.txt; non-empty files are kept unchecked" >&2
  touch "$sums"

  comm -23 "$list" <(sort -u "$done_list") > "$work/todo"
  if [ "$LIMIT" -gt 0 ]; then
    head -n "$LIMIT" "$work/todo" > "$work/todo.limit"
    mv "$work/todo.limit" "$work/todo"
  fi
  echo "$project $version from $src: $(wc -l < "$list") listed, $(wc -l < "$work/todo") to download"
  [ -s "$work/todo" ] || return 0
  split -l "$CHUNK" "$work/todo" "$work/chunk_"

  local chunk shard
  for chunk in "$work"/chunk_*; do
    rm -rf "$work/root" "$work/flat"
    mkdir -p "$work/root" "$work/flat"

    if [ "$src" = gcs ]; then
      # Copy in parallel into one flat folder, then move each file to its relative_path.
      sed "s|^|gs://$bucket/|" "$chunk" | gcs cp -I -c "$work/flat/" || true
      python3 "$HERE/restore_paths.py" "$chunk" "$work/flat" "$work/root"
    else
      sed "s|^|$base/|" "$chunk" \
        | (cd "$work/root" && xargs -P "$WORKERS" -n 50 \
             wget -q -x -nH --cut-dirs=3 --tries=5 --waitretry=10 --timeout=60 || true)
    fi

    python3 "$HERE/check_chunk.py" "$chunk" "$sums" "$work/root" "$work/ok" "$failed"

    # One .tar per chunk, written straight to Drive. The rename marks it complete, and the
    # .txt index written after it is what verify trusts.
    if [ -s "$work/ok" ]; then
      shard="$project-$(date -u +%Y%m%dT%H%M%S)-$$-$(basename "$chunk")"
      tar -C "$work/root" -cf "$dest/$shard.tar.part" -T "$work/ok"
      mv "$dest/$shard.tar.part" "$dest/$shard.tar"
      cp "$work/ok" "$dest/$shard.txt"
      cat "$work/ok" >> "$done_list"
    fi

    rm -f "$chunk"
    echo "  $(basename "$chunk"): $(wc -l < "$work/ok") saved to Drive, $(wc -l < "$failed") failed so far"
  done
  rm -rf "$work"
  [ -s "$failed" ] && echo "  $project: $(wc -l < "$failed") failed, listed in $failed; rerun the download part to retry them"
  return 0
}

part_download() {
  setup_physionet_auth
  for p in "${PROJECTS[@]}"; do download_project "$p"; done
}

# Rebuild the done lists from the shards actually on Drive, so files lost to a dropped
# session are downloaded again on the next run. A shard counts only with both its .tar
# and .txt; leftover .tar.part files from a dropped session are deleted.
part_verify() {
  local p version shards idx bad
  for p in "${PROJECTS[@]}"; do
    version=$(version_of "$p")
    shards="$OUT/physionet/$p/$version/shards"
    mkdir -p "$shards"
    rm -f "$shards"/*.tar.part
    bad=""
    for idx in "$shards"/*.txt; do
      [ -f "$idx" ] || continue
      if [ ! -f "${idx%.txt}.tar" ]; then
        rm -f "$idx"
      elif [ "$DEEP_VERIFY" = 1 ] && ! cmp -s <(tar -tf "${idx%.txt}.tar" 2>/dev/null | sort) "$idx"; then
        echo "  $(basename "${idx%.txt}.tar"): contents do not match its index; removed" >&2
        rm -f "$idx" "${idx%.txt}.tar"
        bad=$((${bad:-0} + 1))
      fi
    done
    for idx in "$shards"/*.txt; do
      if [ -f "$idx" ]; then cat "$idx"; fi
    done | sort -u > "$STAGE/$p.present"
    comm -12 "$OUT/lists/$p.txt" "$STAGE/$p.present" > "$OUT/logs/$p.done"
    comm -23 "$OUT/lists/$p.txt" "$STAGE/$p.present" > "$OUT/logs/$p.missing"
    echo "$p: $(wc -l < "$OUT/lists/$p.txt") listed, $(wc -l < "$OUT/logs/$p.done") on Drive in $(find "$shards" -name '*.tar' | wc -l) shards, $(wc -l < "$OUT/logs/$p.missing") missing${bad:+, $bad bad shards removed}"
  done
  [ -f "$OUT/logs/mimic-iv-ecg.done" ] && echo "ECG records with only one of .hea/.dat: $(sed -E 's/[.](hea|dat)$//' "$OUT/logs/mimic-iv-ecg.done" | uniq -c | awk '$1 < 2' | wc -l)"
  echo "Tables:"
  ls -lh "$OUT/tables"
}

# Unpack every shard onto local disk, where reads are far faster than from Drive:
#   $EXTRACT_TO/<project>/<version>/files/...  (the same layout as physionet.org)
# Unpacks the projects in PROJECTS.
part_extract() {
  local p version target t
  for p in "${PROJECTS[@]}"; do
    version=$(version_of "$p")
    target="$EXTRACT_TO/$p/$version"
    mkdir -p "$target"
    for t in "$OUT/physionet/$p/$version/shards"/*.tar; do
      if [ -f "$t" ]; then tar -C "$target" -xf "$t"; fi
    done
    echo "$p: $(find "$target" -type f | wc -l) files in $target"
  done
}

for part in "${PARTS[@]}"; do
  echo "######## $part"
  "part_$part"
done

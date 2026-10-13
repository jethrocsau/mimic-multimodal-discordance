"""BigQuery side of download_all.sh: size report, table export, PhysioNet file lists.

    python3 bq_export.py size | tables | lists

Settings come from the environment (download_all.sh exports them):
PROJECT, DATASET, OUT, STAGE, IMAGE_FILTER.
"""
import os
import shutil
import sys

from google.cloud import bigquery

PROJECT = os.environ["PROJECT"]
DATASET = os.environ.get("DATASET", "phase0")
OUT = os.environ["OUT"]
STAGE = os.environ["STAGE"]
IMAGE_FILTER = os.environ.get("IMAGE_FILTER", "frontal_one_per_study")

# Every x_* table already holds the union of all levels, so whole tables are exported.
TABLES = ["cohort", "x_stays", "stay_summary", "analysis_dataset",
          "x_vitals", "x_labs", "x_notes", "x_ecg", "x_cxr", "x_file_manifest"]


def t(name):
    return f"`{PROJECT}.{DATASET}.{name}`"


# CXR images in the manifest; `ranked` keeps frontal views and ranks them within a study (PA first).
IMG_SQL = f"""
WITH img AS (
  SELECT DISTINCT relative_path, study_id, view_position, record_id
  FROM {t('x_file_manifest')}
  WHERE file_role = 'cxr_image_jpg'
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY study_id ORDER BY view_position = 'PA' DESC, record_id) AS rn_study
  FROM img
  WHERE view_position IN ('AP', 'PA')
)"""

IMG_SELECT = {
    "all": "SELECT relative_path FROM img",
    "frontal": "SELECT relative_path FROM ranked",
    "frontal_one_per_study": "SELECT relative_path FROM ranked WHERE rn_study = 1",
}

client = bigquery.Client(project=PROJECT)


def query(sql):
    return client.query(sql).result()


def show(title, sql):
    """Print a query result as an aligned text table."""
    print(f"== {title}")
    rows = query(sql)
    cols = [f.name for f in rows.schema]
    data = [["" if v is None else str(v) for v in row.values()] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in data)) if data else len(c) for i, c in enumerate(cols)]
    for r in [cols] + data:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))
    print()


def column(sql):
    """One string per row from a single-column query."""
    return [row[0] for row in query(sql)]


def write_lines(path, lines):
    # Byte-order sort, matching LC_ALL=C sort/comm in download_all.sh.
    with open(path, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(set(lines)))
    return len(set(lines))


def size():
    show("Files per PhysioNet project and level (levels overlap; the download is the union)", f"""
        SELECT level_id, file_role, phase, n_stays, n_files
        FROM {t('report_download_summary')}
        WHERE file_role NOT IN ('bigquery_row', 'cxr_image_dicom')
        ORDER BY file_role, level_id, phase""")
    show("CXR JPGs to download, all levels and phases, by IMAGE_FILTER (about 1.5 MB per image)", f"""{IMG_SQL}
        SELECT image_filter, n_images, ROUND(n_images * 1.5 / 1024, 1) AS est_gb
        FROM (
          SELECT 'all' AS image_filter, (SELECT COUNT(*) FROM img) AS n_images
          UNION ALL SELECT 'frontal', COUNT(*) FROM ranked
          UNION ALL SELECT 'frontal_one_per_study', COUNTIF(rn_study = 1) FROM ranked
        )""")
    show("ECG files to download, all levels and phases (about 120 KB per record, .hea + .dat)", f"""
        SELECT COUNT(DISTINCT relative_path) AS n_files,
               ROUND(COUNT(DISTINCT relative_path) / 2 * 120 / 1024 / 1024, 1) AS est_gb
        FROM {t('x_file_manifest')}
        WHERE file_role IN ('ecg_header', 'ecg_signal')""")
    show("BigQuery rows per modality", f"SELECT * FROM {t('report_extraction_summary')}")


def tables():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from google.cloud import bigquery_storage

    bqs = bigquery_storage.BigQueryReadClient()
    out = f"{OUT}/tables"
    for name in TABLES:
        dst = f"{out}/{name}.parquet"
        if os.path.exists(dst):
            print(f"{name}: already on Drive, skipped")
            continue
        ref = f"{PROJECT}.{DATASET}.{name}"
        table = client.get_table(ref)
        # Tables are read with the Storage API (no query cost); views go through a query.
        rows = client.list_rows(table) if table.table_type == "TABLE" else query(f"SELECT * FROM `{ref}`")
        tmp = f"{STAGE}/{name}.parquet"
        writer, n = None, 0
        for batch in rows.to_arrow_iterable(bqstorage_client=bqs):
            if writer is None:
                writer = pq.ParquetWriter(tmp, batch.schema)
            writer.write_table(pa.Table.from_batches([batch]))
            n += batch.num_rows
        if writer is None:
            print(f"{name}: no rows, nothing written")
            continue
        writer.close()
        # Copy then rename, so a dropped session never leaves a half-written table on Drive.
        shutil.copy(tmp, dst + ".part")
        os.replace(dst + ".part", dst)
        os.remove(tmp)
        print(f"{name}: {n:,} rows")


def lists():
    if IMAGE_FILTER not in IMG_SELECT:
        sys.exit(f"unknown IMAGE_FILTER: {IMAGE_FILTER} (use {' | '.join(IMG_SELECT)})")
    manifest = t("x_file_manifest")
    d = f"{OUT}/lists"
    write_lines(f"{d}/versions.csv", column(f"""
        SELECT DISTINCT CONCAT(physionet_project, ',', physionet_version)
        FROM {manifest} WHERE relative_path IS NOT NULL"""))
    counts = {
        "mimic-cxr": write_lines(f"{d}/mimic-cxr.txt", column(f"""
            SELECT DISTINCT relative_path FROM {manifest} WHERE file_role = 'cxr_report_txt'""")),
        "mimic-iv-ecg": write_lines(f"{d}/mimic-iv-ecg.txt", column(f"""
            SELECT DISTINCT relative_path FROM {manifest}
            WHERE file_role IN ('ecg_header', 'ecg_signal')""")),
        "mimic-cxr-jpg": write_lines(f"{d}/mimic-cxr-jpg.txt", column(f"{IMG_SQL} {IMG_SELECT[IMAGE_FILTER]}")),
    }
    for project, n in counts.items():
        print(f"{project}: {n:,} files")


if __name__ == "__main__":
    commands = {"size": size, "tables": tables, "lists": lists}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(f"usage: {sys.argv[0]} {' | '.join(commands)}")
    commands[sys.argv[1]]()

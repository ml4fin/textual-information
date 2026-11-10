import csv
import os
from pathlib import Path
from typing import List, Optional

import psycopg2
from psycopg2.extras import execute_batch


# ============================================================================
# CONFIGURATION
# ============================================================================

# Allow large CSV fields (e.g. long article bodies).
csv.field_size_limit(10_000_000)

MANIFEST_FILE_PATH = os.getenv(
    "INGESTION_MANIFEST_PATH",
    "ingestion_manifest.csv",
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "research"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

SCHEMA_NAME = "risk_emergence"
STAGING_TABLE = "news_articles_staging"

TRUNCATE_STAGING_BEFORE_BATCH = True

EXPECTED_COLUMNS = [
    "Title",
    "Body",
    "Date",
    "Language",
    "Source",
    "Sentiment",
    "IsDuplicate",
    "dataType",
    "Url",
]

BATCH_SIZE = 2000
CSV_DELIMITER = "|"
CSV_ENCODING = "utf-8"


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_connection():
    """Open a connection to the PostgreSQL database."""

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "Database password not configured. "
            "Set the DB_PASSWORD environment variable."
        )

    return psycopg2.connect(**DB_CONFIG)


# ============================================================================
# INITIAL VALIDATION
# ============================================================================

def check_schema_and_tables(conn) -> None:
    """Verify that the required database tables exist."""

    query = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema = %s
      AND table_name IN (
          'news_articles_staging',
          'news_articles',
          'news_sources'
      )
    ORDER BY table_name;
    """

    with conn.cursor() as cur:
        cur.execute(query, (SCHEMA_NAME,))
        rows = cur.fetchall()

    found = {(row[0], row[1]) for row in rows}

    required = {
        (SCHEMA_NAME, "news_articles_staging"),
        (SCHEMA_NAME, "news_articles"),
        (SCHEMA_NAME, "news_sources"),
    }

    missing = required - found

    if missing:
        raise RuntimeError(
            f"Missing tables in schema {SCHEMA_NAME}: {missing}"
        )

    print("Schema validation completed successfully.")


def get_staging_columns(conn) -> List[str]:
    """Return and display the columns of the staging table."""

    query = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position;
    """

    with conn.cursor() as cur:
        cur.execute(query, (SCHEMA_NAME, STAGING_TABLE))
        rows = cur.fetchall()

    columns = [row[0] for row in rows]

    print("Staging table columns:")
    for column in columns:
        print(f"  - {column}")

    return columns


def validate_csv_header(header: List[str]) -> None:
    """Verify that the CSV header matches the expected structure."""

    if header != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected CSV header.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Found:    {header}"
        )

    print("CSV header validation completed successfully.")


# ============================================================================
# STAGING TABLE OPERATIONS
# ============================================================================

def truncate_staging(conn) -> None:
    """Remove all existing rows from the staging table."""

    sql = f"TRUNCATE TABLE {SCHEMA_NAME}.{STAGING_TABLE};"

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()
    print("Staging table truncated.")


def insert_batch_into_staging(conn, batch_rows: List[tuple]) -> None:
    """Insert a batch of rows into the staging table."""

    sql = f"""
    INSERT INTO {SCHEMA_NAME}.{STAGING_TABLE}
    (
        title,
        body,
        publication_date,
        language,
        source,
        sentiment,
        is_duplicate,
        data_type,
        url,
        _raw_line_number,
        _source_file
    )
    VALUES
    (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """

    with conn.cursor() as cur:
        execute_batch(
            cur,
            sql,
            batch_rows,
            page_size=BATCH_SIZE,
        )


# ============================================================================
# MANIFEST
# ============================================================================

def read_ingestion_manifest(manifest_path: str) -> List[dict]:
    """
    Read the ingestion manifest and return the enabled CSV files.

    Expected columns:
    enabled, file_path, max_rows, description
    """

    manifest_file = Path(manifest_path)

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_file}"
        )

    files_to_load = []

    with open(
        manifest_file,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file, delimiter=";")

        required_columns = {
            "enabled",
            "file_path",
            "max_rows",
            "description",
        }

        if (
            reader.fieldnames is None
            or not required_columns.issubset(set(reader.fieldnames))
        ):
            raise ValueError(
                "Invalid manifest.\n"
                f"Required columns: {required_columns}\n"
                f"Found columns:    {reader.fieldnames}"
            )

        for manifest_line, row in enumerate(reader, start=2):

            if row["enabled"].strip() != "1":
                print(
                    f"[MANIFEST] Line {manifest_line}: "
                    "file disabled, skipping."
                )
                continue

            file_path = row["file_path"].strip()
            description = row["description"].strip()
            max_rows_raw = row["max_rows"].strip()

            max_rows = (
                None
                if max_rows_raw == ""
                else int(max_rows_raw)
            )

            files_to_load.append(
                {
                    "file_path": file_path,
                    "max_rows": max_rows,
                    "description": description,
                }
            )

    print(
        f"Enabled files in manifest: {len(files_to_load)}"
    )

    return files_to_load


# ============================================================================
# CSV INGESTION
# ============================================================================

def load_csv_to_staging(
    csv_path: str,
    truncate_before_load: bool = True,
    max_rows: Optional[int] = None,
) -> dict:
    """
    Load a pipe-delimited CSV file into the staging table.

    max_rows can be used to limit ingestion during development or testing.
    """

    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_file}"
        )

    source_file_name = csv_file.name

    inserted_rows = 0
    skipped_rows = 0
    batch = []

    with get_connection() as conn:

        print("Database connection established.")

        check_schema_and_tables(conn)
        get_staging_columns(conn)

        if truncate_before_load:
            truncate_staging(conn)

        with open(
            csv_file,
            "r",
            encoding=CSV_ENCODING,
            newline="",
        ) as file:

            reader = csv.reader(
                file,
                delimiter=CSV_DELIMITER,
                quotechar='"',
            )

            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV file is empty.")

            validate_csv_header(header)

            for raw_idx, row in enumerate(reader, start=2):

                if (
                    max_rows is not None
                    and (inserted_rows + len(batch)) >= max_rows
                ):
                    print(
                        f"Maximum row limit reached: {max_rows}"
                    )
                    break

                if len(row) != len(EXPECTED_COLUMNS):
                    skipped_rows += 1

                    print(
                        f"[WARNING] Line {raw_idx}: "
                        f"unexpected number of columns "
                        f"({len(row)} instead of "
                        f"{len(EXPECTED_COLUMNS)}). "
                        "Row skipped."
                    )
                    continue

                (
                    title,
                    body,
                    date_,
                    language,
                    source,
                    sentiment,
                    is_duplicate,
                    data_type,
                    url,
                ) = row

                # Preserve source file and original line number for traceability.
                batch.append(
                    (
                        title,
                        body,
                        date_,
                        language,
                        source,
                        sentiment,
                        is_duplicate,
                        data_type,
                        url,
                        raw_idx,
                        source_file_name,
                    )
                )

                if len(batch) >= BATCH_SIZE:
                    insert_batch_into_staging(conn, batch)
                    conn.commit()

                    inserted_rows += len(batch)

                    print(
                        f"Inserted {inserted_rows} rows "
                        "into staging..."
                    )

                    batch = []

            if batch:
                insert_batch_into_staging(conn, batch)
                conn.commit()

                inserted_rows += len(batch)

    print()
    print("Staging ingestion completed.")
    print(f"Total rows inserted: {inserted_rows}")
    print(f"Total rows skipped:  {skipped_rows}")

    return {
        "file": source_file_name,
        "inserted_rows": inserted_rows,
        "skipped_rows": skipped_rows,
    }


def load_manifest_to_staging(
    manifest_path: str,
    truncate_before_batch: bool = True,
) -> None:
    """
    Load all enabled CSV files listed in the ingestion manifest.

    The staging table is truncated once before the first file.
    Subsequent files are appended.
    """

    files_to_load = read_ingestion_manifest(manifest_path)

    if not files_to_load:
        print(
            "No enabled files found in the manifest. "
            "Nothing to load."
        )
        return

    reports = []

    for file_index, item in enumerate(files_to_load):

        file_path = item["file_path"]
        max_rows = item["max_rows"]
        description = item["description"]

        print()
        print("=" * 80)
        print(f"Loading file: {file_path}")
        print(f"Description:  {description}")
        print(f"max_rows:     {max_rows}")
        print("=" * 80)

        # Truncate staging only before the first file in the batch.
        truncate_this_file = (
            truncate_before_batch
            and file_index == 0
        )

        report = load_csv_to_staging(
            csv_path=file_path,
            truncate_before_load=truncate_this_file,
            max_rows=max_rows,
        )

        reports.append(report)

    total_inserted = sum(
        report["inserted_rows"]
        for report in reports
    )

    total_skipped = sum(
        report["skipped_rows"]
        for report in reports
    )

    print()
    print("=" * 80)
    print("FINAL INGESTION REPORT")
    print("=" * 80)

    for report in reports:
        print(
            f"{report['file']}: "
            f"inserted={report['inserted_rows']}, "
            f"skipped={report['skipped_rows']}"
        )

    print("-" * 80)
    print(f"Total rows inserted: {total_inserted}")
    print(f"Total rows skipped:  {total_skipped}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the ingestion pipeline from the manifest."""

    load_manifest_to_staging(
        manifest_path=MANIFEST_FILE_PATH,
        truncate_before_batch=TRUNCATE_STAGING_BEFORE_BATCH,
    )


if __name__ == "__main__":
    main()
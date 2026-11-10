import hashlib
import os
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_batch


# ============================================================================
# CONFIGURATION
# ============================================================================

SOURCE_FILE_FILTER = None

FETCH_SIZE = 10000
INSERT_BATCH_SIZE = 2000
PROGRESS_EVERY_N_ROWS = 50000

MAX_EXAMPLES_PER_CATEGORY = 5
PRINT_REJECTION_EXAMPLES = True

# Set to False to test the complete pipeline without persisting changes.
COMMIT_FINAL_INSERTS = True

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "research"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

SCHEMA_NAME = "risk_emergence"
STAGING_TABLE = "news_articles_staging"
NEWS_TABLE = "news_articles"
SOURCES_TABLE = "news_sources"


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
# STRUCTURAL VALIDATION
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
        (SCHEMA_NAME, STAGING_TABLE),
        (SCHEMA_NAME, NEWS_TABLE),
        (SCHEMA_NAME, SOURCES_TABLE),
    }

    missing = required - found

    if missing:
        raise RuntimeError(
            f"Missing tables in schema {SCHEMA_NAME}: {missing}"
        )

    print("Schema validation completed successfully.")


def get_staging_row_count(
    conn,
    source_file_filter: Optional[str] = None,
) -> int:
    """Count records available in the staging table."""

    query = f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{STAGING_TABLE}"
    params: tuple = ()

    if source_file_filter is not None:
        query += " WHERE _source_file = %s"
        params = (source_file_filter,)

    with conn.cursor() as cur:
        cur.execute(query, params)
        count = cur.fetchone()[0]

    return count


def get_source_mapping(conn) -> Dict[str, int]:
    """Build a normalized source-to-source_id mapping."""

    query_cols = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position;
    """

    with conn.cursor() as cur:
        cur.execute(query_cols, (SCHEMA_NAME, SOURCES_TABLE))
        columns = [row[0] for row in cur.fetchall()]

    # Support alternative source-column names across schema versions.
    candidate_columns = [
        "source_domain",
        "domain",
        "source_raw",
        "source_code",
        "source_name",
    ]

    source_col = next(
        (
            candidate
            for candidate in candidate_columns
            if candidate in columns
        ),
        None,
    )

    if source_col is None:
        raise RuntimeError(
            "Unable to identify a source column in news_sources. "
            f"Available columns: {columns}"
        )

    query = f"""
    SELECT source_id, {source_col}
    FROM {SCHEMA_NAME}.{SOURCES_TABLE}
    WHERE {source_col} IS NOT NULL
    """

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    source_mapping = {
        str(row[1]).strip().lower(): row[0]
        for row in rows
        if row[1] is not None
    }

    print(
        f"Source column used for matching: {source_col}"
    )

    return source_mapping


# ============================================================================
# NORMALIZATION AND VALIDATION
# ============================================================================

def is_blank(value: Optional[str]) -> bool:
    """Return True if a value is null or empty."""

    return value is None or str(value).strip() == ""


def normalize_text_or_none(
    value: Optional[str],
) -> Optional[str]:
    """Return normalized text or None for blank values."""

    if is_blank(value):
        return None

    return str(value).strip()


def parse_date(value: Optional[str]) -> Optional[date]:
    """Parse supported date representations."""

    if is_blank(value):
        return None

    text = str(value).strip()

    candidate_formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]

    for fmt in candidate_formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    return None


def parse_float_or_none(
    value: Optional[str],
) -> Optional[float]:
    """Convert a numeric value to float when possible."""

    if is_blank(value):
        return None

    try:
        return float(str(value).strip())
    except ValueError:
        return None


def parse_duplicate_flag(
    value: Optional[str],
) -> Optional[bool]:
    """Parse supported boolean representations."""

    if is_blank(value):
        return None

    text = str(value).strip().lower()

    truthy = {"true", "t", "1", "yes"}
    falsy = {"false", "f", "0", "no"}

    if text in truthy:
        return True

    if text in falsy:
        return False

    return None


def looks_english(value: Optional[str]) -> bool:
    """Return True for recognized English language labels."""

    if is_blank(value):
        return False

    return str(value).strip().lower() in {
        "en",
        "eng",
        "english",
    }


def compute_content_hash_bytes(body: str) -> bytes:
    """Compute the SHA-256 hash of an article body."""

    return hashlib.sha256(
        body.encode("utf-8", errors="ignore")
    ).digest()


# ============================================================================
# REPORTING AND PROGRESS
# ============================================================================

def add_example(
    examples_dict: Dict[str, List[dict]],
    category: str,
    record: dict,
) -> None:
    """Store a limited number of rejected-record examples."""

    if category not in examples_dict:
        examples_dict[category] = []

    if len(examples_dict[category]) < MAX_EXAMPLES_PER_CATEGORY:
        examples_dict[category].append(record)


def format_seconds(seconds: float) -> str:
    """Format seconds as hh:mm:ss."""

    if seconds < 0:
        return "n/a"

    total = int(seconds)

    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(
    processed: int,
    total: int,
    inserted: int,
    rejected: int,
    start_time: float,
) -> None:
    """Print ingestion progress and estimated completion time."""

    elapsed = time.time() - start_time
    speed = processed / elapsed if elapsed > 0 else 0.0
    pct = (processed / total * 100.0) if total > 0 else 0.0

    remaining = max(total - processed, 0)
    eta_seconds = remaining / speed if speed > 0 else -1

    print(
        f"[PROGRESS] processed={processed:,}/{total:,} "
        f"({pct:6.2f}%) | inserted={inserted:,} "
        f"| rejected={rejected:,} "
        f"| speed={speed:,.0f} rec/s "
        f"| elapsed={format_seconds(elapsed)} "
        f"| ETA={format_seconds(eta_seconds)}"
    )


# ============================================================================
# RECORD TRANSFORMATION
# ============================================================================

def transform_staging_record(
    record: Dict[str, Any],
    source_mapping: Dict[str, int],
    rejection_examples: Dict[str, List[dict]],
) -> Optional[Tuple]:
    """
    Transform a staging record into the format expected by news_articles.

    Invalid records return None and are excluded from insertion.
    """

    title = normalize_text_or_none(record["title"])
    body = normalize_text_or_none(record["body"])

    if title is None:
        add_example(
            rejection_examples,
            "blank_title",
            record,
        )
        return None

    if body is None:
        add_example(
            rejection_examples,
            "blank_body",
            record,
        )
        return None

    publication_date = parse_date(
        record["publication_date"]
    )

    if publication_date is None:
        add_example(
            rejection_examples,
            "invalid_date",
            record,
        )
        return None

    source_raw = normalize_text_or_none(record["source"])

    if source_raw is None:
        add_example(
            rejection_examples,
            "blank_source",
            record,
        )
        return None

    source_key = source_raw.lower()
    source_id = source_mapping.get(source_key)

    if source_id is None:
        add_example(
            rejection_examples,
            "unmapped_source",
            record,
        )
        return None

    language = normalize_text_or_none(
        record["language"]
    )

    is_english = looks_english(language)

    sentiment_original = parse_float_or_none(
        record["sentiment"]
    )

    duplicate_flag = parse_duplicate_flag(
        record["is_duplicate"]
    )

    if duplicate_flag is None:
        add_example(
            rejection_examples,
            "invalid_is_duplicate",
            record,
        )
        return None

    data_type = normalize_text_or_none(
        record["data_type"]
    )

    url = normalize_text_or_none(
        record["url"]
    )

    ingestion_batch = normalize_text_or_none(
        record["_source_file"]
    )

    content_hash = compute_content_hash_bytes(body)

    return (
        title,
        body,
        publication_date,
        language,
        source_raw,
        sentiment_original,
        duplicate_flag,
        data_type,
        url,
        source_id,
        is_english,
        False,  # is_duplicate_verified
        False,  # is_excluded
        ingestion_batch,
        content_hash,
    )


# ============================================================================
# FINAL TABLE INSERT
# ============================================================================

def insert_batch_into_news(
    conn,
    batch_rows: List[Tuple],
) -> None:
    """Insert a batch of records into news_articles."""

    sql = f"""
    INSERT INTO {SCHEMA_NAME}.{NEWS_TABLE}
    (
        title,
        body,
        publication_date,
        language,
        source_raw,
        sentiment_original,
        is_duplicate_flag,
        data_type,
        url,
        source_id,
        is_english,
        is_duplicate_verified,
        is_excluded,
        ingestion_batch,
        content_hash
    )
    VALUES
    (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s
    )
    """

    with conn.cursor() as cur:
        execute_batch(
            cur,
            sql,
            batch_rows,
            page_size=INSERT_BATCH_SIZE,
        )


# ============================================================================
# STAGING -> NEWS INGESTION
# ============================================================================

def load_staging_into_news(
    conn,
    source_mapping: Dict[str, int],
    source_file_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stream records from staging, validate and transform them,
    and insert valid records into news_articles.
    """

    total_records = get_staging_row_count(
        conn,
        source_file_filter=source_file_filter,
    )

    report = {
        "total_records": total_records,
        "processed_records": 0,
        "inserted_records": 0,
        "rejected_records": 0,
        "rejection_examples": {},
    }

    if total_records == 0:
        return report

    query = f"""
    SELECT
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
    FROM {SCHEMA_NAME}.{STAGING_TABLE}
    """

    params: tuple = ()

    if source_file_filter is not None:
        query += " WHERE _source_file = %s"
        params = (source_file_filter,)

    query += " ORDER BY _source_file, _raw_line_number"

    insert_batch: List[Tuple] = []
    start_time = time.time()
    next_progress_threshold = PROGRESS_EVERY_N_ROWS

    # Server-side cursor prevents the full staging table from being
    # loaded into application memory.
    with conn.cursor(name="staging_to_news_cursor") as cur:

        cur.execute(query, params)
        cur.itersize = FETCH_SIZE

        while True:
            rows = cur.fetchmany(FETCH_SIZE)

            if not rows:
                break

            for row in rows:

                record = {
                    "title": row[0],
                    "body": row[1],
                    "publication_date": row[2],
                    "language": row[3],
                    "source": row[4],
                    "sentiment": row[5],
                    "is_duplicate": row[6],
                    "data_type": row[7],
                    "url": row[8],
                    "_raw_line_number": row[9],
                    "_source_file": row[10],
                }

                report["processed_records"] += 1

                final_row = transform_staging_record(
                    record=record,
                    source_mapping=source_mapping,
                    rejection_examples=report[
                        "rejection_examples"
                    ],
                )

                if final_row is None:
                    report["rejected_records"] += 1
                else:
                    insert_batch.append(final_row)

                if len(insert_batch) >= INSERT_BATCH_SIZE:
                    insert_batch_into_news(
                        conn,
                        insert_batch,
                    )

                    report["inserted_records"] += len(
                        insert_batch
                    )

                    insert_batch = []

                if (
                    report["processed_records"]
                    >= next_progress_threshold
                ):
                    print_progress(
                        processed=report[
                            "processed_records"
                        ],
                        total=report["total_records"],
                        inserted=report[
                            "inserted_records"
                        ],
                        rejected=report[
                            "rejected_records"
                        ],
                        start_time=start_time,
                    )

                    next_progress_threshold += (
                        PROGRESS_EVERY_N_ROWS
                    )

    if insert_batch:
        insert_batch_into_news(
            conn,
            insert_batch,
        )

        report["inserted_records"] += len(
            insert_batch
        )

    print_progress(
        processed=report["processed_records"],
        total=report["total_records"],
        inserted=report["inserted_records"],
        rejected=report["rejected_records"],
        start_time=start_time,
    )

    return report


# ============================================================================
# FINAL REPORT
# ============================================================================

def print_examples(
    title: str,
    examples: List[dict],
) -> None:
    """Print a compact sample of rejected records."""

    if not examples:
        return

    print()
    print(title)

    for example in examples:
        print(
            f"  file={example['_source_file']}, "
            f"line={example['_raw_line_number']}, "
            f"date={example['publication_date']}, "
            f"source={example['source']}, "
            f"language={example['language']}, "
            f"is_duplicate={example['is_duplicate']}, "
            f"title_preview="
            f"{repr((example['title'] or '')[:60])}"
        )


def print_load_report(
    report: Dict[str, Any],
) -> None:
    """Print the final staging-to-news ingestion report."""

    print()
    print("=" * 80)
    print("STAGING -> NEWS INGESTION REPORT")
    print("=" * 80)

    print(
        f"Total records read      : "
        f"{report['total_records']}"
    )
    print(
        f"Total records processed : "
        f"{report['processed_records']}"
    )
    print(
        f"Total records inserted  : "
        f"{report['inserted_records']}"
    )
    print(
        f"Total records rejected  : "
        f"{report['rejected_records']}"
    )

    if not PRINT_REJECTION_EXAMPLES:
        return

    rejection_examples = report["rejection_examples"]

    print_examples(
        "EXAMPLES - blank titles",
        rejection_examples.get("blank_title", []),
    )

    print_examples(
        "EXAMPLES - blank bodies",
        rejection_examples.get("blank_body", []),
    )

    print_examples(
        "EXAMPLES - invalid dates",
        rejection_examples.get("invalid_date", []),
    )

    print_examples(
        "EXAMPLES - blank sources",
        rejection_examples.get("blank_source", []),
    )

    print_examples(
        "EXAMPLES - unmapped sources",
        rejection_examples.get("unmapped_source", []),
    )

    print_examples(
        "EXAMPLES - invalid is_duplicate",
        rejection_examples.get(
            "invalid_is_duplicate",
            [],
        ),
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the staging-to-news ingestion pipeline."""

    with get_connection() as conn:

        # Explicit transaction control allows validation runs with rollback.
        conn.autocommit = False

        print("Database connection established.")

        check_schema_and_tables(conn)

        total_in_staging = get_staging_row_count(
            conn,
            source_file_filter=SOURCE_FILE_FILTER,
        )

        if total_in_staging == 0:
            print(
                "No records available in the staging table."
            )
            return

        print(
            f"Records to process: {total_in_staging:,}"
        )

        if SOURCE_FILE_FILTER is not None:
            print(
                f"Source file filter: {SOURCE_FILE_FILTER}"
            )

        source_mapping = get_source_mapping(conn)

        report = load_staging_into_news(
            conn=conn,
            source_mapping=source_mapping,
            source_file_filter=SOURCE_FILE_FILTER,
        )

        if COMMIT_FINAL_INSERTS:
            conn.commit()
            print("Transaction committed.")
        else:
            conn.rollback()
            print(
                "Transaction rolled back: "
                "no records persisted in news_articles."
            )

        print_load_report(report)


if __name__ == "__main__":
    main()
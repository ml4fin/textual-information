import csv
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2


# ============================================================================
# CONFIGURATION
# ============================================================================

def increase_csv_field_size_limit() -> None:
    """Set the largest CSV field size supported by the platform."""

    max_int = sys.maxsize

    while True:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int //= 10


increase_csv_field_size_limit()


SOURCE_FILE_FILTER = None

MAX_EXAMPLES_PER_CATEGORY = 5

PRINT_VALUE_DISTRIBUTIONS = True
MAX_DISTRIBUTION_ITEMS = 30

FETCH_SIZE = 10000
PROGRESS_EVERY_N_ROWS = 50000

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "research"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

SCHEMA_NAME = "risk_emergence"
STAGING_TABLE = "news_articles_staging"
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
        (SCHEMA_NAME, "news_articles"),
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


def get_allowed_sources(conn) -> set:
    """Return normalized source values available in news_sources."""

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

    selected_col = next(
        (
            candidate
            for candidate in candidate_columns
            if candidate in columns
        ),
        None,
    )

    if selected_col is None:
        raise RuntimeError(
            "Unable to identify a source column in news_sources. "
            f"Available columns: {columns}"
        )

    query_values = f"""
    SELECT DISTINCT {selected_col}
    FROM {SCHEMA_NAME}.{SOURCES_TABLE}
    WHERE {selected_col} IS NOT NULL
    """

    with conn.cursor() as cur:
        cur.execute(query_values)
        rows = cur.fetchall()

    allowed_sources = {
        str(row[0]).strip().lower()
        for row in rows
        if row[0] is not None
    }

    print(
        f"Source column used for matching: {selected_col}"
    )

    return allowed_sources


# ============================================================================
# FIELD VALIDATION
# ============================================================================

def is_blank(value: Optional[str]) -> bool:
    """Return True if a value is null or empty."""

    return value is None or str(value).strip() == ""


def is_valid_date_string(value: Optional[str]) -> bool:
    """Check whether a date matches one of the supported formats."""

    if is_blank(value):
        return False

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
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue

    return False


def is_valid_numeric_string(value: Optional[str]) -> bool:
    """Check whether a value is numeric or missing."""

    if is_blank(value):
        return True

    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False


def is_valid_duplicate_flag(value: Optional[str]) -> bool:
    """Check whether is_duplicate has a supported representation."""

    if is_blank(value):
        return True

    valid_values = {
        "true",
        "false",
        "t",
        "f",
        "1",
        "0",
        "yes",
        "no",
    }

    return str(value).strip().lower() in valid_values


def looks_english(value: Optional[str]) -> bool:
    """Return True for recognized English language labels."""

    if is_blank(value):
        return False

    return str(value).strip().lower() in {
        "en",
        "eng",
        "english",
    }


# ============================================================================
# REPORTING UTILITIES
# ============================================================================

def add_example(
    examples_dict: Dict[str, List[dict]],
    category: str,
    record: dict,
) -> None:
    """Store a limited number of problematic-record examples."""

    if category not in examples_dict:
        examples_dict[category] = []

    if len(examples_dict[category]) < MAX_EXAMPLES_PER_CATEGORY:
        examples_dict[category].append(record)


def update_distribution(
    distribution: Dict[str, int],
    key: Optional[str],
) -> None:
    """Update a value-frequency distribution."""

    normalized_key = "" if key is None else str(key).strip()

    distribution[normalized_key] = (
        distribution.get(normalized_key, 0) + 1
    )


def print_examples(
    title: str,
    examples: List[dict],
) -> None:
    """Print a compact sample of problematic records."""

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
            f"sentiment={example['sentiment']}, "
            f"title_preview="
            f"{repr((example['title'] or '')[:60])}"
        )


def print_distribution(
    title: str,
    distribution: Dict[str, int],
    max_items: int = 30,
) -> None:
    """Print the most frequent values in a distribution."""

    if not distribution:
        return

    sorted_items = sorted(
        distribution.items(),
        key=lambda item: (-item[1], item[0]),
    )

    print()
    print(title)

    for key, count in sorted_items[:max_items]:
        print(f"  {repr(key)} -> {count}")

    if len(sorted_items) > max_items:
        print(
            f"  ... {len(sorted_items) - max_items} "
            "additional values not shown"
        )


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
    start_time: float,
) -> None:
    """Print validation progress and estimated completion time."""

    elapsed = time.time() - start_time
    speed = processed / elapsed if elapsed > 0 else 0.0

    pct = (
        processed / total * 100.0
        if total > 0
        else 0.0
    )

    remaining = max(total - processed, 0)

    eta_seconds = (
        remaining / speed
        if speed > 0
        else -1
    )

    print(
        f"[PROGRESS] processed={processed:,}/{total:,} "
        f"({pct:6.2f}%) | speed={speed:,.0f} rec/s "
        f"| elapsed={format_seconds(elapsed)} "
        f"| ETA={format_seconds(eta_seconds)}"
    )


# ============================================================================
# REPORT INITIALIZATION
# ============================================================================

def initialize_report(
    total_records: int,
) -> Dict[str, Any]:
    """Initialize the staging-validation report."""

    return {
        "total_records": total_records,
        "valid_dates": 0,
        "invalid_dates": 0,
        "mapped_sources": 0,
        "unmapped_sources": 0,
        "valid_sentiment": 0,
        "invalid_sentiment": 0,
        "valid_is_duplicate": 0,
        "invalid_is_duplicate": 0,
        "blank_title": 0,
        "blank_body": 0,
        "english_like": 0,
        "non_english_like": 0,
        "examples": {},
        "language_distribution": {},
        "source_distribution": {},
        "duplicate_distribution": {},
    }


# ============================================================================
# RECORD VALIDATION
# ============================================================================

def process_record(
    record: Dict[str, Any],
    allowed_sources: set,
    report: Dict[str, Any],
) -> None:
    """Validate one staging record and update the report."""

    title = record["title"]
    body = record["body"]
    publication_date = record["publication_date"]
    language = record["language"]
    source = record["source"]
    sentiment = record["sentiment"]
    is_duplicate = record["is_duplicate"]

    update_distribution(
        report["language_distribution"],
        language,
    )
    update_distribution(
        report["source_distribution"],
        source,
    )
    update_distribution(
        report["duplicate_distribution"],
        is_duplicate,
    )

    if is_valid_date_string(publication_date):
        report["valid_dates"] += 1
    else:
        report["invalid_dates"] += 1
        add_example(
            report["examples"],
            "invalid_dates",
            record,
        )

    source_norm = (
        ""
        if source is None
        else str(source).strip().lower()
    )

    if source_norm and source_norm in allowed_sources:
        report["mapped_sources"] += 1
    else:
        report["unmapped_sources"] += 1
        add_example(
            report["examples"],
            "unmapped_sources",
            record,
        )

    if is_valid_numeric_string(sentiment):
        report["valid_sentiment"] += 1
    else:
        report["invalid_sentiment"] += 1
        add_example(
            report["examples"],
            "invalid_sentiment",
            record,
        )

    if is_valid_duplicate_flag(is_duplicate):
        report["valid_is_duplicate"] += 1
    else:
        report["invalid_is_duplicate"] += 1
        add_example(
            report["examples"],
            "invalid_is_duplicate",
            record,
        )

    if is_blank(title):
        report["blank_title"] += 1
        add_example(
            report["examples"],
            "blank_title",
            record,
        )

    if is_blank(body):
        report["blank_body"] += 1
        add_example(
            report["examples"],
            "blank_body",
            record,
        )

    if looks_english(language):
        report["english_like"] += 1
    else:
        report["non_english_like"] += 1


# ============================================================================
# STREAMING VALIDATION
# ============================================================================

def stream_validate_staging(
    conn,
    allowed_sources: set,
    source_file_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate staging records using server-side streaming.

    This avoids loading the complete staging table into application memory.
    """

    total_records = get_staging_row_count(
        conn,
        source_file_filter=source_file_filter,
    )

    report = initialize_report(
        total_records=total_records
    )

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

    start_time = time.time()
    processed = 0
    next_progress_threshold = PROGRESS_EVERY_N_ROWS

    # Server-side cursor keeps memory usage bounded for large datasets.
    with conn.cursor(name="staging_validation_cursor") as cur:

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

                process_record(
                    record,
                    allowed_sources,
                    report,
                )

                processed += 1

                if processed >= next_progress_threshold:
                    print_progress(
                        processed,
                        total_records,
                        start_time,
                    )

                    next_progress_threshold += (
                        PROGRESS_EVERY_N_ROWS
                    )

    print_progress(
        processed,
        total_records,
        start_time,
    )

    return report


# ============================================================================
# VALIDATION REPORT
# ============================================================================

def print_validation_report(
    report: Dict[str, Any],
) -> None:
    """Print the final staging-validation report."""

    print()
    print("=" * 80)
    print("STAGING VALIDATION REPORT")
    print("=" * 80)

    print(
        f"Total records analyzed        : "
        f"{report['total_records']}"
    )
    print(
        f"Valid dates                   : "
        f"{report['valid_dates']}"
    )
    print(
        f"Invalid dates                 : "
        f"{report['invalid_dates']}"
    )
    print(
        f"Mapped sources                : "
        f"{report['mapped_sources']}"
    )
    print(
        f"Unmapped sources              : "
        f"{report['unmapped_sources']}"
    )
    print(
        f"Valid sentiment values        : "
        f"{report['valid_sentiment']}"
    )
    print(
        f"Invalid sentiment values      : "
        f"{report['invalid_sentiment']}"
    )
    print(
        f"Valid is_duplicate values     : "
        f"{report['valid_is_duplicate']}"
    )
    print(
        f"Invalid is_duplicate values   : "
        f"{report['invalid_is_duplicate']}"
    )
    print(
        f"Blank titles                  : "
        f"{report['blank_title']}"
    )
    print(
        f"Blank bodies                  : "
        f"{report['blank_body']}"
    )
    print(
        f"English-like language values  : "
        f"{report['english_like']}"
    )
    print(
        f"Non-English/unknown values    : "
        f"{report['non_english_like']}"
    )

    examples = report["examples"]

    print_examples(
        "EXAMPLES - invalid dates",
        examples.get("invalid_dates", []),
    )

    print_examples(
        "EXAMPLES - unmapped sources",
        examples.get("unmapped_sources", []),
    )

    print_examples(
        "EXAMPLES - invalid sentiment",
        examples.get("invalid_sentiment", []),
    )

    print_examples(
        "EXAMPLES - invalid is_duplicate",
        examples.get("invalid_is_duplicate", []),
    )

    print_examples(
        "EXAMPLES - blank titles",
        examples.get("blank_title", []),
    )

    print_examples(
        "EXAMPLES - blank bodies",
        examples.get("blank_body", []),
    )

    if PRINT_VALUE_DISTRIBUTIONS:
        print_distribution(
            "LANGUAGE DISTRIBUTION",
            report["language_distribution"],
            max_items=MAX_DISTRIBUTION_ITEMS,
        )

        print_distribution(
            "SOURCE DISTRIBUTION",
            report["source_distribution"],
            max_items=MAX_DISTRIBUTION_ITEMS,
        )

        print_distribution(
            "IS_DUPLICATE DISTRIBUTION",
            report["duplicate_distribution"],
            max_items=MAX_DISTRIBUTION_ITEMS,
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run streaming validation on the staging table."""

    with get_connection() as conn:

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
            f"Records to validate: {total_in_staging:,}"
        )

        if SOURCE_FILE_FILTER is not None:
            print(
                f"Source file filter: {SOURCE_FILE_FILTER}"
            )

        allowed_sources = get_allowed_sources(conn)

        report = stream_validate_staging(
            conn=conn,
            allowed_sources=allowed_sources,
            source_file_filter=SOURCE_FILE_FILTER,
        )

        print_validation_report(report)


if __name__ == "__main__":
    main()
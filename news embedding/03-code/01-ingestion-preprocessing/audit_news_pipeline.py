#!/usr/bin/env python3

"""
Read-only audit of the Risk Emergence news pipeline.

The script compares the current PostgreSQL database state with the
expected project architecture and reports the implementation status
of the main pipeline components. No database modifications are made.
"""

from __future__ import annotations

import getpass
import os
import sys
from collections import OrderedDict

import psycopg2
from psycopg2 import sql


# ============================================================================
# CONFIGURATION
# ============================================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "research")
DB_USER = os.getenv("DB_USER", "postgres")

SCHEMA = "risk_emergence"

EXPECTED_TABLES = [
    "news_sources",
    "news_articles_staging",
    "news_articles",
    "firm_universe",
    "news_firm_mentions",
    "candidate_themes",
    "news_theme_scores",
    "news_embeddings",
    "news_indicators",
]

EXPECTED_VIEWS = [
    "v_valid_articles",
    "v_coverage_by_source",
    "v_theme_coverage",
]

EXPECTED_THEMES = [
    "ai_risk",
    "climate_risk",
    "cyber_risk",
    "pandemic_risk",
    "supply_chain_risk",
]

EXPECTED_NEWS_ARTICLE_COLUMNS = [
    "article_id",
    "publication_date",
    "title",
    "body",
    "source_raw",
    "source_id",
    "sentiment_original",
    "word_count",
    "char_count",
    "is_english",
    "is_duplicate_verified",
    "is_excluded",
    "content_hash",
    "body_tsv",
    "title_tsv",
]

RECOMMENDED_ARTICLE_INDEXES = [
    "idx_articles_pubdate",
    "idx_articles_source_date",
    "idx_articles_body_tsv",
    "idx_articles_title_tsv",
    "idx_articles_content_hash",
    "idx_articles_valid",
    "idx_articles_batch",
]

THEMATIC_SCORE_COLUMN = "thematic scores"

LINE = "=" * 78
SUBLINE = "-" * 78


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def section(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def subsection(title: str) -> None:
    print()
    print(title)
    print(SUBLINE)


def ok(message: str) -> None:
    print(f"[OK]       {message}")


def warning(message: str) -> None:
    print(f"[WARNING]  {message}")


def missing(message: str) -> None:
    print(f"[MISSING]  {message}")


def info(message: str) -> None:
    print(f"[INFO]     {message}")


def not_done(message: str) -> None:
    print(f"[NOT DONE] {message}")


def format_number(value) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_password() -> str:
    """Read the database password from the environment or prompt securely."""

    password = os.getenv("DB_PASSWORD")

    if password:
        return password

    return getpass.getpass("PostgreSQL password: ")


def connect():
    """Open a read-only PostgreSQL connection."""

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=get_password(),
    )

    # Enforce read-only mode for the entire audit session.
    conn.set_session(
        readonly=True,
        autocommit=True,
    )

    return conn


# ============================================================================
# DATABASE UTILITIES
# ============================================================================

def schema_exists(cur, schema_name: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.schemata
            WHERE schema_name = %s
        )
        """,
        (schema_name,),
    )

    return bool(cur.fetchone()[0])


def relation_exists(
    cur,
    schema_name: str,
    relation_name: str,
    relation_types=None,
) -> bool:

    if relation_types is None:
        relation_types = (
            "BASE TABLE",
            "VIEW",
        )

    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
              AND table_type = ANY(%s)
        )
        """,
        (
            schema_name,
            relation_name,
            list(relation_types),
        ),
    )

    return bool(cur.fetchone()[0])


def table_exists(
    cur,
    schema_name: str,
    table_name: str,
) -> bool:

    return relation_exists(
        cur,
        schema_name,
        table_name,
        ("BASE TABLE",),
    )


def view_exists(
    cur,
    schema_name: str,
    view_name: str,
) -> bool:

    return relation_exists(
        cur,
        schema_name,
        view_name,
        ("VIEW",),
    )


def get_columns(
    cur,
    schema_name: str,
    relation_name: str,
):

    cur.execute(
        """
        SELECT
            column_name,
            data_type,
            udt_name,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
        """,
        (
            schema_name,
            relation_name,
        ),
    )

    return cur.fetchall()


def get_column_names(
    cur,
    schema_name: str,
    relation_name: str,
):

    return [
        row[0]
        for row in get_columns(
            cur,
            schema_name,
            relation_name,
        )
    ]


def count_rows(
    cur,
    schema_name: str,
    relation_name: str,
):

    query = sql.SQL(
        "SELECT COUNT(*) FROM {}.{}"
    ).format(
        sql.Identifier(schema_name),
        sql.Identifier(relation_name),
    )

    cur.execute(query)

    return cur.fetchone()[0]


def scalar(cur, query, params=None):
    cur.execute(
        query,
        params or (),
    )

    row = cur.fetchone()

    return None if row is None else row[0]


# ============================================================================
# SCHEMA AND DATABASE OBJECTS
# ============================================================================

def audit_schema_and_objects(cur, results) -> None:
    section("1. EXPECTED SCHEMA AND OBJECTS")

    if not schema_exists(cur, SCHEMA):
        missing(f"Schema {SCHEMA} not found.")
        results["schema"] = "MISSING"
        return

    ok(f"Schema {SCHEMA} found.")
    results["schema"] = "OK"

    subsection("Tables")

    for table in EXPECTED_TABLES:
        if table_exists(cur, SCHEMA, table):
            n = count_rows(cur, SCHEMA, table)

            ok(
                f"{table:<28} "
                f"rows = {format_number(n)}"
            )

            results[f"table:{table}"] = "OK"
        else:
            missing(table)
            results[f"table:{table}"] = "MISSING"

    subsection("Views")

    for view in EXPECTED_VIEWS:
        if not view_exists(cur, SCHEMA, view):
            missing(view)
            results[f"view:{view}"] = "MISSING"
            continue

        try:
            n = count_rows(cur, SCHEMA, view)

            ok(
                f"{view:<28} "
                f"rows = {format_number(n)}"
            )

            results[f"view:{view}"] = "OK"

        except Exception as exc:
            warning(
                f"{view} exists but cannot be queried: {exc}"
            )

            results[f"view:{view}"] = "BROKEN"


# ============================================================================
# STAGING
# ============================================================================

def audit_staging(cur, results) -> None:
    section("2. INGESTION / STAGING")

    table = "news_articles_staging"

    if not table_exists(cur, SCHEMA, table):
        missing(f"{table} not found.")
        results["staging"] = "MISSING"
        return

    n = count_rows(cur, SCHEMA, table)

    info(
        f"Rows currently in staging: "
        f"{format_number(n)}"
    )

    columns = get_column_names(
        cur,
        SCHEMA,
        table,
    )

    for trace_col in [
        "_raw_line_number",
        "_source_file",
    ]:
        if trace_col in columns:
            ok(
                f"Traceability field "
                f"{trace_col} found."
            )
        else:
            warning(
                f"Traceability field "
                f"{trace_col} not found."
            )

    if n == 0:
        ok(
            "Staging is empty, consistent with "
            "truncation after transfer."
        )
    else:
        warning(
            "Staging still contains records. "
            "Verify whether transfer to news_articles "
            "has already been completed."
        )

    results["staging"] = "CHECK"


# ============================================================================
# NEWS ARTICLES
# ============================================================================

def audit_news_articles(cur, results) -> None:
    section("3. NEWS_ARTICLES — CENTRAL ANALYTICAL TABLE")

    table = "news_articles"

    if not table_exists(cur, SCHEMA, table):
        missing(f"{table} not found.")
        results["news_articles"] = "MISSING"
        return

    total = count_rows(cur, SCHEMA, table)

    info(
        f"Articles in main table: "
        f"{format_number(total)}"
    )

    columns = get_column_names(
        cur,
        SCHEMA,
        table,
    )

    subsection("Expected columns")

    missing_columns = []

    for column in EXPECTED_NEWS_ARTICLE_COLUMNS:
        if column in columns:
            ok(column)
        else:
            missing(column)
            missing_columns.append(column)

    results["news_articles_columns"] = (
        "OK"
        if not missing_columns
        else f"MISSING {len(missing_columns)}"
    )

    if "publication_date" in columns:
        subsection("Temporal coverage")

        query = sql.SQL(
            """
            SELECT
                MIN(publication_date),
                MAX(publication_date),
                COUNT(*) FILTER (
                    WHERE publication_date IS NULL
                )
            FROM {}.{}
            """
        ).format(
            sql.Identifier(SCHEMA),
            sql.Identifier(table),
        )

        cur.execute(query)

        min_date, max_date, null_dates = cur.fetchone()

        info(f"Earliest date: {min_date}")
        info(f"Latest date:   {max_date}")
        info(
            f"Missing dates: "
            f"{format_number(null_dates)}"
        )

    subsection("Text integrity")

    if "body" in columns:
        body_null = scalar(
            cur,
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.{table}
            WHERE body IS NULL
               OR BTRIM(body) = ''
            """,
        )

        info(
            f"Missing/blank bodies: "
            f"{format_number(body_null)}"
        )

    if "title" in columns:
        title_null = scalar(
            cur,
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.{table}
            WHERE title IS NULL
               OR BTRIM(title) = ''
            """,
        )

        info(
            f"Missing/blank titles: "
            f"{format_number(title_null)}"
        )

    results["news_articles"] = "OK"


# ============================================================================
# QUALITY FLAGS
# ============================================================================

def audit_quality_flags(cur, results) -> None:
    section(
        "4. NORMALIZATION, LANGUAGE, "
        "DEDUPLICATION AND EXCLUSIONS"
    )

    table = "news_articles"

    if not table_exists(cur, SCHEMA, table):
        return

    columns = get_column_names(
        cur,
        SCHEMA,
        table,
    )

    subsection("Language")

    if "is_english" in columns:
        cur.execute(
            f"""
            SELECT
                is_english,
                COUNT(*)
            FROM {SCHEMA}.{table}
            GROUP BY is_english
            ORDER BY is_english NULLS FIRST
            """
        )

        for value, count in cur.fetchall():
            info(
                f"is_english={value}: "
                f"{format_number(count)}"
            )

        results["language_check"] = "OK"
    else:
        missing("is_english not found.")
        results["language_check"] = "MISSING"

    subsection("Deduplication")

    if "is_duplicate_verified" in columns:
        cur.execute(
            f"""
            SELECT
                is_duplicate_verified,
                COUNT(*)
            FROM {SCHEMA}.{table}
            GROUP BY is_duplicate_verified
            ORDER BY is_duplicate_verified
            """
        )

        for value, count in cur.fetchall():
            info(
                f"is_duplicate_verified={value}: "
                f"{format_number(count)}"
            )

        results["minhash_or_verified_dedup"] = "PRESENT"
    else:
        missing("is_duplicate_verified not found.")
        results["minhash_or_verified_dedup"] = "MISSING"

    if "content_hash" in columns:
        null_hashes = scalar(
            cur,
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.{table}
            WHERE content_hash IS NULL
            """,
        )

        duplicate_hash_groups = scalar(
            cur,
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT content_hash
                FROM {SCHEMA}.{table}
                WHERE content_hash IS NOT NULL
                GROUP BY content_hash
                HAVING COUNT(*) > 1
            ) x
            """,
        )

        ok("content_hash found.")

        info(
            f"Missing hashes: "
            f"{format_number(null_hashes)}"
        )
        info(
            f"Distinct duplicated hashes: "
            f"{format_number(duplicate_hash_groups)}"
        )

        results["exact_dedup"] = "OK"
    else:
        missing("content_hash not found.")
        results["exact_dedup"] = "MISSING"

    subsection("Exclusions")

    if "is_excluded" in columns:
        cur.execute(
            f"""
            SELECT
                is_excluded,
                COUNT(*)
            FROM {SCHEMA}.{table}
            GROUP BY is_excluded
            ORDER BY is_excluded
            """
        )

        for value, count in cur.fetchall():
            info(
                f"is_excluded={value}: "
                f"{format_number(count)}"
            )

        results["exclusions"] = "OK"
    else:
        missing("is_excluded not found.")
        results["exclusions"] = "MISSING"


# ============================================================================
# VALID ARTICLES
# ============================================================================

def audit_valid_articles(cur, results) -> None:
    section("5. ANALYTICAL CORPUS — v_valid_articles")

    view = "v_valid_articles"

    if not view_exists(cur, SCHEMA, view):
        missing(f"{view} not found.")
        results["valid_articles"] = "MISSING"
        return

    valid_count = count_rows(
        cur,
        SCHEMA,
        view,
    )

    ok(f"{view} found.")

    info(
        f"Analytically valid articles: "
        f"{format_number(valid_count)}"
    )

    if table_exists(cur, SCHEMA, "news_articles"):
        total = count_rows(
            cur,
            SCHEMA,
            "news_articles",
        )

        if total:
            share = valid_count / total * 100

            info(
                f"Valid share of total corpus: "
                f"{share:.2f}%"
            )

    results["valid_articles"] = "OK"


# ============================================================================
# SOURCES AND COVERAGE
# ============================================================================

def audit_sources(cur, results) -> None:
    section("6. NEWS SOURCES AND COVERAGE")

    table = "news_sources"

    if table_exists(cur, SCHEMA, table):
        rows = count_rows(
            cur,
            SCHEMA,
            table,
        )

        info(
            f"news_sources: "
            f"{format_number(rows)} sources."
        )

        try:
            cur.execute(
                f"""
                SELECT *
                FROM {SCHEMA}.{table}
                ORDER BY 1
                """
            )

            columns = [
                desc[0]
                for desc in cur.description
            ]

            for record in cur.fetchall():
                info(
                    str(dict(zip(columns, record)))
                )

        except Exception as exc:
            warning(
                f"Unable to display sources: {exc}"
            )
    else:
        missing(table)

    subsection("v_coverage_by_source")

    view = "v_coverage_by_source"

    if not view_exists(cur, SCHEMA, view):
        missing(view)
        results["coverage_source"] = "MISSING"
        return

    ok(f"{view} found.")

    try:
        cur.execute(
            f"""
            SELECT *
            FROM {SCHEMA}.{view}
            ORDER BY 1, 2
            LIMIT 20
            """
        )

        rows = cur.fetchall()

        info(
            f"Successfully read the first "
            f"{len(rows)} rows."
        )

        results["coverage_source"] = "OK"

    except Exception as exc:
        warning(str(exc))
        results["coverage_source"] = "BROKEN"


# ============================================================================
# CANDIDATE THEMES
# ============================================================================

def audit_candidate_themes(cur, results) -> None:
    section("7. CANDIDATE THEMES")

    table = "candidate_themes"

    if not table_exists(cur, SCHEMA, table):
        missing(table)
        results["candidate_themes"] = "MISSING"
        return

    columns = get_column_names(
        cur,
        SCHEMA,
        table,
    )

    info("Columns: " + ", ".join(columns))

    candidate_columns = [
        "theme_name",
        "name",
        "theme_key",
        "code",
    ]

    theme_column = next(
        (
            candidate
            for candidate in candidate_columns
            if candidate in columns
        ),
        None,
    )

    if theme_column is None:
        warning(
            "Unable to identify the theme-name column "
            "automatically."
        )
        results["candidate_themes"] = "CHECK"
        return

    query = sql.SQL(
        """
        SELECT {}
        FROM {}.{}
        """
    ).format(
        sql.Identifier(theme_column),
        sql.Identifier(SCHEMA),
        sql.Identifier(table),
    )

    cur.execute(query)

    present = {
        str(row[0])
        for row in cur.fetchall()
    }

    subsection("Expected themes")

    all_present = True

    for theme in EXPECTED_THEMES:
        if theme in present:
            ok(theme)
        else:
            missing(theme)
            all_present = False

    if "seed_words" in columns:
        ok("seed_words found.")
    else:
        warning("seed_words not found.")

    results["candidate_themes"] = (
        "OK"
        if all_present
        else "INCOMPLETE"
    )


# ============================================================================
# FIRM UNIVERSE AND NER
# ============================================================================

def audit_firm_ner(cur, results) -> None:
    section("8. FIRM UNIVERSE AND NER")

    if table_exists(cur, SCHEMA, "firm_universe"):
        firms = count_rows(
            cur,
            SCHEMA,
            "firm_universe",
        )

        info(
            f"firm_universe: "
            f"{format_number(firms)} rows."
        )

        if firms > 0:
            ok("Firm universe is populated.")
        else:
            warning(
                "firm_universe exists but is empty."
            )
    else:
        missing("firm_universe")

    if not table_exists(
        cur,
        SCHEMA,
        "news_firm_mentions",
    ):
        missing("news_firm_mentions")
        results["firm_ner"] = "MISSING"
        return

    mentions = count_rows(
        cur,
        SCHEMA,
        "news_firm_mentions",
    )

    info(
        f"news_firm_mentions: "
        f"{format_number(mentions)} rows."
    )

    if mentions > 0:
        ok(
            "Firm-mention extraction has produced results."
        )
        results["firm_ner"] = "PRESENT"
    else:
        not_done(
            "Table exists but is empty; firm NER "
            "has not produced results yet."
        )
        results["firm_ner"] = "NOT DONE"


# ============================================================================
# THEME SCORES
# ============================================================================

def audit_theme_scores(cur, results) -> None:
    section("9. NEWS THEME SCORES")

    table = "news_theme_scores"

    if not table_exists(cur, SCHEMA, table):
        missing(table)
        results["theme_scores"] = "MISSING"
        return

    n = count_rows(cur, SCHEMA, table)

    info(f"Rows: {format_number(n)}")

    columns = get_column_names(
        cur,
        SCHEMA,
        table,
    )

    info("Columns: " + ", ".join(columns))

    expected_columns = [
        "article_id",
        "theme_id",
        THEMATIC_SCORE_COLUMN,
        "method",
    ]

    for expected in expected_columns:
        if expected in columns:
            ok(expected)
        else:
            warning(
                f"Expected column not found: {expected}"
            )

    if n == 0:
        not_done(
            "Table exists but contains no theme scores."
        )
        results["theme_scores"] = "NOT DONE"
        return

    if THEMATIC_SCORE_COLUMN in columns:
        score_identifier = sql.Identifier(
            THEMATIC_SCORE_COLUMN
        )

        query = sql.SQL(
            """
            SELECT
                MIN({score}),
                MAX({score}),
                AVG({score}),
                COUNT(*) FILTER (
                    WHERE {score} < 0
                       OR {score} > 1
                )
            FROM {schema}.{table}
            """
        ).format(
            score=score_identifier,
            schema=sql.Identifier(SCHEMA),
            table=sql.Identifier(table),
        )

        cur.execute(query)

        min_score, max_score, avg_score, bad = (
            cur.fetchone()
        )

        info(f"Minimum score: {min_score}")
        info(f"Maximum score: {max_score}")
        info(f"Average score: {avg_score}")
        info(
            f"Scores outside [0,1]: "
            f"{format_number(bad)}"
        )

    if "method" in columns:
        cur.execute(
            f"""
            SELECT
                method,
                COUNT(*)
            FROM {SCHEMA}.{table}
            GROUP BY method
            ORDER BY COUNT(*) DESC
            """
        )

        subsection("Methods")

        for method, count in cur.fetchall():
            info(
                f"{method}: "
                f"{format_number(count)}"
            )

    results["theme_scores"] = "PRESENT"


# ============================================================================
# EMBEDDINGS
# ============================================================================

def audit_embeddings(cur, results) -> None:
    section("10. NEWS EMBEDDINGS")

    table = "news_embeddings"

    if not table_exists(cur, SCHEMA, table):
        missing(table)
        results["embeddings"] = "MISSING"
        return

    n = count_rows(cur, SCHEMA, table)

    info(
        f"Embedding rows: "
        f"{format_number(n)}"
    )

    columns = get_column_names(
        cur,
        SCHEMA,
        table,
    )

    info("Columns: " + ", ".join(columns))

    if n == 0:
        not_done(
            "Embeddings table exists but is empty."
        )
        results["embeddings"] = "NOT DONE"
        return

    ok("Embeddings found.")

    if "model_name" in columns:
        cur.execute(
            f"""
            SELECT
                model_name,
                COUNT(*)
            FROM {SCHEMA}.{table}
            GROUP BY model_name
            ORDER BY COUNT(*) DESC
            """
        )

        for model, count in cur.fetchall():
            info(
                f"{model}: "
                f"{format_number(count)}"
            )

    results["embeddings"] = "PRESENT"


# ============================================================================
# INDICATORS
# ============================================================================

def audit_indicators(cur, results) -> None:
    section("11. NEWS INDICATORS")

    table = "news_indicators"

    if not table_exists(cur, SCHEMA, table):
        missing(table)
        results["indicators"] = "MISSING"
        return

    n = count_rows(cur, SCHEMA, table)

    info(
        f"Indicator rows: "
        f"{format_number(n)}"
    )

    columns = get_column_names(
        cur,
        SCHEMA,
        table,
    )

    info("Columns: " + ", ".join(columns))

    if n == 0:
        not_done(
            "Indicators table exists but is empty."
        )
        results["indicators"] = "NOT DONE"
    else:
        ok("Aggregated indicators found.")
        results["indicators"] = "PRESENT"


# ============================================================================
# THEME COVERAGE
# ============================================================================

def audit_theme_coverage(cur, results) -> None:
    section("12. THEME COVERAGE / TIME SERIES")

    view = "v_theme_coverage"

    if not view_exists(cur, SCHEMA, view):
        missing(view)
        results["theme_coverage"] = "MISSING"
        return

    try:
        n = count_rows(cur, SCHEMA, view)

        info(
            f"Rows in {view}: "
            f"{format_number(n)}"
        )

        if n == 0:
            not_done(
                "View exists but does not yet produce "
                "theme time series."
            )
            results["theme_coverage"] = "NOT DONE"
        else:
            ok("Theme-coverage time series found.")
            results["theme_coverage"] = "PRESENT"

    except Exception as exc:
        warning(
            f"View exists but cannot be queried: {exc}"
        )
        results["theme_coverage"] = "BROKEN"


# ============================================================================
# INDEXES
# ============================================================================

def audit_indexes(cur, results) -> None:
    section("13. NEWS_ARTICLES INDEXES")

    if not table_exists(
        cur,
        SCHEMA,
        "news_articles",
    ):
        return

    cur.execute(
        """
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE schemaname = %s
          AND tablename = 'news_articles'
        ORDER BY indexname
        """,
        (SCHEMA,),
    )

    rows = cur.fetchall()

    if not rows:
        warning("No indexes found.")
        results["indexes"] = "MISSING"
        return

    for name, definition in rows:
        ok(name)
        print(f"           {definition}")

    index_names = {
        row[0]
        for row in rows
    }

    subsection("Recommended index names")

    for name in RECOMMENDED_ARTICLE_INDEXES:
        if name in index_names:
            ok(name)
        else:
            warning(
                f"{name} not found under "
                "the expected name."
            )

    results["indexes"] = "CHECKED"


# ============================================================================
# TRIGGERS
# ============================================================================

def audit_triggers(cur, results) -> None:
    section("14. TRIGGERS")

    cur.execute(
        """
        SELECT
            event_object_table,
            trigger_name,
            event_manipulation,
            action_timing
        FROM information_schema.triggers
        WHERE trigger_schema = %s
        ORDER BY
            event_object_table,
            trigger_name,
            event_manipulation
        """,
        (SCHEMA,),
    )

    rows = cur.fetchall()

    if not rows:
        warning("No triggers found in the schema.")
        results["triggers"] = "MISSING"
        return

    names = set()

    for table, trigger, event, timing in rows:
        info(
            f"{trigger} | {table} | "
            f"{timing} {event}"
        )
        names.add(trigger)

    if "trg_articles_tsv" in names:
        ok("trg_articles_tsv found.")
        results["tsvector_trigger"] = "OK"
    else:
        warning(
            "trg_articles_tsv not found under "
            "the expected name."
        )
        results["tsvector_trigger"] = "CHECK"


# ============================================================================
# STAGING VS MAIN TABLE
# ============================================================================

def audit_staging_vs_main(cur, results) -> None:
    section("15. STAGING VS NEWS_ARTICLES")

    if not (
        table_exists(
            cur,
            SCHEMA,
            "news_articles_staging",
        )
        and table_exists(
            cur,
            SCHEMA,
            "news_articles",
        )
    ):
        return

    staging = count_rows(
        cur,
        SCHEMA,
        "news_articles_staging",
    )

    main = count_rows(
        cur,
        SCHEMA,
        "news_articles",
    )

    info(
        f"Staging:       "
        f"{format_number(staging)}"
    )
    info(
        f"news_articles: "
        f"{format_number(main)}"
    )

    if main == 0 and staging > 0:
        warning(
            "Records appear to remain only in staging."
        )
        results["staging_to_main"] = "NOT DONE"
    elif main > 0:
        ok("news_articles contains records.")
        results["staging_to_main"] = "DONE"
    else:
        warning("Both tables are empty.")
        results["staging_to_main"] = "CHECK"


# ============================================================================
# PIPELINE SUMMARY
# ============================================================================

def print_pipeline_summary(results) -> None:
    section("16. PIPELINE STATUS SUMMARY")

    stages = OrderedDict([
        (
            "PHASE 1 — PostgreSQL schema",
            results.get("schema", "UNKNOWN"),
        ),
        (
            "PHASE 1 — Staging ingestion",
            results.get("staging", "UNKNOWN"),
        ),
        (
            "PHASE 1 — Staging → news_articles",
            results.get("staging_to_main", "UNKNOWN"),
        ),
        (
            "PHASE 1 — news_articles table",
            results.get("news_articles", "UNKNOWN"),
        ),
        (
            "PHASE 1 — Language check",
            results.get("language_check", "UNKNOWN"),
        ),
        (
            "PHASE 1 — Exact dedup / hash",
            results.get("exact_dedup", "UNKNOWN"),
        ),
        (
            "PHASE 1 — Verified / MinHash dedup",
            results.get(
                "minhash_or_verified_dedup",
                "UNKNOWN",
            ),
        ),
        (
            "PHASE 1 — v_valid_articles",
            results.get("valid_articles", "UNKNOWN"),
        ),
        (
            "PHASE 1 — Coverage by source",
            results.get("coverage_source", "UNKNOWN"),
        ),
        (
            "PHASE 1 — Firm NER",
            results.get("firm_ner", "UNKNOWN"),
        ),
        (
            "PHASE 2 — Candidate themes",
            results.get("candidate_themes", "UNKNOWN"),
        ),
        (
            "PHASE 2 — Theme scores [0,1]",
            results.get("theme_scores", "UNKNOWN"),
        ),
        (
            "PHASE 2 — Embeddings",
            results.get("embeddings", "UNKNOWN"),
        ),
        (
            "PHASE 2 — Theme coverage",
            results.get("theme_coverage", "UNKNOWN"),
        ),
        (
            "PHASE 3 — News indicators",
            results.get("indicators", "UNKNOWN"),
        ),
    ])

    width = max(
        len(name)
        for name in stages
    )

    for name, status in stages.items():
        print(
            f"{name:<{width}} : {status}"
        )

    print()
    print(SUBLINE)
    print(
        "NOTE: UNKNOWN does not necessarily mean that "
        "a stage is missing; it means that its status "
        "could not be determined automatically."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run the complete read-only pipeline audit."""

    print(LINE)
    print("NEWS PIPELINE AUDIT — RISK EMERGENCE PROJECT")
    print(LINE)
    print()
    print(f"Database: {DB_NAME}")
    print(f"Schema:   {SCHEMA}")
    print()
    print("Mode: READ ONLY")
    print("No database modifications will be performed.")

    conn = None
    results = {}

    try:
        conn = connect()

        with conn.cursor() as cur:
            print()
            ok("PostgreSQL connection established.")

            audit_schema_and_objects(cur, results)

            # Remaining checks require the project schema.
            if results.get("schema") != "OK":
                print_pipeline_summary(results)
                return

            audit_staging(cur, results)
            audit_news_articles(cur, results)
            audit_quality_flags(cur, results)
            audit_valid_articles(cur, results)
            audit_sources(cur, results)
            audit_candidate_themes(cur, results)
            audit_firm_ner(cur, results)
            audit_theme_scores(cur, results)
            audit_embeddings(cur, results)
            audit_indicators(cur, results)
            audit_theme_coverage(cur, results)
            audit_indexes(cur, results)
            audit_triggers(cur, results)
            audit_staging_vs_main(cur, results)

            print_pipeline_summary(results)

        print()
        print(LINE)
        print("AUDIT COMPLETED")
        print(LINE)
        print()
        print("The database was not modified.")

    except Exception as exc:
        print()
        print("ERROR DURING AUDIT:")
        print(str(exc))
        raise

    finally:
        if conn is not None:
            conn.close()
            print()
            print("Database connection closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
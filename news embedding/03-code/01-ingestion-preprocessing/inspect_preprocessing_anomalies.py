#!/usr/bin/env python3

"""
Read-only audit of preprocessing anomalies in the Risk Emergence news pipeline.

The script inspects duplicated content, short or repeated bodies,
potential encoding issues, staging-field distributions, and possible
causes of records rejected during the staging-to-main transfer.

No database modifications are made.
"""

from __future__ import annotations

import getpass
import os
import sys

import psycopg2


# ============================================================================
# CONFIGURATION
# ============================================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "research")
DB_USER = os.getenv("DB_USER", "postgres")

SCHEMA = "risk_emergence"
STAGING = "news_articles_staging"
ARTICLES = "news_articles"

LINE = "=" * 80
SUBLINE = "-" * 80


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


def info(message: str) -> None:
    print(f"[INFO] {message}")


def warning(message: str) -> None:
    print(f"[WARNING] {message}")


def ok(message: str) -> None:
    print(f"[OK] {message}")


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


def scalar(cur, query, params=None):
    cur.execute(query, params or ())
    row = cur.fetchone()

    return row[0] if row else None


# ============================================================================
# DUPLICATED CONTENT
# ============================================================================

def inspect_top_duplicate_bodies(cur) -> None:
    """Inspect article bodies associated with the most frequent hashes."""

    section("1. BODIES ASSOCIATED WITH THE MOST DUPLICATED HASHES")

    cur.execute(
        f"""
        WITH top_hashes AS (
            SELECT
                content_hash,
                COUNT(*) AS n
            FROM {SCHEMA}.{ARTICLES}
            WHERE content_hash IS NOT NULL
            GROUP BY content_hash
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 20
        )
        SELECT
            encode(a.content_hash, 'hex') AS hash_hex,
            th.n,
            MIN(a.publication_date) AS min_date,
            MAX(a.publication_date) AS max_date,
            MIN(LENGTH(a.body)) AS body_length,
            MIN(a.body) AS body_example
        FROM {SCHEMA}.{ARTICLES} a
        JOIN top_hashes th
          ON a.content_hash = th.content_hash
        GROUP BY
            a.content_hash,
            th.n
        ORDER BY th.n DESC
        """
    )

    rows = cur.fetchall()

    for index, row in enumerate(rows, start=1):
        (
            hash_hex,
            count,
            min_date,
            max_date,
            body_length,
            body_example,
        ) = row

        print()
        print(f"[{index}] HASH {hash_hex[:20]}...")
        print(
            f"    n={format_number(count)} | "
            f"dates={min_date} -> {max_date} | "
            f"body_length={format_number(body_length)}"
        )

        cleaned = (
            body_example
            .replace("\n", " ")
            .replace("\r", " ")
            if body_example
            else ""
        )

        print("    BODY:")
        print("    " + cleaned[:700])


# ============================================================================
# SHORT AND REPEATED BODIES
# ============================================================================

def inspect_short_bodies(cur) -> None:
    """Inspect unusually short and frequently repeated article bodies."""

    section("2. SHORT ARTICLE BODIES")

    thresholds = [
        50,
        100,
        200,
        300,
        500,
    ]

    for threshold in thresholds:
        count = scalar(
            cur,
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.{ARTICLES}
            WHERE LENGTH(body) <= %s
            """,
            (threshold,),
        )

        info(
            f"Bodies <= {threshold} characters: "
            f"{format_number(count)}"
        )

    subsection("Most frequent short bodies")

    cur.execute(
        f"""
        SELECT
            LEFT(body, 300) AS body_preview,
            LENGTH(body) AS body_length,
            COUNT(*) AS n,
            COUNT(DISTINCT title) AS distinct_titles,
            MIN(publication_date),
            MAX(publication_date)
        FROM {SCHEMA}.{ARTICLES}
        WHERE LENGTH(body) <= 500
        GROUP BY
            body,
            LENGTH(body)
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC
        LIMIT 30
        """
    )

    for (
        body_preview,
        body_length,
        count,
        distinct_titles,
        min_date,
        max_date,
    ) in cur.fetchall():

        print()

        info(
            f"n={format_number(count)} | "
            f"distinct_titles={format_number(distinct_titles)} | "
            f"len={format_number(body_length)} | "
            f"{min_date} -> {max_date}"
        )

        preview = (
            body_preview
            .replace("\n", " ")
            .replace("\r", " ")
            if body_preview
            else ""
        )

        print("      " + preview)


# ============================================================================
# SAME BODY WITH DIFFERENT TITLES
# ============================================================================

def inspect_same_body_different_titles(cur) -> None:
    """Inspect identical article bodies associated with different titles."""

    section("3. IDENTICAL BODIES WITH DIFFERENT TITLES")

    cur.execute(
        f"""
        SELECT
            encode(content_hash, 'hex') AS hash_hex,
            COUNT(*) AS n_articles,
            COUNT(DISTINCT title) AS n_titles,
            MIN(LENGTH(body)) AS body_length
        FROM {SCHEMA}.{ARTICLES}
        WHERE content_hash IS NOT NULL
        GROUP BY content_hash
        HAVING
            COUNT(*) > 1
            AND COUNT(DISTINCT title) > 1
        ORDER BY COUNT(*) DESC
        LIMIT 20
        """
    )

    rows = cur.fetchall()

    for (
        hash_hex,
        n_articles,
        n_titles,
        body_length,
    ) in rows:

        print()

        info(
            f"HASH {hash_hex[:20]}... | "
            f"articles={format_number(n_articles)} | "
            f"titles={format_number(n_titles)} | "
            f"body_len={format_number(body_length)}"
        )

        cur.execute(
            f"""
            SELECT
                article_id,
                publication_date,
                source_raw,
                LEFT(title, 180)
            FROM {SCHEMA}.{ARTICLES}
            WHERE encode(content_hash, 'hex') = %s
            ORDER BY article_id
            LIMIT 8
            """,
            (hash_hex,),
        )

        for (
            article_id,
            publication_date,
            source,
            title,
        ) in cur.fetchall():

            print(
                f"    {article_id} | "
                f"{publication_date} | "
                f"{source} | "
                f"{title}"
            )


# ============================================================================
# ENCODING
# ============================================================================

def inspect_encoding(cur) -> None:
    """Search for common mojibake patterns in titles and article bodies."""

    section("4. POTENTIAL ENCODING ISSUES")

    patterns = [
        "â€",
        "â€™",
        "â€œ",
        "Ã©",
        "Ã",
    ]

    for pattern in patterns:
        subsection(f"Pattern: {pattern!r}")

        cur.execute(
            f"""
            SELECT
                article_id,
                publication_date,
                source_raw,
                LEFT(title, 180),
                LEFT(body, 500)
            FROM {SCHEMA}.{ARTICLES}
            WHERE
                title LIKE %s
                OR body LIKE %s
            ORDER BY article_id
            LIMIT 15
            """,
            (
                f"%{pattern}%",
                f"%{pattern}%",
            ),
        )

        rows = cur.fetchall()

        if not rows:
            info("No examples found.")
            continue

        for (
            article_id,
            publication_date,
            source,
            title,
            body,
        ) in rows:

            print()
            print(
                f"{article_id} | "
                f"{publication_date} | "
                f"{source}"
            )

            print(f"  TITLE: {title}")

            body_clean = (
                body.replace("\n", " ")
                .replace("\r", " ")
                if body
                else ""
            )

            print(f"  BODY:  {body_clean}")


# ============================================================================
# STAGING LANGUAGE
# ============================================================================

def inspect_staging_language(cur) -> None:
    """Inspect the distribution of language values in staging."""

    section("5. LANGUAGE DISTRIBUTION IN STAGING")

    cur.execute(
        f"""
        SELECT
            language,
            COUNT(*) AS n
        FROM {SCHEMA}.{STAGING}
        GROUP BY language
        ORDER BY COUNT(*) DESC
        """
    )

    for language, count in cur.fetchall():
        info(
            f"language={language!r}: "
            f"{format_number(count)}"
        )


# ============================================================================
# STAGING DATA QUALITY
# ============================================================================

def inspect_staging_quality(cur) -> None:
    """Count missing values in key staging fields."""

    section("6. STAGING DATA QUALITY")

    checks = {
        "Blank/null title":
            "title IS NULL OR BTRIM(title) = ''",

        "Blank/null body":
            "body IS NULL OR BTRIM(body) = ''",

        "Blank/null publication_date":
            "publication_date IS NULL OR BTRIM(publication_date) = ''",

        "Blank/null source":
            "source IS NULL OR BTRIM(source) = ''",

        "Blank/null language":
            "language IS NULL OR BTRIM(language) = ''",

        "Blank/null is_duplicate":
            "is_duplicate IS NULL OR BTRIM(is_duplicate) = ''",
    }

    for label, condition in checks.items():
        count = scalar(
            cur,
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.{STAGING}
            WHERE {condition}
            """,
        )

        info(
            f"{label}: "
            f"{format_number(count)}"
        )


# ============================================================================
# STAGING DATES
# ============================================================================

def inspect_staging_dates(cur) -> None:
    """Inspect publication-date values that do not match YYYY-MM-DD."""

    section("7. PUBLICATION DATES IN STAGING")

    invalid_format = scalar(
        cur,
        f"""
        SELECT COUNT(*)
        FROM {SCHEMA}.{STAGING}
        WHERE publication_date IS NOT NULL
          AND BTRIM(publication_date) <> ''
          AND publication_date !~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$'
        """,
    )

    info(
        "Dates not matching YYYY-MM-DD: "
        f"{format_number(invalid_format)}"
    )

    if invalid_format:
        cur.execute(
            f"""
            SELECT
                publication_date,
                COUNT(*)
            FROM {SCHEMA}.{STAGING}
            WHERE publication_date IS NOT NULL
              AND BTRIM(publication_date) <> ''
              AND publication_date !~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$'
            GROUP BY publication_date
            ORDER BY COUNT(*) DESC
            LIMIT 30
            """
        )

        for value, count in cur.fetchall():
            info(
                f"{value!r}: "
                f"{format_number(count)}"
            )


# ============================================================================
# STAGING SOURCES
# ============================================================================

def inspect_staging_sources(cur) -> None:
    """Inspect source-value frequencies in staging."""

    section("8. SOURCE DISTRIBUTION IN STAGING")

    cur.execute(
        f"""
        SELECT
            source,
            COUNT(*)
        FROM {SCHEMA}.{STAGING}
        GROUP BY source
        ORDER BY COUNT(*) DESC
        """
    )

    for source, count in cur.fetchall():
        info(
            f"{source!r}: "
            f"{format_number(count)}"
        )


# ============================================================================
# STAGING DUPLICATE FLAG
# ============================================================================

def inspect_staging_duplicate_flag(cur) -> None:
    """Inspect is_duplicate values in staging."""

    section("9. IS_DUPLICATE DISTRIBUTION IN STAGING")

    cur.execute(
        f"""
        SELECT
            is_duplicate,
            COUNT(*)
        FROM {SCHEMA}.{STAGING}
        GROUP BY is_duplicate
        ORDER BY COUNT(*) DESC
        """
    )

    for value, count in cur.fetchall():
        info(
            f"is_duplicate={value!r}: "
            f"{format_number(count)}"
        )


# ============================================================================
# POSSIBLE TRANSFER LOSSES
# ============================================================================

def inspect_possible_transfer_losses(cur) -> None:
    """Inspect anomalies that may explain staging-to-main rejections."""

    section("10. POSSIBLE CAUSES OF RECORDS NOT TRANSFERRED")

    tests = [
        (
            "Blank/null title",
            "title IS NULL OR BTRIM(title) = ''",
        ),
        (
            "Blank/null body",
            "body IS NULL OR BTRIM(body) = ''",
        ),
        (
            "Blank/null publication_date",
            "publication_date IS NULL OR BTRIM(publication_date) = ''",
        ),
        (
            "Blank/null source",
            "source IS NULL OR BTRIM(source) = ''",
        ),
        (
            "Blank/null language",
            "language IS NULL OR BTRIM(language) = ''",
        ),
    ]

    total_flagged = 0

    for label, condition in tests:
        count = scalar(
            cur,
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.{STAGING}
            WHERE {condition}
            """,
        )

        info(
            f"{label}: "
            f"{format_number(count)}"
        )

        total_flagged += count

    print()

    info(
        "Simple sum of anomalies above "
        "(categories may overlap): "
        f"{format_number(total_flagged)}"
    )

    # Without a shared source-level identifier, rejected staging rows
    # cannot be matched directly to records in news_articles.
    warning(
        "Without _source_file/_raw_line_number in news_articles, "
        "rejected staging rows cannot be identified directly. "
        "These checks only assess whether invalid records could "
        "explain the staging-to-main difference."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run the read-only preprocessing anomaly audit."""

    print(LINE)
    print("PREPROCESSING ANOMALIES AUDIT")
    print(LINE)
    print()
    print(f"Database: {DB_NAME}")
    print(f"Schema:   {SCHEMA}")
    print()
    print("Mode: READ ONLY")
    print("No database modifications will be performed.")

    conn = None

    try:
        conn = connect()

        with conn.cursor() as cur:
            print()
            ok("PostgreSQL connection established.")

            inspect_top_duplicate_bodies(cur)
            inspect_short_bodies(cur)
            inspect_same_body_different_titles(cur)
            inspect_encoding(cur)
            inspect_staging_language(cur)
            inspect_staging_quality(cur)
            inspect_staging_dates(cur)
            inspect_staging_sources(cur)
            inspect_staging_duplicate_flag(cur)
            inspect_possible_transfer_losses(cur)

        section("AUDIT COMPLETED")
        print("The database was not modified.")

    finally:
        if conn is not None:
            conn.close()
            print()
            print("Database connection closed.")


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
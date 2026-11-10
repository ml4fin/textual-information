#!/usr/bin/env python3

"""
Phase 2A — Keyword baseline for candidate risk themes.

For each active theme, the script builds a PostgreSQL full-text search query
from its seed words, identifies matching articles, computes ts_rank scores,
normalizes them using the 99th percentile of the matched-score distribution,
and stores one score for each article-theme pair.

Non-matching articles receive a score of zero.

This is a lexical keyword baseline, not the weakly supervised classifier.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_batch


# =============================================================================
# CONFIGURATION
# =============================================================================

THEME_NAME_FILTER: Optional[str] = None

COMMIT_FINAL_INSERTS = True

NORMALIZATION_PERCENTILE = 0.99

FETCH_SIZE = 10000
UPSERT_BATCH_SIZE = 2000
PROGRESS_EVERY_N_ROWS = 50000

TS_CONFIG = "english"

METHOD_NAME = "keyword"
MODEL_VERSION = "seedwords_tsrank_v1"


# =============================================================================
# DATABASE
# =============================================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "research"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

SCHEMA_NAME = "risk_emergence"

THEMES_TABLE = "candidate_themes"
VALID_ARTICLES_VIEW = "v_valid_articles"
SCORES_TABLE = "news_theme_scores"

SCORE_COLUMN = '"thematic scores"'


# =============================================================================
# CONNECTION
# =============================================================================

def get_connection():
    """Open a PostgreSQL connection."""

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "Database password not configured. "
            "Set the DB_PASSWORD environment variable."
        )

    return psycopg2.connect(**DB_CONFIG)


# =============================================================================
# DATABASE VALIDATION
# =============================================================================

def check_database_structure(conn) -> None:
    """Validate the database objects required by the keyword baseline."""

    required_objects = {
        THEMES_TABLE,
        VALID_ARTICLES_VIEW,
        SCORES_TABLE,
    }

    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = %s
      AND table_name IN (%s, %s, %s);
    """

    with conn.cursor() as cur:
        cur.execute(
            query,
            (
                SCHEMA_NAME,
                THEMES_TABLE,
                VALID_ARTICLES_VIEW,
                SCORES_TABLE,
            ),
        )

        found = {
            row[0]
            for row in cur.fetchall()
        }

    missing = required_objects - found

    if missing:
        raise RuntimeError(
            f"Missing database objects in schema "
            f"{SCHEMA_NAME}: {sorted(missing)}"
        )

    query_body_tsv = """
    SELECT data_type
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
      AND column_name = 'body_tsv';
    """

    with conn.cursor() as cur:
        cur.execute(
            query_body_tsv,
            (
                SCHEMA_NAME,
                VALID_ARTICLES_VIEW,
            ),
        )

        result = cur.fetchone()

    if result is None:
        raise RuntimeError(
            f"{SCHEMA_NAME}.{VALID_ARTICLES_VIEW} "
            f"does not contain body_tsv."
        )

    if result[0] != "tsvector":
        raise RuntimeError(
            f"body_tsv has type {result[0]}, "
            f"expected tsvector."
        )

    print(
        f"[OK] {SCHEMA_NAME}.{VALID_ARTICLES_VIEW}.body_tsv "
        f"is present and has type tsvector."
    )


# =============================================================================
# CANDIDATE THEMES
# =============================================================================

def get_active_themes(
    conn,
    theme_name_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return active candidate themes and their seed words."""

    query = f"""
    SELECT
        theme_id,
        theme_name,
        theme_label,
        seed_words
    FROM {SCHEMA_NAME}.{THEMES_TABLE}
    WHERE is_active = TRUE
    """

    params: Tuple = ()

    if theme_name_filter is not None:
        query += """
        AND theme_name = %s
        """
        params = (theme_name_filter,)

    query += """
    ORDER BY theme_id;
    """

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    themes: List[Dict[str, Any]] = []

    for row in rows:
        themes.append(
            {
                "theme_id": row[0],
                "theme_name": row[1],
                "theme_label": row[2],
                "seed_words": row[3] or [],
            }
        )

    return themes


# =============================================================================
# FULL-TEXT SEARCH QUERY
# =============================================================================

def build_theme_tsquery_sql(
    seed_words: List[str],
) -> Tuple[str, List[str]]:
    """
    Build a full-text search query from a theme's seed words.

    Single-word seeds use plainto_tsquery, while multi-word seeds use
    phraseto_tsquery. All seeds belonging to the same theme are joined by OR.
    """

    expressions: List[str] = []
    params: List[str] = []

    for seed in seed_words:
        if seed is None:
            continue

        seed = str(seed).strip()

        if not seed:
            continue

        if len(seed.split()) > 1:
            expressions.append(
                "phraseto_tsquery(%s, %s)"
            )
        else:
            expressions.append(
                "plainto_tsquery(%s, %s)"
            )

        params.extend(
            [
                TS_CONFIG,
                seed,
            ]
        )

    if not expressions:
        return "", []

    query_sql = (
        "("
        + " || ".join(expressions)
        + ")"
    )

    return query_sql, params


# =============================================================================
# MATCH STATISTICS
# =============================================================================

def count_theme_matches(
    conn,
    query_sql: str,
    query_params: List[str],
) -> int:
    """
    Count articles satisfying the full-text search predicate.

    Matching is defined by the PostgreSQL @@ operator. ts_rank is used only
    to measure relevance after an article has been classified as a match.
    """

    query = f"""
    SELECT COUNT(*)
    FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW}
    WHERE body_tsv @@ {query_sql};
    """

    with conn.cursor() as cur:
        cur.execute(
            query,
            query_params,
        )

        return int(
            cur.fetchone()[0]
        )


# =============================================================================
# SCORE NORMALIZATION
# =============================================================================

def compute_normalization_cap(
    conn,
    query_sql: str,
    query_params: List[str],
) -> Optional[float]:
    """
    Compute the normalization cap from the matched articles only.

    The cap is the configured percentile of the ts_rank distribution.
    Restricting the calculation to actual FTS matches prevents zero-rank
    non-matches from affecting the normalization distribution.
    """

    query = f"""
    SELECT
        percentile_cont(%s)
        WITHIN GROUP (
            ORDER BY ts_rank(
                body_tsv,
                {query_sql}
            )
        )
    FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW}
    WHERE body_tsv @@ {query_sql};
    """

    # The dynamic tsquery occurs twice in the SQL expression.
    params = (
        [NORMALIZATION_PERCENTILE]
        + query_params
        + query_params
    )

    with conn.cursor() as cur:
        cur.execute(
            query,
            params,
        )

        result = cur.fetchone()

    if result is None:
        return None

    cap = result[0]

    if cap is None:
        return None

    cap = float(cap)

    if cap <= 0:
        return None

    return cap


# =============================================================================
# PROGRESS
# =============================================================================

def format_seconds(seconds: float) -> str:
    """Format seconds as hh:mm:ss."""

    if seconds < 0:
        return "n/a"

    total = int(seconds)

    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def print_progress(
    theme_name: str,
    processed: int,
    total: int,
    start_time: float,
) -> None:
    """Print progress, processing speed, elapsed time, and ETA."""

    elapsed = time.time() - start_time

    speed = (
        processed / elapsed
        if elapsed > 0
        else 0.0
    )

    percentage = (
        processed / total * 100.0
        if total > 0
        else 0.0
    )

    remaining = max(
        total - processed,
        0,
    )

    eta = (
        remaining / speed
        if speed > 0
        else -1
    )

    print(
        f"[{theme_name}] "
        f"{processed:,}/{total:,} "
        f"({percentage:6.2f}%) | "
        f"{speed:,.0f} articles/s | "
        f"elapsed={format_seconds(elapsed)} | "
        f"ETA={format_seconds(eta)}"
    )


# =============================================================================
# SCORE UPSERT
# =============================================================================

def upsert_scores(
    conn,
    rows: List[Tuple],
) -> None:
    """
    Insert or update article-theme scores.

    The operation assumes a unique constraint on
    (article_id, theme_id, method).
    """

    query = f"""
    INSERT INTO {SCHEMA_NAME}.{SCORES_TABLE}
    (
        article_id,
        theme_id,
        {SCORE_COLUMN},
        method,
        model_version
    )
    VALUES
    (
        %s,
        %s,
        %s,
        %s,
        %s
    )

    ON CONFLICT
    (
        article_id,
        theme_id,
        method
    )

    DO UPDATE SET
        {SCORE_COLUMN} = EXCLUDED.{SCORE_COLUMN},
        model_version = EXCLUDED.model_version,
        computed_at = NOW();
    """

    with conn.cursor() as cur:
        execute_batch(
            cur,
            query,
            rows,
            page_size=UPSERT_BATCH_SIZE,
        )


# =============================================================================
# THEME SCORING
# =============================================================================

def score_theme(
    conn,
    theme: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute keyword-baseline scores for one candidate theme."""

    theme_id = theme["theme_id"]
    theme_name = theme["theme_name"]
    theme_label = theme["theme_label"]
    seeds = theme["seed_words"]

    print()
    print("=" * 100)
    print(f"THEME: {theme_name}")
    print(f"LABEL: {theme_label}")
    print(f"SEEDS: {seeds}")
    print("=" * 100)

    report = {
        "theme_name": theme_name,
        "total_records": 0,
        "matched_records": 0,
        "scored_records": 0,
        "normalization_cap": None,
    }

    query_sql, query_params = (
        build_theme_tsquery_sql(
            seeds
        )
    )

    if not query_sql:
        print(
            f"[SKIP] No valid seeds for "
            f"{theme_name}."
        )
        return report

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW};
            """
        )

        total_records = int(
            cur.fetchone()[0]
        )

    report["total_records"] = (
        total_records
    )

    print(
        f"[{theme_name}] "
        f"Valid articles: {total_records:,}"
    )

    matched_records = count_theme_matches(
        conn,
        query_sql,
        query_params,
    )

    report["matched_records"] = (
        matched_records
    )

    coverage = (
        matched_records / total_records
        if total_records > 0
        else 0.0
    )

    print(
        f"[{theme_name}] "
        f"FTS matches: {matched_records:,} "
        f"({coverage:.3%})"
    )

    if matched_records == 0:
        print(
            f"[WARNING] No matches for "
            f"{theme_name}. All scores "
            f"will be zero."
        )

        cap = None

    else:
        print(
            f"[{theme_name}] "
            f"Computing p"
            f"{int(NORMALIZATION_PERCENTILE * 100)} "
            f"of ts_rank over matched articles..."
        )

        cap = compute_normalization_cap(
            conn,
            query_sql,
            query_params,
        )

    report["normalization_cap"] = cap

    if cap is None:
        print(
            f"[{theme_name}] "
            f"Normalization cap unavailable."
        )
    else:
        print(
            f"[{theme_name}] "
            f"Normalization cap = {cap:.8f}"
        )

    scoring_sql = f"""
    SELECT
        article_id,

        body_tsv @@ {query_sql}
            AS is_match,

        CASE
            WHEN body_tsv @@ {query_sql}
            THEN ts_rank(
                body_tsv,
                {query_sql}
            )
            ELSE 0.0
        END AS raw_rank

    FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW}

    ORDER BY article_id;
    """

    # The dynamic tsquery occurs three times in the scoring query.
    scoring_params = (
        query_params
        + query_params
        + query_params
    )

    cursor_name = (
        f"keyword_scoring_{theme_id}"
    )

    batch: List[Tuple] = []

    start_time = time.time()

    next_progress = (
        PROGRESS_EVERY_N_ROWS
    )

    matched_seen_in_stream = 0

    # A server-side cursor keeps memory usage bounded for large corpora.
    with conn.cursor(
        name=cursor_name
    ) as cur:

        cur.itersize = FETCH_SIZE

        cur.execute(
            scoring_sql,
            scoring_params,
        )

        while True:
            rows = cur.fetchmany(
                FETCH_SIZE
            )

            if not rows:
                break

            for (
                article_id,
                is_match,
                raw_rank,
            ) in rows:

                raw_rank = (
                    float(raw_rank)
                    if raw_rank is not None
                    else 0.0
                )

                if not is_match:
                    score = 0.0

                else:
                    matched_seen_in_stream += 1

                    if cap is None or cap <= 0:
                        score = 0.0
                    else:
                        score = min(
                            max(
                                raw_rank / cap,
                                0.0,
                            ),
                            1.0,
                        )

                batch.append(
                    (
                        article_id,
                        theme_id,
                        score,
                        METHOD_NAME,
                        MODEL_VERSION,
                    )
                )

                report["scored_records"] += 1

                if (
                    len(batch)
                    >= UPSERT_BATCH_SIZE
                ):
                    upsert_scores(
                        conn,
                        batch,
                    )

                    batch = []

                if (
                    report["scored_records"]
                    >= next_progress
                ):
                    print_progress(
                        theme_name,
                        report["scored_records"],
                        total_records,
                        start_time,
                    )

                    next_progress += (
                        PROGRESS_EVERY_N_ROWS
                    )

    if batch:
        upsert_scores(
            conn,
            batch,
        )

    # The independent match count must agree with the streamed scoring pass.
    if (
        matched_seen_in_stream
        != matched_records
    ):
        raise RuntimeError(
            f"[{theme_name}] Validation error: "
            f"COUNT @@ = {matched_records:,}, "
            f"but the scoring stream found "
            f"{matched_seen_in_stream:,} matches."
        )

    print_progress(
        theme_name,
        report["scored_records"],
        total_records,
        start_time,
    )

    print(
        f"[OK][{theme_name}] "
        f"Match validation passed: "
        f"{matched_seen_in_stream:,} = "
        f"{matched_records:,}"
    )

    return report


# =============================================================================
# FINAL REPORT
# =============================================================================

def print_final_report(
    reports: List[Dict[str, Any]],
) -> None:
    """Print the aggregate keyword-baseline report."""

    print()
    print("=" * 110)
    print("PHASE 2A — KEYWORD BASELINE")
    print("=" * 110)

    for report in reports:
        total = report["total_records"]
        matched = report["matched_records"]
        scored = report["scored_records"]

        coverage = (
            matched / total
            if total > 0
            else 0.0
        )

        print(
            f"{report['theme_name']:<20} | "
            f"articles={total:>10,} | "
            f"matches={matched:>10,} | "
            f"coverage={coverage:>8.3%} | "
            f"scored={scored:>10,} | "
            f"cap={report['normalization_cap']}"
        )

    total_scores = sum(
        report["scored_records"]
        for report in reports
    )

    print("-" * 110)

    print(
        f"Total thematic scores processed: "
        f"{total_scores:,}"
    )
    print(
        f"method        = {METHOD_NAME}"
    )
    print(
        f"model_version = {MODEL_VERSION}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the Phase 2A keyword baseline."""

    with get_connection() as conn:
        conn.autocommit = False

        print(
            "PostgreSQL connection established."
        )

        check_database_structure(conn)

        themes = get_active_themes(
            conn,
            THEME_NAME_FILTER,
        )

        if not themes:
            print(
                "No active themes found."
            )
            return

        print()
        print("Themes to process:")

        for theme in themes:
            print(
                f"  {theme['theme_id']} "
                f"- {theme['theme_name']} "
                f"- {theme['theme_label']}"
            )

        reports = []

        try:
            for theme in themes:
                report = score_theme(
                    conn,
                    theme,
                )

                reports.append(report)

            if COMMIT_FINAL_INSERTS:
                conn.commit()
                print("\nCOMMIT completed.")

            else:
                conn.rollback()

                print("\nROLLBACK completed.")
                print(
                    "TEST MODE: no data was retained "
                    "in news_theme_scores."
                )

        except Exception:
            conn.rollback()

            print(
                "\nERROR: automatic rollback completed."
            )

            raise

        print_final_report(reports)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

"""
Weak V3 Continuous — Full-Corpus Production Scorer

Compute and store the frozen Weak V3 Continuous Candidate C score
for every valid article and each of the five active risk themes.

Frozen output:
    method        = weakly_supervised
    model_version = weak_v3_continuous_frozen

The thematic score is a continuous article-theme relevance score in [0, 1].
It is not a calibrated probability.

The script is resumable. An article is considered complete only when
PostgreSQL contains five distinct theme scores for the frozen method and
model version. Missing or incomplete articles are rescored, while committed
batches remain valid.

Required database uniqueness:
    UNIQUE (article_id, theme_id, method, model_version)

Expected production coverage:
    2,249,255 articles
    5 themes
    11,246,275 article-theme scores
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from weak_supervision_v3_continuous_config import (
    VERSION as CONTINUOUS_VERSION,
    METHOD,
    MODEL_VERSION,
    SELECTED_CANDIDATE,
    PROXIMITY_WINDOW,
    DENSITY_REINFORCEMENT,
    SCORE_MIN,
    SCORE_MAX,
)

from validate_weak_v3_continuous_article_engine import (
    score_article_all_themes,
)


# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================

ARTICLE_BATCH_SIZE = 1_000

EXPECTED_THEMES = 5
EXPECTED_CORPUS_ARTICLES = 2_249_255
EXPECTED_FINAL_SCORES = EXPECTED_CORPUS_ARTICLES * EXPECTED_THEMES


# =============================================================================
# FROZEN CONFIGURATION LOCKS
# =============================================================================

EXPECTED_VERSION = (
    "Weak Supervision V3 Continuous - Frozen Development Version"
)

EXPECTED_METHOD = "weakly_supervised"
EXPECTED_MODEL_VERSION = "weak_v3_continuous_frozen"
EXPECTED_CANDIDATE = "score_C_density"
EXPECTED_WINDOW = 50
EXPECTED_DENSITY_REINFORCEMENT = 0.35


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

SCHEMA_NAME = "risk_emergence"
VALID_ARTICLES_VIEW = "v_valid_articles"
THEMES_TABLE = "candidate_themes"
SCORES_TABLE = "news_theme_scores"

# The database column name contains a space and therefore requires quoting.
SCORE_COLUMN = '"thematic scores"'

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "research"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": (
        os.getenv("THESIS_DB_PASSWORD")
        or os.getenv("DB_PASSWORD")
    ),
}


def get_connection():

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "PostgreSQL password not configured. "
            "Set THESIS_DB_PASSWORD or DB_PASSWORD."
        )

    return psycopg2.connect(
        **DB_CONFIG,
        connect_timeout=10,
    )


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def format_seconds(seconds):

    if (
        seconds is None
        or not np.isfinite(seconds)
        or seconds < 0
    ):
        return "n/a"

    total = int(round(seconds))

    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def verify_frozen_configuration():

    if CONTINUOUS_VERSION != EXPECTED_VERSION:
        raise RuntimeError(
            "Continuous version mismatch."
        )

    if METHOD != EXPECTED_METHOD:
        raise RuntimeError(
            "Method mismatch."
        )

    if MODEL_VERSION != EXPECTED_MODEL_VERSION:
        raise RuntimeError(
            "Model version mismatch."
        )

    if SELECTED_CANDIDATE != EXPECTED_CANDIDATE:
        raise RuntimeError(
            "Candidate mismatch."
        )

    if PROXIMITY_WINDOW != EXPECTED_WINDOW:
        raise RuntimeError(
            "Proximity window mismatch."
        )

    if not np.isclose(
        DENSITY_REINFORCEMENT,
        EXPECTED_DENSITY_REINFORCEMENT,
    ):
        raise RuntimeError(
            "Density reinforcement mismatch."
        )

    if SCORE_MIN != 0.0 or SCORE_MAX != 1.0:
        raise RuntimeError(
            "Unexpected thematic score range."
        )


# =============================================================================
# THEMES
# =============================================================================

def load_active_themes(conn):

    sql = f"""
        SELECT
            theme_id,
            theme_label
        FROM {SCHEMA_NAME}.{THEMES_TABLE}
        WHERE is_active = TRUE
        ORDER BY theme_id
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    themes = [
        (
            int(theme_id),
            str(theme_label),
        )
        for theme_id, theme_label in rows
    ]

    if len(themes) != EXPECTED_THEMES:
        raise RuntimeError(
            f"Expected {EXPECTED_THEMES} active themes, "
            f"found {len(themes)}."
        )

    theme_ids = [
        theme_id
        for theme_id, _ in themes
    ]

    if len(set(theme_ids)) != EXPECTED_THEMES:
        raise RuntimeError(
            "Duplicate active theme IDs."
        )

    return themes


# =============================================================================
# DATABASE STATE
# =============================================================================

def get_corpus_count(conn):

    sql = f"""
        SELECT COUNT(*)
        FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW}
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        return int(cur.fetchone()[0])


def get_score_state(conn):

    sql = f"""
        SELECT
            COUNT(*) AS n_scores,
            COUNT(DISTINCT article_id) AS n_articles,
            COUNT(DISTINCT theme_id) AS n_themes,
            MIN({SCORE_COLUMN}),
            AVG({SCORE_COLUMN}),
            MAX({SCORE_COLUMN})
        FROM {SCHEMA_NAME}.{SCORES_TABLE}
        WHERE method = %s
          AND model_version = %s
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                METHOD,
                MODEL_VERSION,
            ),
        )

        return cur.fetchone()


def get_complete_article_count(conn):

    sql = f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                article_id
            FROM {SCHEMA_NAME}.{SCORES_TABLE}
            WHERE method = %s
              AND model_version = %s
            GROUP BY article_id
            HAVING COUNT(DISTINCT theme_id) = %s
        ) x
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                METHOD,
                MODEL_VERSION,
                EXPECTED_THEMES,
            ),
        )

        return int(cur.fetchone()[0])


# =============================================================================
# RESUMABLE ARTICLE READER
# =============================================================================

def load_incomplete_article_batch(
    conn,
    after_article_id,
    batch_size,
):

    """
    Return the next valid articles without all five frozen theme scores.

    Keyset pagination is used within the current run. Articles with partial
    score coverage are selected again and repaired through the UPSERT.
    """

    sql = f"""
        SELECT
            a.article_id,
            a.title,
            a.body
        FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW} a
        WHERE a.article_id > %s
          AND NOT EXISTS (
              SELECT 1
              FROM {SCHEMA_NAME}.{SCORES_TABLE} s
              WHERE s.article_id = a.article_id
                AND s.method = %s
                AND s.model_version = %s
              GROUP BY s.article_id
              HAVING COUNT(DISTINCT s.theme_id) = %s
          )
        ORDER BY a.article_id
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                after_article_id,
                METHOD,
                MODEL_VERSION,
                EXPECTED_THEMES,
                batch_size,
            ),
        )

        return cur.fetchall()


# =============================================================================
# ARTICLE SCORING
# =============================================================================

def score_article(
    article_id,
    title,
    body,
    themes,
):

    results = score_article_all_themes(
        article_id=article_id,
        title=title,
        body=body,
        themes=themes,
    )

    if len(results) != EXPECTED_THEMES:
        raise RuntimeError(
            f"Article {article_id}: expected "
            f"{EXPECTED_THEMES} scores, got {len(results)}."
        )

    rows = []
    seen_theme_ids = set()

    for (
        scored_article_id,
        theme_id,
        _theme_name,
        score,
    ) in results:

        if int(scored_article_id) != int(article_id):
            raise RuntimeError(
                "Article ID mismatch inside scorer."
            )

        theme_id = int(theme_id)

        if theme_id in seen_theme_ids:
            raise RuntimeError(
                f"Duplicate theme {theme_id} "
                f"for article {article_id}."
            )

        seen_theme_ids.add(theme_id)

        score = float(score)

        if not np.isfinite(score):
            raise RuntimeError(
                f"Non-finite thematic score: "
                f"article={article_id}, "
                f"theme={theme_id}."
            )

        if score < SCORE_MIN or score > SCORE_MAX:
            raise RuntimeError(
                f"Score outside [0, 1]: "
                f"article={article_id}, "
                f"theme={theme_id}, "
                f"score={score}."
            )

        rows.append(
            (
                int(article_id),
                theme_id,
                score,
                METHOD,
                MODEL_VERSION,
            )
        )

    expected_theme_ids = {
        int(theme_id)
        for theme_id, _ in themes
    }

    if seen_theme_ids != expected_theme_ids:
        raise RuntimeError(
            f"Article {article_id}: theme set mismatch. "
            f"Expected {sorted(expected_theme_ids)}, "
            f"got {sorted(seen_theme_ids)}."
        )

    return rows


# =============================================================================
# UPSERT
# =============================================================================

def upsert_scores(
    conn,
    score_rows,
):

    if not score_rows:
        return

    sql = f"""
        INSERT INTO {SCHEMA_NAME}.{SCORES_TABLE}
        (
            article_id,
            theme_id,
            {SCORE_COLUMN},
            method,
            model_version
        )
        VALUES %s

        ON CONFLICT
        (
            article_id,
            theme_id,
            method,
            model_version
        )

        DO UPDATE SET
            {SCORE_COLUMN} = EXCLUDED.{SCORE_COLUMN},
            computed_at = CURRENT_TIMESTAMP
    """

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            score_rows,
            page_size=5_000,
        )


# =============================================================================
# FINAL INTEGRITY CHECK
# =============================================================================

def final_integrity_check(
    conn,
    corpus_count,
):

    print()
    print("=" * 100)
    print("FINAL DATABASE INTEGRITY CHECK")
    print("=" * 100)

    sql = f"""
        SELECT
            COUNT(*) AS n_scores,
            COUNT(DISTINCT article_id) AS n_articles,
            COUNT(DISTINCT theme_id) AS n_themes,
            MIN({SCORE_COLUMN}),
            AVG({SCORE_COLUMN}),
            MAX({SCORE_COLUMN})
        FROM {SCHEMA_NAME}.{SCORES_TABLE}
        WHERE method = %s
          AND model_version = %s
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                METHOD,
                MODEL_VERSION,
            ),
        )

        (
            n_scores,
            n_articles,
            n_themes,
            min_score,
            mean_score,
            max_score,
        ) = cur.fetchone()

    n_scores = int(n_scores)
    n_articles = int(n_articles)
    n_themes = int(n_themes)

    expected_scores = (
        corpus_count
        * EXPECTED_THEMES
    )

    print()
    print(
        f"Corpus articles:       "
        f"{corpus_count:,}"
    )
    print(
        f"Scored articles:       "
        f"{n_articles:,}"
    )
    print(
        f"Distinct themes:       "
        f"{n_themes:,}"
    )
    print(
        f"Stored scores:         "
        f"{n_scores:,}"
    )
    print(
        f"Expected scores:       "
        f"{expected_scores:,}"
    )
    print(
        f"Min thematic score:    "
        f"{float(min_score):.8f}"
    )
    print(
        f"Mean thematic score:   "
        f"{float(mean_score):.8f}"
    )
    print(
        f"Max thematic score:    "
        f"{float(max_score):.8f}"
    )

    incomplete_sql = f"""
        SELECT COUNT(*)
        FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW} a
        WHERE NOT EXISTS (
            SELECT 1
            FROM {SCHEMA_NAME}.{SCORES_TABLE} s
            WHERE s.article_id = a.article_id
              AND s.method = %s
              AND s.model_version = %s
            GROUP BY s.article_id
            HAVING COUNT(DISTINCT s.theme_id) = %s
        )
    """

    with conn.cursor() as cur:
        cur.execute(
            incomplete_sql,
            (
                METHOD,
                MODEL_VERSION,
                EXPECTED_THEMES,
            ),
        )

        incomplete_articles = int(
            cur.fetchone()[0]
        )

    print(
        f"Incomplete articles:   "
        f"{incomplete_articles:,}"
    )

    if n_articles != corpus_count:
        raise RuntimeError(
            "FINAL CHECK FAILED: not every valid "
            "article has stored scores."
        )

    if n_themes != EXPECTED_THEMES:
        raise RuntimeError(
            "FINAL CHECK FAILED: "
            "theme count is not 5."
        )

    if n_scores != expected_scores:
        raise RuntimeError(
            "FINAL CHECK FAILED: "
            "unexpected number of scores."
        )

    if incomplete_articles != 0:
        raise RuntimeError(
            "FINAL CHECK FAILED: "
            "some articles are incomplete."
        )

    if (
        float(min_score) < SCORE_MIN
        or float(max_score) > SCORE_MAX
    ):
        raise RuntimeError(
            "FINAL CHECK FAILED: "
            "thematic scores outside [0, 1]."
        )

    print()
    print(
        "[OK] FINAL DATABASE "
        "INTEGRITY CHECK PASSED"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "WEAK V3 CONTINUOUS — "
        "FULL-CORPUS PRODUCTION"
    )
    print("=" * 100)

    verify_frozen_configuration()

    print()
    print("Frozen specification:")
    print(
        f"  version               = "
        f"{CONTINUOUS_VERSION}"
    )
    print(
        f"  method                = "
        f"{METHOD}"
    )
    print(
        f"  model_version         = "
        f"{MODEL_VERSION}"
    )
    print(
        f"  candidate             = "
        f"{SELECTED_CANDIDATE}"
    )
    print(
        f"  proximity_window      = "
        f"{PROXIMITY_WINDOW}"
    )
    print(
        f"  density_reinforcement = "
        f"{DENSITY_REINFORCEMENT}"
    )

    print()
    print("MODE = FULL CORPUS")
    print(
        f"ARTICLE BATCH SIZE = "
        f"{ARTICLE_BATCH_SIZE:,}"
    )

    print()
    print("=" * 100)
    print("1. CONNECT TO POSTGRESQL")
    print("=" * 100)

    conn = get_connection()

    try:

        conn.autocommit = False

        print()
        print(
            "[OK] PostgreSQL connection established"
        )

        print()
        print("=" * 100)
        print("2. VERIFY ACTIVE THEMES")
        print("=" * 100)

        themes = load_active_themes(
            conn
        )

        for theme_id, theme_name in themes:
            print(
                f"{theme_id} | {theme_name}"
            )

        print()
        print("=" * 100)
        print("3. PRE-RUN DATABASE STATE")
        print("=" * 100)

        corpus_count = get_corpus_count(
            conn
        )

        if corpus_count != EXPECTED_CORPUS_ARTICLES:
            raise RuntimeError(
                f"Corpus count changed: expected "
                f"{EXPECTED_CORPUS_ARTICLES:,}, "
                f"found {corpus_count:,}."
            )

        (
            existing_scores,
            existing_articles,
            existing_themes,
            existing_min,
            existing_mean,
            existing_max,
        ) = get_score_state(
            conn
        )

        complete_articles = (
            get_complete_article_count(
                conn
            )
        )

        existing_scores = int(
            existing_scores
        )
        existing_articles = int(
            existing_articles
        )
        existing_themes = int(
            existing_themes
        )

        remaining_articles = (
            corpus_count
            - complete_articles
        )

        print()
        print(
            f"Valid corpus articles: "
            f"{corpus_count:,}"
        )
        print(
            f"Expected final scores: "
            f"{EXPECTED_FINAL_SCORES:,}"
        )
        print()
        print(
            f"Existing scores:       "
            f"{existing_scores:,}"
        )
        print(
            f"Existing articles:     "
            f"{existing_articles:,}"
        )
        print(
            f"Complete articles:     "
            f"{complete_articles:,}"
        )
        print(
            f"Remaining articles:    "
            f"{remaining_articles:,}"
        )

        if existing_scores > 0:

            print(
                f"Existing themes:       "
                f"{existing_themes:,}"
            )
            print(
                f"Existing min score:    "
                f"{float(existing_min):.8f}"
            )
            print(
                f"Existing mean score:   "
                f"{float(existing_mean):.8f}"
            )
            print(
                f"Existing max score:    "
                f"{float(existing_max):.8f}"
            )

        if complete_articles == corpus_count:

            print()
            print(
                "All articles are already complete."
            )

            final_integrity_check(
                conn,
                corpus_count,
            )

            return

        print()
        print("=" * 100)
        print("4. FULL-CORPUS SCORE + UPSERT")
        print("=" * 100)

        total_start = time.perf_counter()

        total_read_seconds = 0.0
        total_scoring_seconds = 0.0
        total_insert_seconds = 0.0

        processed_articles = 0
        processed_scores = 0

        # The keyset advances only after a successful batch commit.
        after_article_id = 0
        batch_number = 0

        while True:

            read_start = time.perf_counter()

            articles = load_incomplete_article_batch(
                conn=conn,
                after_article_id=after_article_id,
                batch_size=ARTICLE_BATCH_SIZE,
            )

            read_elapsed = (
                time.perf_counter()
                - read_start
            )

            total_read_seconds += read_elapsed

            if not articles:
                break

            batch_number += 1

            score_start = time.perf_counter()

            batch_score_rows = []

            for (
                article_id,
                title,
                body,
            ) in articles:

                article_rows = score_article(
                    article_id=article_id,
                    title=title,
                    body=body,
                    themes=themes,
                )

                batch_score_rows.extend(
                    article_rows
                )

            score_elapsed = (
                time.perf_counter()
                - score_start
            )

            total_scoring_seconds += (
                score_elapsed
            )

            expected_batch_scores = (
                len(articles)
                * EXPECTED_THEMES
            )

            if (
                len(batch_score_rows)
                != expected_batch_scores
            ):
                raise RuntimeError(
                    "Batch score count mismatch."
                )

            insert_start = time.perf_counter()

            upsert_scores(
                conn,
                batch_score_rows,
            )

            conn.commit()

            insert_elapsed = (
                time.perf_counter()
                - insert_start
            )

            total_insert_seconds += (
                insert_elapsed
            )

            after_article_id = int(
                articles[-1][0]
            )

            processed_articles += len(
                articles
            )

            processed_scores += len(
                batch_score_rows
            )

            total_elapsed = (
                time.perf_counter()
                - total_start
            )

            speed = (
                processed_articles
                / total_elapsed
                if total_elapsed > 0
                else 0.0
            )

            still_to_process = max(
                remaining_articles
                - processed_articles,
                0,
            )

            eta_seconds = (
                still_to_process / speed
                if speed > 0
                else float("nan")
            )

            print(
                f"Batch {batch_number:,} | "
                f"processed "
                f"{processed_articles:,}/"
                f"{remaining_articles:,} "
                f"articles | "
                f"scores "
                f"{processed_scores:,} | "
                f"{speed:,.1f} articles/s | "
                f"ETA "
                f"{format_seconds(eta_seconds)} | "
                f"read "
                f"{read_elapsed:.2f}s | "
                f"score "
                f"{score_elapsed:.2f}s | "
                f"insert "
                f"{insert_elapsed:.2f}s"
            )

        total_elapsed = (
            time.perf_counter()
            - total_start
        )

        print()
        print("=" * 100)
        print("5. PRODUCTION RUNTIME SUMMARY")
        print("=" * 100)

        print()
        print(
            f"Articles processed this run: "
            f"{processed_articles:,}"
        )
        print(
            f"Scores generated this run:   "
            f"{processed_scores:,}"
        )
        print()
        print(
            f"DB read time:        "
            f"{total_read_seconds:,.2f} s"
        )
        print(
            f"Scoring time:        "
            f"{total_scoring_seconds:,.2f} s"
        )
        print(
            f"Insert+commit time:  "
            f"{total_insert_seconds:,.2f} s"
        )
        print(
            f"Total elapsed:       "
            f"{total_elapsed:,.2f} s"
        )
        print(
            f"Total elapsed:       "
            f"{format_seconds(total_elapsed)}"
        )

        if (
            total_elapsed > 0
            and processed_articles > 0
        ):
            print(
                f"Overall speed:       "
                f"{processed_articles / total_elapsed:,.2f} "
                f"articles/s"
            )

        final_integrity_check(
            conn,
            corpus_count,
        )

        print()
        print("=" * 100)
        print(
            "FULL-CORPUS WEAK V3 CONTINUOUS "
            "SCORING COMPLETE"
        )
        print("=" * 100)

        print()
        print(
            f"method        = {METHOD}"
        )
        print(
            f"model_version = {MODEL_VERSION}"
        )
        print(
            f"articles      = {corpus_count:,}"
        )
        print(
            f"themes        = {EXPECTED_THEMES:,}"
        )
        print(
            f"scores        = {EXPECTED_FINAL_SCORES:,}"
        )

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print()
        print(
            "RUN INTERRUPTED BY USER."
        )
        print(
            "Already committed batches "
            "remain in PostgreSQL."
        )
        print(
            "Rerun the same script to "
            "resume safely."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("FATAL ERROR:")
        print(repr(exc))
        print()
        print(
            "Already committed batches "
            "remain valid."
        )
        print(
            "After resolving the error, "
            "rerun this script to resume."
        )

        sys.exit(1)
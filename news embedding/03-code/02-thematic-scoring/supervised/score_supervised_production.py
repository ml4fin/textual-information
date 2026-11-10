#!/usr/bin/env python3

"""
SUPERVISED MODEL V1 — PRODUCTION DIAGNOSTIC SCORER
===================================================

Apply the frozen Supervised V1 model to valid PostgreSQL articles under the
production article × theme scoring regime.

This script belongs to the V1 development/diagnostic stage and is retained
for reproducibility. Supervised V1 was later superseded by Supervised V3.

Input
-----
    risk_emergence.v_valid_articles
    risk_emergence.candidate_themes

Model
-----
    supervised_model_v1.joblib

Output
------
    risk_emergence.news_theme_scores

For each article-theme pair:

    article_id
    theme_id
    thematic score = uncalibrated predict_proba() score for class 1
    method = supervised
    model_version = supervised_v1

The frozen V1 threshold (0.35) is used only for diagnostic binary counts.
Continuous model scores are stored without thresholding.

By default the script processes only 1,000 articles.
"""

from __future__ import annotations

import os
import time
from getpass import getpass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psycopg2
from psycopg2.extras import execute_values


# =============================================================================
# CONFIGURATION
# =============================================================================

SCHEMA = "risk_emergence"

ARTICLES_VIEW = "v_valid_articles"
THEMES_TABLE = "candidate_themes"
SCORES_TABLE = "news_theme_scores"

METHOD = "supervised"
MODEL_VERSION = "supervised_v1"

FROZEN_THRESHOLD = 0.35

MAX_ARTICLES = 1_000

ARTICLES_PER_BATCH = 1_000
PREDICTION_BATCH_SIZE = 2_000
UPSERT_BATCH_SIZE = 5_000

# Safe default. Set SUPERVISED_V1_WRITE=1 only when DB writes are intended.
WRITE_TO_DATABASE = (
    os.getenv(
        "SUPERVISED_V1_WRITE",
        "0",
    )
    == "1"
)

BASE_DIR = Path(
    os.getenv(
        "SUPERVISED_MODELS_DIR",
        "07_supervised_models",
    )
)

MODEL_FILE = (
    BASE_DIR
    / "06_frozen"
    / "supervised_model_v1.joblib"
)


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DB_CONFIG = {
    "host": os.getenv(
        "DB_HOST",
        "localhost",
    ),
    "port": int(
        os.getenv(
            "DB_PORT",
            "5432",
        )
    ),
    "dbname": os.getenv(
        "DB_NAME",
        "research",
    ),
    "user": os.getenv(
        "DB_USER",
        "postgres",
    ),
}


# =============================================================================
# FROZEN THEMES
# =============================================================================

EXPECTED_THEME_LABELS = {
    "AI-Related Risk",
    "Climate / Environmental Risk",
    "Cybersecurity Risk",
    "Pandemic Risk",
    "Supply Chain Risk",
}


# =============================================================================
# UTILITIES
# =============================================================================

def clean_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    return str(value)


def format_seconds(
    seconds: float,
) -> str:

    total = max(
        int(seconds),
        0,
    )

    hours = total // 3600
    minutes = (
        total % 3600
    ) // 60
    secs = total % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def build_model_text(
    theme_label: str,
    title: str,
    body: str,
) -> str:
    """
    Reproduce the exact V1 training representation.
    """

    return (
        f"THEME: {clean_text(theme_label)} "
        f"TITLE: {clean_text(title)} "
        f"BODY: {clean_text(body)}"
    )


# =============================================================================
# DATABASE
# =============================================================================

def get_connection():

    password = (
        os.getenv("THESIS_DB_PASSWORD")
        or os.getenv("DB_PASSWORD")
    )

    if not password:
        password = getpass(
            "PostgreSQL password: "
        )

    conn = psycopg2.connect(
        **DB_CONFIG,
        password=password,
        connect_timeout=10,
    )

    conn.autocommit = False

    print(
        "[OK] PostgreSQL connection established."
    )

    return conn


def check_database_structure(
    conn,
):

    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                SCHEMA,
                ARTICLES_VIEW,
            ),
        )

        article_columns = {
            row[0]
            for row in cur.fetchall()
        }

        cur.execute(
            sql,
            (
                SCHEMA,
                SCORES_TABLE,
            ),
        )

        score_columns = {
            row[0]
            for row in cur.fetchall()
        }

    required_articles = {
        "article_id",
        "title",
        "body",
    }

    required_scores = {
        "article_id",
        "theme_id",
        "thematic scores",
        "method",
        "model_version",
        "computed_at",
    }

    missing_articles = (
        required_articles
        - article_columns
    )

    missing_scores = (
        required_scores
        - score_columns
    )

    if missing_articles:
        raise RuntimeError(
            f"{SCHEMA}.{ARTICLES_VIEW} is missing columns: "
            f"{sorted(missing_articles)}"
        )

    if missing_scores:
        raise RuntimeError(
            f"{SCHEMA}.{SCORES_TABLE} is missing columns: "
            f"{sorted(missing_scores)}"
        )

    print(
        "[OK] database structure validated."
    )


# =============================================================================
# MODEL
# =============================================================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Frozen V1 model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise RuntimeError(
            "Loaded V1 model does not expose predict_proba()."
        )

    print(
        "[OK] frozen Supervised V1 model loaded."
    )

    return model


# =============================================================================
# THEMES
# =============================================================================

def get_active_themes(
    conn,
):

    sql = f"""
        SELECT
            theme_id,
            theme_name,
            theme_label
        FROM {SCHEMA}.{THEMES_TABLE}
        WHERE is_active = TRUE
        ORDER BY theme_id;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        rows = cur.fetchall()

    themes = [
        {
            "theme_id": int(theme_id),
            "theme_name": clean_text(
                theme_name
            ),
            "theme_label": clean_text(
                theme_label
            ),
        }
        for (
            theme_id,
            theme_name,
            theme_label,
        ) in rows
    ]

    if len(themes) != 5:
        raise RuntimeError(
            "Frozen V1 expects exactly five active themes. "
            f"Found {len(themes)}."
        )

    actual_labels = {
        theme["theme_label"]
        for theme in themes
    }

    if (
        actual_labels
        != EXPECTED_THEME_LABELS
    ):
        raise RuntimeError(
            "PostgreSQL theme labels do not match "
            "the labels used during V1 training.\n"
            f"Expected: {sorted(EXPECTED_THEME_LABELS)}\n"
            f"Found:    {sorted(actual_labels)}"
        )

    return themes


# =============================================================================
# ARTICLE RETRIEVAL
# =============================================================================

def fetch_article_page(
    conn,
    last_article_id,
    limit,
):

    sql = f"""
        SELECT
            article_id,
            title,
            body
        FROM {SCHEMA}.{ARTICLES_VIEW}
        WHERE article_id > %s
        ORDER BY article_id
        LIMIT %s;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                last_article_id,
                limit,
            ),
        )

        rows = cur.fetchall()

    return [
        (
            int(article_id),
            clean_text(title),
            clean_text(body),
        )
        for (
            article_id,
            title,
            body,
        ) in rows
    ]


# =============================================================================
# SCORING
# =============================================================================

def score_articles(
    model,
    articles,
    themes,
):

    metadata = []
    texts = []

    for (
        article_id,
        title,
        body,
    ) in articles:

        for theme in themes:

            texts.append(
                build_model_text(
                    theme["theme_label"],
                    title,
                    body,
                )
            )

            metadata.append(
                (
                    article_id,
                    theme["theme_id"],
                )
            )

    expected = (
        len(articles)
        * len(themes)
    )

    if (
        len(texts)
        != expected
    ):
        raise RuntimeError(
            "Unexpected number of V1 article-theme inputs."
        )

    output = []

    for start in range(
        0,
        len(texts),
        PREDICTION_BATCH_SIZE,
    ):

        stop = min(
            start + PREDICTION_BATCH_SIZE,
            len(texts),
        )

        probabilities = np.asarray(
            model.predict_proba(
                texts[start:stop]
            )[:, 1],
            dtype=float,
        )

        if not np.isfinite(
            probabilities
        ).all():
            raise RuntimeError(
                "Non-finite V1 model scores detected."
            )

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise RuntimeError(
                "V1 model score outside [0, 1]."
            )

        for offset, probability in enumerate(
            probabilities
        ):

            article_id, theme_id = (
                metadata[
                    start + offset
                ]
            )

            output.append(
                (
                    article_id,
                    theme_id,
                    float(probability),
                    METHOD,
                    MODEL_VERSION,
                )
            )

    if len(output) != expected:
        raise RuntimeError(
            "V1 output count does not match "
            "article-theme input count."
        )

    return output


# =============================================================================
# DATABASE WRITE
# =============================================================================

# This assumes that the database column is literally named
# "thematic scores".
#
# The conflict target must match the actual PostgreSQL UNIQUE constraint.
UPSERT_SQL = f"""
    INSERT INTO {SCHEMA}.{SCORES_TABLE}
    (
        article_id,
        theme_id,
        "thematic scores",
        method,
        model_version
    )
    VALUES %s

    ON CONFLICT
        (article_id, theme_id, method, model_version)

    DO UPDATE SET
        "thematic scores" =
            EXCLUDED."thematic scores",
        computed_at = NOW();
"""


def upsert_scores(
    conn,
    rows,
):

    if not rows:
        return

    with conn.cursor() as cur:

        execute_values(
            cur,
            UPSERT_SQL,
            rows,
            template=(
                "(%s, %s, %s, %s, %s)"
            ),
            page_size=UPSERT_BATCH_SIZE,
        )


# =============================================================================
# AUDIT
# =============================================================================

def count_v1_scores(
    conn,
):

    sql = f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT article_id)
        FROM {SCHEMA}.{SCORES_TABLE}
        WHERE method = %s
          AND model_version = %s;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                METHOD,
                MODEL_VERSION,
            ),
        )

        n_scores, n_articles = (
            cur.fetchone()
        )

    return (
        int(n_scores),
        int(n_articles),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "SUPERVISED MODEL V1 — "
        "PRODUCTION DIAGNOSTIC SCORER"
    )
    print("=" * 110)

    print(
        f"\nmethod        = {METHOD}"
    )

    print(
        f"model_version = {MODEL_VERSION}"
    )

    print(
        f"threshold     = {FROZEN_THRESHOLD}"
    )

    print(
        f"MAX_ARTICLES  = {MAX_ARTICLES}"
    )

    print(
        f"DB writes     = {WRITE_TO_DATABASE}"
    )

    model = load_model()

    conn = get_connection()

    try:

        check_database_structure(
            conn
        )

        themes = get_active_themes(
            conn
        )

        print(
            "\nFrozen V1 themes:"
        )

        for theme in themes:

            print(
                f"  {theme['theme_id']} | "
                f"{theme['theme_label']}"
            )

        existing_scores, existing_articles = (
            count_v1_scores(
                conn
            )
        )

        print(
            f"\nExisting V1 scores:   "
            f"{existing_scores:,}"
        )

        print(
            f"Existing V1 articles: "
            f"{existing_articles:,}"
        )

        last_article_id = -1

        articles_processed = 0
        scores_processed = 0

        started = time.perf_counter()

        while True:

            if (
                MAX_ARTICLES is not None
            ):

                remaining = (
                    MAX_ARTICLES
                    - articles_processed
                )

                if remaining <= 0:
                    break

                page_limit = min(
                    ARTICLES_PER_BATCH,
                    remaining,
                )

            else:

                page_limit = (
                    ARTICLES_PER_BATCH
                )

            articles = fetch_article_page(
                conn,
                last_article_id,
                page_limit,
            )

            if not articles:
                break

            predict_started = (
                time.perf_counter()
            )

            rows = score_articles(
                model,
                articles,
                themes,
            )

            predict_elapsed = (
                time.perf_counter()
                - predict_started
            )

            values = np.asarray(
                [
                    row[2]
                    for row in rows
                ],
                dtype=float,
            )

            print(
                f"\nArticles "
                f"{articles[0][0]:,} -> "
                f"{articles[-1][0]:,}"
            )

            print(
                f"[MODEL] {len(rows):,} scores "
                f"in {predict_elapsed:.2f}s"
            )

            print(
                f"        min={values.min():.6f} "
                f"mean={values.mean():.6f} "
                f"max={values.max():.6f}"
            )

            print(
                f"        >= {FROZEN_THRESHOLD:.2f}: "
                f"{int((values >= FROZEN_THRESHOLD).sum()):,}"
            )

            if WRITE_TO_DATABASE:

                try:

                    upsert_scores(
                        conn,
                        rows,
                    )

                    conn.commit()

                except Exception:

                    conn.rollback()
                    raise

                print(
                    f"[COMMIT] {len(rows):,} V1 scores"
                )

            else:

                print(
                    "[DRY RUN] no database rows written"
                )

            last_article_id = int(
                articles[-1][0]
            )

            articles_processed += (
                len(articles)
            )

            scores_processed += (
                len(rows)
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            rate = (
                articles_processed / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"[TOTAL] articles="
                f"{articles_processed:,} | "
                f"scores="
                f"{scores_processed:,} | "
                f"elapsed="
                f"{format_seconds(elapsed)} | "
                f"rate="
                f"{rate:.2f} art/s"
            )

        if WRITE_TO_DATABASE:

            final_scores, final_articles = (
                count_v1_scores(
                    conn
                )
            )

            print(
                f"\nStored V1 scores:   "
                f"{final_scores:,}"
            )

            print(
                f"Stored V1 articles: "
                f"{final_articles:,}"
            )

        print()
        print("=" * 110)
        print("V1 DIAGNOSTIC RUN COMPLETE")
        print("=" * 110)

        print(
            f"Articles processed: "
            f"{articles_processed:,}"
        )

        print(
            f"Scores processed:   "
            f"{scores_processed:,}"
        )

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

        print(
            "\nPostgreSQL connection closed."
        )


if __name__ == "__main__":
    main()
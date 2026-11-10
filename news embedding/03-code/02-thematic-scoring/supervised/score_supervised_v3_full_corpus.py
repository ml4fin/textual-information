#!/usr/bin/env python3

"""
SUPERVISED V3 FROZEN — FULL-CORPUS SCORING
==========================================

Apply the frozen Supervised V3 model to every valid article and all five
candidate risk themes.

Target:
    risk_emergence.news_theme_scores

Production identifiers:
    method        = supervised_v3
    model_version = supervised_v3_frozen

The stored thematic score is the continuous, uncalibrated predict_proba()
output for class 1.

The frozen classification threshold (0.34) is used only for diagnostics.

No fitting, retraining, calibration, or threshold selection is performed.

Production mode is restart-safe at completed article batches. Each committed
article must have exactly one score for each of the five frozen themes.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(
    os.getenv(
        "SUPERVISED_MODELS_DIR",
        "07_supervised_models",
    )
)

MODEL_FILE = (
    BASE_DIR
    / "06_frozen"
    / "supervised_model_v3.joblib"
)

METHOD = "supervised_v3"
MODEL_VERSION = "supervised_v3_frozen"

FROZEN_THRESHOLD = 0.34

SCHEMA = "risk_emergence"

ARTICLE_BATCH_SIZE = 1_000
TEST_ARTICLES = 1_000

# Safe default. Set SUPERVISED_V3_PRODUCTION=1 for production execution.
TEST_MODE = (
    os.getenv(
        "SUPERVISED_V3_PRODUCTION",
        "0",
    )
    != "1"
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
# FROZEN THEME MAPPING
# =============================================================================

FROZEN_THEME_NAMES = {
    1: "Cybersecurity Risk",
    2: "Climate / Environmental Risk",
    3: "Pandemic Risk",
    4: "Supply Chain Risk",
    5: "AI-Related Risk",
}

EXPECTED_THEME_IDS = set(
    FROZEN_THEME_NAMES
)

N_THEMES = len(
    FROZEN_THEME_NAMES
)


# =============================================================================
# FROZEN V3 REPRESENTATION
# =============================================================================

SEPARATOR = "|||"

TOKEN_RE = re.compile(
    r"(?u)\b\w\w+\b"
)


def safe_text(value) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def normalize_theme(
    theme,
) -> str:

    theme = safe_text(
        theme
    ).lower()

    theme = re.sub(
        r"[^a-z0-9]+",
        "_",
        theme,
    )

    return theme.strip("_")


def validate_separator_absence(
    title: str,
    body: str,
):

    if (
        SEPARATOR in title
        or SEPARATOR in body
    ):
        raise RuntimeError(
            "The frozen V3 separator occurs in article text. "
            "Scoring is stopped because the serialized model "
            "representation would be ambiguous."
        )


def build_v3_document(
    theme_name: str,
    title: str,
    body: str,
) -> str:

    title = safe_text(
        title
    )

    body = safe_text(
        body
    )

    validate_separator_absence(
        title,
        body,
    )

    theme_key = normalize_theme(
        theme_name
    )

    return (
        f"{theme_key} {SEPARATOR} "
        f"{title} {SEPARATOR} "
        f"{body}"
    )


def parse_v3_document(
    document: str,
):

    parts = str(
        document
    ).split(
        SEPARATOR,
        maxsplit=2,
    )

    if len(parts) != 3:
        raise ValueError(
            "Invalid frozen V3 document representation."
        )

    return (
        parts[0].strip(),
        parts[1].strip(),
        parts[2].strip(),
    )


class SharedTextExtractor(
    BaseEstimator,
    TransformerMixin,
):

    def fit(
        self,
        X,
        y=None,
    ):
        return self

    def transform(
        self,
        X,
    ):

        output = []

        for document in X:

            _, title, body = (
                parse_v3_document(
                    document
                )
            )

            output.append(
                f"TITLE: {title} "
                f"BODY: {body}"
            )

        return output


def theme_interaction_analyzer(
    document,
):

    theme_key, title, body = (
        parse_v3_document(
            document
        )
    )

    text = (
        f"{title} {body}"
    ).lower()

    tokens = TOKEN_RE.findall(
        text
    )

    features = []

    for token in tokens:
        features.append(
            f"{theme_key}__{token}"
        )

    for i in range(
        len(tokens) - 1
    ):
        features.append(
            f"{theme_key}__"
            f"{tokens[i]}_"
            f"{tokens[i + 1]}"
        )

    return features


# =============================================================================
# DATABASE
# =============================================================================

def connect_db():

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

    print(
        "[OK] PostgreSQL connection established"
    )

    return conn


def load_themes(
    conn,
):

    sql = f"""
        SELECT
            theme_id,
            theme_name
        FROM {SCHEMA}.candidate_themes
        WHERE is_active = TRUE
        ORDER BY theme_id;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        rows = cur.fetchall()

    themes = pd.DataFrame(
        rows,
        columns=[
            "theme_id",
            "theme_name",
        ],
    )

    if len(themes) != N_THEMES:
        raise RuntimeError(
            f"Expected exactly {N_THEMES} active themes, "
            f"found {len(themes)}."
        )

    actual_ids = set(
        themes[
            "theme_id"
        ]
        .astype(int)
        .tolist()
    )

    if actual_ids != EXPECTED_THEME_IDS:
        raise RuntimeError(
            "Database theme IDs do not match the frozen "
            "Supervised V3 mapping.\n"
            f"Expected: {sorted(EXPECTED_THEME_IDS)}\n"
            f"Found:    {sorted(actual_ids)}"
        )

    print(
        "[OK] database themes match frozen V3 mapping"
    )

    return themes


def count_valid_articles(
    conn,
):

    sql = f"""
        SELECT COUNT(*)
        FROM {SCHEMA}.v_valid_articles;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql
        )

        return int(
            cur.fetchone()[0]
        )


def count_exact_target_rows(
    conn,
):

    sql = f"""
        SELECT COUNT(*)
        FROM {SCHEMA}.news_theme_scores
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

        return int(
            cur.fetchone()[0]
        )


# =============================================================================
# MODEL
# =============================================================================

def load_frozen_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Frozen model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise RuntimeError(
            "Frozen model does not expose predict_proba()."
        )

    print(
        "[OK] frozen Supervised V3 model loaded"
    )

    return model


# =============================================================================
# RESUME AUDIT
# =============================================================================

def get_production_resume_state(
    conn,
):

    sql = f"""
        WITH article_status AS (
            SELECT
                article_id,
                COUNT(DISTINCT theme_id) AS n_themes
            FROM {SCHEMA}.news_theme_scores
            WHERE method = %s
              AND model_version = %s
            GROUP BY article_id
        )
        SELECT
            COUNT(*) FILTER (
                WHERE n_themes = %s
            ),
            COALESCE(
                SUM(n_themes),
                0
            ),
            COALESCE(
                MAX(article_id) FILTER (
                    WHERE n_themes = %s
                ),
                -1
            ),
            COUNT(*) FILTER (
                WHERE n_themes <> %s
            )
        FROM article_status;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                METHOD,
                MODEL_VERSION,
                N_THEMES,
                N_THEMES,
                N_THEMES,
            ),
        )

        row = cur.fetchone()

    return {
        "completed_articles": int(
            row[0] or 0
        ),
        "existing_scores": int(
            row[1] or 0
        ),
        "last_completed_article_id": int(
            row[2]
            if row[2] is not None
            else -1
        ),
        "incomplete_articles": int(
            row[3] or 0
        ),
    }


def count_missing_valid_articles_before_resume(
    conn,
    last_article_id,
):

    if last_article_id < 0:
        return 0

    sql = f"""
        SELECT COUNT(*)
        FROM {SCHEMA}.v_valid_articles v
        WHERE v.article_id <= %s
          AND NOT EXISTS (
              SELECT 1
              FROM {SCHEMA}.news_theme_scores s
              WHERE s.article_id = v.article_id
                AND s.method = %s
                AND s.model_version = %s
              GROUP BY s.article_id
              HAVING COUNT(DISTINCT s.theme_id) = %s
          );
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                last_article_id,
                METHOD,
                MODEL_VERSION,
                N_THEMES,
            ),
        )

        return int(
            cur.fetchone()[0]
        )


# =============================================================================
# SCORING
# =============================================================================

def build_scoring_frame(
    articles,
    themes,
):

    rows = []

    for (
        article_id,
        title,
        body,
    ) in articles:

        article_id = int(
            article_id
        )

        title = safe_text(
            title
        )

        body = safe_text(
            body
        )

        for theme in themes.itertuples(
            index=False
        ):

            theme_id = int(
                theme.theme_id
            )

            if (
                theme_id
                not in FROZEN_THEME_NAMES
            ):
                raise RuntimeError(
                    f"Unknown theme_id: {theme_id}"
                )

            model_theme_name = (
                FROZEN_THEME_NAMES[
                    theme_id
                ]
            )

            model_document = (
                build_v3_document(
                    model_theme_name,
                    title,
                    body,
                )
            )

            rows.append(
                (
                    article_id,
                    theme_id,
                    model_theme_name,
                    model_document,
                )
            )

    expected = (
        len(articles)
        * N_THEMES
    )

    if len(rows) != expected:
        raise RuntimeError(
            "Unexpected number of article-theme observations."
        )

    return rows


def score_rows(
    model,
    scoring_rows,
):

    if not scoring_rows:
        return np.array(
            [],
            dtype=float,
        )

    documents = [
        row[3]
        for row in scoring_rows
    ]

    probabilities = np.asarray(
        model.predict_proba(
            documents
        )[:, 1],
        dtype=float,
    )

    if (
        len(probabilities)
        != len(scoring_rows)
    ):
        raise RuntimeError(
            "Model output length does not match scoring input."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Non-finite model scores detected."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise RuntimeError(
            "Model scores outside [0, 1]."
        )

    return probabilities


# =============================================================================
# DATABASE WRITE
# =============================================================================

# IMPORTANT:
# This assumes that the PostgreSQL column is literally named
# "thematic scores".
#
# The ON CONFLICT target must match the actual UNIQUE constraint.
INSERT_SQL = f"""
    INSERT INTO {SCHEMA}.news_theme_scores
    (
        article_id,
        theme_id,
        "thematic scores",
        method,
        model_version,
        computed_at
    )
    VALUES %s

    ON CONFLICT
        (article_id, theme_id, method, model_version)

    DO UPDATE SET
        "thematic scores" =
            EXCLUDED."thematic scores",
        computed_at =
            EXCLUDED.computed_at;
"""


def write_scores(
    conn,
    scoring_rows,
    probabilities,
):

    if (
        len(scoring_rows)
        != len(probabilities)
    ):
        raise RuntimeError(
            "Cannot write scores: row/probability length mismatch."
        )

    computed_at = datetime.now(
        timezone.utc
    )

    values = [
        (
            int(row[0]),
            int(row[1]),
            float(probability),
            METHOD,
            MODEL_VERSION,
            computed_at,
        )
        for row, probability
        in zip(
            scoring_rows,
            probabilities,
        )
    ]

    try:

        with conn.cursor() as cur:

            execute_values(
                cur,
                INSERT_SQL,
                values,
                page_size=5_000,
            )

        conn.commit()

    except Exception:

        conn.rollback()

        raise


# =============================================================================
# TEST MODE
# =============================================================================

def run_test_mode(
    conn,
    model,
    themes,
):

    print()
    print("=" * 110)
    print(
        "TEST MODE — NO DATABASE WRITES"
    )
    print("=" * 110)

    sql = f"""
        SELECT
            article_id,
            title,
            body
        FROM {SCHEMA}.v_valid_articles
        ORDER BY article_id
        LIMIT %s;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (TEST_ARTICLES,),
        )

        articles = cur.fetchall()

    if not articles:
        raise RuntimeError(
            "No valid articles returned."
        )

    scoring_rows = build_scoring_frame(
        articles,
        themes,
    )

    probabilities = score_rows(
        model,
        scoring_rows,
    )

    predictions = (
        probabilities
        >= FROZEN_THRESHOLD
    ).astype(int)

    result = pd.DataFrame(
        {
            "article_id": [
                row[0]
                for row in scoring_rows
            ],
            "theme_id": [
                row[1]
                for row in scoring_rows
            ],
            "theme_name": [
                row[2]
                for row in scoring_rows
            ],
            "thematic_score": (
                probabilities
            ),
            "pred_at_034": (
                predictions
            ),
        }
    )

    print(
        f"\nArticles: "
        f"{len(articles):,}"
    )

    print(
        f"Article-theme scores: "
        f"{len(result):,}"
    )

    print(
        f"\nScore min:    "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Score mean:   "
        f"{probabilities.mean():.6f}"
    )

    print(
        f"Score median: "
        f"{np.median(probabilities):.6f}"
    )

    print(
        f"Score max:    "
        f"{probabilities.max():.6f}"
    )

    positive = int(
        predictions.sum()
    )

    print(
        f"\nPositive @ {FROZEN_THRESHOLD:.2f}: "
        f"{positive:,}/{len(result):,} "
        f"({positive / len(result):.2%})"
    )

    pivot = result.pivot(
        index="article_id",
        columns="theme_id",
        values="thematic_score",
    )

    if pivot.shape[1] != N_THEMES:
        raise RuntimeError(
            "Test output does not contain all frozen themes."
        )

    article_ranges = (
        pivot.max(axis=1)
        - pivot.min(axis=1)
    )

    print(
        "\nWithin-article score separation:"
    )

    print(
        f"  mean range:   "
        f"{article_ranges.mean():.6f}"
    )

    print(
        f"  median range: "
        f"{article_ranges.median():.6f}"
    )

    print(
        "\nPairwise theme score correlations:"
    )

    print(
        pivot.corr()
        .round(6)
        .to_string()
    )

    print(
        "\n[OK] TEST MODE completed."
    )

    print(
        "[OK] No rows were written to PostgreSQL."
    )


# =============================================================================
# PRODUCTION MODE
# =============================================================================

def run_production(
    conn,
    model,
    themes,
):

    print()
    print("=" * 110)
    print(
        "PRODUCTION MODE — RESTART-SAFE"
    )
    print("=" * 110)

    total_articles = count_valid_articles(
        conn
    )

    expected_scores = (
        total_articles
        * N_THEMES
    )

    resume = get_production_resume_state(
        conn
    )

    completed_before = (
        resume[
            "completed_articles"
        ]
    )

    scores_before = (
        resume[
            "existing_scores"
        ]
    )

    last_article_id = (
        resume[
            "last_completed_article_id"
        ]
    )

    incomplete = (
        resume[
            "incomplete_articles"
        ]
    )

    print(
        f"\nValid articles: "
        f"{total_articles:,}"
    )

    print(
        f"Expected scores: "
        f"{expected_scores:,}"
    )

    print(
        f"Completed articles: "
        f"{completed_before:,}"
    )

    print(
        f"Existing scores: "
        f"{scores_before:,}"
    )

    print(
        f"Last completed article_id: "
        f"{last_article_id}"
    )

    if incomplete != 0:
        raise RuntimeError(
            f"Found {incomplete:,} incomplete "
            "article-theme production records."
        )

    if (
        scores_before
        != completed_before * N_THEMES
    ):
        raise RuntimeError(
            "Existing production score count is inconsistent "
            "with completed article count."
        )

    if completed_before > 0:

        missing = (
            count_missing_valid_articles_before_resume(
                conn,
                last_article_id,
            )
        )

        if missing != 0:
            raise RuntimeError(
                f"Found {missing:,} valid articles before "
                "the resume point without complete frozen scores."
            )

    else:
        last_article_id = -1

    remaining = (
        total_articles
        - completed_before
    )

    if remaining < 0:
        raise RuntimeError(
            "Completed article count exceeds current valid corpus."
        )

    print(
        f"Articles remaining: "
        f"{remaining:,}"
    )

    session_articles = 0
    session_scores = 0
    session_positive = 0

    session_min = np.inf
    session_max = -np.inf
    session_sum = 0.0

    batch_number = 0
    started = time.time()

    while True:

        sql = f"""
            SELECT
                article_id,
                title,
                body
            FROM {SCHEMA}.v_valid_articles
            WHERE article_id > %s
            ORDER BY article_id
            LIMIT %s;
        """

        with conn.cursor() as cur:

            cur.execute(
                sql,
                (
                    last_article_id,
                    ARTICLE_BATCH_SIZE,
                ),
            )

            articles = cur.fetchall()

        if not articles:
            break

        scoring_rows = build_scoring_frame(
            articles,
            themes,
        )

        probabilities = score_rows(
            model,
            scoring_rows,
        )

        batch_positive = int(
            (
                probabilities
                >= FROZEN_THRESHOLD
            ).sum()
        )

        write_scores(
            conn,
            scoring_rows,
            probabilities,
        )

        batch_number += 1

        n_articles = len(
            articles
        )

        n_scores = len(
            scoring_rows
        )

        session_articles += n_articles
        session_scores += n_scores
        session_positive += batch_positive

        session_min = min(
            session_min,
            float(
                probabilities.min()
            ),
        )

        session_max = max(
            session_max,
            float(
                probabilities.max()
            ),
        )

        session_sum += float(
            probabilities.sum()
        )

        last_article_id = int(
            articles[-1][0]
        )

        cumulative_articles = (
            completed_before
            + session_articles
        )

        cumulative_scores = (
            scores_before
            + session_scores
        )

        elapsed = (
            time.time()
            - started
        )

        rate = (
            session_articles / elapsed
            if elapsed > 0
            else 0.0
        )

        remaining = max(
            total_articles
            - cumulative_articles,
            0,
        )

        eta = (
            remaining / rate
            if rate > 0
            else np.nan
        )

        print(
            f"Batch {batch_number:5d} | "
            f"articles="
            f"{cumulative_articles:,}/"
            f"{total_articles:,} | "
            f"scores="
            f"{cumulative_scores:,}/"
            f"{expected_scores:,} | "
            f"last_id="
            f"{last_article_id} | "
            f"batch_pos@.34="
            f"{batch_positive:,} | "
            f"elapsed="
            f"{elapsed / 60:.1f} min | "
            f"ETA="
            f"{eta / 60:.1f} min"
        )

    elapsed = (
        time.time()
        - started
    )

    actual_rows = count_exact_target_rows(
        conn
    )

    if actual_rows != expected_scores:
        raise RuntimeError(
            "Full-corpus frozen model coverage failed.\n"
            f"Expected: {expected_scores:,}\n"
            f"Found:    {actual_rows:,}"
        )

    final_audit_sql = f"""
        SELECT
            COUNT(*) AS article_count,
            COUNT(*) FILTER (
                WHERE n_themes <> %s
            ) AS incomplete_articles
        FROM (
            SELECT
                article_id,
                COUNT(DISTINCT theme_id)
                    AS n_themes
            FROM {SCHEMA}.news_theme_scores
            WHERE method = %s
              AND model_version = %s
            GROUP BY article_id
        ) x;
    """

    with conn.cursor() as cur:

        cur.execute(
            final_audit_sql,
            (
                N_THEMES,
                METHOD,
                MODEL_VERSION,
            ),
        )

        (
            final_article_count,
            final_incomplete,
        ) = cur.fetchone()

    final_article_count = int(
        final_article_count
    )

    final_incomplete = int(
        final_incomplete
    )

    if (
        final_article_count
        != total_articles
    ):
        raise RuntimeError(
            "Final article coverage does not match "
            "the valid article universe."
        )

    if final_incomplete != 0:
        raise RuntimeError(
            f"Found {final_incomplete:,} articles "
            "without all five frozen theme scores."
        )

    print()
    print("=" * 110)
    print(
        "PRODUCTION SCORING COMPLETE"
    )
    print("=" * 110)

    print(
        f"\nArticles processed this session: "
        f"{session_articles:,}"
    )

    print(
        f"Scores processed this session: "
        f"{session_scores:,}"
    )

    print(
        f"Final frozen score rows: "
        f"{actual_rows:,}"
    )

    if session_scores > 0:

        print(
            "\nThis-session score diagnostics:"
        )

        print(
            f"  min:  "
            f"{session_min:.6f}"
        )

        print(
            f"  mean: "
            f"{session_sum / session_scores:.6f}"
        )

        print(
            f"  max:  "
            f"{session_max:.6f}"
        )

        print(
            f"  positive @ "
            f"{FROZEN_THRESHOLD:.2f}: "
            f"{session_positive:,}/"
            f"{session_scores:,} "
            f"({session_positive / session_scores:.2%})"
        )

    print(
        f"\nElapsed this session: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "\n[OK] full article-theme coverage verified."
    )

    print(
        "[OK] exact frozen model-version coverage verified."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "SUPERVISED V3 FROZEN — "
        "FULL-CORPUS SCORER"
    )
    print("=" * 110)

    print(
        f"\nModel:\n{MODEL_FILE}"
    )

    print(
        f"\nTEST_MODE = {TEST_MODE}"
    )

    print(
        f"method = {METHOD}"
    )

    print(
        f"model_version = {MODEL_VERSION}"
    )

    print(
        "\nFrozen theme mapping:"
    )

    for (
        theme_id,
        theme_name,
    ) in FROZEN_THEME_NAMES.items():

        print(
            f"  {theme_id} -> "
            f"{theme_name} -> "
            f"{normalize_theme(theme_name)}"
        )

    model = load_frozen_model()

    conn = connect_db()

    try:

        themes = load_themes(
            conn
        )

        if TEST_MODE:

            run_test_mode(
                conn,
                model,
                themes,
            )

        else:

            run_production(
                conn,
                model,
                themes,
            )

    except KeyboardInterrupt:

        conn.rollback()

        print(
            "\nProduction interrupted. "
            "Previously committed batches remain stored."
        )

        print(
            "Run the script again to resume from the "
            "last verified completed article."
        )

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

        print(
            "\nDatabase connection closed."
        )


if __name__ == "__main__":
    main()
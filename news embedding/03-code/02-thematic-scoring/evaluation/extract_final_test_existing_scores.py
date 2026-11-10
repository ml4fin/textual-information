#!/usr/bin/env python3

"""
EXTRACT EXISTING KEYWORD + WEAK SCORES FOR FROZEN FINAL TEST
=============================================================

Retrieve the already-computed keyword and weak-supervision scores for the
500 frozen final-test article-theme observations.

IMPORTANT
---------
- No scores are recomputed.
- No database rows are modified.
- No model, rule, or threshold tuning is performed.
- Only the exact frozen method/model-version combinations are accepted.
- The final-test row universe is defined exclusively by the frozen gold file.

This script prepares existing baseline scores for final held-out comparison.
"""

from __future__ import annotations

from getpass import getpass
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


# =============================================================================
# CONFIGURATION
# =============================================================================

GOLD_FILE = Path(
    os.getenv(
        "FINAL_TEST_GOLD_FILE",
        "final_test_gold_frozen.csv",
    )
)

OUTPUT_FILE = Path(
    os.getenv(
        "FINAL_TEST_EXISTING_SCORES_FILE",
        "final_test_existing_keyword_weak_scores.csv",
    )
)

EXPECTED_ROWS = 500

KEYWORD_METHOD = "keyword"
KEYWORD_MODEL_VERSION = "seedwords_tsrank_v1"

WEAK_METHOD = "weakly_supervised"
WEAK_MODEL_VERSION = "weak_v3_continuous_frozen"

EXPECTED_MODELS = {
    KEYWORD_METHOD: KEYWORD_MODEL_VERSION,
    WEAK_METHOD: WEAK_MODEL_VERSION,
}

SCHEMA = "risk_emergence"
SCORES_TABLE = "news_theme_scores"

# Verify this against the actual PostgreSQL DDL.
SCORE_COLUMN = '"thematic scores"'


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DB_HOST = os.getenv(
    "DB_HOST",
    "localhost",
)

DB_PORT = int(
    os.getenv(
        "DB_PORT",
        "5432",
    )
)

DB_NAME = os.getenv(
    "DB_NAME",
    "research",
)

DB_USER = os.getenv(
    "DB_USER",
    "postgres",
)


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

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=password,
        connect_timeout=10,
    )


# =============================================================================
# GOLD DATA
# =============================================================================

def load_frozen_gold() -> pd.DataFrame:

    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Frozen gold file not found:\n"
            f"{GOLD_FILE}"
        )

    gold = pd.read_csv(
        GOLD_FILE
    )

    required_columns = {
        "article_id",
        "theme_id",
    }

    missing_columns = (
        required_columns
        - set(gold.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Frozen gold file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(gold) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} frozen gold rows, "
            f"found {len(gold)}."
        )

    if gold[
        ["article_id", "theme_id"]
    ].isna().any().any():

        raise RuntimeError(
            "Missing article_id or theme_id values "
            "in the frozen gold file."
        )

    try:
        gold["article_id"] = (
            pd.to_numeric(
                gold["article_id"],
                errors="raise",
            )
            .astype(int)
        )

        gold["theme_id"] = (
            pd.to_numeric(
                gold["theme_id"],
                errors="raise",
            )
            .astype(int)
        )

    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "article_id and theme_id must be valid integers."
        ) from exc

    if gold[
        ["article_id", "theme_id"]
    ].duplicated().any():

        duplicates = gold.loc[
            gold[
                ["article_id", "theme_id"]
            ].duplicated(
                keep=False
            ),
            ["article_id", "theme_id"],
        ]

        raise RuntimeError(
            "Duplicate article-theme pairs found "
            "in the frozen gold file.\n"
            f"{duplicates.to_string(index=False)}"
        )

    return gold


# =============================================================================
# TEMPORARY FINAL-TEST PAIRS
# =============================================================================

def create_final_pairs_table(
    conn,
    gold: pd.DataFrame,
):

    create_sql = """
        CREATE TEMP TABLE final_test_pairs (
            article_id BIGINT NOT NULL,
            theme_id INTEGER NOT NULL,
            PRIMARY KEY (article_id, theme_id)
        )
        ON COMMIT DROP;
    """

    pairs = [
        (
            int(row.article_id),
            int(row.theme_id),
        )
        for row in gold.itertuples(
            index=False
        )
    ]

    insert_sql = """
        INSERT INTO final_test_pairs
            (article_id, theme_id)
        VALUES %s;
    """

    with conn.cursor() as cur:

        cur.execute(
            create_sql
        )

        execute_values(
            cur,
            insert_sql,
            pairs,
            page_size=500,
        )

        cur.execute(
            """
            SELECT COUNT(*)
            FROM final_test_pairs;
            """
        )

        n_pairs = int(
            cur.fetchone()[0]
        )

    if n_pairs != EXPECTED_ROWS:
        raise RuntimeError(
            "Temporary final-test pair table "
            "does not contain the expected number "
            f"of observations: {n_pairs}."
        )


# =============================================================================
# SCORE EXTRACTION
# =============================================================================

def extract_existing_scores(
    conn,
) -> pd.DataFrame:

    sql = f"""
        SELECT
            s.article_id,
            s.theme_id,
            s.method,
            s.model_version,
            s.{SCORE_COLUMN} AS score
        FROM {SCHEMA}.{SCORES_TABLE} AS s
        INNER JOIN final_test_pairs AS f
            ON f.article_id = s.article_id
           AND f.theme_id = s.theme_id
        WHERE
            (
                s.method = %s
                AND s.model_version = %s
            )
            OR
            (
                s.method = %s
                AND s.model_version = %s
            )
        ORDER BY
            s.article_id,
            s.theme_id,
            s.method,
            s.model_version;
    """

    return pd.read_sql_query(
        sql,
        conn,
        params=(
            KEYWORD_METHOD,
            KEYWORD_MODEL_VERSION,
            WEAK_METHOD,
            WEAK_MODEL_VERSION,
        ),
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_scores(
    scores: pd.DataFrame,
):

    expected_total = (
        EXPECTED_ROWS
        * len(EXPECTED_MODELS)
    )

    if len(scores) != expected_total:
        raise RuntimeError(
            "Unexpected number of extracted scores.\n"
            f"Expected: {expected_total}\n"
            f"Found:    {len(scores)}"
        )

    if scores[
        [
            "article_id",
            "theme_id",
            "method",
            "model_version",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate stored scores detected for the "
            "same article-theme-method-model combination."
        )

    if scores["score"].isna().any():
        raise RuntimeError(
            "Missing stored scores detected."
        )

    score_values = pd.to_numeric(
        scores["score"],
        errors="coerce",
    )

    if score_values.isna().any():
        raise RuntimeError(
            "Non-numeric stored scores detected."
        )

    if not np.isfinite(
        score_values.to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            "Non-finite stored scores detected."
        )

    if (
        (score_values < 0.0)
        | (score_values > 1.0)
    ).any():

        raise RuntimeError(
            "Stored scores outside [0, 1] detected."
        )

    scores["score"] = (
        score_values.astype(float)
    )

    for (
        method,
        expected_version,
    ) in EXPECTED_MODELS.items():

        subset = scores[
            scores["method"] == method
        ]

        if len(subset) != EXPECTED_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_ROWS} scores for "
                f"method='{method}', found {len(subset)}."
            )

        actual_versions = set(
            subset[
                "model_version"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        if actual_versions != {
            expected_version
        }:
            raise RuntimeError(
                f"Unexpected model version for "
                f"method='{method}'.\n"
                f"Expected: {expected_version!r}\n"
                f"Found:    {sorted(actual_versions)}"
            )

        n_pairs = (
            subset[
                ["article_id", "theme_id"]
            ]
            .drop_duplicates()
            .shape[0]
        )

        if n_pairs != EXPECTED_ROWS:
            raise RuntimeError(
                f"Incomplete article-theme coverage for "
                f"method='{method}'.\n"
                f"Expected: {EXPECTED_ROWS}\n"
                f"Found:    {n_pairs}"
            )


# =============================================================================
# WIDE FORMAT
# =============================================================================

def build_comparison_file(
    gold: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:

    score_wide = (
        scores.pivot(
            index=[
                "article_id",
                "theme_id",
            ],
            columns="method",
            values="score",
        )
        .reset_index()
    )

    score_wide.columns.name = None

    score_wide = score_wide.rename(
        columns={
            KEYWORD_METHOD:
                "keyword_score",
            WEAK_METHOD:
                "weak_score",
        }
    )

    required_score_columns = {
        "keyword_score",
        "weak_score",
    }

    missing = (
        required_score_columns
        - set(score_wide.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing score columns after pivot: "
            f"{sorted(missing)}"
        )

    result = gold.merge(
        score_wide,
        on=[
            "article_id",
            "theme_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(result) != EXPECTED_ROWS:
        raise RuntimeError(
            "Merge changed the frozen final-test row count."
        )

    if result[
        [
            "keyword_score",
            "weak_score",
        ]
    ].isna().any().any():

        raise RuntimeError(
            "Missing keyword or weak-supervision scores "
            "after the final merge."
        )

    return result


# =============================================================================
# REPORTING
# =============================================================================

def print_coverage(
    scores: pd.DataFrame,
):

    coverage = (
        scores.groupby(
            [
                "method",
                "model_version",
            ],
            dropna=False,
        )
        .agg(
            n_scores=(
                "score",
                "size",
            ),
            n_articles=(
                "article_id",
                "nunique",
            ),
            n_themes=(
                "theme_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    print()
    print("=" * 100)
    print("COVERAGE")
    print("=" * 100)

    print(
        "\n"
        + coverage.to_string(
            index=False
        )
    )


def print_score_distributions(
    result: pd.DataFrame,
):

    print()
    print("=" * 100)
    print("SCORE DISTRIBUTIONS")
    print("=" * 100)

    for column in [
        "keyword_score",
        "weak_score",
    ]:

        values = result[column]

        print(
            f"\n{column}"
        )

        print(
            f"  min:    "
            f"{values.min():.6f}"
        )

        print(
            f"  mean:   "
            f"{values.mean():.6f}"
        )

        print(
            f"  median: "
            f"{values.median():.6f}"
        )

        print(
            f"  max:    "
            f"{values.max():.6f}"
        )

        print(
            f"  unique: "
            f"{values.nunique():,}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "EXTRACT EXISTING KEYWORD + WEAK SCORES "
        "FOR FROZEN FINAL TEST"
    )
    print("=" * 100)

    gold = load_frozen_gold()

    print(
        f"\nFrozen observations: "
        f"{len(gold):,}"
    )

    print(
        f"Unique articles:     "
        f"{gold['article_id'].nunique():,}"
    )

    print(
        f"Unique themes:       "
        f"{gold['theme_id'].nunique():,}"
    )

    print()
    print(
        "Expected stored models:"
    )

    for method, version in (
        EXPECTED_MODELS.items()
    ):
        print(
            f"  {method} -> {version}"
        )

    print(
        "\nConnecting to PostgreSQL..."
    )

    conn = connect_db()

    try:

        conn.autocommit = False

        create_final_pairs_table(
            conn,
            gold,
        )

        print(
            "\nRetrieving already-computed scores..."
        )

        scores = extract_existing_scores(
            conn
        )

        print(
            f"Rows retrieved: "
            f"{len(scores):,}"
        )

        validate_scores(
            scores
        )

        print_coverage(
            scores
        )

        print(
            "\n[OK] Exact frozen keyword and "
            "weak-supervision coverage verified."
        )

        result = build_comparison_file(
            gold,
            scores,
        )

        print_score_distributions(
            result
        )

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if OUTPUT_FILE.exists():
            raise FileExistsError(
                "Output file already exists; "
                "refusing silent overwrite:\n"
                f"{OUTPUT_FILE}"
            )

        result.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        # The temporary table is discarded with the rollback.
        # No persistent database data are modified.
        conn.rollback()

        print()
        print("=" * 100)
        print("SUCCESS")
        print("=" * 100)

        print(
            f"\nSaved:\n"
            f"{OUTPUT_FILE}"
        )

        print(
            "\nNo scores were recomputed."
        )

        print(
            "No persistent database rows "
            "were modified."
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

    try:
        main()

    except Exception as exc:

        print(
            "\nFATAL ERROR:"
        )

        print(
            repr(exc)
        )

        sys.exit(1)
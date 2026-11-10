#!/usr/bin/env python3

"""
SUPERVISED V3 — PRODUCTION DIAGNOSTIC ON 1,000 ARTICLES
=======================================================

Evaluate the fitted Supervised V3 candidate under the production scoring
regime:

    each article × all five candidate themes

This is a structural diagnostic, not final predictive evaluation and not
full-corpus scoring.

The diagnostic measures whether V3 avoids the near-perfect cross-theme
coupling previously observed for V1.

Constraints:
    - exactly 1,000 production articles;
    - full article body, without truncation;
    - five scores per article;
    - raw uncalibrated model scores retained;
    - threshold 0.34 used only for diagnostic interpretation;
    - separate database method to avoid overwriting other model outputs.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(
    os.getenv(
        "SUPERVISED_MODELS_DIR",
        "07_supervised_models",
    )
)

MODEL_FILE = (
    BASE_DIR
    / "05_v3_hybrid"
    / "production_candidate"
    / "supervised_v3_candidate.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "05_v3_hybrid"
    / "production_candidate"
    / "diagnostic_1000"
)

SCORES_FILE = (
    OUTPUT_DIR
    / "supervised_v3_diagnostic_1000_scores.csv"
)

ARTICLE_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "supervised_v3_diagnostic_1000_article_diagnostics.csv"
)

CORRELATIONS_FILE = (
    OUTPUT_DIR
    / "supervised_v3_diagnostic_1000_correlations.csv"
)

THEME_SUMMARY_FILE = (
    OUTPUT_DIR
    / "supervised_v3_diagnostic_1000_by_theme.csv"
)

TOP_RANGE_FILE = (
    OUTPUT_DIR
    / "supervised_v3_diagnostic_highest_ranges.csv"
)

LOW_RANGE_FILE = (
    OUTPUT_DIR
    / "supervised_v3_diagnostic_lowest_ranges.csv"
)

TOP_SCORES_FILE = (
    OUTPUT_DIR
    / "supervised_v3_diagnostic_top_scores_by_theme.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_ARTICLES = 1_000
EXPECTED_SCORE_ROWS = 5_000

SELECTED_THRESHOLD = 0.34

METHOD = "supervised_v3_diagnostic"
MODEL_VERSION = "supervised_v3_candidate"

SEPARATOR = "|||"

EXPECTED_FIRST_ARTICLE_ID = 859026
EXPECTED_LAST_ARTICLE_ID = 860068

EXPECTED_THEME_LABELS = [
    "AI-Related Risk",
    "Climate / Environmental Risk",
    "Cybersecurity Risk",
    "Pandemic Risk",
    "Supply Chain Risk",
]


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
    "password": (
        os.getenv("THESIS_DB_PASSWORD")
        or os.getenv("DB_PASSWORD")
    ),
}


# =============================================================================
# CUSTOM OBJECTS REQUIRED BY JOBLIB
# =============================================================================

TOKEN_PATTERN = re.compile(
    r"(?u)\b\w\w+\b"
)


def clean_text(value) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def normalize_theme(
    theme: str,
) -> str:

    theme = clean_text(
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
        SEPARATOR in clean_text(title)
        or SEPARATOR in clean_text(body)
    ):
        raise RuntimeError(
            "The frozen V3 separator occurs in article text. "
            "Scoring is stopped because document parsing would "
            "be ambiguous."
        )


def build_v3_document(
    theme_name: str,
    title: str,
    body: str,
) -> str:

    validate_separator_absence(
        title,
        body,
    )

    theme_key = normalize_theme(
        theme_name
    )

    return (
        f"{theme_key} {SEPARATOR} "
        f"{clean_text(title)} {SEPARATOR} "
        f"{clean_text(body)}"
    )


def parse_v3_document(
    document: str,
):

    parts = document.split(
        SEPARATOR,
        maxsplit=2,
    )

    if len(parts) != 3:
        raise ValueError(
            "Invalid V3 document. Expected format: "
            "'theme ||| title ||| body'."
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
    document: str,
):

    theme_key, title, body = (
        parse_v3_document(
            document
        )
    )

    text = (
        clean_text(title)
        + " "
        + clean_text(body)
    ).lower()

    tokens = TOKEN_PATTERN.findall(
        text
    )

    for token in tokens:
        yield (
            f"{theme_key}__{token}"
        )

    for i in range(
        len(tokens) - 1
    ):
        yield (
            f"{theme_key}__"
            f"{tokens[i]}_"
            f"{tokens[i + 1]}"
        )


# =============================================================================
# DATABASE
# =============================================================================

def connect_db():

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "Database password not configured. Set "
            "THESIS_DB_PASSWORD or DB_PASSWORD."
        )

    return psycopg2.connect(
        **DB_CONFIG
    )


def load_themes(
    conn,
):

    sql = """
        SELECT
            theme_id,
            theme_label
        FROM risk_emergence.candidate_themes
        WHERE is_active = TRUE
        ORDER BY theme_id;
    """

    df = pd.read_sql_query(
        sql,
        conn,
    )

    if len(df) != len(
        EXPECTED_THEME_LABELS
    ):
        raise RuntimeError(
            f"Expected {len(EXPECTED_THEME_LABELS)} active themes, "
            f"found {len(df)}."
        )

    actual_labels = sorted(
        df["theme_label"].tolist()
    )

    expected_labels = sorted(
        EXPECTED_THEME_LABELS
    )

    if actual_labels != expected_labels:
        raise RuntimeError(
            "Active database themes do not match the frozen "
            "V3 theme set.\n"
            f"Database: {actual_labels}\n"
            f"Expected: {expected_labels}"
        )

    return df


def load_diagnostic_articles(
    conn,
):

    sql = """
        SELECT
            article_id,
            title,
            body
        FROM risk_emergence.v_valid_articles
        ORDER BY article_id
        LIMIT %s;
    """

    df = pd.read_sql_query(
        sql,
        conn,
        params=(MAX_ARTICLES,),
    )

    if len(df) != MAX_ARTICLES:
        raise RuntimeError(
            f"Expected {MAX_ARTICLES} articles, "
            f"found {len(df)}."
        )

    if (
        df["article_id"].nunique()
        != MAX_ARTICLES
    ):
        raise RuntimeError(
            "Duplicate article IDs found in the diagnostic sample."
        )

    return df


def sample_fingerprint(
    articles,
) -> str:

    ordered_ids = (
        articles[
            "article_id"
        ]
        .astype("int64")
        .tolist()
    )

    payload = "\n".join(
        str(article_id)
        for article_id in ordered_ids
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# =============================================================================
# ARTICLE × THEME FRAME
# =============================================================================

def build_scoring_frame(
    articles,
    themes,
):

    rows = []

    for article in articles.itertuples(
        index=False
    ):

        for theme in themes.itertuples(
            index=False
        ):

            rows.append(
                {
                    "article_id": int(
                        article.article_id
                    ),
                    "theme_id": int(
                        theme.theme_id
                    ),
                    "theme_name": (
                        theme.theme_label
                    ),
                    "title": clean_text(
                        article.title
                    ),
                    "v3_document": (
                        build_v3_document(
                            theme_name=(
                                theme.theme_label
                            ),
                            title=article.title,
                            body=article.body,
                        )
                    ),
                }
            )

    scoring_df = pd.DataFrame(
        rows
    )

    if (
        len(scoring_df)
        != EXPECTED_SCORE_ROWS
    ):
        raise RuntimeError(
            "Unexpected number of article-theme rows."
        )

    if scoring_df[
        [
            "article_id",
            "theme_id",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate article-theme pairs detected."
        )

    return scoring_df


# =============================================================================
# DATABASE WRITE
# =============================================================================

def check_existing_rows(
    conn,
):

    sql = """
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT article_id) AS n_articles
        FROM risk_emergence.news_theme_scores
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

        n_rows, n_articles = (
            cur.fetchone()
        )

    return (
        int(n_rows),
        int(n_articles),
    )


def delete_existing_diagnostic_rows(
    conn,
):

    sql = """
        DELETE FROM risk_emergence.news_theme_scores
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

        deleted = cur.rowcount

    conn.commit()

    return int(
        deleted
    )


def write_scores(
    conn,
    scoring_df,
):

    # Verify the actual PostgreSQL column name before running.
    # This version assumes that the database column is literally
    # named "thematic scores".
    sql = """
        INSERT INTO risk_emergence.news_theme_scores
        (
            article_id,
            theme_id,
            "thematic scores",
            method,
            model_version
        )
        VALUES %s

        ON CONFLICT
            (article_id, theme_id, method)

        DO UPDATE SET
            "thematic scores" =
                EXCLUDED."thematic scores",
            model_version =
                EXCLUDED.model_version,
            computed_at = NOW();
    """

    values = [
        (
            int(row.article_id),
            int(row.theme_id),
            float(row.score),
            METHOD,
            MODEL_VERSION,
        )
        for row in scoring_df.itertuples(
            index=False
        )
    ]

    with conn.cursor() as cur:

        execute_values(
            cur,
            sql,
            values,
            page_size=1000,
        )

    conn.commit()


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def calculate_diagnostics(
    scoring_df,
):

    wide = scoring_df.pivot(
        index="article_id",
        columns="theme_name",
        values="thematic scores",
    )

    wide = wide[
        EXPECTED_THEME_LABELS
    ]

    if wide.shape != (
        MAX_ARTICLES,
        len(EXPECTED_THEME_LABELS),
    ):
        raise RuntimeError(
            "Unexpected diagnostic score matrix shape."
        )

    if wide.isna().any().any():
        raise RuntimeError(
            "Missing scores in article-theme matrix."
        )

    correlation_matrix = (
        wide.corr()
    )

    correlation_rows = []

    for i in range(
        len(EXPECTED_THEME_LABELS)
    ):
        for j in range(
            i + 1,
            len(EXPECTED_THEME_LABELS),
        ):

            theme_a = (
                EXPECTED_THEME_LABELS[i]
            )

            theme_b = (
                EXPECTED_THEME_LABELS[j]
            )

            correlation_rows.append(
                {
                    "theme_a": theme_a,
                    "theme_b": theme_b,
                    "correlation": float(
                        correlation_matrix.loc[
                            theme_a,
                            theme_b,
                        ]
                    ),
                }
            )

    correlations_df = pd.DataFrame(
        correlation_rows
    )

    article_diag = wide.copy()

    article_diag["score_min"] = (
        wide.min(axis=1)
    )

    article_diag["score_max"] = (
        wide.max(axis=1)
    )

    article_diag["score_range"] = (
        article_diag["score_max"]
        - article_diag["score_min"]
    )

    article_diag["score_std"] = (
        wide.std(
            axis=1,
            ddof=0,
        )
    )

    article_diag[
        "highest_scoring_theme"
    ] = wide.idxmax(
        axis=1
    )

    article_diag[
        "highest_score"
    ] = wide.max(
        axis=1
    )

    article_diag = (
        article_diag.reset_index()
    )

    titles = (
        scoring_df[
            [
                "article_id",
                "title",
            ]
        ]
        .drop_duplicates(
            "article_id"
        )
    )

    article_diag = article_diag.merge(
        titles,
        on="article_id",
        how="left",
        validate="one_to_one",
    )

    theme_summary = (
        scoring_df
        .groupby(
            "theme_name"
        )
        .agg(
            n=(
                "thematic scores",
                "size",
            ),
            mean_score=(
                "thematic scores",
                "mean",
            ),
            median_score=(
                "thematic scores",
                "median",
            ),
            min_score=(
                "thematic scores",
                "min",
            ),
            max_score=(
                "thematic scores",
                "max",
            ),
            std_score=(
                "thematic scores",
                "std",
            ),
            positive_at_034=(
                "prediction_at_034",
                "sum",
            ),
        )
        .reset_index()
    )

    theme_summary[
        "positive_rate_at_034"
    ] = (
        theme_summary[
            "positive_at_034"
        ]
        / theme_summary["n"]
    )

    highest_ranges = (
        article_diag
        .sort_values(
            "score_range",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    lowest_ranges = (
        article_diag
        .sort_values(
            "score_range",
            ascending=True,
        )
        .head(20)
        .copy()
    )

    top_frames = []

    for theme in EXPECTED_THEME_LABELS:

        subset = (
            scoring_df[
                scoring_df[
                    "theme_name"
                ] == theme
            ]
            .sort_values(
                "thematic scores",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        subset[
            "rank_within_theme"
        ] = np.arange(
            1,
            len(subset) + 1,
        )

        top_frames.append(
            subset
        )

    top_scores = pd.concat(
        top_frames,
        ignore_index=True,
    )

    return (
        correlations_df,
        article_diag,
        theme_summary,
        highest_ranges,
        lowest_ranges,
        top_scores,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "SUPERVISED V3 — "
        "1,000-ARTICLE PRODUCTION DIAGNOSTIC"
    )
    print("=" * 110)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    print(
        "\nLoading fitted V3 candidate..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "[OK] model loaded"
    )

    print(
        "\nConnecting to database..."
    )

    conn = connect_db()

    print(
        "[OK] database connected"
    )

    try:

        themes = load_themes(
            conn
        )

        print(
            "\nActive themes:"
        )

        print(
            themes.to_string(
                index=False
            )
        )

        print(
            f"\nLoading first "
            f"{MAX_ARTICLES:,} valid articles..."
        )

        t0 = time.time()

        articles = load_diagnostic_articles(
            conn
        )

        print(
            f"[OK] {len(articles):,} articles loaded "
            f"in {time.time() - t0:.2f}s"
        )

        current_min = int(
            articles[
                "article_id"
            ].min()
        )

        current_max = int(
            articles[
                "article_id"
            ].max()
        )

        print(
            f"Article ID range: "
            f"{current_min:,} -> "
            f"{current_max:,}"
        )

        if (
            current_min
            != EXPECTED_FIRST_ARTICLE_ID
            or current_max
            != EXPECTED_LAST_ARTICLE_ID
        ):
            raise RuntimeError(
                "Diagnostic sample range differs from the "
                "previous V1 sample.\n"
                f"Current:  {current_min} -> {current_max}\n"
                f"Expected: {EXPECTED_FIRST_ARTICLE_ID} -> "
                f"{EXPECTED_LAST_ARTICLE_ID}"
            )

        fingerprint = sample_fingerprint(
            articles
        )

        print(
            f"[OK] sample SHA-256: "
            f"{fingerprint}"
        )

        print(
            "\nBuilding article-theme documents..."
        )

        scoring_df = build_scoring_frame(
            articles,
            themes,
        )

        print(
            f"[OK] {len(scoring_df):,} "
            "article-theme documents"
        )

        print(
            "\nScoring V3..."
        )

        t0 = time.time()

        probabilities = model.predict_proba(
            scoring_df[
                "v3_document"
            ]
        )[:, 1]

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

        scoring_df[
            "thematic scores"
        ] = probabilities

        scoring_df[
            "prediction_at_034"
        ] = (
            probabilities
            >= SELECTED_THRESHOLD
        ).astype(int)

        scoring_seconds = (
            time.time() - t0
        )

        print(
            f"[OK] {len(scoring_df):,} scores "
            f"in {scoring_seconds:.2f}s"
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

        n_positive = int(
            scoring_df[
                "prediction_at_034"
            ].sum()
        )

        positive_rate = (
            n_positive
            / len(scoring_df)
        )

        print(
            f"\n>= {SELECTED_THRESHOLD:.2f}: "
            f"{n_positive:,} / "
            f"{len(scoring_df):,} "
            f"({positive_rate:.2%})"
        )

        (
            correlations_df,
            article_diag,
            theme_summary,
            highest_ranges,
            lowest_ranges,
            top_scores,
        ) = calculate_diagnostics(
            scoring_df
        )

        mean_corr = float(
            correlations_df[
                "correlation"
            ].mean()
        )

        min_corr = float(
            correlations_df[
                "correlation"
            ].min()
        )

        max_corr = float(
            correlations_df[
                "correlation"
            ].max()
        )

        mean_range = float(
            article_diag[
                "score_range"
            ].mean()
        )

        median_range = float(
            article_diag[
                "score_range"
            ].median()
        )

        mean_std = float(
            article_diag[
                "score_std"
            ].mean()
        )

        print()
        print("=" * 110)
        print(
            "SCORE DISTRIBUTION BY THEME"
        )
        print("=" * 110)

        print(
            theme_summary.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.6f}"
                ),
            )
        )

        print()
        print("=" * 110)
        print(
            "CROSS-THEME CORRELATIONS"
        )
        print("=" * 110)

        print(
            correlations_df.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.6f}"
                ),
            )
        )

        print(
            f"\nMean pairwise correlation: "
            f"{mean_corr:.6f}"
        )

        print(
            f"Min pairwise correlation:  "
            f"{min_corr:.6f}"
        )

        print(
            f"Max pairwise correlation:  "
            f"{max_corr:.6f}"
        )

        print()
        print("=" * 110)
        print(
            "WITHIN-ARTICLE THEME SENSITIVITY"
        )
        print("=" * 110)

        print(
            f"\nMean score range:   "
            f"{mean_range:.6f}"
        )

        print(
            f"Median score range: "
            f"{median_range:.6f}"
        )

        print(
            f"Mean score std:     "
            f"{mean_std:.6f}"
        )

        export_scores = scoring_df[
            [
                "article_id",
                "theme_id",
                "theme_name",
                "title",
                "thematic scores",
                "prediction_at_034",
            ]
        ].copy()

        export_scores.to_csv(
            SCORES_FILE,
            index=False,
        )

        article_diag.to_csv(
            ARTICLE_DIAGNOSTICS_FILE,
            index=False,
        )

        correlations_df.to_csv(
            CORRELATIONS_FILE,
            index=False,
        )

        theme_summary.to_csv(
            THEME_SUMMARY_FILE,
            index=False,
        )

        highest_ranges.to_csv(
            TOP_RANGE_FILE,
            index=False,
        )

        lowest_ranges.to_csv(
            LOW_RANGE_FILE,
            index=False,
        )

        top_scores[
            [
                "article_id",
                "theme_id",
                "theme_name",
                "title",
                "thematic scores",
                "prediction_at_034",
                "rank_within_theme",
            ]
        ].to_csv(
            TOP_SCORES_FILE,
            index=False,
        )

        existing_rows, existing_articles = (
            check_existing_rows(
                conn
            )
        )

        print()
        print("=" * 110)
        print(
            "DATABASE DIAGNOSTIC OUTPUT"
        )
        print("=" * 110)

        print(
            f"\nmethod        = {METHOD}"
        )

        print(
            f"model_version = {MODEL_VERSION}"
        )

        print(
            f"Existing rows: "
            f"{existing_rows:,}"
        )

        print(
            f"Existing articles: "
            f"{existing_articles:,}"
        )

        if existing_rows > 0:

            deleted = (
                delete_existing_diagnostic_rows(
                    conn
                )
            )

            print(
                f"[OK] deleted "
                f"{deleted:,} previous rows"
            )

        print(
            f"\nWriting "
            f"{EXPECTED_SCORE_ROWS:,} "
            "diagnostic scores..."
        )

        t0 = time.time()

        write_scores(
            conn,
            scoring_df,
        )

        print(
            f"[OK] database write "
            f"in {time.time() - t0:.2f}s"
        )

        final_rows, final_articles = (
            check_existing_rows(
                conn
            )
        )

        if (
            final_rows
            != EXPECTED_SCORE_ROWS
            or final_articles
            != MAX_ARTICLES
        ):
            raise RuntimeError(
                "Final database diagnostic counts are incorrect."
            )

        print()
        print("=" * 110)
        print(
            "DIAGNOSTIC SUMMARY"
        )
        print("=" * 110)

        print(
            f"""
Articles:
  {MAX_ARTICLES:,}

Article-theme scores:
  {len(scoring_df):,}

Diagnostic sample SHA-256:
  {fingerprint}

Threshold:
  {SELECTED_THRESHOLD:.2f}

Positive rate at threshold:
  {positive_rate:.2%}

Cross-theme correlations:
  mean = {mean_corr:.6f}
  min  = {min_corr:.6f}
  max  = {max_corr:.6f}

Within-article theme sensitivity:
  mean range   = {mean_range:.6f}
  median range = {median_range:.6f}
  mean std     = {mean_std:.6f}

Reference — V1 production diagnostic:
  pairwise correlations were approximately 0.98–1.00.

Reference — V3 OOF counterfactual diagnostic:
  mean score range   = 0.254409
  median score range = 0.258506
  mean score std     = 0.097765

This diagnostic does not estimate predictive performance because
the production articles do not have gold labels for all five themes.

It evaluates the structural behavior of V3 when the same article is
scored against different candidate themes.
"""
        )

        print()
        print("=" * 110)
        print(
            "FILES SAVED"
        )
        print("=" * 110)

        for path in [
            SCORES_FILE,
            ARTICLE_DIAGNOSTICS_FILE,
            CORRELATIONS_FILE,
            THEME_SUMMARY_FILE,
            TOP_RANGE_FILE,
            LOW_RANGE_FILE,
            TOP_SCORES_FILE,
        ]:
            print(
                path
            )

        print()
        print(
            "Diagnostic completed. "
            "Do not run full-corpus scoring before "
            "the planned final-test evaluation."
        )

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
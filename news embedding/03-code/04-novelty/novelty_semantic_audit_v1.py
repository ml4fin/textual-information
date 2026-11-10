#!/usr/bin/env python3

"""
STEP 4 — NOVELTY SEMANTIC AUDIT
===============================

Purpose
-------
Qualitatively inspect the semantic meaning of the novelty measure at
different points of its empirical distribution.

Input:
    novelty_calibration_pilot_v1.csv

Diagnostic specification:
    historical window = 365 days
    K = 10 nearest historical neighbours

Embedding model:
    all-MiniLM-L6-v2_chunk254_mean_l2_v1

Audit quantiles:
    P01, P05, P25, P50, P75, P95, P99

For each selected article, the script:
1. retrieves the stored document embedding;
2. reconstructs the exact strictly historical 365-day candidate set;
3. recomputes exact cosine similarities;
4. identifies the top-K historical neighbours;
5. verifies the novelty score against the calibration CSV;
6. prints the target and neighbour titles for qualitative inspection.

No approximate FAISS search is used.

No database data are modified.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psycopg2


# =============================================================================
# PARAMETERS
# =============================================================================

CSV_FILE = Path(
    os.getenv(
        "NOVELTY_CALIBRATION_FILE",
        "novelty_calibration_pilot_v1.csv",
    )
)

MODEL_NAME = (
    "all-MiniLM-L6-v2_"
    "chunk254_mean_l2_v1"
)

EXPECTED_DIMENSION = 384
EXPECTED_CALIBRATION_ROWS = 1_080

WINDOW_DAYS = 365
K = 10

QUANTILES = [
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99,
]

NOVELTY_TOLERANCE = 1e-5
NORM_TOLERANCE = 1e-5


# =============================================================================
# DATABASE
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


def get_connection():

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "Database password not configured. "
            "Set THESIS_DB_PASSWORD or DB_PASSWORD."
        )

    conn = psycopg2.connect(
        **DB_CONFIG
    )

    conn.set_session(
        readonly=True,
        autocommit=False,
    )

    return conn


# =============================================================================
# SQL
# =============================================================================

TARGET_QUERY = """
SELECT
    a.article_id,
    a.publication_date,
    a.source_code,
    a.title,
    e.embedding
FROM risk_emergence.v_valid_articles AS a
JOIN risk_emergence.news_embeddings AS e
    ON e.article_id = a.article_id
WHERE a.article_id = %s
  AND e.model_name = %s;
"""


HISTORY_QUERY = """
SELECT
    a.article_id,
    a.publication_date,
    a.source_code,
    a.title,
    e.embedding
FROM risk_emergence.v_valid_articles AS a
JOIN risk_emergence.news_embeddings AS e
    ON e.article_id = a.article_id
WHERE e.model_name = %s
  AND a.publication_date >= %s
  AND a.publication_date < %s
ORDER BY
    a.publication_date,
    a.article_id;
"""


# =============================================================================
# CALIBRATION DATA
# =============================================================================

def load_calibration_data():

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: "
            f"{CSV_FILE}"
        )

    df = pd.read_csv(
        CSV_FILE
    )

    required_columns = {
        "article_id",
        "window_days",
        "k",
        "novelty_mean_top_k",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing calibration columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    spec = df.loc[
        (df["window_days"] == WINDOW_DAYS)
        & (df["k"] == K)
    ].copy()

    if (
        len(spec)
        != EXPECTED_CALIBRATION_ROWS
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_CALIBRATION_ROWS:,} "
            f"calibration rows for "
            f"W={WINDOW_DAYS}, K={K}; "
            f"found {len(spec):,}."
        )

    if spec[
        [
            "article_id",
            "novelty_mean_top_k",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Missing values detected in "
            "calibration data."
        )

    spec["article_id"] = pd.to_numeric(
        spec["article_id"],
        errors="raise",
    )

    spec[
        "novelty_mean_top_k"
    ] = pd.to_numeric(
        spec[
            "novelty_mean_top_k"
        ],
        errors="raise",
    )

    if not np.isfinite(
        spec[
            "novelty_mean_top_k"
        ].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "Non-finite novelty values "
            "in calibration data."
        )

    if spec[
        "article_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate article IDs detected "
            "for the selected calibration "
            "specification."
        )

    return (
        spec.sort_values(
            [
                "novelty_mean_top_k",
                "article_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def select_quantile_articles(
    spec,
):

    selected = []

    used_article_ids = set()

    for quantile in QUANTILES:

        quantile_value = float(
            spec[
                "novelty_mean_top_k"
            ].quantile(
                quantile
            )
        )

        distances = np.abs(
            spec[
                "novelty_mean_top_k"
            ].to_numpy(
                dtype=np.float64
            )
            - quantile_value
        )

        order = np.lexsort(
            (
                spec[
                    "article_id"
                ].to_numpy(
                    dtype=np.int64
                ),
                distances,
            )
        )

        selected_index = None

        for idx in order:

            article_id = int(
                spec.iloc[
                    idx
                ][
                    "article_id"
                ]
            )

            if (
                article_id
                not in used_article_ids
            ):
                selected_index = int(
                    idx
                )
                break

        if selected_index is None:
            raise RuntimeError(
                f"Could not select a unique "
                f"article for quantile "
                f"{quantile:.2f}."
            )

        row = spec.iloc[
            selected_index
        ].copy()

        article_id = int(
            row[
                "article_id"
            ]
        )

        used_article_ids.add(
            article_id
        )

        selected.append(
            (
                quantile,
                quantile_value,
                row,
            )
        )

    return selected


# =============================================================================
# EMBEDDING VALIDATION
# =============================================================================

def validate_vector(
    vector,
    label,
):

    if vector.shape != (
        EXPECTED_DIMENSION,
    ):
        raise RuntimeError(
            f"{label}: unexpected shape "
            f"{vector.shape}."
        )

    if not np.isfinite(
        vector
    ).all():
        raise RuntimeError(
            f"{label}: NaN or Inf detected."
        )

    norm = float(
        np.linalg.norm(
            vector
        )
    )

    if not np.isclose(
        norm,
        1.0,
        atol=NORM_TOLERANCE,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{label}: embedding is not "
            f"L2 normalized; norm={norm:.8f}."
        )

    return norm


def validate_matrix(
    vectors,
    label,
):

    if (
        vectors.ndim != 2
        or vectors.shape[1]
        != EXPECTED_DIMENSION
    ):
        raise RuntimeError(
            f"{label}: unexpected shape "
            f"{vectors.shape}."
        )

    if not np.isfinite(
        vectors
    ).all():
        raise RuntimeError(
            f"{label}: NaN or Inf detected."
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=NORM_TOLERANCE,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{label}: historical embeddings "
            "are not consistently "
            "L2 normalized."
        )

    return (
        float(
            norms.min()
        ),
        float(
            norms.max()
        ),
    )


# =============================================================================
# DATABASE RETRIEVAL
# =============================================================================

def load_target(
    conn,
    article_id,
):

    with conn.cursor() as cur:

        cur.execute(
            TARGET_QUERY,
            (
                article_id,
                MODEL_NAME,
            ),
        )

        rows = cur.fetchall()

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one target "
            f"row for article_id="
            f"{article_id}; "
            f"found {len(rows)}."
        )

    (
        target_id,
        target_date,
        target_source,
        target_title,
        target_embedding,
    ) = rows[0]

    target_vector = np.asarray(
        target_embedding,
        dtype=np.float32,
    )

    target_norm = validate_vector(
        target_vector,
        label=(
            f"target {article_id}"
        ),
    )

    return {
        "article_id":
            int(target_id),
        "date":
            target_date,
        "source":
            target_source,
        "title":
            target_title,
        "vector":
            target_vector,
        "norm":
            target_norm,
    }


def load_history(
    conn,
    target_date,
):

    history_start = (
        target_date
        - timedelta(
            days=WINDOW_DAYS
        )
    )

    with conn.cursor() as cur:

        cur.execute(
            HISTORY_QUERY,
            (
                MODEL_NAME,
                history_start,
                target_date,
            ),
        )

        rows = cur.fetchall()

    if len(rows) < K:
        raise RuntimeError(
            f"Only {len(rows)} historical "
            f"candidates for target date "
            f"{target_date}."
        )

    ids = np.asarray(
        [
            row[0]
            for row
            in rows
        ],
        dtype=np.int64,
    )

    dates = np.asarray(
        [
            row[1]
            for row
            in rows
        ],
        dtype=object,
    )

    sources = np.asarray(
        [
            row[2]
            for row
            in rows
        ],
        dtype=object,
    )

    titles = [
        row[3]
        for row
        in rows
    ]

    vectors = np.asarray(
        [
            row[4]
            for row
            in rows
        ],
        dtype=np.float32,
    )

    if (
        len(
            np.unique(
                ids
            )
        )
        != len(ids)
    ):
        raise RuntimeError(
            "Duplicate historical article "
            "IDs detected."
        )

    norm_min, norm_max = (
        validate_matrix(
            vectors,
            label=(
                f"history for "
                f"{target_date}"
            ),
        )
    )

    return {
        "start_date":
            history_start,
        "ids":
            ids,
        "dates":
            dates,
        "sources":
            sources,
        "titles":
            titles,
        "vectors":
            vectors,
        "norm_min":
            norm_min,
        "norm_max":
            norm_max,
    }


# =============================================================================
# EXACT NOVELTY
# =============================================================================

def calculate_exact_novelty(
    target_vector,
    history,
):

    similarities = (
        history["vectors"]
        @ target_vector
    )

    similarities = np.clip(
        similarities,
        -1.0,
        1.0,
    )

    if not np.isfinite(
        similarities
    ).all():
        raise RuntimeError(
            "Non-finite similarities "
            "detected."
        )

    # Primary key: descending cosine similarity.
    # Secondary key: ascending article_id.
    order = np.lexsort(
        (
            history["ids"],
            -similarities,
        )
    )

    top_idx = order[
        :K
    ]

    top_sims = similarities[
        top_idx
    ]

    novelty = (
        1.0
        - float(
            np.mean(
                top_sims,
                dtype=np.float64,
            )
        )
    )

    return (
        novelty,
        similarities,
        top_idx,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    start = perf_counter()

    print("=" * 110)
    print(
        "STEP 4 — NOVELTY "
        "SEMANTIC AUDIT"
    )
    print("=" * 110)

    print(
        f"Calibration file     : "
        f"{CSV_FILE}"
    )

    print(
        f"Embedding model      : "
        f"{MODEL_NAME}"
    )

    print(
        f"Window               : "
        f"{WINDOW_DAYS} days"
    )

    print(
        f"K                    : "
        f"{K}"
    )

    print(
        f"Quantiles            : "
        + ", ".join(
            f"P{int(q * 100):02d}"
            for q
            in QUANTILES
        )
    )

    # =========================================================================
    # CALIBRATION SAMPLE
    # =========================================================================

    spec = load_calibration_data()

    selected = (
        select_quantile_articles(
            spec
        )
    )

    print(
        f"Calibration rows     : "
        f"{len(spec):,}"
    )

    print(
        f"Audit articles       : "
        f"{len(selected)}"
    )

    # =========================================================================
    # DATABASE
    # =========================================================================

    conn = get_connection()

    max_difference = 0.0

    try:

        for audit_number, (
            quantile,
            quantile_value,
            calibration_row,
        ) in enumerate(
            selected,
            start=1,
        ):

            article_id = int(
                calibration_row[
                    "article_id"
                ]
            )

            target = load_target(
                conn,
                article_id,
            )

            history = load_history(
                conn,
                target["date"],
            )

            (
                recalculated_novelty,
                similarities,
                top_idx,
            ) = calculate_exact_novelty(
                target_vector=
                    target["vector"],
                history=history,
            )

            stored_novelty = float(
                calibration_row[
                    "novelty_mean_top_k"
                ]
            )

            difference = abs(
                recalculated_novelty
                - stored_novelty
            )

            max_difference = max(
                max_difference,
                difference,
            )

            print()
            print("=" * 110)

            print(
                f"AUDIT "
                f"{audit_number}/"
                f"{len(selected)} "
                f"| P"
                f"{int(quantile * 100):02d}"
            )

            print(
                f"Quantile value       : "
                f"{quantile_value:.8f}"
            )

            print(
                f"Article ID           : "
                f"{target['article_id']}"
            )

            print(
                f"Date                 : "
                f"{target['date']}"
            )

            print(
                f"Source               : "
                f"{target['source']}"
            )

            print(
                f"Title                : "
                f"{target['title']}"
            )

            print(
                f"Target norm          : "
                f"{target['norm']:.8f}"
            )

            print(
                f"History              : "
                f"{history['start_date']} "
                f"<= date < "
                f"{target['date']}"
            )

            print(
                f"Candidates           : "
                f"{len(history['ids']):,}"
            )

            print(
                f"History norms        : "
                f"{history['norm_min']:.8f} "
                f"-> "
                f"{history['norm_max']:.8f}"
            )

            print(
                f"Stored novelty       : "
                f"{stored_novelty:.8f}"
            )

            print(
                f"Recalculated novelty : "
                f"{recalculated_novelty:.8f}"
            )

            print(
                f"Absolute difference  : "
                f"{difference:.10f}"
            )

            if (
                difference
                > NOVELTY_TOLERANCE
            ):
                raise RuntimeError(
                    f"Novelty parity failed "
                    f"for article_id="
                    f"{article_id}: "
                    f"difference="
                    f"{difference:.10f}."
                )

            print()
            print(
                f"TOP {K} HISTORICAL "
                "NEIGHBOURS"
            )
            print("-" * 110)

            for rank, idx in enumerate(
                top_idx,
                start=1,
            ):

                print(
                    f"{rank:>2}. "
                    f"sim="
                    f"{similarities[idx]:.8f} "
                    f"| article_id="
                    f"{history['ids'][idx]} "
                    f"| "
                    f"{history['dates'][idx]} "
                    f"| "
                    f"{history['sources'][idx]}"
                )

                print(
                    f"    "
                    f"{history['titles'][idx]}"
                )

    finally:

        conn.rollback()
        conn.close()

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 110)
    print("AUDIT SUMMARY")
    print("=" * 110)

    print(
        f"Articles audited     : "
        f"{len(selected)}"
    )

    print(
        f"Maximum parity error : "
        f"{max_difference:.10f}"
    )

    print(
        f"Parity tolerance     : "
        f"{NOVELTY_TOLERANCE:.10f}"
    )

    print(
        f"Runtime              : "
        f"{perf_counter() - start:.3f} s"
    )

    print()
    print(
        "NOVELTY RECOMPUTATION: PASS"
    )

    print(
        "DATABASE STATUS: READ ONLY"
    )

    print("=" * 110)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nInterrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print(
            "\nFATAL ERROR:"
        )

        print(
            repr(exc)
        )

        sys.exit(1)
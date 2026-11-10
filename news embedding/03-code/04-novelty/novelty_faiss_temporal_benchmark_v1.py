#!/usr/bin/env python3

"""
STEP 4 — FAISS HNSW TEMPORAL VALIDATION
========================================

Purpose
-------
Validate the candidate HNSW configuration across multiple dates spanning
the corpus.

Embedding model:
    all-MiniLM-L6-v2_chunk254_mean_l2_v1

Novelty definition:
    1 - mean cosine similarity to the K = 10 nearest historical articles.

Historical candidate window:
    [target_date - 365 days, target_date)

Target sampling:
    20 deterministic articles per source per benchmark date,
    for 100 targets per date.

Exact reference:
    FAISS IndexFlatIP.

Candidate approximate configuration:
    M = 48
    efConstruction = 400
    efSearch = 128

Evaluation:
    Recall@10 against exact nearest neighbours
    fraction of targets with perfect Recall@10
    absolute and signed novelty-score error
    HNSW build and batched-query runtime

This script performs validation only.
No database data are modified.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from time import perf_counter

import faiss
import numpy as np
import psycopg2


# =============================================================================
# PARAMETERS
# =============================================================================

MODEL_NAME = (
    "all-MiniLM-L6-v2_"
    "chunk254_mean_l2_v1"
)

EXPECTED_DIMENSION = 384

BENCHMARK_DATES = [
    date(2015, 6, 1),
    date(2016, 6, 30),
    date(2017, 2, 16),
    date(2018, 5, 6),
    date(2019, 4, 25),
    date(2020, 8, 30),
    date(2021, 5, 22),
    date(2022, 10, 10),
    date(2023, 4, 4),
]

WINDOW_DAYS = 365
K = 10

EXPECTED_SOURCES = 5
TARGETS_PER_SOURCE = 20

EXPECTED_TARGETS_PER_DATE = (
    EXPECTED_SOURCES
    * TARGETS_PER_SOURCE
)

FETCH_SIZE = 5_000

HNSW_M = 48
HNSW_EF_CONSTRUCTION = 400
HNSW_EF_SEARCH = 128

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
WITH ranked AS (
    SELECT
        a.article_id,
        a.publication_date,
        a.source_code,
        a.title,
        e.embedding,
        ROW_NUMBER() OVER (
            PARTITION BY a.source_code
            ORDER BY md5(a.article_id::text)
        ) AS rn
    FROM risk_emergence.v_valid_articles AS a
    JOIN risk_emergence.news_embeddings AS e
        ON e.article_id = a.article_id
    WHERE e.model_name = %s
      AND a.publication_date = %s
)
SELECT
    article_id,
    publication_date,
    source_code,
    title,
    embedding
FROM ranked
WHERE rn <= %s
ORDER BY
    source_code,
    rn;
"""


HISTORY_QUERY = """
SELECT
    a.article_id,
    a.publication_date,
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
# VALIDATION
# =============================================================================

def validate_embedding_matrix(
    vectors,
    expected_rows,
    label,
):

    expected_shape = (
        expected_rows,
        EXPECTED_DIMENSION,
    )

    if vectors.shape != expected_shape:
        raise RuntimeError(
            f"{label}: unexpected shape "
            f"{vectors.shape}; expected "
            f"{expected_shape}."
        )

    if vectors.dtype != np.float32:
        raise RuntimeError(
            f"{label}: expected float32, "
            f"found {vectors.dtype}."
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
            f"{label}: embeddings are not "
            "consistently L2 normalized."
        )

    return (
        float(norms.min()),
        float(norms.max()),
    )


def validate_search_output(
    distances,
    indices,
    n_queries,
    label,
):

    expected_shape = (
        n_queries,
        K,
    )

    if distances.shape != expected_shape:
        raise RuntimeError(
            f"{label}: unexpected distance "
            f"shape {distances.shape}."
        )

    if indices.shape != expected_shape:
        raise RuntimeError(
            f"{label}: unexpected index "
            f"shape {indices.shape}."
        )

    if not np.isfinite(
        distances
    ).all():
        raise RuntimeError(
            f"{label}: non-finite "
            "similarities detected."
        )

    if np.any(
        indices < 0
    ):
        raise RuntimeError(
            f"{label}: invalid FAISS "
            "index detected."
        )


# =============================================================================
# LOAD TARGETS
# =============================================================================

def load_targets(
    conn,
    target_date,
):

    with conn.cursor() as cur:

        cur.execute(
            TARGET_QUERY,
            (
                MODEL_NAME,
                target_date,
                TARGETS_PER_SOURCE,
            ),
        )

        rows = cur.fetchall()

    if len(
        rows
    ) != EXPECTED_TARGETS_PER_DATE:
        raise RuntimeError(
            f"{target_date}: expected "
            f"{EXPECTED_TARGETS_PER_DATE} "
            f"targets, found {len(rows)}."
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

    sources = [
        row[2]
        for row
        in rows
    ]

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

    if len(
        np.unique(
            ids
        )
    ) != EXPECTED_TARGETS_PER_DATE:
        raise RuntimeError(
            f"{target_date}: duplicate "
            "target article IDs."
        )

    if not np.all(
        dates == target_date
    ):
        raise RuntimeError(
            f"{target_date}: target-date "
            "validation failed."
        )

    source_counts = {
        source: sources.count(
            source
        )
        for source
        in sorted(
            set(
                sources
            )
        )
    }

    if (
        len(source_counts)
        != EXPECTED_SOURCES
        or any(
            count
            != TARGETS_PER_SOURCE
            for count
            in source_counts.values()
        )
    ):
        raise RuntimeError(
            f"{target_date}: invalid source "
            f"balance {source_counts}."
        )

    norm_min, norm_max = (
        validate_embedding_matrix(
            vectors=vectors,
            expected_rows=
                EXPECTED_TARGETS_PER_DATE,
            label=(
                f"{target_date} targets"
            ),
        )
    )

    return {
        "ids": ids,
        "sources": sources,
        "titles": titles,
        "vectors": vectors,
        "source_counts": source_counts,
        "norm_min": norm_min,
        "norm_max": norm_max,
    }


# =============================================================================
# LOAD HISTORY
# =============================================================================

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

    ids_parts = []
    vectors_parts = []

    cursor_name = (
        "history_"
        + target_date.strftime(
            "%Y%m%d"
        )
    )

    stream = conn.cursor(
        name=cursor_name
    )

    stream.itersize = FETCH_SIZE

    total_rows = 0

    try:

        stream.execute(
            HISTORY_QUERY,
            (
                MODEL_NAME,
                history_start,
                target_date,
            ),
        )

        while True:

            rows = stream.fetchmany(
                FETCH_SIZE
            )

            if not rows:
                break

            ids_parts.append(
                np.asarray(
                    [
                        row[0]
                        for row
                        in rows
                    ],
                    dtype=np.int64,
                )
            )

            vectors_parts.append(
                np.asarray(
                    [
                        row[2]
                        for row
                        in rows
                    ],
                    dtype=np.float32,
                )
            )

            total_rows += len(
                rows
            )

    finally:

        stream.close()

    if total_rows < K:
        raise RuntimeError(
            f"{target_date}: only "
            f"{total_rows} historical "
            "candidates."
        )

    ids = np.concatenate(
        ids_parts
    )

    vectors = np.vstack(
        vectors_parts
    ).astype(
        np.float32,
        copy=False,
    )

    if len(
        np.unique(
            ids
        )
    ) != total_rows:
        raise RuntimeError(
            f"{target_date}: duplicate "
            "historical article IDs."
        )

    norm_min, norm_max = (
        validate_embedding_matrix(
            vectors=vectors,
            expected_rows=total_rows,
            label=(
                f"{target_date} history"
            ),
        )
    )

    return {
        "history_start":
            history_start,
        "ids":
            ids,
        "vectors":
            vectors,
        "norm_min":
            norm_min,
        "norm_max":
            norm_max,
    }


# =============================================================================
# BENCHMARK ONE DATE
# =============================================================================

def benchmark_date(
    conn,
    target_date,
):

    date_start = perf_counter()

    print()
    print("=" * 110)
    print(
        f"DATE {target_date}"
    )
    print("=" * 110)

    target = load_targets(
        conn,
        target_date,
    )

    print(
        f"Targets              : "
        f"{len(target['ids'])}"
    )

    print(
        f"Sources              : "
        f"{target['source_counts']}"
    )

    print(
        f"Target norms         : "
        f"{target['norm_min']:.8f} -> "
        f"{target['norm_max']:.8f}"
    )

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------

    history_load_start = (
        perf_counter()
    )

    history = load_history(
        conn,
        target_date,
    )

    history_load_seconds = (
        perf_counter()
        - history_load_start
    )

    history_ids = history[
        "ids"
    ]

    history_vectors = history[
        "vectors"
    ]

    n_candidates = len(
        history_ids
    )

    print(
        f"History              : "
        f"{history['history_start']} "
        f"<= date < {target_date}"
    )

    print(
        f"Candidates           : "
        f"{n_candidates:,}"
    )

    print(
        f"History matrix       : "
        f"{history_vectors.shape}"
    )

    print(
        f"History norms        : "
        f"{history['norm_min']:.8f} -> "
        f"{history['norm_max']:.8f}"
    )

    print(
        f"History load         : "
        f"{history_load_seconds:.3f} s"
    )

    # -------------------------------------------------------------------------
    # Exact reference
    # -------------------------------------------------------------------------

    exact_index = faiss.IndexFlatIP(
        EXPECTED_DIMENSION
    )

    exact_index.add(
        history_vectors
    )

    if (
        exact_index.ntotal
        != n_candidates
    ):
        raise RuntimeError(
            f"{target_date}: unexpected "
            "exact-index size."
        )

    exact_start = perf_counter()

    exact_D, exact_I = (
        exact_index.search(
            target["vectors"],
            K,
        )
    )

    exact_seconds = (
        perf_counter()
        - exact_start
    )

    validate_search_output(
        distances=exact_D,
        indices=exact_I,
        n_queries=
            EXPECTED_TARGETS_PER_DATE,
        label=(
            f"{target_date} exact"
        ),
    )

    exact_D = np.clip(
        exact_D,
        -1.0,
        1.0,
    )

    exact_ids = (
        history_ids[
            exact_I
        ]
    )

    exact_novelty = (
        1.0
        - np.mean(
            exact_D,
            axis=1,
        )
    )

    # -------------------------------------------------------------------------
    # HNSW
    # -------------------------------------------------------------------------

    hnsw_index = (
        faiss.IndexHNSWFlat(
            EXPECTED_DIMENSION,
            HNSW_M,
            faiss.METRIC_INNER_PRODUCT,
        )
    )

    hnsw_index.hnsw.efConstruction = (
        HNSW_EF_CONSTRUCTION
    )

    hnsw_build_start = (
        perf_counter()
    )

    hnsw_index.add(
        history_vectors
    )

    hnsw_build_seconds = (
        perf_counter()
        - hnsw_build_start
    )

    if (
        hnsw_index.ntotal
        != n_candidates
    ):
        raise RuntimeError(
            f"{target_date}: unexpected "
            "HNSW index size."
        )

    hnsw_index.hnsw.efSearch = (
        HNSW_EF_SEARCH
    )

    hnsw_search_start = (
        perf_counter()
    )

    hnsw_D, hnsw_I = (
        hnsw_index.search(
            target["vectors"],
            K,
        )
    )

    hnsw_search_seconds = (
        perf_counter()
        - hnsw_search_start
    )

    validate_search_output(
        distances=hnsw_D,
        indices=hnsw_I,
        n_queries=
            EXPECTED_TARGETS_PER_DATE,
        label=(
            f"{target_date} HNSW"
        ),
    )

    hnsw_D = np.clip(
        hnsw_D,
        -1.0,
        1.0,
    )

    hnsw_ids = (
        history_ids[
            hnsw_I
        ]
    )

    hnsw_novelty = (
        1.0
        - np.mean(
            hnsw_D,
            axis=1,
        )
    )

    # -------------------------------------------------------------------------
    # Accuracy
    # -------------------------------------------------------------------------

    recalls = np.empty(
        EXPECTED_TARGETS_PER_DATE,
        dtype=np.float64,
    )

    for i in range(
        EXPECTED_TARGETS_PER_DATE
    ):

        exact_set = set(
            exact_ids[
                i
            ].tolist()
        )

        approximate_set = set(
            hnsw_ids[
                i
            ].tolist()
        )

        recalls[i] = (
            len(
                exact_set
                & approximate_set
            )
            / K
        )

    novelty_errors = (
        hnsw_novelty
        - exact_novelty
    )

    abs_errors = np.abs(
        novelty_errors
    )

    if not np.isfinite(
        novelty_errors
    ).all():
        raise RuntimeError(
            f"{target_date}: non-finite "
            "novelty errors."
        )

    perfect_mask = np.isclose(
        recalls,
        1.0,
        atol=1e-12,
        rtol=0.0,
    )

    worst_idx = int(
        np.argmax(
            abs_errors
        )
    )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    print()
    print(
        f"Exact search         : "
        f"{exact_seconds:.6f} s"
    )

    print(
        f"HNSW build           : "
        f"{hnsw_build_seconds:.3f} s"
    )

    print(
        f"HNSW search          : "
        f"{hnsw_search_seconds:.6f} s"
    )

    print(
        f"HNSW batch avg/query : "
        f"{1000.0 * hnsw_search_seconds / EXPECTED_TARGETS_PER_DATE:.6f} ms"
    )

    print()
    print(
        f"Recall@{K} mean       : "
        f"{np.mean(recalls):.4f}"
    )

    print(
        f"Recall@{K} median     : "
        f"{np.median(recalls):.4f}"
    )

    print(
        f"Recall@{K} min        : "
        f"{np.min(recalls):.4f}"
    )

    print(
        f"Recall@{K} p05        : "
        f"{np.quantile(recalls, 0.05):.4f}"
    )

    print(
        f"Perfect targets      : "
        f"{int(np.sum(perfect_mask))}/"
        f"{EXPECTED_TARGETS_PER_DATE}"
    )

    print(
        f"Mean |nov error|     : "
        f"{np.mean(abs_errors):.8f}"
    )

    print(
        f"P95 |nov error|      : "
        f"{np.quantile(abs_errors, 0.95):.8f}"
    )

    print(
        f"Max |nov error|      : "
        f"{np.max(abs_errors):.8f}"
    )

    print(
        f"Mean signed error    : "
        f"{np.mean(novelty_errors):+.8f}"
    )

    print()
    print(
        f"Worst article        : "
        f"{target['ids'][worst_idx]}"
    )

    print(
        f"Worst source         : "
        f"{target['sources'][worst_idx]}"
    )

    print(
        f"Worst title          : "
        f"{target['titles'][worst_idx]}"
    )

    print(
        f"Worst exact novelty  : "
        f"{exact_novelty[worst_idx]:.8f}"
    )

    print(
        f"Worst novelty error  : "
        f"{novelty_errors[worst_idx]:+.8f}"
    )

    print(
        f"Worst Recall@{K}      : "
        f"{recalls[worst_idx]:.4f}"
    )

    date_seconds = (
        perf_counter()
        - date_start
    )

    print(
        f"Date total runtime   : "
        f"{date_seconds:.3f} s"
    )

    result = {
        "date":
            target_date,
        "candidates":
            n_candidates,
        "recalls":
            recalls.copy(),
        "novelty_errors":
            novelty_errors.copy(),
        "mean_recall":
            float(
                np.mean(recalls)
            ),
        "min_recall":
            float(
                np.min(recalls)
            ),
        "perfect_targets":
            int(
                np.sum(
                    perfect_mask
                )
            ),
        "mean_abs_error":
            float(
                np.mean(
                    abs_errors
                )
            ),
        "p95_abs_error":
            float(
                np.quantile(
                    abs_errors,
                    0.95,
                )
            ),
        "max_abs_error":
            float(
                np.max(
                    abs_errors
                )
            ),
        "mean_signed_error":
            float(
                np.mean(
                    novelty_errors
                )
            ),
        "exact_search_seconds":
            exact_seconds,
        "hnsw_build_seconds":
            hnsw_build_seconds,
        "hnsw_search_seconds":
            hnsw_search_seconds,
    }

    del exact_index
    del hnsw_index
    del history
    del history_ids
    del history_vectors
    del exact_D
    del exact_I
    del exact_ids
    del hnsw_D
    del hnsw_I
    del hnsw_ids

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    total_start = perf_counter()

    if len(
        set(
            BENCHMARK_DATES
        )
    ) != len(
        BENCHMARK_DATES
    ):
        raise RuntimeError(
            "Duplicate benchmark dates."
        )

    total_expected_targets = (
        len(BENCHMARK_DATES)
        * EXPECTED_TARGETS_PER_DATE
    )

    print("=" * 110)
    print(
        "STEP 4 — FAISS HNSW "
        "TEMPORAL VALIDATION"
    )
    print("=" * 110)

    print(
        f"Embedding model      : "
        f"{MODEL_NAME}"
    )

    print(
        f"Dates                : "
        f"{len(BENCHMARK_DATES)}"
    )

    print(
        f"Targets/date         : "
        f"{EXPECTED_TARGETS_PER_DATE}"
    )

    print(
        f"Total targets        : "
        f"{total_expected_targets}"
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
        f"HNSW M               : "
        f"{HNSW_M}"
    )

    print(
        f"efConstruction       : "
        f"{HNSW_EF_CONSTRUCTION}"
    )

    print(
        f"efSearch             : "
        f"{HNSW_EF_SEARCH}"
    )

    conn = get_connection()

    results = []

    try:

        for target_date in (
            BENCHMARK_DATES
        ):

            results.append(
                benchmark_date(
                    conn,
                    target_date,
                )
            )

    finally:

        conn.rollback()
        conn.close()

    if (
        len(results)
        != len(BENCHMARK_DATES)
    ):
        raise RuntimeError(
            "Not all benchmark dates "
            "completed."
        )

    # =========================================================================
    # TEMPORAL SUMMARY
    # =========================================================================

    print()
    print("=" * 110)
    print("TEMPORAL SUMMARY")
    print("=" * 110)

    print(
        "Date       | Candidates | "
        "Recall | MinRec | Perfect | "
        "Mean|ΔN|   | P95|ΔN|    | Max|ΔN|"
    )

    print("-" * 110)

    for result in results:

        print(
            f"{result['date']} | "
            f"{result['candidates']:>10,} | "
            f"{result['mean_recall']:.4f} | "
            f"{result['min_recall']:.4f} | "
            f"{result['perfect_targets']:>3}/"
            f"{EXPECTED_TARGETS_PER_DATE} | "
            f"{result['mean_abs_error']:.8f} | "
            f"{result['p95_abs_error']:.8f} | "
            f"{result['max_abs_error']:.8f}"
        )

    # =========================================================================
    # GLOBAL TARGET-LEVEL SUMMARY
    # =========================================================================

    all_recalls = np.concatenate(
        [
            result["recalls"]
            for result
            in results
        ]
    )

    all_novelty_errors = (
        np.concatenate(
            [
                result[
                    "novelty_errors"
                ]
                for result
                in results
            ]
        )
    )

    all_abs_errors = np.abs(
        all_novelty_errors
    )

    total_perfect = int(
        np.sum(
            np.isclose(
                all_recalls,
                1.0,
                atol=1e-12,
                rtol=0.0,
            )
        )
    )

    total_hnsw_build = sum(
        result[
            "hnsw_build_seconds"
        ]
        for result
        in results
    )

    total_hnsw_search = sum(
        result[
            "hnsw_search_seconds"
        ]
        for result
        in results
    )

    print()
    print("=" * 110)
    print("GLOBAL TARGET-LEVEL SUMMARY")
    print("=" * 110)

    print(
        f"Dates tested          : "
        f"{len(results)}"
    )

    print(
        f"Targets tested        : "
        f"{len(all_recalls)}"
    )

    print(
        f"Global Recall@{K} mean: "
        f"{np.mean(all_recalls):.6f}"
    )

    print(
        f"Global Recall@{K} med : "
        f"{np.median(all_recalls):.6f}"
    )

    print(
        f"Global Recall@{K} p05 : "
        f"{np.quantile(all_recalls, 0.05):.6f}"
    )

    print(
        f"Worst target Recall  : "
        f"{np.min(all_recalls):.4f}"
    )

    print(
        f"Perfect targets      : "
        f"{total_perfect}/"
        f"{len(all_recalls)}"
    )

    print(
        f"Global mean |ΔN|     : "
        f"{np.mean(all_abs_errors):.8f}"
    )

    print(
        f"Global P95 |ΔN|      : "
        f"{np.quantile(all_abs_errors, 0.95):.8f}"
    )

    print(
        f"Global max |ΔN|      : "
        f"{np.max(all_abs_errors):.8f}"
    )

    print(
        f"Global mean ΔN       : "
        f"{np.mean(all_novelty_errors):+.8f}"
    )

    print(
        f"Total HNSW build     : "
        f"{total_hnsw_build:.3f} s"
    )

    print(
        f"Total HNSW search    : "
        f"{total_hnsw_search:.6f} s"
    )

    print(
        f"Batch avg/query      : "
        f"{1_000.0 * total_hnsw_search / len(all_recalls):.6f} ms"
    )

    print(
        f"Total runtime        : "
        f"{perf_counter() - total_start:.3f} s"
    )

    print()
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
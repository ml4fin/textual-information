#!/usr/bin/env python3

"""
STEP 4 — FAISS HNSW efSearch VALIDATION
========================================

Purpose
-------
Evaluate the retrieval accuracy / query-time trade-off of different
HNSW efSearch values on real MiniLM document embeddings.

Fixed components:
- embedding model:
    all-MiniLM-L6-v2_chunk254_mean_l2_v1
- target date:
    2015-01-30
- historical window:
    365 days, strictly preceding the target date
- novelty neighbours:
    K = 10
- HNSW M:
    48
- HNSW efConstruction:
    400

Variable component:
- efSearch:
    16, 32, 64, 128, 256

Exact reference:
    FAISS IndexFlatIP.

Evaluation:
- Recall@10 against exact nearest neighbours
- fraction of targets with perfect Recall@10
- novelty-score approximation error
- query latency

The 100 target articles are deterministically sampled with equal
allocation across the five news sources.

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

TARGET_DATE = date(
    2015,
    1,
    30,
)

WINDOW_DAYS = 365

K = 10

TARGETS_PER_SOURCE = 20
EXPECTED_SOURCES = 5

EXPECTED_TARGETS = (
    TARGETS_PER_SOURCE
    * EXPECTED_SOURCES
)

FETCH_SIZE = 5_000

HNSW_M = 48
HNSW_EF_CONSTRUCTION = 400

EF_SEARCH_VALUES = [
    16,
    32,
    64,
    128,
    256,
]

NORM_TOLERANCE = 1e-5

# Optional diagnostic value only.
# This is not interpreted as the selected final configuration.
DIAGNOSTIC_EF_SEARCH = 64


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
# QUERIES
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

    return {
        "norm_min": float(
            norms.min()
        ),
        "norm_max": float(
            norms.max()
        ),
        "max_norm_error": float(
            np.max(
                np.abs(
                    norms - 1.0
                )
            )
        ),
    }


def validate_search_output(
    distances,
    indices,
    n_queries,
    k,
    label,
):

    expected_shape = (
        n_queries,
        k,
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
            "neighbor index detected."
        )


# =============================================================================
# HISTORY LOADING
# =============================================================================

def load_history(
    conn,
    history_start,
    history_end,
):

    ids_parts = []
    vectors_parts = []

    stream = conn.cursor(
        name="faiss_efsearch_history_stream"
    )

    stream.itersize = FETCH_SIZE

    total_rows = 0

    try:

        stream.execute(
            HISTORY_QUERY,
            (
                MODEL_NAME,
                history_start,
                history_end,
            ),
        )

        while True:

            rows = stream.fetchmany(
                FETCH_SIZE
            )

            if not rows:
                break

            ids_batch = np.asarray(
                [
                    row[0]
                    for row
                    in rows
                ],
                dtype=np.int64,
            )

            vectors_batch = np.asarray(
                [
                    row[2]
                    for row
                    in rows
                ],
                dtype=np.float32,
            )

            if vectors_batch.ndim != 2:
                raise RuntimeError(
                    "Unexpected embedding "
                    "structure in history."
                )

            ids_parts.append(
                ids_batch
            )

            vectors_parts.append(
                vectors_batch
            )

            total_rows += len(
                rows
            )

    finally:

        stream.close()

    if total_rows < K:
        raise RuntimeError(
            f"Only {total_rows} historical "
            f"candidates found."
        )

    history_ids = np.concatenate(
        ids_parts
    )

    history_vectors = np.vstack(
        vectors_parts
    ).astype(
        np.float32,
        copy=False,
    )

    if len(
        np.unique(
            history_ids
        )
    ) != total_rows:
        raise RuntimeError(
            "Duplicate article IDs detected "
            "in historical candidate set."
        )

    validation = validate_embedding_matrix(
        vectors=history_vectors,
        expected_rows=total_rows,
        label="historical candidates",
    )

    return (
        history_ids,
        history_vectors,
        validation,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    total_start = perf_counter()

    print("=" * 100)
    print(
        "STEP 4 — FAISS HNSW "
        "efSearch VALIDATION"
    )
    print("=" * 100)

    print(
        f"\nEmbedding model:      "
        f"{MODEL_NAME}"
    )

    print(
        f"Target date:          "
        f"{TARGET_DATE}"
    )

    print(
        f"History window:       "
        f"{WINDOW_DAYS} days"
    )

    print(
        f"K:                    "
        f"{K}"
    )

    print(
        f"HNSW M:               "
        f"{HNSW_M}"
    )

    print(
        f"HNSW efConstruction:  "
        f"{HNSW_EF_CONSTRUCTION}"
    )

    print(
        f"efSearch values:      "
        f"{EF_SEARCH_VALUES}"
    )

    if (
        DIAGNOSTIC_EF_SEARCH
        not in EF_SEARCH_VALUES
    ):
        raise RuntimeError(
            "DIAGNOSTIC_EF_SEARCH must "
            "belong to EF_SEARCH_VALUES."
        )

    conn = get_connection()

    try:

        # =====================================================================
        # 1. TARGET SAMPLE
        # =====================================================================

        print()
        print("=" * 100)
        print("1. LOAD TARGET SAMPLE")
        print("=" * 100)

        with conn.cursor() as cur:

            cur.execute(
                TARGET_QUERY,
                (
                    MODEL_NAME,
                    TARGET_DATE,
                    TARGETS_PER_SOURCE,
                ),
            )

            rows = cur.fetchall()

        if len(rows) != EXPECTED_TARGETS:
            raise RuntimeError(
                f"Expected {EXPECTED_TARGETS} "
                f"targets, found {len(rows)}."
            )

        target_ids = np.asarray(
            [
                row[0]
                for row
                in rows
            ],
            dtype=np.int64,
        )

        target_dates = np.asarray(
            [
                row[1]
                for row
                in rows
            ],
            dtype=object,
        )

        target_sources = [
            row[2]
            for row
            in rows
        ]

        target_titles = [
            row[3]
            for row
            in rows
        ]

        target_vectors = np.asarray(
            [
                row[4]
                for row
                in rows
            ],
            dtype=np.float32,
        )

        if len(
            np.unique(
                target_ids
            )
        ) != EXPECTED_TARGETS:
            raise RuntimeError(
                "Duplicate target article IDs."
            )

        if not np.all(
            target_dates
            == TARGET_DATE
        ):
            raise RuntimeError(
                "Not all targets have the "
                "expected publication date."
            )

        source_counts = {
            source: target_sources.count(
                source
            )
            for source
            in sorted(
                set(
                    target_sources
                )
            )
        }

        if len(
            source_counts
        ) != EXPECTED_SOURCES:
            raise RuntimeError(
                "Unexpected number of sources: "
                f"{source_counts}"
            )

        if any(
            count != TARGETS_PER_SOURCE
            for count
            in source_counts.values()
        ):
            raise RuntimeError(
                "Unbalanced target sample: "
                f"{source_counts}"
            )

        target_validation = (
            validate_embedding_matrix(
                vectors=target_vectors,
                expected_rows=EXPECTED_TARGETS,
                label="targets",
            )
        )

        print(
            f"\nTargets:              "
            f"{EXPECTED_TARGETS}"
        )

        print(
            f"Sources:              "
            f"{source_counts}"
        )

        print(
            f"Target matrix:        "
            f"{target_vectors.shape}"
        )

        print(
            f"Target norm range:    "
            f"{target_validation['norm_min']:.8f}"
            " -> "
            f"{target_validation['norm_max']:.8f}"
        )

        # =====================================================================
        # 2. HISTORICAL WINDOW
        # =====================================================================

        history_start = (
            TARGET_DATE
            - timedelta(
                days=WINDOW_DAYS
            )
        )

        print()
        print("=" * 100)
        print("2. LOAD HISTORICAL CANDIDATES")
        print("=" * 100)

        print(
            f"\nHistory:              "
            f"{history_start} <= date < "
            f"{TARGET_DATE}"
        )

        (
            history_ids,
            history_vectors,
            history_validation,
        ) = load_history(
            conn=conn,
            history_start=history_start,
            history_end=TARGET_DATE,
        )

        history_rows = len(
            history_ids
        )

        print(
            f"Candidates:           "
            f"{history_rows:,}"
        )

        print(
            f"History matrix:       "
            f"{history_vectors.shape}"
        )

        print(
            f"History norm range:   "
            f"{history_validation['norm_min']:.8f}"
            " -> "
            f"{history_validation['norm_max']:.8f}"
        )

        # =====================================================================
        # 3. EXACT REFERENCE
        # =====================================================================

        print()
        print("=" * 100)
        print("3. EXACT GROUND TRUTH")
        print("=" * 100)

        exact_build_start = (
            perf_counter()
        )

        exact_index = faiss.IndexFlatIP(
            EXPECTED_DIMENSION
        )

        exact_index.add(
            history_vectors
        )

        exact_build_seconds = (
            perf_counter()
            - exact_build_start
        )

        exact_search_start = (
            perf_counter()
        )

        exact_D, exact_I = (
            exact_index.search(
                target_vectors,
                K,
            )
        )

        exact_search_seconds = (
            perf_counter()
            - exact_search_start
        )

        validate_search_output(
            distances=exact_D,
            indices=exact_I,
            n_queries=EXPECTED_TARGETS,
            k=K,
            label="exact search",
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

        exact_mean_similarity = (
            np.mean(
                exact_D,
                axis=1,
            )
        )

        exact_novelty = (
            1.0
            - exact_mean_similarity
        )

        if not np.isfinite(
            exact_novelty
        ).all():
            raise RuntimeError(
                "Non-finite exact novelty "
                "values detected."
            )

        print(
            f"\nExact build:          "
            f"{exact_build_seconds:.6f} s"
        )

        print(
            f"Exact search:         "
            f"{exact_search_seconds:.6f} s"
        )

        print(
            f"Exact/query:          "
            f"{1_000.0 * exact_search_seconds / EXPECTED_TARGETS:.6f} ms"
        )

        print(
            f"Novelty min:          "
            f"{exact_novelty.min():.8f}"
        )

        print(
            f"Novelty mean:         "
            f"{exact_novelty.mean():.8f}"
        )

        print(
            f"Novelty max:          "
            f"{exact_novelty.max():.8f}"
        )

        # =====================================================================
        # 4. HNSW BUILD
        # =====================================================================

        print()
        print("=" * 100)
        print("4. HNSW BUILD")
        print("=" * 100)

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
            != history_rows
        ):
            raise RuntimeError(
                "Unexpected HNSW index size."
            )

        print(
            f"\nM:                    "
            f"{HNSW_M}"
        )

        print(
            f"efConstruction:       "
            f"{HNSW_EF_CONSTRUCTION}"
        )

        print(
            f"Indexed vectors:      "
            f"{hnsw_index.ntotal:,}"
        )

        print(
            f"Build time:           "
            f"{hnsw_build_seconds:.3f} s"
        )

        # =====================================================================
        # 5. efSearch SWEEP
        # =====================================================================

        print()
        print("=" * 100)
        print("5. HNSW ACCURACY / SPEED")
        print("=" * 100)

        print(
            "\nefSearch | Recall@10 | Perfect | "
            "Mean |ΔN| | P95 |ΔN| | Max |ΔN| | "
            "Mean ΔN | Search s | ms/query"
        )

        print("-" * 112)

        exact_sets = [
            set(
                exact_ids[
                    i
                ].tolist()
            )
            for i
            in range(
                EXPECTED_TARGETS
            )
        ]

        all_results = []

        for ef_search in (
            EF_SEARCH_VALUES
        ):

            hnsw_index.hnsw.efSearch = (
                ef_search
            )

            search_start = (
                perf_counter()
            )

            hnsw_D, hnsw_I = (
                hnsw_index.search(
                    target_vectors,
                    K,
                )
            )

            search_seconds = (
                perf_counter()
                - search_start
            )

            validate_search_output(
                distances=hnsw_D,
                indices=hnsw_I,
                n_queries=EXPECTED_TARGETS,
                k=K,
                label=(
                    f"HNSW efSearch="
                    f"{ef_search}"
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

            hnsw_mean_similarity = (
                np.mean(
                    hnsw_D,
                    axis=1,
                )
            )

            hnsw_novelty = (
                1.0
                - hnsw_mean_similarity
            )

            recalls = np.empty(
                EXPECTED_TARGETS,
                dtype=np.float64,
            )

            for i in range(
                EXPECTED_TARGETS
            ):

                approximate_set = set(
                    hnsw_ids[
                        i
                    ].tolist()
                )

                recalls[i] = (
                    len(
                        exact_sets[
                            i
                        ]
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

            mean_recall = float(
                np.mean(
                    recalls
                )
            )

            perfect_mask = (
                np.isclose(
                    recalls,
                    1.0,
                    atol=1e-12,
                    rtol=0.0,
                )
            )

            perfect_fraction = float(
                np.mean(
                    perfect_mask
                )
            )

            mean_abs_error = float(
                np.mean(
                    abs_errors
                )
            )

            p95_abs_error = float(
                np.quantile(
                    abs_errors,
                    0.95,
                )
            )

            max_abs_error = float(
                np.max(
                    abs_errors
                )
            )

            mean_signed_error = float(
                np.mean(
                    novelty_errors
                )
            )

            ms_per_query = (
                1_000.0
                * search_seconds
                / EXPECTED_TARGETS
            )

            all_results.append(
                {
                    "ef_search":
                        ef_search,
                    "recalls":
                        recalls,
                    "novelty_errors":
                        novelty_errors,
                    "mean_recall":
                        mean_recall,
                    "perfect_fraction":
                        perfect_fraction,
                    "mean_abs_error":
                        mean_abs_error,
                    "p95_abs_error":
                        p95_abs_error,
                    "max_abs_error":
                        max_abs_error,
                    "mean_signed_error":
                        mean_signed_error,
                    "search_seconds":
                        search_seconds,
                    "ms_per_query":
                        ms_per_query,
                }
            )

            print(
                f"{ef_search:>8} | "
                f"{mean_recall:>9.4f} | "
                f"{perfect_fraction:>7.2%} | "
                f"{mean_abs_error:>9.7f} | "
                f"{p95_abs_error:>8.7f} | "
                f"{max_abs_error:>8.7f} | "
                f"{mean_signed_error:>8.7f} | "
                f"{search_seconds:>8.6f} | "
                f"{ms_per_query:>8.4f}"
            )

        # =====================================================================
        # 6. DETAILED DISTRIBUTIONS
        # =====================================================================

        print()
        print("=" * 100)
        print("6. DETAILED ACCURACY")
        print("=" * 100)

        for result in (
            all_results
        ):

            recalls = result[
                "recalls"
            ]

            errors = np.abs(
                result[
                    "novelty_errors"
                ]
            )

            perfect_count = int(
                np.sum(
                    np.isclose(
                        recalls,
                        1.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                )
            )

            print()
            print(
                f"efSearch = "
                f"{result['ef_search']}"
            )

            print(
                f"  Recall@{K} mean:    "
                f"{np.mean(recalls):.4f}"
            )

            print(
                f"  Recall@{K} median:  "
                f"{np.median(recalls):.4f}"
            )

            print(
                f"  Recall@{K} min:     "
                f"{np.min(recalls):.4f}"
            )

            print(
                f"  Recall@{K} p05:     "
                f"{np.quantile(recalls, 0.05):.4f}"
            )

            print(
                f"  Perfect targets:    "
                f"{perfect_count}/"
                f"{EXPECTED_TARGETS}"
            )

            print(
                f"  Mean |nov error|:   "
                f"{np.mean(errors):.8f}"
            )

            print(
                f"  P95 |nov error|:    "
                f"{np.quantile(errors, 0.95):.8f}"
            )

            print(
                f"  Max |nov error|:    "
                f"{np.max(errors):.8f}"
            )

        # =====================================================================
        # 7. PREDEFINED DIAGNOSTIC efSearch
        # =====================================================================

        selected_result = next(
            result
            for result
            in all_results
            if (
                result["ef_search"]
                == DIAGNOSTIC_EF_SEARCH
            )
        )

        selected_recalls = (
            selected_result[
                "recalls"
            ]
        )

        selected_errors = np.abs(
            selected_result[
                "novelty_errors"
            ]
        )

        print()
        print("=" * 100)
        print(
            "7. SOURCE DIAGNOSTIC — "
            f"efSearch = "
            f"{DIAGNOSTIC_EF_SEARCH}"
        )
        print("=" * 100)

        for source in sorted(
            source_counts
        ):

            source_mask = np.asarray(
                [
                    value == source
                    for value
                    in target_sources
                ],
                dtype=bool,
            )

            print(
                f"\n{source:<4} | "
                f"n="
                f"{int(np.sum(source_mask)):>2} | "
                f"mean recall="
                f"{np.mean(selected_recalls[source_mask]):.4f} | "
                f"mean |ΔN|="
                f"{np.mean(selected_errors[source_mask]):.8f} | "
                f"max |ΔN|="
                f"{np.max(selected_errors[source_mask]):.8f}"
            )

        # =====================================================================
        # 8. WORST DIAGNOSTIC TARGET
        # =====================================================================

        worst_idx = int(
            np.argmax(
                selected_errors
            )
        )

        print()
        print("=" * 100)
        print(
            "8. WORST NOVELTY ERROR — "
            f"efSearch = "
            f"{DIAGNOSTIC_EF_SEARCH}"
        )
        print("=" * 100)

        print(
            f"\nArticle ID:           "
            f"{target_ids[worst_idx]}"
        )

        print(
            f"Source:               "
            f"{target_sources[worst_idx]}"
        )

        print(
            f"Title:                "
            f"{target_titles[worst_idx]}"
        )

        print(
            f"Exact novelty:        "
            f"{exact_novelty[worst_idx]:.8f}"
        )

        print(
            f"Novelty error:        "
            f"{selected_result['novelty_errors'][worst_idx]:+.8f}"
        )

        print(
            f"Recall@{K}:            "
            f"{selected_recalls[worst_idx]:.4f}"
        )

        # =====================================================================
        # 9. FINAL
        # =====================================================================

        print()
        print("=" * 100)
        print("9. FINAL")
        print("=" * 100)

        print(
            f"\nCandidates:           "
            f"{history_rows:,}"
        )

        print(
            f"Targets:              "
            f"{EXPECTED_TARGETS}"
        )

        print(
            f"Exact build time:     "
            f"{exact_build_seconds:.6f} s"
        )

        print(
            f"Exact search time:    "
            f"{exact_search_seconds:.6f} s"
        )

        print(
            f"HNSW build time:      "
            f"{hnsw_build_seconds:.3f} s"
        )

        print(
            f"Total runtime:        "
            f"{perf_counter() - total_start:.3f} s"
        )

        print()
        print(
            "DATABASE STATUS: READ ONLY"
        )

        print("=" * 100)

    finally:

        conn.rollback()
        conn.close()


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
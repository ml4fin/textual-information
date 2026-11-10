#!/usr/bin/env python3

"""
STEP 4 — FAISS ROLLING TEMPORAL FILTER BENCHMARK
=================================================

Purpose
-------
Validate a rolling HNSW retrieval strategy for the novelty indicator
under the worst-case monthly stale-vector share identified by the
corpus diagnostics.

Target date:
    2019-10-31

Monthly rebuild date:
    2019-10-01

Exact novelty candidate window:
    [target_date - 365 days, target_date)

Operational HNSW strategy:
1. rebuild the index at the beginning of the month using the
   365-day history valid at the rebuild date;
2. add new articles incrementally during the month;
3. retain vectors that become temporally expired before the next rebuild;
4. retrieve K' approximate neighbours;
5. remove temporally invalid neighbours;
6. retain the first K valid neighbours.

Exact reference:
    FAISS IndexFlatIP over the exact valid 365-day candidate set.

Approximate index:
    FAISS IndexHNSWFlat + IndexIDMap2.

Evaluation:
- retrieval completeness
- Recall@K
- novelty approximation error
- stale-vector contamination of the raw candidate pool
- HNSW query latency

IMPORTANT
---------
The HNSW configuration is treated as fixed for this benchmark.

The oversampling values are candidate-pool sizes K', not alternative
values of the final novelty K.

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
    2019,
    10,
    31,
)

REBUILD_DATE = date(
    2019,
    10,
    1,
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
HNSW_EF_SEARCH = 128

OVERSAMPLING_VALUES = [
    20,
    50,
    100,
    200,
]

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
ORDER BY source_code, rn;
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
# VALIDATION HELPERS
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


def validate_faiss_results(
    indices,
    scores,
    n_queries,
    requested_k,
    label,
):

    expected_shape = (
        n_queries,
        requested_k,
    )

    if indices.shape != expected_shape:

        raise RuntimeError(
            f"{label}: unexpected index "
            f"shape {indices.shape}; "
            f"expected {expected_shape}."
        )

    if scores.shape != expected_shape:

        raise RuntimeError(
            f"{label}: unexpected score "
            f"shape {scores.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(
        scores
    ).all():

        raise RuntimeError(
            f"{label}: non-finite "
            "similarities returned."
        )


# =============================================================================
# LOAD HISTORY
# =============================================================================

def load_history(
    conn,
    start_date,
    end_date,
    cursor_name,
):

    ids_parts = []
    dates_parts = []
    vectors_parts = []

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
                start_date,
                end_date,
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

            dates_batch = np.asarray(
                [
                    row[1]
                    for row
                    in rows
                ],
                dtype=object,
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
                    "structure while loading "
                    "historical vectors."
                )

            ids_parts.append(
                ids_batch
            )

            dates_parts.append(
                dates_batch
            )

            vectors_parts.append(
                vectors_batch
            )

            total_rows += len(
                rows
            )

    finally:

        stream.close()

    if total_rows == 0:

        raise RuntimeError(
            "No rows found for interval "
            f"{start_date} <= date < "
            f"{end_date}."
        )

    ids = np.concatenate(
        ids_parts
    )

    dates = np.concatenate(
        dates_parts
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
            "Duplicate article IDs detected "
            f"in interval {start_date} <= "
            f"date < {end_date}."
        )

    validate_embedding_matrix(
        vectors=vectors,
        expected_rows=total_rows,
        label=(
            f"history "
            f"[{start_date}, {end_date})"
        ),
    )

    return (
        ids,
        dates,
        vectors,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    total_start = perf_counter()

    print("=" * 110)
    print(
        "STEP 4 — FAISS ROLLING "
        "TEMPORAL FILTER BENCHMARK"
    )
    print("=" * 110)

    print(
        f"\nEmbedding model:      "
        f"{MODEL_NAME}"
    )

    print(
        f"Target date:          "
        f"{TARGET_DATE}"
    )

    print(
        f"Rebuild date:         "
        f"{REBUILD_DATE}"
    )

    print(
        f"History window:       "
        f"{WINDOW_DAYS} days"
    )

    print(
        f"Novelty K:            "
        f"{K}"
    )

    print(
        f"Candidate K' values:  "
        f"{OVERSAMPLING_VALUES}"
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
        f"HNSW efSearch base:   "
        f"{HNSW_EF_SEARCH}"
    )

    if REBUILD_DATE >= TARGET_DATE:

        raise RuntimeError(
            "REBUILD_DATE must precede "
            "TARGET_DATE."
        )

    if any(
        k_prime < K
        for k_prime
        in OVERSAMPLING_VALUES
    ):

        raise RuntimeError(
            "Every K' must be >= K."
        )

    conn = get_connection()

    try:

        # =====================================================================
        # 1. LOAD BALANCED TARGET SAMPLE
        # =====================================================================

        print()
        print("=" * 110)
        print("1. LOAD BALANCED TARGET SAMPLE")
        print("=" * 110)

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
                "Unexpected publication date "
                "in target sample."
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

        if (
            len(source_counts)
            != EXPECTED_SOURCES
        ):

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
            f"{len(target_ids)}"
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
        # 2. DEFINE TEMPORAL RANGES
        # =====================================================================

        exact_start_date = (
            TARGET_DATE
            - timedelta(
                days=WINDOW_DAYS
            )
        )

        rebuild_history_start = (
            REBUILD_DATE
            - timedelta(
                days=WINDOW_DAYS
            )
        )

        print()
        print("=" * 110)
        print("2. TEMPORAL RANGES")
        print("=" * 110)

        print(
            f"\nExact valid history:  "
            f"{exact_start_date} <= date < "
            f"{TARGET_DATE}"
        )

        print(
            f"Rebuild history:      "
            f"{rebuild_history_start} <= date < "
            f"{REBUILD_DATE}"
        )

        print(
            f"Incremental additions:"
            f" {REBUILD_DATE} <= date < "
            f"{TARGET_DATE}"
        )

        # =====================================================================
        # 3. LOAD REBUILD HISTORY
        # =====================================================================

        rebuild_load_start = (
            perf_counter()
        )

        (
            rebuild_ids,
            rebuild_dates,
            rebuild_vectors,
        ) = load_history(
            conn=conn,
            start_date=
                rebuild_history_start,
            end_date=
                REBUILD_DATE,
            cursor_name=
                "rolling_rebuild_stream",
        )

        rebuild_load_seconds = (
            perf_counter()
            - rebuild_load_start
        )

        print()
        print("=" * 110)
        print("3. REBUILD DATA")
        print("=" * 110)

        print(
            f"\nRows:                 "
            f"{len(rebuild_ids):,}"
        )

        print(
            f"Matrix:               "
            f"{rebuild_vectors.shape}"
        )

        print(
            f"Load time:            "
            f"{rebuild_load_seconds:.3f} s"
        )

        # =====================================================================
        # 4. LOAD INCREMENTAL ADDITIONS
        # =====================================================================

        add_load_start = (
            perf_counter()
        )

        (
            add_ids,
            add_dates,
            add_vectors,
        ) = load_history(
            conn=conn,
            start_date=
                REBUILD_DATE,
            end_date=
                TARGET_DATE,
            cursor_name=
                "rolling_add_stream",
        )

        add_load_seconds = (
            perf_counter()
            - add_load_start
        )

        print()
        print("=" * 110)
        print("4. INCREMENTAL DATA")
        print("=" * 110)

        print(
            f"\nRows:                 "
            f"{len(add_ids):,}"
        )

        print(
            f"Matrix:               "
            f"{add_vectors.shape}"
        )

        print(
            f"Load time:            "
            f"{add_load_seconds:.3f} s"
        )

        # =====================================================================
        # 5. BUILD INDEX UNIVERSE
        # =====================================================================

        index_ids = np.concatenate(
            (
                rebuild_ids,
                add_ids,
            )
        )

        index_dates = np.concatenate(
            (
                rebuild_dates,
                add_dates,
            )
        )

        index_vectors = np.vstack(
            (
                rebuild_vectors,
                add_vectors,
            )
        ).astype(
            np.float32,
            copy=False,
        )

        if len(
            np.unique(
                index_ids
            )
        ) != len(
            index_ids
        ):

            raise RuntimeError(
                "Duplicate article IDs in "
                "index universe."
            )

        validate_embedding_matrix(
            vectors=index_vectors,
            expected_rows=len(
                index_ids
            ),
            label="index universe",
        )

        id_to_date = {
            int(article_id):
                publication_date
            for (
                article_id,
                publication_date,
            ) in zip(
                index_ids,
                index_dates,
            )
        }

        # =====================================================================
        # 6. EXACT GROUND TRUTH
        # =====================================================================

        exact_mask = (
            (index_dates >= exact_start_date)
            &
            (index_dates < TARGET_DATE)
        )

        exact_ids_all = (
            index_ids[
                exact_mask
            ]
        )

        exact_vectors = (
            index_vectors[
                exact_mask
            ]
        )

        if len(
            exact_ids_all
        ) < K:

            raise RuntimeError(
                "Insufficient exact candidates."
            )

        print()
        print("=" * 110)
        print("5. EXACT GROUND TRUTH")
        print("=" * 110)

        print(
            f"\nValid candidates:     "
            f"{len(exact_ids_all):,}"
        )

        exact_build_start = (
            perf_counter()
        )

        exact_index = faiss.IndexFlatIP(
            EXPECTED_DIMENSION
        )

        exact_index.add(
            exact_vectors
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

        validate_faiss_results(
            indices=exact_I,
            scores=exact_D,
            n_queries=EXPECTED_TARGETS,
            requested_k=K,
            label="exact",
        )

        if np.any(
            exact_I < 0
        ):

            raise RuntimeError(
                "Exact search returned "
                "invalid neighbor indices."
            )

        exact_sims = np.clip(
            exact_D,
            -1.0,
            1.0,
        )

        exact_neighbor_ids = (
            exact_ids_all[
                exact_I
            ]
        )

        exact_novelty = (
            1.0
            - np.mean(
                exact_sims,
                axis=1,
            )
        )

        if not np.isfinite(
            exact_novelty
        ).all():

            raise RuntimeError(
                "Non-finite exact novelty "
                "values detected."
            )

        print(
            f"Exact build:          "
            f"{exact_build_seconds:.3f} s"
        )

        print(
            f"Exact search:         "
            f"{exact_search_seconds:.6f} s"
        )

        print(
            f"Exact novelty mean:   "
            f"{exact_novelty.mean():.8f}"
        )

        # =====================================================================
        # 7. MONTHLY HNSW REBUILD
        # =====================================================================

        print()
        print("=" * 110)
        print("6. HNSW MONTHLY REBUILD")
        print("=" * 110)

        hnsw_base = (
            faiss.IndexHNSWFlat(
                EXPECTED_DIMENSION,
                HNSW_M,
                faiss.METRIC_INNER_PRODUCT,
            )
        )

        hnsw_base.hnsw.efConstruction = (
            HNSW_EF_CONSTRUCTION
        )

        hnsw_base.hnsw.efSearch = (
            HNSW_EF_SEARCH
        )

        hnsw_index = faiss.IndexIDMap2(
            hnsw_base
        )

        hnsw_build_start = (
            perf_counter()
        )

        hnsw_index.add_with_ids(
            rebuild_vectors,
            rebuild_ids,
        )

        hnsw_build_seconds = (
            perf_counter()
            - hnsw_build_start
        )

        if (
            hnsw_index.ntotal
            != len(rebuild_ids)
        ):

            raise RuntimeError(
                "Unexpected HNSW size "
                "after monthly rebuild."
            )

        print(
            f"\nInitial vectors:      "
            f"{hnsw_index.ntotal:,}"
        )

        print(
            f"Build time:           "
            f"{hnsw_build_seconds:.3f} s"
        )

        # =====================================================================
        # 8. INCREMENTAL ADDITIONS
        # =====================================================================

        incremental_start = (
            perf_counter()
        )

        unique_add_dates = sorted(
            set(
                add_dates.tolist()
            )
        )

        total_added = 0

        for current_date in (
            unique_add_dates
        ):

            day_mask = (
                add_dates
                == current_date
            )

            day_ids = (
                add_ids[
                    day_mask
                ]
            )

            day_vectors = (
                add_vectors[
                    day_mask
                ]
            )

            hnsw_index.add_with_ids(
                day_vectors,
                day_ids,
            )

            total_added += len(
                day_ids
            )

        incremental_seconds = (
            perf_counter()
            - incremental_start
        )

        expected_ntotal = (
            len(rebuild_ids)
            + len(add_ids)
        )

        if (
            hnsw_index.ntotal
            != expected_ntotal
        ):

            raise RuntimeError(
                "Unexpected HNSW size: "
                f"{hnsw_index.ntotal:,}; "
                f"expected "
                f"{expected_ntotal:,}."
            )

        if total_added != len(
            add_ids
        ):

            raise RuntimeError(
                "Incremental add count "
                "mismatch."
            )

        print()
        print("=" * 110)
        print("7. HNSW INCREMENTAL ADD")
        print("=" * 110)

        print(
            f"\nDays added:           "
            f"{len(unique_add_dates)}"
        )

        print(
            f"Vectors added:        "
            f"{total_added:,}"
        )

        print(
            f"Final index size:     "
            f"{hnsw_index.ntotal:,}"
        )

        print(
            f"Incremental add time: "
            f"{incremental_seconds:.3f} s"
        )

        # =====================================================================
        # 9. TEMPORAL COMPOSITION
        # =====================================================================

        expired_mask = (
            index_dates
            < exact_start_date
        )

        n_expired = int(
            np.sum(
                expired_mask
            )
        )

        n_valid = int(
            np.sum(
                exact_mask
            )
        )

        if (
            n_valid
            + n_expired
            != len(index_ids)
        ):

            raise RuntimeError(
                "Temporal composition does not "
                "sum to total index size."
            )

        print()
        print("=" * 110)
        print("8. INDEX TEMPORAL COMPOSITION")
        print("=" * 110)

        print(
            f"\nValid for target:     "
            f"{n_valid:,}"
        )

        print(
            f"Expired in index:     "
            f"{n_expired:,}"
        )

        print(
            f"Total before target:  "
            f"{len(index_ids):,}"
        )

        print(
            f"Expired share:        "
            f"{n_expired / len(index_ids):.4%}"
        )

        # =====================================================================
        # 10. OVERSAMPLING + TEMPORAL FILTER
        # =====================================================================

        print()
        print("=" * 110)
        print("9. OVERSAMPLING + TEMPORAL FILTER")
        print("=" * 110)

        print(
            "\nK'   | Complete | Recall@10 | Perfect | "
            "Mean |ΔN| | P95 |ΔN| | Max |ΔN| | "
            "Expired raw | Search ms/q"
        )

        print("-" * 110)

        exact_sets = [
            set(
                exact_neighbor_ids[
                    i
                ].tolist()
            )
            for i
            in range(
                EXPECTED_TARGETS
            )
        ]

        results = {}

        for k_prime in (
            OVERSAMPLING_VALUES
        ):

            hnsw_base.hnsw.efSearch = max(
                HNSW_EF_SEARCH,
                k_prime,
            )

            search_start = (
                perf_counter()
            )

            raw_D, raw_I = (
                hnsw_index.search(
                    target_vectors,
                    k_prime,
                )
            )

            search_seconds = (
                perf_counter()
                - search_start
            )

            validate_faiss_results(
                indices=raw_I,
                scores=raw_D,
                n_queries=EXPECTED_TARGETS,
                requested_k=k_prime,
                label=f"HNSW K'={k_prime}",
            )

            filtered_ids = []
            filtered_sims = []

            complete = np.zeros(
                EXPECTED_TARGETS,
                dtype=bool,
            )

            expired_raw_counts = (
                np.zeros(
                    EXPECTED_TARGETS,
                    dtype=np.int64,
                )
            )

            for i in range(
                EXPECTED_TARGETS
            ):

                valid_ids_i = []
                valid_sims_i = []

                seen_ids = set()

                for (
                    candidate_id,
                    similarity,
                ) in zip(
                    raw_I[i],
                    raw_D[i],
                ):

                    candidate_id = int(
                        candidate_id
                    )

                    if candidate_id < 0:
                        continue

                    if candidate_id in seen_ids:

                        raise RuntimeError(
                            "Duplicate candidate ID "
                            "returned by HNSW."
                        )

                    seen_ids.add(
                        candidate_id
                    )

                    if candidate_id not in (
                        id_to_date
                    ):

                        raise RuntimeError(
                            "HNSW returned unknown "
                            f"article_id="
                            f"{candidate_id}."
                        )

                    candidate_date = (
                        id_to_date[
                            candidate_id
                        ]
                    )

                    if (
                        candidate_date
                        < exact_start_date
                    ):

                        expired_raw_counts[
                            i
                        ] += 1

                        continue

                    if (
                        candidate_date
                        >= TARGET_DATE
                    ):

                        continue

                    valid_ids_i.append(
                        candidate_id
                    )

                    valid_sims_i.append(
                        float(
                            np.clip(
                                similarity,
                                -1.0,
                                1.0,
                            )
                        )
                    )

                    if len(
                        valid_ids_i
                    ) == K:

                        break

                if len(
                    valid_ids_i
                ) == K:

                    complete[i] = True

                    filtered_ids.append(
                        np.asarray(
                            valid_ids_i,
                            dtype=np.int64,
                        )
                    )

                    filtered_sims.append(
                        np.asarray(
                            valid_sims_i,
                            dtype=np.float32,
                        )
                    )

                else:

                    filtered_ids.append(
                        None
                    )

                    filtered_sims.append(
                        None
                    )

            valid_target_indices = (
                np.flatnonzero(
                    complete
                )
            )

            recalls = []
            novelty_errors = []

            for i in (
                valid_target_indices
            ):

                approximate_set = set(
                    filtered_ids[
                        i
                    ].tolist()
                )

                recall = (
                    len(
                        exact_sets[
                            i
                        ]
                        & approximate_set
                    )
                    / K
                )

                recalls.append(
                    recall
                )

                approximate_novelty = (
                    1.0
                    - float(
                        np.mean(
                            filtered_sims[
                                i
                            ]
                        )
                    )
                )

                novelty_errors.append(
                    approximate_novelty
                    - exact_novelty[i]
                )

            recalls = np.asarray(
                recalls,
                dtype=np.float64,
            )

            novelty_errors = (
                np.asarray(
                    novelty_errors,
                    dtype=np.float64,
                )
            )

            if len(
                valid_target_indices
            ) > 0:

                mean_recall = float(
                    np.mean(
                        recalls
                    )
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

                abs_errors = np.abs(
                    novelty_errors
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

            else:

                mean_recall = np.nan
                perfect_count = 0
                mean_abs_error = np.nan
                p95_abs_error = np.nan
                max_abs_error = np.nan

            mean_expired_raw = float(
                np.mean(
                    expired_raw_counts
                )
            )

            search_ms_per_query = (
                search_seconds
                / EXPECTED_TARGETS
                * 1_000.0
            )

            results[k_prime] = {
                "complete":
                    complete,
                "recalls":
                    recalls,
                "novelty_errors":
                    novelty_errors,
                "expired_raw_counts":
                    expired_raw_counts,
                "filtered_ids":
                    filtered_ids,
            }

            print(
                f"{k_prime:>4} | "
                f"{int(np.sum(complete)):>3}/"
                f"{EXPECTED_TARGETS:<3} | "
                f"{mean_recall:>9.4f} | "
                f"{perfect_count:>3}/"
                f"{len(valid_target_indices):<3} | "
                f"{mean_abs_error:>9.7f} | "
                f"{p95_abs_error:>8.7f} | "
                f"{max_abs_error:>8.7f} | "
                f"{mean_expired_raw:>11.2f} | "
                f"{search_ms_per_query:>11.4f}"
            )

        # =====================================================================
        # 11. DETAILED WORST CASES
        # =====================================================================

        print()
        print("=" * 110)
        print("10. DETAILED WORST CASES")
        print("=" * 110)

        for k_prime in (
            OVERSAMPLING_VALUES
        ):

            result = results[
                k_prime
            ]

            complete = result[
                "complete"
            ]

            valid_indices = (
                np.flatnonzero(
                    complete
                )
            )

            print()
            print(
                f"K' = {k_prime}"
            )

            print(
                f"Complete targets:     "
                f"{int(np.sum(complete))}/"
                f"{EXPECTED_TARGETS}"
            )

            print(
                f"Max expired in raw:   "
                f"{int(np.max(result['expired_raw_counts']))}"
            )

            if len(
                valid_indices
            ) == 0:

                print(
                    "No complete target."
                )

                continue

            errors = result[
                "novelty_errors"
            ]

            worst_local = int(
                np.argmax(
                    np.abs(
                        errors
                    )
                )
            )

            worst_target_idx = int(
                valid_indices[
                    worst_local
                ]
            )

            approximate_ids = result[
                "filtered_ids"
            ][
                worst_target_idx
            ]

            worst_recall = (
                len(
                    exact_sets[
                        worst_target_idx
                    ]
                    & set(
                        approximate_ids.tolist()
                    )
                )
                / K
            )

            print(
                f"Worst article:        "
                f"{target_ids[worst_target_idx]}"
            )

            print(
                f"Worst source:         "
                f"{target_sources[worst_target_idx]}"
            )

            print(
                f"Worst title:          "
                f"{target_titles[worst_target_idx]}"
            )

            print(
                f"Exact novelty:        "
                f"{exact_novelty[worst_target_idx]:.8f}"
            )

            print(
                f"Novelty error:        "
                f"{errors[worst_local]:+.8f}"
            )

            print(
                f"Recall@{K}:            "
                f"{worst_recall:.4f}"
            )

        # =====================================================================
        # 12. FINAL SUMMARY
        # =====================================================================

        print()
        print("=" * 110)
        print("11. FINAL SUMMARY")
        print("=" * 110)

        print(
            f"\nTargets tested:       "
            f"{EXPECTED_TARGETS}"
        )

        print(
            f"Valid candidates:     "
            f"{n_valid:,}"
        )

        print(
            f"Expired vectors:      "
            f"{n_expired:,}"
        )

        print(
            f"HNSW final size:      "
            f"{hnsw_index.ntotal:,}"
        )

        print(
            f"HNSW build time:      "
            f"{hnsw_build_seconds:.3f} s"
        )

        print(
            f"Incremental add time: "
            f"{incremental_seconds:.3f} s"
        )

        print(
            f"Total runtime:        "
            f"{perf_counter() - total_start:.3f} s"
        )

        print()
        print(
            "DATABASE STATUS: READ ONLY"
        )

        print("=" * 110)

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
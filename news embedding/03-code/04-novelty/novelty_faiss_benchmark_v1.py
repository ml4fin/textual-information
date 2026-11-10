#!/usr/bin/env python3

"""
STEP 4 — REAL-CORPUS FAISS NOVELTY VALIDATION BENCHMARK
=======================================================

Purpose
-------
Compare exact and approximate nearest-neighbor retrieval for one
deterministic real article using the frozen MiniLM document embeddings.

Embedding model:
    all-MiniLM-L6-v2_chunk254_mean_l2_v1

Candidate set:
    strictly historical articles from the previous 365 days

Similarity:
    cosine similarity, implemented as inner product on
    L2-normalized embeddings

Exact reference:
    FAISS IndexFlatIP

Approximate method:
    FAISS IndexHNSWFlat

Novelty diagnostic:
    1 - mean(top-K cosine similarity)

IMPORTANT
---------
This is a validation benchmark, not an HNSW parameter-selection study.

The HNSW parameters used here are provisional and are not final
methodological choices.

No database data are modified.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
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

WINDOW_DAYS = 365
TOP_K = 10
FETCH_SIZE = 5_000

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

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
            PARTITION BY DATE_TRUNC(
                'month',
                a.publication_date
            )
            ORDER BY md5(
                a.article_id::text
            )
        ) AS rn
    FROM risk_emergence.v_valid_articles AS a
    JOIN risk_emergence.news_embeddings AS e
        ON e.article_id = a.article_id
    WHERE e.model_name = %s
      AND a.publication_date >= DATE '2015-01-01'
)
SELECT
    article_id,
    publication_date,
    source_code,
    title,
    embedding
FROM ranked
WHERE rn = 1
ORDER BY publication_date
LIMIT 1;
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

def validate_vector(
    vector,
    label,
):

    if vector.shape != (
        EXPECTED_DIMENSION,
    ):

        raise RuntimeError(
            f"{label}: unexpected shape "
            f"{vector.shape}; expected "
            f"({EXPECTED_DIMENSION},)."
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
            f"{label}: expected L2-normalized "
            f"vector, found norm={norm:.8f}."
        )

    return norm


def validate_matrix(
    matrix,
    expected_rows,
    label,
):

    expected_shape = (
        expected_rows,
        EXPECTED_DIMENSION,
    )

    if matrix.shape != expected_shape:

        raise RuntimeError(
            f"{label}: unexpected shape "
            f"{matrix.shape}; expected "
            f"{expected_shape}."
        )

    if not np.isfinite(
        matrix
    ).all():

        raise RuntimeError(
            f"{label}: NaN or Inf detected."
        )

    norms = np.linalg.norm(
        matrix,
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
        float(
            np.max(
                np.abs(
                    norms - 1.0
                )
            )
        ),
    )


def validate_search_results(
    positions,
    similarities,
    candidate_count,
    label,
):

    if positions.shape != (
        TOP_K,
    ):

        raise RuntimeError(
            f"{label}: unexpected index "
            f"shape {positions.shape}."
        )

    if similarities.shape != (
        TOP_K,
    ):

        raise RuntimeError(
            f"{label}: unexpected score "
            f"shape {similarities.shape}."
        )

    if np.any(
        positions < 0
    ):

        raise RuntimeError(
            f"{label}: FAISS returned "
            "invalid neighbor indices."
        )

    if np.any(
        positions >= candidate_count
    ):

        raise RuntimeError(
            f"{label}: FAISS returned an "
            "out-of-range neighbor index."
        )

    if not np.isfinite(
        similarities
    ).all():

        raise RuntimeError(
            f"{label}: non-finite "
            "similarities returned."
        )

    if len(
        np.unique(
            positions
        )
    ) != TOP_K:

        raise RuntimeError(
            f"{label}: duplicate neighbors "
            "returned in top-K."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    total_start = perf_counter()

    print("=" * 100)
    print(
        "STEP 4 — REAL-CORPUS FAISS "
        "NOVELTY VALIDATION BENCHMARK"
    )
    print("=" * 100)

    print(
        f"\nEmbedding model:       "
        f"{MODEL_NAME}"
    )

    print(
        f"History window:        "
        f"{WINDOW_DAYS} days"
    )

    print(
        f"Top-K:                 "
        f"{TOP_K}"
    )

    print(
        f"HNSW M:                "
        f"{HNSW_M}"
    )

    print(
        f"HNSW efConstruction:   "
        f"{HNSW_EF_CONSTRUCTION}"
    )

    print(
        f"HNSW efSearch:         "
        f"{HNSW_EF_SEARCH}"
    )

    conn = get_connection()

    try:

        # =====================================================================
        # 1. LOAD DETERMINISTIC REAL TARGET
        # =====================================================================

        print()
        print("=" * 100)
        print("1. LOAD DETERMINISTIC REAL TARGET")
        print("=" * 100)

        with conn.cursor() as cur:

            cur.execute(
                TARGET_QUERY,
                (MODEL_NAME,),
            )

            row = cur.fetchone()

        if row is None:

            raise RuntimeError(
                "No eligible target article found."
            )

        (
            target_id,
            target_date,
            target_source,
            target_title,
            target_embedding,
        ) = row

        target_id = int(
            target_id
        )

        target_vector_1d = np.asarray(
            target_embedding,
            dtype=np.float32,
        )

        target_norm = validate_vector(
            target_vector_1d,
            "target",
        )

        target_vector = (
            target_vector_1d
            .reshape(1, -1)
        )

        print(
            f"\nTarget ID:             "
            f"{target_id}"
        )

        print(
            f"Target date:           "
            f"{target_date}"
        )

        print(
            f"Source:                "
            f"{target_source}"
        )

        print(
            f"Title:                 "
            f"{target_title}"
        )

        print(
            f"Target shape:          "
            f"{target_vector.shape}"
        )

        print(
            f"Target norm:           "
            f"{target_norm:.8f}"
        )

        # =====================================================================
        # 2. LOAD STRICTLY HISTORICAL CANDIDATES
        # =====================================================================

        print()
        print("=" * 100)
        print("2. LOAD STRICTLY HISTORICAL CANDIDATES")
        print("=" * 100)

        history_start = (
            target_date
            - timedelta(
                days=WINDOW_DAYS
            )
        )

        print(
            f"\nHistory interval:      "
            f"{history_start} <= date < "
            f"{target_date}"
        )

        history_ids_parts = []
        history_vectors_parts = []

        stream = conn.cursor(
            name=(
                "faiss_novelty_"
                "history_stream"
            )
        )

        stream.itersize = FETCH_SIZE

        history_load_start = (
            perf_counter()
        )

        try:

            stream.execute(
                HISTORY_QUERY,
                (
                    MODEL_NAME,
                    history_start,
                    target_date,
                ),
            )

            history_rows = 0

            while True:

                rows_batch = stream.fetchmany(
                    FETCH_SIZE
                )

                if not rows_batch:
                    break

                ids_batch = np.asarray(
                    [
                        row[0]
                        for row
                        in rows_batch
                    ],
                    dtype=np.int64,
                )

                vectors_batch = np.asarray(
                    [
                        row[2]
                        for row
                        in rows_batch
                    ],
                    dtype=np.float32,
                )

                if vectors_batch.ndim != 2:

                    raise RuntimeError(
                        "Unexpected history "
                        "embedding structure."
                    )

                history_ids_parts.append(
                    ids_batch
                )

                history_vectors_parts.append(
                    vectors_batch
                )

                history_rows += len(
                    rows_batch
                )

        finally:

            stream.close()

        history_load_seconds = (
            perf_counter()
            - history_load_start
        )

        if history_rows < TOP_K:

            raise RuntimeError(
                "Insufficient historical "
                f"candidates: {history_rows:,}; "
                f"need at least {TOP_K}."
            )

        history_ids = np.concatenate(
            history_ids_parts
        )

        history_vectors = np.vstack(
            history_vectors_parts
        ).astype(
            np.float32,
            copy=False,
        )

        del history_ids_parts
        del history_vectors_parts

        if len(
            np.unique(
                history_ids
            )
        ) != history_rows:

            raise RuntimeError(
                "Duplicate article IDs detected "
                "in the historical candidate set."
            )

        (
            norm_min,
            norm_max,
            max_norm_error,
        ) = validate_matrix(
            matrix=history_vectors,
            expected_rows=history_rows,
            label="history",
        )

        print(
            f"\nCandidates:            "
            f"{history_rows:,}"
        )

        print(
            f"History matrix:        "
            f"{history_vectors.shape}"
        )

        print(
            f"History load time:     "
            f"{history_load_seconds:.3f} s"
        )

        print(
            f"History norm range:    "
            f"{norm_min:.8f} -> "
            f"{norm_max:.8f}"
        )

        print(
            f"Max norm error:        "
            f"{max_norm_error:.10f}"
        )

        # =====================================================================
        # 3. EXACT REFERENCE — INDEXFLATIP
        # =====================================================================

        print()
        print("=" * 100)
        print("3. EXACT REFERENCE — INDEXFLATIP")
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

        if exact_index.ntotal != history_rows:

            raise RuntimeError(
                "Exact index size mismatch."
            )

        exact_search_start = (
            perf_counter()
        )

        exact_D, exact_I = (
            exact_index.search(
                target_vector,
                TOP_K,
            )
        )

        exact_search_seconds = (
            perf_counter()
            - exact_search_start
        )

        exact_positions = (
            exact_I[0]
        )

        exact_sims_raw = (
            exact_D[0]
        )

        validate_search_results(
            positions=exact_positions,
            similarities=exact_sims_raw,
            candidate_count=history_rows,
            label="exact",
        )

        exact_sims = np.clip(
            exact_sims_raw,
            -1.0,
            1.0,
        )

        exact_ids = history_ids[
            exact_positions
        ]

        exact_mean_similarity = float(
            np.mean(
                exact_sims
            )
        )

        exact_novelty = (
            1.0
            - exact_mean_similarity
        )

        # =====================================================================
        # 4. APPROXIMATE SEARCH — HNSW
        # =====================================================================

        print()
        print("=" * 100)
        print("4. APPROXIMATE SEARCH — HNSW")
        print("=" * 100)

        hnsw_build_start = (
            perf_counter()
        )

        hnsw_index = faiss.IndexHNSWFlat(
            EXPECTED_DIMENSION,
            HNSW_M,
            faiss.METRIC_INNER_PRODUCT,
        )

        hnsw_index.hnsw.efConstruction = (
            HNSW_EF_CONSTRUCTION
        )

        hnsw_index.add(
            history_vectors
        )

        hnsw_build_seconds = (
            perf_counter()
            - hnsw_build_start
        )

        if hnsw_index.ntotal != history_rows:

            raise RuntimeError(
                "HNSW index size mismatch."
            )

        hnsw_index.hnsw.efSearch = (
            HNSW_EF_SEARCH
        )

        hnsw_search_start = (
            perf_counter()
        )

        hnsw_D, hnsw_I = (
            hnsw_index.search(
                target_vector,
                TOP_K,
            )
        )

        hnsw_search_seconds = (
            perf_counter()
            - hnsw_search_start
        )

        hnsw_positions = (
            hnsw_I[0]
        )

        hnsw_sims_raw = (
            hnsw_D[0]
        )

        validate_search_results(
            positions=hnsw_positions,
            similarities=hnsw_sims_raw,
            candidate_count=history_rows,
            label="HNSW",
        )

        hnsw_sims = np.clip(
            hnsw_sims_raw,
            -1.0,
            1.0,
        )

        hnsw_ids = history_ids[
            hnsw_positions
        ]

        hnsw_mean_similarity = float(
            np.mean(
                hnsw_sims
            )
        )

        hnsw_novelty = (
            1.0
            - hnsw_mean_similarity
        )

        # =====================================================================
        # 5. RETRIEVAL ACCURACY
        # =====================================================================

        exact_id_set = set(
            exact_ids.tolist()
        )

        hnsw_id_set = set(
            hnsw_ids.tolist()
        )

        overlap = len(
            exact_id_set
            & hnsw_id_set
        )

        recall_at_k = (
            overlap
            / TOP_K
        )

        similarity_error = (
            hnsw_mean_similarity
            - exact_mean_similarity
        )

        novelty_error = (
            hnsw_novelty
            - exact_novelty
        )

        absolute_novelty_error = abs(
            novelty_error
        )

        # =====================================================================
        # 6. RESULTS
        # =====================================================================

        print()
        print("=" * 100)
        print(f"EXACT TOP-{TOP_K}")
        print("=" * 100)

        for rank in range(
            TOP_K
        ):

            print(
                f"{rank + 1:>2} | "
                f"article_id="
                f"{int(exact_ids[rank]):>10} | "
                f"sim="
                f"{float(exact_sims[rank]):.8f}"
            )

        print()
        print("=" * 100)
        print(f"HNSW TOP-{TOP_K}")
        print("=" * 100)

        for rank in range(
            TOP_K
        ):

            article_id = int(
                hnsw_ids[rank]
            )

            marker = (
                "MATCH"
                if article_id
                in exact_id_set
                else "MISS"
            )

            print(
                f"{rank + 1:>2} | "
                f"article_id="
                f"{article_id:>10} | "
                f"sim="
                f"{float(hnsw_sims[rank]):.8f} | "
                f"{marker}"
            )

        print()
        print("=" * 100)
        print("COMPARISON")
        print("=" * 100)

        print(
            f"\nExact mean similarity: "
            f"{exact_mean_similarity:.8f}"
        )

        print(
            f"HNSW mean similarity:  "
            f"{hnsw_mean_similarity:.8f}"
        )

        print(
            f"Exact novelty:         "
            f"{exact_novelty:.8f}"
        )

        print(
            f"HNSW novelty:          "
            f"{hnsw_novelty:.8f}"
        )

        print(
            f"Overlap:               "
            f"{overlap}/{TOP_K}"
        )

        print(
            f"Recall@{TOP_K}:             "
            f"{recall_at_k:.4f}"
        )

        print(
            f"Similarity error:      "
            f"{similarity_error:+.8f}"
        )

        print(
            f"Novelty error:         "
            f"{novelty_error:+.8f}"
        )

        print(
            f"Absolute novelty error:"
            f" {absolute_novelty_error:.8f}"
        )

        print()
        print(
            f"Exact build:           "
            f"{exact_build_seconds:.3f} s"
        )

        print(
            f"Exact search:          "
            f"{exact_search_seconds:.6f} s"
        )

        print(
            f"HNSW build:            "
            f"{hnsw_build_seconds:.3f} s"
        )

        print(
            f"HNSW search:           "
            f"{hnsw_search_seconds:.6f} s"
        )

        if hnsw_search_seconds > 0:

            print(
                f"Search speed ratio:    "
                f"{exact_search_seconds / hnsw_search_seconds:.2f}x "
                f"(exact/HNSW)"
            )

        print(
            f"\nTotal runtime:         "
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
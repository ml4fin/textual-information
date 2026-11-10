#!/usr/bin/env python3

"""
STEP 4 — FAISS HNSW ENVIRONMENT SMOKE TEST
===========================================

Purpose
-------
Verify that the local FAISS installation can create and query
an HNSW index compatible with the frozen MiniLM document embeddings.

Configuration:
- dimension: 384
- index: IndexHNSWFlat
- HNSW M: 32
- metric: inner product

Because vectors are L2 normalized, inner product corresponds
to cosine similarity.

IMPORTANT
---------
This test uses deterministic synthetic vectors only.

No PostgreSQL connection is opened.
No production embeddings are read or modified.
No persistent FAISS index is created.
"""

from __future__ import annotations

import sys

import faiss
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSION = 384
HNSW_M = 32

N_VECTORS = 1_000
N_QUERIES = 5
TOP_K = 10

RANDOM_SEED = 12_345

SELF_SIMILARITY_TOLERANCE = 1e-5


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STEP 4 — FAISS HNSW ENVIRONMENT SMOKE TEST")
    print("=" * 80)

    # =========================================================================
    # 1. ENVIRONMENT
    # =========================================================================

    print()
    print("1. CHECK ENVIRONMENT")
    print("-" * 80)

    print(
        f"FAISS version:       "
        f"{faiss.__version__}"
    )

    print(
        f"NumPy version:       "
        f"{np.__version__}"
    )

    print(
        f"Embedding dimension: "
        f"{DIMENSION}"
    )

    print(
        f"HNSW M:              "
        f"{HNSW_M}"
    )

    # =========================================================================
    # 2. CREATE INDEX
    # =========================================================================

    print()
    print("2. CREATE HNSW INDEX")
    print("-" * 80)

    index = faiss.IndexHNSWFlat(
        DIMENSION,
        HNSW_M,
        faiss.METRIC_INNER_PRODUCT,
    )

    print(
        f"Index type:          "
        f"{type(index).__name__}"
    )

    print(
        f"Metric type:         "
        f"{index.metric_type}"
    )

    print(
        f"Initial size:        "
        f"{index.ntotal}"
    )

    if index.d != DIMENSION:

        raise RuntimeError(
            "Unexpected FAISS index dimension: "
            f"expected {DIMENSION}, "
            f"found {index.d}."
        )

    if (
        index.metric_type
        != faiss.METRIC_INNER_PRODUCT
    ):

        raise RuntimeError(
            "Unexpected FAISS metric type."
        )

    if index.ntotal != 0:

        raise RuntimeError(
            "New FAISS index is not empty."
        )

    # =========================================================================
    # 3. GENERATE SYNTHETIC VECTORS
    # =========================================================================

    print()
    print("3. GENERATE SYNTHETIC VECTORS")
    print("-" * 80)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    vectors = rng.normal(
        size=(
            N_VECTORS,
            DIMENSION,
        )
    ).astype(
        np.float32
    )

    if vectors.shape != (
        N_VECTORS,
        DIMENSION,
    ):

        raise RuntimeError(
            "Unexpected synthetic vector shape."
        )

    if not np.isfinite(
        vectors
    ).all():

        raise RuntimeError(
            "Synthetic vectors contain NaN or Inf."
        )

    faiss.normalize_L2(
        vectors
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
    ):

        raise RuntimeError(
            "Synthetic vector normalization failed."
        )

    print(
        f"Vectors generated:   "
        f"{N_VECTORS:,}"
    )

    print(
        f"Vector dtype:        "
        f"{vectors.dtype}"
    )

    print(
        f"Max norm error:      "
        f"{np.max(np.abs(norms - 1.0)):.10f}"
    )

    # =========================================================================
    # 4. ADD TO INDEX
    # =========================================================================

    print()
    print("4. ADD VECTORS TO INDEX")
    print("-" * 80)

    index.add(
        vectors
    )

    print(
        f"Indexed vectors:     "
        f"{index.ntotal:,}"
    )

    if index.ntotal != N_VECTORS:

        raise RuntimeError(
            "Unexpected number of indexed vectors: "
            f"expected {N_VECTORS:,}, "
            f"found {index.ntotal:,}."
        )

    # =========================================================================
    # 5. SEARCH
    # =========================================================================

    print()
    print("5. SEARCH INDEX")
    print("-" * 80)

    queries = vectors[
        :N_QUERIES
    ].copy()

    similarities, indices = index.search(
        queries,
        TOP_K,
    )

    expected_shape = (
        N_QUERIES,
        TOP_K,
    )

    print(
        f"Index result shape:  "
        f"{indices.shape}"
    )

    print(
        f"Score result shape:  "
        f"{similarities.shape}"
    )

    if indices.shape != expected_shape:

        raise RuntimeError(
            "Unexpected FAISS index-result shape: "
            f"{indices.shape}; "
            f"expected {expected_shape}."
        )

    if similarities.shape != expected_shape:

        raise RuntimeError(
            "Unexpected FAISS score-result shape: "
            f"{similarities.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(
        similarities
    ).all():

        raise RuntimeError(
            "FAISS search returned NaN or Inf."
        )

    # =========================================================================
    # 6. SELF-MATCH VALIDATION
    # =========================================================================

    print()
    print("6. SELF-MATCH VALIDATION")
    print("-" * 80)

    expected_self_indices = np.arange(
        N_QUERIES,
        dtype=indices.dtype,
    )

    returned_self_indices = indices[
        :,
        0,
    ]

    returned_self_scores = similarities[
        :,
        0,
    ]

    print(
        "Expected top indices: "
        f"{expected_self_indices.tolist()}"
    )

    print(
        "Returned top indices: "
        f"{returned_self_indices.tolist()}"
    )

    print(
        "Top similarities:     "
        + ", ".join(
            f"{value:.6f}"
            for value
            in returned_self_scores
        )
    )

    if not np.array_equal(
        returned_self_indices,
        expected_self_indices,
    ):

        raise RuntimeError(
            "HNSW self-match validation failed."
        )

    if not np.allclose(
        returned_self_scores,
        1.0,
        atol=SELF_SIMILARITY_TOLERANCE,
    ):

        raise RuntimeError(
            "Unexpected self-match similarity."
        )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print()
    print("=" * 80)
    print("FAISS HNSW SMOKE TEST: PASS")
    print("=" * 80)

    print()
    print(
        "FAISS successfully created and queried "
        "an HNSW inner-product index using "
        "384-dimensional L2-normalized vectors."
    )

    print(
        "No database data or production "
        "embeddings were modified."
    )


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
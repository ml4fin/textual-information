#!/usr/bin/env python3

"""
STEP 4 — EMBEDDING MODEL ENVIRONMENT TEST
==========================================

Purpose
-------
Verify that the local environment can load and run the two
candidate SentenceTransformer models used in the embedding study:

1. sentence-transformers/all-MiniLM-L6-v2
2. sentence-transformers/all-mpnet-base-v2

Checks:
- PyTorch installation
- sentence-transformers installation
- available compute device
- model loading
- embedding dimensionality
- maximum sequence length
- basic inference
- numerical validity
- L2 normalization

IMPORTANT
---------
No PostgreSQL connection is opened.
No database data are modified.
"""

from __future__ import annotations

import sys
import time

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_SPECS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimension": 384,
        "max_seq_length": 256,
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "dimension": 768,
        "max_seq_length": 384,
    },
}

TEST_TEXTS = [
    (
        "Cybersecurity risks increased after hackers "
        "targeted the company's computer systems."
    ),
    (
        "Supply chain disruptions caused shortages "
        "and delayed deliveries."
    ),
    (
        "The company reported stronger quarterly "
        "earnings and higher revenue."
    ),
]


# =============================================================================
# HELPERS
# =============================================================================

def select_device(torch):

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def clear_device_cache(
    torch,
    device,
):

    if device == "cuda":
        torch.cuda.empty_cache()

    elif device == "mps":

        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 90)
    print(
        "STEP 4 — EMBEDDING MODEL "
        "ENVIRONMENT TEST"
    )
    print("=" * 90)

    # =========================================================================
    # 1. PACKAGES
    # =========================================================================

    print()
    print("1. CHECK PACKAGES")
    print("-" * 90)

    try:
        import torch

        print(
            f"[OK] torch version: "
            f"{torch.__version__}"
        )

    except Exception as exc:

        raise RuntimeError(
            "PyTorch import failed."
        ) from exc

    try:
        import sentence_transformers

        from sentence_transformers import (
            SentenceTransformer,
        )

        print(
            "[OK] sentence-transformers version: "
            f"{sentence_transformers.__version__}"
        )

    except Exception as exc:

        raise RuntimeError(
            "sentence-transformers import failed."
        ) from exc

    # =========================================================================
    # 2. DEVICE
    # =========================================================================

    print()
    print("2. COMPUTE DEVICE")
    print("-" * 90)

    device = select_device(
        torch
    )

    print(
        f"Selected device: {device}"
    )

    if device == "cuda":

        print(
            "CUDA device: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # =========================================================================
    # 3. MODEL TESTS
    # =========================================================================

    print()
    print("3. LOAD AND TEST MODELS")
    print("-" * 90)

    for (
        model_name,
        expected,
    ) in MODEL_SPECS.items():

        print()
        print("=" * 90)
        print(
            f"MODEL: {model_name}"
        )
        print("=" * 90)

        load_start = (
            time.perf_counter()
        )

        try:

            model = SentenceTransformer(
                model_name,
                device=device,
            )

        except Exception as exc:

            raise RuntimeError(
                "Model loading failed for "
                f"{model_name}."
            ) from exc

        load_seconds = (
            time.perf_counter()
            - load_start
        )

        dimension = int(
            model.get_sentence_embedding_dimension()
        )

        max_seq_length = int(
            model.max_seq_length
        )

        print(
            f"Load time:             "
            f"{load_seconds:.3f} s"
        )

        print(
            f"Embedding dimension:   "
            f"{dimension}"
        )

        print(
            f"Model max_seq_length:  "
            f"{max_seq_length}"
        )

        if (
            dimension
            != expected["dimension"]
        ):

            raise RuntimeError(
                "Unexpected embedding dimension "
                f"for {model_name}: "
                f"expected "
                f"{expected['dimension']}, "
                f"found {dimension}."
            )

        if (
            max_seq_length
            != expected["max_seq_length"]
        ):

            raise RuntimeError(
                "Unexpected max_seq_length "
                f"for {model_name}: "
                f"expected "
                f"{expected['max_seq_length']}, "
                f"found {max_seq_length}."
            )

        # =====================================================================
        # Tiny inference test
        # =====================================================================

        encode_start = (
            time.perf_counter()
        )

        embeddings = model.encode(
            TEST_TEXTS,
            batch_size=len(TEST_TEXTS),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        encode_seconds = (
            time.perf_counter()
            - encode_start
        )

        expected_shape = (
            len(TEST_TEXTS),
            dimension,
        )

        print(
            f"Output shape:          "
            f"{embeddings.shape}"
        )

        print(
            f"Output dtype:          "
            f"{embeddings.dtype}"
        )

        print(
            f"Encoding time:         "
            f"{encode_seconds:.3f} s"
        )

        if (
            embeddings.shape
            != expected_shape
        ):

            raise RuntimeError(
                "Unexpected embedding shape: "
                f"{embeddings.shape}; "
                f"expected {expected_shape}."
            )

        if not np.issubdtype(
            embeddings.dtype,
            np.floating,
        ):

            raise RuntimeError(
                "Embedding output is not "
                "floating-point."
            )

        if not np.isfinite(
            embeddings
        ).all():

            raise RuntimeError(
                "Embedding contains NaN or Inf."
            )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
        )

        print(
            "L2 norms:              "
            + ", ".join(
                f"{value:.6f}"
                for value in norms
            )
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-5,
        ):

            raise RuntimeError(
                "Normalized embeddings "
                "do not have unit norm."
            )

        print(
            "[OK] model test passed"
        )

        del embeddings
        del model

        clear_device_cache(
            torch,
            device,
        )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print()
    print("=" * 90)
    print(
        "ENVIRONMENT TEST COMPLETE"
    )
    print("=" * 90)

    print()
    print(
        "Both candidate embedding models "
        "loaded successfully and produced "
        "valid L2-normalized embeddings."
    )

    print(
        "No database connection was opened."
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
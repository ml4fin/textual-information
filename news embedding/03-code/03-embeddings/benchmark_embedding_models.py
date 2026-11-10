#!/usr/bin/env python3

"""
STEP 4 — REAL-CORPUS EMBEDDING MODEL BENCHMARK
===============================================

Compare two candidate SentenceTransformer models on the same fixed
1,000-article sample from:

    risk_emergence.v_valid_articles

Models
------
1. sentence-transformers/all-MiniLM-L6-v2
2. sentence-transformers/all-mpnet-base-v2

The benchmark measures:
- embedding dimension
- maximum sequence length
- full article token-length distribution
- proportion of articles exceeding max_seq_length
- direct encoding time
- articles per second
- linearly projected direct-encoding time for the full corpus
- raw float8[] storage requirements
- numerical validity
- L2 normalization

Representation
--------------
title + body

IMPORTANT
---------
This benchmark uses the models' standard direct encoding behavior.
Articles longer than max_seq_length are therefore truncated during
model.encode().

The benchmark does NOT estimate the computational cost of a later
full-document chunking pipeline.

READ ONLY.

No embeddings are written to PostgreSQL.
No PostgreSQL data are modified.
No final embedding model is selected or frozen by this script.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np
import psycopg2
import torch

from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIGURATION
# =============================================================================

BENCHMARK_ARTICLES = 1_000
BATCH_SIZE = 32

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]

SCHEMA_NAME = "risk_emergence"
VALID_ARTICLES_VIEW = "v_valid_articles"


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

    return psycopg2.connect(
        **DB_CONFIG
    )


# =============================================================================
# HELPERS
# =============================================================================

def format_seconds(
    seconds,
):

    if (
        seconds is None
        or not np.isfinite(seconds)
        or seconds < 0
    ):
        return "n/a"

    total = int(
        round(seconds)
    )

    hours = (
        total // 3600
    )

    minutes = (
        (total % 3600) // 60
    )

    secs = (
        total % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def format_gib(
    n_bytes,
):

    return (
        n_bytes
        / (1024 ** 3)
    )


def select_device():

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and
        torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def clear_device_cache(
    device,
):

    if device == "cuda":
        torch.cuda.empty_cache()

    elif device == "mps":

        try:
            torch.mps.empty_cache()

        except Exception:
            pass


def fingerprint_ids(
    article_ids,
):

    payload = ",".join(
        str(article_id)
        for article_id in article_ids
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# =============================================================================
# ARTICLE TEXT
# =============================================================================

def safe_text(
    value,
):

    if value is None:
        return ""

    return str(value).strip()


def build_model_text(
    title,
    body,
):

    """
    Build the document representation used consistently across models.

    No manual truncation is applied here. Direct SentenceTransformer
    encoding handles truncation according to model.max_seq_length.
    """

    title = safe_text(
        title
    )

    body = safe_text(
        body
    )

    if title and body:

        return (
            title
            + "\n\n"
            + body
        )

    if title:
        return title

    return body


# =============================================================================
# DATABASE READ
# =============================================================================

def load_full_corpus_count(
    conn,
):

    sql = f"""
        SELECT COUNT(*)
        FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW};
    """

    with conn.cursor() as cur:

        cur.execute(sql)

        count = int(
            cur.fetchone()[0]
        )

    if count <= 0:

        raise RuntimeError(
            "Valid-article corpus is empty."
        )

    return count


def load_articles(
    conn,
):

    """
    Load a fixed deterministic benchmark sample.

    The first 1,000 article_ids are used for reproducibility. This is
    a technical benchmark sample and is not assumed to be statistically
    representative of the complete corpus.
    """

    sql = f"""
        SELECT
            article_id,
            title,
            body
        FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW}
        ORDER BY article_id
        LIMIT %s;
    """

    start = (
        time.perf_counter()
    )

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                BENCHMARK_ARTICLES,
            ),
        )

        rows = (
            cur.fetchall()
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    if (
        len(rows)
        != BENCHMARK_ARTICLES
    ):
        raise RuntimeError(
            f"Expected "
            f"{BENCHMARK_ARTICLES:,} articles, "
            f"found {len(rows):,}."
        )

    article_ids = []
    texts = []

    for (
        article_id,
        title,
        body,
    ) in rows:

        text = build_model_text(
            title,
            body,
        )

        if not text:

            raise RuntimeError(
                "Empty model text for "
                f"article_id={article_id}."
            )

        article_ids.append(
            int(article_id)
        )

        texts.append(
            text
        )

    if (
        len(set(article_ids))
        != BENCHMARK_ARTICLES
    ):
        raise RuntimeError(
            "Duplicate article_id in "
            "benchmark sample."
        )

    return (
        article_ids,
        texts,
        elapsed,
    )


# =============================================================================
# TOKEN LENGTH ANALYSIS
# =============================================================================

def analyze_token_lengths(
    model,
    texts,
):

    tokenizer = (
        model.tokenizer
    )

    max_seq_length = int(
        model.max_seq_length
    )

    lengths = []

    start = (
        time.perf_counter()
    )

    for text in texts:

        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        lengths.append(
            len(
                encoded["input_ids"]
            )
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    lengths = np.asarray(
        lengths,
        dtype=np.int64,
    )

    truncated = (
        lengths
        > max_seq_length
    )

    n_truncated = int(
        truncated.sum()
    )

    truncation_rate = float(
        n_truncated
        / len(lengths)
    )

    return {
        "lengths":
            lengths,

        "max_seq_length":
            max_seq_length,

        "n_truncated":
            n_truncated,

        "truncation_rate":
            truncation_rate,

        "elapsed":
            elapsed,
    }


def print_token_statistics(
    stats,
):

    lengths = (
        stats["lengths"]
    )

    print()
    print(
        "Full article token-length "
        "distribution:"
    )

    print(
        f"  min:       "
        f"{int(lengths.min()):,}"
    )

    print(
        f"  mean:      "
        f"{lengths.mean():,.2f}"
    )

    print(
        f"  median:    "
        f"{np.median(lengths):,.2f}"
    )

    print(
        f"  p75:       "
        f"{np.quantile(lengths, 0.75):,.2f}"
    )

    print(
        f"  p90:       "
        f"{np.quantile(lengths, 0.90):,.2f}"
    )

    print(
        f"  p95:       "
        f"{np.quantile(lengths, 0.95):,.2f}"
    )

    print(
        f"  p99:       "
        f"{np.quantile(lengths, 0.99):,.2f}"
    )

    print(
        f"  max:       "
        f"{int(lengths.max()):,}"
    )

    print()
    print(
        f"Model max_seq_length: "
        f"{stats['max_seq_length']:,}"
    )

    print(
        f"Articles above limit: "
        f"{stats['n_truncated']:,}/"
        f"{len(lengths):,} "
        f"({stats['truncation_rate']:.2%})"
    )

    print(
        f"Token analysis time:  "
        f"{stats['elapsed']:.3f} s"
    )


# =============================================================================
# MODEL BENCHMARK
# =============================================================================

def benchmark_model(
    model_name,
    texts,
    device,
    full_corpus_articles,
):

    print()
    print("=" * 100)
    print(
        f"MODEL: {model_name}"
    )
    print("=" * 100)

    load_start = (
        time.perf_counter()
    )

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    dimension = int(
        model.get_sentence_embedding_dimension()
    )

    print(
        f"\nDevice:               "
        f"{device}"
    )

    print(
        f"Load time:            "
        f"{load_seconds:.3f} s"
    )

    print(
        f"Embedding dimension:  "
        f"{dimension:,}"
    )

    print(
        f"Max sequence length:  "
        f"{model.max_seq_length:,}"
    )

    token_stats = (
        analyze_token_lengths(
            model,
            texts,
        )
    )

    print_token_statistics(
        token_stats
    )

    print(
        f"\nEncoding "
        f"{len(texts):,} articles "
        f"with batch_size="
        f"{BATCH_SIZE:,}..."
    )

    clear_device_cache(
        device
    )

    encode_start = (
        time.perf_counter()
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    encode_seconds = (
        time.perf_counter()
        - encode_start
    )

    expected_shape = (
        len(texts),
        dimension,
    )

    if (
        embeddings.shape
        != expected_shape
    ):
        raise RuntimeError(
            f"Unexpected shape: "
            f"{embeddings.shape}; "
            f"expected {expected_shape}."
        )

    if not np.issubdtype(
        embeddings.dtype,
        np.floating,
    ):
        raise RuntimeError(
            "Embedding array does not "
            "have a floating-point dtype."
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise RuntimeError(
            "Embeddings contain NaN or Inf."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    max_norm_error = float(
        np.max(
            np.abs(
                norms - 1.0
            )
        )
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            "L2 normalization check failed."
        )

    if encode_seconds <= 0:

        raise RuntimeError(
            "Invalid encoding duration."
        )

    articles_per_second = (
        len(texts)
        / encode_seconds
    )

    projected_seconds = (
        full_corpus_articles
        / articles_per_second
    )

    # PostgreSQL float8[] uses 8 bytes per numeric component for the
    # raw vector payload. Array, row and TOAST overhead are excluded.
    raw_bytes_per_embedding = (
        dimension
        * 8
    )

    raw_full_corpus_bytes = (
        full_corpus_articles
        * raw_bytes_per_embedding
    )

    print()
    print(
        "Direct encoding results:"
    )

    print(
        f"  output shape:        "
        f"{embeddings.shape}"
    )

    print(
        f"  dtype returned:      "
        f"{embeddings.dtype}"
    )

    print(
        f"  encoding time:       "
        f"{encode_seconds:.3f} s"
    )

    print(
        f"  articles/second:     "
        f"{articles_per_second:,.2f}"
    )

    print(
        f"  max L2 norm error:   "
        f"{max_norm_error:.10f}"
    )

    print()
    print(
        "Linearly projected full-corpus "
        "DIRECT encoding time:"
    )

    print(
        f"  {projected_seconds:,.1f} s"
    )

    print(
        f"  {format_seconds(projected_seconds)}"
    )

    print(
        "\nThis projection assumes the same "
        "hardware, batch size and workload "
        "characteristics as the benchmark."
    )

    print(
        "It does not represent the cost of "
        "full-document chunked encoding."
    )

    print()
    print(
        "Raw PostgreSQL float8[] "
        "payload estimate:"
    )

    print(
        f"  bytes/vector:        "
        f"{raw_bytes_per_embedding:,}"
    )

    print(
        f"  full corpus:         "
        f"{format_gib(raw_full_corpus_bytes):,.2f} GiB"
    )

    print(
        "  NOTE: PostgreSQL array, row and "
        "TOAST overhead are excluded."
    )

    result = {
        "model_name":
            model_name,

        "dimension":
            dimension,

        "max_seq_length":
            token_stats[
                "max_seq_length"
            ],

        "n_truncated":
            token_stats[
                "n_truncated"
            ],

        "truncation_rate":
            token_stats[
                "truncation_rate"
            ],

        "encode_seconds":
            float(
                encode_seconds
            ),

        "articles_per_second":
            float(
                articles_per_second
            ),

        "projected_seconds":
            float(
                projected_seconds
            ),

        "raw_storage_gib":
            float(
                format_gib(
                    raw_full_corpus_bytes
                )
            ),

        "max_norm_error":
            max_norm_error,

        "embedding_dtype":
            str(
                embeddings.dtype
            ),
    }

    del embeddings
    del model

    clear_device_cache(
        device
    )

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "STEP 4 — REAL-CORPUS "
        "EMBEDDING MODEL BENCHMARK"
    )
    print("=" * 100)

    print(
        "\nREAD ONLY — no embeddings "
        "will be written to PostgreSQL."
    )

    device = (
        select_device()
    )

    print(
        f"\nCompute device: "
        f"{device}"
    )

    # -------------------------------------------------------------------------
    # Load fixed benchmark sample
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("1. LOAD REAL ARTICLES")
    print("=" * 100)

    conn = (
        get_connection()
    )

    try:

        conn.set_session(
            readonly=True,
            autocommit=False,
        )

        full_corpus_articles = (
            load_full_corpus_count(
                conn
            )
        )

        (
            article_ids,
            texts,
            db_read_seconds,
        ) = load_articles(
            conn
        )

    finally:

        conn.rollback()
        conn.close()

    sample_sha256 = (
        fingerprint_ids(
            article_ids
        )
    )

    print(
        f"\nArticles loaded:      "
        f"{len(texts):,}"
    )

    print(
        f"Full corpus articles: "
        f"{full_corpus_articles:,}"
    )

    print(
        f"First article_id:     "
        f"{article_ids[0]:,}"
    )

    print(
        f"Last article_id:      "
        f"{article_ids[-1]:,}"
    )

    print(
        f"DB read time:         "
        f"{db_read_seconds:.3f} s"
    )

    print(
        f"Sample SHA-256:       "
        f"{sample_sha256}"
    )

    # -------------------------------------------------------------------------
    # Benchmark both models on exactly the same texts
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("2. BENCHMARK MODELS")
    print("=" * 100)

    results = []

    for model_name in MODELS:

        result = (
            benchmark_model(
                model_name=model_name,
                texts=texts,
                device=device,
                full_corpus_articles=(
                    full_corpus_articles
                ),
            )
        )

        results.append(
            result
        )

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("3. COMPARISON SUMMARY")
    print("=" * 100)

    for result in results:

        print()
        print(
            result[
                "model_name"
            ]
        )

        print(
            f"  dimension:              "
            f"{result['dimension']:,}"
        )

        print(
            f"  max sequence length:    "
            f"{result['max_seq_length']:,}"
        )

        print(
            f"  articles above limit:   "
            f"{result['n_truncated']:,}/"
            f"{BENCHMARK_ARTICLES:,} "
            f"({result['truncation_rate']:.2%})"
        )

        print(
            f"  articles/second:        "
            f"{result['articles_per_second']:,.2f}"
        )

        print(
            f"  projected direct time:  "
            f"{format_seconds(result['projected_seconds'])}"
        )

        print(
            f"  raw float8[] storage:   "
            f"{result['raw_storage_gib']:,.2f} GiB"
        )

    print()
    print("=" * 100)
    print(
        "BENCHMARK COMPLETE — "
        "DATABASE UNCHANGED"
    )
    print("=" * 100)


# =============================================================================
# ENTRY POINT
# =============================================================================

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
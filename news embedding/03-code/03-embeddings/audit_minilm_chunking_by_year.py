#!/usr/bin/env python3

"""
STEP 4 — MINILM CHUNKING AUDIT BY YEAR
=======================================

Audit document length and expected chunking cost using a reproducible
sample distributed across the full 2014–2023 corpus.

Sampling
--------
- 1,000 valid articles per year
- 10 years
- target sample size: 10,000 articles
- deterministic pseudo-random sampling within each year

Representation
--------------
title + body

Tokenizer
---------
sentence-transformers/all-MiniLM-L6-v2

Chunking
--------
- maximum sequence length detected from the SentenceTransformer model
- content capacity = max_seq_length - required special tokens
- no overlap

The full-corpus chunk projection is stratified by year and weighted by
the actual number of valid articles in each year.

IMPORTANT
---------
Read only.

No embeddings are calculated.
No PostgreSQL data are modified.
"""

from __future__ import annotations

import math
import os
import sys
import time

from collections import Counter

import numpy as np
import psycopg2
import torch

from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

START_YEAR = 2014
END_YEAR = 2023

ARTICLES_PER_YEAR = 1_000

SCHEMA_NAME = "risk_emergence"
VALID_ARTICLES_VIEW = "v_valid_articles"

SAMPLE_KEY = "minilm_chunk_audit_v1"

ORIGINAL_BENCHMARK_MEAN_CHUNKS = 4.166


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
# TEXT
# =============================================================================

def safe_text(
    value,
) -> str:

    if value is None:
        return ""

    return str(value).strip()


def build_model_text(
    title,
    body,
) -> str:

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
# DEVICE
# =============================================================================

def select_device() -> str:

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


# =============================================================================
# CORPUS COUNTS
# =============================================================================

def load_year_counts(
    conn,
) -> dict[int, int]:

    """
    Return the number of valid articles for each audit year.

    These counts are used to convert the equal-per-year audit sample
    into a corpus-weighted chunk-volume projection.
    """

    sql = f"""
        SELECT
            EXTRACT(
                YEAR FROM publication_date
            )::int AS publication_year,
            COUNT(*)::bigint AS article_count
        FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW}
        WHERE publication_date >= %s
          AND publication_date < %s
        GROUP BY 1
        ORDER BY 1;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                f"{START_YEAR}-01-01",
                f"{END_YEAR + 1}-01-01",
            ),
        )

        rows = cur.fetchall()

    counts = {
        int(year): int(count)
        for year, count in rows
    }

    expected_years = set(
        range(
            START_YEAR,
            END_YEAR + 1,
        )
    )

    if (
        set(counts)
        != expected_years
    ):
        raise RuntimeError(
            "Unexpected year coverage in "
            "valid-article corpus counts.\n"
            f"Found: {sorted(counts)}"
        )

    return counts


# =============================================================================
# STRATIFIED SAMPLE
# =============================================================================

def load_stratified_sample(
    conn,
):

    """
    Select a reproducible sample of articles within each year.

    The sample is balanced across years for temporal diagnostics.
    It is therefore not treated as a self-weighting sample when
    projecting total corpus chunk volume.
    """

    rows = []

    start = time.perf_counter()

    with conn.cursor() as cur:

        for year in range(
            START_YEAR,
            END_YEAR + 1,
        ):

            sql = f"""
                SELECT
                    article_id,
                    publication_date,
                    title,
                    body
                FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW}
                WHERE publication_date >= %s
                  AND publication_date < %s
                ORDER BY md5(
                    article_id::text
                    || %s
                )
                LIMIT %s;
            """

            cur.execute(
                sql,
                (
                    f"{year}-01-01",
                    f"{year + 1}-01-01",
                    SAMPLE_KEY,
                    ARTICLES_PER_YEAR,
                ),
            )

            year_rows = (
                cur.fetchall()
            )

            if (
                len(year_rows)
                != ARTICLES_PER_YEAR
            ):
                raise RuntimeError(
                    f"Year {year}: expected "
                    f"{ARTICLES_PER_YEAR:,} "
                    f"articles, found "
                    f"{len(year_rows):,}."
                )

            rows.extend(
                year_rows
            )

            print(
                f"[OK] {year}: "
                f"{len(year_rows):,} "
                "articles loaded"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    return rows, elapsed


# =============================================================================
# TOKENIZATION AUDIT
# =============================================================================

def audit_documents(
    tokenizer,
    rows,
    content_tokens_per_chunk,
):

    records = []

    start = time.perf_counter()

    for index, row in enumerate(
        rows,
        start=1,
    ):

        (
            article_id,
            publication_date,
            title,
            body,
        ) = row

        text = build_model_text(
            title,
            body,
        )

        if not text:

            raise RuntimeError(
                "Empty model text for "
                f"article_id={article_id}."
            )

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        n_tokens = len(
            encoded["input_ids"]
        )

        if n_tokens <= 0:

            raise RuntimeError(
                "Zero tokens for "
                f"article_id={article_id}."
            )

        n_chunks = int(
            math.ceil(
                n_tokens
                / content_tokens_per_chunk
            )
        )

        records.append(
            (
                int(article_id),
                int(
                    publication_date.year
                ),
                int(n_tokens),
                int(n_chunks),
            )
        )

        if index % 1_000 == 0:

            print(
                f"Tokenized "
                f"{index:,}/"
                f"{len(rows):,}"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    return records, elapsed


# =============================================================================
# STATISTICS
# =============================================================================

def summarize(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:

        raise RuntimeError(
            "Cannot summarize an empty array."
        )

    return {
        "n":
            int(len(values)),

        "mean":
            float(
                np.mean(values)
            ),

        "median":
            float(
                np.median(values)
            ),

        "p75":
            float(
                np.quantile(
                    values,
                    0.75,
                )
            ),

        "p90":
            float(
                np.quantile(
                    values,
                    0.90,
                )
            ),

        "p95":
            float(
                np.quantile(
                    values,
                    0.95,
                )
            ),

        "p99":
            float(
                np.quantile(
                    values,
                    0.99,
                )
            ),

        "max":
            float(
                np.max(values)
            ),
    }


def print_summary(
    label,
    stats,
):

    print()
    print(label)

    print(
        f"  n:       "
        f"{stats['n']:,}"
    )

    print(
        f"  mean:    "
        f"{stats['mean']:,.2f}"
    )

    print(
        f"  median:  "
        f"{stats['median']:,.2f}"
    )

    print(
        f"  p75:     "
        f"{stats['p75']:,.2f}"
    )

    print(
        f"  p90:     "
        f"{stats['p90']:,.2f}"
    )

    print(
        f"  p95:     "
        f"{stats['p95']:,.2f}"
    )

    print(
        f"  p99:     "
        f"{stats['p99']:,.2f}"
    )

    print(
        f"  max:     "
        f"{stats['max']:,.0f}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "STEP 4 — MINILM "
        "CHUNKING AUDIT BY YEAR"
    )
    print("=" * 100)

    print(
        "\nREAD ONLY — no PostgreSQL "
        "data will be modified."
    )

    # -------------------------------------------------------------------------
    # Model/tokenizer
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("1. LOAD MINILM TOKENIZER")
    print("=" * 100)

    device = select_device()

    load_start = (
        time.perf_counter()
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    tokenizer = (
        model.tokenizer
    )

    dimension = int(
        model.get_sentence_embedding_dimension()
    )

    max_seq_length = int(
        model.max_seq_length
    )

    special_tokens = int(
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

    content_tokens_per_chunk = (
        max_seq_length
        - special_tokens
    )

    if (
        content_tokens_per_chunk
        <= 0
    ):
        raise RuntimeError(
            "Invalid content-token capacity."
        )

    print(
        f"\nModel:                "
        f"{MODEL_NAME}"
    )

    print(
        f"Device:               "
        f"{device}"
    )

    print(
        f"Load time:            "
        f"{load_seconds:.3f} s"
    )

    print(
        f"Embedding dimension:  "
        f"{dimension}"
    )

    print(
        f"Max sequence length:  "
        f"{max_seq_length}"
    )

    print(
        f"Special tokens:       "
        f"{special_tokens}"
    )

    print(
        f"Content/chunk:        "
        f"{content_tokens_per_chunk}"
    )

    # -------------------------------------------------------------------------
    # Database sample + corpus counts
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("2. LOAD STRATIFIED SAMPLE")
    print("=" * 100)
    print()

    conn = get_connection()

    try:

        conn.set_session(
            readonly=True,
            autocommit=False,
        )

        year_counts = (
            load_year_counts(
                conn
            )
        )

        (
            rows,
            db_seconds,
        ) = (
            load_stratified_sample(
                conn
            )
        )

    finally:

        conn.rollback()
        conn.close()

    expected_total = (
        (
            END_YEAR
            - START_YEAR
            + 1
        )
        * ARTICLES_PER_YEAR
    )

    if (
        len(rows)
        != expected_total
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_total:,} rows, "
            f"found {len(rows):,}."
        )

    full_corpus_articles = int(
        sum(
            year_counts.values()
        )
    )

    print(
        f"\nAudit articles:      "
        f"{len(rows):,}"
    )

    print(
        f"Full corpus articles:"
        f" {full_corpus_articles:,}"
    )

    print(
        f"DB read time:        "
        f"{db_seconds:.3f} s"
    )

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("3. TOKENIZE DOCUMENTS")
    print("=" * 100)
    print()

    (
        records,
        token_seconds,
    ) = audit_documents(
        tokenizer=tokenizer,
        rows=rows,
        content_tokens_per_chunk=(
            content_tokens_per_chunk
        ),
    )

    print(
        f"\nTokenization time: "
        f"{token_seconds:.3f} s"
    )

    # -------------------------------------------------------------------------
    # Overall unweighted sample statistics
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("4. OVERALL SAMPLE RESULTS")
    print("=" * 100)

    token_counts = [
        row[2]
        for row in records
    ]

    chunk_counts = [
        row[3]
        for row in records
    ]

    token_stats = summarize(
        token_counts
    )

    chunk_stats = summarize(
        chunk_counts
    )

    print_summary(
        "TOKEN COUNTS",
        token_stats,
    )

    print_summary(
        "CHUNKS PER ARTICLE",
        chunk_stats,
    )

    chunk_array = np.asarray(
        chunk_counts,
        dtype=np.int64,
    )

    one_chunk = int(
        np.sum(
            chunk_array == 1
        )
    )

    multi_chunk = int(
        np.sum(
            chunk_array > 1
        )
    )

    print(
        f"\nOne-chunk articles:   "
        f"{one_chunk:,}/"
        f"{len(records):,} "
        f"({one_chunk / len(records):.2%})"
    )

    print(
        f"Multi-chunk articles: "
        f"{multi_chunk:,}/"
        f"{len(records):,} "
        f"({multi_chunk / len(records):.2%})"
    )

    # -------------------------------------------------------------------------
    # Year-level statistics
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("5. RESULTS BY YEAR")
    print("=" * 100)
    print()

    header = (
        f"{'YEAR':<6}"
        f"{'CORPUS N':>12}"
        f"{'SAMPLE N':>12}"
        f"{'MEAN TOK':>14}"
        f"{'MED TOK':>14}"
        f"{'P95 TOK':>14}"
        f"{'MEAN CH':>12}"
        f"{'MED CH':>10}"
        f"{'P95 CH':>10}"
        f"{'>1 CHUNK':>14}"
    )

    print(header)
    print("-" * len(header))

    mean_chunks_by_year = {}

    sample_counts = Counter(
        row[1]
        for row in records
    )

    for year in range(
        START_YEAR,
        END_YEAR + 1,
    ):

        year_records = [
            row
            for row in records
            if row[1] == year
        ]

        if (
            len(year_records)
            != ARTICLES_PER_YEAR
        ):
            raise RuntimeError(
                f"Unexpected audit sample "
                f"size for year {year}."
            )

        year_tokens = np.asarray(
            [
                row[2]
                for row in year_records
            ],
            dtype=np.float64,
        )

        year_chunks = np.asarray(
            [
                row[3]
                for row in year_records
            ],
            dtype=np.int64,
        )

        mean_chunks_by_year[
            year
        ] = float(
            year_chunks.mean()
        )

        multi_rate = float(
            np.mean(
                year_chunks > 1
            )
        )

        print(
            f"{year:<6}"
            f"{year_counts[year]:>12,}"
            f"{sample_counts[year]:>12,}"
            f"{year_tokens.mean():>14.2f}"
            f"{np.median(year_tokens):>14.2f}"
            f"{np.quantile(year_tokens, 0.95):>14.2f}"
            f"{year_chunks.mean():>12.3f}"
            f"{np.median(year_chunks):>10.2f}"
            f"{np.quantile(year_chunks, 0.95):>10.2f}"
            f"{multi_rate:>13.2%}"
        )

    # -------------------------------------------------------------------------
    # Corpus-weighted projection
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "6. FULL-CORPUS CHUNK PROJECTION"
    )
    print("=" * 100)

    projected_chunks_by_year = {
        year: (
            year_counts[year]
            * mean_chunks_by_year[year]
        )
        for year in range(
            START_YEAR,
            END_YEAR + 1,
        )
    }

    projected_total_chunks = float(
        sum(
            projected_chunks_by_year.values()
        )
    )

    weighted_mean_chunks = (
        projected_total_chunks
        / full_corpus_articles
    )

    unweighted_sample_mean = float(
        np.mean(
            chunk_array
        )
    )

    print(
        f"\nFull corpus articles:          "
        f"{full_corpus_articles:,}"
    )

    print(
        f"Unweighted audit mean:         "
        f"{unweighted_sample_mean:.4f}"
    )

    print(
        f"Corpus-weighted mean chunks:   "
        f"{weighted_mean_chunks:.4f}"
    )

    print(
        f"Projected total chunks:        "
        f"{projected_total_chunks:,.0f}"
    )

    print(
        "\nProjection method:"
    )

    print(
        "  sum over years of "
        "(actual yearly article count "
        "× sampled yearly mean chunks/article)"
    )

    print(
        "\nThis estimates computational "
        "chunk volume."
    )

    print(
        "Only one final document-level "
        f"{dimension}-dimensional vector "
        "would be stored per article."
    )

    # -------------------------------------------------------------------------
    # Initial benchmark comparison
    # -------------------------------------------------------------------------

    difference = (
        weighted_mean_chunks
        - ORIGINAL_BENCHMARK_MEAN_CHUNKS
    )

    relative_difference = (
        difference
        / ORIGINAL_BENCHMARK_MEAN_CHUNKS
    )

    print()
    print("=" * 100)
    print(
        "7. COMPARISON WITH INITIAL "
        "1,000-ARTICLE BENCHMARK"
    )
    print("=" * 100)

    print(
        f"\nInitial sample mean chunks: "
        f"{ORIGINAL_BENCHMARK_MEAN_CHUNKS:.4f}"
    )

    print(
        f"Corpus-weighted audit mean:  "
        f"{weighted_mean_chunks:.4f}"
    )

    print(
        f"Absolute difference:        "
        f"{difference:+.4f}"
    )

    print(
        f"Relative difference:        "
        f"{relative_difference:+.2%}"
    )

    print()
    print("=" * 100)
    print(
        "AUDIT COMPLETE — "
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
#!/usr/bin/env python3

"""
STEP 4 — MINILM DOCUMENT-LEVEL EMBEDDING BENCHMARK
===================================================

Compare two document representations on the same 1,000 real articles:

A. STANDARD TRUNCATION
   title + body -> MiniLM -> standard model truncation

B. FULL-DOCUMENT CHUNKING
   title + body
       -> tokenize full document
       -> split into model-compatible chunks
       -> encode every chunk
       -> mean pool chunk embeddings
       -> L2 normalize final document embedding

Model
-----
sentence-transformers/all-MiniLM-L6-v2

IMPORTANT
---------
READ ONLY.

No embeddings are written to PostgreSQL.
No PostgreSQL data are modified.
The final embedding specification is not frozen by this script.
"""

from __future__ import annotations

import math
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

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

BENCHMARK_ARTICLES = 1_000
FULL_CORPUS_ARTICLES = 2_249_255
BATCH_SIZE = 32

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

def format_seconds(seconds):

    if (
        seconds is None
        or not np.isfinite(seconds)
        or seconds < 0
    ):
        return "n/a"

    total = int(
        round(seconds)
    )

    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def select_device():

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def clear_device_cache(device):

    if device == "cuda":
        torch.cuda.empty_cache()

    elif device == "mps":

        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def safe_text(value):

    if value is None:
        return ""

    return str(value).strip()


def build_model_text(
    title,
    body,
):

    title = safe_text(title)
    body = safe_text(body)

    if title and body:
        return title + "\n\n" + body

    if title:
        return title

    return body


# =============================================================================
# LOAD ARTICLES
# =============================================================================

def load_articles(conn):

    sql = f"""
        SELECT
            article_id,
            title,
            body
        FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW}
        ORDER BY article_id
        LIMIT %s;
    """

    start = time.perf_counter()

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (BENCHMARK_ARTICLES,),
        )

        rows = cur.fetchall()

    elapsed = (
        time.perf_counter()
        - start
    )

    if len(rows) != BENCHMARK_ARTICLES:

        raise RuntimeError(
            f"Expected {BENCHMARK_ARTICLES:,} "
            f"articles, found {len(rows):,}."
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
                "Empty text for "
                f"article_id={article_id}."
            )

        article_ids.append(
            int(article_id)
        )

        texts.append(
            text
        )

    if len(set(article_ids)) != len(article_ids):

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
# TOKENIZE FULL DOCUMENTS
# =============================================================================

def tokenize_documents(
    tokenizer,
    texts,
):

    """
    Tokenize complete documents without truncation or special tokens.
    """

    all_token_ids = []

    start = time.perf_counter()

    for text in texts:

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        token_ids = encoded[
            "input_ids"
        ]

        if not token_ids:

            raise RuntimeError(
                "Tokenizer produced "
                "an empty document."
            )

        all_token_ids.append(
            token_ids
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        all_token_ids,
        elapsed,
    )


# =============================================================================
# BUILD CHUNKS
# =============================================================================

def build_chunks(
    tokenizer,
    all_token_ids,
    model_max_seq_length,
):

    special_tokens = int(
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

    content_tokens_per_chunk = (
        model_max_seq_length
        - special_tokens
    )

    if content_tokens_per_chunk <= 0:

        raise RuntimeError(
            "Invalid content chunk size."
        )

    all_chunk_texts = []
    article_chunk_ranges = []
    chunks_per_article = []

    start = time.perf_counter()

    for token_ids in all_token_ids:

        start_index = len(
            all_chunk_texts
        )

        n_tokens = len(
            token_ids
        )

        n_chunks = int(
            math.ceil(
                n_tokens
                / content_tokens_per_chunk
            )
        )

        chunks_per_article.append(
            n_chunks
        )

        for chunk_number in range(
            n_chunks
        ):

            left = (
                chunk_number
                * content_tokens_per_chunk
            )

            right = min(
                left
                + content_tokens_per_chunk,
                n_tokens,
            )

            chunk_ids = token_ids[
                left:right
            ]

            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if not chunk_text.strip():

                raise RuntimeError(
                    "Empty chunk after decoding."
                )

            all_chunk_texts.append(
                chunk_text
            )

        end_index = len(
            all_chunk_texts
        )

        article_chunk_ranges.append(
            (
                start_index,
                end_index,
            )
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "chunk_texts":
            all_chunk_texts,

        "article_chunk_ranges":
            article_chunk_ranges,

        "chunks_per_article":
            np.asarray(
                chunks_per_article,
                dtype=np.int64,
            ),

        "special_tokens":
            special_tokens,

        "content_tokens_per_chunk":
            content_tokens_per_chunk,

        "elapsed":
            elapsed,
    }


def validate_chunk_lengths(
    tokenizer,
    chunk_texts,
    model_max_seq_length,
):

    """
    Verify that decoded chunks remain within the model input limit
    after tokenization by SentenceTransformer.
    """

    max_observed = 0

    for chunk_text in chunk_texts:

        encoded = tokenizer(
            chunk_text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        n_tokens = len(
            encoded["input_ids"]
        )

        max_observed = max(
            max_observed,
            n_tokens,
        )

        if n_tokens > model_max_seq_length:

            raise RuntimeError(
                "Decoded chunk exceeds "
                "model.max_seq_length after "
                "re-tokenization: "
                f"{n_tokens} > "
                f"{model_max_seq_length}."
            )

    return max_observed


# =============================================================================
# STANDARD EMBEDDINGS
# =============================================================================

def benchmark_standard_embedding(
    model,
    texts,
    device,
):

    clear_device_cache(
        device
    )

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        embeddings,
        elapsed,
    )


# =============================================================================
# CHUNK EMBEDDINGS
# =============================================================================

def benchmark_chunk_embeddings(
    model,
    chunk_texts,
    device,
):

    clear_device_cache(
        device
    )

    start = time.perf_counter()

    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        chunk_embeddings,
        elapsed,
    )


# =============================================================================
# DOCUMENT POOLING
# =============================================================================

def mean_pool_documents(
    chunk_embeddings,
    article_chunk_ranges,
):

    dimension = int(
        chunk_embeddings.shape[1]
    )

    document_embeddings = np.empty(
        (
            len(article_chunk_ranges),
            dimension,
        ),
        dtype=np.float32,
    )

    start = time.perf_counter()

    for article_index, (
        left,
        right,
    ) in enumerate(
        article_chunk_ranges
    ):

        if right <= left:

            raise RuntimeError(
                "Article has no chunks."
            )

        document_vector = np.mean(
            chunk_embeddings[
                left:right
            ],
            axis=0,
        )

        norm = float(
            np.linalg.norm(
                document_vector
            )
        )

        if (
            not np.isfinite(norm)
            or norm <= 0
        ):

            raise RuntimeError(
                "Invalid pooled "
                "document norm."
            )

        document_embeddings[
            article_index
        ] = (
            document_vector
            / norm
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        document_embeddings,
        elapsed,
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_embeddings(
    embeddings,
    expected_rows,
    expected_dimension,
    label,
):

    expected_shape = (
        expected_rows,
        expected_dimension,
    )

    if embeddings.shape != expected_shape:

        raise RuntimeError(
            f"{label}: unexpected shape "
            f"{embeddings.shape}; expected "
            f"{expected_shape}."
        )

    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            f"{label}: NaN/Inf found."
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
            f"{label}: L2 normalization failed."
        )

    return max_norm_error


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "STEP 4 — MINILM DOCUMENT-LEVEL "
        "EMBEDDING BENCHMARK"
    )
    print("=" * 100)

    print(
        "\nREAD ONLY — PostgreSQL "
        "will not be modified."
    )

    device = select_device()

    print(
        f"\nDevice: {device}"
    )

    # =========================================================================
    # 1. LOAD ARTICLES
    # =========================================================================

    print()
    print("=" * 100)
    print("1. LOAD REAL ARTICLES")
    print("=" * 100)

    conn = get_connection()

    try:

        conn.set_session(
            readonly=True,
            autocommit=False,
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

    print(
        f"\nArticles loaded: "
        f"{len(texts):,}"
    )

    print(
        f"First article_id: "
        f"{article_ids[0]:,}"
    )

    print(
        f"Last article_id:  "
        f"{article_ids[-1]:,}"
    )

    print(
        f"DB read time:     "
        f"{db_read_seconds:.3f} s"
    )

    # =========================================================================
    # 2. LOAD MODEL
    # =========================================================================

    print()
    print("=" * 100)
    print("2. LOAD MINILM")
    print("=" * 100)

    load_start = time.perf_counter()

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    tokenizer = model.tokenizer

    dimension = int(
        model.get_sentence_embedding_dimension()
    )

    max_seq_length = int(
        model.max_seq_length
    )

    print(
        f"\nModel:                "
        f"{MODEL_NAME}"
    )

    print(
        f"Load time:            "
        f"{load_seconds:.3f} s"
    )

    print(
        f"Dimension:            "
        f"{dimension:,}"
    )

    print(
        f"Max sequence length:  "
        f"{max_seq_length:,}"
    )

    # =========================================================================
    # 3. TOKENIZE FULL DOCUMENTS
    # =========================================================================

    print()
    print("=" * 100)
    print("3. TOKENIZE FULL DOCUMENTS")
    print("=" * 100)

    (
        all_token_ids,
        tokenization_seconds,
    ) = tokenize_documents(
        tokenizer,
        texts,
    )

    token_lengths = np.asarray(
        [
            len(token_ids)
            for token_ids in all_token_ids
        ],
        dtype=np.int64,
    )

    print(
        f"\nTokenization time: "
        f"{tokenization_seconds:.3f} s"
    )

    print(
        "\nFull-document token lengths:"
    )

    print(
        f"  min:       "
        f"{token_lengths.min():,}"
    )

    print(
        f"  mean:      "
        f"{token_lengths.mean():,.2f}"
    )

    print(
        f"  median:    "
        f"{np.median(token_lengths):,.2f}"
    )

    print(
        f"  p90:       "
        f"{np.quantile(token_lengths, 0.90):,.2f}"
    )

    print(
        f"  p95:       "
        f"{np.quantile(token_lengths, 0.95):,.2f}"
    )

    print(
        f"  p99:       "
        f"{np.quantile(token_lengths, 0.99):,.2f}"
    )

    print(
        f"  max:       "
        f"{token_lengths.max():,}"
    )

    # =========================================================================
    # 4. BUILD CHUNKS
    # =========================================================================

    print()
    print("=" * 100)
    print("4. BUILD FULL-DOCUMENT CHUNKS")
    print("=" * 100)

    chunk_info = build_chunks(
        tokenizer=tokenizer,
        all_token_ids=all_token_ids,
        model_max_seq_length=max_seq_length,
    )

    chunk_texts = chunk_info[
        "chunk_texts"
    ]

    article_chunk_ranges = chunk_info[
        "article_chunk_ranges"
    ]

    chunks_per_article = chunk_info[
        "chunks_per_article"
    ]

    total_chunks = len(
        chunk_texts
    )

    max_retokenized_length = (
        validate_chunk_lengths(
            tokenizer=tokenizer,
            chunk_texts=chunk_texts,
            model_max_seq_length=max_seq_length,
        )
    )

    print(
        f"\nSpecial tokens/model input: "
        f"{chunk_info['special_tokens']}"
    )

    print(
        f"Content tokens/chunk:       "
        f"{chunk_info['content_tokens_per_chunk']}"
    )

    print(
        f"Maximum re-tokenized chunk: "
        f"{max_retokenized_length}"
    )

    print(
        f"Chunk construction time:    "
        f"{chunk_info['elapsed']:.3f} s"
    )

    print(
        f"\nTotal documents:            "
        f"{len(texts):,}"
    )

    print(
        f"Total chunks:               "
        f"{total_chunks:,}"
    )

    print(
        f"Mean chunks/article:        "
        f"{chunks_per_article.mean():.3f}"
    )

    print(
        f"Median chunks/article:      "
        f"{np.median(chunks_per_article):.2f}"
    )

    print(
        f"P90 chunks/article:         "
        f"{np.quantile(chunks_per_article, 0.90):.2f}"
    )

    print(
        f"P95 chunks/article:         "
        f"{np.quantile(chunks_per_article, 0.95):.2f}"
    )

    print(
        f"P99 chunks/article:         "
        f"{np.quantile(chunks_per_article, 0.99):.2f}"
    )

    print(
        f"Max chunks/article:         "
        f"{chunks_per_article.max():,}"
    )

    one_chunk_articles = int(
        np.sum(
            chunks_per_article == 1
        )
    )

    print(
        f"One-chunk articles:         "
        f"{one_chunk_articles:,}/"
        f"{len(texts):,} "
        f"({one_chunk_articles / len(texts):.2%})"
    )

    # =========================================================================
    # 5. STANDARD TRUNCATION
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "5. A — STANDARD TRUNCATED "
        "DOCUMENT EMBEDDINGS"
    )
    print("=" * 100)

    (
        standard_embeddings,
        standard_seconds,
    ) = benchmark_standard_embedding(
        model,
        texts,
        device,
    )

    standard_norm_error = (
        validate_embeddings(
            standard_embeddings,
            len(texts),
            dimension,
            "standard",
        )
    )

    standard_speed = (
        len(texts)
        / standard_seconds
    )

    standard_projected = (
        FULL_CORPUS_ARTICLES
        / standard_speed
    )

    print(
        f"\nEncoding time:       "
        f"{standard_seconds:.3f} s"
    )

    print(
        f"Articles/second:     "
        f"{standard_speed:,.2f}"
    )

    print(
        f"Max norm error:      "
        f"{standard_norm_error:.10f}"
    )

    print(
        f"Projected corpus:    "
        f"{format_seconds(standard_projected)}"
    )

    # =========================================================================
    # 6. FULL-DOCUMENT CHUNKING
    # =========================================================================

    print()
    print("=" * 100)
    print("6. B — FULL-DOCUMENT CHUNKING")
    print("=" * 100)

    print(
        f"\nEncoding "
        f"{total_chunks:,} chunks..."
    )

    (
        chunk_embeddings,
        chunk_encode_seconds,
    ) = benchmark_chunk_embeddings(
        model,
        chunk_texts,
        device,
    )

    print(
        "\nPooling chunks to "
        "document embeddings..."
    )

    (
        document_embeddings,
        pooling_seconds,
    ) = mean_pool_documents(
        chunk_embeddings,
        article_chunk_ranges,
    )

    chunked_norm_error = (
        validate_embeddings(
            document_embeddings,
            len(texts),
            dimension,
            "chunked",
        )
    )

    chunk_pipeline_seconds = (
        tokenization_seconds
        + chunk_info["elapsed"]
        + chunk_encode_seconds
        + pooling_seconds
    )

    chunk_articles_per_second = (
        len(texts)
        / chunk_pipeline_seconds
    )

    chunk_projected_seconds = (
        FULL_CORPUS_ARTICLES
        / chunk_articles_per_second
    )

    chunks_per_second = (
        total_chunks
        / chunk_encode_seconds
    )

    print(
        f"\nChunk encoding time: "
        f"{chunk_encode_seconds:.3f} s"
    )

    print(
        f"Chunks/second:       "
        f"{chunks_per_second:,.2f}"
    )

    print(
        f"Pooling time:        "
        f"{pooling_seconds:.3f} s"
    )

    print(
        f"Full pipeline time:  "
        f"{chunk_pipeline_seconds:.3f} s"
    )

    print(
        f"Articles/second:     "
        f"{chunk_articles_per_second:,.2f}"
    )

    print(
        f"Max norm error:      "
        f"{chunked_norm_error:.10f}"
    )

    print(
        f"Projected corpus:    "
        f"{format_seconds(chunk_projected_seconds)}"
    )

    # =========================================================================
    # 7. REPRESENTATION COMPARISON
    # =========================================================================

    print()
    print("=" * 100)
    print("7. REPRESENTATION COMPARISON")
    print("=" * 100)

    cosine_same_article = np.sum(
        standard_embeddings
        * document_embeddings,
        axis=1,
    )

    print(
        "\nCosine similarity between "
        "standard truncated and "
        "full-document chunked embeddings:"
    )

    print(
        f"\n  mean:    "
        f"{cosine_same_article.mean():.6f}"
    )

    print(
        f"  median:  "
        f"{np.median(cosine_same_article):.6f}"
    )

    print(
        f"  p10:     "
        f"{np.quantile(cosine_same_article, 0.10):.6f}"
    )

    print(
        f"  p25:     "
        f"{np.quantile(cosine_same_article, 0.25):.6f}"
    )

    print(
        f"  p75:     "
        f"{np.quantile(cosine_same_article, 0.75):.6f}"
    )

    print(
        f"  p90:     "
        f"{np.quantile(cosine_same_article, 0.90):.6f}"
    )

    print(
        f"  min:     "
        f"{cosine_same_article.min():.6f}"
    )

    print(
        f"  max:     "
        f"{cosine_same_article.max():.6f}"
    )

    # =========================================================================
    # 8. FULL-CORPUS PROJECTION
    # =========================================================================

    projected_chunks = (
        FULL_CORPUS_ARTICLES
        * chunks_per_article.mean()
    )

    slowdown = (
        chunk_pipeline_seconds
        / standard_seconds
    )

    print()
    print("=" * 100)
    print("8. FULL-CORPUS PROJECTION")
    print("=" * 100)

    print(
        f"\nCorpus articles:             "
        f"{FULL_CORPUS_ARTICLES:,}"
    )

    print(
        f"Observed chunks/article:     "
        f"{chunks_per_article.mean():.3f}"
    )

    print(
        f"Naive projected chunks:      "
        f"{projected_chunks:,.0f}"
    )

    print(
        "\nA — standard truncation:"
    )

    print(
        f"  projected runtime:         "
        f"{format_seconds(standard_projected)}"
    )

    print(
        "\nB — full-document chunking:"
    )

    print(
        f"  projected runtime:         "
        f"{format_seconds(chunk_projected_seconds)}"
    )

    print(
        f"\nChunking pipeline slowdown:  "
        f"{slowdown:.2f}x"
    )

    # =========================================================================
    # 9. STORAGE
    # =========================================================================

    raw_bytes_per_vector = (
        dimension
        * 8
    )

    raw_storage_bytes = (
        FULL_CORPUS_ARTICLES
        * raw_bytes_per_vector
    )

    raw_storage_gib = (
        raw_storage_bytes
        / (1024 ** 3)
    )

    print()
    print("=" * 100)
    print("9. STORAGE")
    print("=" * 100)

    print(
        "\nFinal database storage uses "
        "one vector per article."
    )

    print(
        f"Embedding dimension:   "
        f"{dimension}"
    )

    print(
        f"Raw float8[] payload:  "
        f"{raw_storage_gib:.2f} GiB"
    )

    print(
        "\nChunks are intermediate only "
        "and are not stored in PostgreSQL."
    )

    print()
    print("=" * 100)
    print(
        "DOCUMENT-LEVEL BENCHMARK "
        "COMPLETE — DATABASE UNCHANGED"
    )
    print("=" * 100)


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
#!/usr/bin/env python3

"""
STEP 4 — FULL-CORPUS MINILM DOCUMENT EMBEDDINGS
================================================

FINAL FROZEN SPECIFICATION
--------------------------
HF model:
    sentence-transformers/all-MiniLM-L6-v2

DB model version:
    all-MiniLM-L6-v2_chunk254_mean_l2_v1

Document:
    title + "\\n\\n" + body

Chunking:
    254 content tokens
    0 overlap

Chunk embeddings:
    L2 normalized

Document aggregation:
    arithmetic mean of chunk embeddings

Final document embedding:
    L2 normalized

Storage:
    risk_emergence.news_embeddings

Primary key:
    (article_id, model_name)

RESUME
------
Only articles missing this MODEL_DB_NAME are selected.

SAFETY
------
Commits occur after every article batch.

Default execution is a controlled 1,000-article run.

Set:
    EMBEDDING_PRODUCTION=1

to enable full-corpus processing.
"""

from __future__ import annotations

import math
import os
import sys
import time
from datetime import timedelta

import numpy as np
import psycopg2
import torch

from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIGURATION
# =============================================================================

HF_MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

MODEL_DB_NAME = (
    "all-MiniLM-L6-v2_"
    "chunk254_mean_l2_v1"
)

SCHEMA_NAME = "risk_emergence"
VALID_ARTICLES_VIEW = "v_valid_articles"
EMBEDDINGS_TABLE = "news_embeddings"

EXPECTED_CORPUS_ARTICLES = 2_249_255

ARTICLE_BATCH_SIZE = 1_000
ENCODE_BATCH_SIZE = 32

CONTROLLED_RUN_LIMIT = 1_000

PRODUCTION_MODE = (
    os.getenv(
        "EMBEDDING_PRODUCTION",
        "0",
    )
    == "1"
)

RUN_LIMIT = (
    None
    if PRODUCTION_MODE
    else CONTROLLED_RUN_LIMIT
)


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


def build_document_text(
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


def format_duration(seconds):

    if (
        seconds is None
        or not np.isfinite(seconds)
        or seconds < 0
    ):
        return "n/a"

    return str(
        timedelta(
            seconds=int(
                round(seconds)
            )
        )
    )


# =============================================================================
# DATABASE COUNTS
# =============================================================================

def get_corpus_count(conn):

    sql = f"""
        SELECT COUNT(*)
        FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW};
    """

    with conn.cursor() as cur:

        cur.execute(sql)

        return int(
            cur.fetchone()[0]
        )


def get_stored_count(conn):

    sql = f"""
        SELECT COUNT(*)
        FROM {SCHEMA_NAME}.{EMBEDDINGS_TABLE}
        WHERE model_name = %s;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (MODEL_DB_NAME,),
        )

        return int(
            cur.fetchone()[0]
        )


# =============================================================================
# LOAD NEXT MISSING BATCH
# =============================================================================

def load_next_batch(
    conn,
    batch_size,
):

    """
    Load the next valid articles without an embedding
    for the frozen MODEL_DB_NAME.

    Successful batches are committed independently, so
    restarting the script skips previously completed rows.
    """

    sql = f"""
        SELECT
            a.article_id,
            a.title,
            a.body
        FROM {SCHEMA_NAME}.{VALID_ARTICLES_VIEW} AS a
        WHERE NOT EXISTS (
            SELECT 1
            FROM {SCHEMA_NAME}.{EMBEDDINGS_TABLE} AS e
            WHERE e.article_id = a.article_id
              AND e.model_name = %s
        )
        ORDER BY a.article_id
        LIMIT %s;
    """

    start = time.perf_counter()

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                MODEL_DB_NAME,
                batch_size,
            ),
        )

        rows = cur.fetchall()

    elapsed = (
        time.perf_counter()
        - start
    )

    return rows, elapsed


# =============================================================================
# CHUNKING
# =============================================================================

def build_chunks(
    tokenizer,
    rows,
    content_tokens_per_chunk,
):

    chunk_texts = []
    article_chunk_ranges = []
    article_ids = []
    chunks_per_article = []

    start = time.perf_counter()

    for (
        article_id,
        title,
        body,
    ) in rows:

        text = build_document_text(
            title,
            body,
        )

        if not text:

            raise RuntimeError(
                "Empty document text for "
                f"article_id={article_id}."
            )

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
                "Zero tokens for "
                f"article_id={article_id}."
            )

        n_chunks = int(
            math.ceil(
                len(token_ids)
                / content_tokens_per_chunk
            )
        )

        range_start = len(
            chunk_texts
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
                len(token_ids),
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
                    "Empty decoded chunk for "
                    f"article_id={article_id}."
                )

            chunk_texts.append(
                chunk_text
            )

        range_end = len(
            chunk_texts
        )

        article_ids.append(
            int(article_id)
        )

        article_chunk_ranges.append(
            (
                range_start,
                range_end,
            )
        )

        chunks_per_article.append(
            n_chunks
        )

    if len(article_ids) != len(
        set(article_ids)
    ):

        raise RuntimeError(
            "Duplicate article IDs detected "
            "within the current batch."
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "chunk_texts":
            chunk_texts,

        "article_chunk_ranges":
            article_chunk_ranges,

        "article_ids":
            article_ids,

        "chunks_per_article":
            np.asarray(
                chunks_per_article,
                dtype=np.int64,
            ),

        "elapsed":
            elapsed,
    }


def validate_chunk_lengths(
    tokenizer,
    chunk_texts,
    max_seq_length,
):

    """
    Verify that token-id chunks remain within the frozen
    sequence limit after decode and re-tokenization.

    This prevents silent truncation inside model.encode().
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

        if n_tokens > max_seq_length:

            raise RuntimeError(
                "Decoded chunk exceeds the frozen "
                "sequence limit after re-tokenization: "
                f"{n_tokens} > {max_seq_length}."
            )

    return max_observed


# =============================================================================
# ENCODING
# =============================================================================

def encode_chunks(
    model,
    chunk_texts,
):

    if not chunk_texts:

        raise RuntimeError(
            "No chunks available for encoding."
        )

    start = time.perf_counter()

    embeddings = model.encode(
        chunk_texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if embeddings.ndim != 2:

        raise RuntimeError(
            "Unexpected chunk embedding matrix."
        )

    if embeddings.shape[0] != len(
        chunk_texts
    ):

        raise RuntimeError(
            "Chunk embedding count mismatch."
        )

    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            "NaN or Inf detected in "
            "chunk embeddings."
        )

    return (
        embeddings,
        elapsed,
    )


# =============================================================================
# DOCUMENT POOLING
# =============================================================================

def pool_documents(
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
                "Article without chunks."
            )

        vector = np.mean(
            chunk_embeddings[
                left:right
            ],
            axis=0,
            dtype=np.float32,
        )

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if (
            not np.isfinite(norm)
            or norm <= 0
        ):

            raise RuntimeError(
                "Invalid document vector norm."
            )

        document_embeddings[
            article_index
        ] = (
            vector
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
    article_ids,
    embeddings,
    expected_dimension,
):

    if len(article_ids) != len(
        set(article_ids)
    ):

        raise RuntimeError(
            "Duplicate article IDs detected."
        )

    expected_shape = (
        len(article_ids),
        expected_dimension,
    )

    if embeddings.shape != expected_shape:

        raise RuntimeError(
            "Unexpected document matrix: "
            f"{embeddings.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            "NaN or Inf detected."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
    ):

        raise RuntimeError(
            "L2 normalization validation failed."
        )

    return float(
        np.max(
            np.abs(
                norms - 1.0
            )
        )
    )


# =============================================================================
# INSERT BATCH
# =============================================================================

def insert_batch(
    conn,
    article_ids,
    embeddings,
):

    sql = f"""
        INSERT INTO
            {SCHEMA_NAME}.{EMBEDDINGS_TABLE}
            (
                article_id,
                model_name,
                embedding
            )
        VALUES %s
        ON CONFLICT (
            article_id,
            model_name
        )
        DO NOTHING;
    """

    values = [
        (
            int(article_id),
            MODEL_DB_NAME,
            [
                float(x)
                for x in vector
            ],
        )
        for article_id, vector
        in zip(
            article_ids,
            embeddings,
            strict=True,
        )
    ]

    start = time.perf_counter()

    with conn.cursor() as cur:

        execute_values(
            cur,
            sql,
            values,
            page_size=250,
        )

    conn.commit()

    elapsed = (
        time.perf_counter()
        - start
    )

    return elapsed


# =============================================================================
# DATABASE VALIDATION
# =============================================================================

def database_validation(
    conn,
    expected_dimension,
):

    sql = f"""
        SELECT
            COUNT(*) AS n_embeddings,
            COUNT(DISTINCT article_id) AS n_articles,
            MIN(array_length(embedding, 1)) AS min_dim,
            MAX(array_length(embedding, 1)) AS max_dim,
            COUNT(*) FILTER (
                WHERE embedding IS NULL
            ) AS null_embeddings
        FROM {SCHEMA_NAME}.{EMBEDDINGS_TABLE}
        WHERE model_name = %s;
    """

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (MODEL_DB_NAME,),
        )

        (
            n_embeddings,
            n_articles,
            min_dim,
            max_dim,
            null_embeddings,
        ) = cur.fetchone()

    if n_embeddings != n_articles:

        raise RuntimeError(
            "Duplicate article/model rows detected."
        )

    if n_embeddings > 0:

        if (
            min_dim != expected_dimension
            or max_dim != expected_dimension
        ):

            raise RuntimeError(
                "Embedding dimension validation "
                "failed: "
                f"min={min_dim}, "
                f"max={max_dim}, "
                f"expected={expected_dimension}."
            )

    if null_embeddings != 0:

        raise RuntimeError(
            "NULL embeddings detected."
        )

    return {
        "n_embeddings":
            int(n_embeddings),

        "n_articles":
            int(n_articles),

        "min_dim":
            min_dim,

        "max_dim":
            max_dim,

        "null_embeddings":
            int(null_embeddings),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    total_start = time.perf_counter()

    print("=" * 100)
    print(
        "STEP 4 — FULL-CORPUS MINILM "
        "DOCUMENT EMBEDDINGS"
    )
    print("=" * 100)

    print(
        f"\nMode:           "
        f"{'FULL PRODUCTION' if PRODUCTION_MODE else 'CONTROLLED RUN'}"
    )

    print(
        f"HF model:       "
        f"{HF_MODEL_NAME}"
    )

    print(
        f"DB model:       "
        f"{MODEL_DB_NAME}"
    )

    print(
        f"Article batch:  "
        f"{ARTICLE_BATCH_SIZE:,}"
    )

    print(
        f"Encode batch:   "
        f"{ENCODE_BATCH_SIZE:,}"
    )

    print(
        f"Run limit:      "
        f"{RUN_LIMIT if RUN_LIMIT is not None else 'FULL CORPUS'}"
    )

    device = select_device()

    print(
        f"Device:         "
        f"{device}"
    )

    # =========================================================================
    # 1. LOAD FROZEN MODEL
    # =========================================================================

    print()
    print("=" * 100)
    print("1. LOAD FROZEN MODEL")
    print("=" * 100)

    start = time.perf_counter()

    model = SentenceTransformer(
        HF_MODEL_NAME,
        device=device,
    )

    tokenizer = model.tokenizer

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

    model_load_seconds = (
        time.perf_counter()
        - start
    )

    print(
        f"\nLoad time:       "
        f"{model_load_seconds:.3f} s"
    )

    print(
        f"Dimension:       "
        f"{dimension}"
    )

    print(
        f"Max sequence:    "
        f"{max_seq_length}"
    )

    print(
        f"Special tokens:  "
        f"{special_tokens}"
    )

    print(
        f"Content/chunk:   "
        f"{content_tokens_per_chunk}"
    )

    if dimension != 384:

        raise RuntimeError(
            "Frozen specification violation: "
            f"dimension={dimension}."
        )

    if max_seq_length != 256:

        raise RuntimeError(
            "Frozen specification violation: "
            f"max_seq_length={max_seq_length}."
        )

    if special_tokens != 2:

        raise RuntimeError(
            "Frozen specification violation: "
            f"special_tokens={special_tokens}."
        )

    if content_tokens_per_chunk != 254:

        raise RuntimeError(
            "Frozen specification violation: "
            "content_tokens_per_chunk="
            f"{content_tokens_per_chunk}."
        )

    # =========================================================================
    # 2. DATABASE INITIAL STATE
    # =========================================================================

    print()
    print("=" * 100)
    print("2. DATABASE INITIAL STATE")
    print("=" * 100)

    conn = get_connection()

    try:

        corpus_count = get_corpus_count(
            conn
        )

        stored_before = get_stored_count(
            conn
        )

        missing_before = (
            corpus_count
            - stored_before
        )

        print(
            f"\nValid corpus:      "
            f"{corpus_count:,}"
        )

        print(
            f"Already stored:    "
            f"{stored_before:,}"
        )

        print(
            f"Missing:           "
            f"{missing_before:,}"
        )

        if (
            corpus_count
            != EXPECTED_CORPUS_ARTICLES
        ):

            raise RuntimeError(
                "Corpus size mismatch: "
                f"expected "
                f"{EXPECTED_CORPUS_ARTICLES:,}, "
                f"found {corpus_count:,}."
            )

        if stored_before > corpus_count:

            raise RuntimeError(
                "Stored embedding count exceeds "
                "valid corpus size."
            )

        # =====================================================================
        # 3. BATCH PROCESSING
        # =====================================================================

        print()
        print("=" * 100)
        print("3. BATCH PROCESSING")
        print("=" * 100)

        processed_this_run = 0
        total_chunks_this_run = 0
        batch_number = 0

        total_read_seconds = 0.0
        total_chunk_seconds = 0.0
        total_encode_seconds = 0.0
        total_pool_seconds = 0.0
        total_insert_seconds = 0.0

        while True:

            if RUN_LIMIT is not None:

                remaining_run = (
                    RUN_LIMIT
                    - processed_this_run
                )

                if remaining_run <= 0:
                    break

                next_batch_size = min(
                    ARTICLE_BATCH_SIZE,
                    remaining_run,
                )

            else:

                next_batch_size = (
                    ARTICLE_BATCH_SIZE
                )

            (
                rows,
                read_seconds,
            ) = load_next_batch(
                conn=conn,
                batch_size=next_batch_size,
            )

            total_read_seconds += (
                read_seconds
            )

            if not rows:
                break

            batch_number += 1

            batch_start = (
                time.perf_counter()
            )

            chunk_info = build_chunks(
                tokenizer=tokenizer,
                rows=rows,
                content_tokens_per_chunk=
                    content_tokens_per_chunk,
            )

            total_chunk_seconds += (
                chunk_info["elapsed"]
            )

            max_retokenized_length = (
                validate_chunk_lengths(
                    tokenizer=tokenizer,
                    chunk_texts=
                        chunk_info["chunk_texts"],
                    max_seq_length=
                        max_seq_length,
                )
            )

            (
                chunk_embeddings,
                encode_seconds,
            ) = encode_chunks(
                model=model,
                chunk_texts=
                    chunk_info["chunk_texts"],
            )

            total_encode_seconds += (
                encode_seconds
            )

            if (
                chunk_embeddings.shape[1]
                != dimension
            ):

                raise RuntimeError(
                    "Unexpected chunk embedding "
                    f"dimension: "
                    f"{chunk_embeddings.shape[1]}."
                )

            (
                document_embeddings,
                pool_seconds,
            ) = pool_documents(
                chunk_embeddings=
                    chunk_embeddings,
                article_chunk_ranges=
                    chunk_info[
                        "article_chunk_ranges"
                    ],
            )

            total_pool_seconds += (
                pool_seconds
            )

            max_norm_error = (
                validate_embeddings(
                    article_ids=
                        chunk_info["article_ids"],
                    embeddings=
                        document_embeddings,
                    expected_dimension=
                        dimension,
                )
            )

            insert_seconds = insert_batch(
                conn=conn,
                article_ids=
                    chunk_info["article_ids"],
                embeddings=
                    document_embeddings,
            )

            total_insert_seconds += (
                insert_seconds
            )

            batch_articles = len(
                chunk_info["article_ids"]
            )

            batch_chunks = len(
                chunk_info["chunk_texts"]
            )

            processed_this_run += (
                batch_articles
            )

            total_chunks_this_run += (
                batch_chunks
            )

            batch_seconds = (
                time.perf_counter()
                - batch_start
            )

            batch_rate = (
                batch_articles
                / batch_seconds
            )

            processing_elapsed = (
                time.perf_counter()
                - total_start
            )

            run_rate = (
                processed_this_run
                / processing_elapsed
            )

            if RUN_LIMIT is None:

                target_this_run = (
                    missing_before
                )

            else:

                target_this_run = min(
                    RUN_LIMIT,
                    missing_before,
                )

            remaining_target = max(
                0,
                target_this_run
                - processed_this_run,
            )

            eta_seconds = (
                remaining_target
                / run_rate
                if run_rate > 0
                else 0.0
            )

            print(
                f"\n[BATCH {batch_number:,}]"
            )

            print(
                f"  article_id range:    "
                f"{rows[0][0]:,} -> "
                f"{rows[-1][0]:,}"
            )

            print(
                f"  articles:            "
                f"{batch_articles:,}"
            )

            print(
                f"  chunks:              "
                f"{batch_chunks:,}"
            )

            print(
                f"  chunks/article:      "
                f"{batch_chunks / batch_articles:.3f}"
            )

            print(
                f"  max chunk tokens:    "
                f"{max_retokenized_length}"
            )

            print(
                f"  read:                "
                f"{read_seconds:.2f} s"
            )

            print(
                f"  chunk build:         "
                f"{chunk_info['elapsed']:.2f} s"
            )

            print(
                f"  encode:              "
                f"{encode_seconds:.2f} s"
            )

            print(
                f"  pool:                "
                f"{pool_seconds:.2f} s"
            )

            print(
                f"  insert+commit:       "
                f"{insert_seconds:.2f} s"
            )

            print(
                f"  batch total:         "
                f"{batch_seconds:.2f} s"
            )

            print(
                f"  batch rate:          "
                f"{batch_rate:.2f} articles/s"
            )

            print(
                f"  processed this run:  "
                f"{processed_this_run:,}/"
                f"{target_this_run:,}"
            )

            print(
                f"  max norm error:      "
                f"{max_norm_error:.10f}"
            )

            print(
                f"  ETA current target:  "
                f"{format_duration(eta_seconds)}"
            )

            del chunk_embeddings
            del document_embeddings
            del chunk_info
            del rows

            clear_device_cache(
                device
            )

        # =====================================================================
        # 4. DATABASE END STATE
        # =====================================================================

        print()
        print("=" * 100)
        print("4. DATABASE END STATE")
        print("=" * 100)

        stored_after = get_stored_count(
            conn
        )

        added = (
            stored_after
            - stored_before
        )

        missing_after = (
            corpus_count
            - stored_after
        )

        print(
            f"\nStored before:       "
            f"{stored_before:,}"
        )

        print(
            f"Stored after:        "
            f"{stored_after:,}"
        )

        print(
            f"Added this run:      "
            f"{added:,}"
        )

        print(
            f"Missing after:       "
            f"{missing_after:,}"
        )

        if added != processed_this_run:

            raise RuntimeError(
                "Stored-row increase does not "
                "match processed article count: "
                f"added={added:,}, "
                f"processed={processed_this_run:,}."
            )

        db_validation = database_validation(
            conn=conn,
            expected_dimension=dimension,
        )

        print(
            f"\nEmbeddings:          "
            f"{db_validation['n_embeddings']:,}"
        )

        print(
            f"Distinct articles:   "
            f"{db_validation['n_articles']:,}"
        )

        print(
            f"Min dimension:       "
            f"{db_validation['min_dim']}"
        )

        print(
            f"Max dimension:       "
            f"{db_validation['max_dim']}"
        )

        print(
            f"NULL embeddings:     "
            f"{db_validation['null_embeddings']}"
        )

        # =====================================================================
        # 5. SUMMARY
        # =====================================================================

        total_seconds = (
            time.perf_counter()
            - total_start
        )

        print()
        print("=" * 100)
        print("5. RUN SUMMARY")
        print("=" * 100)

        print(
            f"\nArticles this run:   "
            f"{processed_this_run:,}"
        )

        print(
            f"Chunks this run:     "
            f"{total_chunks_this_run:,}"
        )

        if processed_this_run > 0:

            print(
                f"Mean chunks/article: "
                f"{total_chunks_this_run / processed_this_run:.4f}"
            )

        print(
            f"\nDB read total:       "
            f"{format_duration(total_read_seconds)}"
        )

        print(
            f"Chunk build total:   "
            f"{format_duration(total_chunk_seconds)}"
        )

        print(
            f"Encoding total:      "
            f"{format_duration(total_encode_seconds)}"
        )

        print(
            f"Pooling total:       "
            f"{format_duration(total_pool_seconds)}"
        )

        print(
            f"Insert total:        "
            f"{format_duration(total_insert_seconds)}"
        )

        print(
            f"Total runtime:       "
            f"{format_duration(total_seconds)}"
        )

        if total_seconds > 0:

            print(
                f"Overall rate:        "
                f"{processed_this_run / total_seconds:.2f} "
                f"articles/s"
            )

        print(
            f"\nCorpus complete:     "
            f"{'YES' if missing_after == 0 else 'NO'}"
        )

        print(
            "\nDATABASE VALIDATION: PASS"
        )

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

    print()
    print("=" * 100)
    print("RUN COMPLETE")
    print("=" * 100)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nInterrupted by user."
        )

        print(
            "Completed batches were already "
            "committed and will be skipped "
            "on the next run."
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
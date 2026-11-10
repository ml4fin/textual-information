#!/usr/bin/env python3

"""
STEP 4 — PRODUCTION DOCUMENT EMBEDDINGS
========================================

FINAL PIPELINE

Model:
    sentence-transformers/all-MiniLM-L6-v2

Document:
    title + "\\n\\n" + body

Long documents:
    token-based full-document chunking

Chunk:
    254 content tokens
    + model special tokens
    = max sequence length 256

Chunk overlap:
    0

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

IMPORTANT
---------
Default mode processes only 100 articles for production validation.

Set EMBEDDING_PRODUCTION=1 to enable full-corpus processing.

The pipeline is resumable:
articles already stored for MODEL_DB_NAME are skipped.
"""

from __future__ import annotations

import math
import os
import sys
import time

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

TEST_LIMIT = 100
ENCODE_BATCH_SIZE = 32

PRODUCTION_MODE = (
    os.getenv(
        "EMBEDDING_PRODUCTION",
        "0",
    )
    == "1"
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
# GENERAL HELPERS
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


# =============================================================================
# LOAD UNSCORED ARTICLES
# =============================================================================

def load_articles(
    conn,
    limit=None,
):

    """
    Load valid articles that do not yet have an embedding
    for the frozen MODEL_DB_NAME.
    """

    limit_clause = ""

    params = [
        MODEL_DB_NAME,
    ]

    if limit is not None:

        limit_clause = "LIMIT %s"

        params.append(
            int(limit)
        )

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
        {limit_clause};
    """

    start = time.perf_counter()

    with conn.cursor() as cur:

        cur.execute(
            sql,
            tuple(params),
        )

        rows = cur.fetchall()

    elapsed = (
        time.perf_counter()
        - start
    )

    return rows, elapsed


# =============================================================================
# TOKENIZATION AND CHUNKING
# =============================================================================

def build_chunks_for_documents(
    tokenizer,
    rows,
    content_tokens_per_chunk,
):

    """
    Convert complete documents into model-compatible token chunks.
    """

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

        document_text = build_document_text(
            title,
            body,
        )

        if not document_text:

            raise RuntimeError(
                "Empty document text for "
                f"article_id={article_id}."
            )

        encoded = tokenizer(
            document_text,
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
    Verify that decoded chunks remain within the frozen model
    sequence limit after re-tokenization.
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
                "model sequence limit after "
                "re-tokenization: "
                f"{n_tokens} > {max_seq_length}."
            )

    return max_observed


# =============================================================================
# ENCODE CHUNKS
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
# POOL TO ONE VECTOR PER ARTICLE
# =============================================================================

def pool_documents(
    chunk_embeddings,
    article_chunk_ranges,
):

    if chunk_embeddings.ndim != 2:

        raise RuntimeError(
            "Chunk embedding matrix must "
            "be two-dimensional."
        )

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
                "Invalid pooled vector norm."
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
# VALIDATION BEFORE INSERT
# =============================================================================

def validate_document_embeddings(
    article_ids,
    embeddings,
    expected_dimension,
):

    if len(article_ids) != len(
        set(article_ids)
    ):

        raise RuntimeError(
            "Duplicate article IDs in "
            "current processing batch."
        )

    expected_shape = (
        len(article_ids),
        expected_dimension,
    )

    if embeddings.shape != expected_shape:

        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{embeddings.shape}; expected "
            f"{expected_shape}."
        )

    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            "NaN or Inf detected before "
            "database insert."
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
            "Final L2 normalization "
            "validation failed."
        )

    return float(
        np.max(
            np.abs(
                norms - 1.0
            )
        )
    )


# =============================================================================
# INSERT INTO POSTGRESQL
# =============================================================================

def insert_embeddings(
    conn,
    article_ids,
    embeddings,
):

    """
    Insert one final document embedding per article.

    The article/model primary key makes repeated execution idempotent.
    """

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
            page_size=100,
        )

    conn.commit()

    return (
        time.perf_counter()
        - start
    )


# =============================================================================
# POST-INSERT VALIDATION
# =============================================================================

def validate_database(
    conn,
    expected_dimension,
):

    sql = f"""
        SELECT
            COUNT(*) AS n_embeddings,
            COUNT(DISTINCT article_id) AS n_articles,
            MIN(array_length(embedding, 1)) AS min_dimension,
            MAX(array_length(embedding, 1)) AS max_dimension,
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
            min_dimension,
            max_dimension,
            null_embeddings,
        ) = cur.fetchone()

    if n_embeddings != n_articles:

        raise RuntimeError(
            "Duplicate article/model rows detected."
        )

    if n_embeddings > 0:

        if (
            min_dimension != expected_dimension
            or max_dimension != expected_dimension
        ):

            raise RuntimeError(
                "Unexpected database embedding "
                "dimension: "
                f"min={min_dimension}, "
                f"max={max_dimension}."
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

        "min_dimension":
            min_dimension,

        "max_dimension":
            max_dimension,

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
        "STEP 4 — PRODUCTION MINILM "
        "DOCUMENT EMBEDDINGS"
    )
    print("=" * 100)

    if PRODUCTION_MODE:

        article_limit = None

        print(
            "\nMODE: FULL PRODUCTION"
        )

    else:

        article_limit = TEST_LIMIT

        print(
            "\nMODE: VALIDATION TEST"
        )

        print(
            f"Article limit: "
            f"{TEST_LIMIT:,}"
        )

    print(
        f"HF model:     "
        f"{HF_MODEL_NAME}"
    )

    print(
        f"DB model:     "
        f"{MODEL_DB_NAME}"
    )

    device = select_device()

    print(
        f"Device:       "
        f"{device}"
    )

    # =========================================================================
    # 1. LOAD MODEL
    # =========================================================================

    print()
    print("=" * 100)
    print("1. LOAD MODEL")
    print("=" * 100)

    start = time.perf_counter()

    model = SentenceTransformer(
        HF_MODEL_NAME,
        device=device,
    )

    model_load_seconds = (
        time.perf_counter()
        - start
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

    print(
        f"\nLoad time:             "
        f"{model_load_seconds:.3f} s"
    )

    print(
        f"Embedding dimension:   "
        f"{dimension}"
    )

    print(
        f"Max sequence length:   "
        f"{max_seq_length}"
    )

    print(
        f"Special tokens:        "
        f"{special_tokens}"
    )

    print(
        f"Content/chunk:         "
        f"{content_tokens_per_chunk}"
    )

    if dimension != 384:

        raise RuntimeError(
            "Frozen specification mismatch: "
            f"expected dimension 384, "
            f"found {dimension}."
        )

    if max_seq_length != 256:

        raise RuntimeError(
            "Frozen specification mismatch: "
            "expected max_seq_length 256, "
            f"found {max_seq_length}."
        )

    if special_tokens != 2:

        raise RuntimeError(
            "Frozen specification mismatch: "
            "expected 2 special tokens, "
            f"found {special_tokens}."
        )

    if content_tokens_per_chunk != 254:

        raise RuntimeError(
            "Frozen specification mismatch: "
            "expected 254 content tokens/chunk, "
            f"found {content_tokens_per_chunk}."
        )

    # =========================================================================
    # 2. LOAD UNSCORED ARTICLES
    # =========================================================================

    print()
    print("=" * 100)
    print("2. LOAD UNSCORED ARTICLES")
    print("=" * 100)

    conn = get_connection()

    try:

        (
            rows,
            db_read_seconds,
        ) = load_articles(
            conn=conn,
            limit=article_limit,
        )

        print(
            f"\nArticles selected: "
            f"{len(rows):,}"
        )

        print(
            f"DB read time:      "
            f"{db_read_seconds:.3f} s"
        )

        if not rows:

            print(
                "\nNo missing articles "
                "for this model."
            )

            print(
                "Resume check: PASS"
            )

            return

        print(
            f"First article_id:   "
            f"{rows[0][0]:,}"
        )

        print(
            f"Last article_id:    "
            f"{rows[-1][0]:,}"
        )

        # =====================================================================
        # 3. BUILD CHUNKS
        # =====================================================================

        print()
        print("=" * 100)
        print("3. BUILD DOCUMENT CHUNKS")
        print("=" * 100)

        chunk_info = (
            build_chunks_for_documents(
                tokenizer=tokenizer,
                rows=rows,
                content_tokens_per_chunk=
                    content_tokens_per_chunk,
            )
        )

        chunk_texts = chunk_info[
            "chunk_texts"
        ]

        article_ids = chunk_info[
            "article_ids"
        ]

        article_chunk_ranges = chunk_info[
            "article_chunk_ranges"
        ]

        chunks_per_article = chunk_info[
            "chunks_per_article"
        ]

        max_retokenized_length = (
            validate_chunk_lengths(
                tokenizer=tokenizer,
                chunk_texts=chunk_texts,
                max_seq_length=max_seq_length,
            )
        )

        print(
            f"\nDocuments:             "
            f"{len(article_ids):,}"
        )

        print(
            f"Chunks:                "
            f"{len(chunk_texts):,}"
        )

        print(
            f"Mean chunks/article:   "
            f"{chunks_per_article.mean():.3f}"
        )

        print(
            f"Median chunks:         "
            f"{np.median(chunks_per_article):.2f}"
        )

        print(
            f"Max chunks:            "
            f"{chunks_per_article.max():,}"
        )

        print(
            f"Max re-tokenized chunk:"
            f" {max_retokenized_length}"
        )

        print(
            f"Chunk build time:      "
            f"{chunk_info['elapsed']:.3f} s"
        )

        # =====================================================================
        # 4. ENCODE CHUNKS
        # =====================================================================

        print()
        print("=" * 100)
        print("4. ENCODE CHUNKS")
        print("=" * 100)

        clear_device_cache(
            device
        )

        (
            chunk_embeddings,
            encode_seconds,
        ) = encode_chunks(
            model=model,
            chunk_texts=chunk_texts,
        )

        if (
            chunk_embeddings.shape[0]
            != len(chunk_texts)
        ):

            raise RuntimeError(
                "Chunk embedding count mismatch."
            )

        if (
            chunk_embeddings.shape[1]
            != dimension
        ):

            raise RuntimeError(
                "Chunk embedding dimension mismatch."
            )

        if not np.isfinite(
            chunk_embeddings
        ).all():

            raise RuntimeError(
                "NaN or Inf detected in "
                "chunk embeddings."
            )

        print(
            f"\nChunk matrix:        "
            f"{chunk_embeddings.shape}"
        )

        print(
            f"Encoding time:       "
            f"{encode_seconds:.3f} s"
        )

        # =====================================================================
        # 5. POOL DOCUMENTS
        # =====================================================================

        print()
        print("=" * 100)
        print("5. POOL TO DOCUMENT EMBEDDINGS")
        print("=" * 100)

        (
            document_embeddings,
            pooling_seconds,
        ) = pool_documents(
            chunk_embeddings,
            article_chunk_ranges,
        )

        max_norm_error = (
            validate_document_embeddings(
                article_ids,
                document_embeddings,
                dimension,
            )
        )

        print(
            f"\nDocument matrix:     "
            f"{document_embeddings.shape}"
        )

        print(
            f"Pooling time:        "
            f"{pooling_seconds:.3f} s"
        )

        print(
            f"Max L2 norm error:   "
            f"{max_norm_error:.10f}"
        )

        print(
            "Pre-insert validation: PASS"
        )

        # =====================================================================
        # 6. INSERT
        # =====================================================================

        print()
        print("=" * 100)
        print("6. INSERT INTO POSTGRESQL")
        print("=" * 100)

        insert_seconds = insert_embeddings(
            conn,
            article_ids,
            document_embeddings,
        )

        print(
            f"\nRows submitted:      "
            f"{len(article_ids):,}"
        )

        print(
            f"Insert + commit:     "
            f"{insert_seconds:.3f} s"
        )

        # =====================================================================
        # 7. DATABASE VALIDATION
        # =====================================================================

        print()
        print("=" * 100)
        print("7. POST-INSERT VALIDATION")
        print("=" * 100)

        db_validation = validate_database(
            conn,
            dimension,
        )

        print(
            f"\nStored embeddings:   "
            f"{db_validation['n_embeddings']:,}"
        )

        print(
            f"Distinct articles:   "
            f"{db_validation['n_articles']:,}"
        )

        print(
            f"Min dimension:       "
            f"{db_validation['min_dimension']}"
        )

        print(
            f"Max dimension:       "
            f"{db_validation['max_dimension']}"
        )

        print(
            f"NULL embeddings:     "
            f"{db_validation['null_embeddings']}"
        )

        print(
            "\nPOST-INSERT VALIDATION: PASS"
        )

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    # =========================================================================
    # 8. SUMMARY
    # =========================================================================

    print()
    print("=" * 100)
    print("8. RUN SUMMARY")
    print("=" * 100)

    print(
        f"\nArticles processed:   "
        f"{len(article_ids):,}"
    )

    print(
        f"Chunks processed:     "
        f"{len(chunk_texts):,}"
    )

    print(
        f"Model name stored:    "
        f"{MODEL_DB_NAME}"
    )

    print(
        f"Vector dimension:     "
        f"{dimension}"
    )

    print(
        f"Total runtime:        "
        f"{total_seconds:.3f} s"
    )

    print()
    print("=" * 100)

    if PRODUCTION_MODE:

        print(
            "PRODUCTION EMBEDDING RUN COMPLETE"
        )

    else:

        print(
            "100-ARTICLE PRODUCTION "
            "VALIDATION COMPLETE"
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
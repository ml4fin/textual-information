#!/usr/bin/env python3

"""
Weak V3 Continuous — Article-Level Engine Parity Test.

Validate an optimized implementation of the already-frozen Weak V3
Continuous Candidate C scorer.

The optimized engine preprocesses article text once and reuses the resulting
representation across article-theme observations. The mathematical scoring
specification is not changed.

Validation target:
    Reproduce the 500 already-frozen final-test Weak V3 Continuous scores
    within numerical tolerance.

Important:
- No training.
- No tuning.
- No PostgreSQL writes.
- Frozen predictions are not overwritten.
- Final-test gold labels are not used for scoring or parity evaluation.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import build_gold_review as gold_builder

from weak_supervision_v3_config import (
    VERSION as BINARY_VERSION,
    PROXIMITY_WINDOW as BINARY_PROXIMITY_WINDOW,
    STRONG_SIGNALS,
    CLIMATE_THEME_SIGNALS,
    CYBER_OBJECT_SIGNALS,
    CYBER_CONTEXT_SIGNALS,
    SUPPLY_OBJECT_SIGNALS,
    SUPPLY_DISRUPTION_SIGNALS,
    AI_STRICT_SIGNALS,
)

from weak_supervision_v3_continuous_config import (
    VERSION as CONTINUOUS_VERSION,
    SELECTED_CANDIDATE,
    PROXIMITY_WINDOW,
    DENSITY_REINFORCEMENT,
    SCORE_MIN,
    SCORE_MAX,
)


# =============================================================================
# FILES / LOCKS
# =============================================================================

GOLD_FILE = Path(
    os.getenv(
        "FINAL_TEST_GOLD_FILE",
        "final_test_gold_frozen.csv",
    )
)

FROZEN_PREDICTIONS_FILE = Path(
    os.getenv(
        "FINAL_WEAK_V3_CONTINUOUS_PREDICTIONS",
        "final_weak_v3_continuous_predictions.csv",
    )
)

EXPECTED_ROWS = 500
EXPECTED_ARTICLES = 300

EXPECTED_BINARY_VERSION = (
    "Weak Supervision V3.0 - Frozen Development Version"
)

EXPECTED_CONTINUOUS_VERSION = (
    "Weak Supervision V3 Continuous - "
    "Frozen Development Version"
)

EXPECTED_CANDIDATE = "score_C_density"
EXPECTED_WINDOW = 50
EXPECTED_DENSITY = 0.35

ATOL = 1e-12
RTOL = 1e-12


# =============================================================================
# THEMES
# =============================================================================

THEMES = [
    (1, "Cybersecurity Risk"),
    (2, "Climate / Environmental Risk"),
    (3, "Pandemic Risk"),
    (4, "Supply Chain Risk"),
    (5, "AI-Related Risk"),
]

THEME_ID_TO_NAME = dict(
    THEMES
)


# =============================================================================
# TEXT HELPERS
# =============================================================================

TOKEN_RE = re.compile(
    r"[a-z0-9\-]+"
)


def safe_text(value) -> str:
    """Convert missing values to empty strings."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def normalize_text(value) -> str:
    """Normalize text using the frozen lowercase convention."""

    return safe_text(
        value
    ).lower()


def tokenize(value) -> list[str]:
    """Tokenize text using the frozen Weak V3 tokenization rule."""

    return TOKEN_RE.findall(
        normalize_text(value)
    )


def phrase_pattern(
    phrase,
) -> str:
    """Build the frozen literal phrase-matching pattern."""

    phrase = (
        normalize_text(
            phrase
        )
        .strip()
    )

    return (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )


def count_phrase(
    normalized_text: str,
    phrase,
) -> int:
    """Count non-overlapping occurrences of one signal."""

    if not phrase:
        return 0

    return len(
        re.findall(
            phrase_pattern(
                phrase
            ),
            normalized_text,
        )
    )


def total_signal_frequency(
    normalized_text: str,
    signals,
) -> int:
    """Count all occurrences across a collection of signals."""

    return sum(
        count_phrase(
            normalized_text,
            signal,
        )
        for signal in signals
    )


def phrase_positions(
    tokens: list[str],
    phrase,
) -> list[int]:
    """Return start-token positions for a phrase."""

    target = tokenize(
        phrase
    )

    if not target:
        return []

    phrase_length = len(
        target
    )

    return [
        i
        for i in range(
            len(tokens)
            - phrase_length
            + 1
        )
        if (
            tokens[
                i:i + phrase_length
            ]
            == target
        )
    ]


def minimum_distance_sets(
    tokens: list[str],
    signals1,
    signals2,
):
    """
    Return the minimum frozen start-token distance between two signal sets.
    """

    best = None

    for signal1 in signals1:

        positions1 = phrase_positions(
            tokens,
            signal1,
        )

        if not positions1:
            continue

        for signal2 in signals2:

            positions2 = phrase_positions(
                tokens,
                signal2,
            )

            if not positions2:
                continue

            for position1 in positions1:
                for position2 in positions2:

                    distance = abs(
                        position1
                        - position2
                    )

                    if (
                        best is None
                        or distance < best
                    ):
                        best = distance

                        if best == 0:
                            return 0

    return best


def proximity_strength(
    distance,
) -> float:
    """Apply the frozen continuous proximity transformation."""

    if distance is None:
        return 0.0

    return float(
        max(
            0.0,
            1.0
            - (
                distance
                / PROXIMITY_WINDOW
            ),
        )
    )


def parse_pipe(
    value,
) -> list[str]:
    """Parse pipe-delimited evidence returned by the frozen baseline."""

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    value = (
        str(value)
        .strip()
        .lower()
    )

    if value in {
        "",
        "nan",
        "none",
    }:
        return []

    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


# =============================================================================
# FROZEN SCORE HELPERS
# =============================================================================

def sat_count(
    value,
) -> float:
    """Apply the frozen saturating count transformation."""

    value = max(
        float(value),
        0.0,
    )

    return float(
        1.0
        - np.exp(-value)
    )


def density_component(
    value,
) -> float:
    """Map non-negative density to [0, 1) using x / (1 + x)."""

    value = max(
        float(value),
        0.0,
    )

    return float(
        value
        / (
            1.0
            + value
        )
    )


def probabilistic_union(
    *components,
) -> float:
    """
    Apply the frozen positive-evidence aggregation rule.

    This is an evidence aggregation operator, not a calibrated probability.
    """

    product = 1.0

    for component in components:

        component = float(
            np.clip(
                component,
                SCORE_MIN,
                SCORE_MAX,
            )
        )

        product *= (
            1.0
            - component
        )

    return float(
        np.clip(
            1.0
            - product,
            SCORE_MIN,
            SCORE_MAX,
        )
    )


# =============================================================================
# ARTICLE PREPROCESSING
# =============================================================================

def prepare_article(
    title,
    body,
) -> dict:
    """Preprocess article text once for reuse across theme observations."""

    title = safe_text(
        title
    )

    body = safe_text(
        body
    )

    full_text = (
        f"{title} {body}"
    )

    normalized_full_text = (
        normalize_text(
            full_text
        )
    )

    tokens = TOKEN_RE.findall(
        normalized_full_text
    )

    token_count = len(
        tokens
    )

    return {
        "title":
            title,

        "body":
            body,

        "normalized_full_text":
            normalized_full_text,

        "tokens":
            tokens,

        "token_count":
            token_count,

        "denominator":
            max(
                token_count,
                1,
            ),
    }


# =============================================================================
# FROZEN BASELINE EVIDENCE
# =============================================================================

def get_theme_and_risk_hits(
    article_id,
    theme_id,
    theme_name,
    title,
    body,
) -> tuple[list[str], list[str]]:
    """
    Reconstruct frozen baseline evidence where Candidate C proximity requires it.
    """

    baseline_row = pd.Series(
        {
            "article_id":
                article_id,

            "theme_id":
                theme_id,

            "theme_name":
                theme_name,

            "title":
                title,

            "body":
                body,
        }
    )

    result = gold_builder.score_row(
        baseline_row
    )

    return (
        parse_pipe(
            result[
                "theme_evidence"
            ]
        ),
        parse_pipe(
            result[
                "risk_evidence"
            ]
        ),
    )


# =============================================================================
# ARTICLE-THEME SCORER
# =============================================================================

def score_prepared_article_theme(
    article_id,
    theme_id,
    theme_name,
    prepared,
) -> float:
    """
    Score one article-theme observation using frozen Candidate C.

    The prepared article representation is reused across themes.
    """

    normalized_full_text = (
        prepared[
            "normalized_full_text"
        ]
    )

    tokens = (
        prepared[
            "tokens"
        ]
    )

    denominator = (
        prepared[
            "denominator"
        ]
    )

    title = (
        prepared[
            "title"
        ]
    )

    body = (
        prepared[
            "body"
        ]
    )

    strong_signals = (
        STRONG_SIGNALS.get(
            theme_name,
            set(),
        )
    )

    strong_frequency = (
        total_signal_frequency(
            normalized_full_text,
            strong_signals,
        )
    )

    strong_density_1000 = (
        1000.0
        * strong_frequency
        / denominator
    )

    # Climate and AI proximity depend on evidence reconstructed by the
    # already-frozen Weak V3 baseline.
    theme_hits = []
    risk_hits = []

    if theme_name in {
        "Climate / Environmental Risk",
        "AI-Related Risk",
    }:
        (
            theme_hits,
            risk_hits,
        ) = get_theme_and_risk_hits(
            article_id=article_id,
            theme_id=theme_id,
            theme_name=theme_name,
            title=title,
            body=body,
        )

    distance = None

    if (
        theme_name
        == "Climate / Environmental Risk"
    ):
        if risk_hits:
            distance = (
                minimum_distance_sets(
                    tokens,
                    CLIMATE_THEME_SIGNALS,
                    risk_hits,
                )
            )

    elif (
        theme_name
        == "Cybersecurity Risk"
    ):
        distance = (
            minimum_distance_sets(
                tokens,
                CYBER_OBJECT_SIGNALS,
                CYBER_CONTEXT_SIGNALS,
            )
        )

    elif (
        theme_name
        == "Supply Chain Risk"
    ):
        distance = (
            minimum_distance_sets(
                tokens,
                SUPPLY_OBJECT_SIGNALS,
                SUPPLY_DISRUPTION_SIGNALS,
            )
        )

    elif (
        theme_name
        == "AI-Related Risk"
    ):
        if theme_hits:
            distance = (
                minimum_distance_sets(
                    tokens,
                    theme_hits,
                    AI_STRICT_SIGNALS,
                )
            )

    elif (
        theme_name
        == "Pandemic Risk"
    ):
        distance = None

    else:
        raise RuntimeError(
            f"Unexpected theme: "
            f"{theme_name!r}"
        )

    proximity = (
        proximity_strength(
            distance
        )
    )

    frequency_component = (
        sat_count(
            strong_frequency
        )
    )

    strong_density_component = (
        density_component(
            strong_density_1000
        )
    )

    density_reinforcement = (
        DENSITY_REINFORCEMENT
        * strong_density_component
    )

    # Frozen Candidate C specification.
    if (
        theme_name
        == "AI-Related Risk"
    ):
        score = proximity

    elif (
        theme_name
        == "Pandemic Risk"
    ):
        score = probabilistic_union(
            frequency_component,
            density_reinforcement,
        )

    else:
        score = probabilistic_union(
            frequency_component,
            proximity,
            density_reinforcement,
        )

    return float(
        np.clip(
            score,
            SCORE_MIN,
            SCORE_MAX,
        )
    )


def score_article_all_themes(
    article_id,
    title,
    body,
    themes=THEMES,
) -> list[tuple[int, int, str, float]]:
    """Preprocess one article and compute all requested frozen theme scores."""

    prepared = prepare_article(
        title=title,
        body=body,
    )

    results = []

    for (
        theme_id,
        theme_name,
    ) in themes:

        score = (
            score_prepared_article_theme(
                article_id=article_id,
                theme_id=theme_id,
                theme_name=theme_name,
                prepared=prepared,
            )
        )

        results.append(
            (
                int(article_id),
                int(theme_id),
                theme_name,
                score,
            )
        )

    return results


# =============================================================================
# VALIDATION
# =============================================================================

def validate_frozen_configuration() -> None:
    """Verify that the imported configuration matches the frozen specification."""

    if (
        BINARY_VERSION
        != EXPECTED_BINARY_VERSION
    ):
        raise RuntimeError(
            "Binary version mismatch."
        )

    if (
        CONTINUOUS_VERSION
        != EXPECTED_CONTINUOUS_VERSION
    ):
        raise RuntimeError(
            "Continuous version mismatch."
        )

    if (
        BINARY_PROXIMITY_WINDOW
        != EXPECTED_WINDOW
    ):
        raise RuntimeError(
            "Binary proximity-window mismatch."
        )

    if (
        PROXIMITY_WINDOW
        != EXPECTED_WINDOW
    ):
        raise RuntimeError(
            "Continuous proximity-window mismatch."
        )

    if (
        BINARY_PROXIMITY_WINDOW
        != PROXIMITY_WINDOW
    ):
        raise RuntimeError(
            "Binary and continuous proximity windows differ."
        )

    if (
        SELECTED_CANDIDATE
        != EXPECTED_CANDIDATE
    ):
        raise RuntimeError(
            "Selected candidate mismatch."
        )

    if not np.isclose(
        DENSITY_REINFORCEMENT,
        EXPECTED_DENSITY,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Density reinforcement mismatch."
        )

    if (
        SCORE_MIN != 0.0
        or SCORE_MAX != 1.0
    ):
        raise RuntimeError(
            "Unexpected frozen score bounds."
        )


def validate_article_consistency(
    df: pd.DataFrame,
) -> None:
    """
    Verify that each article_id maps to one unique title/body pair.

    This prevents drop_duplicates(article_id) from silently choosing one
    representation when duplicated article rows contain inconsistent text.
    """

    consistency = (
        df.groupby(
            "article_id",
            dropna=False,
        )
        .agg(
            title_variants=(
                "title",
                lambda values:
                    values.fillna("")
                    .astype(str)
                    .nunique()
            ),
            body_variants=(
                "body",
                lambda values:
                    values.fillna("")
                    .astype(str)
                    .nunique()
            ),
        )
    )

    inconsistent = consistency[
        (consistency["title_variants"] > 1)
        | (consistency["body_variants"] > 1)
    ]

    if not inconsistent.empty:
        raise RuntimeError(
            "Some article_id values map to inconsistent "
            "title/body text across final-test observations."
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 100)
    print(
        "WEAK V3 CONTINUOUS — "
        "ARTICLE-LEVEL ENGINE PARITY TEST"
    )
    print("=" * 100)

    print()
    print("=" * 100)
    print("1. VERIFY FROZEN CONFIG")
    print("=" * 100)

    validate_frozen_configuration()

    print()
    print(
        f"Binary config: "
        f"{BINARY_VERSION}"
    )

    print(
        f"Continuous config: "
        f"{CONTINUOUS_VERSION}"
    )

    print(
        f"Candidate: "
        f"{SELECTED_CANDIDATE}"
    )

    print(
        f"Window: "
        f"{PROXIMITY_WINDOW}"
    )

    print(
        f"Density reinforcement: "
        f"{DENSITY_REINFORCEMENT}"
    )

    print()
    print(
        "[OK] Frozen configuration verified"
    )

    print()
    print("=" * 100)
    print("2. LOAD FROZEN REFERENCE")
    print("=" * 100)

    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Gold file not found:\n"
            f"{GOLD_FILE}"
        )

    if not FROZEN_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found:\n"
            f"{FROZEN_PREDICTIONS_FILE}"
        )

    gold = pd.read_csv(
        GOLD_FILE
    )

    reference = pd.read_csv(
        FROZEN_PREDICTIONS_FILE
    )

    required_gold_columns = {
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "title",
        "body",
    }

    missing_gold_columns = (
        required_gold_columns
        - set(gold.columns)
    )

    if missing_gold_columns:
        raise RuntimeError(
            f"Missing gold columns: "
            f"{sorted(missing_gold_columns)}"
        )

    required_reference_columns = {
        "final_test_id",
        "article_id",
        "theme_id",
        "weak_continuous_score",
    }

    missing_reference_columns = (
        required_reference_columns
        - set(reference.columns)
    )

    if missing_reference_columns:
        raise RuntimeError(
            f"Missing reference columns: "
            f"{sorted(missing_reference_columns)}"
        )

    if (
        len(gold)
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} gold rows, "
            f"found {len(gold)}."
        )

    if (
        len(reference)
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} reference rows, "
            f"found {len(reference)}."
        )

    if (
        gold["final_test_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate final_test_id values in gold file."
        )

    if (
        reference["final_test_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate final_test_id values in reference predictions."
        )

    if (
        gold["article_id"]
        .nunique()
        != EXPECTED_ARTICLES
    ):
        raise RuntimeError(
            "Unexpected number of unique final-test articles."
        )

    validate_article_consistency(
        gold
    )

    reference_small = (
        reference[
            [
                "final_test_id",
                "article_id",
                "theme_id",
                "weak_continuous_score",
            ]
        ]
        .rename(
            columns={
                "article_id":
                    "reference_article_id",

                "theme_id":
                    "reference_theme_id",

                "weak_continuous_score":
                    "reference_score",
            }
        )
    )

    df = gold.merge(
        reference_small,
        on="final_test_id",
        how="left",
        validate="one_to_one",
    )

    if (
        df["reference_score"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing reference scores after merge."
        )

    if not np.isfinite(
        df["reference_score"]
        .to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Reference scores contain non-finite values."
        )

    if not np.array_equal(
        df["article_id"]
        .astype(int)
        .to_numpy(),
        df["reference_article_id"]
        .astype(int)
        .to_numpy(),
    ):
        raise RuntimeError(
            "Article alignment mismatch."
        )

    if not np.array_equal(
        df["theme_id"]
        .astype(int)
        .to_numpy(),
        df["reference_theme_id"]
        .astype(int)
        .to_numpy(),
    ):
        raise RuntimeError(
            "Theme alignment mismatch."
        )

    expected_theme_names = (
        df["theme_id"]
        .astype(int)
        .map(
            THEME_ID_TO_NAME
        )
    )

    if (
        expected_theme_names
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Unexpected theme_id in final-test data."
        )

    if not (
        expected_theme_names
        == df["theme_name"]
    ).all():
        raise RuntimeError(
            "theme_id/theme_name mapping mismatch."
        )

    print()
    print(
        f"Frozen rows: "
        f"{len(df):,}"
    )

    print(
        f"Unique articles: "
        f"{df['article_id'].nunique():,}"
    )

    print(
        "[OK] Frozen reference loaded"
    )

    print()
    print("=" * 100)
    print("3. PREPROCESS EACH ARTICLE ONCE")
    print("=" * 100)

    article_cache = {}

    article_rows = (
        df[
            [
                "article_id",
                "title",
                "body",
            ]
        ]
        .drop_duplicates(
            subset=[
                "article_id"
            ]
        )
    )

    for row in (
        article_rows.itertuples(
            index=False
        )
    ):
        article_cache[
            int(
                row.article_id
            )
        ] = prepare_article(
            title=row.title,
            body=row.body,
        )

    if (
        len(article_cache)
        != EXPECTED_ARTICLES
    ):
        raise RuntimeError(
            "Article cache size mismatch."
        )

    print()
    print(
        f"Articles preprocessed: "
        f"{len(article_cache):,}"
    )

    print()
    print("=" * 100)
    print("4. SCORE FROZEN OBSERVATIONS")
    print("=" * 100)

    optimized_scores = []

    for i, row in enumerate(
        df.itertuples(
            index=False
        ),
        start=1,
    ):

        prepared = (
            article_cache[
                int(
                    row.article_id
                )
            ]
        )

        score = (
            score_prepared_article_theme(
                article_id=row.article_id,
                theme_id=row.theme_id,
                theme_name=row.theme_name,
                prepared=prepared,
            )
        )

        optimized_scores.append(
            score
        )

        if (
            i % 100
            == 0
        ):
            print(
                f"Processed "
                f"{i}/{len(df)}"
            )

    optimized_scores = np.asarray(
        optimized_scores,
        dtype=float,
    )

    reference_scores = (
        df["reference_score"]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        optimized_scores
    ).all():
        raise RuntimeError(
            "Optimized scores contain non-finite values."
        )

    print()
    print("=" * 100)
    print("5. PARITY CHECK")
    print("=" * 100)

    close = np.isclose(
        optimized_scores,
        reference_scores,
        atol=ATOL,
        rtol=RTOL,
    )

    absolute_diff = np.abs(
        optimized_scores
        - reference_scores
    )

    n_exact = int(
        np.sum(
            optimized_scores
            == reference_scores
        )
    )

    n_close = int(
        close.sum()
    )

    n_mismatch = (
        len(df)
        - n_close
    )

    print()
    print(
        f"Exact matches:            "
        f"{n_exact:,} / {len(df):,}"
    )

    print(
        f"Matches within tolerance: "
        f"{n_close:,} / {len(df):,}"
    )

    print(
        f"Mismatches:               "
        f"{n_mismatch:,}"
    )

    print(
        f"Mean absolute difference: "
        f"{absolute_diff.mean():.16g}"
    )

    print(
        f"Maximum absolute diff:    "
        f"{absolute_diff.max():.16g}"
    )

    if n_mismatch:

        bad = df.loc[
            ~close,
            [
                "final_test_id",
                "article_id",
                "theme_id",
                "theme_name",
            ],
        ].copy()

        bad[
            "reference_score"
        ] = (
            reference_scores[
                ~close
            ]
        )

        bad[
            "optimized_score"
        ] = (
            optimized_scores[
                ~close
            ]
        )

        bad[
            "abs_diff"
        ] = (
            absolute_diff[
                ~close
            ]
        )

        print()
        print(
            bad
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "STOP: article-level engine failed frozen parity."
        )

    print()
    print("=" * 100)
    print("6. DISTRIBUTION CHECK")
    print("=" * 100)

    print()
    print(
        f"Reference mean: "
        f"{reference_scores.mean():.12f}"
    )

    print(
        f"Optimized mean: "
        f"{optimized_scores.mean():.12f}"
    )

    print()
    print(
        f"Reference max:  "
        f"{reference_scores.max():.12f}"
    )

    print(
        f"Optimized max:  "
        f"{optimized_scores.max():.12f}"
    )

    print()
    print(
        f"Reference zeros: "
        f"{int((reference_scores == 0.0).sum())}"
    )

    print(
        f"Optimized zeros: "
        f"{int((optimized_scores == 0.0).sum())}"
    )

    print()
    print("=" * 100)
    print(
        "ARTICLE-LEVEL ENGINE PARITY PASSED"
    )
    print("=" * 100)

    print()
    print(
        "The article-level implementation reproduces the frozen "
        "Weak V3 Continuous final-test scores."
    )

    print(
        "The scoring specification has not changed."
    )

    print(
        "PostgreSQL was not modified."
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print()
        print("FATAL ERROR:")
        print(repr(exc))

        sys.exit(1)
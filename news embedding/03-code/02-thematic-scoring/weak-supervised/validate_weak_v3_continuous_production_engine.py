#!/usr/bin/env python3

"""
Weak V3 Continuous — Production Engine Parity Test.

Validate the production implementation of the frozen Weak V3 Continuous
Candidate C against the already-frozen final-test scores.

The test verifies implementation equivalence only. It does not train,
tune, or modify the frozen scoring specification.

Requirements:
- 500 frozen final-test observations;
- no PostgreSQL writes;
- no modification of frozen predictions;
- optimized scores must reproduce the frozen reference scores within
  the locked floating-point tolerance.
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

THEMES = {
    1: "Cybersecurity Risk",
    2: "Climate / Environmental Risk",
    3: "Pandemic Risk",
    4: "Supply Chain Risk",
    5: "AI-Related Risk",
}


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
    """Apply the frozen lowercase normalization."""

    return safe_text(
        value
    ).lower()


def tokenize(value) -> list[str]:
    """Apply the frozen Weak V3 tokenization."""

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

    n = len(
        target
    )

    return [
        i
        for i in range(
            len(tokens) - n + 1
        )
        if tokens[i:i + n] == target
    ]


def minimum_distance_sets(
    tokens: list[str],
    signals1,
    signals2,
):
    """Return the minimum frozen start-token distance between signal sets."""

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
    Reconstruct the frozen baseline evidence required by Climate and AI
    proximity scoring.
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
# FROZEN CONTINUOUS SCORE
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
    """Apply the frozen density transformation x / (1 + x)."""

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

    This operator does not represent a calibrated probability.
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
# PRODUCTION SCORER
# =============================================================================

def score_article_theme(
    article_id,
    theme_id,
    theme_name,
    title,
    body,
) -> float:
    """Compute the frozen Candidate C score for one article-theme pair."""

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

    denominator = max(
        len(tokens),
        1,
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

    theme_hits = []
    risk_hits = []

    # Climate and AI proximity depend on evidence reconstructed by the
    # already-frozen Weak V3 baseline.
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
            distance = minimum_distance_sets(
                tokens,
                CLIMATE_THEME_SIGNALS,
                risk_hits,
            )

    elif (
        theme_name
        == "Cybersecurity Risk"
    ):
        distance = minimum_distance_sets(
            tokens,
            CYBER_OBJECT_SIGNALS,
            CYBER_CONTEXT_SIGNALS,
        )

    elif (
        theme_name
        == "Supply Chain Risk"
    ):
        distance = minimum_distance_sets(
            tokens,
            SUPPLY_OBJECT_SIGNALS,
            SUPPLY_DISRUPTION_SIGNALS,
        )

    elif (
        theme_name
        == "AI-Related Risk"
    ):
        if theme_hits:
            distance = minimum_distance_sets(
                tokens,
                theme_hits,
                AI_STRICT_SIGNALS,
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

    proximity = proximity_strength(
        distance
    )

    frequency_component = sat_count(
        strong_frequency
    )

    density_reinforcement = (
        DENSITY_REINFORCEMENT
        * density_component(
            strong_density_1000
        )
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


# =============================================================================
# VALIDATION
# =============================================================================

def validate_frozen_configuration() -> None:
    """Verify the complete frozen scoring lock."""

    if (
        BINARY_VERSION
        != EXPECTED_BINARY_VERSION
    ):
        raise RuntimeError(
            "Binary Weak V3 version mismatch."
        )

    if (
        CONTINUOUS_VERSION
        != EXPECTED_CONTINUOUS_VERSION
    ):
        raise RuntimeError(
            "Continuous Weak V3 version mismatch."
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


def validate_theme_mapping(
    df: pd.DataFrame,
) -> None:
    """Verify the frozen theme ID/name mapping."""

    expected_names = (
        df["theme_id"]
        .astype(int)
        .map(
            THEMES
        )
    )

    if (
        expected_names
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Unexpected theme_id in final-test data."
        )

    if not (
        expected_names
        == df["theme_name"]
    ).all():
        raise RuntimeError(
            "theme_id/theme_name mapping mismatch."
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 100)
    print(
        "WEAK V3 CONTINUOUS — "
        "PRODUCTION ENGINE PARITY TEST"
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
            "Frozen continuous prediction file not found:\n"
            f"{FROZEN_PREDICTIONS_FILE}"
        )

    gold = pd.read_csv(
        GOLD_FILE
    )

    reference = pd.read_csv(
        FROZEN_PREDICTIONS_FILE
    )

    if (
        len(gold)
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            f"Gold rows: {len(gold)} "
            f"instead of {EXPECTED_ROWS}."
        )

    if (
        len(reference)
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            f"Reference rows: {len(reference)} "
            f"instead of {EXPECTED_ROWS}."
        )

    required_gold = {
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "title",
        "body",
    }

    missing_gold = (
        required_gold
        - set(gold.columns)
    )

    if missing_gold:
        raise RuntimeError(
            f"Missing gold columns: "
            f"{sorted(missing_gold)}"
        )

    required_reference = {
        "final_test_id",
        "article_id",
        "theme_id",
        "weak_continuous_score",
    }

    missing_reference = (
        required_reference
        - set(reference.columns)
    )

    if missing_reference:
        raise RuntimeError(
            f"Missing reference columns: "
            f"{sorted(missing_reference)}"
        )

    if (
        gold["final_test_id"]
        .isna()
        .any()
        or gold["final_test_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Gold final_test_id values must be unique and non-missing."
        )

    if (
        reference["final_test_id"]
        .isna()
        .any()
        or reference["final_test_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Reference final_test_id values must be unique and non-missing."
        )

    validate_theme_mapping(
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
            "Missing frozen reference scores."
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
            "Article ID mismatch between gold and reference."
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
            "Theme ID mismatch between gold and reference."
        )

    reference_scores = (
        df["reference_score"]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        reference_scores
    ).all():
        raise RuntimeError(
            "Reference scores contain non-finite values."
        )

    if (
        (reference_scores < SCORE_MIN).any()
        or (reference_scores > SCORE_MAX).any()
    ):
        raise RuntimeError(
            "Reference scores fall outside frozen score bounds."
        )

    print()
    print(
        f"Frozen observations loaded: "
        f"{len(df):,}"
    )

    print(
        "[OK] Gold/reference alignment verified"
    )

    print()
    print("=" * 100)
    print("3. RUN PRODUCTION ENGINE")
    print("=" * 100)

    optimized_scores = []

    for i, row in enumerate(
        df.itertuples(
            index=False
        ),
        start=1,
    ):

        score = score_article_theme(
            article_id=row.article_id,
            theme_id=row.theme_id,
            theme_name=row.theme_name,
            title=row.title,
            body=row.body,
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

    if not np.isfinite(
        optimized_scores
    ).all():
        raise RuntimeError(
            "Production scores contain non-finite values."
        )

    print()
    print("=" * 100)
    print("4. PARITY CHECK")
    print("=" * 100)

    absolute_diff = np.abs(
        optimized_scores
        - reference_scores
    )

    parity_mask = np.isclose(
        optimized_scores,
        reference_scores,
        atol=ATOL,
        rtol=RTOL,
    )

    n_exact = int(
        np.sum(
            optimized_scores
            == reference_scores
        )
    )

    n_close = int(
        parity_mask.sum()
    )

    n_mismatch = int(
        len(df)
        - n_close
    )

    print()
    print(
        f"Exact floating-point matches: "
        f"{n_exact:,} / {len(df):,}"
    )

    print(
        f"Matches within tolerance:      "
        f"{n_close:,} / {len(df):,}"
    )

    print(
        f"Mismatches:                    "
        f"{n_mismatch:,}"
    )

    print(
        f"Mean absolute difference:      "
        f"{absolute_diff.mean():.16g}"
    )

    print(
        f"Maximum absolute difference:   "
        f"{absolute_diff.max():.16g}"
    )

    if n_mismatch:

        diagnostic = df.loc[
            ~parity_mask,
            [
                "final_test_id",
                "article_id",
                "theme_id",
                "theme_name",
            ],
        ].copy()

        diagnostic[
            "reference_score"
        ] = reference_scores[
            ~parity_mask
        ]

        diagnostic[
            "production_score"
        ] = optimized_scores[
            ~parity_mask
        ]

        diagnostic[
            "abs_diff"
        ] = absolute_diff[
            ~parity_mask
        ]

        print()
        print(
            "First mismatches:"
        )

        print(
            diagnostic
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "STOP: production engine does not reproduce "
            "the frozen scores."
        )

    print()
    print("=" * 100)
    print("5. DISTRIBUTION CHECK")
    print("=" * 100)

    print()
    print(
        f"Reference min:    "
        f"{reference_scores.min():.12f}"
    )

    print(
        f"Production min:   "
        f"{optimized_scores.min():.12f}"
    )

    print()
    print(
        f"Reference mean:   "
        f"{reference_scores.mean():.12f}"
    )

    print(
        f"Production mean:  "
        f"{optimized_scores.mean():.12f}"
    )

    print()
    print(
        f"Reference median: "
        f"{np.median(reference_scores):.12f}"
    )

    print(
        f"Production median:"
        f" {np.median(optimized_scores):.12f}"
    )

    print()
    print(
        f"Reference max:    "
        f"{reference_scores.max():.12f}"
    )

    print(
        f"Production max:   "
        f"{optimized_scores.max():.12f}"
    )

    print()
    print(
        f"Reference zeros:  "
        f"{int((reference_scores == 0.0).sum())}"
    )

    print(
        f"Production zeros: "
        f"{int((optimized_scores == 0.0).sum())}"
    )

    print()
    print("=" * 100)
    print("PARITY TEST PASSED")
    print("=" * 100)

    print()
    print(
        "The production engine reproduces all 500 frozen "
        "Weak V3 Continuous final-test scores within the "
        "locked tolerance."
    )

    print(
        "The frozen scoring specification was not modified."
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
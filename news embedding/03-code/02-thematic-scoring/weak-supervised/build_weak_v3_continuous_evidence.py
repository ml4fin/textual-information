#!/usr/bin/env python3

"""
Weak V3 Continuous — Continuous Evidence Matrix.

Build interpretable continuous evidence features for the closed
485-observation development set.

Target concept:
    How relevant is article i to risk theme t?

Important:
- Development set only.
- Final test set is not loaded.
- No model is fitted.
- No PostgreSQL writes.
- Gold labels are used only for diagnostics after evidence extraction.
- Frozen Weak V3 signals are not modified.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import build_gold_review as gold_builder

from weak_supervision_v3_config import (
    VERSION,
    PROXIMITY_WINDOW,
    STRONG_SIGNALS,
    AMBIGUOUS_SIGNALS,
    CLIMATE_THEME_SIGNALS,
    CYBER_OBJECT_SIGNALS,
    CYBER_CONTEXT_SIGNALS,
    SUPPLY_OBJECT_SIGNALS,
    SUPPLY_DISRUPTION_SIGNALS,
    AI_STRICT_SIGNALS,
)


# =============================================================================
# PATHS / FROZEN LOCKS
# =============================================================================

DEV_FILE = Path(
    os.getenv(
        "SUPERVISED_CV_DATASET",
        "supervised_cv_dataset.csv",
    )
)

OUTPUT_FILE = Path(
    os.getenv(
        "WEAK_V3_CONTINUOUS_EVIDENCE_OUTPUT",
        "weak_v3_continuous_evidence_matrix.csv",
    )
)

EXPECTED_N = 485

EXPECTED_VERSION = (
    "Weak Supervision V3.0 - Frozen Development Version"
)

EXPECTED_WINDOW = 50


# =============================================================================
# TEXT HELPERS
# =============================================================================

def safe_text(value) -> str:
    """Convert missing values to empty strings."""

    if pd.isna(value):
        return ""

    return str(value)


def normalize_text(value) -> str:
    """Normalize text for case-insensitive lexical matching."""

    return safe_text(
        value
    ).lower()


def tokenize(value) -> list[str]:
    """Tokenize text for frequency and proximity calculations."""

    return re.findall(
        r"[a-z0-9\-]+",
        normalize_text(value),
    )


def phrase_pattern(
    phrase,
) -> str:
    """Build a literal phrase pattern with alphanumeric boundaries."""

    phrase = normalize_text(
        phrase
    ).strip()

    return (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )


def count_phrase(
    text,
    phrase,
) -> int:
    """Count non-overlapping occurrences of a phrase."""

    if not phrase:
        return 0

    return len(
        re.findall(
            phrase_pattern(phrase),
            normalize_text(text),
        )
    )


def matching_signals(
    text,
    signals,
) -> list[str]:
    """Return unique configured signals found in the text."""

    return sorted(
        {
            str(signal).lower()
            for signal in signals
            if count_phrase(
                text,
                signal,
            ) > 0
        }
    )


def total_signal_frequency(
    text,
    signals,
) -> int:
    """Return the total number of occurrences across supplied signals."""

    return sum(
        count_phrase(
            text,
            signal,
        )
        for signal in signals
    )


def phrase_positions(
    tokens: list[str],
    phrase,
) -> list[int]:
    """Return start-token positions of a phrase."""

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
    Return the minimum start-token distance between two signal sets.

    This preserves the proximity definition used by frozen Weak V3.
    """

    distances = []

    for signal_1 in signals1:

        positions_1 = phrase_positions(
            tokens,
            signal_1,
        )

        if not positions_1:
            continue

        for signal_2 in signals2:

            positions_2 = phrase_positions(
                tokens,
                signal_2,
            )

            if not positions_2:
                continue

            distances.extend(
                abs(a - b)
                for a in positions_1
                for b in positions_2
            )

    if not distances:
        return None

    return min(
        distances
    )


def proximity_strength(
    distance,
) -> float:
    """
    Convert proximity distance into continuous evidence.

    distance = 0       -> 1.0
    distance = window  -> 0.0
    distance > window  -> 0.0
    no pair            -> 0.0

    No parameters are fitted from the gold labels.
    """

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

    if pd.isna(value):
        return []

    value = str(
        value
    ).strip().lower()

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

def get_baseline(
    row,
) -> dict:
    """
    Reconstruct evidence using the frozen Weak V3 baseline procedure.

    The returned evidence is reused as input to the continuous evidence
    features; the baseline rules themselves are not modified.
    """

    baseline_row = pd.Series(
        {
            "article_id":
                row.article_id,

            "theme_id":
                row.theme_id,

            "theme_name":
                row.theme_name,

            "title":
                row.title,

            "body":
                row.body,
        }
    )

    result = gold_builder.score_row(
        baseline_row
    )

    return {
        "baseline_pred":
            int(
                result[
                    "suggested_label"
                ]
            ),

        "rule_reason":
            str(
                result[
                    "rule_reason"
                ]
            ),

        "theme_hits":
            parse_pipe(
                result[
                    "theme_evidence"
                ]
            ),

        "risk_hits":
            parse_pipe(
                result[
                    "risk_evidence"
                ]
            ),

        "exclusion_hits":
            parse_pipe(
                result[
                    "exclusion_evidence"
                ]
            ),
    }


# =============================================================================
# EVIDENCE EXTRACTION
# =============================================================================

def extract_evidence(
    row,
) -> dict:
    """Extract interpretable continuous evidence features for one row."""

    title = safe_text(
        row.title
    )

    body = safe_text(
        row.body
    )

    full_text = (
        f"{title} {body}"
    )

    tokens = tokenize(
        full_text
    )

    theme = str(
        row.theme_name
    )

    base = get_baseline(
        row
    )

    theme_hits = base[
        "theme_hits"
    ]

    risk_hits = base[
        "risk_hits"
    ]

    exclusion_hits = base[
        "exclusion_hits"
    ]

    strong_signals = (
        STRONG_SIGNALS.get(
            theme,
            set(),
        )
    )

    ambiguous_signals = (
        AMBIGUOUS_SIGNALS.get(
            theme,
            set(),
        )
    )

    # Strong evidence.
    strong_unique = matching_signals(
        full_text,
        strong_signals,
    )

    strong_title_unique = matching_signals(
        title,
        strong_signals,
    )

    strong_body_unique = matching_signals(
        body,
        strong_signals,
    )

    strong_frequency = total_signal_frequency(
        full_text,
        strong_signals,
    )

    strong_title_frequency = total_signal_frequency(
        title,
        strong_signals,
    )

    strong_body_frequency = total_signal_frequency(
        body,
        strong_signals,
    )

    # Ambiguous evidence.
    ambiguous_unique = matching_signals(
        full_text,
        ambiguous_signals,
    )

    ambiguous_title_unique = matching_signals(
        title,
        ambiguous_signals,
    )

    ambiguous_body_unique = matching_signals(
        body,
        ambiguous_signals,
    )

    ambiguous_frequency = total_signal_frequency(
        full_text,
        ambiguous_signals,
    )

    ambiguous_title_frequency = total_signal_frequency(
        title,
        ambiguous_signals,
    )

    ambiguous_body_frequency = total_signal_frequency(
        body,
        ambiguous_signals,
    )

    # Theme evidence returned by the frozen baseline.
    theme_frequency = (
        total_signal_frequency(
            full_text,
            theme_hits,
        )
        if theme_hits
        else 0
    )

    theme_title_frequency = (
        total_signal_frequency(
            title,
            theme_hits,
        )
        if theme_hits
        else 0
    )

    theme_body_frequency = (
        total_signal_frequency(
            body,
            theme_hits,
        )
        if theme_hits
        else 0
    )

    # Risk-context evidence returned by the frozen baseline.
    risk_frequency = (
        total_signal_frequency(
            full_text,
            risk_hits,
        )
        if risk_hits
        else 0
    )

    risk_title_frequency = (
        total_signal_frequency(
            title,
            risk_hits,
        )
        if risk_hits
        else 0
    )

    risk_body_frequency = (
        total_signal_frequency(
            body,
            risk_hits,
        )
        if risk_hits
        else 0
    )

    # Theme-specific proximity.
    proximity_distance = None
    proximity_type = "none"

    if (
        theme
        == "Climate / Environmental Risk"
    ):
        if risk_hits:
            proximity_distance = (
                minimum_distance_sets(
                    tokens,
                    CLIMATE_THEME_SIGNALS,
                    risk_hits,
                )
            )

            proximity_type = (
                "climate_theme_risk"
            )

    elif (
        theme
        == "Cybersecurity Risk"
    ):
        proximity_distance = (
            minimum_distance_sets(
                tokens,
                CYBER_OBJECT_SIGNALS,
                CYBER_CONTEXT_SIGNALS,
            )
        )

        proximity_type = (
            "cyber_object_context"
        )

    elif (
        theme
        == "Supply Chain Risk"
    ):
        proximity_distance = (
            minimum_distance_sets(
                tokens,
                SUPPLY_OBJECT_SIGNALS,
                SUPPLY_DISRUPTION_SIGNALS,
            )
        )

        proximity_type = (
            "supply_object_disruption"
        )

    elif (
        theme
        == "AI-Related Risk"
    ):
        if theme_hits:
            proximity_distance = (
                minimum_distance_sets(
                    tokens,
                    theme_hits,
                    AI_STRICT_SIGNALS,
                )
            )

            proximity_type = (
                "ai_theme_adverse"
            )

    elif (
        theme
        == "Pandemic Risk"
    ):
        # Pandemic V3 relies primarily on lexical strong/ambiguous evidence
        # rather than on a separate proximity rescue rule.
        proximity_type = (
            "not_applicable"
        )

    prox_strength = (
        proximity_strength(
            proximity_distance
        )
    )

    # Article length and normalized frequencies.
    token_count = len(
        tokens
    )

    denominator = max(
        token_count,
        1,
    )

    theme_density_1000 = (
        1000.0
        * theme_frequency
        / denominator
    )

    strong_density_1000 = (
        1000.0
        * strong_frequency
        / denominator
    )

    ambiguous_density_1000 = (
        1000.0
        * ambiguous_frequency
        / denominator
    )

    risk_density_1000 = (
        1000.0
        * risk_frequency
        / denominator
    )

    return {
        "article_id":
            int(
                row.article_id
            ),

        "theme_id":
            int(
                row.theme_id
            ),

        "theme_name":
            theme,

        "human_label":
            int(
                row.human_review_decision
            ),

        # Frozen Weak V3 audit fields.
        "baseline_pred":
            base[
                "baseline_pred"
            ],

        "rule_reason":
            base[
                "rule_reason"
            ],

        # Number of distinct evidence terms.
        "n_theme_unique":
            len(
                theme_hits
            ),

        "n_risk_unique":
            len(
                risk_hits
            ),

        "n_exclusion_unique":
            len(
                exclusion_hits
            ),

        "n_strong_unique":
            len(
                strong_unique
            ),

        "n_ambiguous_unique":
            len(
                ambiguous_unique
            ),

        # Raw frequencies.
        "theme_frequency":
            theme_frequency,

        "strong_frequency":
            strong_frequency,

        "ambiguous_frequency":
            ambiguous_frequency,

        "risk_frequency":
            risk_frequency,

        # Title frequencies.
        "theme_title_frequency":
            theme_title_frequency,

        "strong_title_frequency":
            strong_title_frequency,

        "ambiguous_title_frequency":
            ambiguous_title_frequency,

        "risk_title_frequency":
            risk_title_frequency,

        # Body frequencies.
        "theme_body_frequency":
            theme_body_frequency,

        "strong_body_frequency":
            strong_body_frequency,

        "ambiguous_body_frequency":
            ambiguous_body_frequency,

        "risk_body_frequency":
            risk_body_frequency,

        # Unique title/body evidence.
        "n_strong_title_unique":
            len(
                strong_title_unique
            ),

        "n_strong_body_unique":
            len(
                strong_body_unique
            ),

        "n_ambiguous_title_unique":
            len(
                ambiguous_title_unique
            ),

        "n_ambiguous_body_unique":
            len(
                ambiguous_body_unique
            ),

        # Binary presence indicators.
        "has_theme":
            int(
                len(theme_hits) > 0
            ),

        "has_strong":
            int(
                len(strong_unique) > 0
            ),

        "has_ambiguous":
            int(
                len(ambiguous_unique) > 0
            ),

        "has_risk":
            int(
                len(risk_hits) > 0
            ),

        "has_exclusion":
            int(
                len(exclusion_hits) > 0
            ),

        "theme_in_title":
            int(
                theme_title_frequency > 0
            ),

        "strong_in_title":
            int(
                strong_title_frequency > 0
            ),

        "ambiguous_in_title":
            int(
                ambiguous_title_frequency > 0
            ),

        "risk_in_title":
            int(
                risk_title_frequency > 0
            ),

        # Proximity evidence.
        "proximity_type":
            proximity_type,

        "proximity_distance":
            (
                np.nan
                if proximity_distance is None
                else float(
                    proximity_distance
                )
            ),

        "proximity_strength":
            prox_strength,

        "within_window":
            int(
                proximity_distance is not None
                and proximity_distance
                <= PROXIMITY_WINDOW
            ),

        # Length-normalized evidence.
        "token_count":
            token_count,

        "theme_density_1000":
            theme_density_1000,

        "strong_density_1000":
            strong_density_1000,

        "ambiguous_density_1000":
            ambiguous_density_1000,

        "risk_density_1000":
            risk_density_1000,
    }


# =============================================================================
# DIAGNOSTICS
# =============================================================================

DIAGNOSTIC_FEATURES = [
    "n_theme_unique",
    "n_strong_unique",
    "n_ambiguous_unique",
    "n_risk_unique",
    "theme_frequency",
    "strong_frequency",
    "ambiguous_frequency",
    "risk_frequency",
    "theme_in_title",
    "strong_in_title",
    "ambiguous_in_title",
    "risk_in_title",
    "proximity_strength",
    "within_window",
    "theme_density_1000",
    "strong_density_1000",
    "ambiguous_density_1000",
    "risk_density_1000",
]


def print_gold_diagnostics(
    out: pd.DataFrame,
) -> None:
    """Compare mean evidence values across the two gold labels."""

    print()
    print("=" * 100)
    print("MEAN EVIDENCE BY GOLD LABEL")
    print("=" * 100)

    means = (
        out
        .groupby(
            "human_label"
        )[DIAGNOSTIC_FEATURES]
        .mean()
        .T
    )

    means["difference_1_minus_0"] = (
        means.get(1, 0)
        - means.get(0, 0)
    )

    print(
        means
        .round(4)
        .to_string()
    )


def print_theme_diagnostics(
    out: pd.DataFrame,
) -> None:
    """Report key evidence features by theme and gold label."""

    print()
    print("=" * 100)
    print("KEY EVIDENCE BY THEME AND GOLD")
    print("=" * 100)

    key_features = [
        "n_theme_unique",
        "n_strong_unique",
        "n_ambiguous_unique",
        "theme_frequency",
        "strong_frequency",
        "ambiguous_frequency",
        "theme_in_title",
        "strong_in_title",
        "proximity_strength",
        "within_window",
    ]

    table = (
        out
        .groupby(
            [
                "theme_name",
                "human_label",
            ]
        )[key_features]
        .mean()
        .round(4)
    )

    print(
        table.to_string()
    )


def print_quantiles(
    out: pd.DataFrame,
) -> None:
    """Report evidence distributions rather than relying only on means."""

    print()
    print("=" * 100)
    print("EVIDENCE QUANTILES BY GOLD LABEL")
    print("=" * 100)

    quantitative = [
        "theme_frequency",
        "strong_frequency",
        "ambiguous_frequency",
        "risk_frequency",
        "proximity_strength",
        "theme_density_1000",
        "strong_density_1000",
    ]

    quantiles = [
        0.00,
        0.25,
        0.50,
        0.75,
        0.90,
        1.00,
    ]

    for feature in quantitative:

        print()
        print(
            f"--- {feature} ---"
        )

        table = (
            out
            .groupby(
                "human_label"
            )[feature]
            .quantile(
                quantiles
            )
            .unstack()
            .round(4)
        )

        print(
            table.to_string()
        )


def print_coverage(
    out: pd.DataFrame,
) -> None:
    """Report whether any theme-related evidence is available."""

    print()
    print("=" * 100)
    print("EVIDENCE COVERAGE")
    print("=" * 100)

    diagnostic = (
        out.copy()
    )

    diagnostic[
        "any_theme_evidence"
    ] = (
        (
            (diagnostic["has_theme"] == 1)
            | (diagnostic["has_strong"] == 1)
            | (diagnostic["has_ambiguous"] == 1)
            | (diagnostic["within_window"] == 1)
        )
        .astype(int)
    )

    print()

    print(
        pd.crosstab(
            diagnostic[
                "any_theme_evidence"
            ],
            diagnostic[
                "human_label"
            ],
            margins=True,
        ).to_string()
    )

    print()
    print(
        "Coverage by theme and gold:"
    )

    coverage = (
        diagnostic
        .groupby(
            [
                "theme_name",
                "human_label",
            ]
        )[
            "any_theme_evidence"
        ]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
    )

    print(
        coverage
        .round(4)
        .to_string()
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 100)
    print(
        "WEAK V3 CONTINUOUS — "
        "CONTINUOUS EVIDENCE MATRIX"
    )
    print("=" * 100)

    # Prevent accidental extraction under a modified Weak V3 configuration.
    if (
        VERSION
        != EXPECTED_VERSION
    ):
        raise RuntimeError(
            "Weak V3 VERSION mismatch."
        )

    if (
        PROXIMITY_WINDOW
        != EXPECTED_WINDOW
    ):
        raise RuntimeError(
            "Weak V3 proximity "
            "window mismatch."
        )

    print()
    print(
        f"[OK] Frozen version: "
        f"{VERSION}"
    )

    print(
        f"[OK] Proximity window: "
        f"{PROXIMITY_WINDOW}"
    )

    if not DEV_FILE.exists():
        raise FileNotFoundError(
            f"Development file not found:\n"
            f"{DEV_FILE}"
        )

    df = pd.read_csv(
        DEV_FILE
    )

    if (
        len(df)
        != EXPECTED_N
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_N} "
            f"rows, found {len(df)}."
        )

    required_columns = {
        "article_id",
        "theme_id",
        "theme_name",
        "title",
        "body",
        "human_review_decision",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"Missing columns: "
            f"{sorted(missing_columns)}"
        )

    print()
    print(
        f"Development rows: "
        f"{len(df):,}"
    )

    print(
        "Gold labels:"
    )

    print(
        df[
            "human_review_decision"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    records = []

    for i, row in enumerate(
        df.itertuples(
            index=False
        ),
        start=1,
    ):

        records.append(
            extract_evidence(
                row
            )
        )

        if (
            i % 100
            == 0
        ):
            print(
                f"Processed "
                f"{i}/{len(df)}"
            )

    out = pd.DataFrame(
        records
    )

    # Gold labels are consulted only after evidence extraction.
    print_gold_diagnostics(
        out
    )

    print_theme_diagnostics(
        out
    )

    print_quantiles(
        out
    )

    print_coverage(
        out
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 100)
    print("COMPLETE")
    print("=" * 100)

    print()
    print(
        f"Saved:\n{OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(out):,}"
    )

    print()
    print(
        "No continuous relevance thematic score "
        "has been defined yet."
    )

    print(
        "Final test not loaded."
    )

    print(
        "PostgreSQL not modified."
    )


if __name__ == "__main__":
    main()
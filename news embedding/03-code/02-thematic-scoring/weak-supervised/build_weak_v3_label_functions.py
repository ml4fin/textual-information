#!/usr/bin/env python3

"""
Weak V3 Continuous — Label Function diagnostics.

Represent the frozen Weak V3 evidence as separate Label Functions (LFs)
and evaluate their behavior on the closed 485-observation development set.

LF outputs:
     1 = positive vote
     0 = abstain
    -1 = negative vote

Important:
- Development set only.
- Final test set is not loaded.
- No PostgreSQL writes.
- No continuous thematic score is fitted.
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
        "WEAK_V3_LF_MATRIX_OUTPUT",
        "weak_v3_label_function_matrix.csv",
    )
)

EXPECTED_N = 485

EXPECTED_VERSION = (
    "Weak Supervision V3.0 - Frozen Development Version"
)

EXPECTED_WINDOW = 50

ABSTAIN = 0
NEGATIVE = -1
POSITIVE = 1


# =============================================================================
# TEXT HELPERS
# =============================================================================

def safe_text(value) -> str:
    """Convert missing values to empty strings."""

    if pd.isna(value):
        return ""

    return str(value)


def normalize_text(text) -> str:
    """Normalize text for case-insensitive matching."""

    return safe_text(text).lower()


def tokenize(text) -> list[str]:
    """Tokenize text for proximity calculations."""

    return re.findall(
        r"[a-z0-9\-]+",
        normalize_text(text),
    )


def contains_phrase(
    text,
    phrase,
) -> bool:
    """Detect a phrase using alphanumeric token-like boundaries."""

    text = normalize_text(text)
    phrase = normalize_text(
        phrase
    ).strip()

    if not phrase:
        return False

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )

    return (
        re.search(
            pattern,
            text,
        )
        is not None
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
            if contains_phrase(
                text,
                signal,
            )
        }
    )


def phrase_positions(
    tokens: list[str],
    phrase,
) -> list[int]:
    """Return token positions where a phrase begins."""

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


def minimum_distance(
    tokens: list[str],
    phrase1,
    phrase2,
):
    """Return the minimum start-token distance between two phrases."""

    positions_1 = phrase_positions(
        tokens,
        phrase1,
    )

    positions_2 = phrase_positions(
        tokens,
        phrase2,
    )

    if (
        not positions_1
        or not positions_2
    ):
        return None

    return min(
        abs(a - b)
        for a in positions_1
        for b in positions_2
    )


def minimum_distance_sets(
    tokens: list[str],
    signals1,
    signals2,
):
    """Return the minimum distance across two sets of signals."""

    distances = []

    for signal_1 in signals1:
        for signal_2 in signals2:

            distance = minimum_distance(
                tokens,
                signal_1,
                signal_2,
            )

            if distance is not None:
                distances.append(
                    distance
                )

    if not distances:
        return None

    return min(
        distances
    )


def parse_pipe(value) -> list[str]:
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

def baseline_evidence(row) -> dict:
    """
    Reconstruct evidence using the frozen baseline scoring procedure.

    The baseline output is reused as evidence input for the diagnostic LFs;
    the underlying baseline rules are not modified here.
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
# LABEL FUNCTIONS
# =============================================================================

def compute_label_functions(
    row,
) -> dict:
    """Compute all frozen Weak V3 diagnostic Label Functions for one row."""

    title = safe_text(
        row.title
    )

    body = safe_text(
        row.body
    )

    text = (
        f"{title} {body}"
    )

    tokens = tokenize(
        text
    )

    theme = str(
        row.theme_name
    )

    base = baseline_evidence(
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

    strong_hits = matching_signals(
        text,
        strong_signals,
    )

    ambiguous_hits = matching_signals(
        text,
        ambiguous_signals,
    )

    # Strong theme evidence provides a positive vote.
    lf_strong_theme = (
        POSITIVE
        if strong_hits
        else ABSTAIN
    )

    # Ambiguous theme evidence becomes positive only when supported by
    # risk-context evidence from the frozen baseline.
    lf_ambiguous_plus_risk = (
        POSITIVE
        if (
            ambiguous_hits
            and risk_hits
        )
        else ABSTAIN
    )

    # Ambiguous evidence without supporting risk context is treated as
    # explicit negative evidence.
    lf_ambiguous_without_risk = (
        NEGATIVE
        if (
            ambiguous_hits
            and not risk_hits
        )
        else ABSTAIN
    )

    # Climate proximity.
    lf_climate_proximity = ABSTAIN
    climate_distance = None

    if (
        theme
        == "Climate / Environmental Risk"
        and risk_hits
    ):
        climate_distance = (
            minimum_distance_sets(
                tokens,
                CLIMATE_THEME_SIGNALS,
                risk_hits,
            )
        )

        if (
            climate_distance is not None
            and climate_distance
            <= PROXIMITY_WINDOW
        ):
            lf_climate_proximity = (
                POSITIVE
            )

    # Cybersecurity proximity.
    lf_cyber_proximity = ABSTAIN
    cyber_distance = None

    if (
        theme
        == "Cybersecurity Risk"
    ):
        cyber_distance = (
            minimum_distance_sets(
                tokens,
                CYBER_OBJECT_SIGNALS,
                CYBER_CONTEXT_SIGNALS,
            )
        )

        if (
            cyber_distance is not None
            and cyber_distance
            <= PROXIMITY_WINDOW
        ):
            lf_cyber_proximity = (
                POSITIVE
            )

    # Supply-chain proximity.
    lf_supply_proximity = ABSTAIN
    supply_distance = None

    if (
        theme
        == "Supply Chain Risk"
    ):
        supply_distance = (
            minimum_distance_sets(
                tokens,
                SUPPLY_OBJECT_SIGNALS,
                SUPPLY_DISRUPTION_SIGNALS,
            )
        )

        if (
            supply_distance is not None
            and supply_distance
            <= PROXIMITY_WINDOW
        ):
            lf_supply_proximity = (
                POSITIVE
            )

    # AI adverse-context evidence.
    lf_ai_strict_adverse = ABSTAIN
    lf_ai_without_adverse = ABSTAIN
    ai_distance = None

    if (
        theme
        == "AI-Related Risk"
    ):
        ai_theme_signals = (
            theme_hits
        )

        if ai_theme_signals:

            ai_distance = (
                minimum_distance_sets(
                    tokens,
                    ai_theme_signals,
                    AI_STRICT_SIGNALS,
                )
            )

            if (
                ai_distance is not None
                and ai_distance
                <= PROXIMITY_WINDOW
            ):
                lf_ai_strict_adverse = (
                    POSITIVE
                )

            else:
                lf_ai_without_adverse = (
                    NEGATIVE
                )

    # Explicit exclusion evidence provides a negative vote.
    lf_explicit_exclusion = (
        NEGATIVE
        if exclusion_hits
        else ABSTAIN
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

        "baseline_pred":
            base[
                "baseline_pred"
            ],

        "rule_reason":
            base[
                "rule_reason"
            ],

        "n_theme_hits":
            len(
                theme_hits
            ),

        "n_risk_hits":
            len(
                risk_hits
            ),

        "n_strong_hits":
            len(
                strong_hits
            ),

        "n_ambiguous_hits":
            len(
                ambiguous_hits
            ),

        "lf_strong_theme":
            lf_strong_theme,

        "lf_ambiguous_plus_risk":
            lf_ambiguous_plus_risk,

        "lf_ambiguous_without_risk":
            lf_ambiguous_without_risk,

        "lf_climate_proximity":
            lf_climate_proximity,

        "lf_cyber_proximity":
            lf_cyber_proximity,

        "lf_supply_proximity":
            lf_supply_proximity,

        "lf_ai_strict_adverse":
            lf_ai_strict_adverse,

        "lf_ai_without_adverse":
            lf_ai_without_adverse,

        "lf_explicit_exclusion":
            lf_explicit_exclusion,

        "climate_distance":
            (
                np.nan
                if climate_distance is None
                else climate_distance
            ),

        "cyber_distance":
            (
                np.nan
                if cyber_distance is None
                else cyber_distance
            ),

        "supply_distance":
            (
                np.nan
                if supply_distance is None
                else supply_distance
            ),

        "ai_distance":
            (
                np.nan
                if ai_distance is None
                else ai_distance
            ),
    }


# =============================================================================
# LABEL FUNCTION DIAGNOSTICS
# =============================================================================

LF_COLUMNS = [
    "lf_strong_theme",
    "lf_ambiguous_plus_risk",
    "lf_ambiguous_without_risk",
    "lf_climate_proximity",
    "lf_cyber_proximity",
    "lf_supply_proximity",
    "lf_ai_strict_adverse",
    "lf_ai_without_adverse",
    "lf_explicit_exclusion",
]


def lf_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute coverage and conditional vote accuracy for each LF."""

    records = []

    gold_labels = (
        df["human_label"]
        .to_numpy()
    )

    for lf in LF_COLUMNS:

        votes = (
            df[lf]
            .to_numpy()
        )

        active = (
            votes
            != ABSTAIN
        )

        positive_mask = (
            votes
            == POSITIVE
        )

        negative_mask = (
            votes
            == NEGATIVE
        )

        n_active = int(
            active.sum()
        )

        n_positive_votes = int(
            positive_mask.sum()
        )

        n_negative_votes = int(
            negative_mask.sum()
        )

        coverage = (
            n_active / len(df)
        )

        if n_active == 0:
            accuracy_when_active = (
                np.nan
            )

        else:
            predicted = (
                votes[active]
                == POSITIVE
            ).astype(int)

            accuracy_when_active = (
                predicted
                == gold_labels[active]
            ).mean()

        if n_positive_votes > 0:
            positive_precision = (
                gold_labels[
                    positive_mask
                ]
                == 1
            ).mean()
        else:
            positive_precision = (
                np.nan
            )

        if n_negative_votes > 0:
            negative_precision = (
                gold_labels[
                    negative_mask
                ]
                == 0
            ).mean()
        else:
            negative_precision = (
                np.nan
            )

        records.append(
            {
                "label_function":
                    lf,

                "active":
                    n_active,

                "coverage":
                    coverage,

                "positive_votes":
                    n_positive_votes,

                "negative_votes":
                    n_negative_votes,

                "accuracy_when_active":
                    accuracy_when_active,

                "positive_precision":
                    positive_precision,

                "negative_precision":
                    negative_precision,
            }
        )

    return pd.DataFrame(
        records
    )


def overlap_conflict_statistics(
    df: pd.DataFrame,
) -> dict:
    """Summarize LF coverage, overlap, and positive/negative conflicts."""

    label_matrix = (
        df[LF_COLUMNS]
        .to_numpy()
    )

    active_count = (
        label_matrix
        != ABSTAIN
    ).sum(axis=1)

    positive_count = (
        label_matrix
        == POSITIVE
    ).sum(axis=1)

    negative_count = (
        label_matrix
        == NEGATIVE
    ).sum(axis=1)

    conflict = (
        (positive_count > 0)
        & (negative_count > 0)
    )

    return {
        "rows_with_any_LF":
            int(
                (active_count > 0)
                .sum()
            ),

        "rows_with_no_LF":
            int(
                (active_count == 0)
                .sum()
            ),

        "rows_with_overlap":
            int(
                (active_count > 1)
                .sum()
            ),

        "rows_with_conflict":
            int(
                conflict.sum()
            ),

        "mean_active_LFs":
            float(
                active_count.mean()
            ),

        "max_active_LFs":
            int(
                active_count.max()
            ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 90)
    print(
        "WEAK V3 CONTINUOUS — "
        "LABEL FUNCTION DIAGNOSTICS"
    )
    print("=" * 90)

    # Prevent accidental diagnostics under a modified Weak V3 configuration.
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
            f"Development file "
            f"not found:\n{DEV_FILE}"
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

    print()
    print(
        f"Development rows: "
        f"{len(df):,}"
    )

    print("Gold:")

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
            compute_label_functions(
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

    stats = lf_statistics(
        out
    )

    print()
    print("=" * 90)
    print("LABEL FUNCTION PERFORMANCE")
    print("=" * 90)
    print()

    display_stats = (
        stats.copy()
    )

    for column in [
        "coverage",
        "accuracy_when_active",
        "positive_precision",
        "negative_precision",
    ]:
        display_stats[column] = (
            display_stats[column]
            .round(4)
        )

    print(
        display_stats.to_string(
            index=False
        )
    )

    diagnostics = (
        overlap_conflict_statistics(
            out
        )
    )

    print()
    print("=" * 90)
    print("OVERLAP / CONFLICT")
    print("=" * 90)

    for key, value in (
        diagnostics.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 90)
    print("ACTIVE LF COUNT")
    print("=" * 90)

    active_count = (
        out[LF_COLUMNS]
        .ne(ABSTAIN)
        .sum(axis=1)
    )

    print(
        active_count
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("=" * 90)
    print("LF VOTES BY GOLD LABEL")
    print("=" * 90)

    for lf in LF_COLUMNS:

        print()
        print(
            f"--- {lf} ---"
        )

        print(
            pd.crosstab(
                out[lf],
                out["human_label"],
                margins=True,
            ).to_string()
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
    print("=" * 90)
    print("COMPLETE")
    print("=" * 90)

    print()
    print(
        f"Saved:\n{OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(out):,}"
    )

    print()
    print(
        "No continuous thematic scores fitted."
    )

    print(
        "Final test not loaded."
    )

    print(
        "PostgreSQL not modified."
    )


if __name__ == "__main__":
    main()
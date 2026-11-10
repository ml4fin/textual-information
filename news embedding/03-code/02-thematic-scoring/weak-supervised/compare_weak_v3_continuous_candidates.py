#!/usr/bin/env python3

"""
Weak V3 Continuous — Candidate Score Comparison.

Compare a small, predefined set of interpretable continuous Weak V3
relevance-score formulas on the closed 485-observation development set.

Target interpretation:
    score(i, t) = how relevant article i is to risk theme t,
                  according to frozen Weak V3 evidence.

Important:
- Development set only.
- Final test set is not loaded.
- No classifier is fitted.
- No PostgreSQL writes.
- Candidate formulas are predefined rather than fitted.
- Gold labels are used only to compare candidate formulas on development data.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# =============================================================================
# PATHS / LOCKS
# =============================================================================

INPUT_FILE = Path(
    os.getenv(
        "WEAK_V3_CONTINUOUS_EVIDENCE_OUTPUT",
        "weak_v3_continuous_evidence_matrix.csv",
    )
)

OUTPUT_FILE = Path(
    os.getenv(
        "WEAK_V3_CANDIDATE_SCORES_OUTPUT",
        "weak_v3_continuous_candidate_scores.csv",
    )
)

EXPECTED_N = 485

TITLE_REINFORCEMENT_WEIGHT = 0.50
DENSITY_REINFORCEMENT_WEIGHT = 0.35


# =============================================================================
# TRANSFORMATIONS
# =============================================================================

def sat_count(values):
    """
    Apply a saturating transformation to count/frequency evidence.

    f(x) = 1 - exp(-x)

    Examples:
        0 -> 0.000
        1 -> 0.632
        2 -> 0.865
        3 -> 0.950

    Repeated mentions therefore provide diminishing rather than linear
    increases in evidence strength.
    """

    values = np.maximum(
        np.asarray(
            values,
            dtype=float,
        ),
        0.0,
    )

    return (
        1.0
        - np.exp(-values)
    )


def clip01(values):
    """Clip numeric values to the [0, 1] interval."""

    return np.clip(
        np.asarray(
            values,
            dtype=float,
        ),
        0.0,
        1.0,
    )


def probabilistic_union(
    *components,
):
    """
    Aggregate positive evidence components with diminishing returns.

    U(c1, ..., ck) = 1 - product(1 - cj)

    The operator keeps the resulting score in [0, 1]. It is used here as an
    evidence-aggregation rule and must not be interpreted as a calibrated
    probability model.
    """

    if not components:
        raise ValueError(
            "At least one component is required."
        )

    result = np.ones_like(
        np.asarray(
            components[0],
            dtype=float,
        )
    )

    for component in components:

        clipped = clip01(
            component
        )

        result *= (
            1.0
            - clipped
        )

    return clip01(
        1.0
        - result
    )


def union_scalar(
    *components,
) -> float:
    """Apply probabilistic union to scalar evidence components."""

    values = np.asarray(
        components,
        dtype=float,
    )

    return float(
        1.0
        - np.prod(
            1.0
            - np.clip(
                values,
                0.0,
                1.0,
            )
        )
    )


# =============================================================================
# EVIDENCE COMPONENTS
# =============================================================================

def build_components(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build normalized evidence-strength components.

    Gold labels are not used in this transformation.
    """

    out = df.copy()

    # Lexical frequency evidence.
    out["c_theme_freq"] = sat_count(
        out["theme_frequency"]
    )

    out["c_strong_freq"] = sat_count(
        out["strong_frequency"]
    )

    out["c_ambiguous_freq"] = sat_count(
        out["ambiguous_frequency"]
    )

    # Density evidence is compressed through x / (1 + x), mapping
    # non-negative values into [0, 1).
    out["c_theme_density"] = (
        out["theme_density_1000"]
        / (
            1.0
            + out["theme_density_1000"]
        )
    )

    out["c_strong_density"] = (
        out["strong_density_1000"]
        / (
            1.0
            + out["strong_density_1000"]
        )
    )

    # Binary title evidence.
    out["c_theme_title"] = (
        out["theme_in_title"]
        .astype(float)
    )

    out["c_strong_title"] = (
        out["strong_in_title"]
        .astype(float)
    )

    # Continuous proximity evidence.
    out["c_proximity"] = (
        out["proximity_strength"]
        .fillna(0.0)
        .clip(
            0.0,
            1.0,
        )
    )

    return out


# =============================================================================
# CANDIDATE SCORES
# =============================================================================

def candidate_A(
    row,
) -> float:
    """
    Candidate A — Conservative core evidence.

    Strong lexical evidence is the primary channel and theme-specific
    proximity is the secondary channel. Ambiguous evidence alone does not
    increase the score.
    """

    theme = row["theme_name"]

    if (
        theme
        == "AI-Related Risk"
    ):
        # AI relevance requires adverse-context proximity.
        return float(
            row["c_proximity"]
        )

    if (
        theme
        == "Pandemic Risk"
    ):
        # Pandemic V3 is primarily driven by strong lexical evidence.
        return float(
            row["c_strong_freq"]
        )

    # Climate, Cybersecurity, and Supply Chain combine strong lexical
    # evidence with their theme-specific proximity channel.
    return union_scalar(
        row["c_strong_freq"],
        row["c_proximity"],
    )


def candidate_B(
    row,
) -> float:
    """
    Candidate B — Core evidence with title reinforcement.

    Candidate A is extended by a predefined reinforcement when strong
    evidence occurs in the title.
    """

    theme = row["theme_name"]

    if (
        theme
        == "AI-Related Risk"
    ):
        # An AI mention in the title is intentionally insufficient by itself.
        return float(
            row["c_proximity"]
        )

    title_component = (
        TITLE_REINFORCEMENT_WEIGHT
        * row["c_strong_title"]
    )

    if (
        theme
        == "Pandemic Risk"
    ):
        return union_scalar(
            row["c_strong_freq"],
            title_component,
        )

    return union_scalar(
        row["c_strong_freq"],
        row["c_proximity"],
        title_component,
    )


def candidate_C(
    row,
) -> float:
    """
    Candidate C — Core evidence with density reinforcement.

    Candidate A is extended by a predefined contribution from the
    concentration of strong lexical evidence.
    """

    theme = row["theme_name"]

    if (
        theme
        == "AI-Related Risk"
    ):
        return float(
            row["c_proximity"]
        )

    density_component = (
        DENSITY_REINFORCEMENT_WEIGHT
        * row["c_strong_density"]
    )

    if (
        theme
        == "Pandemic Risk"
    ):
        return union_scalar(
            row["c_strong_freq"],
            density_component,
        )

    return union_scalar(
        row["c_strong_freq"],
        row["c_proximity"],
        density_component,
    )


def candidate_D(
    row,
) -> float:
    """
    Candidate D — Full interpretable evidence score.

    Combines:
    - strong lexical evidence;
    - theme-specific proximity;
    - title reinforcement;
    - density reinforcement.

    Ambiguous evidence is deliberately not rewarded independently.
    """

    theme = row["theme_name"]

    if (
        theme
        == "AI-Related Risk"
    ):
        return float(
            row["c_proximity"]
        )

    title_component = (
        TITLE_REINFORCEMENT_WEIGHT
        * row["c_strong_title"]
    )

    density_component = (
        DENSITY_REINFORCEMENT_WEIGHT
        * row["c_strong_density"]
    )

    if (
        theme
        == "Pandemic Risk"
    ):
        return union_scalar(
            row["c_strong_freq"],
            title_component,
            density_component,
        )

    return union_scalar(
        row["c_strong_freq"],
        row["c_proximity"],
        title_component,
        density_component,
    )


CANDIDATES = {
    "score_A_core":
        candidate_A,

    "score_B_title":
        candidate_B,

    "score_C_density":
        candidate_C,

    "score_D_full":
        candidate_D,
}


# =============================================================================
# METRICS
# =============================================================================

def safe_auc(
    y,
    score,
) -> float:
    """Return ROC-AUC when both gold classes are present."""

    if (
        len(
            np.unique(y)
        )
        < 2
    ):
        return np.nan

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def safe_ap(
    y,
    score,
) -> float:
    """Return average precision when at least one positive is present."""

    if (
        np.sum(y)
        == 0
    ):
        return np.nan

    return float(
        average_precision_score(
            y,
            score,
        )
    )


def evaluate_overall(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate candidate rankings across the complete development set."""

    y = (
        df["human_label"]
        .astype(int)
        .to_numpy()
    )

    records = []

    for score_name in CANDIDATES:

        score = (
            df[score_name]
            .astype(float)
            .to_numpy()
        )

        records.append(
            {
                "candidate":
                    score_name,

                "roc_auc":
                    safe_auc(
                        y,
                        score,
                    ),

                "average_precision":
                    safe_ap(
                        y,
                        score,
                    ),

                "mean_score":
                    score.mean(),

                "median_score":
                    np.median(
                        score
                    ),

                "zero_score_rate":
                    np.mean(
                        score == 0.0
                    ),

                "max_score":
                    score.max(),
            }
        )

    return pd.DataFrame(
        records
    )


def evaluate_by_theme(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate candidate rankings separately for each risk theme."""

    records = []

    for theme, part in (
        df.groupby(
            "theme_name"
        )
    ):

        y = (
            part["human_label"]
            .astype(int)
            .to_numpy()
        )

        prevalence = (
            y.mean()
        )

        for score_name in CANDIDATES:

            score = (
                part[score_name]
                .astype(float)
                .to_numpy()
            )

            records.append(
                {
                    "theme":
                        theme,

                    "candidate":
                        score_name,

                    "n":
                        len(part),

                    "positives":
                        int(
                            y.sum()
                        ),

                    "prevalence":
                        prevalence,

                    "roc_auc":
                        safe_auc(
                            y,
                            score,
                        ),

                    "average_precision":
                        safe_ap(
                            y,
                            score,
                        ),

                    "mean_score":
                        score.mean(),

                    "zero_score_rate":
                        np.mean(
                            score == 0.0
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


def score_by_gold(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize candidate-score distributions by gold label."""

    records = []

    for score_name in CANDIDATES:

        for gold_label, part in (
            df.groupby(
                "human_label"
            )
        ):

            values = (
                part[score_name]
                .astype(float)
            )

            records.append(
                {
                    "candidate":
                        score_name,

                    "gold":
                        int(
                            gold_label
                        ),

                    "n":
                        len(values),

                    "mean":
                        values.mean(),

                    "median":
                        values.median(),

                    "p75":
                        values.quantile(
                            0.75
                        ),

                    "p90":
                        values.quantile(
                            0.90
                        ),

                    "max":
                        values.max(),
                }
            )

    return pd.DataFrame(
        records
    )


def rank_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise Spearman rank correlations between candidates."""

    score_columns = list(
        CANDIDATES
    )

    return (
        df[score_columns]
        .corr(
            method="spearman"
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 100)
    print(
        "WEAK V3 CONTINUOUS — "
        "CANDIDATE SCORE COMPARISON"
    )
    print("=" * 100)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    if (
        len(df)
        != EXPECTED_N
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_N} rows, "
            f"found {len(df)}."
        )

    required_columns = {
        "article_id",
        "theme_id",
        "theme_name",
        "human_label",
        "theme_frequency",
        "strong_frequency",
        "ambiguous_frequency",
        "theme_in_title",
        "strong_in_title",
        "proximity_strength",
        "theme_density_1000",
        "strong_density_1000",
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
        df["human_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Component construction is label-independent.
    df = build_components(
        df
    )

    print()
    print(
        "Building candidate scores..."
    )

    for name, function in (
        CANDIDATES.items()
    ):

        df[name] = df.apply(
            function,
            axis=1,
        )

        if not np.isfinite(
            df[name]
        ).all():
            raise RuntimeError(
                f"{name} contains non-finite values."
            )

        if (
            (df[name] < 0.0).any()
            or (df[name] > 1.0).any()
        ):
            raise RuntimeError(
                f"{name} outside [0, 1]."
            )

        print(
            f"[OK] {name}"
        )

    overall = evaluate_overall(
        df
    )

    print()
    print("=" * 100)
    print("OVERALL DEVELOPMENT PERFORMANCE")
    print("=" * 100)

    print(
        overall
        .round(6)
        .to_string(
            index=False
        )
    )

    by_theme = evaluate_by_theme(
        df
    )

    print()
    print("=" * 100)
    print("PERFORMANCE BY THEME")
    print("=" * 100)

    print(
        by_theme
        .round(6)
        .to_string(
            index=False
        )
    )

    distributions = score_by_gold(
        df
    )

    print()
    print("=" * 100)
    print("SCORE DISTRIBUTIONS BY GOLD LABEL")
    print("=" * 100)

    print(
        distributions
        .round(6)
        .to_string(
            index=False
        )
    )

    correlations = rank_correlations(
        df
    )

    print()
    print("=" * 100)
    print("SPEARMAN RANK CORRELATIONS")
    print("=" * 100)

    print(
        correlations
        .round(4)
        .to_string()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
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
        f"Rows: {len(df):,}"
    )

    print()
    print("IMPORTANT:")
    print(
        "- No candidate score has been "
        "declared final yet."
    )
    print(
        "- Final test set was not loaded."
    )
    print(
        "- PostgreSQL was not modified."
    )


if __name__ == "__main__":
    main()
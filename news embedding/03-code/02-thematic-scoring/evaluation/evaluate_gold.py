#!/usr/bin/env python3

"""
GOLD REVIEW — AUTOMATIC LABEL EVALUATION
========================================

Evaluate the automatically suggested binary labels against the manually
reviewed human decisions.

Ground truth:
    human_review_decision

Automatic prediction:
    suggested_label

The script reports:
- global class distributions;
- confusion matrix;
- global classification metrics;
- classification metrics by theme;
- false-positive and false-negative analysis.

It also exports:
- all classification errors;
- false negatives;
- false positives;
- theme-level metrics.

IMPORTANT
---------
This evaluates the automatic label-generation procedure used before manual
review. It does not evaluate the final supervised model.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = Path(
    os.getenv(
        "GOLD_REVIEW_FILE",
        "gold_review_high.csv",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "GOLD_EVALUATION_DIR",
        "gold_evaluation",
    )
)

ERRORS_FILE = (
    OUTPUT_DIR
    / "gold_errors.csv"
)

FN_FILE = (
    OUTPUT_DIR
    / "gold_false_negatives.csv"
)

FP_FILE = (
    OUTPUT_DIR
    / "gold_false_positives.csv"
)

THEME_METRICS_FILE = (
    OUTPUT_DIR
    / "gold_metrics_by_theme.csv"
)

TRUE_COL = "human_review_decision"
PRED_COL = "suggested_label"
THEME_COL = "theme_name"

EXPECTED_LABELS = {0, 1}


# =============================================================================
# LOAD AND VALIDATE DATA
# =============================================================================

def load_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = {
        TRUE_COL,
        PRED_COL,
        THEME_COL,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "Input dataset is empty."
        )

    return df


def prepare_evaluation_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    eval_df = df.copy()

    eval_df[TRUE_COL] = pd.to_numeric(
        eval_df[TRUE_COL],
        errors="coerce",
    )

    eval_df[PRED_COL] = pd.to_numeric(
        eval_df[PRED_COL],
        errors="coerce",
    )

    missing_true = int(
        eval_df[TRUE_COL]
        .isna()
        .sum()
    )

    missing_pred = int(
        eval_df[PRED_COL]
        .isna()
        .sum()
    )

    if (
        missing_true > 0
        or missing_pred > 0
    ):
        raise ValueError(
            "Missing or non-numeric labels detected.\n"
            f"{TRUE_COL}: {missing_true}\n"
            f"{PRED_COL}: {missing_pred}"
        )

    eval_df[TRUE_COL] = (
        eval_df[TRUE_COL]
        .astype(int)
    )

    eval_df[PRED_COL] = (
        eval_df[PRED_COL]
        .astype(int)
    )

    true_values = set(
        eval_df[TRUE_COL]
        .unique()
        .tolist()
    )

    pred_values = set(
        eval_df[PRED_COL]
        .unique()
        .tolist()
    )

    if not true_values.issubset(
        EXPECTED_LABELS
    ):
        raise ValueError(
            "Ground-truth labels must be binary {0, 1}. "
            f"Found: {sorted(true_values)}"
        )

    if not pred_values.issubset(
        EXPECTED_LABELS
    ):
        raise ValueError(
            "Suggested labels must be binary {0, 1}. "
            f"Found: {sorted(pred_values)}"
        )

    if eval_df[THEME_COL].isna().any():
        raise ValueError(
            "Missing theme names detected."
        )

    return eval_df


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred,
) -> dict:

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
        .ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "n": len(y_true),
        "actual_0": int(
            (y_true == 0).sum()
        ),
        "actual_1": int(
            (y_true == 1).sum()
        ),
        "predicted_0": int(
            (y_pred == 0).sum()
        ),
        "predicted_1": int(
            (y_pred == 1).sum()
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_distribution(
    series: pd.Series,
    title: str,
):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    counts = (
        series
        .value_counts()
        .sort_index()
    )

    percentages = (
        series
        .value_counts(
            normalize=True
        )
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nCounts:")
    print(counts)

    print("\nPercentages:")
    print(percentages)


def print_global_metrics(
    metrics: dict,
):

    print()
    print("=" * 100)
    print("GLOBAL CONFUSION MATRIX")
    print("=" * 100)

    print(
        "\n"
        "                     PREDICTED\n"
        "                  0           1\n"
        f"ACTUAL 0      "
        f"{metrics['TN']:5d}       "
        f"{metrics['FP']:5d}\n"
        f"ACTUAL 1      "
        f"{metrics['FN']:5d}       "
        f"{metrics['TP']:5d}"
    )

    print()
    print("=" * 100)
    print("GLOBAL METRICS")
    print("=" * 100)

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
    ]:

        value = metrics[metric]

        if np.isnan(value):
            display = "n/a"
        else:
            display = (
                f"{value:.4f} "
                f"({value * 100:.2f}%)"
            )

        print(
            f"{metric:20s}: {display}"
        )


# =============================================================================
# THEME-LEVEL EVALUATION
# =============================================================================

def calculate_theme_metrics(
    eval_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for theme, group in eval_df.groupby(
        THEME_COL,
        sort=True,
    ):

        metrics = calculate_metrics(
            group[TRUE_COL],
            group[PRED_COL],
        )

        rows.append(
            {
                "theme_name": theme,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def classify_errors(
    eval_df: pd.DataFrame,
) -> pd.DataFrame:

    result = eval_df.copy()

    result["error_type"] = "CORRECT"

    result.loc[
        (
            (result[TRUE_COL] == 1)
            & (result[PRED_COL] == 0)
        ),
        "error_type",
    ] = "FN"

    result.loc[
        (
            (result[TRUE_COL] == 0)
            & (result[PRED_COL] == 1)
        ),
        "error_type",
    ] = "FP"

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "GOLD REVIEW — "
        "AUTOMATIC LABEL EVALUATION"
    )
    print("=" * 100)

    df = load_data()

    print(
        f"\nInput observations: "
        f"{len(df):,}"
    )

    eval_df = prepare_evaluation_data(
        df
    )

    print(
        f"Evaluated observations: "
        f"{len(eval_df):,}"
    )

    print_distribution(
        eval_df[TRUE_COL],
        "GROUND-TRUTH DISTRIBUTION",
    )

    print_distribution(
        eval_df[PRED_COL],
        "AUTOMATIC LABEL DISTRIBUTION",
    )

    y_true = eval_df[TRUE_COL]
    y_pred = eval_df[PRED_COL]

    global_metrics = (
        calculate_metrics(
            y_true,
            y_pred,
        )
    )

    print_global_metrics(
        global_metrics
    )

    print()
    print("=" * 100)
    print("CLASSIFICATION REPORT")
    print("=" * 100)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            digits=4,
            zero_division=0,
        )
    )

    theme_metrics = (
        calculate_theme_metrics(
            eval_df
        )
    )

    print()
    print("=" * 100)
    print("METRICS BY THEME")
    print("=" * 100)

    print(
        theme_metrics
        .round(4)
        .to_string(
            index=False
        )
    )

    evaluated = classify_errors(
        eval_df
    )

    errors = evaluated[
        evaluated["error_type"].isin(
            ["FN", "FP"]
        )
    ].copy()

    false_negatives = evaluated[
        evaluated["error_type"] == "FN"
    ].copy()

    false_positives = evaluated[
        evaluated["error_type"] == "FP"
    ].copy()

    print()
    print("=" * 100)
    print("ERROR ANALYSIS")
    print("=" * 100)

    print(
        f"\nTotal errors:     "
        f"{len(errors):,}"
    )

    print(
        f"False negatives: "
        f"{len(false_negatives):,}"
    )

    print(
        f"False positives: "
        f"{len(false_positives):,}"
    )

    error_summary = (
        evaluated
        .groupby(
            [
                THEME_COL,
                "error_type",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print(
        "\nErrors by theme:"
    )

    print(
        error_summary
        .to_string()
    )

    print(
        "\nFalse negatives by theme:"
    )

    print(
        false_negatives[
            THEME_COL
        ]
        .value_counts()
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    print(
        "\nFalse positives by theme:"
    )

    print(
        false_positives[
            THEME_COL
        ]
        .value_counts()
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors.to_csv(
        ERRORS_FILE,
        index=False,
    )

    false_negatives.to_csv(
        FN_FILE,
        index=False,
    )

    false_positives.to_csv(
        FP_FILE,
        index=False,
    )

    theme_metrics.to_csv(
        THEME_METRICS_FILE,
        index=False,
    )

    print()
    print("=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print(
        f"\nAll errors:      "
        f"{ERRORS_FILE}"
    )

    print(
        f"False negatives: "
        f"{FN_FILE}"
    )

    print(
        f"False positives: "
        f"{FP_FILE}"
    )

    print(
        f"Theme metrics:   "
        f"{THEME_METRICS_FILE}"
    )

    print(
        "\nEvaluation completed."
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

"""
FINAL EVALUATION — SUPERVISED V3 FROZEN
========================================

Evaluate the already-frozen Supervised V3 predictions on the
untouched final test.

Methodological constraints
--------------------------
- No training.
- No refitting.
- No hyperparameter tuning.
- No threshold tuning.
- No calibration.
- No modification of predictions or gold labels.
- The frozen prediction artifact is verified by SHA-256 before evaluation.

Outputs
-------
- Overall metrics
- Metrics by theme
- Metrics by sample block
- Metrics by theme × sample block
- Confusion matrix
- Observation-level error file
- Summary JSON
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(
    os.getenv(
        "FINAL_EVALUATION_DIR",
        "08_final_evaluation",
    )
)

PREDICTIONS_FILE = Path(
    os.getenv(
        "FINAL_SUPERVISED_V3_PREDICTIONS",
        str(
            BASE_DIR
            / "04_supervised"
            / "final_supervised_v3_predictions.csv"
        ),
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "FINAL_SUPERVISED_V3_EVAL_DIR",
        str(
            BASE_DIR
            / "04_supervised"
            / "evaluation"
        ),
    )
)

OVERALL_FILE = (
    OUTPUT_DIR
    / "supervised_v3_final_overall_metrics.csv"
)

BY_THEME_FILE = (
    OUTPUT_DIR
    / "supervised_v3_final_metrics_by_theme.csv"
)

BY_BLOCK_FILE = (
    OUTPUT_DIR
    / "supervised_v3_final_metrics_by_block.csv"
)

BY_THEME_BLOCK_FILE = (
    OUTPUT_DIR
    / "supervised_v3_final_metrics_by_theme_block.csv"
)

CONFUSION_FILE = (
    OUTPUT_DIR
    / "supervised_v3_final_confusion_matrix.csv"
)

ERRORS_FILE = (
    OUTPUT_DIR
    / "supervised_v3_final_errors.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "supervised_v3_final_evaluation_summary.json"
)

EXPECTED_PREDICTIONS_SHA256 = (
    "838a353221c4532d2b0687cbaa221f07"
    "f9b9daa85e2126e70146968eacce6d4c"
)

EXPECTED_ROWS = 500
EXPECTED_ARTICLES = 300
EXPECTED_THEMES = 5
EXPECTED_THRESHOLD = 0.34

EXPECTED_THEME_NAMES = {
    "AI-Related Risk",
    "Climate / Environmental Risk",
    "Cybersecurity Risk",
    "Pandemic Risk",
    "Supply Chain Risk",
}

MODEL_NAME = "Supervised V3 Hybrid"
MODEL_VERSION = "supervised_v3_frozen"


# =============================================================================
# UTILITIES
# =============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return np.nan

    return numerator / denominator


def safe_auc(
    y_true,
    y_score,
):

    if len(
        np.unique(y_true)
    ) < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        y_score,
    )


def safe_ap(
    y_true,
    y_score,
):

    if np.sum(
        y_true == 1
    ) == 0:
        return np.nan

    return average_precision_score(
        y_true,
        y_score,
    )


def json_safe(
    value,
):

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    if pd.isna(value):
        return None

    return value


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    df: pd.DataFrame,
) -> dict:

    y_true = (
        df["human_label"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        df["supervised_prediction"]
        .astype(int)
        .to_numpy()
    )

    y_score = (
        df["supervised_probability"]
        .astype(float)
        .to_numpy()
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
        .ravel()
    )

    n = len(df)

    n_negative = int(
        np.sum(y_true == 0)
    )

    n_positive = int(
        np.sum(y_true == 1)
    )

    predicted_negative = int(
        np.sum(y_pred == 0)
    )

    predicted_positive = int(
        np.sum(y_pred == 1)
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    balanced_accuracy = (
        (recall + specificity) / 2
        if np.isfinite(specificity)
        else np.nan
    )

    npv = safe_divide(
        tn,
        tn + fn,
    )

    return {
        "n":
            int(n),

        "unique_articles":
            int(
                df[
                    "article_id"
                ]
                .nunique()
            ),

        "gold_negative":
            n_negative,

        "gold_positive":
            n_positive,

        "gold_positive_rate":
            safe_divide(
                n_positive,
                n,
            ),

        "predicted_negative":
            predicted_negative,

        "predicted_positive":
            predicted_positive,

        "predicted_positive_rate":
            safe_divide(
                predicted_positive,
                n,
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(recall),

        "specificity":
            (
                float(specificity)
                if np.isfinite(
                    specificity
                )
                else np.nan
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "balanced_accuracy":
            (
                float(
                    balanced_accuracy
                )
                if np.isfinite(
                    balanced_accuracy
                )
                else np.nan
            ),

        "npv":
            (
                float(npv)
                if np.isfinite(npv)
                else np.nan
            ),

        "roc_auc":
            float(
                safe_auc(
                    y_true,
                    y_score,
                )
            )
            if np.isfinite(
                safe_auc(
                    y_true,
                    y_score,
                )
            )
            else np.nan,

        "average_precision":
            float(
                safe_ap(
                    y_true,
                    y_score,
                )
            )
            if np.isfinite(
                safe_ap(
                    y_true,
                    y_score,
                )
            )
            else np.nan,

        "score_min":
            float(
                np.min(y_score)
            ),

        "score_mean":
            float(
                np.mean(y_score)
            ),

        "score_median":
            float(
                np.median(y_score)
            ),

        "score_max":
            float(
                np.max(y_score)
            ),
    }


def grouped_metrics(
    df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:

    rows = []

    grouper = (
        group_columns[0]
        if len(group_columns) == 1
        else group_columns
    )

    for keys, subset in df.groupby(
        grouper,
        sort=True,
        dropna=False,
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (keys,)

        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        row.update(
            calculate_metrics(
                subset
            )
        )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FORMATTING
# =============================================================================

def fmt_metric(
    value,
) -> str:

    if pd.isna(value):
        return "NA"

    return f"{value:.6f}"


def print_metric_summary(
    metrics,
):

    print(
        f"""
N:                     {metrics['n']:,}
Unique articles:       {metrics['unique_articles']:,}

Gold negatives:        {metrics['gold_negative']:,}
Gold positives:        {metrics['gold_positive']:,}
Gold positive rate:    {metrics['gold_positive_rate']:.2%}

Predicted negatives:   {metrics['predicted_negative']:,}
Predicted positives:   {metrics['predicted_positive']:,}
Predicted positive rate:
                       {metrics['predicted_positive_rate']:.2%}

Confusion matrix:
  TN = {metrics['tn']:,}
  FP = {metrics['fp']:,}
  FN = {metrics['fn']:,}
  TP = {metrics['tp']:,}

Accuracy:              {fmt_metric(metrics['accuracy'])}
Precision:             {fmt_metric(metrics['precision'])}
Recall / Sensitivity:  {fmt_metric(metrics['recall'])}
Specificity:           {fmt_metric(metrics['specificity'])}
F1:                    {fmt_metric(metrics['f1'])}
Balanced Accuracy:     {fmt_metric(metrics['balanced_accuracy'])}
NPV:                   {fmt_metric(metrics['npv'])}
ROC-AUC:               {fmt_metric(metrics['roc_auc'])}
Average Precision:     {fmt_metric(metrics['average_precision'])}
"""
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 105)
    print(
        "FINAL EVALUATION — "
        "SUPERVISED V3 FROZEN"
    )
    print("=" * 105)

    # -------------------------------------------------------------------------
    # Verify frozen prediction artifact
    # -------------------------------------------------------------------------

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            "Prediction file not found:\n"
            f"{PREDICTIONS_FILE}"
        )

    actual_hash = sha256_file(
        PREDICTIONS_FILE
    )

    if (
        actual_hash
        != EXPECTED_PREDICTIONS_SHA256
    ):
        raise RuntimeError(
            "Frozen prediction SHA-256 mismatch.\n"
            f"Expected: "
            f"{EXPECTED_PREDICTIONS_SHA256}\n"
            f"Found:    {actual_hash}"
        )

    print(
        "\n[OK] frozen prediction "
        "SHA-256 verified"
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    required = {
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "sample_block",
        "human_label",
        "supervised_probability",
        "supervised_prediction",
        "threshold",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    # -------------------------------------------------------------------------
    # Structural integrity
    # -------------------------------------------------------------------------

    print()
    print("=" * 105)
    print("1. INTEGRITY CHECKS")
    print("=" * 105)

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(df)}."
        )

    if (
        df["article_id"].nunique()
        != EXPECTED_ARTICLES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_ARTICLES} "
            "unique articles."
        )

    if (
        df["theme_id"].nunique()
        != EXPECTED_THEMES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_THEMES} themes."
        )

    if (
        df["final_test_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate final_test_id."
        )

    if df[
        [
            "article_id",
            "theme_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate article-theme pairs."
        )

    actual_themes = set(
        df[
            "theme_name"
        ]
        .dropna()
        .astype(str)
    )

    if (
        actual_themes
        != EXPECTED_THEME_NAMES
    ):
        raise RuntimeError(
            "Theme universe does not match "
            "the frozen Supervised V3 themes.\n"
            f"Found: {sorted(actual_themes)}"
        )

    # Each theme_id must correspond to exactly one theme_name.
    theme_mapping = (
        df[
            [
                "theme_id",
                "theme_name",
            ]
        ]
        .drop_duplicates()
    )

    if (
        theme_mapping[
            "theme_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "A theme_id maps to multiple "
            "theme_name values."
        )

    if (
        theme_mapping[
            "theme_name"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "A theme_name maps to multiple "
            "theme_id values."
        )

    # -------------------------------------------------------------------------
    # Validate labels, scores and threshold
    # -------------------------------------------------------------------------

    df[
        "human_label"
    ] = pd.to_numeric(
        df["human_label"],
        errors="raise",
    )

    df[
        "supervised_prediction"
    ] = pd.to_numeric(
        df[
            "supervised_prediction"
        ],
        errors="raise",
    )

    df[
        "supervised_probability"
    ] = pd.to_numeric(
        df[
            "supervised_probability"
        ],
        errors="raise",
    )

    df[
        "threshold"
    ] = pd.to_numeric(
        df["threshold"],
        errors="raise",
    )

    numeric_columns = [
        "human_label",
        "supervised_prediction",
        "supervised_probability",
        "threshold",
    ]

    if (
        df[
            numeric_columns
        ]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Missing numeric values "
            "in frozen prediction artifact."
        )

    probabilities = (
        df[
            "supervised_probability"
        ]
        .to_numpy(
            dtype=float
        )
    )

    thresholds = (
        df[
            "threshold"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Non-finite supervised scores."
        )

    if (
        (probabilities < 0).any()
        or
        (probabilities > 1).any()
    ):
        raise RuntimeError(
            "Supervised scores outside [0,1]."
        )

    if not np.isfinite(
        thresholds
    ).all():
        raise RuntimeError(
            "Non-finite threshold values."
        )

    if not set(
        df[
            "human_label"
        ].unique()
    ).issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "Invalid human labels."
        )

    if not set(
        df[
            "supervised_prediction"
        ].unique()
    ).issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "Invalid supervised predictions."
        )

    df[
        "human_label"
    ] = (
        df[
            "human_label"
        ]
        .astype(int)
    )

    df[
        "supervised_prediction"
    ] = (
        df[
            "supervised_prediction"
        ]
        .astype(int)
    )

    if not np.all(
        np.isclose(
            thresholds,
            EXPECTED_THRESHOLD,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError(
            "Stored threshold differs "
            "from frozen threshold 0.34."
        )

    reconstructed = (
        probabilities
        >= EXPECTED_THRESHOLD
    ).astype(int)

    if not np.array_equal(
        reconstructed,
        df[
            "supervised_prediction"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "Stored predictions do not match "
            "score >= 0.34."
        )

    print(
        f"\nRows: "
        f"{len(df):,} OK"
    )

    print(
        f"Unique articles: "
        f"{df['article_id'].nunique():,} OK"
    )

    print(
        f"Themes: "
        f"{df['theme_id'].nunique():,} OK"
    )

    print(
        "Identifiers and theme mapping: OK"
    )

    print(
        "Labels and predictions binary: OK"
    )

    print(
        "Scores finite and within [0,1]: OK"
    )

    print(
        "Frozen threshold = 0.34: OK"
    )

    print(
        "Stored predictions reproduce "
        "frozen thresholding: OK"
    )

    # -------------------------------------------------------------------------
    # Overall
    # -------------------------------------------------------------------------

    print()
    print("=" * 105)
    print(
        "2. OVERALL FINAL-TEST PERFORMANCE"
    )
    print("=" * 105)

    overall = calculate_metrics(
        df
    )

    print_metric_summary(
        overall
    )

    overall_df = pd.DataFrame(
        [overall]
    )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    confusion_df = pd.DataFrame(
        [
            {
                "actual":
                    "negative",

                "predicted_negative":
                    overall["tn"],

                "predicted_positive":
                    overall["fp"],
            },
            {
                "actual":
                    "positive",

                "predicted_negative":
                    overall["fn"],

                "predicted_positive":
                    overall["tp"],
            },
        ]
    )

    # -------------------------------------------------------------------------
    # Grouped diagnostics
    # -------------------------------------------------------------------------

    print()
    print("=" * 105)
    print("3. PERFORMANCE BY THEME")
    print("=" * 105)

    by_theme = grouped_metrics(
        df,
        [
            "theme_id",
            "theme_name",
        ],
    )

    print(
        by_theme.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print()
    print("=" * 105)
    print(
        "4. PERFORMANCE BY SAMPLE BLOCK"
    )
    print("=" * 105)

    by_block = grouped_metrics(
        df,
        [
            "sample_block",
        ],
    )

    print(
        by_block.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print()
    print("=" * 105)
    print(
        "5. PERFORMANCE BY THEME × BLOCK"
    )
    print("=" * 105)

    by_theme_block = grouped_metrics(
        df,
        [
            "sample_block",
            "theme_id",
            "theme_name",
        ],
    )

    print(
        by_theme_block.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    # -------------------------------------------------------------------------
    # Error analysis
    # -------------------------------------------------------------------------

    print()
    print("=" * 105)
    print("6. ERROR ANALYSIS")
    print("=" * 105)

    errors = df[
        df[
            "human_label"
        ]
        !=
        df[
            "supervised_prediction"
        ]
    ].copy()

    errors[
        "error_type"
    ] = np.where(
        (
            (errors["human_label"] == 0)
            &
            (
                errors[
                    "supervised_prediction"
                ]
                == 1
            )
        ),
        "FP",
        "FN",
    )

    # Distance from the frozen decision boundary is
    # useful for reviewing the most confident errors.
    errors[
        "distance_from_threshold"
    ] = np.abs(
        errors[
            "supervised_probability"
        ]
        - EXPECTED_THRESHOLD
    )

    errors = errors.sort_values(
        [
            "error_type",
            "distance_from_threshold",
        ],
        ascending=[
            True,
            False,
        ],
    )

    print(
        f"\nTotal errors: "
        f"{len(errors):,}"
    )

    print(
        "\nError types:"
    )

    print(
        errors[
            "error_type"
        ]
        .value_counts()
        .to_string()
    )

    if not errors.empty:

        print(
            "\nErrors by theme:"
        )

        print(
            pd.crosstab(
                errors[
                    "theme_name"
                ],
                errors[
                    "error_type"
                ],
            )
            .to_string()
        )

        print(
            "\nErrors by block:"
        )

        print(
            pd.crosstab(
                errors[
                    "sample_block"
                ],
                errors[
                    "error_type"
                ],
            )
            .to_string()
        )

    # -------------------------------------------------------------------------
    # Protect final evaluation artifacts
    # -------------------------------------------------------------------------

    print()
    print("=" * 105)
    print("7. SAVE FINAL EVALUATION")
    print("=" * 105)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = [
        OVERALL_FILE,
        BY_THEME_FILE,
        BY_BLOCK_FILE,
        BY_THEME_BLOCK_FILE,
        CONFUSION_FILE,
        ERRORS_FILE,
        SUMMARY_FILE,
    ]

    existing = [
        path
        for path in output_files
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Final evaluation artifacts "
            "already exist; refusing "
            "silent overwrite:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------

    overall_df.to_csv(
        OVERALL_FILE,
        index=False,
    )

    by_theme.to_csv(
        BY_THEME_FILE,
        index=False,
    )

    by_block.to_csv(
        BY_BLOCK_FILE,
        index=False,
    )

    by_theme_block.to_csv(
        BY_THEME_BLOCK_FILE,
        index=False,
    )

    confusion_df.to_csv(
        CONFUSION_FILE,
        index=False,
    )

    errors.to_csv(
        ERRORS_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = {
        "status":
            "FINAL_TEST_EVALUATION",

        "evaluated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "model":
            MODEL_NAME,

        "model_version":
            MODEL_VERSION,

        "evaluation_set":
            "untouched_final_test",

        "threshold":
            EXPECTED_THRESHOLD,

        "prediction_file":
            str(
                PREDICTIONS_FILE
            ),

        "prediction_sha256":
            actual_hash,

        "overall_metrics": {
            key:
                json_safe(value)

            for key, value
            in overall.items()
        },

        "methodological_lock": {
            "training_performed":
                False,

            "refit_performed":
                False,

            "model_changed":
                False,

            "features_changed":
                False,

            "hyperparameters_changed":
                False,

            "threshold_changed":
                False,

            "calibration_changed":
                False,

            "final_test_used_for_tuning":
                False,
        },

        "score_interpretation": (
            "supervised_probability is the "
            "uncalibrated class-1 score returned "
            "by the frozen logistic-regression "
            "pipeline. It is used as a continuous "
            "ranking score for ROC-AUC and average "
            "precision, but should not be interpreted "
            "as an empirically calibrated probability."
        ),
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    # -------------------------------------------------------------------------
    # Output fingerprints
    # -------------------------------------------------------------------------

    print(
        "\nSaved final evaluation artifacts:"
    )

    for path in output_files:

        print(
            f"\n{path}"
        )

        print(
            f"SHA-256: "
            f"{sha256_file(path)}"
        )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    print()
    print("=" * 105)
    print(
        "FINAL SUPERVISED V3 "
        "EVALUATION COMPLETE"
    )
    print("=" * 105)

    print(
        """
These are final untouched-test results.

Do not change the model, feature representation, hyperparameters,
threshold, or calibration on the basis of these results.
"""
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            "\nFATAL ERROR:"
        )

        print(
            repr(exc)
        )

        sys.exit(1)
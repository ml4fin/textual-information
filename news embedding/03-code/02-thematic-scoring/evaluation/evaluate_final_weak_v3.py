#!/usr/bin/env python3

"""
FINAL TEST — WEAK SUPERVISION V3.0 FROZEN EVALUATION
=====================================================

Evaluate the already-frozen Weak Supervision V3.0 binary predictions
against the frozen final-test gold labels.

Methodological constraints
--------------------------
- Predictions are not regenerated here.
- Rules and signals are not modified here.
- No threshold or window is selected here.
- Frozen gold and prediction SHA-256 fingerprints are verified.
- Final-test labels are used only for evaluation.
- This script evaluates binary classification performance.
"""

from __future__ import annotations

import hashlib
import json
import os

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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

GOLD_FILE = Path(
    os.getenv(
        "FINAL_TEST_GOLD_FILE",
        str(
            BASE_DIR
            / "01_gold"
            / "final_test_gold_frozen.csv"
        ),
    )
)

PREDICTION_FILE = Path(
    os.getenv(
        "FINAL_WEAK_V3_PREDICTIONS",
        str(
            BASE_DIR
            / "03_weak_supervision"
            / "final_weak_v3_predictions.csv"
        ),
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "FINAL_WEAK_V3_EVAL_DIR",
        str(
            BASE_DIR
            / "03_weak_supervision"
            / "evaluation"
        ),
    )
)

EXPECTED_GOLD_SHA256 = (
    "d4e65929a1831f7823bd92546b100647"
    "96379a55e3924fd642e5366bade19e21"
)

EXPECTED_PREDICTION_SHA256 = (
    "2fa56210b48d074596340f5ba2abfe04"
    "b9f503236904c54658b5f78482eeea41"
)

EXPECTED_ROWS = 500
EXPECTED_ARTICLES = 300
EXPECTED_THEMES = 5

MODEL_VERSION = (
    "Weak Supervision V3.0 - "
    "Frozen Development Version"
)

EXPECTED_THEME_NAMES = {
    "AI-Related Risk",
    "Climate / Environmental Risk",
    "Cybersecurity Risk",
    "Pandemic Risk",
    "Supply Chain Risk",
}


# =============================================================================
# OUTPUT FILES
# =============================================================================

OVERALL_FILE = (
    OUTPUT_DIR
    / "weak_v3_final_overall_metrics.csv"
)

THEME_FILE = (
    OUTPUT_DIR
    / "weak_v3_final_by_theme.csv"
)

BLOCK_FILE = (
    OUTPUT_DIR
    / "weak_v3_final_by_block.csv"
)

THEME_BLOCK_FILE = (
    OUTPUT_DIR
    / "weak_v3_final_by_theme_block.csv"
)

CONFUSION_FILE = (
    OUTPUT_DIR
    / "weak_v3_final_confusion_matrix.csv"
)

ERRORS_FILE = (
    OUTPUT_DIR
    / "weak_v3_final_errors.csv"
)

MERGED_FILE = (
    OUTPUT_DIR
    / "weak_v3_final_evaluated_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "weak_v3_final_evaluation_summary.json"
)


# =============================================================================
# HASHING
# =============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    group: pd.DataFrame,
) -> dict:

    y_true = (
        group["human_label"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        group["pred_weak_v3"]
        .astype(int)
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

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    balanced_accuracy = (
        (recall + specificity) / 2
        if np.isfinite(specificity)
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    return {
        "n":
            int(len(group)),

        "gold_negative":
            int((y_true == 0).sum()),

        "gold_positive":
            int((y_true == 1).sum()),

        "predicted_negative":
            int((y_pred == 0).sum()),

        "predicted_positive":
            int((y_pred == 1).sum()),

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
                if np.isfinite(specificity)
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
                float(balanced_accuracy)
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

    for keys, group in df.groupby(
        grouper,
        sort=True,
        dropna=False,
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (keys,)

        result = dict(
            zip(
                group_columns,
                keys,
            )
        )

        result.update(
            calculate_metrics(
                group
            )
        )

        rows.append(result)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "FINAL TEST — WEAK SUPERVISION V3.0 "
        "FROZEN EVALUATION"
    )
    print("=" * 110)

    # -------------------------------------------------------------------------
    # Frozen artifact verification
    # -------------------------------------------------------------------------

    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Gold file not found:\n"
            f"{GOLD_FILE}"
        )

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PREDICTION_FILE}"
        )

    gold_hash = sha256_file(
        GOLD_FILE
    )

    prediction_hash = sha256_file(
        PREDICTION_FILE
    )

    if (
        gold_hash
        != EXPECTED_GOLD_SHA256
    ):
        raise RuntimeError(
            "Frozen gold SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_GOLD_SHA256}\n"
            f"Found:    {gold_hash}"
        )

    if (
        prediction_hash
        != EXPECTED_PREDICTION_SHA256
    ):
        raise RuntimeError(
            "Frozen prediction SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_PREDICTION_SHA256}\n"
            f"Found:    {prediction_hash}"
        )

    print(
        "\n[OK] frozen gold SHA-256 verified"
    )

    print(
        "[OK] frozen prediction SHA-256 verified"
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    gold = pd.read_csv(
        GOLD_FILE
    )

    pred = pd.read_csv(
        PREDICTION_FILE
    )

    gold_required = {
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "sample_block",
        "human_label",
    }

    pred_required = {
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "pred_weak_v3",
    }

    missing_gold = (
        gold_required
        - set(gold.columns)
    )

    missing_pred = (
        pred_required
        - set(pred.columns)
    )

    if missing_gold:
        raise RuntimeError(
            "Missing frozen-gold columns: "
            f"{sorted(missing_gold)}"
        )

    if missing_pred:
        raise RuntimeError(
            "Missing prediction columns: "
            f"{sorted(missing_pred)}"
        )

    if "human_label" in pred.columns:
        raise RuntimeError(
            "human_label unexpectedly present "
            "in the frozen prediction artifact."
        )

    # -------------------------------------------------------------------------
    # Structural validation
    # -------------------------------------------------------------------------

    if len(gold) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} gold rows, "
            f"found {len(gold)}."
        )

    if len(pred) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} prediction rows, "
            f"found {len(pred)}."
        )

    if (
        gold["article_id"].nunique()
        != EXPECTED_ARTICLES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_ARTICLES} "
            "unique gold articles."
        )

    if (
        pred["article_id"].nunique()
        != EXPECTED_ARTICLES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_ARTICLES} "
            "unique prediction articles."
        )

    if (
        gold["theme_id"].nunique()
        != EXPECTED_THEMES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_THEMES} "
            "themes in frozen gold."
        )

    if (
        pred["theme_id"].nunique()
        != EXPECTED_THEMES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_THEMES} "
            "themes in predictions."
        )

    if (
        gold["final_test_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate final_test_id "
            "in frozen gold."
        )

    if (
        pred["final_test_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate final_test_id "
            "in prediction artifact."
        )

    if gold[
        [
            "article_id",
            "theme_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate article-theme pairs "
            "in frozen gold."
        )

    if pred[
        [
            "article_id",
            "theme_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate article-theme pairs "
            "in prediction artifact."
        )

    actual_gold_themes = set(
        gold[
            "theme_name"
        ]
        .dropna()
        .astype(str)
    )

    actual_pred_themes = set(
        pred[
            "theme_name"
        ]
        .dropna()
        .astype(str)
    )

    if (
        actual_gold_themes
        != EXPECTED_THEME_NAMES
    ):
        raise RuntimeError(
            "Unexpected theme universe "
            "in frozen gold."
        )

    if (
        actual_pred_themes
        != EXPECTED_THEME_NAMES
    ):
        raise RuntimeError(
            "Unexpected theme universe "
            "in prediction artifact."
        )

    # -------------------------------------------------------------------------
    # Validate labels and predictions
    # -------------------------------------------------------------------------

    gold[
        "human_label"
    ] = pd.to_numeric(
        gold["human_label"],
        errors="raise",
    )

    pred[
        "pred_weak_v3"
    ] = pd.to_numeric(
        pred["pred_weak_v3"],
        errors="raise",
    )

    if (
        gold["human_label"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing human labels."
        )

    if (
        pred["pred_weak_v3"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing Weak V3 predictions."
        )

    if not set(
        gold[
            "human_label"
        ].unique()
    ).issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "Gold labels outside {0,1}."
        )

    if not set(
        pred[
            "pred_weak_v3"
        ].unique()
    ).issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "Weak V3 predictions outside {0,1}."
        )

    gold[
        "human_label"
    ] = (
        gold["human_label"]
        .astype(int)
    )

    pred[
        "pred_weak_v3"
    ] = (
        pred["pred_weak_v3"]
        .astype(int)
    )

    print(
        "[OK] structural integrity verified"
    )

    # -------------------------------------------------------------------------
    # Merge gold and frozen predictions
    # -------------------------------------------------------------------------

    prediction_columns = [
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "suggested_label",
        "rule_reason",
        "theme_evidence",
        "risk_evidence",
        "pred_A_strong_weak",
        "pred_proximity50",
        "pred_weak_v3",
    ]

    prediction_columns = [
        column
        for column in prediction_columns
        if column in pred.columns
    ]

    merged = gold.merge(
        pred[
            prediction_columns
        ],
        on="final_test_id",
        how="inner",
        suffixes=(
            "_gold",
            "_pred",
        ),
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_ROWS:
        raise RuntimeError(
            "Gold/prediction merge did not "
            f"produce {EXPECTED_ROWS} rows."
        )

    # Identity fields must agree independently of final_test_id.
    for column in [
        "article_id",
        "theme_id",
        "theme_name",
    ]:

        gold_column = (
            f"{column}_gold"
        )

        pred_column = (
            f"{column}_pred"
        )

        mismatch = (
            merged[
                gold_column
            ].astype(str)
            !=
            merged[
                pred_column
            ].astype(str)
        )

        if mismatch.any():
            raise RuntimeError(
                f"Identity mismatch for {column}: "
                f"{int(mismatch.sum())} rows."
            )

        merged[
            column
        ] = merged[
            gold_column
        ]

    print(
        "[OK] one-to-one gold/prediction "
        "identity verified"
    )

    # -------------------------------------------------------------------------
    # Overall evaluation
    # -------------------------------------------------------------------------

    overall = calculate_metrics(
        merged
    )

    overall_df = pd.DataFrame(
        [overall]
    )

    print()
    print("=" * 110)
    print("OVERALL")
    print("=" * 110)

    print(
        f"""
N:                    {overall['n']}
Gold positives:       {overall['gold_positive']}
Predicted positives:  {overall['predicted_positive']}

Confusion matrix:
  TN = {overall['tn']}
  FP = {overall['fp']}
  FN = {overall['fn']}
  TP = {overall['tp']}

Accuracy:             {overall['accuracy']:.6f}
Precision:            {overall['precision']:.6f}
Recall:               {overall['recall']:.6f}
Specificity:          {overall['specificity']:.6f}
F1:                   {overall['f1']:.6f}
Balanced Accuracy:    {overall['balanced_accuracy']:.6f}
NPV:                  {overall['npv']:.6f}
"""
    )

    # -------------------------------------------------------------------------
    # Grouped diagnostics
    # -------------------------------------------------------------------------

    theme_df = grouped_metrics(
        merged,
        [
            "theme_id",
            "theme_name",
        ],
    )

    block_df = grouped_metrics(
        merged,
        [
            "sample_block",
        ],
    )

    theme_block_df = grouped_metrics(
        merged,
        [
            "theme_id",
            "theme_name",
            "sample_block",
        ],
    )

    print("-" * 110)
    print("BY THEME")
    print("-" * 110)

    print(
        theme_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("-" * 110)
    print("BY SAMPLE BLOCK")
    print("-" * 110)

    print(
        block_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Confusion matrix artifact
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
    # Error analysis
    # -------------------------------------------------------------------------

    merged[
        "error_type"
    ] = np.select(
        [
            (
                (merged["human_label"] == 0)
                & (merged["pred_weak_v3"] == 1)
            ),
            (
                (merged["human_label"] == 1)
                & (merged["pred_weak_v3"] == 0)
            ),
        ],
        [
            "FP",
            "FN",
        ],
        default="CORRECT",
    )

    errors = merged[
        merged["error_type"]
        != "CORRECT"
    ].copy()

    print()
    print("-" * 110)
    print("ERRORS")
    print("-" * 110)

    print(
        f"Total errors:   "
        f"{len(errors)}"
    )

    print(
        f"False positives: "
        f"{int((errors['error_type'] == 'FP').sum())}"
    )

    print(
        f"False negatives: "
        f"{int((errors['error_type'] == 'FN').sum())}"
    )

    if not errors.empty:

        print(
            "\nErrors by theme:"
        )

        print(
            errors.groupby(
                [
                    "theme_name",
                    "error_type",
                ]
            )
            .size()
            .unstack(
                fill_value=0
            )
            .to_string()
        )

    # -------------------------------------------------------------------------
    # Protect final evaluation artifacts
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = [
        OVERALL_FILE,
        THEME_FILE,
        BLOCK_FILE,
        THEME_BLOCK_FILE,
        CONFUSION_FILE,
        ERRORS_FILE,
        MERGED_FILE,
        SUMMARY_FILE,
    ]

    existing_files = [
        path
        for path in output_files
        if path.exists()
    ]

    if existing_files:
        raise FileExistsError(
            "Final evaluation artifacts already "
            "exist; refusing silent overwrite:\n"
            + "\n".join(
                str(path)
                for path in existing_files
            )
        )

    # -------------------------------------------------------------------------
    # Save tabular artifacts
    # -------------------------------------------------------------------------

    overall_df.to_csv(
        OVERALL_FILE,
        index=False,
    )

    theme_df.to_csv(
        THEME_FILE,
        index=False,
    )

    block_df.to_csv(
        BLOCK_FILE,
        index=False,
    )

    theme_block_df.to_csv(
        THEME_BLOCK_FILE,
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

    merged.to_csv(
        MERGED_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Evaluation summary
    # -------------------------------------------------------------------------

    overall_json = {
        key: (
            None
            if pd.isna(value)
            else (
                int(value)
                if key in {
                    "n",
                    "gold_negative",
                    "gold_positive",
                    "predicted_negative",
                    "predicted_positive",
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                }
                else float(value)
            )
        )
        for key, value
        in overall.items()
    }

    summary = {
        "status":
            "FINAL_TEST_EVALUATION",

        "evaluated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "method":
            "weak_supervision",

        "model_version":
            MODEL_VERSION,

        "evaluation_set":
            "untouched_final_test",

        "n":
            int(
                len(merged)
            ),

        "gold_sha256":
            gold_hash,

        "expected_gold_sha256":
            EXPECTED_GOLD_SHA256,

        "prediction_sha256":
            prediction_hash,

        "expected_prediction_sha256":
            EXPECTED_PREDICTION_SHA256,

        "overall_metrics":
            overall_json,

        "methodological_lock": {
            "predictions_regenerated":
                False,

            "rules_modified":
                False,

            "signals_modified":
                False,

            "proximity_window_modified":
                False,

            "threshold_selected":
                False,

            "final_test_labels_used_only_for_evaluation":
                True,
        },

        "metric_scope": {
            "classification_metrics":
                (
                    "Computed from the frozen "
                    "binary Weak V3 predictions."
                ),

            "ranking_metrics":
                (
                    "Not reported because this "
                    "artifact contains binary "
                    "Weak V3 predictions rather "
                    "than the frozen continuous "
                    "Weak V3 relevance score."
                ),
        },

        "artifacts": {
            "overall":
                str(
                    OVERALL_FILE
                ),

            "by_theme":
                str(
                    THEME_FILE
                ),

            "by_block":
                str(
                    BLOCK_FILE
                ),

            "by_theme_block":
                str(
                    THEME_BLOCK_FILE
                ),

            "confusion_matrix":
                str(
                    CONFUSION_FILE
                ),

            "errors":
                str(
                    ERRORS_FILE
                ),

            "evaluated_predictions":
                str(
                    MERGED_FILE
                ),
        },
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

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    for path in output_files:

        print(
            f"\n{path}"
        )

        print(
            f"SHA-256: "
            f"{sha256_file(path)}"
        )

    print()
    print("=" * 110)
    print(
        "WEAK V3 FINAL EVALUATION COMPLETE"
    )
    print("=" * 110)

    print(
        "\nNo rules, signals, windows, "
        "thresholds, or model parameters "
        "were selected from final-test performance."
    )


if __name__ == "__main__":
    main()
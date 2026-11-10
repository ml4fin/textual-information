#!/usr/bin/env python3

"""
FINAL TEST — SUPERVISED V3 FROZEN SCORING
=========================================

Apply the frozen Supervised V3 model to the frozen final-test set.

Methodological constraints
--------------------------
- No training.
- No refitting.
- No hyperparameter tuning.
- No threshold tuning.
- No calibration.
- The classification threshold is frozen at 0.34.
- Exactly 500 frozen article-theme observations are scored.
- Human labels are never used as model inputs.
- Existing final-test prediction artifacts are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_FILE = Path(
    os.getenv(
        "SUPERVISED_V3_FROZEN_MODEL",
        "supervised_model_v3.joblib",
    )
)

GOLD_FILE = Path(
    os.getenv(
        "FINAL_TEST_GOLD_FILE",
        "final_test_gold_frozen.csv",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "FINAL_SUPERVISED_V3_OUTPUT_DIR",
        "final_supervised_v3",
    )
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "final_supervised_v3_predictions.csv"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "final_supervised_v3_predictions_manifest.json"
)

EXPECTED_GOLD_SHA256 = (
    "d4e65929a1831f7823bd92546b10064796379a55e3924fd642e5366bade19e21"
)

# Add the SHA-256 recorded when the final V3 model was frozen.
EXPECTED_MODEL_SHA256 = os.getenv(
    "SUPERVISED_V3_EXPECTED_MODEL_SHA256"
)

EXPECTED_ROWS = 500
EXPECTED_ARTICLES = 300
EXPECTED_THEMES = 5

SELECTED_THRESHOLD = 0.34

MODEL_NAME = "Supervised V3 Hybrid"
MODEL_VERSION = "supervised_v3_frozen"
MODEL_STATUS = "FROZEN"

SEPARATOR = "|||"

EXPECTED_THEME_LABELS = {
    "AI-Related Risk",
    "Climate / Environmental Risk",
    "Cybersecurity Risk",
    "Pandemic Risk",
    "Supply Chain Risk",
}


# =============================================================================
# CUSTOM OBJECTS REQUIRED BY JOBLIB
# =============================================================================

TOKEN_PATTERN = re.compile(
    r"(?u)\b\w\w+\b"
)


def clean_text(value) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def normalize_theme(
    theme: str,
) -> str:

    theme = clean_text(
        theme
    ).lower()

    theme = re.sub(
        r"[^a-z0-9]+",
        "_",
        theme,
    )

    return theme.strip("_")


def build_v3_document(
    theme_name: str,
    title: str,
    body: str,
) -> str:

    theme_key = normalize_theme(
        theme_name
    )

    return (
        f"{theme_key} {SEPARATOR} "
        f"{clean_text(title)} {SEPARATOR} "
        f"{clean_text(body)}"
    )


def parse_v3_document(
    document: str,
):

    parts = str(
        document
    ).split(
        SEPARATOR,
        maxsplit=2,
    )

    if len(parts) != 3:
        raise ValueError(
            "Invalid V3 document. Expected "
            "'theme ||| title ||| body'."
        )

    theme_key = (
        parts[0].strip()
    )

    title = (
        parts[1].strip()
    )

    body = (
        parts[2].strip()
    )

    return (
        theme_key,
        title,
        body,
    )


class SharedTextExtractor(
    BaseEstimator,
    TransformerMixin,
):

    def fit(
        self,
        X,
        y=None,
    ):
        return self

    def transform(
        self,
        X,
    ):

        output = []

        for document in X:

            _, title, body = (
                parse_v3_document(
                    document
                )
            )

            output.append(
                f"TITLE: {title} "
                f"BODY: {body}"
            )

        return output


def theme_interaction_analyzer(
    document: str,
):

    theme_key, title, body = (
        parse_v3_document(
            document
        )
    )

    text = (
        f"{clean_text(title)} "
        f"{clean_text(body)}"
    ).lower()

    tokens = TOKEN_PATTERN.findall(
        text
    )

    for token in tokens:

        yield (
            f"{theme_key}__{token}"
        )

    for i in range(
        len(tokens) - 1
    ):

        yield (
            f"{theme_key}__"
            f"{tokens[i]}_"
            f"{tokens[i + 1]}"
        )


# =============================================================================
# HASHING
# =============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "FINAL TEST — SUPERVISED V3 "
        "FROZEN SCORING"
    )
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Frozen artifact existence
    # -------------------------------------------------------------------------

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Frozen model not found:\n"
            f"{MODEL_FILE}"
        )

    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Frozen gold not found:\n"
            f"{GOLD_FILE}"
        )

    if PREDICTIONS_FILE.exists():
        raise FileExistsError(
            "Final-test prediction file already exists; "
            "refusing silent overwrite:\n"
            f"{PREDICTIONS_FILE}"
        )

    if MANIFEST_FILE.exists():
        raise FileExistsError(
            "Final-test manifest already exists; "
            "refusing silent overwrite:\n"
            f"{MANIFEST_FILE}"
        )

    print(
        f"\nFrozen model:\n"
        f"{MODEL_FILE}"
    )

    print(
        f"\nFrozen gold:\n"
        f"{GOLD_FILE}"
    )

    # -------------------------------------------------------------------------
    # Verify frozen artifacts
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("1. VERIFY FROZEN ARTIFACTS")
    print("=" * 100)

    actual_gold_hash = (
        sha256_file(
            GOLD_FILE
        )
    )

    if (
        actual_gold_hash
        != EXPECTED_GOLD_SHA256
    ):
        raise RuntimeError(
            "Frozen gold SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_GOLD_SHA256}\n"
            f"Found:    {actual_gold_hash}"
        )

    print(
        "\n[OK] frozen gold SHA-256 verified"
    )

    model_hash = sha256_file(
        MODEL_FILE
    )

    print(
        f"[OK] frozen model SHA-256: "
        f"{model_hash}"
    )

    if EXPECTED_MODEL_SHA256:

        if (
            model_hash
            != EXPECTED_MODEL_SHA256
        ):
            raise RuntimeError(
                "Frozen model SHA-256 mismatch.\n"
                f"Expected: {EXPECTED_MODEL_SHA256}\n"
                f"Found:    {model_hash}"
            )

        print(
            "[OK] frozen model SHA-256 "
            "matches the expected fingerprint"
        )

    else:

        print(
            "[WARNING] no expected frozen model "
            "SHA-256 was configured."
        )

    # -------------------------------------------------------------------------
    # Load frozen gold
    # -------------------------------------------------------------------------

    gold = pd.read_csv(
        GOLD_FILE
    )

    required_columns = {
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "publication_date",
        "title",
        "body",
        "sample_block",
        "human_label",
    }

    missing_columns = (
        required_columns
        - set(gold.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # -------------------------------------------------------------------------
    # Structural checks
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("2. STRUCTURAL CHECKS")
    print("=" * 100)

    if len(gold) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(gold)}."
        )

    identifier_columns = [
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "sample_block",
    ]

    if gold[
        identifier_columns
    ].isna().any().any():

        raise RuntimeError(
            "Missing final-test identifiers "
            "or theme information."
        )

    n_articles = int(
        gold[
            "article_id"
        ].nunique()
    )

    if (
        n_articles
        != EXPECTED_ARTICLES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_ARTICLES} "
            f"unique articles, "
            f"found {n_articles}."
        )

    n_themes = int(
        gold[
            "theme_id"
        ].nunique()
    )

    if (
        n_themes
        != EXPECTED_THEMES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_THEMES} themes, "
            f"found {n_themes}."
        )

    if (
        gold[
            "final_test_id"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate final_test_id detected."
        )

    if gold[
        [
            "article_id",
            "theme_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate article-theme pairs detected."
        )

    actual_theme_labels = set(
        gold[
            "theme_name"
        ].astype(str)
    )

    if (
        actual_theme_labels
        != EXPECTED_THEME_LABELS
    ):
        raise RuntimeError(
            "Theme labels do not match "
            "the frozen V3 theme universe.\n"
            f"Expected: "
            f"{sorted(EXPECTED_THEME_LABELS)}\n"
            f"Found:    "
            f"{sorted(actual_theme_labels)}"
        )

    print(
        f"\n[OK] rows = {len(gold):,}"
    )

    print(
        f"[OK] unique articles = "
        f"{n_articles:,}"
    )

    print(
        f"[OK] themes = "
        f"{n_themes}"
    )

    print(
        "[OK] identifiers and "
        "article-theme pairs are unique"
    )

    print(
        "[OK] theme labels match "
        "the frozen V3 model"
    )

    # -------------------------------------------------------------------------
    # Gold-label integrity
    #
    # Labels are validated for integrity but are not passed to the model.
    # -------------------------------------------------------------------------

    labels = pd.to_numeric(
        gold[
            "human_label"
        ],
        errors="raise",
    )

    if labels.isna().any():
        raise RuntimeError(
            "Missing human labels detected."
        )

    if not set(
        labels.unique()
    ).issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "human_label contains values "
            "outside {0,1}."
        )

    gold[
        "human_label"
    ] = labels.astype(
        int
    )

    print(
        "[OK] human labels are complete "
        "and binary"
    )

    # -------------------------------------------------------------------------
    # Load frozen model
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("3. LOAD FROZEN SUPERVISED V3")
    print("=" * 100)

    print(
        f"\nFrozen threshold: "
        f"{SELECTED_THRESHOLD:.2f}"
    )

    model = joblib.load(
        MODEL_FILE
    )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise RuntimeError(
            "Loaded frozen model does not "
            "expose predict_proba()."
        )

    print(
        "[OK] frozen model loaded"
    )

    # -------------------------------------------------------------------------
    # Blind scoring frame
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("4. BUILD FROZEN V3 INPUT")
    print("=" * 100)

    scoring_df = gold[
        [
            "final_test_id",
            "article_id",
            "theme_id",
            "theme_name",
            "title",
            "body",
        ]
    ].copy()

    if (
        "human_label"
        in scoring_df.columns
    ):
        raise RuntimeError(
            "human_label entered the model "
            "scoring dataframe."
        )

    documents = [
        build_v3_document(
            theme_name=row.theme_name,
            title=row.title,
            body=row.body,
        )
        for row in scoring_df.itertuples(
            index=False
        )
    ]

    if (
        len(documents)
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            "Incorrect number of V3 documents."
        )

    print(
        f"\n[OK] V3 documents built: "
        f"{len(documents):,}"
    )

    print(
        "[OK] human_label excluded "
        "from model input"
    )

    # -------------------------------------------------------------------------
    # Frozen inference
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("5. FROZEN MODEL INFERENCE")
    print("=" * 100)

    probabilities = (
        model.predict_proba(
            documents
        )[:, 1]
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if (
        len(probabilities)
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            "Incorrect number of model scores."
        )

    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            "Non-finite model scores detected."
        )

    if (
        (probabilities < 0.0).any()
        or
        (probabilities > 1.0).any()
    ):
        raise RuntimeError(
            "Model scores outside [0,1]."
        )

    predictions = (
        probabilities
        >= SELECTED_THRESHOLD
    ).astype(
        int
    )

    n_positive = int(
        predictions.sum()
    )

    print(
        f"\nScores generated: "
        f"{len(probabilities):,}"
    )

    print(
        f"Score min:    "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Score mean:   "
        f"{probabilities.mean():.6f}"
    )

    print(
        f"Score median: "
        f"{np.median(probabilities):.6f}"
    )

    print(
        f"Score max:    "
        f"{probabilities.max():.6f}"
    )

    print(
        f"\nPredicted positive @ "
        f"{SELECTED_THRESHOLD:.2f}: "
        f"{n_positive:,}/"
        f"{EXPECTED_ROWS:,} "
        f"({n_positive / EXPECTED_ROWS:.2%})"
    )

    # -------------------------------------------------------------------------
    # Prediction artifact
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("6. BUILD PREDICTION ARTIFACT")
    print("=" * 100)

    output = gold[
        [
            "final_test_id",
            "article_id",
            "theme_id",
            "theme_name",
            "publication_date",
            "sample_block",
            "human_label",
        ]
    ].copy()

    output[
        "supervised_probability"
    ] = probabilities

    output[
        "supervised_prediction"
    ] = predictions

    output[
        "threshold"
    ] = SELECTED_THRESHOLD

    output[
        "model_name"
    ] = MODEL_NAME

    output[
        "model_version"
    ] = MODEL_VERSION

    if len(
        output
    ) != EXPECTED_ROWS:

        raise RuntimeError(
            "Prediction artifact row-count mismatch."
        )

    if (
        output[
            "final_test_id"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate final_test_id "
            "in prediction artifact."
        )

    if output[
        [
            "article_id",
            "theme_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate article-theme pair "
            "in prediction artifact."
        )

    if output[
        "supervised_probability"
    ].isna().any():

        raise RuntimeError(
            "Missing supervised model scores."
        )

    if not set(
        output[
            "supervised_prediction"
        ].unique()
    ).issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "Non-binary supervised predictions."
        )

    print(
        "\n[OK] prediction artifact "
        "integrity verified"
    )

    # -------------------------------------------------------------------------
    # Save and freeze
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("7. SAVE AND FREEZE PREDICTIONS")
    print("=" * 100)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    predictions_hash = sha256_file(
        PREDICTIONS_FILE
    )

    manifest = {
        "status":
            "FROZEN",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "purpose":
            (
                "Final untouched test inference "
                "for frozen Supervised V3"
            ),

        "model": {
            "name":
                MODEL_NAME,

            "version":
                MODEL_VERSION,

            "status":
                MODEL_STATUS,

            "path":
                str(
                    MODEL_FILE
                ),

            "sha256":
                model_hash,

            "expected_sha256":
                EXPECTED_MODEL_SHA256,
        },

        "gold": {
            "path":
                str(
                    GOLD_FILE
                ),

            "sha256":
                actual_gold_hash,

            "expected_sha256":
                EXPECTED_GOLD_SHA256,

            "rows":
                int(
                    len(gold)
                ),

            "unique_articles":
                n_articles,

            "themes":
                n_themes,
        },

        "inference": {
            "threshold":
                SELECTED_THRESHOLD,

            "calibration":
                "none",

            "score_type":
                "uncalibrated_predict_proba",

            "training_performed":
                False,

            "refit_performed":
                False,

            "hyperparameter_tuning_performed":
                False,

            "threshold_tuning_performed":
                False,

            "calibration_performed":
                False,

            "human_label_used_as_model_input":
                False,
        },

        "predictions": {
            "path":
                str(
                    PREDICTIONS_FILE
                ),

            "sha256":
                predictions_hash,

            "rows":
                int(
                    len(output)
                ),

            "predicted_negative":
                int(
                    (
                        predictions == 0
                    ).sum()
                ),

            "predicted_positive":
                int(
                    (
                        predictions == 1
                    ).sum()
                ),

            "positive_rate":
                float(
                    predictions.mean()
                ),

            "score_min":
                float(
                    probabilities.min()
                ),

            "score_mean":
                float(
                    probabilities.mean()
                ),

            "score_median":
                float(
                    np.median(
                        probabilities
                    )
                ),

            "score_max":
                float(
                    probabilities.max()
                ),
        },

        "lock": {
            "predictions_frozen":
                True,

            "model_must_not_be_changed":
                True,

            "threshold_must_not_be_changed":
                True,

            "gold_must_not_be_changed":
                True,

            "no_final_test_tuning":
                True,
        },
    }

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    manifest_hash = sha256_file(
        MANIFEST_FILE
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print(
        f"\nPredictions written to:\n"
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"\nPrediction SHA-256:\n"
        f"{predictions_hash}"
    )

    print(
        f"\nManifest written to:\n"
        f"{MANIFEST_FILE}"
    )

    print(
        f"\nManifest SHA-256:\n"
        f"{manifest_hash}"
    )

    print()
    print("=" * 100)
    print(
        "SUPERVISED V3 FINAL-TEST "
        "PREDICTIONS SUCCESSFULLY FROZEN"
    )
    print("=" * 100)

    print(
        f"\nRows scored:          "
        f"{len(output):,}"
    )

    print(
        f"Unique articles:      "
        f"{output['article_id'].nunique():,}"
    )

    print(
        f"Threshold:            "
        f"{SELECTED_THRESHOLD:.2f}"
    )

    print(
        f"Predicted negatives:  "
        f"{int((predictions == 0).sum()):,}"
    )

    print(
        f"Predicted positives:  "
        f"{int((predictions == 1).sum()):,}"
    )

    print(
        f"Positive rate:        "
        f"{predictions.mean():.2%}"
    )

    print(
        "\nNo training, refitting, "
        "hyperparameter tuning, calibration, "
        "or threshold tuning was performed."
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
#!/usr/bin/env python3

"""
FINAL TEST — WEAK SUPERVISION V3.0 FROZEN SCORING
==================================================

Apply the frozen Weak Supervision V3.0 rules to the 500 frozen final-test
article-theme observations.

CRITICAL METHODOLOGICAL RULE
----------------------------
Human labels are never used for prediction.

Pipeline:
    final-test text
        -> frozen baseline logic from build_gold_review
        -> A strong/weak refinement
        -> proximity rescue (window = 50)
        -> AI strict rule
        -> frozen binary prediction

No evaluation is performed here.
No rule is tuned here.
No threshold is selected here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re

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
# CONFIGURATION
# =============================================================================

INPUT_FILE = Path(
    os.getenv(
        "FINAL_TEST_GOLD_FILE",
        "final_test_gold_frozen.csv",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "FINAL_WEAK_V3_OUTPUT_DIR",
        "final_weak_v3",
    )
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "final_weak_v3_predictions.csv"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "final_weak_v3_predictions_manifest.json"
)

EXPECTED_ROWS = 500

EXPECTED_VERSION = (
    "Weak Supervision V3.0 - "
    "Frozen Development Version"
)

EXPECTED_WINDOW = 50

EXPECTED_THEMES = {
    "AI-Related Risk",
    "Climate / Environmental Risk",
    "Cybersecurity Risk",
    "Pandemic Risk",
    "Supply Chain Risk",
}


# =============================================================================
# HASHING
# =============================================================================

def sha256_file(path: Path) -> str:

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


def canonicalize_for_hash(value):

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                canonicalize_for_hash(
                    item
                )
            for key, item in sorted(
                value.items(),
                key=lambda x: str(x[0]),
            )
        }

    if isinstance(
        value,
        (set, frozenset),
    ):
        return sorted(
            canonicalize_for_hash(
                item
            )
            for item in value
        )

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            canonicalize_for_hash(
                item
            )
            for item in value
        ]

    return value


def frozen_config_snapshot() -> dict:

    return {
        "version":
            VERSION,

        "proximity_window":
            PROXIMITY_WINDOW,

        "strong_signals":
            STRONG_SIGNALS,

        "ambiguous_signals":
            AMBIGUOUS_SIGNALS,

        "climate_theme_signals":
            CLIMATE_THEME_SIGNALS,

        "cyber_object_signals":
            CYBER_OBJECT_SIGNALS,

        "cyber_context_signals":
            CYBER_CONTEXT_SIGNALS,

        "supply_object_signals":
            SUPPLY_OBJECT_SIGNALS,

        "supply_disruption_signals":
            SUPPLY_DISRUPTION_SIGNALS,

        "ai_strict_signals":
            AI_STRICT_SIGNALS,
    }


def frozen_config_sha256() -> str:

    canonical = canonicalize_for_hash(
        frozen_config_snapshot()
    )

    payload = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def normalize(value) -> str:

    if pd.isna(value):
        return ""

    return str(
        value
    ).strip().lower()


def has_risk_evidence(value) -> bool:

    if pd.isna(value):
        return False

    value = str(
        value
    ).strip().lower()

    return value not in {
        "",
        "nan",
        "none",
    }


def tokenize(text) -> list[str]:

    if pd.isna(text):
        return []

    return re.findall(
        r"[a-z0-9\-]+",
        str(text).lower(),
    )


def phrase_positions(
    tokens,
    phrase,
):

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


def minimum_distance(
    tokens,
    phrase1,
    phrase2,
):

    positions1 = phrase_positions(
        tokens,
        phrase1,
    )

    positions2 = phrase_positions(
        tokens,
        phrase2,
    )

    if (
        not positions1
        or not positions2
    ):
        return None

    return min(
        abs(a - b)
        for a in positions1
        for b in positions2
    )


def within_window(
    tokens,
    signals1,
    signals2,
    window,
) -> bool:

    for signal1 in signals1:

        for signal2 in signals2:

            distance = minimum_distance(
                tokens,
                signal1,
                signal2,
            )

            if (
                distance is not None
                and distance <= window
            ):
                return True

    return False


def ai_strict_match(
    tokens,
    theme_phrase,
    window,
) -> bool:

    if not theme_phrase:
        return False

    for signal in AI_STRICT_SIGNALS:

        distance = minimum_distance(
            tokens,
            theme_phrase,
            signal,
        )

        if (
            distance is not None
            and distance <= window
        ):
            return True

    return False


def parse_risk_evidence(value):

    if pd.isna(value):
        return []

    value = str(
        value
    ).lower().strip()

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
# FROZEN BASELINE
# =============================================================================

def get_baseline_classifier():

    classifier = getattr(
        gold_builder,
        "score_row",
        None,
    )

    if not callable(
        classifier
    ):
        raise RuntimeError(
            "Frozen baseline function "
            "build_gold_review.score_row "
            "was not found."
        )

    print(
        "[OK] Frozen baseline classifier: "
        "build_gold_review.score_row"
    )

    return classifier


def normalize_baseline_result(
    result,
):

    required = {
        "suggested_label",
        "rule_reason",
        "theme_evidence",
        "risk_evidence",
    }

    if isinstance(
        result,
        pd.Series,
    ):

        missing = (
            required
            - set(result.index)
        )

        if missing:
            raise RuntimeError(
                "score_row returned a pandas Series "
                "without required fields: "
                f"{sorted(missing)}"
            )

        values = result

    elif isinstance(
        result,
        dict,
    ):

        missing = (
            required
            - set(result)
        )

        if missing:
            raise RuntimeError(
                "score_row returned a dictionary "
                "without required fields: "
                f"{sorted(missing)}"
            )

        values = result

    elif isinstance(
        result,
        (tuple, list),
    ):

        if len(result) < 4:
            raise RuntimeError(
                "score_row returned fewer than "
                "four values."
            )

        return {
            "suggested_label":
                int(result[0]),

            "rule_reason":
                result[1],

            "theme_evidence":
                result[2],

            "risk_evidence":
                result[3],
        }

    else:
        raise RuntimeError(
            "Unsupported return type from "
            "build_gold_review.score_row: "
            f"{type(result)}"
        )

    return {
        "suggested_label":
            int(
                values[
                    "suggested_label"
                ]
            ),

        "rule_reason":
            values[
                "rule_reason"
            ],

        "theme_evidence":
            values[
                "theme_evidence"
            ],

        "risk_evidence":
            values[
                "risk_evidence"
            ],
    }


def call_baseline_classifier(
    classifier,
    row,
):

    """
    Call the frozen baseline using its frozen row-based API.

    If the historical score_row API differs from this signature,
    change this adapter only after verifying the original
    build_gold_review implementation. Do not guess alternative
    argument orders during final-test scoring.
    """

    result = classifier(
        row
    )

    return normalize_baseline_result(
        result
    )


# =============================================================================
# V3 FROZEN REFINEMENT
# =============================================================================

def apply_v3_refinement(
    row,
):

    baseline = int(
        row["suggested_label"]
    )

    rule_reason = str(
        row["rule_reason"]
    )

    theme = str(
        row["theme_name"]
    )

    theme_signal = normalize(
        row["theme_evidence"]
    )

    tokens = row["_tokens"]

    # A strong/weak refinement.
    pred_a = baseline

    if (
        rule_reason
        == "single_theme_signal"
    ):

        strong = (
            STRONG_SIGNALS.get(
                theme,
                set(),
            )
        )

        ambiguous = (
            AMBIGUOUS_SIGNALS.get(
                theme,
                set(),
            )
        )

        if theme_signal in strong:
            pred_a = 1

        elif theme_signal in ambiguous:

            pred_a = int(
                has_risk_evidence(
                    row[
                        "risk_evidence"
                    ]
                )
            )

    # Proximity rescue.
    pred_proximity = pred_a

    if (
        rule_reason
        == "no_theme_signal"
    ):

        activated = False

        if (
            theme
            == "Climate / Environmental Risk"
        ):

            risk_terms = (
                parse_risk_evidence(
                    row[
                        "risk_evidence"
                    ]
                )
            )

            if risk_terms:

                activated = within_window(
                    tokens,
                    CLIMATE_THEME_SIGNALS,
                    risk_terms,
                    PROXIMITY_WINDOW,
                )

        elif (
            theme
            == "Cybersecurity Risk"
        ):

            activated = within_window(
                tokens,
                CYBER_OBJECT_SIGNALS,
                CYBER_CONTEXT_SIGNALS,
                PROXIMITY_WINDOW,
            )

        elif (
            theme
            == "Supply Chain Risk"
        ):

            activated = within_window(
                tokens,
                SUPPLY_OBJECT_SIGNALS,
                SUPPLY_DISRUPTION_SIGNALS,
                PROXIMITY_WINDOW,
            )

        if activated:
            pred_proximity = 1

    # AI strict rule.
    pred_final = pred_proximity

    if (
        theme
        == "AI-Related Risk"
    ):

        activated = ai_strict_match(
            tokens,
            theme_signal,
            PROXIMITY_WINDOW,
        )

        if activated:
            pred_final = 1

    return (
        int(pred_a),
        int(pred_proximity),
        int(pred_final),
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def load_final_test() -> tuple[pd.DataFrame, str]:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Frozen final-test file not found:\n"
            f"{INPUT_FILE}"
        )

    input_hash = sha256_file(
        INPUT_FILE
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = {
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "title",
        "body",
        "sample_block",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Frozen final-test file is missing "
            "required columns: "
            f"{sorted(missing)}"
        )

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(df)}."
        )

    if df[
        [
            "final_test_id",
            "article_id",
            "theme_id",
            "theme_name",
        ]
    ].isna().any().any():

        raise RuntimeError(
            "Missing final-test identifiers "
            "or theme information."
        )

    if (
        df["final_test_id"].nunique()
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            "final_test_id is not unique."
        )

    if df[
        [
            "article_id",
            "theme_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate article-theme pairs "
            "in the frozen final test."
        )

    actual_themes = set(
        df[
            "theme_name"
        ].astype(str)
    )

    if (
        actual_themes
        != EXPECTED_THEMES
    ):
        raise RuntimeError(
            "Final-test themes do not match "
            "the frozen V3 theme universe.\n"
            f"Expected: {sorted(EXPECTED_THEMES)}\n"
            f"Found:    {sorted(actual_themes)}"
        )

    return (
        df,
        input_hash,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "FINAL TEST — WEAK SUPERVISION V3.0 "
        "FROZEN SCORING"
    )
    print("=" * 110)

    # Frozen configuration lock.
    if VERSION != EXPECTED_VERSION:
        raise RuntimeError(
            "Unexpected frozen V3 version.\n"
            f"Expected: {EXPECTED_VERSION}\n"
            f"Found:    {VERSION}"
        )

    if (
        PROXIMITY_WINDOW
        != EXPECTED_WINDOW
    ):
        raise RuntimeError(
            "Unexpected frozen proximity window.\n"
            f"Expected: {EXPECTED_WINDOW}\n"
            f"Found:    {PROXIMITY_WINDOW}"
        )

    config_hash = (
        frozen_config_sha256()
    )

    print(
        f"\n[OK] version = {VERSION}"
    )

    print(
        "[OK] proximity window = "
        f"{PROXIMITY_WINDOW}"
    )

    print(
        "[OK] frozen configuration SHA-256 = "
        f"{config_hash}"
    )

    # Refuse to overwrite final-test artifacts.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            "Final-test prediction file already exists; "
            "refusing silent overwrite:\n"
            f"{OUTPUT_FILE}"
        )

    if MANIFEST_FILE.exists():
        raise FileExistsError(
            "Final-test manifest already exists; "
            "refusing silent overwrite:\n"
            f"{MANIFEST_FILE}"
        )

    df, input_hash = (
        load_final_test()
    )

    print(
        f"\nRows loaded: "
        f"{len(df):,}"
    )

    print(
        "[OK] final-test structure verified"
    )

    # Build a blind scoring dataframe containing no gold label.
    scoring_columns = [
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
        "title",
        "body",
        "sample_block",
    ]

    if (
        "publication_date"
        in df.columns
    ):
        scoring_columns.insert(
            4,
            "publication_date",
        )

    score_df = (
        df[
            scoring_columns
        ]
        .copy()
    )

    if (
        "human_label"
        in score_df.columns
    ):
        raise RuntimeError(
            "human_label entered the scoring dataframe."
        )

    print(
        "[OK] human_label excluded from "
        "the scoring dataframe"
    )

    classifier = (
        get_baseline_classifier()
    )

    print(
        "\nReconstructing frozen baseline..."
    )

    baseline_rows = []

    for (
        position,
        (_, row),
    ) in enumerate(
        score_df.iterrows(),
        start=1,
    ):

        baseline_rows.append(
            call_baseline_classifier(
                classifier,
                row,
            )
        )

        if position % 100 == 0:
            print(
                f"  {position}/{EXPECTED_ROWS}"
            )

    baseline_df = pd.DataFrame(
        baseline_rows
    )

    score_df = pd.concat(
        [
            score_df.reset_index(
                drop=True
            ),
            baseline_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    if (
        score_df[
            "suggested_label"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing baseline suggested_label."
        )

    baseline_labels = set(
        score_df[
            "suggested_label"
        ]
        .astype(int)
        .unique()
    )

    if not baseline_labels.issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "Baseline labels outside {0,1}: "
            f"{baseline_labels}"
        )

    print(
        "\nBaseline suggested positives: "
        f"{int(score_df['suggested_label'].sum())}"
    )

    print(
        "\nBaseline rule reasons:"
    )

    print(
        score_df[
            "rule_reason"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # Frozen text representation.
    score_df["_text"] = (
        score_df[
            "title"
        ]
        .fillna("")
        .astype(str)
        + " "
        + score_df[
            "body"
        ]
        .fillna("")
        .astype(str)
    )

    score_df["_tokens"] = (
        score_df[
            "_text"
        ].apply(
            tokenize
        )
    )

    print(
        "\nApplying frozen V3 refinement..."
    )

    v3_results = score_df.apply(
        apply_v3_refinement,
        axis=1,
        result_type="expand",
    )

    v3_results.columns = [
        "pred_A_strong_weak",
        "pred_proximity50",
        "pred_weak_v3",
    ]

    score_df = pd.concat(
        [
            score_df,
            v3_results,
        ],
        axis=1,
    )

    prediction_columns = [
        "pred_A_strong_weak",
        "pred_proximity50",
        "pred_weak_v3",
    ]

    for column in prediction_columns:

        if score_df[
            column
        ].isna().any():

            raise RuntimeError(
                f"{column} contains missing values."
            )

        values = set(
            score_df[
                column
            ]
            .astype(int)
            .unique()
        )

        if not values.issubset(
            {0, 1}
        ):
            raise RuntimeError(
                f"{column} contains "
                f"non-binary values: {values}"
            )

    print(
        "\nSuggested baseline positives: "
        f"{int(score_df['suggested_label'].sum())}"
    )

    print(
        "A strong/weak positives:       "
        f"{int(score_df['pred_A_strong_weak'].sum())}"
    )

    print(
        "Proximity-50 positives:        "
        f"{int(score_df['pred_proximity50'].sum())}"
    )

    print(
        "Final Weak V3 positives:       "
        f"{int(score_df['pred_weak_v3'].sum())}"
    )

    output_columns = [
        "final_test_id",
        "article_id",
        "theme_id",
        "theme_name",
    ]

    if (
        "publication_date"
        in score_df.columns
    ):
        output_columns.append(
            "publication_date"
        )

    output_columns += [
        "sample_block",
        "suggested_label",
        "rule_reason",
        "theme_evidence",
        "risk_evidence",
        "pred_A_strong_weak",
        "pred_proximity50",
        "pred_weak_v3",
    ]

    output_df = (
        score_df[
            output_columns
        ]
        .copy()
    )

    if (
        "human_label"
        in output_df.columns
    ):
        raise RuntimeError(
            "human_label found in prediction output."
        )

    if len(
        output_df
    ) != EXPECTED_ROWS:
        raise RuntimeError(
            "Prediction output row count changed."
        )

    if (
        output_df[
            "final_test_id"
        ].nunique()
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            "Prediction output does not contain "
            "exactly one row per final_test_id."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    output_hash = sha256_file(
        OUTPUT_FILE
    )

    manifest = {
        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "method":
            "weak_supervision",

        "model_version":
            VERSION,

        "status":
            "FROZEN_FINAL_TEST_PREDICTIONS",

        "n_predictions":
            int(
                len(output_df)
            ),

        "n_unique_articles":
            int(
                output_df[
                    "article_id"
                ].nunique()
            ),

        "n_themes":
            int(
                output_df[
                    "theme_id"
                ].nunique()
            ),

        "predicted_positive":
            int(
                output_df[
                    "pred_weak_v3"
                ].sum()
            ),

        "frozen_configuration": {
            "version":
                VERSION,

            "proximity_window":
                PROXIMITY_WINDOW,

            "configuration_sha256":
                config_hash,

            "strong_signals":
                True,

            "ambiguous_signals":
                True,

            "proximity_rescue":
                True,

            "ai_strict":
                True,
        },

        "prediction_generation": {
            "uses_human_label":
                False,

            "evaluation_performed":
                False,

            "threshold_tuning":
                False,

            "rule_tuning":
                False,
        },

        "input_file":
            str(INPUT_FILE),

        "input_sha256":
            input_hash,

        "prediction_file":
            str(OUTPUT_FILE),

        "prediction_sha256":
            output_hash,
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

    print()
    print("=" * 110)
    print(
        "WEAK V3 FINAL-TEST SCORING COMPLETE"
    )
    print("=" * 110)

    print(
        f"\nPrediction file:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nPrediction SHA-256:\n"
        f"{output_hash}"
    )

    print(
        f"\nManifest:\n"
        f"{MANIFEST_FILE}"
    )

    print(
        f"\nManifest SHA-256:\n"
        f"{manifest_hash}"
    )

    print(
        "\nNo human labels were used "
        "during prediction."
    )

    print(
        "No final-test metrics were calculated."
    )

    print(
        "No rules or thresholds were modified."
    )


if __name__ == "__main__":
    main()
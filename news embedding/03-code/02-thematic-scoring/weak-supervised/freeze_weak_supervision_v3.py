#!/usr/bin/env python3

"""
Freeze the Weak Supervision V3 development configuration to JSON.

The generated snapshot records:
- the frozen Weak V3 version;
- the development-set role and size;
- development-set performance;
- the frozen proximity window;
- all lexical and contextual signal sets;
- the methodological restriction against further tuning on the same
  development observations.

The JSON file is intended as a reproducibility record. It does not modify
the operational Weak V3 configuration.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

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

OUTPUT_FILE = Path(
    os.getenv(
        "WEAK_V3_FROZEN_SNAPSHOT",
        "weak_supervision_v3_frozen.json",
    )
)

EXPECTED_DEVELOPMENT_N = 485

DEVELOPMENT_PERFORMANCE = {
    "TN": 314,
    "FP": 23,
    "FN": 42,
    "TP": 106,
    "accuracy": 0.865979,
    "precision": 0.821705,
    "recall": 0.716216,
    "specificity": 0.931751,
    "f1": 0.765343,
    "balanced_accuracy": 0.823983,
}


# =============================================================================
# HELPERS
# =============================================================================

def to_json_serializable(obj):
    """Recursively convert sets into deterministically ordered lists."""

    if isinstance(obj, dict):
        return {
            key: to_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, set):
        return sorted(
            to_json_serializable(value)
            for value in obj
        )

    if isinstance(obj, (list, tuple)):
        return [
            to_json_serializable(value)
            for value in obj
        ]

    return obj


def validate_development_record():
    """Check internal consistency of the frozen development record."""

    confusion_total = (
        DEVELOPMENT_PERFORMANCE["TN"]
        + DEVELOPMENT_PERFORMANCE["FP"]
        + DEVELOPMENT_PERFORMANCE["FN"]
        + DEVELOPMENT_PERFORMANCE["TP"]
    )

    if confusion_total != EXPECTED_DEVELOPMENT_N:
        raise RuntimeError(
            "Development confusion-matrix total does not match "
            f"the expected sample size: "
            f"{confusion_total} != {EXPECTED_DEVELOPMENT_N}."
        )


# =============================================================================
# SNAPSHOT
# =============================================================================

def build_snapshot():
    """Build the frozen Weak V3 reproducibility snapshot."""

    return {
        "version":
            VERSION,

        "status":
            "FROZEN",

        "frozen_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "development_set": {
            "n":
                EXPECTED_DEVELOPMENT_N,

            "role":
                (
                    "Development and rule-tuning set; "
                    "not an untouched final test set."
                ),
        },

        "development_performance":
            DEVELOPMENT_PERFORMANCE,

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

        "methodological_note":
            (
                "No further rule, keyword, or window tuning should be "
                "performed using the same 485 manually reviewed "
                "development observations. Subsequent performance "
                "claims should be evaluated on unseen manually "
                "annotated data."
            ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Validate, serialize, and save the frozen Weak V3 snapshot."""

    validate_development_record()

    snapshot = to_json_serializable(
        build_snapshot()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            snapshot,
            file,
            indent=4,
            ensure_ascii=False,
        )

        file.write("\n")

    metrics = DEVELOPMENT_PERFORMANCE

    print("=" * 100)
    print("WEAK SUPERVISION V3 FREEZE")
    print("=" * 100)

    print()
    print(f"Version: {VERSION}")
    print("Status: FROZEN")

    print()
    print("Development performance:")
    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Precision:         "
        f"{metrics['precision']:.4f}"
    )
    print(
        f"Recall:            "
        f"{metrics['recall']:.4f}"
    )
    print(
        f"F1:                "
        f"{metrics['f1']:.4f}"
    )
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print()
    print(
        "IMPORTANT: Do not tune this configuration further "
        "using the 485 development observations."
    )

    print()
    print(
        f"File saved:\n{OUTPUT_FILE}"
    )

    print()
    print("Freeze completed.")


if __name__ == "__main__":
    main()
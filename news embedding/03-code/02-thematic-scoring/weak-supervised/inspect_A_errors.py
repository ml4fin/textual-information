#!/usr/bin/env python3

"""
Detailed error analysis for the A_strong_weak scenario.

The script examines classification changes produced by the strong/weak
theme-signal configuration and separates them into:

- FIXED_ERROR: previously incorrect classifications that become correct;
- NEW_ERROR: previously correct classifications that become incorrect.

Changes are summarized by theme and theme signal. For each signal, the script
also computes:

    net_gain = fixed_errors - new_errors

Positive values indicate that the signal fixes more reviewed errors than it
introduces within the evaluated sample.

The analysis is descriptive and is based on the supplied scenario-comparison
dataset.
"""

import os

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = os.getenv(
    "STRONG_WEAK_CHANGES_FILE",
    "strong_weak_signal_changes.csv",
)

OUT_NEW = os.getenv(
    "STRONG_WEAK_NEW_ERRORS_OUTPUT",
    "A_new_errors.csv",
)

OUT_FIXED = os.getenv(
    "STRONG_WEAK_FIXED_ERRORS_OUTPUT",
    "A_fixed_errors.csv",
)

SCENARIO_NAME = "A_strong_weak"


# =============================================================================
# DATA LOADING
# =============================================================================

df = pd.read_csv(
    INPUT_FILE
)

scenario_df = df[
    df["scenario"] == SCENARIO_NAME
].copy()

new_errors = scenario_df[
    scenario_df["effect"] == "NEW_ERROR"
].copy()

fixed_errors = scenario_df[
    scenario_df["effect"] == "FIXED_ERROR"
].copy()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 110)
print("A_STRONG_WEAK - DETAILED ANALYSIS")
print("=" * 110)

print(
    f"\nTotal changes: {len(scenario_df):,}"
)

print(
    f"Fixed errors:  {len(fixed_errors):,}"
)

print(
    f"New errors:    {len(new_errors):,}"
)


# =============================================================================
# NEW ERRORS BY THEME
# =============================================================================

print("\n" + "=" * 110)
print("NEW ERRORS BY THEME")
print("=" * 110)

print(
    new_errors["theme_name"]
    .value_counts()
    .to_string()
)


# =============================================================================
# NEW ERRORS BY SIGNAL
# =============================================================================

new_by_signal = (
    new_errors.groupby(
        [
            "theme_name",
            "theme_signal",
        ]
    )
    .size()
    .reset_index(name="n")
    .sort_values(
        [
            "theme_name",
            "n",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

print("\n" + "=" * 110)
print("NEW ERRORS BY SIGNAL")
print("=" * 110)

print(
    new_by_signal.to_string(
        index=False
    )
)


# =============================================================================
# FIXED ERRORS BY SIGNAL
# =============================================================================

fixed_by_signal = (
    fixed_errors.groupby(
        [
            "theme_name",
            "theme_signal",
        ]
    )
    .size()
    .reset_index(name="n")
    .sort_values(
        [
            "theme_name",
            "n",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

print("\n" + "=" * 110)
print("FIXED ERRORS BY SIGNAL")
print("=" * 110)

print(
    fixed_by_signal.to_string(
        index=False
    )
)


# =============================================================================
# NET ERROR CHANGE BY SIGNAL
# =============================================================================

fixed_counts = (
    fixed_errors.groupby(
        [
            "theme_name",
            "theme_signal",
        ]
    )
    .size()
    .rename("fixed_errors")
)

new_counts = (
    new_errors.groupby(
        [
            "theme_name",
            "theme_signal",
        ]
    )
    .size()
    .rename("new_errors")
)

comparison = pd.concat(
    [
        fixed_counts,
        new_counts,
    ],
    axis=1,
).fillna(0)

comparison[
    [
        "fixed_errors",
        "new_errors",
    ]
] = comparison[
    [
        "fixed_errors",
        "new_errors",
    ]
].astype(int)

# Positive values indicate a net reduction in errors for the reviewed sample.
comparison["net_gain"] = (
    comparison["fixed_errors"]
    - comparison["new_errors"]
)

comparison = (
    comparison
    .reset_index()
    .sort_values(
        [
            "net_gain",
            "fixed_errors",
        ],
        ascending=[
            False,
            False,
        ],
    )
)

print("\n" + "=" * 110)
print("NET ERROR CHANGE BY SIGNAL")
print("=" * 110)

print(
    comparison.to_string(
        index=False
    )
)


# =============================================================================
# NEW-ERROR INSPECTION
# =============================================================================

print("\n" + "=" * 110)
print(
    f"NEW ERRORS INTRODUCED BY SCENARIO A "
    f"({len(new_errors):,})"
)
print("=" * 110)

for theme, group in new_errors.groupby(
    "theme_name"
):

    print()
    print("#" * 100)
    print(theme)
    print("#" * 100)

    for _, row in group.iterrows():

        print(
            f"\nID: {row['annotation_id']}"
        )

        print(
            f"Signal: {row['theme_signal']}"
        )

        print(
            f"Title: {row['title']}"
        )

        print(
            f"Risk evidence: "
            f"{row['risk_evidence']}"
        )


# =============================================================================
# SAVE
# =============================================================================

new_errors.to_csv(
    OUT_NEW,
    index=False,
)

fixed_errors.to_csv(
    OUT_FIXED,
    index=False,
)

print("\n" + "=" * 110)
print("FILES SAVED")
print("=" * 110)

print(OUT_NEW)
print(OUT_FIXED)

print(
    "\nAnalysis completed."
)
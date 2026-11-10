#!/usr/bin/env python3

"""
STEP 4 — NOVELTY TEMPORAL STABILITY AUDIT
==========================================

Purpose
-------
Assess the temporal stability of the novelty distribution for the
selected diagnostic specification:

    historical window = 365 days
    K = 10 nearest historical neighbours

Input:
    novelty_calibration_pilot_v1.csv

Expected calibration design:
    2015-2023
    120 observations per year
    10 observations per month
    1,080 observations in total

The analysis reports:
- yearly novelty distributions;
- global calibration-sample distribution;
- yearly deviations from the global mean and median;
- a descriptive linear trend in yearly means;
- exact validation of the balanced monthly sampling design.

The linear trend is descriptive only and is not treated as an
econometric or inferential model.

Because the calibration sample is balanced by month, global statistics
describe the calibration design and are not automatically corpus-weighted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PARAMETERS
# =============================================================================

CSV_FILE = Path(
    os.getenv(
        "NOVELTY_CALIBRATION_FILE",
        "novelty_calibration_pilot_v1.csv",
    )
)

WINDOW_DAYS = 365
K = 10

NOVELTY_COLUMN = "novelty_mean_top_k"

EXPECTED_YEARS = list(
    range(
        2015,
        2024,
    )
)

EXPECTED_PER_YEAR = 120
EXPECTED_PER_MONTH = 10

EXPECTED_MONTHS = (
    len(EXPECTED_YEARS)
    * 12
)

EXPECTED_ROWS = (
    EXPECTED_MONTHS
    * EXPECTED_PER_MONTH
)

YEARLY_QUANTILES = [
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_calibration_data():

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: "
            f"{CSV_FILE}"
        )

    df = pd.read_csv(
        CSV_FILE,
        parse_dates=[
            "publication_date"
        ],
    )

    required_columns = {
        "article_id",
        "publication_date",
        "window_days",
        "k",
        NOVELTY_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if df[
        "publication_date"
    ].isna().any():
        raise RuntimeError(
            "Missing or invalid publication "
            "dates detected."
        )

    spec = df.loc[
        (
            df["window_days"]
            == WINDOW_DAYS
        )
        & (
            df["k"]
            == K
        )
    ].copy()

    if len(
        spec
    ) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} "
            f"observations for "
            f"W={WINDOW_DAYS}, K={K}; "
            f"found {len(spec):,}."
        )

    if spec[
        [
            "article_id",
            "publication_date",
            NOVELTY_COLUMN,
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Missing values detected in "
            "the selected specification."
        )

    spec[
        NOVELTY_COLUMN
    ] = pd.to_numeric(
        spec[
            NOVELTY_COLUMN
        ],
        errors="raise",
    )

    novelty_values = spec[
        NOVELTY_COLUMN
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        novelty_values
    ).all():
        raise RuntimeError(
            "Non-finite novelty values "
            "detected."
        )

    # Cosine-based novelty is theoretically in [0, 2].
    if (
        np.any(
            novelty_values < 0.0
        )
        or np.any(
            novelty_values > 2.0
        )
    ):
        raise RuntimeError(
            "Novelty values outside the "
            "theoretical [0, 2] range."
        )

    if spec[
        "article_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate article IDs detected "
            "for the selected specification."
        )

    spec["year"] = (
        spec[
            "publication_date"
        ].dt.year
    )

    spec["year_month"] = (
        spec[
            "publication_date"
        ].dt.to_period(
            "M"
        )
    )

    return spec


# =============================================================================
# SAMPLE VALIDATION
# =============================================================================

def validate_sample_design(
    spec,
):

    observed_years = sorted(
        spec[
            "year"
        ].unique().tolist()
    )

    if (
        observed_years
        != EXPECTED_YEARS
    ):
        raise RuntimeError(
            "Unexpected year coverage: "
            f"{observed_years}"
        )

    year_counts = (
        spec.groupby(
            "year"
        )
        .size()
        .reindex(
            EXPECTED_YEARS
        )
    )

    if not (
        year_counts.notna().all()
        and (
            year_counts
            == EXPECTED_PER_YEAR
        ).all()
    ):
        raise RuntimeError(
            "Unexpected observations "
            "per year:\n"
            + year_counts.to_string()
        )

    expected_months = (
        pd.period_range(
            start=(
                f"{EXPECTED_YEARS[0]}"
                "-01"
            ),
            end=(
                f"{EXPECTED_YEARS[-1]}"
                "-12"
            ),
            freq="M",
        )
    )

    monthly_counts = (
        spec.groupby(
            "year_month"
        )
        .size()
        .reindex(
            expected_months
        )
    )

    if len(
        monthly_counts
    ) != EXPECTED_MONTHS:
        raise RuntimeError(
            "Unexpected number of expected "
            "monthly periods."
        )

    if monthly_counts.isna().any():
        missing_months = (
            monthly_counts[
                monthly_counts.isna()
            ]
            .index
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Missing calibration months: "
            + ", ".join(
                missing_months
            )
        )

    if not (
        monthly_counts
        == EXPECTED_PER_MONTH
    ).all():
        raise RuntimeError(
            "Unexpected observations "
            "per month:\n"
            + monthly_counts.to_string()
        )

    return (
        year_counts,
        monthly_counts,
    )


# =============================================================================
# YEARLY DISTRIBUTION
# =============================================================================

def calculate_yearly_distribution(
    spec,
):

    grouped = spec.groupby(
        "year"
    )[
        NOVELTY_COLUMN
    ]

    yearly = grouped.agg(
        n="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )

    quantiles = (
        grouped.quantile(
            YEARLY_QUANTILES
        )
        .unstack()
        .rename(
            columns={
                0.05: "p05",
                0.25: "p25",
                0.50: "p50",
                0.75: "p75",
                0.95: "p95",
            }
        )
    )

    yearly = yearly.join(
        quantiles
    )

    yearly["iqr"] = (
        yearly["p75"]
        - yearly["p25"]
    )

    yearly = yearly[
        [
            "n",
            "mean",
            "median",
            "std",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "iqr",
            "min",
            "max",
        ]
    ]

    return yearly


# =============================================================================
# GLOBAL DISTRIBUTION
# =============================================================================

def calculate_global_distribution(
    spec,
):

    values = spec[
        NOVELTY_COLUMN
    ]

    return {
        "mean":
            float(
                values.mean()
            ),
        "median":
            float(
                values.median()
            ),
        "std":
            float(
                values.std()
            ),
        "p05":
            float(
                values.quantile(
                    0.05
                )
            ),
        "p25":
            float(
                values.quantile(
                    0.25
                )
            ),
        "p75":
            float(
                values.quantile(
                    0.75
                )
            ),
        "p95":
            float(
                values.quantile(
                    0.95
                )
            ),
        "iqr":
            float(
                values.quantile(
                    0.75
                )
                - values.quantile(
                    0.25
                )
            ),
    }


# =============================================================================
# LINEAR TREND
# =============================================================================

def calculate_linear_trend(
    yearly,
):

    years = (
        yearly.index.to_numpy(
            dtype=np.float64
        )
    )

    yearly_means = (
        yearly[
            "mean"
        ].to_numpy(
            dtype=np.float64
        )
    )

    slope, intercept = np.polyfit(
        years,
        yearly_means,
        1,
    )

    fitted = (
        intercept
        + slope * years
    )

    ss_res = float(
        np.sum(
            (
                yearly_means
                - fitted
            )
            ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                yearly_means
                - yearly_means.mean()
            )
            ** 2
        )
    )

    r_squared = (
        1.0
        - ss_res / ss_tot
        if ss_tot > 0.0
        else np.nan
    )

    total_fitted_change = (
        slope
        * (
            years.max()
            - years.min()
        )
    )

    return {
        "slope":
            float(slope),
        "intercept":
            float(intercept),
        "r_squared":
            float(r_squared),
        "total_fitted_change":
            float(
                total_fitted_change
            ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "STEP 4 — NOVELTY "
        "TEMPORAL STABILITY AUDIT"
    )
    print("=" * 110)

    print(
        f"Calibration file     : "
        f"{CSV_FILE}"
    )

    print(
        f"Specification        : "
        f"W={WINDOW_DAYS}, K={K}"
    )

    # =========================================================================
    # 1. LOAD AND VALIDATE SAMPLE
    # =========================================================================

    spec = load_calibration_data()

    (
        year_counts,
        monthly_counts,
    ) = validate_sample_design(
        spec
    )

    print()
    print(
        f"Observations         : "
        f"{len(spec):,}"
    )

    print(
        f"Years                : "
        f"{EXPECTED_YEARS[0]}-"
        f"{EXPECTED_YEARS[-1]}"
    )

    print(
        f"Months               : "
        f"{len(monthly_counts)}"
    )

    print(
        f"Targets/month        : "
        f"{EXPECTED_PER_MONTH}"
    )

    print(
        "YEAR SAMPLE          : PASS"
    )

    print(
        "MONTHLY SAMPLE       : PASS"
    )

    # =========================================================================
    # 2. YEARLY DISTRIBUTION
    # =========================================================================

    yearly = (
        calculate_yearly_distribution(
            spec
        )
    )

    print()
    print("=" * 110)
    print(
        "YEARLY NOVELTY DISTRIBUTION"
    )
    print("=" * 110)

    print(
        yearly.to_string(
            float_format=(
                lambda value:
                f"{value:.6f}"
            )
        )
    )

    # =========================================================================
    # 3. GLOBAL CALIBRATION-SAMPLE DISTRIBUTION
    # =========================================================================

    global_stats = (
        calculate_global_distribution(
            spec
        )
    )

    print()
    print("=" * 110)
    print(
        "GLOBAL CALIBRATION-SAMPLE "
        "DISTRIBUTION"
    )
    print("=" * 110)

    for name, value in (
        global_stats.items()
    ):

        print(
            f"{name:>8} : "
            f"{value:.6f}"
        )

    # =========================================================================
    # 4. YEARLY DEVIATIONS
    # =========================================================================

    yearly[
        "mean_minus_global"
    ] = (
        yearly["mean"]
        - global_stats["mean"]
    )

    yearly[
        "mean_pct_global"
    ] = (
        100.0
        * yearly[
            "mean_minus_global"
        ]
        / global_stats["mean"]
    )

    yearly[
        "median_minus_global"
    ] = (
        yearly["median"]
        - global_stats["median"]
    )

    print()
    print("=" * 110)
    print(
        "YEARLY DEVIATION FROM "
        "GLOBAL CALIBRATION SAMPLE"
    )
    print("=" * 110)

    print(
        yearly[
            [
                "mean",
                "mean_minus_global",
                "mean_pct_global",
                "median",
                "median_minus_global",
            ]
        ].to_string(
            float_format=(
                lambda value:
                f"{value:.6f}"
            )
        )
    )

    # =========================================================================
    # 5. DESCRIPTIVE LINEAR TREND
    # =========================================================================

    trend = (
        calculate_linear_trend(
            yearly
        )
    )

    print()
    print("=" * 110)
    print(
        "LINEAR YEAR TREND — "
        "DESCRIPTIVE DIAGNOSTIC ONLY"
    )
    print("=" * 110)

    print(
        f"Slope per year       : "
        f"{trend['slope']:.8f}"
    )

    print(
        f"Fitted "
        f"{EXPECTED_YEARS[0]}-"
        f"{EXPECTED_YEARS[-1]} Δ   : "
        f"{trend['total_fitted_change']:+.8f}"
    )

    print(
        f"R-squared            : "
        f"{trend['r_squared']:.6f}"
    )

    # =========================================================================
    # 6. TEMPORAL RANGE DIAGNOSTICS
    # =========================================================================

    yearly_mean_range = (
        float(
            yearly["mean"].max()
            - yearly["mean"].min()
        )
    )

    yearly_median_range = (
        float(
            yearly["median"].max()
            - yearly["median"].min()
        )
    )

    max_abs_mean_deviation = float(
        np.max(
            np.abs(
                yearly[
                    "mean_minus_global"
                ].to_numpy()
            )
        )
    )

    max_abs_median_deviation = float(
        np.max(
            np.abs(
                yearly[
                    "median_minus_global"
                ].to_numpy()
            )
        )
    )

    print()
    print("=" * 110)
    print(
        "TEMPORAL RANGE DIAGNOSTICS"
    )
    print("=" * 110)

    print(
        f"Yearly mean range    : "
        f"{yearly_mean_range:.8f}"
    )

    print(
        f"Yearly median range  : "
        f"{yearly_median_range:.8f}"
    )

    print(
        f"Max |mean-global|    : "
        f"{max_abs_mean_deviation:.8f}"
    )

    print(
        f"Max |median-global|  : "
        f"{max_abs_median_deviation:.8f}"
    )

    # =========================================================================
    # 7. FINAL VALIDATION
    # =========================================================================

    print()
    print("=" * 110)
    print("FINAL VALIDATION")
    print("=" * 110)

    print(
        f"Years represented    : "
        f"{len(year_counts)}"
    )

    print(
        f"Targets/year         : "
        f"{year_counts.min()} -> "
        f"{year_counts.max()}"
    )

    print(
        f"Months represented   : "
        f"{len(monthly_counts)}"
    )

    print(
        f"Targets/month        : "
        f"{monthly_counts.min()} -> "
        f"{monthly_counts.max()}"
    )

    print(
        "SAMPLE DESIGN        : PASS"
    )

    print()
    print(
        "Temporal stability metrics are "
        "reported descriptively; no "
        "automatic stability threshold "
        "is imposed."
    )

    print("=" * 110)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nInterrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print(
            "\nFATAL ERROR:"
        )

        print(
            repr(exc)
        )

        sys.exit(1)
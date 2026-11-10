#!/usr/bin/env python3

"""
Check the raw IsDuplicate field directly in the source CSV files.

The script reads the enabled files listed in the ingestion manifest,
bypassing the database entirely, and reports the distinct values of
IsDuplicate for each file and across the full corpus.

The purpose is to determine whether a constant IsDuplicate value observed
in staging originates from the raw data or was introduced during ingestion.

The script is read-only and does not modify any files or database objects.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


# ============================================================================
# CONFIGURATION
# ============================================================================

MANIFEST_FILE_PATH = os.getenv(
    "INGESTION_MANIFEST_PATH",
    "ingestion_manifest.csv",
)

CSV_DELIMITER = "|"
CSV_ENCODING = "utf-8"

EXPECTED_COLUMNS = [
    "Title",
    "Body",
    "Date",
    "Language",
    "Source",
    "Sentiment",
    "IsDuplicate",
    "dataType",
    "Url",
]

ISDUPLICATE_INDEX = EXPECTED_COLUMNS.index("IsDuplicate")

MAX_EXAMPLES_PER_VALUE = 3


# ============================================================================
# CSV CONFIGURATION
# ============================================================================

def increase_csv_field_size_limit() -> None:
    """Increase the maximum CSV field size to support long article bodies."""

    max_int = sys.maxsize

    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)


increase_csv_field_size_limit()


# ============================================================================
# MANIFEST
# ============================================================================

def read_enabled_files(manifest_path: str) -> List[str]:
    """Return the file paths enabled in the ingestion manifest."""

    manifest_file = Path(manifest_path)

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_file}"
        )

    enabled_files = []

    with open(
        manifest_file,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter=";",
        )

        if reader.fieldnames is None:
            raise ValueError(
                "The manifest does not contain a header."
            )

        reader.fieldnames = [
            name.strip()
            for name in reader.fieldnames
        ]

        required_columns = {
            "enabled",
            "file_path",
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Missing manifest columns: "
                f"{sorted(missing_columns)}. "
                f"Columns found: {reader.fieldnames}"
            )

        for row in reader:
            if row["enabled"].strip() == "1":
                enabled_files.append(
                    row["file_path"].strip()
                )

    return enabled_files


# ============================================================================
# RAW CSV ANALYSIS
# ============================================================================

def analyze_isduplicate_in_file(
    csv_path: str,
) -> Dict[str, Any]:
    """
    Count distinct IsDuplicate values in one raw CSV file and retain
    a small number of examples for each observed value.
    """

    csv_file = Path(csv_path)

    if not csv_file.exists():
        print(
            f"[WARNING] File not found, skipping: "
            f"{csv_file}"
        )

        return {
            "file": str(csv_file),
            "found": False,
        }

    value_counts: Dict[str, int] = {}
    examples: Dict[str, List[dict]] = {}

    total_rows = 0
    malformed_rows = 0

    with open(
        csv_file,
        "r",
        encoding=CSV_ENCODING,
        newline="",
    ) as file:

        reader = csv.reader(
            file,
            delimiter=CSV_DELIMITER,
            quotechar='"',
        )

        try:
            header = next(reader)

        except StopIteration:
            print(
                f"[WARNING] Empty file: {csv_file}"
            )

            return {
                "file": str(csv_file),
                "found": True,
                "empty": True,
            }

        if header != EXPECTED_COLUMNS:
            print(
                f"[WARNING] Unexpected header in "
                f"{csv_file.name}: {header}"
            )

        for raw_line_number, row in enumerate(
            reader,
            start=2,
        ):

            if len(row) != len(EXPECTED_COLUMNS):
                malformed_rows += 1
                continue

            total_rows += 1

            raw_value = row[ISDUPLICATE_INDEX]

            # repr() preserves visible differences such as whitespace.
            key = repr(raw_value)

            value_counts[key] = (
                value_counts.get(key, 0) + 1
            )

            if key not in examples:
                examples[key] = []

            if (
                len(examples[key])
                < MAX_EXAMPLES_PER_VALUE
            ):
                examples[key].append(
                    {
                        "line": raw_line_number,
                        "title_preview": (
                            row[0] or ""
                        )[:60],
                        "date": row[2],
                        "source": row[4],
                    }
                )

    return {
        "file": csv_file.name,
        "found": True,
        "total_rows": total_rows,
        "malformed_rows": malformed_rows,
        "value_counts": value_counts,
        "examples": examples,
    }


# ============================================================================
# FILE REPORT
# ============================================================================

def print_file_report(
    report: Dict[str, Any],
) -> None:
    """Print the IsDuplicate analysis for one CSV file."""

    print()
    print("=" * 80)
    print(f"FILE: {report.get('file')}")
    print("=" * 80)

    if not report.get("found", False):
        return

    if report.get("empty", False):
        print("Empty file.")
        return

    print(
        f"Total rows read        : "
        f"{report['total_rows']:,}"
    )

    print(
        f"Malformed rows skipped : "
        f"{report['malformed_rows']:,}"
    )

    print()
    print("Distinct IsDuplicate values:")

    sorted_items = sorted(
        report["value_counts"].items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    for key, count in sorted_items:
        percentage = (
            count
            / report["total_rows"]
            * 100.0
            if report["total_rows"]
            else 0.0
        )

        print(
            f"  {key:>12} -> "
            f"{count:>10,} "
            f"({percentage:6.2f}%)"
        )

    # Multiple distinct values require inspection because staging should
    # preserve the original IsDuplicate values during ingestion.
    if len(sorted_items) > 1:
        print()
        print(
            "Examples for each distinct value:"
        )

        for key, _ in sorted_items:
            print(
                f"  -- value {key} --"
            )

            for example in report[
                "examples"
            ].get(key, []):

                print(
                    f"     line={example['line']}, "
                    f"date={example['date']}, "
                    f"source={example['source']}, "
                    f"title="
                    f"{example['title_preview']!r}"
                )


# ============================================================================
# GLOBAL SUMMARY
# ============================================================================

def print_global_summary(
    reports: List[Dict[str, Any]],
) -> None:
    """Print the aggregate IsDuplicate distribution across all raw CSV files."""

    print()
    print("=" * 80)
    print(
        "GLOBAL SUMMARY — DISTINCT IsDuplicate "
        "VALUES ACROSS ALL FILES"
    )
    print("=" * 80)

    global_counts: Dict[str, int] = {}

    for report in reports:
        if (
            not report.get("found")
            or report.get("empty")
        ):
            continue

        for key, count in report[
            "value_counts"
        ].items():

            global_counts[key] = (
                global_counts.get(key, 0)
                + count
            )

    total = sum(global_counts.values())

    sorted_items = sorted(
        global_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    for key, count in sorted_items:
        percentage = (
            count / total * 100.0
            if total
            else 0.0
        )

        print(
            f"  {key:>12} -> "
            f"{count:>10,} "
            f"({percentage:6.2f}%)"
        )

    print()

    if not sorted_items:
        print(
            "CONCLUSION: no valid IsDuplicate "
            "values were found."
        )

    elif len(sorted_items) == 1:
        print(
            "CONCLUSION: only one distinct value "
            "was found across all raw CSV files. "
            "The IsDuplicate field is therefore "
            "already constant in the source data "
            "and was not made constant by the "
            "ingestion pipeline."
        )

    else:
        print(
            "CONCLUSION: multiple distinct "
            "IsDuplicate values are present in "
            "the raw CSV files. If staging "
            "contains only one value, the "
            "ingestion step should be investigated."
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run the raw IsDuplicate audit."""

    files = read_enabled_files(
        MANIFEST_FILE_PATH
    )

    if not files:
        print(
            "No enabled files found in the manifest."
        )
        return

    print(
        f"Enabled files to analyze: {len(files)}"
    )

    reports = []

    for file_path in files:
        report = analyze_isduplicate_in_file(
            file_path
        )

        print_file_report(report)

        reports.append(report)

    print_global_summary(reports)


if __name__ == "__main__":
    main()
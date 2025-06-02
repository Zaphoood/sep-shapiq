"""Parse a coverage report and check whether code coverage satsifies a minimum threshold provided by the environment."""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

MIN_CODE_COVERAGE_ENV = "MIN_CODE_COVERAGE"


def parse_coverage(xml_file: str) -> float:
    """Parses line-rate coverage percentage from a coverage XML file."""
    tree = ET.parse(xml_file)  # noqa: S314
    root = tree.getroot()
    coverage_attr = root.attrib.get("line-rate")
    if coverage_attr is not None:
        return float(coverage_attr) * 100

    msg = "Could not find coverage data."
    raise ValueError(msg)


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(description="Check code coverage against a minimum threshold.")
    parser.add_argument("coverage_xml", help="Path to the coverage XML report.")
    parser.add_argument(
        "--fail-too-low",
        action="store_true",
        help="Exit with status 1 if coverage is below threshold.",
    )

    args = parser.parse_args()

    try:
        coverage = parse_coverage(args.coverage_xml)
    except Exception as e:  # noqa: BLE001
        print(f"Error reading coverage: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        min_coverage = float(os.environ.get(MIN_CODE_COVERAGE_ENV))
    except (TypeError, ValueError):
        print(
            f"Missing or invalid environment variable {MIN_CODE_COVERAGE_ENV} (must be a float).",
            file=sys.stderr,
        )
        sys.exit(1)

    msg = (
        f"✅ Code coverage is **{coverage:.2f}%** -- OK!"
        if coverage >= min_coverage
        else f"❌ Code coverage is **{coverage:.2f}%** -- below required **{min_coverage}%**."
    )
    print(msg)

    if args.fail_too_low and coverage < min_coverage:
        print("Code coverage is too low. Exiting with code 1 since --fail-too-low is set.")
        sys.exit(1)


if __name__ == "__main__":
    main()
